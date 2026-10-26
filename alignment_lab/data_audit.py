"""CPU-only, immutable HelpSteer3 preference audit and grouped partitioning.

Near duplicates use an exact Jaccard threshold *within the declared token-shingle
representation*. A prefix-filtered inverted index bounds candidate generation;
if its work budget is exceeded, readiness is blocked. This is not semantic
deduplication and cannot establish absence of pretraining contamination.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any, Callable, Iterable

from .chat_codec import Qwen3ChatCodec, SERIALIZATION_VERSION, validate_context, validate_response

VERSION = "alignlab-data-audit-v3-native-nfc"
ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 20261020
    dev_fraction: float = 0.1
    diagnostic_clusters: int = 100
    prompt_limit: int = 1024
    response_limit: int = 512
    pair_limit: int = 24000
    sft_limit: int = 4000
    minimum_pairs: int = 12000
    minimum_clusters: int = 8000
    minimum_sft: int = 2000
    near_threshold: float = 0.9
    near_min_chars: int = 40
    prefix_min_chars: int = 40
    candidate_budget: int = 2_000_000
    posting_budget: int = 20_000_000
    expected_train: int | None = 38459
    expected_validation: int | None = 2017

    def validate(self) -> None:
        if not 0 < self.dev_fraction < 1:
            raise ValueError("dev_fraction must be between zero and one")
        if not 0.5 <= self.near_threshold <= 1:
            raise ValueError("near_threshold must be in [0.5, 1]")
        for key, value in asdict(self).items():
            if key not in {"seed", "dev_fraction", "near_threshold", "expected_train", "expected_validation"} and value < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not self.candidate_budget or not self.posting_budget:
            raise ValueError("near-duplicate work budgets must be positive")
        if not self.prompt_limit or not self.response_limit:
            raise ValueError("prompt and response limits must be positive; response limit includes EOS")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        # JSON arrays and line-delimited objects are supported, including gzip.
        first = ""
        while not first:
            char = stream.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
        stream.seek(0)
        if first == "[":
            rows = json.load(stream)
            if not isinstance(rows, list):
                raise ValueError("expected a JSON array")
            yield from rows
        else:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON at {path}:{line_no}: {error}") from error


def parse_row(raw: Any, split: str, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("row_not_object")
    score = raw.get("overall_preference")
    if type(score) is not int or not -3 <= score <= 3:
        raise ValueError("invalid_overall_preference")
    clean_context = validate_context(raw.get("context"))
    answers = [validate_response(raw.get("response1")), validate_response(raw.get("response2"))]
    domain = raw.get("domain")
    language = raw.get("language")
    if not isinstance(domain, str) or domain.casefold() not in {"general", "stem", "code", "multilingual"}:
        raise ValueError("invalid_domain")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("invalid_language")
    individuals = raw.get("individual_preference", [])
    if not isinstance(individuals, list):
        raise ValueError("invalid_individual_preference")
    individual_scores = []
    for item in individuals:
        value = item.get("score") if isinstance(item, dict) else None
        if type(value) is not int or not -3 <= value <= 3:
            raise ValueError("invalid_individual_score")
        individual_scores.append(value)
    norm_context = [{"role": m["role"], "content": normalized(m["content"])} for m in clean_context]
    user_turns = [m["content"] for m in norm_context if m["role"] == "user"]
    if not user_turns:
        raise ValueError("missing_user")
    payload = {"context": clean_context, "response1": answers[0], "response2": answers[1],
               "overall_preference": score, "domain": domain.casefold(), "language": language,
               "individual_scores": individual_scores}
    row = {**payload, "record_id": digest(payload), "source_split": split, "source_index": index,
           "prompt_id": digest(norm_context), "user_path_id": digest(user_turns),
           "_user_turns": user_turns, "_user_text": "\n".join(user_turns)}
    if score:
        chosen = 0 if score < 0 else 1
        row.update(chosen=answers[chosen], rejected=answers[1 - chosen])
    else:
        row.update(chosen=None, rejected=None)
    row["annotation_summary"] = {
        "count": len(individual_scores),
        "mean": sum(individual_scores) / len(individual_scores) if individual_scores else None,
        "same_sign_as_overall": sum((x > 0) - (x < 0) == (score > 0) - (score < 0) for x in individual_scores),
        "positive_votes": sum(x > 0 for x in individual_scores),
        "negative_votes": sum(x < 0 for x in individual_scores),
        "zero_votes": sum(x == 0 for x in individual_scores),
    }
    return row


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def shingles(text: str) -> array:
    tokens = TOKEN_RE.findall(text)
    grams = (tokens[i:i + 3] for i in range(max(1, len(tokens) - 2)))
    hashed = {int.from_bytes(hashlib.blake2b("\x1f".join(gram).encode("utf-8"), digest_size=8).digest(), "big")
              for gram in grams}
    return array("Q", sorted(hashed))


def jaccard_at_least(a: array, b: array, threshold: float) -> bool:
    # Decimal avoids float ceil errors at exact threshold boundaries.
    t = Decimal(str(threshold))
    need = int((t * (len(a) + len(b)) / (1 + t)).to_integral_value(rounding=ROUND_CEILING))
    i = j = overlap = 0
    while i < len(a) and j < len(b):
        if overlap + min(len(a) - i, len(b) - j) < need:
            return False
        if a[i] == b[j]:
            overlap += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return overlap >= need


def build_clusters(rows: list[dict[str, Any]], config: AuditConfig,
                   progress: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    uf = UnionFind(len(rows))
    exact: dict[str, int] = {}
    prefixes: dict[str, int] = {}
    links = Counter()
    for i, row in enumerate(rows):
        for kind, key in (("context", row["prompt_id"]), ("user_path", row["user_path_id"])):
            combined = kind + ":" + key
            if combined in exact:
                links[kind] += uf.union(i, exact[combined])
            else:
                exact[combined] = i
        turns = row["_user_turns"]
        for end in range(1, len(turns) + 1):
            prefix = turns[:end]
            if sum(map(len, prefix)) < config.prefix_min_chars:
                continue
            key = digest(prefix)
            if key in prefixes:
                links["user_prefix"] += uf.union(i, prefixes[key])
            else:
                prefixes[key] = i
    progress("Exact context and user-prefix clustering complete; building near-duplicate index")
    # One representative per identical user path, not per already-merged cluster:
    # dropping distinct paths here would miss transitive near-duplicate edges.
    representatives: dict[str, int] = {}
    for i, row in enumerate(rows):
        if len(row["_user_text"]) >= config.near_min_chars:
            representatives.setdefault(row["user_path_id"], i)
    sketches = [(shingles(rows[i]["_user_text"]), i) for i in representatives.values()]
    sketches.sort(key=lambda pair: (len(pair[0]), rows[pair[1]]["user_path_id"]))
    inverted: dict[int, list[int]] = defaultdict(list)
    candidates = postings = 0
    complete = True
    t = Decimal(str(config.near_threshold))
    for position, (values, row_index) in enumerate(sketches):
        prefix_length = len(values) - int((t * len(values)).to_integral_value(rounding=ROUND_CEILING)) + 1
        minimum_length = int((t * len(values)).to_integral_value(rounding=ROUND_CEILING))
        seen: set[int] = set()
        for value in values[:prefix_length]:
            for previous in inverted[value]:
                postings += 1
                if postings > config.posting_budget:
                    complete = False
                    break
                other_values, other_index = sketches[previous]
                if len(other_values) < minimum_length or previous in seen:
                    continue
                seen.add(previous)
                if uf.find(row_index) == uf.find(other_index):
                    continue
                candidates += 1
                if candidates > config.candidate_budget:
                    complete = False
                    break
                if jaccard_at_least(values, other_values, config.near_threshold):
                    links["near_jaccard"] += uf.union(row_index, other_index)
            if not complete:
                break
        if not complete:
            break
        for value in values[:prefix_length]:
            inverted[value].append(position)
        if position % 2000 == 0:
            progress(f"Near duplicates {position}/{len(sketches)}; candidates={candidates}; postings={postings}")
    component_keys: dict[int, str] = {}
    for i, row in enumerate(rows):
        root = uf.find(i)
        component_keys[root] = min(component_keys.get(root, row["prompt_id"]), row["prompt_id"])
    for i, row in enumerate(rows):
        row["cluster_id"] = component_keys[uf.find(i)]
    members_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members_by_cluster[row["cluster_id"]].append(row)
    largest = []
    for key, members in sorted(members_by_cluster.items(), key=lambda item: (-len(item[1]), item[0]))[:10]:
        largest.append({"cluster_id": key, "rows": len(members),
                        "unique_normalized_contexts": len({r["prompt_id"] for r in members}),
                        "domains": dict(sorted(Counter(r["domain"] for r in members).items())),
                        "source_splits": dict(sorted(Counter(r["source_split"] for r in members).items()))})
    return {"algorithm": "prefix-filtered exact Jaccard join over sorted 64-bit token-trigram shingles",
            "basis": "NFKC/casefold/whitespace-normalized concatenated user turns",
            "threshold": config.near_threshold, "minimum_characters": config.near_min_chars,
            "user_prefix_minimum_characters": config.prefix_min_chars,
            "representatives": len(sketches), "verified_candidates": candidates, "posting_visits": postings,
            "candidate_budget": config.candidate_budget, "posting_budget": config.posting_budget,
            "complete_for_declared_representation": complete, "successful_unions": dict(links),
            "largest_ten_clusters": largest,
            "limits": ["Not semantic/paraphrase deduplication", "64-bit shingle hashes have a nonzero collision risk",
                       "Texts below the minimum length receive exact/prefix checks only",
                       "No claim about base-model pretraining contamination"]}


def stable_rank(seed: int, namespace: str, key: str) -> str:
    return digest([seed, namespace, key])


def grouped_split(rows: list[dict[str, Any]], config: AuditConfig) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_id"]].append(row)
    strata: dict[str, list[str]] = defaultdict(list)
    for cluster, members in groups.items():
        # Majority domain/language, with a lexical tie break, is deterministic.
        counts = Counter((r["domain"], r["language"]) for r in members)
        label = sorted(counts, key=lambda x: (-counts[x], x))[0]
        strata[canonical(label)].append(cluster)
    assignment = {}
    for stratum, clusters in sorted(strata.items()):
        ordered = sorted(clusters, key=lambda key: stable_rank(config.seed, "split:" + stratum, key))
        count = int(math.floor(len(ordered) * config.dev_fraction + 0.5))
        # Do not force rare singleton strata into dev or alter the requested ratio.
        for i, key in enumerate(ordered):
            assignment[key] = "internal_dev" if i < count else "train"
    dev_clusters = [key for key, value in assignment.items() if value == "internal_dev"]
    eligible = [key for key in dev_clusters if any(row["overall_preference"] for row in groups[key])]
    diagnostics = set(sorted(eligible, key=lambda key: stable_rank(config.seed, "diagnostic", key))[:config.diagnostic_clusters])
    for key in dev_clusters:
        assignment[key] = "diagnostic_holdout" if key in diagnostics else "tuning_dev"
    return assignment


class LocalTokenCounter:
    def __init__(self, directory: Path):
        # Transformers tokenizer utilities work without torch; never query CUDA.
        os.environ["USE_TORCH"] = "0"
        os.environ["USE_TF"] = "0"
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.codec = Qwen3ChatCodec(self.tokenizer)
        self.cache = {}

    def validate_input(self, row: dict[str, Any]) -> None:
        self.codec.validate_input(row["context"], row["response1"], row["response2"])

    def __call__(self, row: dict[str, Any]) -> dict[str, int]:
        # Cache by raw context too: normalized IDs may conflate token-distinct text.
        key = digest(row["context"])
        if key not in self.cache:
            self.cache[key] = self.codec.encode_prompt(row["context"])
        prompt = self.cache[key]
        result = {"prompt": len(prompt.input_ids),
                  "prompt_nfc_changed": prompt.rendered != prompt.normalized_rendered}
        for name in ("response1", "response2"):
            completion = self.codec.encode_completion(prompt, row[name])
            result[name] = len(completion.completion_ids)
            result[name + "_nfc_changed"] = row[name] != self.codec.normalized(row[name])
        return result


def quantiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {key: None for key in ("min", "p50", "p90", "p95", "p99", "max")}
    values = sorted(values)
    result = {"min": values[0], "max": values[-1]}
    result.update({f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)] for p in (50, 90, 95, 99)})
    return result


def summarize(rows: list[dict[str, Any]], config: AuditConfig) -> dict[str, Any]:
    tokenized = [r for r in rows if "tokens" in r]
    return {"rows": len(rows), "non_tie_pairs": sum(r["overall_preference"] != 0 for r in rows),
            "ties": sum(r["overall_preference"] == 0 for r in rows),
            "unique_normalized_contexts": len({r["prompt_id"] for r in rows}),
            "independent_prompt_clusters": len({r["cluster_id"] for r in rows}),
            "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
            "languages": dict(sorted(Counter(r["language"] for r in rows).items())),
            "preference_scores": dict(sorted(Counter(str(r["overall_preference"]) for r in rows).items())),
            "tokenized_rows": len(tokenized),
            "lengths": {key: quantiles([r["tokens"][key] for r in tokenized]) for key in ("prompt", "response1", "response2")},
            "within_training_limits": sum(r["tokens"]["prompt"] <= config.prompt_limit and
                                          max(r["tokens"]["response1"], r["tokens"]["response2"]) <= config.response_limit
                                          for r in tokenized)}


def stratified_limit(rows: list[dict[str, Any]], maximum: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return sorted(rows, key=lambda r: r["record_id"])
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[row["domain"], row["language"]].append(row)
    exact = {key: len(value) * maximum / len(rows) for key, value in strata.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(strata, key=lambda key: (-(exact[key] - quotas[key]), key))[:maximum - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key, values in sorted(strata.items()):
        values.sort(key=lambda r: stable_rank(seed, namespace, r["record_id"]))
        selected.extend(values[:quotas[key]])
    return sorted(selected, key=lambda r: r["record_id"])


def training_length_grid(rows: list[dict[str, Any]], config: AuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare lengths on isolated, deduplicated TRAIN candidates only.

    Input is captured after grouped assignment and before the configured length
    filter or pair cap. Dev, diagnostic and official validation rows are forbidden.
    This descriptive grid does not validate GPU capacity or waive count minima.
    """
    if any(row["source_split"] != "train" or row.get("local_split") != "train"
           or not row["overall_preference"] for row in rows):
        raise ValueError("length_grid_requires_assigned_non_tie_train_candidates_only")
    metadata = []
    for row in rows:
        metadata.append({key: row[key] for key in
                         ("record_id", "source_split", "source_index", "prompt_id", "cluster_id", "domain", "language", "local_split")})
        metadata[-1].update(preference_strength=abs(row["overall_preference"]),
                            tokens=row.get("tokens"), token_error=row.get("token_error"))
    available = [row for row in rows if "tokens" in row]
    grid = []
    for response_limit in (512, 640, 768, 1024):
        eligible = [row for row in available if row["tokens"]["prompt"] <= 1024
                    and max(row["tokens"]["response1"], row["tokens"]["response2"]) <= response_limit]
        strong = [row for row in eligible if abs(row["overall_preference"]) >= 2]
        clusters = len({row["cluster_id"] for row in eligible})
        sft_clusters = len({row["cluster_id"] for row in strong})
        grid.append({"prompt_limit": 1024, "response_limit_including_eos": response_limit,
                     "pairs_before_pair_cap": len(eligible), "independent_prompt_clusters": clusters,
                     "unique_normalized_contexts": len({row["prompt_id"] for row in eligible}),
                     "sft_strength_eligible_pairs": len(strong), "sft_strength_eligible_clusters": sft_clusters,
                     "sft_strength_eligible_contexts": len({row["prompt_id"] for row in strong}),
                     "domains": dict(sorted(Counter(row["domain"] for row in eligible).items())),
                     "languages": dict(sorted(Counter(row["language"] for row in eligible).items())),
                     "meets_count_minima_before_pair_cap": len(eligible) >= config.minimum_pairs
                     and clusters >= config.minimum_clusters and sft_clusters >= config.minimum_sft,
                     "gpu_capacity_verified": False})
    return metadata, {"population": "official train, deduplicated, cross-split isolated, assigned TRAIN, non-tie, before length filtering and pair cap",
                      "candidate_pairs": len(rows), "tokenized_pairs": len(available),
                      "missing_token_lengths": len(rows) - len(available),
                      "heldout_lengths_used_for_grid": False, "serialization_version": SERIALIZATION_VERSION,
                      "grid": grid, "interpretation": "Counts before pair/SFT caps; not final training coverage, capacity acceptance or a relaxed scale gate"}


def audit_records(train: Iterable[Any], validation: Iterable[Any], config: AuditConfig,
                  token_counter: Callable[[dict[str, Any]], dict[str, int]] | None = None,
                  progress: Callable[[str], None] = lambda _: None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    config.validate()
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    raw_counts = {}
    for split, source in (("train", train), ("validation", validation)):
        count = 0
        for count, raw in enumerate(source, 1):
            try:
                row = parse_row(raw, split, count - 1)
                if token_counter is not None and hasattr(token_counter, "validate_input"):
                    token_counter.validate_input(row)
                rows.append(row)
            except ValueError as error:
                exclusions.append({"source_split": split, "source_index": count - 1, "reason": str(error)})
        raw_counts[split] = count
        progress(f"Loaded {split}: {count} raw records")
    near = build_clusters(rows, config, progress)
    validation_clusters = {r["cluster_id"] for r in rows if r["source_split"] == "validation"}
    final = [r for r in rows if r["source_split"] == "validation"]
    candidates = []
    for row in rows:
        if row["source_split"] != "train":
            continue
        if row["cluster_id"] in validation_clusters:
            exclusions.append({"record_id": row["record_id"], "source_split": "train", "source_index": row["source_index"],
                               "cluster_id": row["cluster_id"], "reason": "cross_official_split_cluster"})
        else:
            candidates.append(row)
    # Exact duplicate pairs and conflicting labels are checked before partitioning.
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = digest([row["prompt_id"], sorted([normalized(row["response1"]), normalized(row["response2"])])])
        pair_groups[key].append(row)
    clean = []
    for group in pair_groups.values():
        choices = {normalized(r["chosen"]) if r["chosen"] is not None else None for r in group}
        contradictory = len(choices) > 1
        for i, row in enumerate(sorted(group, key=lambda r: r["record_id"])):
            reason = ("contradictory_duplicate_pair" if contradictory else
                      "identical_responses" if normalized(row["response1"]) == normalized(row["response2"]) else
                      "duplicate_pair" if i else None)
            if reason:
                exclusions.append({"record_id": row["record_id"], "source_split": "train", "source_index": row["source_index"],
                                   "cluster_id": row["cluster_id"], "reason": reason})
            else:
                clean.append(row)
    assignments = grouped_split(clean, config)
    token_errors = Counter()
    if token_counter:
        for index, row in enumerate(rows):
            try:
                row["tokens"] = token_counter(row)
            except (ValueError, TypeError) as error:
                row["token_error"] = str(error)
                token_errors[str(error)] += 1
            if index % 1000 == 0:
                progress(f"Token lengths {index}/{len(rows)}")
    train_pool, tuning, diagnostics, ties = [], [], [], []
    assigned_train_candidates = []
    diagnostic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        split = assignments[row["cluster_id"]]
        row["local_split"] = split
        if not row["overall_preference"]:
            ties.append(row)
            continue
        if split == "diagnostic_holdout":
            diagnostic_groups[row["cluster_id"]].append(row)
        elif split == "tuning_dev":
            tuning.append(row)
        else:
            assigned_train_candidates.append(row)
            reasons = []
            if token_counter:
                if "tokens" not in row:
                    reasons.append("tokenization_failed")
                else:
                    if row["tokens"]["prompt"] > config.prompt_limit:
                        reasons.append("prompt_too_long")
                    if max(row["tokens"]["response1"], row["tokens"]["response2"]) > config.response_limit:
                        reasons.append("response_too_long")
            if reasons:
                exclusions.append({"record_id": row["record_id"], "source_split": "train", "source_index": row["source_index"],
                                   "cluster_id": row["cluster_id"], "reason": "+".join(reasons)})
            else:
                train_pool.append(row)
    length_metadata, length_grid = training_length_grid(assigned_train_candidates, config)
    for cluster, members in sorted(diagnostic_groups.items()):
        diagnostics.append(min(members, key=lambda r: stable_rank(config.seed, "diagnostic-row", r["record_id"])))
    selected = stratified_limit(train_pool, config.pair_limit, config.seed, "preference-pairs")
    sft_candidates: dict[str, dict[str, Any]] = {}
    for row in selected:
        if abs(row["overall_preference"]) >= 2:
            previous = sft_candidates.get(row["cluster_id"])
            if previous is None or stable_rank(config.seed, "sft", row["record_id"]) < stable_rank(config.seed, "sft", previous["record_id"]):
                sft_candidates[row["cluster_id"]] = row
    sft = stratified_limit(list(sft_candidates.values()), config.sft_limit, config.seed, "sft-limit")
    ppo_candidates: dict[str, dict[str, Any]] = {}
    for row in selected:
        key = row["prompt_id"]
        if key not in ppo_candidates or row["record_id"] < ppo_candidates[key]["record_id"]:
            ppo_candidates[key] = row
    partitions = {"preference_train": selected, "eligible_train_pool": train_pool,
                  "sft_train": sft, "ppo_prompts": list(ppo_candidates.values()),
                  "tuning_dev": tuning, "diagnostic_holdout": diagnostics,
                  "final_holdout": final, "ties": ties, "exclusions": exclusions,
                  "assigned_train_lengths": length_metadata}
    stats = {name: summarize(items, config) for name, items in partitions.items()
             if name not in {"exclusions", "assigned_train_lengths"}}
    blockers = []
    if token_counter is None:
        blockers.append("local_tokenizer_missing: token lengths and final training eligibility unverified")
    if token_errors:
        blockers.append("tokenization_errors: template/prefix boundaries must be resolved")
    if any(r["source_split"] == "validation" for r in exclusions):
        blockers.append("invalid_official_validation_rows: final holdout coverage requires review; full original source retained")
    if not near["complete_for_declared_representation"]:
        blockers.append("near_duplicate_budget_exceeded: partial clustering must not be used for training")
    for name, expected in (("train", config.expected_train), ("validation", config.expected_validation)):
        if expected is not None and raw_counts[name] != expected:
            blockers.append(f"official_{name}_row_count_mismatch: {raw_counts[name]} != {expected}")
    if len(selected) < config.minimum_pairs:
        blockers.append(f"insufficient_preference_pairs: {len(selected)} < {config.minimum_pairs}")
    if stats["preference_train"]["independent_prompt_clusters"] < config.minimum_clusters:
        blockers.append(f"insufficient_prompt_clusters: {stats['preference_train']['independent_prompt_clusters']} < {config.minimum_clusters}")
    if len(sft) < config.minimum_sft:
        blockers.append(f"insufficient_sft_clusters: {len(sft)} < {config.minimum_sft}")
    if len(diagnostics) != config.diagnostic_clusters:
        blockers.append(f"insufficient_diagnostic_clusters: {len(diagnostics)} != {config.diagnostic_clusters}")
    # All train inputs must be disjoint from all holdouts, including their ties.
    split_clusters = {name: {r["cluster_id"] for r in items} for name, items in
                      (("train", selected), ("tuning_dev", tuning), ("diagnostic_holdout", diagnostics), ("final_holdout", final))}
    overlap = {f"{a}:{b}": len(split_clusters[a] & split_clusters[b]) for i, a in enumerate(split_clusters)
               for b in list(split_clusters)[i + 1:]}
    if any(overlap.values()):
        raise AssertionError(f"cluster leakage detected: {overlap}")
    coverage = {}
    for field in ("domain", "language"):
        coverage[field] = {value: summarize([r for r in rows if r[field] == value], config)
                           for value in sorted({r[field] for r in rows})}
    source_coverage = {split: {field: {value: summarize([r for r in rows if r["source_split"] == split and r[field] == value], config)
                                     for value in sorted({r[field] for r in rows if r["source_split"] == split})}
                              for field in ("domain", "language")}
                       for split in ("train", "validation")}
    report = {"raw_counts": raw_counts, "parsed": summarize(rows, config), "near_duplicates": near,
              "partition_statistics": stats, "coverage": coverage, "source_coverage": source_coverage,
              "cluster_overlaps": overlap,
              "exclusion_reasons": dict(Counter(r["reason"] for r in exclusions)),
              "tokenization_errors": dict(token_errors), "data_ready": not blockers,
              "native_nfc_normalization": {
                  "raw_text_preserved": True,
                  "normalizer": "NFC from pinned tokenizer; whitespace is not stripped",
                  "changed_rows_by_source": {split: {key: sum(bool(r.get("tokens", {}).get(key)) for r in rows
                                                               if r["source_split"] == split)
                                                       for key in ("prompt_nfc_changed", "response1_nfc_changed", "response2_nfc_changed")}
                                             for split in ("train", "validation")}},
              "training_only_length_grid": length_grid,
              "readiness_blockers": blockers, "training_started": False,
              "labels": {"source": "official overall_preference; negative selects response1, positive response2",
                         "ties": "zero scores preserved separately; never assigned chosen/rejected",
                         "individual_annotations": "agreement statistics only; no averaging or voting override; never model inputs"},
              "split_semantics": {"final_holdout": "official validation reserved as project final holdout, not an official test split",
                                  "fraction_unit": "prompt clusters within majority domain/language strata; row ratio may differ",
                                  "diagnostic": "one representative from each reserved independent internal-dev cluster",
                                  "sft": "at most one strong-preference row per cluster, selected from the common preference train set"},
              "serialization_version": SERIALIZATION_VERSION,
              "token_counting": "official non-thinking generation prompt; prompt and raw response encoded separately with pinned native NFC normalization, response plus one EOS; exact NFC-segment decode checked, raw text retained, no trailing template newline or whitespace stripping",
              "gpu_capacity_verified": False,
              "provisional_without_tokenizer": token_counter is None}
    return partitions, report


def public_row(row: dict[str, Any], name: str) -> dict[str, Any]:
    value = {key: item for key, item in row.items() if not key.startswith("_")}
    if name == "ppo_prompts":
        value = {key: value[key] for key in ("record_id", "prompt_id", "cluster_id", "context", "domain", "language", "source_split", "source_index")}
    return value


def input_identity(train: Path, validation: Path, tokenizer: Path | None, revision: str | None,
                   model_revision: str | None, config: AuditConfig) -> dict[str, Any]:
    source_files = [Path(__file__), ROOT / "alignment_lab/__init__.py", ROOT / "alignment_lab/chat_codec.py",
                    ROOT / "scripts/audit_alignlab_data.py"]
    tokenizer_files = {}
    if tokenizer:
        if not tokenizer.is_dir():
            raise ValueError("tokenizer must be an existing local directory")
        for path in sorted(tokenizer.rglob("*")):
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(tokenizer).parts):
                if path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                    raise ValueError("tokenizer directory must not contain model weights")
                tokenizer_files[path.relative_to(tokenizer).as_posix()] = file_hash(path)
        if "tokenizer.json" not in tokenizer_files or "tokenizer_config.json" not in tokenizer_files:
            raise ValueError("local tokenizer.json and tokenizer_config.json are required")
    packages = {}
    if tokenizer:
        for package in ("transformers", "tokenizers", "jinja2"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
    return {"version": VERSION, "serialization_version": SERIALIZATION_VERSION,
            "config": asdict(config), "source_revision": revision,
            "model_tokenizer_revision": model_revision,
            "inputs": {"train": {"path": str(train.resolve()), "sha256": file_hash(train)},
                       "validation": {"path": str(validation.resolve()), "sha256": file_hash(validation)}},
            "tokenizer": {"path": str(tokenizer.resolve()), "files": tokenizer_files} if tokenizer else None,
            "tokenizer_library_versions": packages,
            "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files},
            "protocol_sha256": file_hash(ROOT / "docs/ALIGNLAB_V2.md")}


def verify_existing(output: Path, identity: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("output already exists without a complete manifest; refusing overwrite")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "audit_complete":
        raise ValueError("existing output is not a completed audit")
    stored_identity = manifest.get("identity")
    if (not isinstance(stored_identity, dict) or not isinstance(identity, dict)
            or manifest.get("identity_sha256") != digest(stored_identity)):
        raise ValueError("existing output identity self-hash is invalid")
    if stored_identity != identity or manifest["identity_sha256"] != digest(identity):
        raise ValueError("existing output identity differs; use a new output directory")
    expected_artifacts = {
        "preference_train.jsonl", "eligible_train_pool.jsonl", "sft_train.jsonl", "ppo_prompts.jsonl",
        "tuning_dev.jsonl", "diagnostic_holdout.jsonl", "final_holdout.jsonl", "ties.jsonl",
        "exclusions.jsonl", "assigned_train_lengths.jsonl", "audit.json",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("existing output artifact inventory does not match the fixed complete set")
    for name, sha in artifacts.items():
        path = output / name
        if Path(name).name != name or not path.is_file() or file_hash(path) != sha:
            raise ValueError(f"existing artifact missing or changed: {name}")
    if {p.name for p in output.iterdir()} != expected_artifacts | {"manifest.json"}:
        raise ValueError("existing output contains untracked files; refusing reuse")
    # Parse the same bytes whose digest is checked, rather than trusting the
    # mutable manifest's readiness summary or a separately-read report snapshot.
    audit_bytes = (output / "audit.json").read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != artifacts["audit.json"]:
        raise ValueError("existing audit.json changed while verifying")
    audit = json.loads(audit_bytes)
    if not isinstance(audit, dict):
        raise ValueError("existing audit.json is not an object")
    for name, summary in (("manifest", manifest), ("audit.json", audit)):
        blockers = summary.get("readiness_blockers")
        if (type(summary.get("data_ready")) is not bool
                or not isinstance(blockers, list)
                or any(not isinstance(reason, str) or not reason.strip() for reason in blockers)
                or summary["data_ready"] != (len(blockers) == 0)):
            raise ValueError(f"existing {name} readiness flags/blockers are inconsistent")
        if summary.get("training_started") is not False:
            raise ValueError(f"existing {name} training_started must be false")
    for key in ("data_ready", "readiness_blockers", "training_started"):
        if manifest[key] != audit[key]:
            raise ValueError(f"existing manifest {key} differs from hashed audit.json")
    return manifest


def run_audit(train: Path, validation: Path, output: Path, tokenizer: Path | None = None,
              source_revision: str | None = None, model_revision: str | None = None,
              config: AuditConfig = AuditConfig(), progress: Callable[[str], None] = print) -> dict[str, Any]:
    config.validate()
    identity = input_identity(train, validation, tokenizer, source_revision, model_revision, config)
    if output.exists():
        result = verify_existing(output, identity)
        progress("Verified immutable existing output; reused without writing")
        return result
    started = time.monotonic()
    counter = LocalTokenCounter(tokenizer) if tokenizer else None
    partitions, report = audit_records(read_rows(train), read_rows(validation), config, counter, progress)
    if not source_revision:
        report["readiness_blockers"].append("upstream_dataset_revision_missing")
    if tokenizer and not model_revision:
        report["readiness_blockers"].append("upstream_tokenizer_revision_missing")
    report["data_ready"] = not report["readiness_blockers"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    # Check inputs again after a potentially long audit. Never seal mixed identities.
    if input_identity(train, validation, tokenizer, source_revision, model_revision, config) != identity:
        raise ValueError("source/config/input/tokenizer changed during audit; refusing to publish")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, values in partitions.items():
        path = output / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in sorted(values, key=lambda r: (r.get("record_id", ""), r.get("source_split", ""), r.get("source_index", 0))):
                stream.write(canonical(public_row(row, name)) + "\n")
        artifacts[path.name] = file_hash(path)
    path = output / "audit.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    artifacts[path.name] = file_hash(path)
    manifest = {"identity": identity, "identity_sha256": digest(identity), "artifacts": artifacts,
                "data_ready": report["data_ready"], "readiness_blockers": report["readiness_blockers"],
                "training_started": False, "status": "audit_complete"}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--model-revision")
    parser.add_argument("--seed", type=int, default=AuditConfig.seed)
    parser.add_argument("--candidate-budget", type=int, default=AuditConfig.candidate_budget)
    parser.add_argument("--posting-budget", type=int, default=AuditConfig.posting_budget)
    parser.add_argument("--prompt-limit", type=int, default=AuditConfig.prompt_limit,
                        help="Candidate training prompt limit; does not establish GPU capacity")
    parser.add_argument("--response-limit", type=int, default=AuditConfig.response_limit,
                        help="Candidate response-token limit including EOS; does not establish GPU capacity")
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.train, args.validation, args.output, args.tokenizer, args.source_revision,
                           args.model_revision, AuditConfig(seed=args.seed, candidate_budget=args.candidate_budget,
                                                            posting_budget=args.posting_budget, prompt_limit=args.prompt_limit,
                                                            response_limit=args.response_limit),
                           lambda text: print(text, flush=True))
    except (ValueError, OSError, ImportError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.resolve()), "data_ready": result["data_ready"],
                      "training_started": False, "readiness_blockers": result["readiness_blockers"]}, ensure_ascii=False), flush=True)
    return 0

