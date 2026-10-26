"""Offline evaluation utilities. No model runtime or GPU is needed here."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


GENERATION_FIELDS = {"id", "prompt", "response", "model_label", "model_path", "seed"}
BLIND_FIELDS = ["pair_id", "prompt", "response_a", "response_b", "judgment"]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_generations(path: str | Path) -> list[dict]:
    """Validate a generation JSONL, including duplicate IDs and model identity."""
    rows = []
    seen = set()
    identity = None
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or not GENERATION_FIELDS.issubset(row):
                raise ValueError(f"{path}:{line_number}: missing generation fields")
            for key in ("id", "prompt", "model_label", "model_path"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_number}: {key} must be a nonempty string")
            if not isinstance(row["response"], str):
                raise ValueError(f"{path}:{line_number}: response must be a string")
            if type(row["seed"]) is not int:
                raise ValueError(f"{path}:{line_number}: seed must be an integer")
            if "reference" in row and not isinstance(row["reference"], str):
                raise ValueError(f"{path}:{line_number}: reference must be a string")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']!r}")
            current_identity = (row["model_label"], row["model_path"])
            if identity is not None and current_identity != identity:
                raise ValueError(f"{path}:{line_number}: mixed model identities")
            identity = current_identity
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: generation file is empty")
    return rows


def export_blind_pairs(
    left_path: str | Path,
    right_path: str | Path,
    csv_path: str | Path,
    mapping_path: str | Path,
    seed: int = 42,
) -> dict:
    """Create a blinded A/B CSV and a separate, private identity mapping.

    A/B order is seeded independently for each prompt and stays stable when input
    rows are reordered. Both input files must contain exactly the same IDs/prompts.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    destinations = (Path(csv_path), Path(mapping_path))
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("blind CSV and private mapping must be separate files")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)
    _check_generation_manifests(left_path, right_path)
    left = {row["id"]: row for row in load_generations(left_path)}
    right = {row["id"]: row for row in load_generations(right_path)}
    if left.keys() != right.keys():
        raise ValueError("generation files must have exactly matched IDs")
    first_left, first_right = next(iter(left.values())), next(iter(right.values()))
    if first_left["model_label"] == first_right["model_label"]:
        raise ValueError("comparison model labels must be different")
    public_rows = []
    private_pairs = []
    for sample_id in sorted(left):
        first, second = left[sample_id], right[sample_id]
        if first["prompt"] != second["prompt"]:
            raise ValueError(f"prompt mismatch for id {sample_id!r}")
        if "reference" in first and "reference" in second and first["reference"] != second["reference"]:
            raise ValueError(f"reference mismatch for id {sample_id!r}")
        identity = json.dumps([seed, sample_id, first["prompt"]], ensure_ascii=False)
        pair_id = _digest("pair:" + identity)
        a_side = random.Random(_digest("order:" + identity)).choice(("left", "right"))
        a, b = (first, second) if a_side == "left" else (second, first)
        public_rows.append({
            "pair_id": pair_id,
            "prompt": first["prompt"],
            "response_a": a["response"],
            "response_b": b["response"],
            "judgment": "",
        })
        private_pairs.append({
            "pair_id": pair_id,
            "id": sample_id,
            "a_side": a_side,
            "b_side": "right" if a_side == "left" else "left",
            "prompt_sha256": _digest(first["prompt"]),
            "response_a_sha256": _digest(a["response"]),
            "response_b_sha256": _digest(b["response"]),
        })
    random.Random(seed).shuffle(public_rows)
    mapping = {
        "schema_version": 1,
        "seed": seed,
        "left_model": {key: first_left[key] for key in ("model_label", "model_path")},
        "right_model": {key: first_right[key] for key in ("model_label", "model_path")},
        "pairs": private_pairs,
    }
    # All inputs are validated before either output is written. Never replace files.
    _write_json(mapping_path, mapping)
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    with destinations[0].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(public_rows)
    return {"pairs": len(public_rows), "blind_csv": str(csv_path), "private_mapping": str(mapping_path)}


def _check_generation_manifests(left_path: str | Path, right_path: str | Path) -> None:
    """Validate provenance/settings when outputs carry generation sidecars."""
    paths = [Path(str(path) + ".manifest.json") for path in (left_path, right_path)]
    if not any(path.exists() for path in paths):
        return  # External generation JSONL remains supported; review settings manually.
    if not all(path.is_file() for path in paths):
        raise ValueError("both generated outputs must have their generation manifest sidecars")
    manifests = []
    for path, output in zip(paths, (left_path, right_path)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            raise ValueError("blind evaluation requires completed generation manifests")
        digest = hashlib.sha256()
        with Path(output).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if manifest.get("output_sha256") != digest.hexdigest():
            raise ValueError("generation JSONL no longer matches its manifest checksum")
        if not isinstance(manifest.get("config"), dict):
            raise ValueError("generation manifest is missing its config")
        manifests.append(manifest)
    for key in ("model_id", "model_revision", "eval_split", "prompts_sha256", "dataset_manifest_sha256", "decoding", "inference_dtype"):
        if key not in manifests[0] or key not in manifests[1] or manifests[0][key] != manifests[1][key]:
            raise ValueError(f"generation settings mismatch: {key}")
    for key in ("max_prompt_tokens", "max_response_tokens", "seed"):
        if key not in manifests[0]["config"] or key not in manifests[1]["config"] or manifests[0]["config"][key] != manifests[1]["config"][key]:
            raise ValueError(f"generation settings mismatch: {key}")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    """95% Wilson interval by default; undefined when there are no trials."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError("require integer 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _load_mapping(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or mapping.get("schema_version") != 1:
        raise ValueError("unsupported private mapping schema")
    for side in ("left_model", "right_model"):
        model = mapping.get(side)
        if not isinstance(model, dict) or any(not isinstance(model.get(key), str) or not model[key].strip() for key in ("model_label", "model_path")):
            raise ValueError("invalid model identity in private mapping")
    pairs = mapping.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("private mapping must contain pairs")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("invalid private mapping pair")
        for key in ("pair_id", "id", "prompt_sha256", "response_a_sha256", "response_b_sha256"):
            if not isinstance(pair.get(key), str) or not pair[key]:
                raise ValueError(f"invalid {key} in private mapping")
        if pair["pair_id"] in seen:
            raise ValueError("duplicate pair in private mapping")
        if {pair.get("a_side"), pair.get("b_side")} != {"left", "right"}:
            raise ValueError("invalid A/B sides in private mapping")
        seen.add(pair["pair_id"])
    return mapping


def score_judgments(
    csv_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Score actual A/B/tie judgments from the left model's perspective.

    Blank judgments remain unjudged. Ties are excluded from the binomial Wilson
    interval, and the tie-adjusted score is reported separately without that CI.
    """
    mapping = _load_mapping(mapping_path)
    pairs = {pair["pair_id"]: pair for pair in mapping["pairs"]}
    seen = set()
    counts = Counter({"wins": 0, "ties": 0, "losses": 0, "unjudged": 0})
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != BLIND_FIELDS:
            raise ValueError(f"judgment CSV header must be {BLIND_FIELDS!r}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed judgment CSV row")
            pair_id = row["pair_id"]
            if pair_id not in pairs or pair_id in seen:
                raise ValueError("unknown or duplicate pair_id in judgment CSV")
            pair = pairs[pair_id]
            for field in ("prompt", "response_a", "response_b"):
                if _digest(row[field]) != pair[field + "_sha256"]:
                    raise ValueError(f"{field} changed for pair {pair_id}")
            seen.add(pair_id)
            judgment = row["judgment"].strip().lower()
            if judgment == "":
                counts["unjudged"] += 1
            elif judgment == "tie":
                counts["ties"] += 1
            elif judgment in ("a", "b"):
                counts["wins" if pair[judgment + "_side"] == "left" else "losses"] += 1
            else:
                raise ValueError("judgment must be A, B, tie, or blank")
    if seen != pairs.keys():
        raise ValueError("judgment CSV is missing pairs; keep rows and leave judgments blank")
    judged = counts["wins"] + counts["ties"] + counts["losses"]
    decisive = counts["wins"] + counts["losses"]
    result = {
        "schema_version": 1,
        "left_model": mapping["left_model"],
        "right_model": mapping["right_model"],
        "perspective": "left_model",
        "total_pairs": len(pairs),
        **dict(counts),
        "judged": judged,
        "decisive_judgments": decisive,
        "decisive_win_rate": counts["wins"] / decisive if decisive else None,
        "decisive_win_rate_wilson_95": wilson_interval(counts["wins"], decisive),
        "tie_adjusted_score": (counts["wins"] + 0.5 * counts["ties"]) / judged if judged else None,
        "notes": [
            "Only entered human judgments are counted; blank judgments never become wins.",
            "Wilson interval describes wins among decisive judgments and excludes ties.",
            "The interval assumes independent examples; a small or biased sample is not proof of general superiority.",
        ],
    }
    if output_path is not None:
        _write_json(output_path, result)
    return result


def response_proxies(response: str) -> dict:
    """Simple text diagnostics, not a factuality or human-quality metric."""
    tokens = re.findall(r"\w+|[^\w\s]", response.lower(), flags=re.UNICODE)
    grams = [tuple(tokens[index:index + 3]) for index in range(max(0, len(tokens) - 2))]
    return {
        "characters": len(response),
        "regex_tokens": len(tokens),
        "empty": not response.strip(),
        "repeated_trigram_fraction_proxy": (len(grams) - len(set(grams))) / len(grams) if grams else 0.0,
    }


def summarize_generations(input_path: str | Path, output_path: str | Path | None = None) -> dict:
    rows = load_generations(input_path)
    diagnostics = [response_proxies(row["response"]) for row in rows]
    count = len(rows)
    report = {
        "schema_version": 1,
        "model_label": rows[0]["model_label"],
        "model_path": rows[0]["model_path"],
        "examples": count,
        "empty_responses": sum(row["empty"] for row in diagnostics),
        "mean_characters": sum(row["characters"] for row in diagnostics) / count,
        "mean_regex_tokens": sum(row["regex_tokens"] for row in diagnostics) / count,
        "mean_repeated_trigram_fraction_proxy": sum(row["repeated_trigram_fraction_proxy"] for row in diagnostics) / count,
        "notes": [
            "Repetition and length are diagnostic proxies, not measures of summary quality or factual correctness.",
            "Regex token counts are not model tokenizer counts; results depend on language and punctuation.",
            "Reward-model scores, if collected elsewhere, must not be reported as human preference wins.",
        ],
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report
