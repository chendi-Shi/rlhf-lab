"""Training-only, deterministic capacity workloads; never quality evaluation.

The caller supplies a TrainingPartition returned by load_training_partition.
This module does not read datasets, tokenize text, alter an accepted audit, or
assert that a user-constructed TrainingPartition has passed that loader.
Synthetic rows are explicit shape stressors, not human preference examples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .training_data import TrainingPartition, TrainingRow

DEFAULT_COUNTS = {"sft": 16, "reward": 16, "dpo": 320, "ppo": 128}
PARTITIONS = {"sft": "sft_train", "reward": "preference_train",
              "dpo": "preference_train", "ppo": "ppo_prompts"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tokens(ids, vocab_size, name):
    if (type(ids) is not tuple or not ids
            or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
        raise ValueError(f"{name} requires nonempty exact vocabulary token tuples")


def _token_payload(row):
    return {"prompt_ids": row.prompt_ids, "chosen_ids": row.chosen_ids,
            "rejected_ids": row.rejected_ids}


@dataclass(frozen=True)
class CapacitySelection:
    rows: tuple[TrainingRow, ...]
    _metadata_json: bytes

    @property
    def metadata(self):
        """Detached JSON copy; callers cannot change the stored selection."""
        return json.loads(self._metadata_json)


def make_capacity_stress_row(rows, *, stage, eos_id, special_token_ids,
                             vocab_size, prompt_limit=1024, completion_limit=768):
    """Repeat actual training non-special IDs to an exact unpadded shape.

    Exactly one EOS terminates each completion. Chosen/rejected content differs
    by construction; their ordering is synthetic and carries no quality claim.
    No chat-template or natural-language claim is made for the stress prompt.
    """
    if stage not in {"sft", "reward", "dpo"}:
        raise ValueError("Synthetic supervised stress stage must be sft, reward or dpo")
    for name, value in (("vocab_size", vocab_size), ("prompt_limit", prompt_limit),
                        ("completion_limit", completion_limit)):
        _positive(value, name)
    if completion_limit < 2:
        raise ValueError("Synthetic completion needs content plus terminal EOS")
    if type(eos_id) is not int or not 0 <= eos_id < vocab_size:
        raise ValueError("EOS must be an exact vocabulary ID")
    specials = frozenset(special_token_ids)
    if (eos_id not in specials or any(type(t) is not int or not 0 <= t < vocab_size for t in specials)):
        raise ValueError("Explicit special IDs must include EOS and be in vocabulary")
    rows = tuple(rows)
    if not rows or any(not isinstance(row, TrainingRow) for row in rows):
        raise ValueError("Stress token sources must be nonempty training rows")
    provenance, content = {}, set()
    for row in rows:
        for field in ("prompt_ids", "chosen_ids", "rejected_ids"):
            ids = getattr(row, field)
            if ids is None:
                continue
            _tokens(ids, vocab_size, field)
            for token in ids:
                if token not in specials:
                    content.add(token)
                    provenance.setdefault(token, row.record_id)
    content = sorted(content)
    needed = 1 if stage == "sft" else 2
    if len(content) < needed:
        raise ValueError("Not enough distinct non-special training tokens for stress")
    a, b = content[0], content[-1]
    prompt = tuple(content[i % len(content)] for i in range(prompt_limit))
    chosen = (a,) * (completion_limit - 1) + (eos_id,)
    rejected = None if stage == "sft" else (b,) * (completion_limit - 1) + (eos_id,)
    token_hash = _digest({"prompt_ids": prompt, "chosen_ids": chosen, "rejected_ids": rejected})
    identity = "synthetic-capacity-" + stage + "-" + token_hash
    row = TrainingRow(identity, identity, identity, prompt, chosen, rejected)
    metadata = {
        "kind": "synthetic_worst_shape", "capacity_only": True,
        "preference_label_is_synthetic": stage != "sft", "token_sha256": token_hash,
        "source_record_ids": sorted(set(provenance.values())),
        "construction": "Repeat real training non-special IDs; one final EOS; no padding",
        "prompt_tokens": prompt_limit, "chosen_tokens": completion_limit,
        "rejected_tokens": None if rejected is None else completion_limit,
        "attention_all_true": True, "padding_tokens": 0,
    }
    return row, metadata


def select_capacity_rows(partition, *, stage, seed=42, count=None, eos_id,
                         special_token_ids, vocab_size, prompt_limit=1024,
                         completion_limit=768, include_synthetic=True):
    """Select fixed training rows with long-shape coverage and seeded variety.

    Half of the real-row slots prioritize prompt/response/total length in turn;
    remaining slots use a SHA-256 seeded order. Distinct clusters are preferred
    throughout and relaxed only when exhausted. PPO always has distinct prompt
    IDs and contains zero synthetic rows. Other stages replace one budgeted row
    with a labeled synthetic stress row when include_synthetic=True.
    """
    if stage not in DEFAULT_COUNTS:
        raise ValueError("Unknown capacity stage")
    if not isinstance(partition, TrainingPartition) or partition.partition != PARTITIONS[stage]:
        raise ValueError("Stage requires its accepted training-only partition")
    for name, value in (("vocab_size", vocab_size), ("prompt_limit", prompt_limit),
                        ("completion_limit", completion_limit)):
        _positive(value, name)
    if type(seed) is not int or seed < 0 or type(include_synthetic) is not bool:
        raise ValueError("Seed must be a nonnegative integer and synthetic flag boolean")
    if type(eos_id) is not int or not 0 <= eos_id < vocab_size:
        raise ValueError("EOS must be an exact vocabulary ID")
    if count is None:
        count = DEFAULT_COUNTS[stage]
    _positive(count, "count")
    synthetic_count = int(include_synthetic and stage != "ppo")
    real_count = count - synthetic_count
    if real_count < 1:
        raise ValueError("At least one real training row is required")
    rows = tuple(partition.rows)
    seen = set()
    for row in rows:
        if not isinstance(row, TrainingRow):
            raise ValueError("Partition must contain immutable TrainingRow values")
        for name in ("record_id", "cluster_id", "prompt_id"):
            if not isinstance(getattr(row, name), str) or not getattr(row, name):
                raise ValueError("Nonempty record, cluster and prompt identities required")
        if row.record_id in seen or row.record_id.startswith("synthetic-capacity-"):
            raise ValueError("Accepted source cannot contain duplicate or synthetic record IDs")
        seen.add(row.record_id)
        _tokens(row.prompt_ids, vocab_size, "prompt_ids")
        if len(row.prompt_ids) > prompt_limit:
            raise ValueError("Real prompt exceeds capacity limit; no silent truncation")
        expected = () if stage == "ppo" else (("chosen_ids",) if stage == "sft"
                                                else ("chosen_ids", "rejected_ids"))
        for field in expected:
            ids = getattr(row, field)
            _tokens(ids, vocab_size, field)
            if len(ids) > completion_limit or ids[-1] != eos_id or eos_id in ids[:-1]:
                raise ValueError("Real completion requires length limit and one terminal EOS")
        if stage == "ppo" and (row.chosen_ids is not None or row.rejected_ids is not None):
            raise ValueError("PPO prompt partition cannot contain completion labels")
    if len(rows) < real_count:
        raise ValueError("Insufficient distinct real rows")
    if stage == "ppo" and len({row.prompt_id for row in rows}) < real_count:
        raise ValueError("Insufficient distinct PPO prompts")
    seeded = lambda row: (_digest([seed, stage, row.record_id]), row.record_id)
    ranked = sorted(rows, key=seeded)
    length = lambda row, field: len(getattr(row, field) or ())
    metrics = [lambda r: len(r.prompt_ids)]
    if stage != "ppo":
        metrics.append(lambda r: length(r, "chosen_ids"))
        if stage != "sft":
            metrics.append(lambda r: length(r, "rejected_ids"))
        metrics.append(lambda r: len(r.prompt_ids) + max(length(r, "chosen_ids"), length(r, "rejected_ids")))
    selected, used_rows, used_clusters, used_prompts = [], set(), set(), set()
    long_count = (real_count + 1) // 2
    for index in range(real_count):
        candidates = [r for r in ranked if r.record_id not in used_rows
                      and (stage != "ppo" or r.prompt_id not in used_prompts)]
        independent = [r for r in candidates if r.cluster_id not in used_clusters]
        candidates = independent or candidates
        if not candidates:
            raise ValueError("Cannot satisfy unique selected row/prompt budget")
        if index < long_count:
            score = metrics[index % len(metrics)]
            row = min(candidates, key=lambda r: (-score(r), seeded(r)))
        else:
            row = candidates[0]
        selected.append(row)
        used_rows.add(row.record_id); used_clusters.add(row.cluster_id); used_prompts.add(row.prompt_id)
    stress_metadata = None
    if synthetic_count:
        stress, stress_metadata = make_capacity_stress_row(selected, stage=stage, eos_id=eos_id,
            special_token_ids=special_token_ids, vocab_size=vocab_size, prompt_limit=prompt_limit,
            completion_limit=completion_limit)
        selected.insert(0, stress)
    details = []
    for row in selected:
        synthetic = row.record_id.startswith("synthetic-capacity-")
        details.append({"record_id": row.record_id, "cluster_id": row.cluster_id, "prompt_id": row.prompt_id,
            "kind": "synthetic_worst_shape" if synthetic else "accepted_training",
            "prompt_tokens": len(row.prompt_ids), "chosen_tokens": None if row.chosen_ids is None else len(row.chosen_ids),
            "rejected_tokens": None if row.rejected_ids is None else len(row.rejected_ids),
            "token_sha256": _digest(_token_payload(row)), "attention_all_true_before_batching": True})
    metadata = {"schema_version": 1, "capacity_only": True, "quality_evaluation": False,
        "stage": stage, "seed": seed, "partition": partition.partition,
        "accepted_manifest_sha256": partition.manifest_sha256,
        "accepted_identity_sha256": partition.identity_sha256,
        "serialization_version": partition.serialization_version,
        "configured_rows": count, "selected_rows": len(selected), "real_rows": real_count,
        "synthetic_rows": synthetic_count, "real_unique_clusters": len(used_clusters),
        "real_unique_prompts": len(used_prompts), "source_unique_clusters": len({r.cluster_id for r in rows}),
        "long_priority_real_slots": long_count,
        "selection_rule": "Seeded SHA order; alternating longest dimensions for first half; unused clusters first",
        "prompt_limit": prompt_limit, "completion_limit": completion_limit,
        "rows": details, "synthetic_stress": stress_metadata,
        "tokens_sha256": _digest([_token_payload(r) for r in selected])}
    return CapacitySelection(tuple(selected), json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
