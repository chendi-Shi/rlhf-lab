"""Prepare auditable preference data without importing a training framework."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import unicodedata


DATASET_ID = "CarperAI/openai_summarize_comparisons"
DATASET_REVISION = "a320fae2cd84f04610a1b57ac83c2a2f674b8bcd"
SCHEMA_VERSION = 1
OUTPUT_FIELDS = {
    "sft_train.jsonl": ("id", "prompt", "response"),
    "sft_valid.jsonl": ("id", "prompt", "response"),
    "preferences_train.jsonl": ("id", "prompt", "chosen", "rejected"),
    "preferences_valid.jsonl": ("id", "prompt", "chosen", "rejected"),
    "preferences_test.jsonl": ("id", "prompt", "chosen", "rejected"),
    "ppo_train.jsonl": ("id", "prompt"),
    "eval_prompts.jsonl": ("id", "prompt", "reference"),
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(*values):
    return hashlib.sha256(_json(values).encode("utf-8")).hexdigest()


def normalize_prompt(text):
    """Identity key only: output text retains case and interior whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
            yield row


def read_jsonl(path):
    """Read a UTF-8 JSONL file, reporting malformed records with line numbers."""
    return list(_iter_jsonl(path))


def _clean(rows):
    counts = {"read": 0, "invalid": 0, "identical": 0, "duplicate": 0,
              "contradictory": 0, "accepted": 0}
    records = {}
    content_hash = hashlib.sha256()
    for raw in rows:
        counts["read"] += 1
        content_hash.update((_json(raw) + "\n").encode("utf-8"))
        if any(not isinstance(raw.get(key), str) or not raw[key].strip()
               for key in ("prompt", "chosen", "rejected")):
            counts["invalid"] += 1
            continue
        row = {key: unicodedata.normalize("NFC", raw[key]).strip()
               for key in ("prompt", "chosen", "rejected")}
        key = tuple(normalize_prompt(row[field]) for field in ("prompt", "chosen", "rejected"))
        if key[1] == key[2]:
            counts["identical"] += 1
            continue
        if key in records:
            counts["duplicate"] += 1
            # Stable representative even when source rows are reordered.
            if _json(row) < _json(records[key]):
                records[key] = row
        else:
            records[key] = row
    conflicts = {key for key in records if (key[0], key[2], key[1]) in records}
    counts["contradictory"] = len(conflicts)
    grouped = {}
    for key, row in records.items():
        if key in conflicts:
            continue
        row = {"id": _digest(*key), **row}
        grouped.setdefault(key[0], []).append(row)
    for rows_for_prompt in grouped.values():
        rows_for_prompt.sort(key=lambda row: row["id"])
    counts["accepted"] = sum(len(group) for group in grouped.values())
    return grouped, counts, content_hash.hexdigest()


def _limits(train_limit, valid_limit, test_limit):
    result = dict(train=train_limit, valid=valid_limit, test=test_limit)
    if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
           for limit in result.values()):
        raise ValueError("Each split limit must be a positive integer")
    return result


def _select(groups, keys, limit):
    return list(itertools.islice((row for key in keys for row in groups[key]), limit))


def _unique_prompts(rows):
    selected = {}
    for row in rows:
        selected.setdefault(normalize_prompt(row["prompt"]), row)
    return list(selected.values())


def _make_outputs(splits):
    output = {}
    for split in ("train", "valid", "test"):
        rows = splits[split]
        if not rows:
            raise ValueError(f"No usable {split} records; supply more distinct prompts or fix the source")
        output[f"preferences_{split}.jsonl"] = rows
    for split in ("train", "valid"):
        seen = set()
        output[f"sft_{split}.jsonl"] = []
        for row in splits[split]:
            key = (normalize_prompt(row["prompt"]), normalize_prompt(row["chosen"]))
            if key not in seen:
                seen.add(key)
                output[f"sft_{split}.jsonl"].append(
                    {"id": _digest(*key), "prompt": row["prompt"], "response": row["chosen"]})
    output["ppo_train.jsonl"] = [
        {"id": _digest(normalize_prompt(row["prompt"])), "prompt": row["prompt"]}
        for row in _unique_prompts(splits["train"])]
    output["eval_prompts.jsonl"] = [
        {"id": _digest(normalize_prompt(row["prompt"])), "prompt": row["prompt"],
         "reference": row["chosen"]} for row in _unique_prompts(splits["test"])]
    return output


def _write(output_dir, splits, source, counts, seed, limits, split_policy):
    output = _make_outputs(splits)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in output.items():
        target = directory / name
        content = "".join(_json(row) + "\n" for row in rows).encode("utf-8")
        target.write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "rows": len(rows)}
    manifest = {
        "schema_version": SCHEMA_VERSION, "source": source, "seed": seed,
        "requested_limits": limits, "split_policy": split_policy, "counts": counts,
        "samples": {name.removesuffix(".jsonl"): item["rows"] for name, item in files.items()},
        "unique_prompts": {split: len(_unique_prompts(rows)) for split, rows in splits.items()},
        "files": files,
        "notes": ["SFT uses preferred summaries from human comparison pairs, not a separate demonstration dataset.",
                  "A chosen summary is a preference label, not a factual gold reference.",
                  "PPO training prompts come only from train; final evaluation prompts come only from test."]}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_dataset(directory)
    return manifest


def prepare(input_path, output_dir, seed=42, train_limit=1024, valid_limit=128, test_limit=128):
    """Prepare local JSONL; an optional <input>.provenance.json declares its origin."""
    limits = _limits(train_limit, valid_limit, test_limit)
    input_path = Path(input_path)
    grouped, counts, _ = _clean(_iter_jsonl(input_path))
    keys = sorted(grouped, key=lambda key: _digest(str(seed), key))
    if len(keys) < 3:
        raise ValueError("At least three distinct usable prompt groups are required")
    holdout_count = max(1, len(keys) // 10)
    partitions = {"test": keys[:holdout_count], "valid": keys[holdout_count:2 * holdout_count],
                  "train": keys[2 * holdout_count:]}
    splits = {split: _select(grouped, partitions[split], limits[split]) for split in limits}
    provenance_path = Path(str(input_path) + ".provenance.json")
    declaration = {}
    if provenance_path.exists():
        declaration = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        if not isinstance(declaration, dict):
            raise ValueError("Local provenance must be a JSON object")
    source = {"kind": "local_jsonl", "path": str(input_path.resolve()),
              "sha256": _file_sha(input_path),
              "human_preferences": declaration.get("human_preferences") is True,
              "synthetic": declaration.get("synthetic") is True,
              "provenance_status": "user_declared" if declaration else "unverified",
              "declaration": declaration}
    if provenance_path.exists():
        source["provenance_sha256"] = _file_sha(provenance_path)
    return _write(output_dir, splits, source, {"all": counts}, seed, limits,
                  "Local source has no official split metadata: deterministic SHA-256 ordering of normalized "
                  "prompt groups, approximately 80/10/10 train/valid/test, then per-split pair limits. "
                  "All comparisons for a prompt stay within its assigned split.")


def prepare_hub(output_dir, seed=42, train_limit=1024, valid_limit=128, test_limit=128,
                dataset_name=DATASET_ID, revision=DATASET_REVISION):
    """Explicit network operation, using bounded streamed prefixes from pinned source splits."""
    if dataset_name != DATASET_ID:
        raise ValueError(f"Only the verified {DATASET_ID} adapter is supported; use prepare() for other data")
    if not revision:
        revision = DATASET_REVISION
    limits = _limits(train_limit, valid_limit, test_limit)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project training dependencies before requesting a Hub download") from exc
    source_splits = {"train": "train", "valid": "valid1", "test": "test"}
    splits, counts, content_hashes, scan_limits = {}, {}, {}, {}
    heldout_prompts = set()
    # Test has precedence. No exported held-out prompt can occur in an earlier training stage.
    for split in ("test", "valid", "train"):
        scan_limit = max(2048, limits[split] * 32)
        stream = load_dataset(dataset_name, split=source_splits[split], streaming=True, revision=revision)
        groups, stats, content_hash = _clean(itertools.islice(stream, scan_limit))
        excluded = set(groups) & heldout_prompts
        stats["overlap_pairs_removed"] = sum(len(groups[key]) for key in excluded)
        keys = sorted(set(groups) - heldout_prompts, key=lambda key: _digest(str(seed), key))
        splits[split] = _select(groups, keys, limits[split])
        heldout_prompts.update(normalize_prompt(row["prompt"]) for row in splits[split])
        counts[split], content_hashes[split], scan_limits[split] = stats, content_hash, scan_limit
    source = {"kind": "huggingface", "dataset_id": dataset_name, "revision": revision,
              "url": f"https://huggingface.co/datasets/{dataset_name}/tree/{revision}",
              "original_source": "https://github.com/openai/summarize-from-feedback",
              "human_preferences": True, "synthetic": False,
              "provenance_status": "public_human_comparison_dataset",
              "split_mapping": source_splits, "canonical_stream_sha256": content_hashes,
              "stream_scan_limits": scan_limits,
              "sampling": "Deterministically hash-rank bounded source prefixes, not a uniform full-dataset sample."}
    return _write(output_dir, splits, source, counts, seed, limits,
                  "Preserve source train/valid1/test splits; valid2 is unused. Remove normalized prompt overlap "
                  "between exported splits with test > valid > train precedence. Sample pairs from whole "
                  "prompt groups in deterministic seeded hash order within each streamed prefix.")


def validate_dataset(directory):
    """Check checksums, schemas, split isolation, and consistency of derived training files."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema")
    if not isinstance(manifest.get("source"), dict):
        raise ValueError("Dataset manifest has no provenance")
    records = {}
    for name, fields in OUTPUT_FIELDS.items():
        meta = manifest.get("files", {}).get(name)
        if not meta or _file_sha(directory / name) != meta.get("sha256"):
            raise ValueError(f"Checksum mismatch or missing manifest entry: {name}")
        rows = read_jsonl(directory / name)
        if not rows or len(rows) != meta.get("rows"):
            raise ValueError(f"Empty file or row count mismatch: {name}")
        ids = set()
        for row in rows:
            if any(not isinstance(row.get(field), str) or not row[field].strip() for field in fields):
                raise ValueError(f"Missing or invalid fields: {name}")
            if row["id"] in ids:
                raise ValueError(f"Duplicate id: {name}")
            ids.add(row["id"])
            if "chosen" in fields and normalize_prompt(row["chosen"]) == normalize_prompt(row["rejected"]):
                raise ValueError(f"Identical preference pair: {name}")
        records[name] = rows
    splits = {split: records[f"preferences_{split}.jsonl"] for split in ("train", "valid", "test")}
    prompt_sets = {split: {normalize_prompt(row["prompt"]) for row in rows} for split, rows in splits.items()}
    for left, right in itertools.combinations(prompt_sets, 2):
        if prompt_sets[left] & prompt_sets[right]:
            raise ValueError(f"Prompt leakage between {left} and {right}")
    expected = _make_outputs(splits)
    for name in OUTPUT_FIELDS:
        if records[name] != expected[name]:
            raise ValueError(f"Derived data inconsistent with preferences: {name}")
    actual_samples = {name.removesuffix(".jsonl"): len(rows) for name, rows in records.items()}
    if manifest.get("samples") != actual_samples:
        raise ValueError("Manifest sample counts do not match files")
    return {"valid": True, "samples": actual_samples, "unique_prompts": {
        split: len(keys) for split, keys in prompt_sets.items()}, "source": manifest["source"]}
