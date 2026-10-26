"""Offline context audit of prepared train/valid only; never opens test labels.

Uses the local Rust tokenizer and pure token-list helpers, without importing
torch or transformers. The destination is created exclusively and never replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
PROMPT_BUDGETS = (192, 384, 512, 576, 768)
RESPONSE_BUDGETS = (64, 96, 128)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pure_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class LocalTokenizer:
    def __init__(self, path, eos_token_id):
        self.backend = Tokenizer.from_file(str(path))
        self.backend.no_truncation()
        self.backend.no_padding()
        self.eos_token_id = eos_token_id

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    result = {
        "count": len(ordered), "min": ordered[0], "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }
    for percentile in (50, 90, 95, 99):
        result[f"p{percentile}"] = ordered[math.ceil(len(ordered) * percentile / 100) - 1]
    return result


def count_rate(count, total):
    return {"count": count, "denominator": total, "rate": count / total if total else None}


def clipped_response(ids, budget, eos):
    return ids[:budget] if len(ids) >= budget else ids + [eos]


def audit_split(path, tokenizer, tokenization, normalize_prompt):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    encoded = []
    unique_prompts = {}
    unlimited = {"max_prompt_tokens": 2**31 - 1, "max_response_tokens": 2**31 - 1,
                 "completion_eos_policy": "natural"}
    for row in rows:
        key = normalize_prompt(row["prompt"])
        prompt_ids = tokenization.encode_prompt(tokenizer, row["prompt"], unlimited)
        chosen = tokenization.encode_response(tokenizer, row["chosen"], unlimited)[:-1]
        rejected = tokenization.encode_response(tokenizer, row["rejected"], unlimited)[:-1]
        unique_prompts.setdefault(key, []).append(len(prompt_ids))
        encoded.append({"prompt_key": key, "prompt_length": len(prompt_ids),
                        "chosen": chosen, "rejected": rejected})
    # Normalized-equivalent text can in principle tokenize differently. Count a
    # group as fully covered only when every observed rendering is covered.
    unique_lengths = [max(lengths) for lengths in unique_prompts.values()]
    prompt_stats = {}
    for budget in PROMPT_BUDGETS:
        prompt_stats[str(budget)] = {
            "pair_weighted_full_coverage": count_rate(sum(x["prompt_length"] <= budget for x in encoded), len(rows)),
            "pair_weighted_truncation": count_rate(sum(x["prompt_length"] > budget for x in encoded), len(rows)),
            "unique_prompt_full_coverage": count_rate(sum(n <= budget for n in unique_lengths), len(unique_lengths)),
            "unique_prompt_truncation": count_rate(sum(n > budget for n in unique_lengths), len(unique_lengths)),
        }
    response_stats = {}
    full_pair_grid = {}
    for budget in RESPONSE_BUDGETS:
        response_stats[str(budget)] = {}
        for side in ("chosen", "rejected"):
            lengths = [len(x[side]) for x in encoded]
            response_stats[str(budget)][side] = {
                "full_content_and_natural_eos": count_rate(sum(n < budget for n in lengths), len(rows)),
                "full_content_only": count_rate(sum(n <= budget for n in lengths), len(rows)),
                "content_truncated": count_rate(sum(n > budget for n in lengths), len(rows)),
                "natural_eos_omitted": count_rate(sum(n >= budget for n in lengths), len(rows)),
            }
        same = sum(clipped_response(x["chosen"], budget, tokenizer.eos_token_id) ==
                   clipped_response(x["rejected"], budget, tokenizer.eos_token_id) for x in encoded)
        response_stats[str(budget)]["pair_identical_after_encoding"] = count_rate(same, len(rows))
        response_stats[str(budget)]["pair_newly_identical_due_to_truncation"] = count_rate(sum(
            x["chosen"] != x["rejected"] and
            clipped_response(x["chosen"], budget, tokenizer.eos_token_id) ==
            clipped_response(x["rejected"], budget, tokenizer.eos_token_id) for x in encoded), len(rows))
        for prompt_budget in PROMPT_BUDGETS:
            kept = [x for x in encoded if x["prompt_length"] <= prompt_budget
                    and len(x["chosen"]) < budget and len(x["rejected"]) < budget
                    and x["chosen"] != x["rejected"]]
            full_pair_grid[f"prompt{prompt_budget}_response{budget}"] = {
                "pairs": len(kept), "unique_prompts": len({x["prompt_key"] for x in kept}),
                "pair_retention_rate": len(kept) / len(rows),
            }
    # Cross-check local slicing against the existing training helper at all
    # requested budgets. This performs no model construction or GPU operations.
    for row, item in zip(rows, encoded):
        for budget in RESPONSE_BUDGETS:
            config = {**unlimited, "max_response_tokens": budget}
            for side in ("chosen", "rejected"):
                assert clipped_response(item[side], budget, tokenizer.eos_token_id) == tokenization.encode_response(tokenizer, row[side], config)
    return {
        "pairs": len(rows), "unique_prompts": len(unique_prompts),
        "prompt_tokens_including_marker_pair_weighted": distribution([x["prompt_length"] for x in encoded]),
        "prompt_tokens_including_marker_unique_prompt": distribution(unique_lengths),
        "chosen_content_tokens": distribution([len(x["chosen"]) for x in encoded]),
        "rejected_content_tokens": distribution([len(x["rejected"]) for x in encoded]),
        "prompt_budgets": prompt_stats, "response_budgets": response_stats,
        "fully_retained_pair_grid": full_pair_grid,
        "identical_pairs_without_truncation": sum(x["chosen"] == x["rejected"] for x in encoded),
        "response_encoding_verified_against_training_helper": True,
    }, set(unique_prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/context-audit-20261020.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    dataset = ROOT / "data/processed/pilot"
    model = ROOT / "models/gpt2"
    manifest_path = dataset / "manifest.json"
    provenance_path = ROOT / "data/raw/pilot_rows.jsonl.provenance.json"
    model_manifest = load_json(model / "local-manifest.json")
    model_config = load_json(model / "config.json")
    manifest = load_json(manifest_path)
    provenance = load_json(provenance_path)
    tokenizer_path = model / "tokenizer.json"
    tokenizer_hash = sha256(tokenizer_path)
    assert tokenizer_hash == model_manifest["files"]["tokenizer.json"]["sha256"], "Tokenizer integrity mismatch"
    tokenization_path = ROOT / "rlhf_lab/tokenization.py"
    data_source_path = ROOT / "rlhf_lab/data.py"
    tokenization = load_pure_module("audit_tokenization", tokenization_path)
    data_helpers = load_pure_module("audit_data", data_source_path)
    tokenizer = LocalTokenizer(tokenizer_path, model_config["eos_token_id"])
    split_results = {}
    prompt_keys = {}
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in
              (manifest_path, provenance_path, tokenizer_path, tokenization_path, data_source_path)}
    for split in ("train", "valid"):
        path = dataset / f"preferences_{split}.jsonl"
        actual = sha256(path)
        assert actual == manifest["files"][path.name]["sha256"], f"Data integrity mismatch: {split}"
        inputs[str(path.relative_to(ROOT))] = actual
        split_results[split], prompt_keys[split] = audit_split(path, tokenizer, tokenization, data_helpers.normalize_prompt)
        assert split_results[split]["pairs"] == manifest["files"][path.name]["rows"]
    overlap = len(prompt_keys["train"] & prompt_keys["valid"])
    assert overlap == 0, "Prepared train and valid overlap"
    counts = manifest["counts"]["all"]
    recorded_selected = sum(manifest["samples"][f"preferences_{split}"] for split in ("train", "valid", "test"))
    raw_hash_counts = Counter(row["canonical_row_sha256"] for row in provenance["row_hashes"])
    source_totals = sorted({page["num_rows_total"] for page in provenance["page_requests"]})
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha256(Path(__file__).resolve()), "input_sha256": inputs,
        "model": {"id": model_manifest["model_id"], "revision": model_manifest["revision"],
                  "max_positions": model_config["n_positions"], "eos_token_id": tokenizer.eos_token_id},
        "method": {
            "offline": True, "model_loaded": False, "gpu_used": False,
            "torch_imported": "torch" in sys.modules, "transformers_imported": "transformers" in sys.modules,
            "labels_decoded": ["preferences_train.jsonl", "preferences_valid.jsonl"],
            "test_labels_opened": False, "raw_rows_opened": False, "split_assignments_changed": False,
            "prompt_marker": "\\n\\nTL;DR:", "prompt_budgets_include_marker": True,
            "response_preprocessing": "Strip leading TL;DR marker, strip whitespace, prepend a space, tokenize, remove terminal EOS if present.",
            "response_budget_includes_natural_eos": True,
            "content_truncated_definition": "content token count > budget",
            "natural_eos_omitted_definition": "content token count >= budget, matching completion_eos_policy=natural",
            "fully_retained_pair_definition": "Provided prompt plus marker fits prompt budget; both summaries plus natural EOS fit response budget; encoded summaries differ.",
            "quantile_method": "nearest rank", "unique_prompt_length": "Maximum over normalized-equivalent renderings",
            "caveat": "Full prompt coverage means all text provided in this dataset, not proof that the original source article was complete.",
        },
        "splits": split_results, "train_valid_normalized_prompt_overlap": overlap,
        "raw_source_metadata_only": {
            "source_split": provenance["source_split"], "dataset_id": provenance["dataset_id"],
            "expected_revision": provenance["expected_dataset_sha"],
            "downloaded_rows": provenance["actual_rows"], "sampling": provenance["sampling"],
            "official_train_total_as_recorded_at_download": source_totals,
            "raw_row_hashes": len(provenance["row_hashes"]), "unique_raw_row_hashes": len(raw_hash_counts),
            "recorded_cleaning_counts": counts, "selected_pairs_all_local_splits_from_manifest": recorded_selected,
            "accepted_but_not_selected_upper_bound": counts["accepted"] - recorded_selected,
            "expansion_constraint": "Unselected accepted pairs are only an upper bound across all local splits. Metadata does not establish how many additional train pairs or prompts are usable. Preserve existing prompt assignments and held-out groups; never repartition the raw prefix to enlarge training. Raw JSONL was not opened because it contains current held-out labels.",
        },
        "recommendations": [
            "Use prompt budget 576 for the next capacity probe only if the reported maximum remains <=576. This preserves all text supplied by the present train/valid dataset plus the task marker; resource feasibility must be measured separately.",
            "Response budget 96 is a candidate, not a claim of full coverage. Compare the reported 64/96/128 coverage and retain natural EOS only for complete summaries. A complete-only subset changes the target distribution and must be reported.",
            "Keep current train/valid/test assignments and pilot artifacts unchanged. This audit does not establish training quality, statistical significance, or availability of extra independent prompts.",
            "For stronger resume evidence, additional independent prompt groups and an untouched final evaluation are needed. The downloaded first 3000 official train rows are an ordered prefix, not a random sample or the official holdout.",
        ],
    }
    assert not report["method"]["torch_imported"]
    assert not report["method"]["transformers_imported"]
    # Exclusive creation protects previous audit evidence even if another process
    # creates this name while tokenization is running.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "splits": split_results,
                      "raw_source_metadata_only": report["raw_source_metadata_only"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
