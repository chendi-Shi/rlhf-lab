"""Read-only completion audit; never import torch, load models, or repair artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlhf_lab.data import read_jsonl
from rlhf_lab.evaluate import load_generations, score_judgments
from rlhf_lab.local_evaluate import _canonical_hash, _select_validation_rows
from rlhf_lab.local_runtime import STAGES, load_local_config
from rlhf_lab.local_training import _configuration, _dataset_identity, _files_hashes, _source_hashes
from rlhf_lab.runtime import resolve_path, sha256_file, utc_now


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_artifacts(directory, manifest, required):
    require(not (directory / ".local-training.lock").exists(), f"Active/unreconciled lock: {directory}")
    require(not any(".tmp" in p.name for p in directory.rglob("*")), f"Temporary artifacts: {directory}")
    recorded = manifest.get("artifacts", {})
    require(required <= set(recorded), f"Missing required artifacts: {sorted(required - set(recorded))}")
    for name in recorded:
        target = (directory / name).resolve()
        require(target.is_relative_to(directory.resolve()) and target.is_file() and target.stat().st_size > 0,
                f"Missing, empty, or unsafe artifact: {name}")
    require(recorded == _files_hashes(directory, exclude=("run_manifest.json",)),
            f"Artifact inventory/hash mismatch: {directory}")


def verify(config_path, result):
    config = load_local_config(config_path)
    c, pause_after = _configuration(config, "sft")
    require(pause_after is None, "Completion audit requires the full configured budget")
    require(c["software_fixture"] is False, "Software fixtures are not real training evidence")
    root, model = Path(c["output_dir"]), Path(c["model_path"])
    evaluation = Path(config.get("eval_output_dir", root / "evaluation")).resolve()
    result.update(device=c["device"], output_dir=str(root), evaluation_dir=str(evaluation),
                  config=c, stages={}, errors=[])
    dataset_hash, _ = _dataset_identity(c)  # Hash all splits; decode train/valid only.
    sources, base_files = _source_hashes(), _files_hashes(model)
    dependencies = {name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "numpy", "safetensors")}
    base_manifest = read_json(model / "local-manifest.json")
    require(base_manifest.get("model_id") == c["model_id"] and base_manifest.get("revision") == c["model_revision"],
            "Base model manifest identity mismatch")
    require("model.safetensors" in base_manifest.get("files", {}), "Base manifest lacks model weights")
    for name, identity in base_manifest["files"].items():
        require(Path(name).name == name and base_files.get(name) == identity.get("sha256"),
                f"Base model file mismatch: {name}")
    stage_hashes = {stage: sha256_file(root / stage / "run_manifest.json") for stage in STAGES}
    for stage in STAGES:
        try:
            directory = root / stage
            manifest = read_json(directory / "run_manifest.json")
            parents = {} if stage == "sft" else {"sft": stage_hashes["sft"]}
            if stage == "ppo":
                parents["reward"] = stage_hashes["reward"]
            identity = {"config": c, "stage": stage, "dataset_manifest_sha256": dataset_hash,
                        "source_sha256": sources, "parent_sha256": parents,
                        "base_model_sha256": base_files, "dependencies": dependencies}
            require(manifest.get("identity") == identity, "Full normalized training identity mismatch")
            require(all(manifest.get(key) == value for key, value in identity.items()),
                    "Top-level training identity differs from embedded identity")
            require(manifest.get("status") == "completed" and manifest.get("engine") == "local_torch_autograd",
                    "Training stage is not completed by the expected engine")
            require(manifest.get("test_labels_decoded") is False and manifest.get("software_fixture") is False,
                    "Invalid training data scope")
            require(manifest.get("weights_changed") is True and manifest.get("weight_fingerprint_before")
                    and manifest.get("weight_fingerprint_after")
                    and manifest["weight_fingerprint_before"] != manifest["weight_fingerprint_after"],
                    "Missing parameter-update evidence")
            batches = c["ppo_updates"] if stage == "ppo" else 0
            steps = (batches * c["ppo_epochs"] * (c["ppo_rollout_batch_size"] // c["ppo_minibatch_size"])
                     if stage == "ppo" else c[stage + "_max_steps"])
            require(manifest.get("completed_optimizer_steps") == steps
                    and manifest.get("completed_rollout_batches") == batches
                    and manifest.get("pending_ppo_minibatch_cursor") is None, "Incomplete or different budget")
            peak = manifest.get("peak_allocated_bytes")
            require(c["device"] != "cuda" or (type(peak) is int and peak > 0), "Missing CUDA allocation evidence")
            required = {"model.safetensors", "config.json", "tokenizer_config.json", "tokenizer.json",
                        "metrics.jsonl", "latest.pt"}
            if stage == "ppo":
                required.add("value_head.safetensors")
                require(manifest.get("value_head_weights_changed") is True, "PPO value head did not change")
            verify_artifacts(directory, manifest, required)
            metrics = read_jsonl(directory / "metrics.jsonl")
            require(len(metrics) == steps, "Metric count differs from optimizer budget")
            for number, entry in enumerate(metrics, 1):
                require(entry.get("stage") == stage and entry.get("optimizer_step") == number,
                        "Metric stage/step sequence mismatch")
                require(all(isinstance(entry.get(key), (int, float)) and math.isfinite(entry[key])
                            for key in ("loss", "grad_norm")), "Nonfinite/missing optimization metrics")
            require(metrics[-1].get("rollout_updates") == batches, "Metric rollout budget mismatch")
            result["stages"][stage] = {"verified": True, "optimizer_steps": steps, "rollout_batches": batches,
                                      "peak_allocated_bytes": peak, "weights_changed": True,
                                      "manifest_sha256": stage_hashes[stage]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["stages"][stage] = {"verified": False, "error": str(exc)}
            result["errors"].append(f"{stage}: {exc}")

    try:
        manifest = read_json(evaluation / "run_manifest.json")
        eval_manifest_hash = sha256_file(evaluation / "run_manifest.json")
        current_eval_source = {"local_evaluate.py": sha256_file(ROOT / "rlhf_lab/local_evaluate.py")}
        source_matches = manifest.get("source_sha256") == current_eval_source
        result["evaluation"] = {"verified": False, "source_matches_current": source_matches,
                                "recorded_source_sha256": manifest.get("source_sha256"),
                                "current_source_sha256": current_eval_source}
        if not source_matches:
            result["errors"].append("evaluation: historical/source hash mismatch; current evaluation source is not verified")
        raw = read_jsonl(Path(c["data_dir"]) / "preferences_valid.jsonl")
        expected_rows = _select_validation_rows(raw, c["eval_limit"])
        selection = [{"id": row["id"], "prompt": row["prompt"], "reference": row["chosen"]} for row in expected_rows]
        prompt_hash, count = _canonical_hash(selection), len(selection)
        shared = {"config": c, "model_id": c["model_id"], "model_revision": c["model_revision"],
                  "eval_split": "valid", "prompts_sha256": prompt_hash, "dataset_manifest_sha256": dataset_hash,
                  "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": c["max_response_tokens"], "use_cache": True},
                  "inference_dtype": "float32" if c["device"] == "cpu" else "bfloat16"}
        require(manifest.get("status") == "completed" and manifest.get("test_labels_decoded") is False,
                "Evaluation incomplete or test labels decoded")
        require(all(manifest.get(key) == value for key, value in shared.items()), "Evaluation config/protocol mismatch")
        require(manifest.get("stage_manifest_sha256") == stage_hashes, "Evaluation used different stage manifests")
        require(manifest.get("base_model_manifest_sha256") == sha256_file(model / "local-manifest.json")
                and manifest.get("validation_file_sha256") == sha256_file(Path(c["data_dir"]) / "preferences_valid.jsonl"),
                "Evaluation base/data hash mismatch")
        labels = ("base", "sft", "dpo", "ppo")
        required = {"summary.json", "report.html"}
        required.update(f"{label}.jsonl{suffix}" for label in labels for suffix in ("", ".manifest.json"))
        required.update(f"{prefix}-{label}-vs-sft.{ext}" for label in ("dpo", "ppo")
                        for prefix, ext in (("blind", "csv"), ("private", "json")))
        verify_artifacts(evaluation, manifest, required)
        summary = read_json(evaluation / "summary.json")
        result["evaluation"].update(human_judgments=summary.get("human_judgments"),
                                    human_win_rate=summary.get("human_win_rate"))
        require(summary.get("status") == "completed" and summary.get("split") == "valid"
                and summary.get("device") == c["device"] and summary.get("test_labels_decoded") is False
                and summary.get("generation_examples") == count and set(summary.get("policies", {})) == set(labels),
                "Evaluation summary status/scope/policy/count mismatch")
        model_inputs = None
        for label in labels:
            path = evaluation / f"{label}.jsonl"
            sidecar, rows = read_json(str(path) + ".manifest.json"), load_generations(path)
            expected_model = str(model if label == "base" else root / label)
            require(all(sidecar.get(key) == value for key, value in shared.items())
                    and sidecar.get("status") == "completed" and sidecar.get("test_labels_decoded") is False
                    and sidecar.get("model_label") == label and sidecar.get("model_path") == expected_model
                    and sidecar.get("examples") == count and sidecar.get("output_sha256") == sha256_file(path),
                    f"{label}: generation sidecar identity/hash mismatch")
            actual_selection = [{key: row[key] for key in ("id", "prompt", "reference")} for row in rows]
            require(actual_selection == selection, f"{label}: different validation prompts/order/count")
            require(all(row["model_label"] == label and row["model_path"] == expected_model and row["seed"] == c["seed"]
                        for row in rows), f"{label}: generation model identity mismatch")
            inputs = [row["model_input_token_ids"] for row in rows]
            require(all(isinstance(ids, list) and ids for ids in inputs), f"{label}: missing model inputs")
            require(model_inputs is None or inputs == model_inputs, f"{label}: models received different token inputs")
            model_inputs = inputs
            require(summary["policies"][label].get("nll_examples") == count, f"{label}: NLL example count mismatch")
        judgments = {}
        for label in ("dpo", "ppo"):
            ballot = score_judgments(evaluation / f"blind-{label}-vs-sft.csv", evaluation / f"private-{label}-vs-sft.json")
            require(ballot["total_pairs"] == count and ballot["left_model"]["model_label"] == label
                    and ballot["right_model"]["model_label"] == "sft", f"{label}: blind ballot identity/count mismatch")
            judgments[label] = {key: ballot[key] for key in ("judged", "wins", "losses", "ties", "unjudged", "decisive_win_rate")}
        actual_judgments = sum(item["judged"] for item in judgments.values())
        require(summary.get("human_judgments") == actual_judgments, "Human judgment count differs from actual ballots")
        require(actual_judgments != 0 or summary.get("human_win_rate") is None, "Unjudged ballots cannot establish a win rate")
        result["evaluation"].update(verified=source_matches, generation_examples=count, prompts_sha256=prompt_hash,
                                    actual_ballots=judgments, reward_pairs=summary.get("reward_model", {}).get("pairs"))
        require(sha256_file(evaluation / "run_manifest.json") == eval_manifest_hash, "Evaluation changed during audit")
        require(current_eval_source["local_evaluate.py"] == sha256_file(ROOT / "rlhf_lab/local_evaluate.py"),
                "Evaluation source changed during audit")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.setdefault("evaluation", {})["verified"] = False
        result["errors"].append(f"evaluation: {exc}")

    require(stage_hashes == {stage: sha256_file(root / stage / "run_manifest.json") for stage in STAGES},
            "Stage manifests changed during audit")
    require(_source_hashes() == sources and _configuration(load_local_config(config_path), "sft")[0] == c,
            "Training source/config changed during audit")
    require("torch" not in sys.modules, "Verifier unexpectedly imported torch")
    result["status"] = "failed" if result["errors"] else "verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local CPU/CUDA profile to verify")
    parser.add_argument("--report", help="Write a new JSON report only if every check passes; never overwrite")
    args = parser.parse_args()
    result = {"schema_version": 1, "status": "failed", "checked_at_utc": utc_now(), "errors": [],
              "scope": "Artifact integrity/completion audit only; process exit must be checked separately; no quality claim"}
    try:
        verify(args.config, result)
        if result["status"] == "verified" and args.report:
            destination = resolve_path(args.report)
            protected = [Path(result["config"][key]) for key in ("model_path", "data_dir", "output_dir")]
            protected.append(Path(result["evaluation_dir"]))
            require(not any(destination == path or path in destination.parents for path in protected),
                    "Audit report must be outside model/data/run directories")
            payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                try:
                    stream.write(payload)
                    stream.flush()
                except BaseException:
                    stream.close()
                    destination.unlink(missing_ok=True)
                    raise
    except (OSError, ValueError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as exc:
        result["status"] = "failed"
        result["errors"].append(str(exc))
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
