"""Run official CPU scoring on immutable, complete saved model outputs.

This does not load a model, select hyperparameters, resume inference, or claim
that the whole research campaign is complete. IFEval token decoding is done in
a separate CPU-only tokenizers process, with every input and executable pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment_lab.evaluation_protocol_v2 import (
    EvaluationProtocol, canonical, parse, verified_bytes,
    load_evaluation_assets, load_final_holdout,
)
from alignment_lab.official_evaluation_v2 import (
    score_ifeval, aggregate_helpsteer_rm, aggregate_rm_bench,
)


def _read(path, sha):
    return parse(verified_bytes(Path(path), sha))


def _files(directory, records):
    if not isinstance(records, dict) or not records:
        raise ValueError("Complete inference file inventory required")
    result = {}
    for name, record in records.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("Unsafe inference artifact path")
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Inference artifact escapes its output directory")
        payload = verified_bytes(path, record["sha256"])
        if len(payload) != record["bytes"]:
            raise ValueError("Inference artifact byte size mismatch")
        result[name] = payload
    if not {"outputs.jsonl", "protocol.snapshot.json", "request.snapshot.json",
            "roster.snapshot.json"} <= result.keys():
        raise ValueError("Incomplete inference artifacts")
    return result


def _verify_roster(roster, protocol):
    expected = {"base": "base", "sft": "sft", "reward": "reward",
                **{f"{stage}{seed}": stage for stage in ("dpo", "ppo") for seed in (42, 43, 44)}}
    if (roster.get("schema_version") != 2
            or roster.get("kind") != "alignlab-accepted-evaluation-roster-v2"
            or set(roster.get("models", {})) != set(expected)
            or set(protocol.identity["model_manifests"]) != set(expected)):
        raise ValueError("The complete preregistered model roster is required")
    for model_id, stage in expected.items():
        record = roster["models"][model_id]
        if record.get("model_id") != model_id or record.get("stage") != stage:
            raise ValueError("Model roster role mismatch")
        accepted = _read(record["acceptance_path"], record["acceptance_sha256"])
        if (accepted.get("kind") != "alignlab-independent-evaluation-model-acceptance-v2"
                or accepted.get("status") != "accepted_for_evaluation"
                or accepted.get("all_checks_passed") is not True
                or accepted.get("processes_exited") is not True
                or accepted.get("snapshot_manifest_sha256") != roster["model_manifest_sha256"]):
            raise ValueError("Independent completed-model acceptance is required")
        row = accepted.get("models", {}).get(model_id, {})
        if any(row.get(key) != record.get(key)
               for key in ("stage", "artifact_dir", "manifest_sha256")):
            raise ValueError("Model acceptance differs from the roster")
        bound = protocol.identity["model_manifests"][model_id]
        if stage == "base":
            if (record["artifact_dir"] is not None or record["manifest_sha256"] is not None
                    or any(row.get(key) is not None for key in ("progress_path", "progress_sha256"))
                    or bound["sha256"] != roster["model_manifest_sha256"]
                    or Path(bound["path"]).resolve()
                    != (Path(roster["model_snapshot"]) / "snapshot-manifest.json").resolve()):
                raise ValueError("Base manifest differs from the official snapshot")
        else:
            if (bound["sha256"] != record["manifest_sha256"]
                    or Path(bound["path"]).resolve()
                    != (Path(record["artifact_dir"]) / "manifest.json").resolve()):
                raise ValueError("Protocol adapter differs from the accepted model")
            progress = _read(row["progress_path"], row["progress_sha256"])
            required_status = ("formal_parent_recovery_passed" if stage == "reward" else "formal_parent_stage_passed" if stage == "sft"
                               else "formal_alignment_stage_passed")
            if (progress.get("status") != required_status
                    or progress.get("formal_training") is not True
                    or progress.get("artifact", {}).get("manifest_sha256") != record["manifest_sha256"]):
                raise ValueError("Model is not a completed formal stage")


def _decode(decoder, sequences, *, deadline, runner=subprocess.run):
    expected = {"python", "python_sha256", "script", "script_sha256",
                "tokenizer_path", "tokenizer_sha256", "tokenizers_version"}
    if not isinstance(decoder, dict) or set(decoder) != expected:
        raise ValueError("An explicitly pinned CPU decoder is required")
    python = Path(decoder["python"]).resolve()
    if python != (ROOT / ".runtime/alignlab-v2/Scripts/python.exe").resolve():
        raise ValueError("Unexpected decoder interpreter")
    verified_bytes(python, decoder["python_sha256"])
    script = Path(decoder["script"]).resolve()
    if script != Path(__file__).with_name("decode_alignlab_evaluation_v2.py"):
        raise ValueError("Unexpected decoder script")
    verified_bytes(script, decoder["script_sha256"])
    verified_bytes(Path(decoder["tokenizer_path"]), decoder["tokenizer_sha256"])
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Absolute evaluation deadline expired before decoding")
    request = {"schema_version": 1,
               "tokenizer_path": decoder["tokenizer_path"],
               "tokenizer_sha256": decoder["tokenizer_sha256"],
               "tokenizers_version": decoder["tokenizers_version"],
               "sequences": sequences}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = runner([str(python), "-I", "-B", str(script)],
                    input=canonical(request), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=False, timeout=remaining,
                    env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("CPU decoder failed: " +
                           result.stderr.decode("utf-8", errors="replace")[-2000:])
    output = parse(result.stdout)
    if (output.get("kind") != "alignlab-cpu-decoded-actions-v1"
            or output.get("tokenizer_sha256") != decoder["tokenizer_sha256"]
            or output.get("tokenizers_version") != decoder["tokenizers_version"]
            or output.get("sequence_count") != len(sequences)
            or output.get("torch_imported") is not False
            or output.get("gpu_operations_performed") is not False
            or not isinstance(output.get("texts"), list)
            or len(output["texts"]) != len(sequences)
            or any(not isinstance(text, str) for text in output["texts"])):
        raise ValueError("Invalid CPU decoder receipt")
    verified_bytes(script, decoder["script_sha256"])
    verified_bytes(python, decoder["python_sha256"])
    mapping = {}
    for ids, text in zip(sequences, output["texts"]):
        key = tuple(ids)
        if key in mapping and mapping[key] != text:
            raise ValueError("Inconsistent decoding of identical token sequences")
        mapping[key] = text
    return mapping.__getitem__, output


def score_request(request_path, request_sha256, *, output):
    request = _read(request_path, request_sha256)
    if (set(request) != {"schema_version", "kind", "inference_manifest",
                         "inference_manifest_sha256", "source_sha256", "decoder",
                         "deadline_epoch"}
            or request["schema_version"] != 1
            or request["kind"] != "alignlab-official-cpu-scoring-request-v1"):
        raise ValueError("Unexpected scoring request")
    deadline = request["deadline_epoch"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Explicit absolute deadline required")
    if time.time() >= deadline:
        raise TimeoutError("Evaluation budget exhausted")
    sources = request["source_sha256"]
    required = {"scripts/score_alignlab_evaluation_v2.py",
                "scripts/decode_alignlab_evaluation_v2.py",
                "alignment_lab/evaluation_protocol_v2.py",
                "alignment_lab/official_evaluation_v2.py"}
    if not isinstance(sources, dict) or not required <= sources.keys():
        raise ValueError("Scoring source identity is incomplete")
    for name, sha in sources.items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or ".." in Path(name).parts:
            raise ValueError("Source escapes project")
        verified_bytes(path, sha)
    manifest_path = Path(request["inference_manifest"])
    manifest = _read(manifest_path, request["inference_manifest_sha256"])
    if (manifest.get("kind") != "alignlab-evaluation-inference-result-v2"
            or manifest.get("status") != "evaluation_inference_passed"
            or manifest.get("inference_complete") is not True
            or manifest.get("scoring_complete") is not False
            or manifest.get("quality_evaluation_complete") is not False
            or manifest.get("research_complete") is not False):
        raise ValueError("Only complete, verified inference can be scored")
    if deadline != manifest["deadline_epoch"]:
        raise ValueError("Scoring cannot reset or extend the evaluation deadline")
    runtime = manifest.get("model_runtime", {})
    audits = manifest.get("frozen_audits")
    if (runtime.get("device") != "cuda:0" or runtime.get("cuda") is not True
            or runtime.get("weights_unchanged") is not True
            or not isinstance(audits, list) or len(audits) < 2
            or any(item.get("unchanged") is not True for item in audits)):
        raise ValueError("Completed CUDA inference and unchanged-weight audits required")
    for audit in (audits[0], audits[-1]):
        if (audit.get("type") != "alignlab-evaluation-frozen-audit-v1"
                or audit.get("model_id") != manifest["model_id"]
                or any(audit.get(key) is not True for key in (
                    "full", "file_contents_rehashed", "untracked_content_writes_checked"))):
            raise ValueError("Full before/after content audits are required")
    for key in ("source_sha256", "base_sha256", "adapter_sha256", "head_sha256"):
        value = audits[0].get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                or audits[-1].get(key) != value):
            raise ValueError("Inference frozen content anchors differ")
    files = _files(manifest_path.parent, manifest["files"])
    if parse(files["request.snapshot.json"]) != manifest["request"]:
        raise ValueError("Inference request snapshot mismatch")
    protocol_info = manifest["protocol"]
    protocol_raw = verified_bytes(Path(protocol_info["path"]), protocol_info["sha256"])
    if parse(protocol_raw) != parse(files["protocol.snapshot.json"]):
        raise ValueError("Protocol snapshot mismatch")
    protocol = EvaluationProtocol(canonical(parse(protocol_raw)))
    if protocol.sha256 != protocol_info["identity_sha256"]:
        raise ValueError("Canonical protocol identity mismatch")
    protocol.verify_bindings()
    roster_info = manifest["accepted_roster"]
    roster = _read(roster_info["path"], roster_info["sha256"])
    if roster != parse(files["roster.snapshot.json"]):
        raise ValueError("Accepted model roster snapshot mismatch")
    _verify_roster(roster, protocol)
    model_id = manifest["model_id"]
    inference_request = manifest["request"]
    if (inference_request.get("model_id") != model_id
            or inference_request.get("task") != manifest["task"]
            or inference_request.get("deadline_epoch") != deadline
            or inference_request.get("protocol_path") != protocol_info["path"]
            or inference_request.get("protocol_sha256") != protocol_info["sha256"]
            or inference_request.get("roster_path") != roster_info["path"]
            or inference_request.get("roster_sha256") != roster_info["sha256"]):
        raise ValueError("Inference request differs from its output bindings")
    inference_sources = inference_request.get("source_sha256")
    if not isinstance(inference_sources, dict) or not inference_sources:
        raise ValueError("Inference sources must remain explicitly pinned")
    for name, sha in inference_sources.items():
        if not (ROOT / name).resolve().is_relative_to(ROOT) or ".." in Path(name).parts:
            raise ValueError("Inference source escapes project")
        verified_bytes(ROOT / name, sha)
    if protocol.identity["model_manifests"].get(model_id) != manifest["model_manifest"]:
        raise ValueError("Model differs from the frozen protocol")
    outputs = [parse(line) for line in files["outputs.jsonl"].splitlines() if line.strip()]
    assets_info = protocol.identity["asset_manifest"]
    assets = load_evaluation_assets(assets_info["path"], assets_info["sha256"])
    holdout_info = protocol.identity["holdout_manifest"]
    holdout = load_final_holdout(Path(holdout_info["path"]).parent, holdout_info["sha256"])
    task = manifest["task"]
    if ((task == "ifeval" and model_id == "reward")
            or (task in ("helpsteer_rm", "rm_bench") and model_id != "reward")):
        raise ValueError("Benchmark task does not match the registered model role")
    decode_receipt = None
    if task == "ifeval":
        if manifest["dataset_identity"] != assets.ifeval.identity:
            raise ValueError("IFEval dataset identity mismatch")
        decoder = request["decoder"]
        # The tokenizer must be present in the actual accepted model inventory.
        model_manifest = _read(manifest["model_manifest"]["path"],
                               manifest["model_manifest"]["sha256"])
        model_inputs = model_manifest.get("identity", {}).get("inputs", {})
        tokenizer_hashes = {v["sha256"] for k, v in model_inputs.items()
                            if k.startswith("tokenizer/") and k.endswith("tokenizer.json")}
        if not tokenizer_hashes:
            # Base snapshots use file inventories rather than stage identities.
            tokenizer_record = model_manifest.get("files", {}).get("tokenizer.json")
            if isinstance(tokenizer_record, dict):
                tokenizer_hashes.add(tokenizer_record.get("sha256"))
        if not decoder or decoder["tokenizer_sha256"] not in tokenizer_hashes:
            raise ValueError("CPU decoder is not bound to the evaluated model tokenizer")
        eos = protocol.identity["generation"]["eos_token_id"]
        sequences = []
        for row in outputs:
            ids = row.get("generated_token_ids")
            if not isinstance(ids, list):
                raise ValueError("Original generated actions required")
            sequences.append(ids[:-1] if ids and ids[-1] == eos else ids)
        decode_tokens, decode_receipt = _decode(decoder, sequences, deadline=deadline)
        scores = score_ifeval(assets, outputs, protocol=protocol,
                             model_id=model_id, decode_tokens=decode_tokens)
    elif task == "helpsteer_rm":
        if request["decoder"] is not None or manifest["dataset_identity"] != holdout.identity:
            raise ValueError("Unexpected RM scoring inputs")
        scores = aggregate_helpsteer_rm(holdout, outputs, protocol=protocol, model_id=model_id)
    elif task == "rm_bench":
        if request["decoder"] is not None or manifest["dataset_identity"] != assets.rm_bench.identity:
            raise ValueError("Unexpected RM scoring inputs")
        scores = aggregate_rm_bench(assets, outputs, protocol=protocol, model_id=model_id)
    else:
        raise ValueError("Unknown evaluation task")
    if time.time() >= deadline:
        raise TimeoutError("Evaluation budget exhausted during official scoring")
    # Re-read all externally bound bytes after scoring, before publishing success.
    verified_bytes(manifest_path, request["inference_manifest_sha256"])
    _files(manifest_path.parent, manifest["files"])
    protocol.verify_bindings()
    _read(roster_info["path"], roster_info["sha256"])
    _verify_roster(roster, protocol)
    for name, sha in sources.items():
        verified_bytes(ROOT / name, sha)
    for name, sha in inference_sources.items():
        verified_bytes(ROOT / name, sha)
    verified_bytes(Path(request_path), request_sha256)
    result = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-result-v1",
              "status": "official_scoring_passed", "task": task, "model_id": model_id,
              "request": {"path": str(Path(request_path).resolve()), "sha256": request_sha256},
              "inference_manifest": {"path": str(manifest_path.resolve()),
                                     "sha256": request["inference_manifest_sha256"]},
              "protocol_sha256": protocol.sha256, "scores": scores,
              "decoder_receipt": decode_receipt,
              "human_labels": 0, "human_win_rate": None, "research_complete": False,
              "gpu_operations_performed": False, "finished_at_epoch": time.time(),
              "deadline_epoch": deadline}
    destination = Path(output)
    destination.mkdir(parents=False, exist_ok=False)
    (destination / "request.snapshot.json").write_bytes(canonical(request))
    (destination / "scores.json").write_bytes(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = score_request(args.request, args.request_sha256, output=args.output)
    print(json.dumps({"status": result["status"], "task": result["task"],
                      "model_id": result["model_id"]}))


if __name__ == "__main__":
    main()
