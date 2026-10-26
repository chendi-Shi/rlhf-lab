"""Deterministic development prompts and the already reserved diagnostic rows.

This selects no model outputs and runs no model. The additional 100
programmatically checkable diagnostic templates are deliberately not supplied
here: these files are partial research inputs, not a completed diagnostic set.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .data_audit import normalized
from .evaluation_protocol import (
    ASSET_MANIFEST_SHA256, HOLDOUT_MANIFEST_SHA256,
    canonical, digest, parse, verified_bytes, load_evaluation_assets,
)

ROOT = Path(__file__).resolve().parents[1]
RULE = "seed42-domain-proportional-largest-remainder-one-cluster-v1"
SOURCES = (
    "alignment_lab/research_inputs.py", "scripts/prepare_alignlab_research_inputs.py",
    "tests_v2/test_research_inputs.py", "alignment_lab/data_audit.py",
    "alignment_lab/evaluation_protocol.py", "alignment_lab/chat_codec.py",
)


def _rank(purpose, key):
    return digest([42, purpose, key])


def select_development(rows, count=128):
    """Choose distinct clusters, stratified by representative-row domain."""
    if type(count) is not int or count < 1:
        raise ValueError("A positive prompt count is required")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate input record identities")
    groups = defaultdict(list)
    for row in rows:
        if row["source_split"] != "train" or row["local_split"] != "tuning_dev":
            raise ValueError("Only the reserved tuning-dev partition is eligible")
        groups[row["cluster_id"]].append(row)
    if len(groups) < count:
        raise ValueError("Insufficient independent development clusters")
    representatives = [
        min(group, key=lambda r: (_rank("dev-representative", r["record_id"]), r["record_id"]))
        for group in groups.values()
    ]
    domains = defaultdict(list)
    for row in representatives:
        domains[row["domain"]].append(row)
    total = len(representatives)
    allocations = {d: count * len(values) // total for d, values in domains.items()}
    remainder = count - sum(allocations.values())
    order = sorted(domains, key=lambda d: (-(count * len(domains[d]) % total), d))
    for domain in order[:remainder]:
        allocations[domain] += 1
    selected = []
    for domain in sorted(domains):
        selected.extend(sorted(domains[domain],
            key=lambda r: (_rank("dev-cluster", r["cluster_id"]), r["cluster_id"]))[:allocations[domain]])
    selected.sort(key=lambda r: (_rank("dev-output-order", r["cluster_id"]), r["cluster_id"]))
    return selected, {
        "rule": RULE, "seed": 42, "representative_rule": "seeded record hash per cluster",
        "eligible_rows": len(rows), "eligible_clusters": total,
        "representative_domain_counts": dict(sorted(Counter(r["domain"] for r in representatives).items())),
        "selected_domain_counts": allocations, "selected_count": count,
        "length_or_label_filter": False, "model_outputs_used": False,
    }


def _rows(directory, manifest, name):
    raw = verified_bytes(directory / name, manifest["artifacts"][name])
    # Split physical LF only. U+2028 inside JSON strings is not a record boundary.
    return [parse(line) for line in raw.split(b"\n") if line.strip()]


def _context_id(context):
    return digest([{"role": m["role"], "content": normalized(m["content"])} for m in context])


def prepare_research_inputs(audit_dir, asset_manifest_path, *, output):
    directory, output = Path(audit_dir).resolve(), Path(output).absolute()
    if (not output.resolve().is_relative_to(ROOT / "data")
            or output.resolve() == ROOT / "data"
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Fresh research-input output must be under project data")
    if output.exists():
        raise FileExistsError("Research input outputs are exclusive; no overwrite")
    manifest = parse(verified_bytes(directory / "manifest.json", HOLDOUT_MANIFEST_SHA256))
    if manifest.get("status") != "audit_complete" or manifest.get("data_ready") is not True:
        raise ValueError("An accepted complete data audit is required")
    partitions = {name: _rows(directory, manifest, name + ".jsonl") for name in
                  ("eligible_train_pool", "tuning_dev", "diagnostic_holdout", "final_holdout")}
    for field in ("cluster_id", "prompt_id", "user_path_id"):
        sets = {name: {r[field] for r in rows} for name, rows in partitions.items()}
        for left in sets:
            for right in sets:
                if left < right and sets[left] & sets[right]:
                    raise ValueError(f"Partition overlap: {field} {left} {right}")
    dev, selection = select_development(partitions["tuning_dev"])
    diagnostic = partitions["diagnostic_holdout"]
    if (len(diagnostic) != 100 or len({r["cluster_id"] for r in diagnostic}) != 100
            or any(r["local_split"] != "diagnostic_holdout" or r["source_split"] != "train"
                   for r in diagnostic)):
        raise ValueError("Keep all 100 previously reserved diagnostic clusters")
    assets = load_evaluation_assets(asset_manifest_path, ASSET_MANIFEST_SHA256)
    external = {_context_id([{"role": "user", "content": r["prompt"]}])
                for dataset in (assets.ifeval, assets.rm_bench) for r in dataset.rows}
    if any(r["prompt_id"] in external for r in dev + diagnostic):
        raise ValueError("Selected context exactly overlaps an external evaluation prompt")
    for row in dev + diagnostic:
        if _context_id(row["context"]) != row["prompt_id"]:
            raise ValueError("Context identity mismatch")
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    # Keep only prompt/source identities in the development file. Selecting a
    # source-preferred answer is not a quality judgment or a diagnostic oracle.
    fields = ("record_id", "cluster_id", "prompt_id", "user_path_id", "context",
              "domain", "language", "source_split", "source_index", "local_split")
    payloads = {
        "development-128.jsonl": b"".join(canonical({k: r[k] for k in fields}) + b"\n" for r in dev),
        "diagnostic-holdout-100.jsonl": b"".join(canonical({k: r[k] for k in fields}) + b"\n"
            for r in sorted(diagnostic, key=lambda r: r["cluster_id"])),
    }
    inputs = {
        "audit_manifest": {"path": str(directory / "manifest.json"), "sha256": HOLDOUT_MANIFEST_SHA256},
        "external_asset_manifest": {"path": str(Path(asset_manifest_path).resolve()),
                                    "sha256": ASSET_MANIFEST_SHA256},
    }
    for name in partitions:
        inputs[name] = {"path": str(directory / (name + ".jsonl")),
                        "sha256": manifest["artifacts"][name + ".jsonl"]}
    report = {
        "kind": "alignlab-partial-research-input-preregistration-v1", "schema_version": 1,
        "status": "development_and_reserved_diagnostic_prompts_prepared",
        "inputs": inputs, "source_sha256": sources, "selection": selection,
        "artifacts": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                      for name, raw in payloads.items()},
        "development_prompts": 128, "reserved_diagnostic_prompts": 100,
        "additional_rule_templates": 0, "diagnostic_set_complete": False,
        "registration_note": "128 and its stratification complete the previously unspecified dev-subset rule before alignment branches; they are not numbers claimed by the original design.",
        "external_overlap_check": "Exact NFKC/casefold/whitespace-normalized full context only; not semantic deduplication",
        "model_outputs_used": False, "model_quality_evaluated": False,
        "gpu_operations_performed": False, "research_complete": False,
        "usage": "Fixed tuning-dev cost curves at actually reached 1/2/4/8-hour checkpoints; no selection on final holdout",
        "cost_comparison_policy": {
            "checkpoint": "First committed optimizer boundary at or after each target; retain actual elapsed, overshoot and save-completion time, not just the cost label",
            "missing_timepoint": "null; never repeat data to reach a target",
            "final_models": "Registered full-workload endpoints; never choose seed/checkpoint using final IFEval or official heldout scores",
            "dpo_cache": "Disclose one-time cache/setup cost and distinguish cold and reused-cache cost; do not claim warm ablation vs cold main is a single-variable cost match",
        },
        "long_context_policy": "Retain selected prompts verbatim without length filtering, re-drawing or truncation; their inference capacity and budget remain unverified",
        "pending": ["100 independently registered rule-template diagnostics and cross-set deduplication",
                    "Diagnostic perturbations/checkers and evaluation workload/budget",
                    "Development-curve inference and independent main evaluation"],
    }
    for binding in inputs.values():
        verified_bytes(Path(binding["path"]), binding["sha256"])
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in sources.items()):
        raise ValueError("Preparation sources changed")
    output.mkdir(parents=False)
    for name, raw in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
        verified_bytes(output / name, report["artifacts"][name]["sha256"])
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical(report))
    return report
