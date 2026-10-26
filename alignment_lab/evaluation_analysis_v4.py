"""Read-only primary instruction-following and reward-model evaluation summary.

Real inference and model scoring are never performed here. Saved official
scores are replayed with the original official CPU checkers and pinned CPU
token decoder. This is a separate, preregistered analysis allocation within the
evaluation budget; it cannot reopen an expired inference/scoring allocation.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import os
import statistics
import time

from .evaluation_protocol_v2 import (
    EvaluationProtocol, canonical, digest, parse, verified_bytes, _relative,
    _finite, _integer, _index, load_evaluation_assets, load_final_holdout,
)
from .evaluation_execution_v4 import (
    MODEL_STAGES, REQUIRED_SOURCES as INFERENCE_SOURCES, verify_evaluation_result,
)
from .official_evaluation_v2 import (
    score_ifeval, aggregate_helpsteer_rm, aggregate_rm_bench, paired_cluster_bootstrap,
)
from scripts.score_alignlab_evaluation_v2 import _decode, _files

ROOT = Path(__file__).resolve().parents[1]
KIND = "alignlab-primary-evaluation-analysis-request-v2"
RESULT_KIND = "alignlab-primary-evaluation-analysis-result-v2"
POLICIES = tuple(name for name in MODEL_STAGES if name != "reward")
SLOTS = frozenset({f"ifeval/{name}" for name in POLICIES} |
                  {"helpsteer_rm/reward", "rm_bench/reward"})
METRICS = ("strict_prompt_accuracy", "strict_instruction_accuracy",
           "loose_prompt_accuracy", "loose_instruction_accuracy")
SCORING_SOURCES = frozenset({"scripts/score_alignlab_evaluation_v2.py",
    "scripts/decode_alignlab_evaluation_v2.py", "alignment_lab/evaluation_protocol_v2.py",
    "alignment_lab/official_evaluation_v2.py"})
REQUIRED_SOURCES = INFERENCE_SOURCES | SCORING_SOURCES | frozenset({
    "alignment_lab/evaluation_analysis_v4.py", "scripts/summarize_alignlab_evaluation_v4.py",
    "tests_v2/test_evaluation_analysis_v4.py", "alignment_lab/evaluation_acceptance_v2.py",
    "scripts/audit_alignlab_evaluation_models_v2.py", "tests_v2/test_evaluation_acceptance_v2.py",
    "alignment_lab/parent_acceptance.py", "alignment_lab/alignment_supervisor.py",
    "alignment_lab/study_supervisor.py", "alignment_lab/capacity_supervisor.py",
})
EXCLUDED = {
    "policy_likelihood_evaluation": "Not included: chosen NLL, likelihood ranking, and policy/SFT log-ratio ranking.",
    "four_regularization_ablations": "Not included: DPO beta 0.05/0.2 and PPO KL coefficient 0/0.1.",
    "reward_exploitation_diagnostics": "Not included: frozen diagnostic prompts, perturbations, KL and failure analysis.",
    "cost_dev_curves": "Not included: matched-cost development curves at 1/2/4/8 active hours.",
}


def _read(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("An explicit path/SHA256 file binding is required")
    if not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute():
        raise ValueError("Bound paths must be absolute")
    return parse(verified_bytes(Path(binding["path"]), binding["sha256"]))


def _sources(values, required):
    if not isinstance(values, dict) or not required <= values.keys():
        raise ValueError("Explicit static source dependency inventory is incomplete")
    for name, sha in values.items():
        verified_bytes(_relative(ROOT, name), sha)


def _allocation(value):
    if (not isinstance(value, dict) or set(value) != {"kind", "wall_seconds", "deadline_epoch",
                                                     "evaluation_deadline_epoch"}
            or value["kind"] != "within_independent_evaluation_budget"):
        raise ValueError("A separately preregistered read-only verification allocation is required")
    wall = _finite(value["wall_seconds"], "Verification wall seconds")
    deadline = _finite(value["deadline_epoch"], "Verification deadline")
    outer = _finite(value["evaluation_deadline_epoch"], "Overall evaluation deadline")
    if wall <= 0 or deadline <= 0 or deadline > outer:
        raise ValueError("Verification allocation must fit inside the independent evaluation budget")
    return value


def _request(request):
    if (not isinstance(request, dict) or set(request) != {"schema_version", "kind", "protocol",
            "roster", "scores", "source_sha256", "verification_allocation"}
            or type(request["schema_version"]) is not int or request["schema_version"] != 1
            or request["kind"] != KIND or not isinstance(request["scores"], dict)
            or set(request["scores"]) != SLOTS):
        raise ValueError("All eight complete policy IFEval scores and both reward tasks are required")
    _allocation(request["verification_allocation"])
    _sources(request["source_sha256"], REQUIRED_SOURCES)
    protocol = EvaluationProtocol(canonical(_read(request["protocol"])))
    protocol.verify_bindings()
    if (protocol.identity.get("primary_metric") != "ifeval_strict_prompt_accuracy"
            or protocol.identity.get("scorer_seed") != 0
            or set(protocol.identity["model_manifests"]) != set(MODEL_STAGES)):
        raise ValueError("The fixed full-model primary evaluation protocol is required")
    _integer(protocol.identity["bootstrap_seed"], "Bootstrap seed")
    _integer(protocol.identity["bootstrap_replicates"], "Bootstrap replicates", 2000)
    # This verifier is deliberately stdlib-only. The separately accepted model
    # audit checked tensor/checkpoint semantics and pinned all underlying bytes.
    from .evaluation_acceptance_v2 import verify_evaluation_roster
    roster = verify_evaluation_roster(request["roster"]["path"], request["roster"]["sha256"])
    if roster != _read(request["roster"]) or set(roster["models"]) != set(MODEL_STAGES):
        raise ValueError("Complete live accepted roster identity differs")
    asset = protocol.identity["asset_manifest"]
    assets = load_evaluation_assets(asset["path"], asset["sha256"])
    held = protocol.identity["holdout_manifest"]
    holdout = load_final_holdout(Path(held["path"]).parent, held["sha256"])
    for dataset in (assets.ifeval, assets.rm_bench, holdout):
        dataset.verify()
        if protocol.identity["dataset_identities"].get(dataset.name) != dataset.identity:
            raise ValueError("Live full evaluation datasets differ from the protocol")
    return protocol, roster, assets, holdout


def _budget(request, started, *, now=None):
    value = request["verification_allocation"]
    current = (now or time.time)()
    if current >= min(value["deadline_epoch"], started + value["wall_seconds"]):
        raise TimeoutError("Read-only analysis allocation exhausted; no completion is published")
    return min(value["deadline_epoch"], started + value["wall_seconds"])


def _scored(binding, slot, request, protocol):
    """Check the original scoring manifest and its original, unextended budget."""
    result = _read(binding)
    path = Path(binding["path"])
    if (path.name != "scores.json" or path.parent.is_symlink()
            or {p.name for p in path.parent.iterdir()} != {"scores.json", "request.snapshot.json"}):
        raise ValueError("Original complete scoring output directory required")
    task, model_id = slot.split("/")
    if (result.get("kind") != "alignlab-official-cpu-scoring-result-v1"
            or result.get("status") != "official_scoring_passed"
            or result.get("task") != task or result.get("model_id") != model_id
            or result.get("protocol_sha256") != protocol.sha256
            or type(result.get("human_labels")) is not int or result["human_labels"] != 0
            or result.get("human_win_rate") is not None
            or result.get("research_complete") is not False
            or result.get("gpu_operations_performed") is not False):
        raise ValueError("Official independent CPU scoring identity/claims differ")
    scoring_request = _read(result["request"])
    if parse((path.parent / "request.snapshot.json").read_bytes()) != scoring_request:
        raise ValueError("Scoring request snapshot differs from the pinned live request")
    if (set(scoring_request) != {"schema_version", "kind", "inference_manifest",
            "inference_manifest_sha256", "source_sha256", "decoder", "deadline_epoch"}
            or scoring_request["kind"] != "alignlab-official-cpu-scoring-request-v1"
            or scoring_request["schema_version"] != 1):
        raise ValueError("Exact original scoring request required")
    _sources(scoring_request["source_sha256"], SCORING_SOURCES)
    if result["inference_manifest"] != {"path": scoring_request["inference_manifest"],
                                      "sha256": scoring_request["inference_manifest_sha256"]}:
        raise ValueError("Score does not descend from its pinned inference manifest")
    inference = _read(result["inference_manifest"])
    if (inference.get("task") != task or inference.get("model_id") != model_id
            or inference.get("accepted_roster") != request["roster"]
            or inference.get("protocol") != {**request["protocol"], "identity_sha256": protocol.sha256}):
        raise ValueError("Score inference differs from the shared protocol/model roster")
    finished, deadline = (_finite(result[key], key) for key in ("finished_at_epoch", "deadline_epoch"))
    inference_finished = _finite(inference["finished_at_epoch"], "Inference finish")
    if (not 0 < inference_finished <= finished <= deadline
            or deadline != scoring_request["deadline_epoch"] or deadline != inference["deadline_epoch"]
            or deadline > request["verification_allocation"]["evaluation_deadline_epoch"]):
        raise ValueError("Original inference/scoring deadline expired, changed or exceeds the evaluation allocation")
    files = _files(Path(result["inference_manifest"]["path"]).parent, inference["files"])
    outputs = [parse(line) for line in files["outputs.jsonl"].split(b"\n") if line.strip()]
    return result, scoring_request, inference, outputs


def _normalize_ifeval(score, dataset):
    """Preserve all original observations; validate flags and recompute four ratios."""
    if (score.get("official_evaluator") is not True
            or score.get("denominators") != {"prompts": 541, "instructions": 834}
            or len(dataset.rows) != 541
            or sum(len(r["instruction_id_list"]) for r in dataset.rows) != 834):
        raise ValueError("Complete official IFEval 541/834 scores required")
    indexed = _index(score["per_prompt"], "key")
    if set(indexed) != set(dataset.ids):
        raise ValueError("Missing/extra IFEval score IDs")
    ordered = []
    metrics = {key: 0 for key in METRICS}
    for source in dataset.rows:
        record = indexed[source["key"]]
        for mode in ("strict", "loose"):
            item = record[mode]
            flags = item.get("follow_instruction_list")
            if (item.get("key") != source["key"]
                    or item.get("instruction_id_list") != source["instruction_id_list"]
                    or not isinstance(flags, list) or len(flags) != len(source["instruction_id_list"])
                    or any(type(flag) is not bool for flag in flags)
                    or type(item.get("follow_all_instructions")) is not bool
                    or item["follow_all_instructions"] != all(flags)):
                raise ValueError("Per-instruction official score schema or IDs differ")
            metrics[mode + "_prompt_accuracy"] += int(all(flags))
            metrics[mode + "_instruction_accuracy"] += sum(flags)
        ordered.append(record)
    for key in metrics:
        metrics[key] /= 541 if "_prompt_" in key else 834
    if score.get("metrics") != metrics:
        raise ValueError("Reported IFEval aggregates differ from complete per-instruction flags")
    return {**score, "per_prompt": ordered}


def _replay(slot, request, protocol, assets, holdout, deadline):
    result, score_request, inference, outputs = _scored(request["scores"][slot], slot, request, protocol)
    task, model_id = slot.split("/")
    decode = None
    if task == "ifeval":
        decoder = score_request["decoder"]
        model_manifest = _read(inference["model_manifest"])
        inputs = model_manifest.get("identity", {}).get("inputs", {})
        token_hashes = {v["sha256"] for k, v in inputs.items()
                       if k.startswith("tokenizer/") and k.endswith("tokenizer.json")}
        if not token_hashes:
            record = model_manifest.get("files", {}).get("tokenizer.json", {})
            token_hashes = {record.get("sha256")}
        if not isinstance(decoder, dict) or decoder.get("tokenizer_sha256") not in token_hashes:
            raise ValueError("Decoder is not bound to the actual evaluated model tokenizer")
        sequences = []
        for row in outputs:
            ids = row.get("generated_token_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError("Original generated action IDs required for replay")
            sequences.append(ids[:-1] if ids[-1] == 151645 else ids)
        decode, decoder_receipt = _decode(decoder, sequences, deadline=deadline)
        if decoder_receipt != result["decoder_receipt"]:
            raise ValueError("Independent CPU decoder replay differs from the saved scoring receipt")
    elif score_request["decoder"] is not None or result["decoder_receipt"] is not None:
        raise ValueError("Reward score aggregation cannot claim a generation decoder")
    verified = verify_evaluation_result(Path(result["inference_manifest"]["path"]).parent,
        expected_manifest_sha256=result["inference_manifest"]["sha256"],
        expected_request=inference["request"], decode_tokens=decode)
    outputs = verified["ordered_results"]
    if task == "ifeval":
        actual = _normalize_ifeval(score_ifeval(assets, outputs, protocol=protocol,
            model_id=model_id, decode_tokens=decode), assets.ifeval)
        saved = _normalize_ifeval(result["scores"], assets.ifeval)
    elif task == "helpsteer_rm":
        actual = aggregate_helpsteer_rm(holdout, outputs, protocol=protocol, model_id=model_id)
        saved = result["scores"]
        if actual["denominators"] != {"all": 2017, "non_ties": 1920, "ties": 97}:
            raise ValueError("Full HelpSteer validation denominator required")
    else:
        actual = aggregate_rm_bench(assets, outputs, protocol=protocol, model_id=model_id)
        saved = result["scores"]
        if actual["denominators"] != {"records": 1327, "answer_scores": 7962, "comparisons": 11943}:
            raise ValueError("Full six-response RM-Bench denominator required")
    if canonical(actual) != canonical(saved):
        raise ValueError("Official CPU replay differs from the supplied saved scores")
    termination = None
    if task == "ifeval":
        termination = {"eos": sum(r["termination"] == "eos" for r in outputs),
                       "length": sum(r["termination"] == "length" for r in outputs),
                       "max_new_tokens": 2048,
                       "generated_action_tokens": sum(len(r["generated_token_ids"]) for r in outputs)}
        termination["length_termination_rate"] = termination["length"] / 541
    return actual, termination


def _statistics(scores, protocol, dataset):
    """Three alignment seeds; shared SFT is a fixed paired baseline, not three draws."""
    normalized = {name: _normalize_ifeval(scores[f"ifeval/{name}"], dataset) for name in POLICIES}
    per_model = {name: normalized[name]["metrics"] for name in POLICIES}
    # The accepted IFEval input has 541 distinct prompts. Each original prompt
    # key is one cluster; all its constraint checks remain together. No semantic
    # clusters are inferred from outcomes and no model/seed observations pool.
    clusters = {row["key"]: "ifeval-prompt:" + str(row["key"]) for row in dataset.rows}
    baseline = {r["key"]: int(r["strict"]["follow_all_instructions"])
                for r in normalized["sft"]["per_prompt"]}
    algorithms = {}
    for algorithm in ("dpo", "ppo"):
        names = [f"{algorithm}{seed}" for seed in (42, 43, 44)]
        aggregate = {metric: {"mean": statistics.mean(per_model[name][metric] for name in names),
                             "sample_std": statistics.stdev(per_model[name][metric] for name in names),
                             "seed_count": 3, "ddof": 1} for metric in METRICS}
        paired = {}
        for seed, name in zip((42, 43, 44), names):
            policy = {r["key"]: int(r["strict"]["follow_all_instructions"])
                      for r in normalized[name]["per_prompt"]}
            paired[str(seed)] = paired_cluster_bootstrap(policy, baseline, clusters,
                seed=protocol.identity["bootstrap_seed"],
                replicates=protocol.identity["bootstrap_replicates"], confidence=.95)
        algorithms[algorithm] = {"seeds": [42, 43, 44], "metrics": aggregate,
            "strict_prompt_difference_vs_shared_sft_by_seed": paired}
    direct = {}
    for seed in (42, 43, 44):
        paired_values = [{r["key"]: int(r["strict"]["follow_all_instructions"])
                          for r in normalized[f"{algorithm}{seed}"]["per_prompt"]}
                         for algorithm in ("ppo", "dpo")]
        direct[str(seed)] = paired_cluster_bootstrap(*paired_values, clusters,
            seed=protocol.identity["bootstrap_seed"],
            replicates=protocol.identity["bootstrap_replicates"], confidence=.95)
    return {"per_model": per_model, "alignment_algorithms": algorithms,
            "strict_prompt_ppo_minus_dpo_by_seed": direct,
            "primary_metric": "ifeval_strict_prompt_accuracy",
            "bootstrap": {"unit": "original_ifeval_prompt_key", "clusters": 541,
                          "constraints_kept_with_prompt": True, "confidence": .95,
                          "seed": protocol.identity["bootstrap_seed"],
                          "replicates": protocol.identity["bootstrap_replicates"],
                          "shared_sft_baseline": True, "across_seed_pooling": False,
                          "conditional_on_fixed_models": True,
                          "full_sft_reward_pipeline_seed_variance_estimated": False}}


def _report(summary):
    def esc(value):
        return html.escape(str(value))
    def percent(value):
        return f"{100 * value:.2f}%"
    rows = []
    for model in POLICIES:
        metrics = summary["instruction_following"]["per_model"][model]
        rows.append("<tr><th>" + esc(model) + "</th>" +
                    "".join("<td>" + percent(metrics[key]) + "</td>" for key in METRICS) +
                    "<td>" + percent(summary["generation"][model]["length_termination_rate"]) + "</td></tr>")
    paired_rows = []
    for algorithm, values in summary["instruction_following"]["alignment_algorithms"].items():
        for seed, item in values["strict_prompt_difference_vs_shared_sft_by_seed"].items():
            low, high = item["confidence_interval"]
            paired_rows.append(f"<tr><td>{esc(algorithm)} {esc(seed)}</td><td>{100*item['estimate']:.2f}</td>"
                               f"<td>[{100*low:.2f}, {100*high:.2f}]</td></tr>")
    direct_rows = []
    for seed, item in summary["instruction_following"]["strict_prompt_ppo_minus_dpo_by_seed"].items():
        low, high = item["confidence_interval"]
        direct_rows.append(f"<tr><td>{esc(seed)}</td><td>{100*item['estimate']:.2f}</td>"
                           f"<td>[{100*low:.2f}, {100*high:.2f}]</td></tr>")
    aggregates = []
    for algorithm, values in summary["instruction_following"]["alignment_algorithms"].items():
        item = values["metrics"]["strict_prompt_accuracy"]
        aggregates.append(f"<li>{esc(algorithm)} strict prompt: {percent(item['mean'])} "
                          f"± {100*item['sample_std']:.2f} percentage points (sample SD, n=3).</li>")
    rm = summary["reward_model"]
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>AlignLab primary evaluation</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ccd3df;padding:9px;"
        "text-align:left}h1{font-size:28px}p,li{line-height:1.6}code{overflow-wrap:anywhere}</style>"
        "<h1>Primary instruction-following and reward-model evaluation</h1>"
        "<p>All ten registered primary scoring tasks passed complete independent CPU replay. "
        "<strong>This is not completion of the whole research design.</strong> No human labels or human win rate.</p>"
        "<h2>Complete IFEval</h2><p>Eight policies; 541 original prompts and 834 constraints each. "
        "Greedy generation, maximum 2,048 new actions; true EOS and length endings retained.</p>"
        "<table><tr><th>Model</th><th>Strict prompt</th><th>Strict instruction</th><th>Loose prompt</th>"
        "<th>Loose instruction</th><th>Length ending</th></tr>" + "".join(rows) + "</table><ul>" +
        "".join(aggregates) + "</ul><h2>Paired primary differences from the shared SFT</h2>"
        "<p>Percentage points; 95% percentile cluster bootstrap intervals. Every original prompt is one cluster, "
        "including all its constraints. Intervals condition on the fitted models; alignment seeds are never "
        "pooled as independent prompt observations. SFT/RM pipeline seed variance is not estimated.</p>"
        "<table><tr><th>Alignment seed</th><th>Difference</th><th>95% interval</th></tr>" +
        "".join(paired_rows) + "</table><h2>PPO minus DPO, paired within each seed</h2>"
        "<p>Direct differences on the same 541 prompts; no inference from overlap of separate intervals.</p>"
        "<table><tr><th>Seed</th><th>Difference (percentage points)</th><th>95% interval</th></tr>" +
        "".join(direct_rows) + "</table><h2>Reward-model diagnostics</h2><p>HelpSteer official validation: "
        f"2,017 observations (1,920 non-ties, 97 ties), pair accuracy {percent(rm['helpsteer_rm']['pair_accuracy'])}. "
        f"RM-Bench: 1,327 records, 7,962 responses, official four-domain macro "
        f"{percent(rm['rm_bench']['metrics']['total_avg_acc'])}. These metrics evaluate the reward model, "
        "not the quality of policy-generated responses; training reward is not a quality metric.</p>"
        "<pre>" + esc(json.dumps(rm["rm_bench"]["domains"], indent=2, sort_keys=True)) + "</pre>"
        "<h2>Excluded from this primary summary</h2><ul>" +
        "".join("<li>" + esc(text) + "</li>" for text in EXCLUDED.values()) + "</ul>"
        "<p>Full coverage and exact artifact identities are recorded in summary.json and run_manifest.json. "
        "No composite quality score or claim of comprehensive usefulness is produced.</p></html>")


def _new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _receipt(output, manifest):
    return {"manifest_path": str(output / "run_manifest.json"),
            "manifest_sha256": hashlib.sha256((output / "run_manifest.json").read_bytes()).hexdigest(),
            "status": manifest["status"], "primary_evaluation_complete": True, "research_complete": False}


def verify_analysis(output, expected_manifest_sha256):
    """Reuse requires an external completion-manifest pin, not self-asserted success."""
    output = Path(output)
    manifest = _read({"path": str((output / "run_manifest.json").resolve()), "sha256": expected_manifest_sha256})
    if (manifest.get("kind") != RESULT_KIND or manifest.get("status") != "primary_evaluation_analysis_passed"
            or manifest.get("primary_evaluation_complete") is not True
            or manifest.get("research_complete") is not False
            or set(manifest.get("files", {})) != {"summary.json", "report.html"}
            or {p.name for p in output.iterdir()} != {"summary.json", "report.html", "run_manifest.json"}):
        raise ValueError("Complete externally anchored primary analysis required")
    request = _read(manifest["request"])
    protocol, _, _, _ = _request(request)
    if manifest["identity_sha256"] != digest(request) or manifest["protocol_sha256"] != protocol.sha256:
        raise ValueError("Analysis identity differs from live inputs")
    for slot, binding in request["scores"].items():
        result, _, inference, outputs = _scored(binding, slot, request, protocol)
        decode = None
        if slot.startswith("ifeval/"):
            receipt = result["decoder_receipt"]
            if (receipt.get("sequence_count") != 541 or len(receipt.get("texts", [])) != 541
                    or receipt.get("torch_imported") is not False
                    or receipt.get("gpu_operations_performed") is not False):
                raise ValueError("Complete previously replayed decoder receipt required")
            mapping = {}
            for row, text in zip(outputs, receipt["texts"]):
                ids = row["generated_token_ids"]
                key = tuple(ids[:-1] if ids[-1] == 151645 else ids)
                if not isinstance(text, str) or key in mapping and mapping[key] != text:
                    raise ValueError("Saved decoder receipt is inconsistent")
                mapping[key] = text
            decode = mapping.__getitem__
        verify_evaluation_result(Path(result["inference_manifest"]["path"]).parent,
            expected_manifest_sha256=result["inference_manifest"]["sha256"],
            expected_request=inference["request"], decode_tokens=decode)
    payloads = {}
    for name, record in manifest["files"].items():
        payloads[name] = verified_bytes(output / name, record["sha256"])
        if len(payloads[name]) != record["bytes"]:
            raise ValueError("Analysis artifact size mismatch")
    summary = parse(payloads["summary.json"])
    if (summary.get("primary_evaluation_complete") is not True
            or summary.get("research_complete") is not False
            or summary.get("excluded") != EXCLUDED or summary.get("human_labels") != 0
            or summary.get("human_win_rate") is not None or summary.get("input_scores") != request["scores"]
            or payloads["report.html"] != _report(summary).encode("utf-8")):
        raise ValueError("Analysis report/summary identity or scope differs")
    verification = summary["verification"]
    allocation = request["verification_allocation"]
    started = _finite(verification["started_at_epoch"], "Analysis start")
    finished = _finite(verification["finished_at_epoch"], "Analysis finish")
    if (verification != manifest.get("verification") or verification.get("allocation") != allocation
            or verification.get("official_cpu_replay") is not True
            or verification.get("token_decoder_replay") is not True
            or verification.get("original_score_deadlines_preserved") is not True
            or not 0 < started <= finished <= allocation["deadline_epoch"]
            or finished - started > allocation["wall_seconds"]
            or verification.get("elapsed_seconds") != finished - started):
        raise ValueError("Completed analysis timing/replay receipt differs")
    return _receipt(output, manifest)


def summarize_evaluation(request_path, request_sha256, *, output, expected_manifest_sha256=None):
    """Replay ten complete saved scoring tasks, then publish one immutable summary."""
    request_binding = {"path": str(Path(request_path).resolve()), "sha256": request_sha256}
    request = _read(request_binding)
    raw_destination = Path(output).absolute()
    if any(p.is_symlink() for p in (raw_destination, *raw_destination.parents)):
        raise ValueError("Analysis output must not traverse symlinks")
    destination = raw_destination.resolve()
    if destination.exists():
        if expected_manifest_sha256 is None:
            raise FileExistsError("Existing analysis requires its external completion SHA; no overwrite/resume")
        receipt = verify_analysis(destination, expected_manifest_sha256)
        manifest = _read({"path": receipt["manifest_path"], "sha256": receipt["manifest_sha256"]})
        if manifest["identity_sha256"] != digest(request):
            raise ValueError("Existing output has a different analysis identity")
        return {**receipt, "reused": True}
    if expected_manifest_sha256 is not None:
        raise ValueError("Reuse pin supplied but completed analysis does not exist")
    started = time.time()
    _allocation(request.get("verification_allocation"))
    _budget(request, started)
    protocol, roster, assets, holdout = _request(request)
    scores, generation = {}, {}
    for slot in sorted(SLOTS):
        deadline = _budget(request, started)
        scores[slot], termination = _replay(slot, request, protocol, assets, holdout, deadline)
        if termination is not None:
            generation[slot.split("/")[1]] = termination
    _budget(request, started)
    summary = {"schema_version": 1, "kind": "alignlab-primary-evaluation-summary-v1",
        "scope": "primary_instruction_following_and_reward_model_evaluation",
        "primary_evaluation_complete": True, "research_complete": False,
        "human_labels": 0, "human_win_rate": None, "gpu_operations_performed": False,
        "protocol_sha256": protocol.sha256, "model_ids": list(MODEL_STAGES),
        "input_scores": request["scores"], "denominators": protocol.identity["denominators"],
        "instruction_following": _statistics(scores, protocol, assets.ifeval),
        "generation": generation,
        "reward_model": {"helpsteer_rm": scores["helpsteer_rm/reward"], "rm_bench": scores["rm_bench/reward"],
                         "policy_response_quality_claim": False, "training_rewards_used_as_quality": False},
        "excluded": EXCLUDED,
        "verification": {"official_cpu_replay": True, "token_decoder_replay": True,
            "allocation": request["verification_allocation"], "original_score_deadlines_preserved": True,
            "started_at_epoch": started}}
    # All externally pinned bytes remain bound after the last CPU replay.
    _request(request)
    for slot in SLOTS:
        _scored(request["scores"][slot], slot, request, protocol)
    verified_bytes(Path(request_binding["path"]), request_sha256)
    _budget(request, started)
    finished = time.time()
    summary["verification"].update(finished_at_epoch=finished, elapsed_seconds=finished - started)
    payloads = {"summary.json": canonical(summary), "report.html": _report(summary).encode("utf-8")}
    manifest = {"schema_version": 1, "kind": RESULT_KIND, "status": "primary_evaluation_analysis_passed",
        "request": request_binding, "identity_sha256": digest(request), "protocol_sha256": protocol.sha256,
        "primary_evaluation_complete": True, "research_complete": False, "human_labels": 0,
        "human_win_rate": None, "verification": summary["verification"],
        "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                  for name, payload in payloads.items()}}
    destination.mkdir(parents=False, exist_ok=False)
    for name, payload in payloads.items():
        _new(destination / name, payload)
    _budget(request, started)
    _new(destination / "run_manifest.json", canonical(manifest))
    return {**_receipt(destination, manifest), "reused": False}
