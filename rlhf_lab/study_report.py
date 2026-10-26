"""Offline study aggregation. Reads completed artifacts; never loads a model."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics

from .data import normalize_prompt
from .evaluate import load_generations, response_proxies, score_judgments, wilson_interval
from .local_runtime import STAGES, load_local_config
from .local_training import _configuration
from .runtime import ROOT, resolve_path, sha256_file

POLICIES = ("base", "sft", "dpo", "ppo")
METRICS = ("chosen_token_nll", "repeat_trigram_proxy", "mean_generated_tokens", "mean_reward_model_score")
BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES = 20261020, 2000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(left, right):
    return _finite(left) and _finite(right) and math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)


def paired_bootstrap(left, right, weights=None, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Percentile CI for candidate-minus-reference, resampling paired prompts.

    NLL uses per-prompt token means with token-count weights. Other metrics use
    equal prompt weights. The interval is conditional on one trained seed.
    """
    _require(len(left) == len(right) and len(left) > 0, "Bootstrap needs nonempty matched prompts")
    weights = [1.0] * len(left) if weights is None else list(weights)
    _require(len(weights) == len(left) and all(_finite(v) for v in list(left) + list(right))
             and all(_finite(w) and w > 0 for w in weights), "Invalid bootstrap values/weights")
    _require(type(samples) is int and samples >= 2, "At least two bootstrap samples required")
    delta = [a - b for a, b in zip(left, right)]

    def estimate(indices):
        return sum(delta[i] * weights[i] for i in indices) / sum(weights[i] for i in indices)

    rng, n = random.Random(seed), len(delta)
    draws = sorted(estimate([rng.randrange(n) for _ in range(n)]) for _ in range(samples))

    def quantile(p):
        position = (samples - 1) * p
        lower = int(position)
        return draws[lower] + (draws[min(lower + 1, samples - 1)] - draws[lower]) * (position - lower)

    return {"estimate": estimate(range(n)), "ci95": [quantile(.025), quantile(.975)],
            "prompts": n, "samples": samples, "rng_seed": seed, "direction": "candidate minus SFT",
            "scope": "paired prompt bootstrap conditional on this trained seed; not seed uncertainty"}


def _artifact_hashes(directory, recorded):
    _require(isinstance(recorded, dict) and recorded, f"Missing artifact inventory: {directory}")
    actual_names = {p.name for p in directory.iterdir() if p.is_file() and p.name != "run_manifest.json"}
    _require(actual_names == set(recorded), f"Artifact inventory mismatch: {directory}")
    verified = {}
    for name, digest in recorded.items():
        path = directory / name
        _require(Path(name).name == name and path.resolve().parent == directory.resolve()
                 and path.is_file() and path.stat().st_size > 0, f"Unsafe/missing artifact: {name}")
        _require(sha256_file(path) == digest, f"Artifact hash mismatch: {path}")
        verified[str(path.resolve())] = digest
    return verified


def _load_run(entry, config_path, c):
    root = Path(c["output_dir"])
    evaluation = Path(c.get("eval_output_dir", root / "evaluation")).resolve()
    files = {str(config_path): sha256_file(config_path)}
    stages = {stage: _read(root / stage / "run_manifest.json") for stage in STAGES}
    stage_hashes = {stage: sha256_file(root / stage / "run_manifest.json") for stage in STAGES}
    for stage, manifest in stages.items():
        _require(manifest.get("status") == "completed" and manifest.get("stage") == stage
                 and manifest.get("config") == c and manifest.get("weights_changed") is True,
                 f"{entry['name']}: stage config/completion mismatch: {stage}")
        expected_parents = {} if stage == "sft" else {"sft": stage_hashes["sft"]}
        if stage == "ppo":
            expected_parents["reward"] = stage_hashes["reward"]
        _require(manifest.get("parent_sha256") == expected_parents, "Stage parent manifest hash mismatch")
        _require(manifest.get("test_labels_decoded") is False, "Test labels must remain sealed")
        files[str((root / stage / "run_manifest.json").resolve())] = stage_hashes[stage]
    manifest_path = evaluation / "run_manifest.json"
    manifest = _read(manifest_path)
    files[str(manifest_path)] = sha256_file(manifest_path)
    _require(manifest.get("status") == "completed" and manifest.get("config") == c
             and manifest.get("eval_split") == "valid" and manifest.get("test_labels_decoded") is False,
             "Evaluation config/completion/split mismatch")
    _require(manifest.get("stage_manifest_sha256") == stage_hashes, "Evaluation stage manifest hashes mismatch")
    _require(manifest.get("source_sha256") == {"local_evaluate.py": sha256_file(ROOT / "rlhf_lab/local_evaluate.py")},
             "Evaluation source hash mismatch")
    _require(all(m.get("dataset_manifest_sha256") == manifest.get("dataset_manifest_sha256")
                 for m in stages.values()), "Evaluation/training data identity mismatch")
    files.update(_artifact_hashes(evaluation, manifest.get("artifacts")))
    summary = _read(evaluation / "summary.json")
    count = summary.get("generation_examples")
    _require(summary.get("status") == "completed" and summary.get("split") == "valid"
             and summary.get("device") == c["device"] and summary.get("test_labels_decoded") is False
             and type(count) is int and count > 0 and set(summary.get("policies", {})) == set(POLICIES),
             "Evaluation summary scope/policy/count mismatch")
    common, vectors, policy_metrics = None, {}, {}
    for label in POLICIES:
        path = evaluation / f"{label}.jsonl"
        rows, sidecar = load_generations(path), _read(str(path) + ".manifest.json")
        _require(len(rows) == count, "Generation prompt count mismatch")
        for key in ("config", "model_id", "model_revision", "eval_split", "prompts_sha256",
                    "dataset_manifest_sha256", "decoding", "inference_dtype"):
            _require(sidecar.get(key) == manifest.get(key), f"{label}: sidecar {key} mismatch")
        model_path = str(Path(c["model_path"]) if label == "base" else root / label)
        _require(sidecar.get("status") == "completed" and sidecar.get("examples") == count
                 and sidecar.get("output_sha256") == files[str(path)] and sidecar.get("model_label") == label
                 and sidecar.get("model_path") == model_path and sidecar.get("test_labels_decoded") is False,
                 f"{label}: generation sidecar identity/hash mismatch")
        _require(all(r["model_label"] == label and r["model_path"] == model_path and r["seed"] == c["seed"] for r in rows),
                 "Generation model/seed mismatch")
        selection = [{key: r[key] for key in ("id", "prompt", "reference")} for r in rows]
        _require(_canonical(selection) == manifest["prompts_sha256"], "Generation prompt hash mismatch")
        _require(len({normalize_prompt(r["prompt"]) for r in rows}) == count, "Duplicate normalized prompts")
        stats = summary["policies"][label]
        nll = stats["nll_rows"]
        _require([r["id"] for r in nll] == [r["id"] for r in rows], "NLL prompt IDs mismatch")
        _require(all(type(r["tokens"]) is int and r["tokens"] > 0 and _finite(r["negative_logprob_sum"]) for r in nll),
                 "Invalid NLL rows")
        inputs = [r["model_input_token_ids"] for r in rows]
        _require(all(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids) for ids in inputs),
                 "Missing/invalid model input tokens")
        signature = {"selection": selection, "input_tokens": inputs, "chosen_token_counts": [r["tokens"] for r in nll]}
        _require(common is None or common == signature, "Policies used different prompts/input tokens")
        common = signature
        values = {"chosen_token_nll": [r["negative_logprob_sum"] / r["tokens"] for r in nll],
                  "repeat_trigram_proxy": [response_proxies(r["response"])["repeated_trigram_fraction_proxy"] for r in rows],
                  "mean_generated_tokens": [r["generated_tokens"] for r in rows],
                  "mean_reward_model_score": [r["reward_model_score"] for r in rows]}
        _require(all(_finite(value) for values_ in values.values() for value in values_), "Nonfinite policy metrics")
        totals = {key: statistics.mean(value) for key, value in values.items()}
        totals["chosen_token_nll"] = sum(r["negative_logprob_sum"] for r in nll) / sum(r["tokens"] for r in nll)
        recorded = {**stats, "repeat_trigram_proxy": stats["text_diagnostics"]["mean_repeated_trigram_fraction_proxy"]}
        _require(all(_close(totals[key], recorded[key]) for key in METRICS), "Policy summary differs from per-prompt metrics")
        _require(stats.get("nll_examples") == count and stats.get("nll_tokens") == sum(r["tokens"] for r in nll),
                 "NLL count mismatch")
        policy_metrics[label], vectors[label] = totals, values
    comparisons, ballots = {}, {}
    for label in ("dpo", "ppo"):
        comparisons[label + "_vs_sft"] = {metric: paired_bootstrap(vectors[label][metric], vectors["sft"][metric],
            common["chosen_token_counts"] if metric == "chosen_token_nll" else None) for metric in METRICS}
        ballot = score_judgments(evaluation / f"blind-{label}-vs-sft.csv", evaluation / f"private-{label}-vs-sft.json")
        mapping = _read(evaluation / f"private-{label}-vs-sft.json")
        _require(ballot["total_pairs"] == count and ballot["left_model"]["model_label"] == label
                 and ballot["right_model"]["model_label"] == "sft"
                 and sorted(p["id"] for p in mapping["pairs"]) == sorted(r["id"] for r in common["selection"]),
                 "Human ballot comparison/IDs mismatch")
        ballots[label] = {key: ballot[key] for key in ("judged", "wins", "losses", "ties", "unjudged", "decisive_win_rate")}
    judged = sum(b["judged"] for b in ballots.values())
    _require(summary.get("human_judgments") == judged and (judged != 0 or summary.get("human_win_rate") is None),
             "Human evaluation count/rate mismatch")
    rm = summary["reward_model"]
    _require(rm.get("pairs") == len(rm["rows"]) > 0 and len({r["id"] for r in rm["rows"]}) == rm["pairs"],
             "RM pair count/duplicate ID mismatch")
    _require(all(_finite(r["chosen_reward"]) and _finite(r["rejected_reward"])
                 and r["correct"] == (r["chosen_reward"] > r["rejected_reward"])
                 and r["tie"] == (r["chosen_reward"] == r["rejected_reward"]) for r in rm["rows"]), "Invalid RM pair scores")
    correct, ties = sum(r["correct"] for r in rm["rows"]), sum(r["tie"] for r in rm["rows"])
    _require(rm["correct"] == correct and rm["ties"] == ties and _close(rm["accuracy"], correct / rm["pairs"]),
             "RM summary count/accuracy mismatch")
    result = {key: entry[key] for key in ("name", "bundle", "seed", "dpo_beta", "kl_coef")}
    result.update(policies=policy_metrics, paired_bootstrap=comparisons, human_judgments=judged,
                  human_win_rate=summary.get("human_win_rate"), human_ballots=ballots,
                  reward_model={"pairs": rm["pairs"], "correct": correct, "ties": ties, "accuracy": rm["accuracy"],
                                "wilson95": wilson_interval(correct, rm["pairs"])},
                  parent_fingerprints={stage: stages[stage]["weight_fingerprint_after"] for stage in ("sft", "reward")})
    signature = {**common, "rm_ids": sorted(r["id"] for r in rm["rows"]),
                 "dataset_manifest_sha256": manifest["dataset_manifest_sha256"], "decoding": manifest["decoding"],
                 "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
                 "inference_dtype": manifest["inference_dtype"]}
    return result, signature, files


def validate_comparability(runs, signatures):
    _require(all(s == signatures[0] for s in signatures), "Runs used different prompts/input tokens/protocol")
    parents = {}
    for run in runs:
        value = run["parent_fingerprints"]
        _require(all(isinstance(v, str) and len(v) == 64 for v in value.values()), "Missing SFT/RM fingerprints")
        _require(run["seed"] not in parents or parents[run["seed"]] == value,
                 f"Unequal SFT/RM parent fingerprints across bundles for seed {run['seed']}; ablation is not matched")
        parents[run["seed"]] = value


def _aggregate(runs):
    groups = {}
    for bundle in sorted({r["bundle"] for r in runs}):
        selected = sorted((r for r in runs if r["bundle"] == bundle), key=lambda r: r["seed"])
        policies = {}
        for label in POLICIES:
            policies[label] = {}
            for metric in METRICS[:-1]:  # Independent RMs have uncalibrated score scales across seeds.
                values = [r["policies"][label][metric] for r in selected]
                policies[label][metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                                          "per_seed": dict(zip((str(r["seed"]) for r in selected), values))}
        accuracies = [r["reward_model"]["accuracy"] for r in selected]
        groups[bundle] = {"seeds": [r["seed"] for r in selected], "policies": policies,
                          "reward_accuracy_mean": statistics.mean(accuracies),
                          "reward_accuracy_sample_sd": statistics.stdev(accuracies)}
    return groups


def _render(summary):
    esc = lambda value: html.escape(str(value), quote=True)
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{esc(v)}</th>" for v in headers) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"
    rows, uncertainty, rm_rows = [], [], []
    for run in summary["runs"]:
        for label, values in run["policies"].items():
            rows.append([run["name"], label, run["dpo_beta"], run["kl_coef"]] + [f"{values[k]:.5f}" for k in METRICS])
        rm = run["reward_model"]
        rm_rows.append([run["name"], f"{rm['correct']}/{rm['pairs']}", rm["ties"],
                        f"[{rm['wilson95'][0]:.4f}, {rm['wilson95'][1]:.4f}]", run["human_judgments"]])
        for comparison, values in run["paired_bootstrap"].items():
            for metric, interval in values.items():
                uncertainty.append([run["name"], comparison, metric, f"{interval['estimate']:.5f}",
                                    f"[{interval['ci95'][0]:.5f}, {interval['ci95'][1]:.5f}]"])
    grouped = [[bundle, label] + [f"{metrics[k]['mean']:.5f} ± {metrics[k]['sample_sd']:.5f}" for k in METRICS[:-1]]
               for bundle, group in summary["by_bundle"].items() for label, metrics in group["policies"].items()]
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>RLHF 三种子对照研究</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:28px auto;padding:0 20px;color:#15243b}"
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:28px}td,th{padding:8px;border-bottom:1px solid #d9e1eb;text-align:left}"
            "th{background:#edf2f8}.note{padding:18px;background:#fff3d9;border-left:5px solid #bc8617}h2{margin-top:36px}</style>"
            f"<h1>{esc(summary['name'])}</h1><p>{summary['completed_runs']} 个已完成运行；{summary['validation_prompts']} 个共同验证原文。</p>"
            "<div class='note'>" + "<br>".join(esc(v) for v in summary["limitations"]) + "</div>"
            "<p>人工评价：" + str(summary["human_judgments"]) + "；没有人工判断的比较胜率保持为空。"
            "所有差值均为候选策略减 SFT；NLL 更低并不等于摘要更好。<a href='summary.json'>完整机器可读结果</a></p>"
            "<h2>每个运行的结果</h2>" + table(["运行", "策略", "β", "KL 系数", "Chosen NLL", "重复三元组代理", "生成 tokens", "训练 RM 分数"], rows)
            + "<h2>三个 seed 的均值 ± 样本标准差</h2>" + table(["配置包", "策略", "Chosen NLL", "重复三元组代理", "生成 tokens"], grouped)
            + "<h2>奖励模型诊断</h2><p>Wilson 区间是 RM 与偏好标签符合率的区间；RM 分数不是独立质量评估。不同 seed 的 RM 分数尺度未校准。</p>"
            + table(["运行", "正确/偏好对", "平分", "准确率 Wilson 95%", "人工评价数"], rm_rows)
            + "<h2>逐 seed 的配对 prompt bootstrap</h2><p>固定 RNG 20261020，2000 次重采样，百分位 95% 区间；"
            "NLL 每次按采样 prompt 的真实 token 数加权。区间条件于当前训练 seed，不替代跨 seed 标准差，亦未作多重比较校正。</p>"
            + table(["运行", "比较", "指标", "差值", "95% 区间"], uncertainty) + "</html>")


def summarize_campaign(plan_path, output):
    plan_path, output = resolve_path(plan_path), resolve_path(output)
    plan = _read(plan_path)
    entries = plan["runs"]
    _require(len(entries) == 9 and len({r["name"] for r in entries}) == 9, "Study requires nine unique runs")
    seeds, bundles = {r["seed"] for r in entries}, {r["bundle"] for r in entries}
    _require(len(seeds) == len(bundles) == 3 and len({(r["bundle"], r["seed"]) for r in entries}) == 9,
             "Study requires three bundles crossed with three seeds")
    for bundle in bundles:
        _require(len({(r["dpo_beta"], r["kl_coef"]) for r in entries if r["bundle"] == bundle}) == 1,
                 "Bundle hyperparameters must be fixed across seeds")
    configs, incomplete, fixed = [], [], None
    for entry in entries:
        path = resolve_path(entry["config"])
        c, _ = _configuration(load_local_config(path), "sft")
        _require(all(c[key] == entry[key] for key in ("seed", "dpo_beta", "kl_coef")), "Plan/config parameters mismatch")
        comparable = {k: v for k, v in c.items() if k not in {"seed", "dpo_beta", "kl_coef", "output_dir", "config_path"}}
        _require(fixed is None or comparable == fixed, "Non-ablation configuration differs between runs")
        fixed = comparable
        root = Path(c["output_dir"])
        evaluation = Path(c.get("eval_output_dir", root / "evaluation")).resolve()
        protected = [root, evaluation, Path(c["model_path"]), Path(c["data_dir"])]
        _require(not any(output == p or p in output.parents or output in p.parents for p in protected), "Unsafe analysis output directory")
        manifests = [root / stage / "run_manifest.json" for stage in STAGES] + [evaluation / "run_manifest.json"]
        if any(not p.is_file() or _read(p).get("status") != "completed" for p in manifests):
            incomplete.append(entry["name"])
        configs.append((entry, path, c))
    _require(len({c["output_dir"] for _, _, c in configs}) == 9, "Run output directories must be distinct")
    if incomplete:
        return {"status": "incomplete", "completed_runs": 9 - len(incomplete), "expected_runs": 9, "incomplete_runs": incomplete}
    runs, signatures = [], []
    files = {str(plan_path): sha256_file(plan_path)}
    for entry, path, c in configs:
        run, signature, inputs = _load_run(entry, path, c)
        runs.append(run)
        signatures.append(signature)
        files.update(inputs)
    validate_comparability(runs, signatures)
    sources = {name: sha256_file(ROOT / name) for name in (
        "rlhf_lab/study_report.py", "scripts/summarize_campaign.py", "rlhf_lab/evaluate.py", "rlhf_lab/local_evaluate.py",
        "rlhf_lab/local_runtime.py", "rlhf_lab/local_training.py", "rlhf_lab/data.py", "rlhf_lab/runtime.py")}
    identity = {"plan_sha256": files[str(plan_path)], "input_sha256": files, "source_sha256": sources,
                "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}
    _require(all(sha256_file(Path(path)) == digest for path, digest in files.items()), "Inputs changed during aggregation")
    if output.exists():
        manifest = _read(output / "run_manifest.json")
        _require(manifest.get("status") == "completed" and manifest.get("identity") == identity,
                 "Existing analysis identity mismatch; refusing overwrite")
        _artifact_hashes(output, manifest.get("artifacts"))
        return {"status": "reused", "output": str(output), "completed_runs": 9}
    count = len(signatures[0]["selection"])
    summary = {"schema_version": 1, "status": "completed", "name": plan.get("name", "RLHF controlled study"),
               "completed_runs": 9, "validation_prompts": count, "prompts_sha256": _canonical(signatures[0]["selection"]),
               "runs": runs, "by_bundle": _aggregate(runs), "parent_fingerprints_matched": True,
               "human_judgments": sum(r["human_judgments"] for r in runs), "human_win_rate": None,
               "test_labels_decoded": False, "protocol": plan.get("protocol", {}),
               "limitations": ["数据来自官方 train 的有序前缀，当前划分不是官方独立留出集。",
                   f"共同验证原文仅 {count} 篇；跨 seed 重复使用这些原文，不能将其相乘当作独立样本。",
                   "只有一个小型 GPT-2 模型；早期 pilot 已被查看并影响本轮协议，因此不是未触碰的确认性实验。",
                   "最终 test 标签仍封存。没有已填人工评价时，人工胜率为空；训练 RM 分数不是独立质量证据。",
                   "重复率和长度只是代理指标；NLL 和 RM 的变化不证明摘要事实性或质量提升。",
                   "配对 prompt bootstrap 条件于单个 seed；跨三个 seed 的样本标准差另行报告。",
                   "模型权重由运行前的独立审计核验；本汇总重验阶段清单与评估文件，不重新加载或哈希全部训练权重。"]}
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("report.html", _render(summary))):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(payload)
    manifest = {"schema_version": 1, "status": "completed", "identity": identity,
                "artifacts": {name: sha256_file(output / name) for name in ("summary.json", "report.html")}}
    with (output / "run_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"status": "completed", "output": str(output), "completed_runs": 9,
            "summary": str(output / "summary.json"), "report": str(output / "report.html")}
