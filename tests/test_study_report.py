"""Small JSON SOFTWARE fixtures only: no training, model loading, or GPU calls."""
import json
from pathlib import Path

import pytest

from rlhf_lab import study_report as report
from rlhf_lab.evaluate import export_blind_pairs, summarize_generations
from rlhf_lab.local_runtime import load_local_config
from rlhf_lab.local_training import _configuration
from rlhf_lab.runtime import ROOT, sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_evaluation(directory):
    path = directory / "run_manifest.json"
    manifest = read(path)
    manifest["artifacts"] = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file() and p != path}
    write(path, manifest)


def fixture_study(tmp_path):
    template = read(ROOT / "configs/local-gpu.json")
    campaign = tmp_path / "campaign"
    entries = []
    for bundle, beta, kl in (("standard", .1, .05), ("low", .05, 0.), ("high", .2, .1)):
        for seed in (42, 43, 44):
            name = f"{bundle}-s{seed}"
            root = campaign / name
            config_path = tmp_path / "configs" / f"{name}.json"
            config = dict(template, seed=seed, dpo_beta=beta, kl_coef=kl, device="cpu", precision="float32",
                          output_dir=str(root), model_path=str(tmp_path / "model"), data_dir=str(tmp_path / "data"),
                          eval_limit=3, eval_preference_limit=3)
            write(config_path, config)
            c, _ = _configuration(load_local_config(config_path), "sft")
            entry = {"name": name, "bundle": bundle, "seed": seed, "dpo_beta": beta, "kl_coef": kl, "config": str(config_path)}
            entries.append(entry)
            stage_hashes = {}
            for stage in report.STAGES:
                parents = {} if stage == "sft" else {"sft": stage_hashes["sft"]}
                if stage == "ppo":
                    parents["reward"] = stage_hashes["reward"]
                path = root / stage / "run_manifest.json"
                write(path, {"stage": stage, "status": "completed", "config": c, "weights_changed": True,
                             "parent_sha256": parents, "dataset_manifest_sha256": "a" * 64,
                             "weight_fingerprint_after": f"{seed:064x}", "test_labels_decoded": False})
                stage_hashes[stage] = sha256_file(path)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            selection = [{"id": str(i), "prompt": f"article {i}", "reference": f"summary {i}"} for i in range(3)]
            manifest = {"status": "completed", "config": c, "eval_split": "valid", "test_labels_decoded": False,
                        "stage_manifest_sha256": stage_hashes, "dataset_manifest_sha256": "a" * 64,
                        "source_sha256": {"local_evaluate.py": sha256_file(ROOT / "rlhf_lab/local_evaluate.py")},
                        "model_id": c["model_id"], "model_revision": c["model_revision"],
                        "prompts_sha256": report._canonical(selection), "inference_dtype": "float32",
                        "decoding": {"do_sample": False, "max_new_tokens": 64}}
            policies = {}
            for label, offset, response in (("base", .3, "hello hello hello hello hello"), ("sft", 0., "clear summary"),
                                             ("dpo", -.2, "brief useful response"), ("ppo", -.1, "word word word word")):
                model_path = str(tmp_path / "model" if label == "base" else root / label)
                rows = [dict(row, response=response, model_label=label, model_path=model_path, seed=seed,
                             model_input_token_ids=[7, 8, i], generated_tokens=i + 2, reward_model_score=.1 + offset + i)
                        for i, row in enumerate(selection)]
                path = evaluation / f"{label}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                sidecar = {k: manifest[k] for k in ("config", "model_id", "model_revision", "eval_split", "prompts_sha256",
                                                   "dataset_manifest_sha256", "decoding", "inference_dtype")}
                sidecar.update(status="completed", model_label=label, model_path=model_path, examples=3,
                               output_sha256=sha256_file(path), test_labels_decoded=False)
                write(Path(str(path) + ".manifest.json"), sidecar)
                nll = [{"id": str(i), "tokens": i + 1, "negative_logprob_sum": (2 + offset + i / 10 + (seed - 42) / 10) * (i + 1)}
                       for i in range(3)]
                policies[label] = {"nll_rows": nll, "nll_examples": 3, "nll_tokens": 6,
                                   "chosen_token_nll": sum(r["negative_logprob_sum"] for r in nll) / 6,
                                   "mean_generated_tokens": 3., "mean_reward_model_score": 1.1 + offset,
                                   "text_diagnostics": summarize_generations(path)}
            for label in ("dpo", "ppo"):
                export_blind_pairs(evaluation / f"{label}.jsonl", evaluation / "sft.jsonl",
                                   evaluation / f"blind-{label}-vs-sft.csv", evaluation / f"private-{label}-vs-sft.json", seed)
            reward_rows = [{"id": str(i), "chosen_reward": value, "rejected_reward": 0., "correct": value > 0, "tie": False}
                           for i, value in enumerate((1., -1., 2.))]
            write(evaluation / "summary.json", {"status": "completed", "split": "valid", "device": "cpu",
                "test_labels_decoded": False, "generation_examples": 3, "policies": policies,
                "reward_model": {"pairs": 3, "correct": 2, "ties": 0, "accuracy": 2 / 3, "rows": reward_rows},
                "human_judgments": 0, "human_win_rate": None})
            (evaluation / "report.html").write_text("Software fixture; not a model result", encoding="utf-8")
            write(evaluation / "run_manifest.json", manifest)
            refresh_evaluation(evaluation)
    plan_path = tmp_path / "plan.json"
    write(plan_path, {"name": "Software fixture <script>", "output_dir": str(campaign), "runs": entries})
    return plan_path, campaign / "analysis"


def test_bootstrap_is_paired_weighted_numeric_and_reproducible():
    constant = report.paired_bootstrap([2., 5.], [1., 4.], [1, 3])
    assert constant["estimate"] == 1 and constant["ci95"] == [1, 1]
    result = report.paired_bootstrap([0., 10.], [0., 0.], [1, 3])
    assert result["estimate"] == 7.5 and result["ci95"] == [0, 10]
    assert result == report.paired_bootstrap([0., 10.], [0., 0.], [1, 3])
    assert result["samples"] == 2000 and result["rng_seed"] == 20261020
    with pytest.raises(ValueError, match="matched"):
        report.paired_bootstrap([1], [])


def test_prompt_sets_cannot_be_pooled_when_different():
    fingerprints = {"sft": "a" * 64, "reward": "b" * 64}
    runs = [{"seed": 42, "parent_fingerprints": fingerprints}] * 2
    with pytest.raises(ValueError, match="different prompts"):
        report.validate_comparability(runs, [{"selection": ["article one"]}, {"selection": ["article two"]}])


def test_complete_study_and_hash_verified_idempotent_reuse(tmp_path):
    plan, output = fixture_study(tmp_path)
    assert report.summarize_campaign(plan, output)["status"] == "completed"
    summary = read(output / "summary.json")
    assert len(summary["runs"]) == 9 and summary["human_judgments"] == 0 and summary["human_win_rate"] is None
    assert summary["parent_fingerprints_matched"] and summary["test_labels_decoded"] is False
    sft = summary["by_bundle"]["standard"]["policies"]["sft"]["chosen_token_nll"]
    assert sft["sample_sd"] == pytest.approx(.1)
    interval = summary["runs"][0]["paired_bootstrap"]["dpo_vs_sft"]["chosen_token_nll"]
    assert interval["estimate"] == pytest.approx(-.2)
    assert interval["ci95"] == pytest.approx([-.2, -.2])
    assert summary["runs"][0]["reward_model"]["wilson95"] == pytest.approx([.2076596008, .9385080553])
    document = (output / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in document and "&lt;script&gt;" in document
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert report.summarize_campaign(plan, output)["status"] == "reused"
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    (output / "report.html").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact hash"):
        report.summarize_campaign(plan, output)
    assert (output / "report.html").read_text() == "tampered"


def test_incomplete_study_has_no_analysis(tmp_path):
    plan, output = fixture_study(tmp_path)
    (output.parent / "high-s44/evaluation/run_manifest.json").unlink()
    result = report.summarize_campaign(plan, output)
    assert result["status"] == "incomplete" and result["completed_runs"] == 8
    assert not output.exists()


def test_consistently_altered_input_tokens_still_fail_across_runs(tmp_path):
    plan, output = fixture_study(tmp_path)
    evaluation = output.parent / "low-s42/evaluation"
    for label in report.POLICIES:
        path = evaluation / f"{label}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["model_input_token_ids"][0] = 999
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        sidecar = Path(str(path) + ".manifest.json")
        value = read(sidecar)
        value["output_sha256"] = sha256_file(path)
        write(sidecar, value)
    refresh_evaluation(evaluation)
    with pytest.raises(ValueError, match="Runs used different prompts/input tokens"):
        report.summarize_campaign(plan, output)
    assert not output.exists()


def test_unequal_seed_parent_fingerprints_are_rejected(tmp_path):
    plan, output = fixture_study(tmp_path)
    root = output.parent / "low-s42"
    hashes = {}
    for stage in report.STAGES:
        path = root / stage / "run_manifest.json"
        value = read(path)
        if stage == "sft":
            value["weight_fingerprint_after"] = "f" * 64
        value["parent_sha256"] = {} if stage == "sft" else {"sft": hashes["sft"]}
        if stage == "ppo":
            value["parent_sha256"]["reward"] = hashes["reward"]
        write(path, value)
        hashes[stage] = sha256_file(path)
    evaluation = root / "evaluation/run_manifest.json"
    value = read(evaluation)
    value["stage_manifest_sha256"] = hashes
    write(evaluation, value)
    with pytest.raises(ValueError, match="Unequal SFT/RM parent fingerprints"):
        report.summarize_campaign(plan, output)
    assert not output.exists()


def test_existing_analysis_refuses_changed_plan_identity(tmp_path):
    plan, output = fixture_study(tmp_path)
    report.summarize_campaign(plan, output)
    before = (output / "summary.json").read_bytes()
    value = read(plan)
    value["name"] = "Changed identity"
    write(plan, value)
    with pytest.raises(ValueError, match="Existing analysis identity mismatch"):
        report.summarize_campaign(plan, output)
    assert (output / "summary.json").read_bytes() == before
