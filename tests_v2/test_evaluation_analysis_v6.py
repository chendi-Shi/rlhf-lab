"""Artificial saved-output fixtures, never real model performance measurements."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_analysis_v6 as an
from alignment_lab import evaluation_protocol_v2 as ep
from alignment_lab import official_evaluation_v6 as official
from alignment_lab import evaluation_execution_v6 as execution
import tests_v2.test_evaluation_execution_v6 as fixture_module


def saved(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ep.canonical(value))
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def artificial_scores(assets, outputs, *, protocol, model_id, decode_tokens):
    """A deliberately simple fixture oracle, not an alternate official scorer."""
    indexed = ep._index(outputs, "key")
    if set(indexed) != set(assets.ifeval.ids):
        raise ValueError("Fixture oracle needs all original IDs")
    records = []
    for source in assets.ifeval.rows:
        passed = indexed[source["key"]]["response"] == "H"
        item = {"key": source["key"], "instruction_id_list": source["instruction_id_list"],
                "follow_instruction_list": [passed] * len(source["instruction_id_list"]),
                "follow_all_instructions": passed}
        records.append({"key": source["key"], "strict": dict(item), "loose": dict(item)})
    prompt = sum(r["strict"]["follow_all_instructions"] for r in records) / 541
    instruction = sum(sum(r["strict"]["follow_instruction_list"]) for r in records) / 834
    return {"protocol_sha256": protocol.sha256, "model_id": model_id, "official_evaluator": True,
            "denominators": {"prompts": 541, "instructions": 834},
            "dependency_diagnostics": {"synthetic_test_fixture": True},
            "per_prompt": records, "metrics": {mode + "_" + unit + "_accuracy": value
                for mode in ("strict", "loose") for unit, value in (("prompt", prompt), ("instruction", instruction))}}


class AnalysisFixture(unittest.TestCase):
    def setUp(self):
        # Reuse its exact accepted-artifact JSON schema, not any real model or
        # outputs. Existing verifier and full-denominator checks still execute.
        self.fixture = fixture_module.ExecutionTests(methodName="test_full_rm_bench_six_answers_per_record")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        f = self.fixture
        self.root = f.root
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(an, "ROOT", self.root))
        for name in an.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"# artificial fixture identity\n")
        tokenizer = self.root / "tokenizer.json"
        tokenizer.write_bytes(b"not a real tokenizer; test decoder is explicitly injected")
        self.token_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        # Repair all fixture manifest/parent/acceptance links after adding the
        # model-bound tokenizer hash required by production replay validation.
        base = Path(f.protocol["model_manifests"]["base"]["path"])
        base_pin = saved(base, {"files": {"tokenizer.json": {"sha256": self.token_sha}}})
        f.protocol["model_manifests"]["base"] = base_pin
        f.roster["model_manifest_sha256"] = base_pin["sha256"]
        for model_id, stage in execution.MODEL_STAGES.items():
            record = f.roster["models"][model_id]
            accepted_path = Path(record["acceptance_path"])
            accepted = ep.parse(accepted_path.read_bytes())
            accepted["snapshot_manifest_sha256"] = base_pin["sha256"]
            if stage != "base":
                path = Path(record["artifact_dir"]) / "manifest.json"
                manifest = ep.parse(path.read_bytes())
                manifest["identity"]["inputs"] = {
                    "tokenizer/tokenizer.json": {"path": str(tokenizer), "sha256": self.token_sha, "size_bytes": tokenizer.stat().st_size}}
                for parent in manifest["identity"]["parents"]:
                    pin = f.protocol["model_manifests"][parent]
                    manifest["identity"]["parents"][parent] = {**pin, "size_bytes": Path(pin["path"]).stat().st_size}
                manifest["identity_sha256"] = ep.digest(manifest["identity"])
                pin = saved(path, manifest)
                f.protocol["model_manifests"][model_id] = pin
                record["manifest_sha256"] = pin["sha256"]
                row = accepted["models"][model_id]
                row["manifest_sha256"] = pin["sha256"]
                progress_path = Path(row["progress_path"])
                progress = ep.parse(progress_path.read_bytes())
                progress["identity"] = manifest["identity"]
                progress["artifact"]["manifest_sha256"] = pin["sha256"]
                row["progress_sha256"] = saved(progress_path, progress)["sha256"]
            record["acceptance_sha256"] = saved(accepted_path, accepted)["sha256"]
        f.roster_sha = saved(f.roster_path, f.roster)["sha256"]
        rmrows = list(f.rm.rows)
        for index, row in enumerate(rmrows):
            row["domain"] = ("chat", "math", "code", "safety-refuse")[index % 4]
        f.rm = fixture_module.dataset("rm_bench", "id", rmrows, f.rm.lengths)
        f.assets.rm_bench = f.rm
        f.assets.verify = lambda: None  # Only fabricated, immutable fixture rows.
        f.protocol["dataset_identities"]["rm_bench"] = f.rm.identity
        f.protocol.update(primary_metric="ifeval_strict_prompt_accuracy", bootstrap_seed=73,
                          bootstrap_replicates=2000)
        f.protocol_sha = saved(f.protocol_path, f.protocol)["sha256"]
        self.protocol = ep.EvaluationProtocol(ep.canonical(f.protocol))
        self.stack.enter_context(patch.object(an, "load_evaluation_assets", return_value=f.assets))
        self.stack.enter_context(patch.object(an, "load_final_holdout", return_value=f.held))
        # Acceptance audit itself is separately tested against actual CPU
        # tensors. Here only this heavy initial-audit boundary is substituted.
        acceptance = importlib.import_module("alignment_lab.evaluation_acceptance_v2")
        self.stack.enter_context(patch.object(acceptance, "verify_evaluation_roster", return_value=f.roster))
        self.stack.enter_context(patch.object(an, "score_ifeval", side_effect=artificial_scores))
        self.stack.enter_context(patch.object(an, "_decode", side_effect=self.decode))
        self.sources = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                        for name in an.REQUIRED_SOURCES}
        current = time.time()
        self.original_deadline = current - 60  # Original execution already ended.
        self.request = {"schema_version": 1, "kind": an.KIND,
            "protocol": {"path": str(f.protocol_path), "sha256": f.protocol_sha},
            "roster": {"path": str(f.roster_path), "sha256": f.roster_sha},
            "scores": {}, "source_sha256": self.sources,
            "verification_allocation": {"kind": "within_independent_evaluation_budget",
                "wall_seconds": 600, "deadline_epoch": current + 600,
                "evaluation_deadline_epoch": current + 1200}}
        plan = {"protocol": self.request["protocol"], "roster": self.request["roster"],
                "source_sha256": self.sources, "deadline_epoch": current + 1200, "inherited_inference": {}}
        self.request["evaluation_plan"] = saved(self.root / "evaluation-plan.json", plan)
        from alignment_lab import evaluation_inheritance_v6 as inheritance
        self.stack.enter_context(patch.object(inheritance, "ROOT", self.root))
        for slot in sorted(an.SLOTS):
            self.request["scores"][slot] = self.scoring_result(slot)
        self.request_path = self.root / "analysis-request.json"
        self.request_pin = saved(self.request_path, self.request)
        self.output = self.root / "primary-analysis"

    def decode(self, decoder, sequences, *, deadline):
        texts = [self.fixture.codec.tokenizer.decode(ids) for ids in sequences]
        receipt = {"schema_version": 1, "kind": "alignlab-cpu-decoded-actions-v1",
                   "tokenizer_sha256": self.token_sha, "tokenizers_version": "fixture-only",
                   "sequence_count": len(sequences), "texts": texts,
                   "torch_imported": False, "gpu_operations_performed": False}
        mapping = {tuple(ids): text for ids, text in zip(sequences, texts)}
        return mapping.__getitem__, receipt

    def scoring_result(self, slot):
        task, model_id = slot.split("/")
        f = self.fixture
        output = self.root / "runs" / slot.replace("/", "-")
        output.mkdir()
        request = {"schema_version": 1, "kind": execution.KIND, "task": task, "model_id": model_id,
                   "output": str(output), "deadline_epoch": self.original_deadline,
                   "protocol_path": str(f.protocol_path), "protocol_sha256": f.protocol_sha,
                   "roster_path": str(f.roster_path), "roster_sha256": f.roster_sha,
                   "source_sha256": self.sources, "resource_profile": f.resource_profile,
                   "capacity_manifest": f.capacity_manifest}
        data = {"ifeval": f.ifeval, "helpsteer_rm": f.held, "rm_bench": f.rm}[task]
        rows = []
        passed_count = {"base": 100, "sft": 200, "dpo42": 240, "dpo43": 200, "dpo44": 160,
                        "ppo42": 300, "ppo43": 320, "ppo44": 340}
        for index, source in enumerate(data.rows):
            row = {data.id_field: source[data.id_field], "model_id": model_id,
                   "model_manifest_sha256": self.protocol.identity["model_manifests"][model_id]["sha256"],
                   "protocol_sha256": self.protocol.sha256, "input_truncated": False,
                   "prompt_token_ids": [10, 20]}
            if task == "ifeval":
                passed = index < passed_count[model_id]
                row.update(prompt=source["prompt"], response="H" if passed else "",
                           generated_token_ids=[33, 151645] if passed else [151645],
                           termination="eos", prompt_tokens=2, generation_seed=0)
            elif task == "helpsteer_rm":
                row.update(record_id=source["record_id"], source_index=source["source_index"],
                    response1_score=1., response2_score=0., input_lengths=[4, 4],
                    response1_token_ids=[30, 151645], response2_token_ids=[31, 151645])
            else:
                row.update(score_chosen=[1., 1., 1.], score_rejected=[0., 0., 0.],
                    input_lengths_chosen=[4] * 3, input_lengths_rejected=[4] * 3,
                    chosen_token_ids=[[30, 151645]] * 3, rejected_token_ids=[[31, 151645]] * 3)
            rows.append(row)
        status = {"status": "evaluation_inference_passed", "request": request, "pid": os.getpid(),
                  "started_at_epoch": self.original_deadline - 120, "finished_at_epoch": self.original_deadline - 30,
                  "deadline_epoch": self.original_deadline, "completed_records": len(rows),
                  "inference_complete": True, "scoring_complete": False, "research_complete": False,
                  "formal_training": False,
                  **f.resource_evidence(self.original_deadline - 60),
                  "attention_execution": fixture_module.attention_fixture(model_id,
                      calls=(4034 * 28 if task == "helpsteer_rm" else 7962 * 28 if task == "rm_bench" else 1))}
        payloads = {"request.snapshot.json": ep.canonical(request), "protocol.snapshot.json": self.protocol._payload,
                    "roster.snapshot.json": ep.canonical(f.roster), "progress.json": ep.canonical(status),
                    "outputs.jsonl": b"\n".join(ep.canonical(r) for r in rows) + b"\n"}
        for name, payload in payloads.items():
            (output / name).write_bytes(payload)
        model = fixture_module.FakeModel(model_id)
        manifest = {"schema_version": 1, "kind": execution.RESULT_KIND, **status, "task": task, "model_id": model_id,
            "protocol": {**self.request["protocol"], "identity_sha256": self.protocol.sha256},
            "accepted_roster": self.request["roster"], "model_manifest": self.protocol.identity["model_manifests"][model_id],
            "dataset_identity": data.identity, "counts": execution._counts(task, len(rows)),
            "model_runtime": {**model.metadata, "cuda": True, "weights_unchanged": True},
            "frozen_audits": [model.frozen_audit(full=True)] * 2,
            "processes": f.system.processes(), "quality_evaluation_complete": False,
            "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                      for name, raw in payloads.items()}}
        inference = saved(output / "manifest.json", manifest)
        decoder = {"tokenizer_sha256": self.token_sha} if task == "ifeval" else None
        score_request = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
            "inference_manifest": inference["path"], "inference_manifest_sha256": inference["sha256"],
            "source_sha256": self.sources, "decoder": decoder, "deadline_epoch": self.original_deadline}
        score_req = saved(self.root / (slot.replace("/", "-") + "-score-request.json"), score_request)
        if task == "ifeval":
            scores = artificial_scores(f.assets, rows, protocol=self.protocol, model_id=model_id, decode_tokens=None)
            _, decoder_receipt = self.decode(decoder, [r["generated_token_ids"][:-1] for r in rows], deadline=1)
        elif task == "helpsteer_rm":
            scores = official.aggregate_helpsteer_rm(f.held, rows, protocol=self.protocol, model_id=model_id)
            decoder_receipt = None
        else:
            # The arithmetic is real; its input scores are explicitly fabricated.
            scores = official.aggregate_rm_bench(f.assets, rows, protocol=self.protocol, model_id=model_id)
            decoder_receipt = None
        result = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-result-v1",
            "status": "official_scoring_passed", "task": task, "model_id": model_id,
            "request": score_req, "inference_manifest": inference, "protocol_sha256": self.protocol.sha256,
            "scores": scores, "decoder_receipt": decoder_receipt, "human_labels": 0, "human_win_rate": None,
            "research_complete": False, "gpu_operations_performed": False,
            "finished_at_epoch": self.original_deadline - 10, "deadline_epoch": self.original_deadline}
        score_dir = self.root / "scoring" / slot.replace("/", "-")
        saved(score_dir / "request.snapshot.json", score_request)
        return saved(score_dir / "scores.json", result)

    def run_summary(self, **kwargs):
        return an.summarize_evaluation(self.request_path, self.request_pin["sha256"], output=self.output, **kwargs)

    def test_complete_replay_statistics_scope_and_pinned_reuse(self):
        result = self.run_summary()
        self.assertTrue(result["primary_evaluation_complete"])
        self.assertFalse(result["research_complete"])
        summary = ep.parse((self.output / "summary.json").read_bytes())
        algorithms = summary["instruction_following"]["alignment_algorithms"]
        self.assertAlmostEqual(algorithms["dpo"]["metrics"]["strict_prompt_accuracy"]["mean"], 200 / 541)
        self.assertAlmostEqual(algorithms["dpo"]["metrics"]["strict_prompt_accuracy"]["sample_std"], 40 / 541)
        self.assertAlmostEqual(algorithms["ppo"]["metrics"]["strict_prompt_accuracy"]["sample_std"], 20 / 541)
        direct = summary["instruction_following"]["strict_prompt_ppo_minus_dpo_by_seed"]
        self.assertAlmostEqual(direct["42"]["estimate"], 60 / 541)
        self.assertAlmostEqual(direct["43"]["estimate"], 120 / 541)
        self.assertAlmostEqual(direct["44"]["estimate"], 180 / 541)
        for values in algorithms.values():
            for pair in values["strict_prompt_difference_vs_shared_sft_by_seed"].values():
                self.assertEqual(pair["records"], 541)
                self.assertEqual(pair["clusters"], 541)
                self.assertEqual(pair["replicates"], 2000)
                self.assertFalse(pair["across_seed_pooling"])
        self.assertEqual(summary["excluded"], an.EXCLUDED)
        self.assertIsNone(summary["human_win_rate"])
        with self.assertRaises(FileExistsError):
            self.run_summary()
        reused = self.run_summary(expected_manifest_sha256=result["manifest_sha256"])
        self.assertTrue(reused["reused"])
        with patch.object(an.time, "time", return_value=time.time() + 10000):
            self.assertTrue(self.run_summary(expected_manifest_sha256=result["manifest_sha256"])["reused"])
        # Mutating an original output, without changing its saved scores, must
        # invalidate even the read-only reuse path.
        raw = self.root / "runs/ifeval-base/outputs.jsonl"
        raw.write_bytes(raw.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_summary(expected_manifest_sha256=result["manifest_sha256"])

    def test_completed_analysis_reuse_rejects_different_inherited_manifest_or_request(self):
        """A saved success never replaces the explicit inference authorization bridge."""
        from alignment_lab import evaluation_inheritance_v6 as inheritance
        plan_path = Path(self.request["evaluation_plan"]["path"])
        plan = ep.parse(plan_path.read_bytes())
        plan["inherited_inference"] = {"ifeval/base": {
            name: saved(self.root / ("bridge-" + name + ".json"), {"synthetic_bridge_fixture": name})
            for name in inheritance.BINDING_FIELDS}}
        self.request["evaluation_plan"] = saved(plan_path, plan)
        self.request_pin = saved(self.request_path, self.request)
        score = an._read(self.request["scores"]["ifeval/base"])
        inference = an._read(score["inference_manifest"])
        accepted = {"manifest_binding": copy.deepcopy(score["inference_manifest"]),
                    "request": copy.deepcopy(inference["request"])}
        # Only the separately tested historical-closure boundary is injected.
        # Full output, denominator, scoring and completed-analysis checks run.
        with patch.object(inheritance, "verify_inherited_inference", return_value=accepted) as bridge:
            completed = self.run_summary()
            self.assertTrue(an.verify_analysis(self.output, completed["manifest_sha256"])["primary_evaluation_complete"])
            before = {p.name: p.read_bytes() for p in self.output.iterdir()}
            for field in ("manifest_binding", "request"):
                with self.subTest(field=field):
                    wrong = copy.deepcopy(accepted)
                    if field == "manifest_binding":
                        wrong[field]["sha256"] = "f" * 64
                    else:
                        wrong[field]["model_id"] = "sft"
                    bridge.return_value = wrong
                    with self.assertRaisesRegex(ValueError, "explicitly inherited inference"):
                        an.verify_analysis(self.output, completed["manifest_sha256"])
                    self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_missing_task_and_source_change_never_create_analysis(self):
        self.request["scores"].pop("ifeval/ppo44")
        self.request_pin = saved(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "eight"):
            self.run_summary()
        self.assertFalse(self.output.exists())
        self.request["scores"]["ifeval/ppo44"] = saved(
            self.root / "extra-score.json", {"unused": True})
        (self.root / "alignment_lab/evaluation_analysis_v6.py").write_bytes(b"changed source")
        self.request_pin = saved(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_summary()

    def test_saved_status_does_not_override_wrong_flags_or_original_deadline(self):
        pin = self.request["scores"]["ifeval/base"]
        original = ep.parse(Path(pin["path"]).read_bytes())
        changed = copy.deepcopy(original)
        changed["scores"]["per_prompt"][0]["strict"]["follow_instruction_list"][0] = False
        self.request["scores"]["ifeval/base"] = saved(Path(pin["path"]), changed)
        self.request_pin = saved(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "schema|aggregates|replay"):
            self.run_summary()
        self.assertFalse(self.output.exists())
        changed = copy.deepcopy(original)
        changed["finished_at_epoch"] = original["deadline_epoch"] + 1
        self.request["scores"]["ifeval/base"] = saved(Path(pin["path"]), changed)
        self.request_pin = saved(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.run_summary()

    def test_reordered_scores_preserve_ids_and_duplicate_scores_fail(self):
        pin = self.request["scores"]["ifeval/base"]
        result = ep.parse(Path(pin["path"]).read_bytes())
        score = result["scores"]
        reordered = {**score, "per_prompt": list(reversed(score["per_prompt"]))}
        self.assertEqual(an._normalize_ifeval(reordered, self.fixture.ifeval)["per_prompt"], score["per_prompt"])
        duplicate = {**score, "per_prompt": score["per_prompt"][:-1] + score["per_prompt"][:1]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            an._normalize_ifeval(duplicate, self.fixture.ifeval)
        with self.assertRaisesRegex(ValueError, "Missing"):
            an._normalize_ifeval({**score, "per_prompt": score["per_prompt"][:-1]}, self.fixture.ifeval)

    def test_expired_analysis_allocation_and_replay_mismatch_never_publish(self):
        self.request["verification_allocation"]["deadline_epoch"] = time.time() - 1
        self.request_pin = saved(self.request_path, self.request)
        with self.assertRaises(TimeoutError):
            self.run_summary()
        self.assertFalse(self.output.exists())
        self.request["verification_allocation"]["deadline_epoch"] = time.time() + 600
        self.request_pin = saved(self.request_path, self.request)
        with patch.object(an, "_decode", return_value=(lambda _: "wrong", {"incorrect": True})):
            with self.assertRaisesRegex(ValueError, "decoder replay"):
                self.run_summary()
        self.assertFalse(self.output.exists())


class LightweightTests(unittest.TestCase):
    def test_cluster_bootstrap_keeps_repeated_observations_as_one_unit(self):
        left = {"a1": 1, "a2": 1, "b": 0}
        right = {k: 0 for k in left}
        clusters = {"a1": "a", "a2": "a", "b": "b"}
        result = official.paired_cluster_bootstrap(left, right, clusters, seed=73, replicates=2000)
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["estimate"], 2 / 3)
        self.assertEqual(result, official.paired_cluster_bootstrap(left, right, clusters, seed=73, replicates=2000))

    def test_real_analysis_and_roster_modules_import_without_torch(self):
        root = Path(__file__).resolve().parents[1]
        python = root / ".runtime/alignlab-eval-20261020/Scripts/python.exe"
        if not python.is_file():
            self.skipTest("The separately accepted CPU scorer environment is not installed here")
        code = ("import sys; sys.path.insert(0,sys.argv[1]); "
                "import alignment_lab.evaluation_analysis_v6, alignment_lab.evaluation_acceptance_v2; "
                "assert 'torch' not in sys.modules; print('torch_imported=False')")
        run = subprocess.run([str(python), "-I", "-B", "-c", code, str(root)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"), capture_output=True,
            text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("torch_imported=False", run.stdout)


if __name__ == "__main__":
    unittest.main()
