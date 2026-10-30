"""CPU-only V10 analysis routing fixtures; no model or performance measurement.

The complete artificial V6 JSON fixtures exercise output, denominator, decoder,
statistics and budget checks. Only the independently tested historical closure,
capacity, and execution-version routing boundaries are injected here.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_analysis_v10 as an
from alignment_lab import evaluation_protocol_v2 as ep
from alignment_lab import evaluation_execution_v6 as fixture_execution
from tests_v2 import test_evaluation_analysis_v6 as legacy


saved = legacy.saved


class V10AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.old = legacy.AnalysisFixture(methodName="test_reordered_scores_preserve_ids_and_duplicate_scores_fail")
        self.old.setUp()
        self.addCleanup(self.old.doCleanups)
        self.root = self.old.root
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in an.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"# synthetic V10 source identity, never an executable model\n")
        self.sources = {**self.old.sources, **{name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                             for name in an.REQUIRED_SOURCES}}
        self.request = copy.deepcopy(self.old.request)
        self.request["source_sha256"] = self.sources
        self.plan = {
            "protocol": self.request["protocol"], "roster": self.request["roster"],
            "source_sha256": self.sources,
            "deadline_epoch": self.request["verification_allocation"]["evaluation_deadline_epoch"],
            "output": str(self.root / "runs"), "resource_profile": self.old.fixture.resource_profile,
            "decoder": {"tokenizer_sha256": self.old.token_sha},
            "inherited_capacity": {"synthetic_test_boundary": "V5-through-V6"},
            "inherited_inference": {}, "inherited_scores": {},
        }
        self.accepted = {}
        for slot in an.INHERITED_SLOTS:
            score = an._read(self.request["scores"][slot])
            inference = an._read(score["inference_manifest"])
            self.plan["inherited_inference"][slot] = {"fixture_slot": slot, "kind": "combined_seven_pins"}
            self.plan["inherited_scores"][slot] = copy.deepcopy(self.plan["inherited_inference"][slot])
            self.accepted[slot] = (
                {"slot": slot, "manifest_binding": score["inference_manifest"], "request": inference["request"]},
                {"slot": slot, "manifest_binding": score["inference_manifest"], "score_binding": self.request["scores"][slot]},
            )
        for slot in an.NEW_SLOTS:
            self._new_slot(slot)
        self.request_path = self.root / "analysis-v10-request.json"
        self.output = self.root / "primary-analysis-v10"
        self.save_plan_request()
        f = self.old.fixture
        self.stack.enter_context(patch.object(an, "ROOT", self.root))
        self.stack.enter_context(patch.object(an, "load_evaluation_assets", return_value=f.assets))
        self.stack.enter_context(patch.object(an, "load_final_holdout", return_value=f.held))
        self.stack.enter_context(patch.object(an, "score_ifeval", side_effect=legacy.artificial_scores))
        self.stack.enter_context(patch.object(an, "_decode", side_effect=self.old.decode))
        self.stack.enter_context(patch.object(an, "verify_evaluation_result", side_effect=fixture_execution.verify_evaluation_result))
        self.bridge = self.stack.enter_context(patch.object(an, "_accepted_inheritance", side_effect=self._bridge))
        capacity = importlib.import_module("alignment_lab.evaluation_capacity_inheritance_v10")
        self.capacity = self.stack.enter_context(patch.object(capacity, "verify_inherited_capacity",
            return_value={"inheritance_verified": True, "synthetic_test_boundary": True}))

    def _bridge(self, plan, slot):
        if (plan["inherited_inference"][slot] != self.plan["inherited_inference"][slot]
                or plan["inherited_scores"][slot] != self.plan["inherited_scores"][slot]):
            raise ValueError("Synthetic historical closure pin drift")
        return copy.deepcopy(self.accepted[slot])

    def _new_slot(self, slot):
        result = an._read(self.request["scores"][slot])
        manifest_pin = result["inference_manifest"]
        manifest = an._read(manifest_pin)
        directory = Path(manifest_pin["path"]).parent
        request = manifest["request"]
        request["source_sha256"] = self.sources
        request["deadline_epoch"] = self.plan["deadline_epoch"]
        manifest["deadline_epoch"] = self.plan["deadline_epoch"]
        progress = ep.parse((directory / "progress.json").read_bytes())
        progress["request"] = request
        progress["deadline_epoch"] = self.plan["deadline_epoch"]
        for name, value in (("request.snapshot.json", request), ("progress.json", progress)):
            pin = saved(directory / name, value)
            manifest["files"][name] = {"sha256": pin["sha256"], "bytes": (directory / name).stat().st_size}
        manifest_pin = saved(directory / "manifest.json", manifest)
        score_request = an._read(result["request"])
        score_request.update(inference_manifest=manifest_pin["path"], inference_manifest_sha256=manifest_pin["sha256"],
            source_sha256=self.sources, deadline_epoch=self.plan["deadline_epoch"])
        result["request"] = saved(Path(result["request"]["path"]), score_request)
        result["inference_manifest"] = manifest_pin
        result["deadline_epoch"] = self.plan["deadline_epoch"]
        score_directory = self.root / "runs" / (slot.replace("/", "-") + "-score")
        saved(score_directory / "request.snapshot.json", score_request)
        self.request["scores"][slot] = saved(score_directory / "scores.json", result)

    def save_plan_request(self):
        self.request["evaluation_plan"] = saved(self.root / "evaluation-v10-plan.json", self.plan)
        self.request_pin = saved(self.request_path, self.request)

    def run_summary(self, **kwargs):
        return an.summarize_evaluation(self.request_path, self.request_pin["sha256"], output=self.output, **kwargs)

    def test_seven_inherited_three_new_full_analysis_and_pinned_reuse(self):
        completed = self.run_summary()
        self.assertTrue(completed["primary_evaluation_complete"])
        self.assertFalse(completed["research_complete"])
        summary = ep.parse((self.output / "summary.json").read_bytes())
        self.assertEqual(set(summary["input_scores"]), an.SLOTS)
        self.assertEqual(len(summary["instruction_following"]["per_model"]), 8)
        self.assertEqual(summary["human_labels"], 0)
        self.assertIsNone(summary["human_win_rate"])
        self.assertEqual(summary["excluded"], an.EXCLUDED)
        self.assertEqual(summary["continuation_provenance"]["inherited_complete_slots"], sorted(an.INHERITED_SLOTS))
        self.assertEqual(summary["continuation_provenance"]["new_complete_slots"], sorted(an.NEW_SLOTS))
        self.assertEqual(summary["continuation_provenance"]["partial_prior_ppo44_records_reused"], 0)
        self.assertGreaterEqual(self.capacity.call_count, 2)
        counts = {slot: sum(call.args[1] == slot for call in self.bridge.call_args_list) for slot in an.INHERITED_SLOTS}
        self.assertTrue(all(count >= 3 for count in counts.values()))
        before = self.capacity.call_count
        receipt = an.verify_analysis(self.output, completed["manifest_sha256"])
        self.assertTrue(receipt["primary_evaluation_complete"])
        self.assertGreater(self.capacity.call_count, before)
        self.assertTrue(self.run_summary(expected_manifest_sha256=completed["manifest_sha256"])["reused"])
        with self.assertRaises(FileExistsError):
            self.run_summary()

    def test_exact_seven_slot_maps_reject_partial_ppo44_missing_and_extra(self):
        for field in ("inherited_inference", "inherited_scores"):
            for change in ("missing", "partial303", "reward"):
                with self.subTest(field=field, change=change):
                    plan = copy.deepcopy(self.plan)
                    if change == "missing":
                        plan[field].pop("ifeval/dpo44")
                    else:
                        plan[field]["ifeval/ppo44" if change == "partial303" else "rm_bench/reward"] = {"completed_records": 303}
                    with self.assertRaisesRegex(ValueError, "seven"):
                        an._inheritance_shape(plan)

    def test_inherited_score_pin_is_not_replaceable_by_another_complete_score(self):
        self.request["scores"]["ifeval/dpo44"] = self.request["scores"]["ifeval/dpo43"]
        self.save_plan_request()
        with self.assertRaisesRegex(ValueError, "inherited accepted score"):
            self.run_summary()
        self.assertFalse(self.output.exists())

    def test_failure_of_capacity_or_historical_recheck_blocks_publication(self):
        self.capacity.side_effect = ValueError("V5 through V6 capacity source drift")
        with self.assertRaisesRegex(ValueError, "capacity source drift"):
            self.run_summary()
        self.assertFalse(self.output.exists())
        self.capacity.side_effect = None
        self.bridge.side_effect = ValueError("V6 score not accepted or closure tampered")
        with self.assertRaisesRegex(ValueError, "not accepted"):
            self.run_summary()
        self.assertFalse(self.output.exists())

    def test_completed_reuse_rechecks_both_inference_and_score_binding(self):
        completed = self.run_summary()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        for index, field in ((0, "manifest_binding"), (0, "request"), (1, "score_binding")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.accepted["ifeval/dpo44"])
                changed = self.accepted["ifeval/dpo44"][index][field]
                changed["sha256" if field != "request" else "model_id"] = "f" * 64
                with self.assertRaises(ValueError):
                    an.verify_analysis(self.output, completed["manifest_sha256"])
                self.accepted["ifeval/dpo44"] = original
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_historical_inference_cannot_enter_fresh_ppo44_slot(self):
        slot = "ifeval/ppo44"
        result = an._read(self.request["scores"][slot])
        inference = an._read(result["inference_manifest"])
        for field, value in (("source_sha256", self.old.sources), ("deadline_epoch", self.plan["deadline_epoch"] + 1),
                             ("output", str(self.root / "old-partial303")), ("resource_profile", {"wrong": True})):
            with self.subTest(field=field):
                changed = copy.deepcopy(inference)
                changed["request"][field] = value
                with self.assertRaisesRegex(ValueError, "New V10 inference"):
                    an._verify_inherited_slot(result["inference_manifest"], changed, slot, self.request)

    def test_new_scores_must_bind_current_source_decoder_directory_and_budget(self):
        slot = "ifeval/ppo44"
        for field, value in (("source_sha256", self.old.sources), ("decoder", {"different": True}),
                             ("deadline_epoch", self.plan["deadline_epoch"] + 1)):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan[field] = value
                request = copy.deepcopy(self.request)
                request["evaluation_plan"] = saved(self.root / ("bad-plan-" + field + ".json"), plan)
                with self.assertRaisesRegex(ValueError, "New V10 scoring"):
                    an._scored(request["scores"][slot], slot, request, self.old.protocol)

    def test_all_ten_scores_source_pins_and_full_denominators_remain_required(self):
        self.request["scores"].pop("ifeval/ppo44")
        self.save_plan_request()
        with self.assertRaisesRegex(ValueError, "eight"):
            self.run_summary()
        self.assertFalse(self.output.exists())
        score = an._read(self.request["scores"]["ifeval/base"])["scores"]
        score["denominators"]["prompts"] = 303
        with self.assertRaisesRegex(ValueError, "541/834"):
            an._normalize_ifeval(score, self.old.fixture.ifeval)

    def test_changed_new_source_invalidates_before_reading_models(self):
        (self.root / "alignment_lab/evaluation_analysis_v10.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_summary()
        self.assertFalse(self.output.exists())


class LightweightV10Tests(unittest.TestCase):
    def test_combined_inheritance_api_checks_inference_and_score_once(self):
        from alignment_lab import evaluation_inheritance_v10 as inheritance
        slot = "ifeval/dpo44"
        binding = {"manifest": {"path": "synthetic-manifest", "sha256": "1" * 64},
                   "score": {"path": "synthetic-scores", "sha256": "2" * 64}}
        plan = {"inherited_inference": {slot: binding}, "inherited_scores": {slot: copy.deepcopy(binding)}}
        accepted = {"slot": slot, "manifest_binding": binding["manifest"], "score_binding": binding["score"],
                    "request": {"task": "ifeval", "model_id": "dpo44"}, "score_request": {"test": "scoring"},
                    "verification": {"official_scoring_complete": True}}
        with patch.object(inheritance, "verify_inherited_inference", return_value=accepted) as verifier:
            first, scored = an._accepted_inheritance(plan, slot)
            self.assertEqual(first, accepted)
            self.assertEqual(scored["score_binding"], binding["score"])
            verifier.assert_called_once_with(binding, plan)
            for field, changed in (("slot", "ifeval/ppo44"), ("score_binding", {"wrong": True}),
                                   ("manifest_binding", {"wrong": True}), ("verification", {"official_scoring_complete": False})):
                with self.subTest(field=field):
                    verifier.return_value = {**accepted, field: changed}
                    with self.assertRaisesRegex(ValueError, "exact accepted inference"):
                        an._accepted_inheritance(plan, slot)
        plan["inherited_scores"][slot]["score"] = {"wrong": True}
        with self.assertRaisesRegex(ValueError, "seven-pin"):
            an._accepted_inheritance(plan, slot)

    def test_analysis_import_is_stdlib_only_and_cuda_uninitialized(self):
        root = Path(__file__).resolve().parents[1]
        code = ("import sys; sys.path.insert(0,sys.argv[1]); import alignment_lab.evaluation_analysis_v10; "
                "assert 'torch' not in sys.modules; print('torch_imported=False')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root)],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"), capture_output=True,
            text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("torch_imported=False", result.stdout)


if __name__ == "__main__":
    unittest.main()
