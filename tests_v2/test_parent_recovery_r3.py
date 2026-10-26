"""Real tiny CPU tests for a second paused RM segment followed by R3.

These use random miniature weights, real backward/Adam/checkpoints, and mocked
resource inventory only. They neither dispatch CUDA nor train a research model.
"""
import contextlib
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from alignment_lab import checkpointing as ck
from alignment_lab import parent_recovery as r1
from alignment_lab import parent_recovery_r2 as r2
from alignment_lab import parent_recovery_r3 as r3
from alignment_lab import study_stage as old
from alignment_lab.stage_engine import SupervisedConfig, _records, parameter_mapping, run_supervised
import test_study_stage as fixtures
from test_parent_recovery import Runtime as OriginalRuntime
from test_stage_artifacts import backend_for

TARGETS = ("q_proj", "k_proj", "v_proj")
TINY = {"sft": (17, 2, "sft_train", 1e-4), "reward": (35, 3, "preference_train", 5e-5)}


class Runtime(OriginalRuntime):
    def load(self, request):
        loaded = super().load(request)
        loaded.metadata["config"] = {"target_modules": TARGETS}
        if self.mode == "changed_metadata":
            loaded.metadata["config"]["target_modules"] = tuple(reversed(TARGETS))
        return loaded

    def processes(self):
        result = super().processes()
        # The fourth sample follows the first newly completed optimizer update.
        if self.mode == "pause_after_one" and self.calls >= 4:
            return []
        return result


class RecoveryR3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    @contextlib.contextmanager
    def cpu_stage(self, request, engine=None):
        request.update(expected_rows=TINY[request["stage"]][0], expected_updates=TINY[request["stage"]][1])
        backend, _ = backend_for(request["stage"])
        role = "reward" if request["stage"] == "reward" else "policy"
        loaded = SimpleNamespace(backend=backend, role=role, heads=("reward",) if role == "reward" else (),
            metadata={"fixture": "tiny random CPU, no CUDA evidence", "config": {"target_modules": TARGETS}},
            device=torch.device("cpu"), autocast=contextlib.nullcontext)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("alignment_lab.training_data.load_training_partition",
                                      return_value=self.fixture.partition(request)))
            stack.enter_context(patch("alignment_lab.gpu_probe.resources", return_value={"blockers": []}))
            stack.enter_context(patch("alignment_lab.model_loader.load_quantized_stage", return_value=loaded))
            stack.enter_context(patch("rlhf_lab.runtime.gpu_inventory", return_value={"fixture_only": True}))
            for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            if engine:
                stack.enter_context(patch("alignment_lab.stage_engine.run_supervised", side_effect=engine))
            yield loaded
        self.assertFalse(torch.cuda.is_initialized())

    def setUp(self):
        self.fixture = fixtures.StudyStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for module, origin in ((r1, 1), (r3, 2)):
            for name, value in (("ROOT", self.root), ("ORIGIN_UPDATE", origin), ("ROWS", 35), ("UPDATES", 3)):
                self.stack.enter_context(patch.object(module, name, value))
        for name, value in (("EXPECTED", TINY), ("PAD", 0), ("EOS", 2)):
            self.stack.enter_context(patch.object(old, name, value))
        sft_request = copy.deepcopy(self.fixture.request)
        with self.cpu_stage(sft_request):
            sft = old.run_study_stage(sft_request, execute=True)
        self.old_request = dict(sft_request, stage="reward", expected_rows=35, expected_updates=3,
            output=str(self.root / "runs/formal/reward"),
            parents={"sft": {k: sft["artifact"][k] for k in ("artifact_dir", "manifest_sha256")}})

        def pause_first(*args, **kwargs):
            callback, done = kwargs["checkpoint_callback"], False
            def save(cursor, extra):
                nonlocal done
                callback(cursor, extra)
                done = True
            kwargs.update(checkpoint_callback=save, stop_requested=lambda: done)
            return run_supervised(*args, **kwargs)

        with self.cpu_stage(self.old_request, pause_first):
            with self.assertRaises(InterruptedError):
                old.run_study_stage(self.old_request, execute=True)
        previous = Path(self.old_request["output"])
        self.old_progress = r1.read_json(previous / "progress.json")
        self.old_identity = self.old_progress["identity"]
        self.partition = self.fixture.partition(self.old_request)
        self.stack.enter_context(patch("alignment_lab.training_data.load_training_partition", return_value=self.partition))
        r1._write(previous.parent / "campaign.json", {"status": "failed", "plan_sha256": self.old_request["parent_plan_sha256"]})
        r1._write(previous.parent / "STOP", {"owner": "alignlab-formal-parents-campaign-v1", "reason": "identity_failure",
                                           "plan_sha256": self.old_request["parent_plan_sha256"]})
        origin = {"plan": r1.pin(self.old_request["parent_plan_path"]), "campaign": r1.pin(previous.parent / "campaign.json"),
                  "stop": r1.pin(previous.parent / "STOP")}
        for name in ("request", "progress", "identity", "selection", "metrics"):
            origin[name] = r1.pin(previous / (name + (".jsonl" if name == "metrics" else ".json")))
        origin["checkpoint"] = {"path": str(previous / "update-000001"),
                                "manifest_sha256": r1.sha(previous / "update-000001/manifest.json")}
        sources = dict(self.old_request["source_sha256"])
        self.add_sources(sources, r1.NEW_SOURCES)
        authorization = self.root / "reports/budget-authorization.json"
        r1._write(authorization, {"kind": "explicit-user-rm-recovery-budget-authorization-v1",
            "evidence": {"source": "direct_user_reply_in_this_chat"}, "authorized_scope": {
                "stage": "reward", "new_wall_seconds": 28800, "resume_origin_optimizer_update": 1,
                "target_total_optimizer_updates": 3, "target_total_pairs": 35, "automatic_retries": False,
                "original_budget_changed": False, "old_campaign_mutation_allowed": False,
                "new_output_directory_required": True}})
        plan_path = self.root / "r1-plan.json"
        plan = {"schema_version": 1, "kind": r1.PLAN_KIND, "output": str(self.root / "runs/r1"),
                "wall_seconds": 28800, "source_sha256": sources, "origin": origin,
                "budget_authorization": r1.pin(authorization), "allowed_background_processes": []}
        r1._write(plan_path, plan)
        start = time.time()
        request = {"schema_version": 1, "kind": r1.KIND, "stage": "reward", "output": plan["output"],
                   "recovery_plan_path": str(plan_path), "recovery_plan_sha256": r1.sha(plan_path),
                   "source_sha256": sources, "origin": origin,
                   "budget": {"started_at_epoch": start, "wall_seconds": 28800, "deadline_epoch": start + 28800}}
        request_path = self.root / "r1-request.json"
        r1._write(request_path, request)
        with self.assertRaisesRegex(ValueError, "Actual loaded model/runtime"):
            r1.run_recovery(request_path, r1.sha(request_path), execute=True, runtime=Runtime(request, self.old_request))
        r2_plan = copy.deepcopy(plan)
        r2_plan.update(execution_revision=r2.REVISION, output=str(self.root / "runs/r2"),
            prior_failed_dispatch={"request": r1.pin(request_path), "progress": r1.pin(Path(request["output"]) / "progress.json")})
        self.add_sources(r2_plan["source_sha256"], r2.NEW_SOURCES)
        retry_auth = self.root / "reports/r2-authorization.json"
        r1._write(retry_auth, {"kind": "explicit-user-rm-r2-dispatch-authorization-v1",
            "source": "direct_user_reply_in_this_chat", "allow_one_corrected_dispatch": True,
            "budget": request["budget"], "prior_request_sha256": r1.sha(request_path)})
        r2_plan["retry_authorization"] = r1.pin(retry_auth)
        r2_plan_path = self.root / "r2-plan.json"
        r1._write(r2_plan_path, r2_plan)
        self.prior_request = dict(request, output=r2_plan["output"], source_sha256=r2_plan["source_sha256"],
                                 recovery_plan_path=str(r2_plan_path), recovery_plan_sha256=r1.sha(r2_plan_path))
        prior_request_path = self.root / "r2-request.json"
        r1._write(prior_request_path, self.prior_request)
        with self.assertRaises(InterruptedError):
            r2.run_recovery(prior_request_path, r1.sha(prior_request_path), execute=True,
                            runtime=Runtime(self.prior_request, self.old_request, "pause_after_one"))
        self.prior_output = Path(self.prior_request["output"])
        prior_progress = r1.read_json(self.prior_output / "progress.json")
        self.assertEqual((prior_progress["status"], prior_progress["actual_optimizer_updates"]), ("paused", 2))
        prior = {}
        filenames = {"request": "request.json", "progress": "progress.json", "identity": "identity.json",
            "selection": "selection.json", "metrics": "metrics.jsonl", "migration_manifest": "migration-manifest.json",
            "migration_state_audit": "migration-state-audit.json", "metadata_comparison": "model-metadata-comparison.json"}
        for name, filename in filenames.items():
            prior[name] = r1.pin(self.prior_output / filename)
        prior["request"] = r1.pin(prior_request_path)
        prior["checkpoint"] = {"path": str(self.prior_output / "update-000002"),
                               "manifest_sha256": r1.sha(self.prior_output / "update-000002/manifest.json")}
        self.plan = copy.deepcopy(r2_plan)
        self.plan.update(kind=r3.PLAN_KIND, execution_revision=r3.REVISION,
                         output=str(self.root / "runs/r3"), prior_segment=prior)
        self.plan.pop("prior_failed_dispatch")
        self.plan.pop("retry_authorization")
        self.add_sources(self.plan["source_sha256"], r3.NEW_SOURCES)
        self.dispatch_auth_path = self.root / "reports/r3-authorization.json"
        self.dispatch_auth = {"kind": "explicit-user-rm-r3-dispatch-authorization-v1",
            "source": "direct_user_message_in_this_chat", "user_message": "现在继续", "allow_one_dispatch": True,
            "budget": request["budget"], "prior_request_sha256": prior["request"]["sha256"],
            "resume_optimizer_update": 2, "target_total_optimizer_updates": 3}
        r1._write(self.dispatch_auth_path, self.dispatch_auth)
        self.plan["dispatch_authorization"] = r1.pin(self.dispatch_auth_path)
        self.plan_path = self.root / "r3-plan.json"
        self.request = copy.deepcopy(self.prior_request)
        self.request.update(kind=r3.KIND, output=self.plan["output"], source_sha256=self.plan["source_sha256"],
                            recovery_plan_path=str(self.plan_path), prior_segment=prior)
        self.request_path = self.root / "r3-request.json"
        self.write_request()

    def add_sources(self, sources, names):
        for name in names:
            if name not in sources:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("independently frozen tiny CPU fixture source", encoding="utf8")
                sources[name] = r1.sha(path)

    def write_request(self):
        r1._write(self.plan_path, self.plan)
        self.request["recovery_plan_sha256"] = r1.sha(self.plan_path)
        r1._write(self.request_path, self.request)

    def run_(self, mode=None, execute=True):
        return r3.run_recovery(self.request_path, r1.sha(self.request_path), execute=execute,
                               runtime=Runtime(self.request, self.old_request, mode))

    def test_two_pauses_match_continuous_adam_parameters_rng_cursor_and_partial_batch(self):
        backend, _ = backend_for("reward")
        mapping = parameter_mapping(backend, "reward", heads=("reward",))
        optimizer = torch.optim.AdamW(mapping.values(), lr=5e-5)
        generator = torch.Generator(device="cpu").manual_seed(42)
        origin = self.request["origin"]["checkpoint"]
        restored = ck.load_checkpoint(origin["path"], expected_identity=self.old_identity, parameters=mapping,
            optimizer=optimizer, generators={"sampling": generator}, expected_manifest_sha256=origin["manifest_sha256"])
        before = SupervisedConfig(**self.old_identity["config"]["work"])
        config = replace(before, deadline_epoch=self.request["budget"]["deadline_epoch"])
        cursor, _ = r1.migrate_cursor(restored["cursor"], _records(self.partition.rows, backend, preference=True, eos_id=2), before, config)
        continuous = run_supervised(backend, "reward", optimizer, self.partition.rows, config, device="cpu", state=cursor)
        rng = ck._capture_rng({"sampling": generator})
        result = self.run_()
        output = Path(self.request["output"])
        saved = torch.load(output / "update-000003/state.pt", map_location="cpu", weights_only=True)
        self.assertTrue(r1._tree_equal(optimizer.state_dict(), saved["optimizer"]))
        self.assertTrue(all(torch.equal(p.detach(), saved["parameters"][n]) for n, p in mapping.items()))
        self.assertTrue(r1._tree_equal(rng, saved["rng"]))
        self.assertEqual(continuous.cursor, saved["cursor"])
        self.assertEqual((result["completion"]["historical_optimizer_updates"], result["completion"]["segment_optimizer_updates"]), (2, 1))
        self.assertEqual((result["completion"]["historical_rows"], result["completion"]["segment_rows"]), (32, 3))
        transition = r1.read_json(output / "migration-manifest.json")
        self.assertEqual(transition["changed_work_fields"], [])
        self.assertEqual(transition["changed_cursor_fields"], [])
        self.assertEqual(transition["old_cursor"], transition["new_cursor"])
        self.assertEqual(result["budget"], self.prior_request["budget"])
        self.assertEqual(result, r3.verify_recovery_output(output, expected_manifest_sha256=r1.sha(output / "manifest.json"),
                                                         expected_request=self.request))
        self.assertEqual(r1.read_json(self.prior_output / "progress.json")["status"], "paused")
        self.assertEqual(r1.read_json(Path(self.old_request["output"]).parent / "campaign.json")["status"], "failed")
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_authorization_cannot_dispatch_or_create_output(self):
        self.plan.pop("dispatch_authorization")
        self.write_request()
        self.assertEqual(self.run_(execute=False)["status"], "plan_only")
        with self.assertRaises((ValueError, KeyError)):
            self.run_()
        self.assertFalse(Path(self.request["output"]).exists())

    def test_authorization_cannot_change_scope_or_user_message(self):
        for key, value in (("allow_one_dispatch", False), ("user_message", "status only"),
                           ("resume_optimizer_update", 1), ("target_total_optimizer_updates", 4)):
            with self.subTest(key=key):
                changed = dict(self.dispatch_auth, **{key: value})
                r1._write(self.dispatch_auth_path, changed)
                self.plan["dispatch_authorization"] = r1.pin(self.dispatch_auth_path)
                self.write_request()
                with self.assertRaises(ValueError):
                    self.run_()
                self.assertFalse(Path(self.request["output"]).exists())

    def test_budget_cannot_reset_extend_or_disagree_with_frozen_plan(self):
        original = copy.deepcopy(self.request["budget"])
        for changes in ({"deadline_epoch": original["deadline_epoch"] + 1},
                        {"started_at_epoch": original["started_at_epoch"] + 1, "deadline_epoch": original["deadline_epoch"] + 1},
                        {"wall_seconds": 43200, "deadline_epoch": original["started_at_epoch"] + 43200}):
            with self.subTest(changes=changes):
                self.request["budget"] = dict(original, **changes)
                self.write_request()
                with self.assertRaises(ValueError):
                    self.run_(execute=False)
                self.assertFalse(Path(self.request["output"]).exists())

    def test_frozen_source_change_is_rejected_before_loading_or_output(self):
        (self.root / "alignment_lab/stage_engine.py").write_text("changed frozen numerical engine")
        with patch.object(Runtime, "load", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                self.run_()
        self.assertFalse(Path(self.request["output"]).exists())

    def test_previous_metrics_gap_cannot_be_relabelled_as_covered_updates(self):
        path = self.prior_output / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if row.get("event") != "objective_after_engine_return"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
        self.request["prior_segment"]["metrics"] = r1.pin(path)
        self.plan["prior_segment"] = self.request["prior_segment"]
        self.write_request()
        with self.assertRaises(ValueError):
            self.run_()
        self.assertFalse(Path(self.request["output"]).exists())

    def test_metadata_difference_is_recorded_without_checkpoint_or_claimed_success(self):
        with self.assertRaises(ValueError):
            self.run_("changed_metadata")
        output = Path(self.request["output"])
        receipt = r1.read_json(output / "model-metadata-comparison.json")
        self.assertFalse(receipt["all_fields_equal"])
        self.assertFalse((output / "origin-000002").exists())
        self.assertFalse((output / "manifest.json").exists())

    def test_existing_output_and_foreign_lock_are_never_overwritten(self):
        output = Path(self.request["output"])
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("prior evidence")
        with self.assertRaises(FileExistsError):
            self.run_()
        self.assertEqual(sentinel.read_text(), "prior evidence")
        # A distinct fresh path still may not remove or acquire an unknown lock.
        self.plan["output"] = str(self.root / "runs/r3-other")
        self.request["output"] = self.plan["output"]
        self.write_request()
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text("foreign identity")
        with self.assertRaises(FileExistsError):
            self.run_()
        self.assertEqual(lock.read_text(), "foreign identity")
        self.assertFalse(Path(self.request["output"]).exists())

    def test_noop_cursor_validation_preserves_exact_work_and_rejects_corruption(self):
        backend, _ = backend_for("reward")
        previous = r1.read_json(self.prior_output / "progress.json")
        config = SupervisedConfig(**previous["identity"]["config"]["work"])
        cursor = copy.deepcopy(previous["cursor"])
        rows = _records(self.partition.rows, backend, preference=True, eos_id=config.eos_id)
        migrated, receipt = r3.migrate_cursor(cursor, rows, config, config)
        self.assertEqual(migrated, cursor)
        self.assertIsNot(migrated, cursor)
        self.assertEqual(receipt["changed_work_fields"], [])
        self.assertEqual(receipt["changed_cursor_fields"], [])
        for config_change in (replace(config, deadline_epoch=config.deadline_epoch + 1), replace(config, seed=43),
                              replace(config, effective_batch=8), replace(config, max_updates=4)):
            with self.subTest(config=config_change), self.assertRaises(ValueError):
                r3.migrate_cursor(cursor, rows, config, config_change)
        for key, value in (("offset", 31), ("optimizer_updates", 1), ("response_tokens", 1),
                           ("active_rollout", True), ("data_sha256", "0" * 64), ("unexpected", 1)):
            with self.subTest(cursor_field=key), self.assertRaises(ValueError):
                r3.migrate_cursor(dict(cursor, **{key: value}), rows, config, config)


if __name__ == "__main__":
    unittest.main()
