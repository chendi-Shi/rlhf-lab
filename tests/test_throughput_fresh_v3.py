"""CPU-only regression proof for scalar PPO telemetry and the new fresh entry."""
import ast
import copy
import inspect
import itertools
import json
from pathlib import Path
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from alignment_lab import throughput_fresh_v1 as previous
from alignment_lab import throughput_fresh_v3 as fresh
from alignment_lab import alignment_process_guard_r3 as guard
from alignment_lab import alignment_supervisor_r3 as acceptance
from alignment_lab.stage_engine_observed import _immutable
from alignment_lab.training_telemetry import TrainingTelemetry as OriginalTelemetry


class FreshV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def observer(self, *, stage="ppo", original=False, name="telemetry"):
        clock = itertools.count(100)
        kind = OriginalTelemetry if original else fresh.TrainingTelemetry
        return kind(self.root / name, stage=stage, started_at_epoch=100,
                    clock=lambda: next(clock))

    def event(self, name, *, stage="ppo", count=1, indices=(0,)):
        common = {"event": name, "stage": stage, "target_update": 1,
                  "cursor": {"optimizer_updates": int(name == "optimizer_end"),
                             "order": (0, 1), "active_rollout": stage == "ppo"}}
        if stage == "ppo":
            common.update(response_indices=indices, record_ids=("a",), epoch=0, rollout_offset=0)
        if name == "optimizer_end":
            common.update(update=1, objective=0.25, gradient_norm=0.75, action_tokens=3)
            common.update({"responses": count} if stage == "ppo" else {"rows": 1})
        return _immutable(common)

    def emit_update(self, observer, *, stage="ppo", count=1, indices=(0,)):
        observer.observe(self.event("optimizer_start", stage=stage, count=count, indices=indices),
                         backend=None, role="policy")
        observer.observe(self.event("optimizer_end", stage=stage, count=count, indices=indices),
                         backend=None, role="policy")

    def test_positive_scalar_count_satisfies_unchanged_live_acceptance(self):
        observer = self.observer()
        self.emit_update(observer)
        rows = [json.loads(line) for line in (observer.path / "cost-events.jsonl").read_text().splitlines()]
        live = next(row for row in rows if row["event"] == "optimizer_end")
        self.assertEqual(live["responses"], 1)
        self.assertIs(type(live["responses"]), int)
        expected = [{"update": 1, "objective": 0.25, "gradient_norm": 0.75,
                     "responses": 1, "action_tokens": 3}]
        acceptance._verify_live_objectives(self.root, "ppo", expected)
        for key in expected[0]:
            changed = copy.deepcopy(expected)
            changed[0][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differ"):
                acceptance._verify_live_objectives(self.root, "ppo", changed)

    def test_old_bug_is_reproduced_and_only_scalar_is_added(self):
        old = self.observer(original=True, name="old/telemetry")
        new = self.observer(name="new/telemetry")
        self.emit_update(old)
        self.emit_update(new)
        new_rows = copy.deepcopy(new.points)
        for row in new_rows:
            if row["event"] == "optimizer_end":
                self.assertEqual(row.pop("responses"), 1)
        self.assertEqual(old.points, new_rows)
        self.assertEqual(old.cost, new.cost)
        expected = [{"update": 1, "objective": 0.25, "gradient_norm": 0.75,
                     "responses": 1, "action_tokens": 3}]
        with self.assertRaisesRegex(ValueError, "differ"):
            acceptance._verify_live_objectives(self.root / "old", "ppo", expected)
        acceptance._verify_live_objectives(self.root / "new", "ppo", expected)

    def test_invalid_counts_or_indices_are_rejected_before_optimizer_end_commit(self):
        for index, (count, indices) in enumerate(((None, (0,)), (True, (0,)), (1.0, (0,)),
                (0, ()), (-1, ()), ((1,), (0,)), (2, (0,)), (2, (0, 0)), (1, (-1,)),
                (1, (False,)))):
            with self.subTest(count=count, indices=indices):
                observer = self.observer(name=str(index))
                observer.observe(self.event("optimizer_start", count=count, indices=indices),
                                 backend=None, role="policy")
                with self.assertRaisesRegex(ValueError, "scalar response count"):
                    observer.observe(self.event("optimizer_end", count=count, indices=indices),
                                     backend=None, role="policy")
                self.assertFalse(any(row["event"] == "optimizer_end" for row in observer.points))

    def test_dpo_observation_and_costs_are_identical(self):
        old = self.observer(stage="dpo", original=True, name="old")
        new = self.observer(stage="dpo", name="new")
        self.emit_update(old, stage="dpo")
        self.emit_update(new, stage="dpo")
        self.assertEqual(old.points, new.points)
        self.assertEqual(old.cost, new.cost)
        self.assertFalse(any("responses" in row for row in new.points))

    def test_rollout_arrays_remain_out_of_cost_events(self):
        observer = self.observer()
        origin = {"rng_before_hex": "01", "rng_after_hex": "02", "model_versions": ("v",)}
        event = _immutable({"event": "rollout_collected", "stage": "ppo",
            "cursor": {"optimizer_updates": 0, "active_rollout": True, "order": (0,)},
            "responses": ({"action_ids": (1, 2, 3)},), "observation_origin": origin,
            "rollout_metadata": {"eos_id": 3, "pad_id": 0, "vocab_size": 8, "max_new_tokens": 3},
            "record_ids": ("a",), "advantages": ((0.1,),), "returns": ((0.2,),),
            "reference_logprobs": ((-0.3,),), "reward_scores": (0.4,)})
        with patch.object(observer, "commit_rollout") as committed:
            observer.observe(event, backend=None, role="policy")
        self.assertEqual(committed.call_count, 1)
        self.assertEqual(committed.call_args.kwargs["active"]["responses"][0]["action_ids"], [1, 2, 3])
        row = observer.points[0]
        for key in ("responses", "observation_origin", "advantages", "returns",
                    "reference_logprobs", "reward_scores"):
            self.assertNotIn(key, row)

    def test_stage_body_and_backend_are_unchanged(self):
        source = lambda function: ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))),
                                          include_attributes=False)
        self.assertEqual(source(previous.run_stage), source(fresh.run_stage))
        self.assertEqual(source(previous._adopt_backend), source(fresh._adopt_backend))
        backend = object()
        loaded = SimpleNamespace(backend=backend)
        for stage in ("dpo", "ppo"):
            self.assertIs(fresh._adopt_backend(loaded, {"legacy_request": {"stage": stage},
                                                       "selected_backend": "original"}), backend)

    def test_wrapper_only_allows_remaining_seed(self):
        plan = {"source_sha256": {}, "selected_backend": "original", "future_ppo_backend": "original",
                "authorization": {}, "validation_report": {}, "original_request": {}, "history": {},
                "recovery_authorization": {}, "supplemental_ppo43": {},
                "budget_authorization": {}, "execution_window": {"deadline_epoch": fresh.resume.DEADLINE + 86400},
                "original_started_at_epoch": fresh.resume.STARTED,
                "original_deadline_epoch": fresh.resume.DEADLINE}
        for stage in ("dpo", "ppo"):
            inner = {"stage": stage, "seed": 44}
            wrapper = fresh.make_request(inner, plan, {})
            self.assertEqual(wrapper["legacy_request"], inner)
            self.assertEqual(wrapper["kind"], "alignlab-throughput-fresh-v3")
            self.assertEqual(wrapper["selected_backend"], "original")
            for seed in (42, 43):
                with self.assertRaisesRegex(ValueError, "limited"):
                    fresh.make_request({"stage": stage, "seed": seed}, plan, {})

    def test_effective_deadline_changes_only_runtime_top_level_and_engine_config(self):
        from dataclasses import asdict
        from alignment_lab.alignment_supervisor import workload
        for stage in ("dpo", "ppo"):
            historical = {"stage": stage, "seed": 44, "deadline_epoch": fresh.resume.DEADLINE,
                          "continuation": {"deadline_epoch": fresh.resume.DEADLINE, "immutable": "history"},
                          "workload": workload(stage, 1024)}
            before = copy.deepcopy(historical)
            active_deadline = fresh.resume.DEADLINE + 86400
            runtime = fresh._runtime_request(historical, {"deadline_epoch": active_deadline})
            self.assertEqual(historical, before)
            expected = {**before, "deadline_epoch": active_deadline}
            self.assertEqual(runtime, expected)
            self.assertEqual(runtime["continuation"], before["continuation"])
            original_config, active_config = asdict(fresh.stage_config(before)), asdict(fresh.stage_config(runtime))
            self.assertEqual(active_config, {**original_config, "deadline_epoch": active_deadline})
            for invalid in ({**before, "seed": 43}, {**before, "deadline_epoch": active_deadline}):
                with self.assertRaisesRegex(ValueError, "historical budget"):
                    fresh._runtime_request(invalid, {"deadline_epoch": active_deadline})

    def test_marker_explicitly_pins_new_window_and_preserves_historical_budget_role(self):
        path = self.root / "ppo44.request.json"
        path.write_text("{}")
        wrapper = {"legacy_request": {"output": str(self.root / "ppo44")},
                   "selected_backend": "original", "plan": {}, "authorization": {}, "validation_report": {},
                   "budget_authorization": {"path": "frozen-authorization", "sha256": "a" * 64},
                   "execution_window": {"started_at_epoch": 1, "deadline_epoch": 2},
                   "original_deadline_epoch": fresh.resume.DEADLINE}
        marker = fresh._marker(wrapper)
        self.assertEqual(marker["budget_authorization"], wrapper["budget_authorization"])
        self.assertEqual(marker["execution_window"], wrapper["execution_window"])
        self.assertEqual(marker["historical_deadline_epoch"], fresh.resume.DEADLINE)
        self.assertEqual(marker["continuation_budget_role"], "historical_provenance_only")

    def test_acceptance_calls_complete_original_verifier_with_effective_request(self):
        deadline = fresh.resume.DEADLINE + 86400
        historical = {"stage": "ppo", "seed": 44, "output": str(self.root / "ppo44"),
                      "deadline_epoch": fresh.resume.DEADLINE, "supervisor_processes": [],
                      "continuation": {"deadline_epoch": fresh.resume.DEADLINE}}
        runtime = fresh._runtime_request(historical, {"deadline_epoch": deadline})
        wrapper = {"legacy_request": historical,
                   "execution_window": {"started_at_epoch": deadline - 3600, "deadline_epoch": deadline}}
        path = self.root / "ppo44.request.json"
        path.write_text("{}")
        observed = {"pid": 101}
        result = {"identity": {"config": {"work": {"deadline_epoch": deadline}, "throughput": {}}, "inputs": {}},
                  "started_at_epoch": deadline - 1800, "request": runtime, "throughput": {}, "pid": 101,
                  "process_guard": {"owned_identities": [observed]}}
        with patch.object(fresh, "validate_request", return_value=(Path(historical["output"]).resolve(), runtime)), \
             patch.object(fresh.legacy_supervisor, "_verify_stage", return_value=result) as complete_original, \
             patch.object(fresh, "read_json", return_value=wrapper), patch.object(fresh, "_marker", return_value={}), \
             patch.object(fresh, "_wrapper_inputs", return_value={}), patch.object(fresh, "_owned", return_value=[observed]):
            self.assertIs(fresh.verify_stage(historical["output"], wrapper, {101: observed}, 0), result)
            self.assertEqual(complete_original.call_args.args[1], runtime)
            result["identity"]["config"]["work"]["deadline_epoch"] -= 1
            with self.assertRaisesRegex(ValueError, "Actual engine deadline"):
                fresh.verify_stage(historical["output"], wrapper, {101: observed}, 0)

    def test_entry_identity_rejects_old_cli_even_with_same_hash(self):
        request_path = self.root / "ppo44.request.json"
        supervisor = {"pid": 100, "parent_pid": 99, "created_at_utc": "2026-10-24T00:00:00Z",
                      "command_line": "python supervisor"}
        child = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-24T00:00:01Z",
                 "command_line": f'python scripts/run_alignlab_throughput_fresh_v3.py --request "{request_path}" --request-sha256 abc --execute'}
        request = {"supervisor_processes": [supervisor]}
        self.assertEqual(fresh._owned([child, supervisor], request, request_path, "abc", pid=101), [child])
        old = {**child, "command_line": child["command_line"].replace("fresh_v3.py", "fresh_v1.py")}
        with self.assertRaisesRegex(RuntimeError, "Unverified"):
            fresh._owned([old, supervisor], request, request_path, "abc", pid=101)

    def test_stop_before_cuda_saves_same_receipt_and_exits(self):
        output = self.root / "ppo44"
        (self.root / "reports").mkdir()
        (self.root / "ppo44.request.json").write_text("{}", encoding="utf-8")
        own = {"pid": 101, "parent_pid": 100, "created_at_utc": "2026-10-24T00:00:00Z",
               "command_line": "python scripts/run_alignlab_throughput_fresh_v3.py --request fixture --execute"}
        inner = {"stage": "ppo", "seed": 44, "output": str(output), "deadline_epoch": time.time() + 3600,
            "alignment_plan_path": str(self.root / "old-plan.json"), "alignment_plan_sha256": "unused",
            "allowed_background_processes": [], "supervisor_processes": [], "source_sha256": {},
            "process_wait_spent_seconds": 0, "process_wait_policy": guard.POLICY, "workload": {"scope": "fixture"}}
        wrapper = {"legacy_request": inner, "plan": {"path": str(self.root / "new-plan.json")}, "source_sha256": {}}
        def stopped(*args):
            (output / "STOP").write_text("explicit stop", encoding="utf-8")
            return {}
        with patch.object(fresh, "ROOT", self.root), patch.object(fresh, "validate_request", return_value=(output, inner)), \
             patch.object(fresh, "_owned", return_value=[own]), patch.object(fresh, "_marker", return_value={}), \
             patch.object(fresh, "_wrapper_inputs", side_effect=stopped), \
             patch("alignment_lab.probe_dispatch._System", return_value=SimpleNamespace(processes=lambda: [own])), \
             patch.object(guard, "strict_admission", return_value={"caller_admission_allowed": True}), \
             patch("alignment_lab.gpu_probe.resources") as resources:
            with self.assertRaisesRegex(InterruptedError, "after saving current state"):
                fresh.run_stage(wrapper, execute=True)
            resources.assert_not_called()
        status = fresh.read_json(output / "progress.json")
        self.assertEqual(status["status"], "paused")
        self.assertTrue(status["process_guard"]["stopping_boundary"]["complete"])
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())


if __name__ == "__main__":
    unittest.main()
