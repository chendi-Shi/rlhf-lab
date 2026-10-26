"""No-GPU proofs that the continuation cannot repeat accepted work or extend time."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from alignment_lab import throughput_supervisor_v1 as subject


class SupervisorTests(unittest.TestCase):
    def plans(self):
        old = {k: "frozen-" + k for k in (
            "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
            "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
            "parent_acceptance_path", "parent_acceptance_sha256")}
        old.update(source_sha256={"old.py": "a" * 64}, allowed_background_processes=[], process_wait_policy={"frozen": True},
            continuation={"deadline_epoch": subject.resume.DEADLINE}, repair_attempt={"frozen": True},
            parents={"sft": {"frozen": "sft"}, "reward": {"frozen": "reward"}}, ppo_rows=1024)
        plan = {"_original": old, "output": str(subject.ROOT / "runs/new-continuation"),
            "original_plan": {"path": str(subject.ROOT / "configs/original.json"), "sha256": "b" * 64},
            "source_sha256": {**old["source_sha256"], "new.py": "c" * 64},
            "original_request": {"path": "original-ppo42", "sha256": "d" * 64},
            "history": {"pinned": "history"}, "authorization": {"pinned": "authorization"},
            "validation_report": {"pinned": "validation"}, "accepted_dpo42": {"pinned": "dpo42"},
            "selected_backend": "original"}
        plan["_authorized"] = {k: v for k, v in plan.items() if not k.startswith("_")}
        return plan

    def test_resume_does_not_reset_history_budget_or_create_new_random_seed(self):
        plan = self.plans()
        own = {1: {"pid": 1}}
        request, script = subject._build_request(plan, Path("new-plan"), "e" * 64, own, 2643.9, "ppo", 42, {})
        self.assertEqual(script, "run_alignlab_throughput_stage_v1.py")
        self.assertEqual(request["history"], plan["history"])
        self.assertEqual(request["original_started_at_epoch"], 1792634845.7777236)
        self.assertEqual(request["original_deadline_epoch"], 1792894045.7777236)
        self.assertEqual(request["process_wait_spent_seconds"], 2643.9)
        self.assertNotIn("seed", request)
        self.assertEqual([f"{s}{n}" for s, n in subject.SEQUENCE[1:]], ["ppo42", "dpo43", "ppo43", "dpo44", "ppo44"])

    def test_later_dpo_reuses_old_cache_original_plan_and_source_bytes(self):
        plan = self.plans()
        before = copy.deepcopy(plan)
        cache = {"path": "accepted-cache", "manifest_sha256": "a" * 64, "identity_sha256": "b" * 64}
        for seed in (43, 44):
            with patch("alignment_lab.throughput_fresh_v1.make_request", side_effect=lambda inner, plan, pin: inner):
                request, script = subject._build_request(plan, Path("new-plan"), "e" * 64, {1: {"pid": 1}}, 3000.1, "dpo", seed, cache)
            self.assertEqual(script, "run_alignlab_throughput_fresh_v1.py")
            self.assertEqual(request["reference_cache"], {"mode": "reuse", **cache})
            self.assertEqual(request["source_sha256"], plan["_original"]["source_sha256"])
            self.assertEqual(request["alignment_plan_sha256"], plan["original_plan"]["sha256"])
            self.assertEqual(request["workload"]["optimizer_updates"], 822)
            self.assertEqual(request["workload"]["rows"], 13139)
            self.assertEqual(request["deadline_epoch"], subject.resume.DEADLINE)
            self.assertEqual(request["process_wait_spent_seconds"], 3000.1)
        self.assertEqual(plan, before)

    def test_fallback_future_ppo_preserves_registered_protocol(self):
        plan = self.plans()
        with patch("alignment_lab.throughput_fresh_v1.make_request", side_effect=lambda inner, plan, pin: inner):
            request, script = subject._build_request(plan, Path("new-plan"), "e" * 64, {1: {"pid": 1}}, 3000.1, "ppo", 43, {})
        self.assertEqual(script, "run_alignlab_throughput_fresh_v1.py")
        self.assertEqual(request["seed"], 43)
        self.assertEqual(request["workload"]["optimizer_updates"], 2048)
        self.assertEqual(request["workload"]["rows"], 1024)
        self.assertEqual(request["workload"]["selection_seed"], 42)
        self.assertIsNone(request["reference_cache"])

    def test_alive_original_process_even_changed_command_blocks_parallel_resume(self):
        old = {"pid": 20, "created_at_utc": "2026-10-22T06:46:47Z", "command_line": "original"}
        plan = {"_prior_campaign": {"supervisor_processes": [old], "stages": [{"known_processes": []}]}}
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            subject._prior_processes_gone(plan, [{**old, "command_line": "changed"}])
        subject._prior_processes_gone(plan, [{**old, "created_at_utc": "2026-10-23T09:00:00Z"}])

    def test_cpu_proof_rejects_partial_test_coverage_or_cuda(self):
        sources = {"tests_throughput_v1/test_stage_v1.py": "a", "tests_throughput_v1/test_telemetry_v1.py": "b"}
        report = {"kind": "alignlab-throughput-cpu-validation-v1", "status": "passed", "device": "cpu",
            "cuda_initialized": False, "tests_run": 12, "errors": 0, "failures": 0, "skipped": 0,
            "source_before": sources, "source_after": sources,
            "modules": ["tests_throughput_v1.test_stage_v1", "tests_throughput_v1.test_telemetry_v1"]}
        with patch.object(subject, "_pin_record", return_value="report"), patch.object(subject, "_parse", return_value=report):
            subject._cpu_proof({"cpu_report": {}, "source_sha256": sources})
            report["modules"].pop()
            with self.assertRaises(ValueError):
                subject._cpu_proof({"cpu_report": {}, "source_sha256": sources})


if __name__ == "__main__":
    unittest.main()
