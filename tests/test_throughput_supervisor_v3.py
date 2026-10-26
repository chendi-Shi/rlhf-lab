"""CPU-only recovery invariants and fake-system dispatch integration."""
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from alignment_lab import throughput_supervisor_v3 as subject


class RecoverySupervisorTests(unittest.TestCase):
    def execution_window(self):
        return {"kind": "authorized_remaining_training_window_v3",
            "started_at_epoch": 1792833376.012, "deadline_epoch": 1793006176.012, "wall_seconds": 172800,
            "scope": ["dpo44", "ppo44"], "completion_stops_early": True,
            "chosen_by": "assistant_under_user_authorization_to_proceed_without_original_deadline",
            "numeric_hours_explicitly_supplied_by_user": False,
            "all_preparation_validation_waiting_training_and_acceptance_count": True,
            "old_deadline_applies_to_historical_runs_only": True,
            "additional_training_or_automatic_retry": False}

    def plan(self):
        old = {k: "frozen-" + k for k in (
            "model_snapshot", "model_manifest_sha256", "tokenizer_dir", "audit_dir", "audit_manifest_sha256",
            "capacity_acceptance_path", "capacity_acceptance_sha256", "parent_plan_path", "parent_plan_sha256",
            "parent_acceptance_path", "parent_acceptance_sha256")}
        old.update(source_sha256={"old.py": "a" * 64}, allowed_background_processes=[], process_wait_policy={"frozen": True},
            continuation={"started_at_epoch": subject.resume.STARTED, "deadline_epoch": subject.resume.DEADLINE,
                          "process_wait_spent_seconds": 2643.9}, repair_attempt={"frozen": True},
            parents={"sft": {"frozen": "sft"}, "reward": {"frozen": "reward"}}, ppo_rows=1024)
        plan = {**old, "_original": old, "output": str(subject.ROOT / "runs/new-v3"),
            "original_plan": {"path": str(subject.ROOT / "configs/original.json"), "sha256": "b" * 64},
            "source_sha256": {**old["source_sha256"], "new.py": "c" * 64},
            "original_request": {"path": "original-ppo42", "sha256": "d" * 64},
            "history": {"pinned": "history"}, "authorization": {"pinned": "authorization"},
            "validation_report": {"pinned": "validation"}, "accepted_dpo42": {"pinned": "dpo42"},
            "selected_backend": "original", "future_ppo_backend": "original", "wall_seconds": 259200,
            "failed_v2_plan": {"path": "prior-plan", "sha256": "f" * 64},
            "failed_v2_campaign": {"path": "prior-campaign", "sha256": "e" * 64},
            "recovery_authorization": {"path": "recovery", "sha256": "a" * 64},
            "supplemental_ppo43": {"path": "acceptance", "sha256": "b" * 64},
            "budget_authorization": {"path": "new-window", "sha256": "c" * 64},
            "execution_window": self.execution_window()}
        plan["continuation"] = {**old["continuation"],
            "started_at_epoch": plan["execution_window"]["started_at_epoch"],
            "deadline_epoch": plan["execution_window"]["deadline_epoch"]}
        plan["_authorized"] = {k: v for k, v in plan.items() if not k.startswith("_")}
        return plan

    def authorization(self, plan):
        return {"kind": "alignlab-explicit-ppo43-supplemental-recovery-authorization-v3",
            "source": "direct_user_reply_in_this_chat", "user_selected_answer": "允许补验收并继续剩余训练",
            "prior_plan": plan["failed_v2_plan"], "prior_campaign": plan["failed_v2_campaign"],
            "original_started_at_epoch": subject.resume.STARTED, "original_deadline_epoch": subject.resume.DEADLINE,
            "wall_seconds": 259200, "scope": {"remaining_training": ["dpo44", "ppo44"], "retrain_ppo43": False,
            "future_automatic_retry_authorized": False, "supplemental_acceptance_existing_ppo43": True,
            "original_sources_outputs_preserved": True, "requires_full_independent_acceptance": True}}

    def test_exact_authorization_rejects_budget_extension_repeat_or_other_incident(self):
        plan = self.plan()
        authorization = self.authorization(plan)
        changes = [lambda a: a.update(original_deadline_epoch=subject.resume.DEADLINE + 1),
                   lambda a: a["scope"].update(retrain_ppo43=True),
                   lambda a: a["scope"].update(future_automatic_retry_authorized=True),
                   lambda a: a["scope"].update(remaining_training=["ppo43", "dpo44", "ppo44"]),
                   lambda a: a.update(prior_campaign={"path": "other", "sha256": "e" * 64}),
                   lambda a: a.update(user_selected_answer="跑就行了")]
        with patch.object(subject, "_pin_record", return_value="authorization"):
            with patch.object(subject, "_parse", return_value=authorization):
                subject._recovery_authority(plan)
            for change in changes:
                altered = copy.deepcopy(authorization)
                change(altered)
                with patch.object(subject, "_parse", return_value=altered):
                    with self.assertRaises(ValueError):
                        subject._recovery_authority(plan)

    def test_effective_window_requires_later_exact_authorization_and_preserves_history(self):
        plan = self.plan()
        authorization = {"kind": "alignlab-explicit-deadline-replacement-authorization-v3",
            "source": "direct_user_message_in_this_chat",
            "user_message": "不管截止时间 你自己跑就行了 后面两天我都不在电脑前 你要自己保持项目进行",
            "execution_window": self.execution_window(), "recovery_authorization": plan["recovery_authorization"],
            "original_started_at_epoch": subject.resume.STARTED, "original_deadline_epoch": subject.resume.DEADLINE,
            "original_wall_seconds": 259200, "guard_policy_unchanged": True}
        with patch.object(subject, "_pin_record", return_value="budget"), patch.object(subject, "_parse", return_value=authorization):
            self.assertEqual(subject.validate_execution_window(plan), self.execution_window())
            self.assertEqual(plan["_original"]["continuation"]["deadline_epoch"], subject.resume.DEADLINE)
            for key, value in (("deadline_epoch", 1793006177.012), ("wall_seconds", 200000),
                               ("additional_training_or_automatic_retry", True),
                               ("numeric_hours_explicitly_supplied_by_user", True)):
                altered = copy.deepcopy(plan)
                altered["execution_window"][key] = value
                with self.assertRaises(ValueError):
                    subject.validate_execution_window(altered)
            authorization["user_message"] = "跑就行了"
            with self.assertRaises(ValueError):
                subject.validate_execution_window(plan)

    def test_dispatch_cannot_repeat_any_inherited_branch(self):
        for stage, seed in subject.SEQUENCE[:4]:
            with self.assertRaisesRegex(ValueError, "Only remaining"):
                subject._build_request(self.plan(), Path("plan"), "a" * 64, {}, 2643.9, stage, seed, {})
        self.assertEqual(subject.REMAINING_SEQUENCE, (("dpo", 44), ("ppo", 44)))

    def test_dpo44_reuses_exact_cache_and_original_workload(self):
        plan = self.plan()
        before = copy.deepcopy(plan)
        cache = {"path": "accepted-cache", "manifest_sha256": "a" * 64, "identity_sha256": "b" * 64}
        with patch("alignment_lab.throughput_fresh_v3.make_request", side_effect=lambda inner, plan, pin: inner):
            inner, script = subject._build_request(plan, Path("plan"), "c" * 64, {1: {"pid": 1}}, 2700, "dpo", 44, cache)
        self.assertEqual(script, "run_alignlab_throughput_fresh_v3.py")
        self.assertEqual(inner["reference_cache"], {"mode": "reuse", **cache})
        self.assertEqual(inner["workload"]["optimizer_updates"], 822)
        self.assertEqual(inner["workload"]["rows"], 13139)
        self.assertEqual(inner["source_sha256"], plan["_original"]["source_sha256"])
        self.assertEqual(inner["deadline_epoch"], subject.resume.DEADLINE)
        self.assertEqual(inner["process_wait_spent_seconds"], 2700)
        self.assertEqual(plan, before)

    def test_ppo44_preserves_all_registered_real_prompt_and_update_denominators(self):
        with patch("alignment_lab.throughput_fresh_v3.make_request", side_effect=lambda inner, plan, pin: inner):
            inner, _ = subject._build_request(self.plan(), Path("plan"), "c" * 64, {}, 2700, "ppo", 44, {})
        self.assertEqual(inner["workload"]["optimizer_updates"], 2048)
        self.assertEqual(inner["workload"]["rows"], 1024)
        self.assertEqual(inner["workload"]["selection_seed"], 42)
        self.assertEqual(set(inner["parents"]), {"sft", "reward"})
        self.assertIsNone(inner["reference_cache"])

    def test_changed_known_identity_or_same_instance_still_alive_rejected(self):
        observed = {"pid": 20, "parent_pid": 10, "name": "python.exe", "created_at_utc": "2026-10-24T00:00:00.123456Z", "command_line": "python pinned"}
        original = {**observed, "created_at_utc": "2026-10-24T00:00:00.1234560Z"}
        record = {"known_processes": [original]}
        progress = {"process_guard": {"owned_identities": [observed]}}
        self.assertEqual(subject._known_for_stage(record, progress), {20: observed})
        with self.assertRaises(ValueError):
            subject._known_for_stage({"known_processes": [{**original, "command_line": "changed"}]}, progress)
        plan = {"_prior_campaign": {"supervisor_processes": [original], "stages": []}}
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            subject._prior_processes_gone(plan, [{**observed, "command_line": "changed"}])
        subject._prior_processes_gone(plan, [{**observed, "created_at_utc": "2026-10-24T01:00:00Z"}])

    def test_current_source_cpu_proof_rejects_missing_new_tests_cuda_and_changed_hashes(self):
        sources = {"tests_throughput_v1/test_stage_v1.py": "a", "tests_throughput_v2/test_stage_v2.py": "b",
                   "tests/test_throughput_supervisor_v3.py": "c", "tests/test_ppo43_supplemental_v3.py": "d"}
        proof = {"kind": "alignlab-throughput-cpu-validation-v3", "status": "passed", "device": "cpu",
            "cuda_initialized": False, "tests_run": 12, "errors": 0, "failures": 0, "skipped": 0,
            "source_before": sources, "source_after": sources,
            "modules": [p[:-3].replace("/", ".") for p in sources]}
        with patch.object(subject, "_pin_record", return_value="proof"):
            with patch.object(subject, "_parse", return_value=proof):
                subject._cpu_proof({"source_sha256": sources, "cpu_report": {}})
            for altered in ({**proof, "modules": proof["modules"][:-1]}, {**proof, "cuda_initialized": True},
                            {**proof, "source_after": {}}, {**proof, "kind": "alignlab-throughput-cpu-validation-v1"}):
                with patch.object(subject, "_parse", return_value=altered):
                    with self.assertRaises(ValueError):
                        subject._cpu_proof({"source_sha256": sources, "cpu_report": {}})

    def test_own_identity_requires_v3_cli_and_exact_plan_hash(self):
        path = subject.ROOT / "configs/v3.json"
        digest = "d" * 64
        own = {"pid": os.getpid(), "parent_pid": 1, "created_at_utc": "2026-10-24T00:00:00Z",
            "command_line": f'python scripts/run_alignlab_throughput_campaign_v3.py --plan "{path}" --plan-sha256 {digest} --execute'}
        with patch.object(subject, "sha", return_value=digest):
            self.assertEqual(subject._own_processes([own], path), {own["pid"]: own})
            for cmd in (own["command_line"].replace("campaign_v3", "campaign_v2"), own["command_line"].replace(digest, "e" * 64)):
                with self.assertRaises(RuntimeError):
                    subject._own_processes([{**own, "command_line": cmd}], path)

    def test_guard_functions_and_original_sequence_remain_frozen(self):
        self.assertIs(subject.process_guard, subject.prior.process_guard)
        self.assertIs(subject._peer_decision, subject.legacy._peer_decision)
        self.assertIs(subject._waitable_peer_decision, subject.legacy._waitable_peer_decision)
        self.assertEqual(subject.SEQUENCE, subject.legacy.SEQUENCE)
        self.assertTrue(subject.prior.NEW_SOURCES <= subject.NEW_SOURCES)

    def test_full_content_hash_rejects_same_size_mtime_plan_source_and_receipt_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / name for name in ("plan.json", "source.py", "receipt.json", "state.pt")]
            for path in files:
                path.write_bytes(b"original-content")
            live = subject.legacy._LiveIdentity({}, trees=[root])
            proof = subject._ContentIdentity(live)
            original_hashes = copy.deepcopy(proof.sha256)
            # Even if all metadata and cheap checks appear unchanged, full byte
            # hashes independently reject changes. No baseline can be refreshed.
            with patch.object(live, "check", return_value=True):
                for path in files:
                    info = path.stat()
                    path.write_bytes(b"changed!-content")
                    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
                    self.assertEqual(path.stat().st_size, info.st_size)
                    self.assertEqual(path.stat().st_mtime_ns, info.st_mtime_ns)
                    with self.assertRaisesRegex(ValueError, "input content changed"):
                        proof.check()
                    path.write_bytes(b"original-content")
                    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
                self.assertTrue(proof.check())
            self.assertEqual(proof.sha256, original_hashes)

    def test_full_content_inventory_rejects_added_or_removed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "state.pt"
            original.write_bytes(b"state")
            live = subject.legacy._LiveIdentity({}, trees=[root])
            proof = subject._ContentIdentity(live)
            with patch.object(live, "check", return_value=True):
                extra = root / "unexpected.pt"
                extra.write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    proof.check()
                extra.unlink()
                original.unlink()
                with self.assertRaisesRegex(ValueError, "inventory changed"):
                    proof.check()

    def test_full_content_inventory_closes_transitive_report_evidence_and_proposal(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof_file, proposal = root / "cpu-proof.json", root / "proposal.md"
            proof_file.write_text("{}")
            proposal.write_text("concrete proposal")
            report = root / "report.json"
            report.write_text(json.dumps({"evidence_pins": {"proof": {
                "path": proof_file.name, "sha256": subject.sha(proof_file)}},
                "proposal": proposal.name, "proposal_sha256": subject.sha(proposal)}))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"validation_report": {"path": str(report), "sha256": subject.sha(report)}}))
            live = subject.legacy._LiveIdentity({str(plan): subject.sha(plan)})
            with patch.object(subject, "ROOT", root):
                subject._add_document_dependencies(live)
            content = subject._ContentIdentity(live)
            self.assertEqual(set(content.sha256), {str(p) for p in (proof_file, proposal, report, plan)})
            with patch.object(live, "check", return_value=True):
                proof_file.write_text("[]")
                with self.assertRaisesRegex(ValueError, "content changed"):
                    content.check()

    def fake_run(self, *, fail=False, expired=False):
        effective_window = self.execution_window()
        class FakeSystem:
            def __init__(self):
                self.t = effective_window["deadline_epoch"] if expired else effective_window["started_at_epoch"] + 1000
                self.spawned = []
                self.queries = 0
                self.child = None
            def now(self): return self.t
            def sleep(self, seconds): self.t += seconds
            def resources(self): return {"blockers": []}
            def processes(self):
                if self.child is None: return []
                self.queries += 1
                return [self.child] if self.queries == 1 else []
            def spawn(self, args, stdout, stderr):
                self.spawned.append(args)
                self.queries = 0
                self.child = {"pid": 100 + len(self.spawned), "parent_pid": 1, "name": "python.exe",
                    "created_at_utc": datetime.fromtimestamp(self.t, timezone.utc).isoformat(), "command_line": " ".join(args)}
                handle = MagicMock(pid=self.child["pid"])
                handle.poll.side_effect = lambda: None if self.queries == 1 else 0
                return handle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reports").mkdir()
            plan = self.plan()
            plan["output"] = str(root / "runs/new")
            (root / "runs").mkdir()
            cache = {"path": "cache", "manifest_sha256": "a" * 64, "identity_sha256": "b" * 64}
            plan["_inherited_stages"] = [{"name": f"{s}{n}", "status": "accepted", "reference_cache": cache}
                                          for s, n in subject.SEQUENCE[:4]]
            system = FakeSystem()
            guard = MagicMock()
            guard.check.return_value = {"total_waited_seconds": 2643.9, "admission_allowed": True}
            sentinel = MagicMock()
            sentinel.record.return_value = {"fixture": True}
            content = MagicMock()
            content.record.return_value = {"full_content_fixture": True}
            def verified(output, request, known, code):
                if fail: raise ValueError("intentional proof failure")
                output.mkdir()
                (output / "progress.json").write_text("{}")
                return {"artifact": {}, "reference_cache": cache, "process_guard": {"total_waited_seconds": 2643.9}}
            with patch.object(subject, "ROOT", root), patch.object(subject, "validate_plan", return_value=plan) as validate, \
                patch.object(subject, "_metadata_plan", return_value=plan), \
                patch.object(subject, "_ContentIdentity", return_value=content), \
                patch.object(subject, "_checked_prefix"), patch.object(subject, "_pin"), \
                patch.object(subject._LiveIdentity, "from_plan", return_value=sentinel), \
                patch.object(subject, "_parents_gone"), patch.object(subject, "_prior_processes_gone"), \
                patch.object(subject, "_own_processes", return_value={1: {"pid": 1}}), \
                patch.object(subject.process_guard, "strict_admission", return_value={"caller_admission_allowed": True}), \
                patch.object(subject.process_guard, "GuardSession", return_value=guard), \
                patch.object(subject, "_peer_decision", return_value={"caller_admission_allowed": True}), \
                patch.object(subject, "_command", return_value=True), \
                patch.object(subject, "verify_snapshot", return_value={"repo_id": subject.REPO_ID, "revision": subject.REVISION}), \
                patch.object(subject, "_build_request", side_effect=lambda *args: ({"kind": "fixture"}, "run_alignlab_throughput_fresh_v3.py")), \
                patch.object(subject, "_verify_stage", side_effect=verified):
                (root / "plan.json").write_text(__import__("json").dumps(plan["_authorized"]))
                if fail:
                    with self.assertRaisesRegex(ValueError, "intentional"):
                        subject.run_alignment(root / "plan.json", "c" * 64, execute=True, system=system)
                    self.assertEqual(len(system.spawned), 1)
                else:
                    state = subject.run_alignment(root / "plan.json", "c" * 64, execute=True, system=system)
                    if expired:
                        self.assertEqual(state["status"], "budget_exhausted")
                        self.assertEqual(system.spawned, [])
                    else:
                        self.assertEqual(state["status"], "primary_alignment_passed")
                        self.assertTrue(state["primary_alignment_verified"])
                        self.assertEqual([s["name"] for s in state["stages"]], ["dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44"])
                        self.assertEqual(len(system.spawned), 2)
                        self.assertEqual(state["started_at_epoch"], effective_window["started_at_epoch"])
                        self.assertEqual(state["deadline_epoch"], effective_window["deadline_epoch"])
                        self.assertEqual(state["original_started_at_epoch"], subject.resume.STARTED)
                        self.assertEqual(state["original_deadline_epoch"], subject.resume.DEADLINE)
                        self.assertEqual(state["total_process_wait_seconds"], 2643.9)
                self.assertEqual(validate.call_count, 1)
                if not fail and not expired:
                    self.assertGreaterEqual(content.check.call_count, 6)

    def test_fake_system_only_dispatches_two_remaining_children(self): self.fake_run()
    def test_fake_system_failure_does_not_retry_or_dispatch_next(self): self.fake_run(fail=True)
    def test_fake_system_expired_deadline_never_dispatches(self): self.fake_run(expired=True)


if __name__ == "__main__":
    unittest.main()
