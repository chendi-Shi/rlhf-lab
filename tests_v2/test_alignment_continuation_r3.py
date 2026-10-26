"""Actual tiny CPU prefix scoring + safe-boundary guard + cache reuse.

These checks exercise the new helper contracts, not production CUDA acceptance.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import io
import os
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_reference_cache_recovery as cache_fixture
from test_reference_cache_recovery import sha
from alignment_lab.alignment_supervisor_r3 import JsonWriter
import time

def write(path, value):
    return JsonWriter(time.time() + 120, lambda: False)(path, value, exclusive=True)
from alignment_lab import alignment_stage_r3 as stage
from alignment_lab import alignment_process_guard_r3 as guard
from alignment_lab import reference_cache_recovery_r3 as recovery
from alignment_lab import gpu_reference as gpu
from alignment_lab import alignment_supervisor_r3 as supervisor
from alignment_lab import windows_atomic_io
from tests_v2 import test_alignment_supervisor_r2 as legacy


OWN = {"pid": 1, "parent_pid": 10, "created_at_utc": "2026-10-22T05:00:00.1234567Z",
       "command_line": "python scripts/run_alignlab_alignment_stage_r3.py --request frozen --execute"}
UNKNOWN = {"pid": 2, "parent_pid": 20, "created_at_utc": "2026-10-22T05:00:01.1234567Z",
           "command_line": "python -m pip install --target .runtime Pillow"}


class AlignmentContinuationIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = cache_fixture.ReferenceCacheRecoveryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def run_connected_cache(self, *, stop_during_wait=False):
        fixture = self.fixture
        output = fixture.root / "new-stage"
        output.mkdir()
        journal = output / "reference-cache-records.jsonl"
        log = output / "process-guard.jsonl"
        original_bytes = {p: p.read_bytes() for p in fixture.old.iterdir()}
        now = [fixture.deadline - 290]
        started = now[0]
        state = {"count": 0, "unknown": False, "sleeps": 0, "stop": False, "scoring_calls": 0}
        parameters = [p.detach().clone() for p in fixture.backend._all_parameters()]
        identity = {"purpose": "tiny CPU boundary test; not formal artifact acceptance"}
        progress = {"started_at_epoch": started, "identity": identity, "checkpoints": [],
            "process_guard": {"owned_identities": [OWN], "total_waited_seconds": 300.0664498806}}
        request = {"stage": "dpo", "seed": 42, "output": str(output), "deadline_epoch": fixture.deadline,
            "reference_cache": {"mode": "create", "path": str(output / "reference-cache")},
            "continuation": {"cache_prefix": copy.deepcopy(fixture.spec)},
            "process_wait_policy": copy.deepcopy(guard.POLICY), "process_wait_spent_seconds": 300.0664498806,
            "allowed_background_processes": [], "supervisor_processes": [],
            "workload": {"optimizer_updates": 1}}
        write(output / "request.json", request)
        def emit(event):
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            if event["event"] == "result":
                progress["process_guard"].update(last=event, total_waited_seconds=event["total_waited_seconds"])
        def inventory():
            return [OWN, UNKNOWN] if state["unknown"] else [OWN]
        def sleep(seconds):
            # Imported records are already committed; no suffix score or model
            # mutation can occur while the synchronous guard owns this boundary.
            self.assertEqual(state["count"], fixture.spec["scored_sequences"])
            self.assertEqual(state["scoring_calls"], 0)
            self.assertEqual(len(journal.read_text().splitlines()), state["count"] + 1)
            self.assertTrue(all(torch.equal(p.detach(), q)
                for p, q in zip(fixture.backend._all_parameters(), parameters)))
            saved = list(output.glob("cache-safe-boundary-*.json"))
            self.assertEqual(len(saved), 1)
            receipt = json.loads(saved[0].read_text())
            self.assertEqual(receipt["committed_record"]["scored_sequences"], state["count"])
            now[0] += seconds
            state["sleeps"] += 1
            state["unknown"] = state["sleeps"] < 2
            state["stop"] = stop_during_wait
        session = guard.GuardSession(inventory, owned_identities=[OWN], deadline_epoch=fixture.deadline,
            policy=guard.POLICY, spent_seconds=300.0664498806, stop_requested=lambda: state["stop"],
            clock=lambda: now[0], sleep=sleep, on_event=emit)
        def receipt(path, kind, update=0):
            return {"complete": True, "optimizer_updates": update, "kind": kind,
                    "path": str(path.resolve()), "sha256": sha(path)}
        def check(name, update, save):
            result = session.check(boundary=name, optimizer_updates=update, save_boundary=save)
            if not result["admission_allowed"]:
                raise InterruptedError(result["reason"])
        check("before_cuda", 0, lambda: receipt(output / "request.json", "before_cuda"))
        chain = "0" * 64
        def boundary(**values):
            nonlocal chain
            payload = {"schema_version": 1, "previous_sha256": chain, **values}
            chain = stage.digest(payload)
            row = {**payload, "record_sha256": chain}
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            count, total = values["scored_sequences"], values["expected_unique_sequences"]
            state["count"] = count
            imported = fixture.spec["scored_sequences"]
            if count == imported:
                state["unknown"] = True
            if count == 0 or count == imported or count == total or (count > imported and count % 16 == 0):
                def save():
                    path = output / f"cache-safe-boundary-{count:06d}-{session.calls:06d}.json"
                    write(path, {"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                        "optimizer_updates": 0, "cache_journal": str(journal), "committed_record": row,
                        "identity": identity})
                    return receipt(path, "cache_record_boundary")
                check(f"cache_record:{count}", 0, save)
        def migration(report):
            path = output / "cache-prefix-migration.json"
            write(path, report)
            progress["cache_prefix_recovery_report"] = {"path": str(path), "sha256": sha(path)}
        original_score = gpu._sequence_scores
        def score(*args, **kwargs):
            self.assertFalse(state["unknown"])
            self.assertFalse(state["stop"])
            state["scoring_calls"] += 1
            return original_score(*args, **kwargs)
        with patch.object(gpu, "_sequence_scores", side_effect=score):
            if stop_during_wait:
                with self.assertRaisesRegex(InterruptedError, "stop_requested"):
                    recovery.create_bounded_cache_from_prefix(fixture.new_receipt, output / "reference-cache", fixture.pairs,
                        prefix_spec=fixture.spec, config=fixture.config, deadline_epoch=fixture.deadline,
                        clock=lambda: now[0], stop_requested=lambda: state["stop"],
                        record_boundary=boundary, migration_callback=migration)
                self.assertEqual(state["scoring_calls"], 0)
                self.assertFalse((output / "reference-cache").exists())
                self.assertEqual({p: p.read_bytes() for p in fixture.old.iterdir()}, original_bytes)
                return
            binding = recovery.create_bounded_cache_from_prefix(fixture.new_receipt, output / "reference-cache", fixture.pairs,
                prefix_spec=fixture.spec, config=fixture.config, deadline_epoch=fixture.deadline,
                clock=lambda: now[0], record_boundary=boundary, migration_callback=migration)
        self.assertEqual(state["scoring_calls"], fixture.total - fixture.spec["scored_sequences"])
        self.assertEqual(state["sleeps"], 2)
        pin = progress["cache_prefix_recovery_report"]
        progress["reference_cache"] = {"mode": "create", "path": str(binding.path),
            "manifest_sha256": binding.manifest_sha256, "identity_sha256": sha(binding.path / "identity.json"),
            "unique_sequences": len(binding._cache.entries), "migration_report_path": pin["path"],
            "migration_report_sha256": pin["sha256"], "imported_sequences": fixture.spec["scored_sequences"],
            "newly_scored_sequences": fixture.total - fixture.spec["scored_sequences"]}
        check("engine_origin", 0, lambda: self.fail("No remaining unknown process"))
        # Real tiny policy update follows cache construction and full clearance.
        backend, _, _, ids, attention, labels = fixture.parts
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        backend.backward("policy", optimizer,
            lambda view: -view.sequence_scores(ids, attention, labels).sums.mean())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        check("optimizer_complete", 1, lambda: self.fail("No remaining unknown process"))
        check("engine_final", 1, lambda: self.fail("No remaining unknown process"))
        accepted = guard.verify_guard_log(log, request=request, progress=progress)
        self.assertTrue(accepted["verified"])
        self.assertEqual(accepted["segment_waited_seconds"], 20)
        from alignment_lab import alignment_supervisor_r3 as supervisor
        with patch.object(supervisor, "ROOT", fixture.root):
            self.assertEqual(supervisor._verify_prefix_migration(output, request, progress,
                binding._cache, binding.identity), pin)
        self.assertEqual({p: p.read_bytes() for p in fixture.old.iterdir()}, original_bytes)
        # The same actual immutable reference can reuse the cache without a
        # scoring forward, after the policy changes independently.
        cache_pin = {"mode": "reuse", "path": str(binding.path),
            "manifest_sha256": binding.manifest_sha256, "identity_sha256": sha(binding.path / "identity.json")}
        with patch.object(gpu, "_sequence_scores", side_effect=AssertionError("Reuse cannot re-score")):
            reused = stage.bind_complete_cache(fixture.new_receipt, cache_pin, fixture.pairs,
                config=fixture.config, deadline_epoch=fixture.deadline)
        self.assertEqual(dict(reused._cache.entries), dict(binding._cache.entries))
        # Even coherently re-pinning a false imported-count report is rejected.
        report_path = Path(pin["path"])
        false = json.loads(report_path.read_text())
        false["imported_sequences"] -= 1
        JsonWriter(time.time() + 120, lambda: False)(report_path, false)
        progress["cache_prefix_recovery_report"]["sha256"] = sha(report_path)
        progress["reference_cache"]["migration_report_sha256"] = sha(report_path)
        with self.assertRaisesRegex(ValueError, "migration provenance"):
            guard.verify_guard_log(log, request=request, progress=progress)

    def test_imported_prefix_wait_suffix_score_complete_replay_and_readonly_reuse(self):
        self.run_connected_cache()

    def test_stop_during_imported_boundary_wait_prevents_any_suffix_score(self):
        self.run_connected_cache(stop_during_wait=True)


class LifecycleR3Tests(legacy.LifecycleTests):
    def setUp(self):
        alias = patch.object(legacy, "ss", supervisor)
        alias.start()
        self.addCleanup(alias.stop)
        super().setUp()
        self.plan["repair_attempt"] = {"fixture": "pinned separately by repair tests"}
        original = self.system.processes
        def current():
            rows = original()
            for row in rows:
                row["command_line"] = row["command_line"].replace("run_alignlab_alignment_r2.py", "run_alignlab_alignment_r3.py")
            return rows
        self.system.processes = current


class RepairIdentityTests(legacy.ContinuationTests):
    def setUp(self):
        # Build the original failure with its own frozen legacy fixture, then
        # an independently pinned R2 attempt; no production evidence is written.
        super().setUp()
        self.r3root = patch.object(supervisor, "ROOT", self.root)
        self.r3root.start()
        self.addCleanup(self.r3root.stop)
        r2out = self.root / "runs/r2"
        (r2out / "dpo42").mkdir(parents=True)
        self.r2path = self.root / "r2-plan.json"
        self.r2plan = {**self.plan, "kind": legacy.ss.KIND, "output": str(r2out),
            "process_wait_policy": copy.deepcopy(guard.POLICY)}
        legacy.write(self.r2path, self.r2plan)
        req = {"kind": legacy.ss.STAGE_KIND, "stage": "dpo", "seed": 42,
            "continuation": self.r2plan["continuation"], "deadline_epoch": 3100,
            "alignment_plan_path": str(self.r2path), "alignment_plan_sha256": legacy.pin(self.r2path)["sha256"],
            "reference_cache": {"mode": "create", "path": str(r2out / "dpo42/reference-cache")}}
        legacy.write(r2out / "dpo42.request.json", req)
        legacy.write(r2out / "dpo42/request.json", req)
        legacy.write(r2out / "dpo42/cache-prefix-migration.json", {
            "kind": "alignlab-reference-cache-prefix-migration-v1", "pins": self.r2plan["continuation"]["cache_prefix"],
            "new_optimizer_updates": 0, "formal_training_complete": False})
        self.r2progress = {"status": "failed", "request": req, "actual_optimizer_updates": 0,
            "latest_safe_checkpoint": None, "checkpoints": [],
            "cache_prefix_recovery_report": legacy.pin(r2out / "dpo42/cache-prefix-migration.json"),
            "cache_scoring_progress": {"scored_sequences": 1, "newly_scored_sequences": 0, "complete": False},
            "process_guard": {"total_waited_seconds": 300.0664498806}}
        legacy.write(r2out / "dpo42/progress.json", self.r2progress)
        self.r2campaign = {"kind": legacy.ss.KIND, "status": "failed", "started_at_epoch": 100,
            "deadline_epoch": 3100, "plan_sha256": legacy.pin(self.r2path)["sha256"],
            "total_process_wait_seconds": 300.0664498806,
            "stages": [{"name": "dpo42", "request": req, "exit_code": 1, "active_processes": []}]}
        legacy.write(r2out / "campaign.json", self.r2campaign)
        auth = {"kind": "explicit-user-primary-alignment-r3-repair-authorization-v1", "source": "direct_user_reply_in_this_chat",
            "allow_one_corrected_dispatch": True, "origin_plan_sha256": legacy.pin(self.r2path)["sha256"],
            "origin_campaign_sha256": legacy.pin(r2out / "campaign.json")["sha256"],
            "original_started_at_epoch": 100, "original_deadline_epoch": 3100}
        legacy.write(self.root / "repair-auth.json", auth)
        self.repair_plan = {**self.r2plan, "kind": supervisor.KIND, "output": str(self.root / "runs/r3"),
            "repair_attempt": {"prior_plan": legacy.pin(self.r2path), "prior_campaign": legacy.pin(r2out / "campaign.json"),
                "prior_stage_request": legacy.pin(r2out / "dpo42.request.json"),
                "prior_stage_progress": legacy.pin(r2out / "dpo42/progress.json"),
                "prior_migration": legacy.pin(r2out / "dpo42/cache-prefix-migration.json"),
                "authorization": legacy.pin(self.root / "repair-auth.json")}}

    def test_r2_import_progress_is_only_history_not_resume(self):
        result = supervisor.validate_repair_attempt(self.repair_plan)
        self.assertEqual(result["progress"]["actual_optimizer_updates"], 0)

    def test_new_scores_or_optimizer_updates_forbid_this_repair_protocol(self):
        path = Path(self.repair_plan["repair_attempt"]["prior_stage_progress"]["path"])
        self.r2progress["cache_scoring_progress"]["newly_scored_sequences"] = 1
        legacy.write(path, self.r2progress)
        self.repair_plan["repair_attempt"]["prior_stage_progress"] = legacy.pin(path)
        with self.assertRaises(ValueError):
            supervisor.validate_repair_attempt(self.repair_plan)

    def test_r2_source_change_or_new_deadline_rejected(self):
        self.repair_plan["source_sha256"] = {"old.py": "0" * 64}
        with self.assertRaises(ValueError):
            supervisor.validate_repair_attempt(self.repair_plan)


class AtomicWiringTests(unittest.TestCase):
    def test_retry_is_visible_and_terminal_flush_does_not_extend_training_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            now, stop, calls = [100.0], [False], [0]
            writer = supervisor.JsonWriter(105, lambda: stop[0], clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
            original = os.replace
            def transient(src, dst):
                calls[0] += 1
                if calls[0] < 3:
                    error = PermissionError("sharing fixture")
                    error.winerror = 5
                    raise error
                return original(src, dst)
            stderr = io.StringIO()
            path = Path(tmp) / "progress.json"
            with patch.object(windows_atomic_io.os, "replace", transient), patch("sys.stderr", stderr):
                receipt = writer(path, {"deadline_epoch": 105, "update": 0})
            self.assertEqual(receipt["attempts"], 3)
            self.assertEqual(stderr.getvalue().count('"event": "atomic_json_retry"'), 2)
            now[0], stop[0] = 106, True
            with self.assertRaises(InterruptedError):
                writer(path, {"must_not_commit": True})
            writer(path, {"deadline_epoch": 105, "status": "paused"}, terminal=True)
            self.assertEqual(json.loads(path.read_text())["deadline_epoch"], 105)

    def test_stage_runtime_uses_new_writer_not_imported_frozen_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            output = root / "runs/new/dpo42"
            (root / "plan.json").write_text("{}")
            request = {"stage": "dpo", "output": str(output), "deadline_epoch": time.time() + 60,
                "alignment_plan_path": str(root / "plan.json"), "allowed_background_processes": [],
                "alignment_plan_sha256": sha(root / "plan.json"), "source_sha256": {},
                "supervisor_processes": [], "process_wait_policy": guard.POLICY,
                "process_wait_spent_seconds": 0, "workload": {"scope": "cpu fixture"}}
            owned = [{**OWN, "pid": os.getpid()}]
            system = SimpleNamespace(processes=lambda: owned)
            with patch.object(stage, "ROOT", root), patch.object(stage, "validate_request", return_value=output), \
                    patch("alignment_lab.probe_dispatch._System", return_value=system), \
                    patch.object(guard, "child_identities", return_value=owned), \
                    patch("alignment_lab.gpu_probe.resources", return_value={"blockers": ["CPU fixture stops before CUDA"]}), \
                    patch("alignment_lab.capacity_stage._write", side_effect=AssertionError("old writer used")), \
                    patch("alignment_lab.probe_dispatch._status", side_effect=AssertionError("old writer used")), \
                    patch("alignment_lab.probe_dispatch._new", side_effect=AssertionError("old writer used")):
                with self.assertRaisesRegex(RuntimeError, "CUDA admission blocked"):
                    stage.run_alignment_stage(request, execute=True)
            result = json.loads((output / "progress.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["actual_optimizer_updates"], 0)
            self.assertFalse((root / "reports/.alignlab-gpu.lock").exists())
            self.assertFalse(torch.cuda.is_initialized())

    def test_guard_accepts_only_precise_r3_stage_entry(self):
        request = {"output": "C:/fixture/dpo42", "supervisor_processes": [
            {"pid": 10, "parent_pid": 1, "created_at_utc": "2026-10-22T05:00:00.0000000Z", "command_line": "python supervisor"}]}
        child = {"pid": 20, "parent_pid": 10, "created_at_utc": "2026-10-22T05:00:01.0000000Z",
            "command_line": "python scripts/run_alignlab_alignment_stage_r3.py --request C:/fixture/dpo42.request.json --execute"}
        self.assertEqual(guard.child_identities([child, *request["supervisor_processes"]], request, pid=20), [child])
        child["command_line"] = child["command_line"].replace("stage_r3.py", "stage_r2.py")
        with self.assertRaises(RuntimeError):
            guard.child_identities([child, *request["supervisor_processes"]], request, pid=20)


if __name__ == "__main__":
    unittest.main()



