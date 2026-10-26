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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_reference_cache_recovery as cache_fixture
from test_reference_cache_recovery import sha, write
from alignment_lab import alignment_stage_r2 as stage
from alignment_lab import alignment_process_guard_r2 as guard
from alignment_lab import reference_cache_recovery as recovery
from alignment_lab import gpu_reference as gpu


OWN = {"pid": 1, "parent_pid": 10, "created_at_utc": "2026-10-22T05:00:00.1234567Z",
       "command_line": "python scripts/run_alignlab_alignment_stage_r2.py --request frozen --execute"}
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
        from alignment_lab import alignment_supervisor_r2 as supervisor
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
        write(report_path, false)
        progress["cache_prefix_recovery_report"]["sha256"] = sha(report_path)
        progress["reference_cache"]["migration_report_sha256"] = sha(report_path)
        with self.assertRaisesRegex(ValueError, "migration provenance"):
            guard.verify_guard_log(log, request=request, progress=progress)

    def test_imported_prefix_wait_suffix_score_complete_replay_and_readonly_reuse(self):
        self.run_connected_cache()

    def test_stop_during_imported_boundary_wait_prevents_any_suffix_score(self):
        self.run_connected_cache(stop_during_wait=True)


if __name__ == "__main__":
    unittest.main()
