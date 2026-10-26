"""Tiny actual CPU tensor/campaign fixtures, never real model-quality evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_acceptance as audit
from alignment_lab import parent_acceptance as parents
from tests_v2 import test_alignment_supervisor as af
from tests_v2 import test_parent_acceptance as pf


def write(path, value):
    Path(path).write_bytes(audit._canonical(value))


class TelemetryInventoryTests(unittest.TestCase):
    """Inventory wiring fixtures; numerical/verifier correctness is tested separately."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stage = Path(self.temporary.name)
        directory = self.stage / "telemetry"
        (directory / "checkpoints").mkdir(parents=True)
        from alignment_lab.training_telemetry import PROTOCOL, REVISION
        self.manifest = {"kind": PROTOCOL, "status": "complete", "rollouts": [],
                         "checkpoint_bindings": [{}], "costs": {"fixture_only": True}, "files": {}}
        pins = {}
        for name in ("rollout-0.json", "checkpoints/sidecar.json", "checkpoints/binding.json"):
            path = directory / name
            write(path, {"fixture": name})
            item = audit._file(path)
            self.manifest["files"][name] = {k: item[k] for k in ("sha256", "size_bytes")}
            pins[name] = {k: item[k] for k in ("path", "sha256")}
        write(directory / "manifest.json", self.manifest)
        self.progress = {
            "identity": {"config": {"stage": "ppo", "execution_revision": REVISION,
                                     "telemetry_protocol": PROTOCOL}},
            "telemetry": {k: audit._file(directory / "manifest.json")[k] for k in ("path", "sha256")},
            "completion": {"fixture_only": True},
            "checkpoints": [{"telemetry_sidecar": pins["checkpoints/sidecar.json"],
                              "telemetry_binding": pins["checkpoints/binding.json"]}]}

    def test_records_manifest_all_payloads_and_checkpoint_links(self):
        evidence = {}
        with patch("alignment_lab.training_telemetry.verify_training_telemetry",
                   return_value=self.manifest) as verifier:
            result = audit._telemetry_files(evidence, self.stage, self.progress)
        self.assertEqual(len(evidence), 4)
        self.assertEqual(result["checkpoint_binding_count"], 1)
        self.assertEqual(verifier.call_args.kwargs["expected_completion"], self.progress["completion"])
        self.assertEqual(verifier.call_args.kwargs["checkpoints"], self.progress["checkpoints"])

    def test_inventory_cannot_drop_checkpoint_sidecar_or_accept_changed_payload(self):
        with patch("alignment_lab.training_telemetry.verify_training_telemetry",
                   return_value=self.manifest):
            del self.manifest["files"]["checkpoints/sidecar.json"]
            with self.assertRaisesRegex(ValueError, "inventory"):
                audit._telemetry_files({}, self.stage, self.progress)
            (self.stage / "telemetry/rollout-0.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit._telemetry_files({}, self.stage, self.progress)

    def test_branch_manifest_cannot_be_substituted_by_another_directory(self):
        other = self.stage / "elsewhere"
        other.mkdir()
        write(other / "manifest.json", self.manifest)
        self.progress["telemetry"] = {k: audit._file(other / "manifest.json")[k] for k in ("path", "sha256")}
        with self.assertRaisesRegex(ValueError, "own stage"):
            audit._telemetry_files({}, self.stage, self.progress)


class EvaluationAcceptanceTests(unittest.TestCase):
    def setUp(self):
        # The process identity must already be complete when admission and
        # child requests record it. Never repair historical commands later.
        original_processes = af.ParentFakeSystem.processes
        def processes_with_plan_hash(system):
            rows = original_processes(system)
            for row in rows:
                if row["pid"] in (os.getpid(), 91):
                    row["command_line"] += " --plan-sha256 " + system.test.plan_sha
            return rows
        process_patch = patch.object(af.ParentFakeSystem, "processes", processes_with_plan_hash)
        process_patch.start()
        self.addCleanup(process_patch.stop)
        fixture = af.AlignmentSupervisorTests("test_plan_only_never_creates_output_or_process")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        self.current = []
        self.system = SimpleNamespace(processes=lambda: copy.deepcopy(self.current))
        for module in (audit, parents):
            change = patch.object(module, "ROOT", self.root)
            change.start()
            self.addCleanup(change.stop)
        for name in set(audit.SOURCES) | set(parents.AUDITOR_SOURCES):
            path = self.root / name
            if not path.exists():
                path.write_text("# CPU audit fixture only\n")
        change = patch.object(parents, "verify_snapshot",
                              return_value={"repo_id": "Qwen/Qwen3-1.7B", "revision": "fixture"})
        change.start()
        self.addCleanup(change.stop)
        def verify_data(plan):
            path = Path(plan["audit_dir"]) / "manifest.json"
            value = json.loads(path.read_text())
            for name, digest in value["artifacts"].items():
                parents._pin(path.parent / name, digest)
            return {"fixture_only": True, "full_fixture_data_hashes": True}
        change = patch.object(parents, "_verify_data", side_effect=verify_data)
        change.start()
        self.addCleanup(change.stop)
        parent = fixture.parent
        parent_campaign_path = parent.output / "campaign.json"
        campaign = json.loads(parent_campaign_path.read_text())
        helper = SimpleNamespace(parent=parent, campaign=campaign, campaign_path=parent_campaign_path)
        pf.ParentAcceptanceTests.rewrite_real_payloads(helper)
        initial_path = self.root / "reports/actual-parent-audit.json"
        initial = parents.audit_parents(parent.path, parent.plan_sha, output=initial_path, system=self.system)
        self.assertEqual(initial["status"], parents.ACCEPTED, initial.get("error"))
        fixture.plan["parent_acceptance_path"] = str(initial_path)
        fixture.plan["parent_acceptance_sha256"] = af.ss.sha(initial_path)
        fixture.plan["parents"] = {name: {k: row[k] for k in ("artifact_dir", "manifest_sha256")}
                                   for name, row in initial["stages"].items()}
        fixture.save_plan()
        # Exercise both actual cache-record and optimizer checkpoint receipts,
        # with a single inherited four-second campaign ledger. Waiting uses
        # the fixture clock; no real model, sleep or GPU is involved.
        fixture.guard_wait_branches.add(("ppo", 42))
        guard_module = af.process_guard
        original_session = guard_module.GuardSession
        cache_output = fixture.output / "dpo42"
        def session(*args, **kwargs):
            value = original_session(*args, **kwargs)
            identities = kwargs.get("owned_identities", ())
            is_cache_child = identities and "dpo42.request.json" in identities[0]["command_line"]
            if not is_cache_child:
                return value
            original_check, held = value.check, [False]
            def check(*, boundary, optimizer_updates, save_boundary):
                if boundary != "cache_record:0" or held[0]:
                    return original_check(boundary=boundary, optimizer_updates=optimizer_updates, save_boundary=save_boundary)
                held[0] = True
                reader, remaining = value.reader, [2]
                def transient():
                    rows = list(reader())
                    if remaining[0]:
                        remaining[0] -= 1
                        rows.append({"pid": 700002, "parent_pid": 0,
                            "created_at_utc": af.parent_fixtures.stamp(fixture.system.child_start),
                            "command_line": "python -m pip install --target .runtime Pillow"})
                    return rows
                def cache_receipt():
                    journal = cache_output / "reference-cache-records.jsonl"
                    path = cache_output / "cache-safe-boundary-fixture-before-reference.json"
                    write(path, {"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                        "optimizer_updates": 0, "cache_journal": str(journal),
                        "committed_record": json.loads(journal.read_text().splitlines()[0]),
                        "identity": json.loads((cache_output / "identity.json").read_text())})
                    return {"complete": True, "optimizer_updates": 0, "kind": "cache_record_boundary",
                            "path": str(path), "sha256": af.ss.sha(path)}
                value.reader = transient
                try:
                    return original_check(boundary=boundary, optimizer_updates=optimizer_updates, save_boundary=cache_receipt)
                finally:
                    value.reader = reader
            value.check = check
            return value
        with patch.object(guard_module, "GuardSession", side_effect=session):
            fixture.run_campaign()
        self.campaign_path = fixture.output / "campaign.json"
        self.campaign = json.loads(self.campaign_path.read_text())
        for process in self.campaign["supervisor_processes"]:
            self.assertIn(" --plan-sha256 " + fixture.plan_sha, process["command_line"])
        self.assertEqual(self.campaign["total_process_wait_seconds"], 4.0)
        self.assertEqual([row["request"]["process_wait_spent_seconds"] for row in self.campaign["stages"]],
                         [0.0, 2.0, 4.0, 4.0, 4.0, 4.0])
        for record in self.campaign["stages"]:
            self.assertEqual(record["request"]["supervisor_processes"], self.campaign["supervisor_processes"])

    def run_audit(self, name="evaluation-model-audit"):
        parent = self.fixture.parent
        return audit.audit_evaluation_models(parent.path, parent.plan_sha, self.fixture.path,
            self.fixture.plan_sha, output_dir=self.root / "reports" / name, system=self.system)

    def assert_no_acceptance(self, result, status=None):
        if status:
            self.assertEqual(result["status"], status, result.get("error"))
        self.assertFalse(result["models_verified"])
        self.assertIsNone(result["accepted_models"])
        self.assertFalse((self.root / "reports/evaluation-model-audit/accepted-models.json").exists())
        self.assertFalse((self.root / "reports/evaluation-model-audit/roster.json").exists())

    def rewrite_progress(self, name, change):
        record = next(s for s in self.campaign["stages"] if s["name"] == name)
        path = Path(record["progress_path"])
        value = json.loads(path.read_text())
        change(value)
        write(path, value)
        record["progress_sha256"] = af.ss.sha(path)
        record["artifact"] = value["artifact"]
        write(self.campaign_path, self.campaign)

    def test_complete_actual_payload_audit_roster_and_consumers(self):
        result = self.run_audit()
        self.assertEqual(result["status"], "evaluation_models_accepted", result.get("error"))
        self.assertFalse(result["cuda_initialized"])
        roster = audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"],
                                                system=self.system)
        self.assertEqual(set(roster["models"]), set(audit.MODELS))
        accepted = json.loads(Path(result["accepted_models"]["path"]).read_text())
        self.assertFalse(accepted["research_complete"])
        self.assertTrue(all(s["final_checkpoint_export_exact"] for s in accepted["alignment_stages"].values()))
        self.assertTrue(any(p.endswith("state.pt") for p in accepted["evidence_files"]))
        from alignment_lab.evaluation_execution import _roster
        from scripts.score_alignlab_evaluation import _verify_roster
        bound = {"base": {"path": str(Path(roster["model_snapshot"]) / "snapshot-manifest.json"),
                          "sha256": roster["model_manifest_sha256"]}}
        bound.update({name: {"path": str(Path(r["artifact_dir"]) / "manifest.json"),
                            "sha256": r["manifest_sha256"]}
                      for name, r in roster["models"].items() if name != "base"})
        protocol = SimpleNamespace(identity={"model_manifests": bound})
        _roster(roster, protocol)
        _verify_roster(roster, protocol)

    def test_guard_request_fields_and_campaign_wait_ledger_cannot_be_rewritten(self):
        accepted = self.run_audit("guard-dispatch-baseline")
        self.assertTrue(accepted["models_verified"], accepted.get("error"))
        original_campaign = copy.deepcopy(self.campaign)
        path = self.fixture.output / "dpo42.request.json"
        original_bytes = path.read_bytes()
        changes = (
            ("allowed_background_processes", lambda value: value.update(allowed_background_processes=[{
                "pid": 799999, "parent_pid": 0, "created_at_utc": "2026-10-21T00:00:00.0000000Z",
                "command_line": "python unrelated.py"}])),
            ("process_wait_policy", lambda value: value["process_wait_policy"].update(max_wait_seconds=301)),
            ("supervisor_processes", lambda value: value["supervisor_processes"][0].update(parent_pid=999)),
            ("process_wait_spent_seconds", lambda value: value.update(process_wait_spent_seconds=0.5)),
        )
        for field, change in changes:
            with self.subTest(field=field):
                self.campaign = copy.deepcopy(original_campaign)
                value = json.loads(original_bytes)
                change(value)
                write(path, value)
                # Even coherent outer hashes cannot authorize a changed
                # request, policy, actual process ancestry or waiting ledger.
                self.campaign["stages"][0].update(request=value, request_sha256=af.ss.sha(path))
                write(self.campaign_path, self.campaign)
                result = self.run_audit("changed-" + field)
                self.assert_no_acceptance(result, "failed")
                path.write_bytes(original_bytes)
                self.campaign = copy.deepcopy(original_campaign)
                write(self.campaign_path, self.campaign)
        self.campaign["total_process_wait_seconds"] = 3.0
        write(self.campaign_path, self.campaign)
        self.assert_no_acceptance(self.run_audit("reset-campaign-wait"), "failed")
        self.campaign = copy.deepcopy(original_campaign)
        self.campaign["supervisor_processes"][0]["command_line"] += " --posthoc-identity-change"
        write(self.campaign_path, self.campaign)
        self.assert_no_acceptance(self.run_audit("posthoc-supervisor-identity"), "failed")

    def test_guard_journals_and_actual_wait_receipts_are_pinned_against_change_or_deletion(self):
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        accepted = json.loads(Path(result["accepted_models"]["path"]).read_text())
        required = {self.fixture.output / "supervisor-process-guard.jsonl", self.fixture.output / "plan.snapshot.json"}
        wait_receipts = []
        for record in self.campaign["stages"]:
            output = self.fixture.output / record["name"]
            log = output / "process-guard.jsonl"
            required.add(log)
            if record["stage"] == "dpo":
                required.add(output / "reference-cache-records.jsonl")
            for event in (json.loads(line) for line in log.read_text().splitlines()):
                if event["event"] == "wait_start":
                    wait_receipts.append(event["saved_boundary"])
                    required.add(Path(event["saved_boundary"]["path"]))
        self.assertEqual({r["kind"] for r in wait_receipts}, {"cache_record_boundary", "optimizer_checkpoint"})
        self.assertTrue({str(path.resolve()) for path in required} <= set(accepted["evidence_files"]))
        chosen = {self.fixture.output / "supervisor-process-guard.jsonl",
                  self.fixture.output / "dpo42/process-guard.jsonl",
                  self.fixture.output / "dpo42/reference-cache-records.jsonl",
                  *(Path(row["path"]) for row in wait_receipts)}
        for path in chosen:
            before = path.read_bytes()
            for operation in ("change", "delete"):
                with self.subTest(path=str(path), operation=operation):
                    try:
                        if operation == "change":
                            path.write_bytes(before + b"tampered")
                        else:
                            path.unlink()
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)
                    finally:
                        path.write_bytes(before)

    def test_incomplete_parents_do_not_publish_roster(self):
        path = self.fixture.parent.output / "campaign.json"
        value = json.loads(path.read_text())
        value.update(status="running", parents_verified=False)
        write(path, value)
        self.assert_no_acceptance(self.run_audit(), "incomplete")

    def test_active_alignment_process_and_unknown_identity_refused(self):
        self.current = [self.campaign["stages"][-1]["known_processes"][-1]]
        self.assert_no_acceptance(self.run_audit(), "incomplete")
        self.current = [{"pid": 123456, "created_at_utc": "2026-10-20T11:00:00+00:00",
                         "command_line": None}]
        result = self.run_audit("unknown-process-audit")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unknown", result["error"])

    def test_pid_reuse_is_distinct_but_same_creation_command_drift_fails(self):
        process = copy.deepcopy(self.campaign["stages"][0]["known_processes"][0])
        process.update(created_at_utc="2030-01-01T00:00:00+00:00", command_line="python unrelated_api.py")
        self.current = [process]
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        process["created_at_utc"] = self.campaign["stages"][0]["known_processes"][0]["created_at_utc"]
        self.current = [process]
        result = self.run_audit("bad-process-audit")
        self.assertEqual(result["status"], "failed")
        self.assertIn("command", result["error"])

    def test_old_valid_checkpoint_cannot_be_relabelled_final(self):
        def change(progress):
            old, final = progress["checkpoints"][0], progress["checkpoints"][-1]
            for name in ("state.pt", "manifest.json"):
                Path(final["checkpoint_path"], name).write_bytes(Path(old["checkpoint_path"], name).read_bytes())
            final["manifest_sha256"] = old["manifest_sha256"]
            progress["latest_safe_checkpoint"] = final
        self.rewrite_progress("dpo42", change)
        result = self.run_audit()
        self.assert_no_acceptance(result, "failed")
        self.assertIn("payload cursor", result["error"])

    def test_modified_final_export_rejected_even_with_rehashed_manifest(self):
        def change(progress):
            import torch
            directory = Path(progress["artifact"]["artifact_dir"])
            manifest = json.loads((directory / "manifest.json").read_text())
            value = af.ss._cpu_payload(directory / "weights.pt", manifest["artifacts"]["weights.pt"]["sha256"])
            next(iter(value.values())).add_(1)
            torch.save(value, directory / "weights.pt")
            manifest["artifacts"]["weights.pt"] = {k: v for k, v in audit._file(directory / "weights.pt").items()
                                                   if k != "path"}
            write(directory / "manifest.json", manifest)
            progress["artifact"]["manifest_sha256"] = af.ss.sha(directory / "manifest.json")
        self.rewrite_progress("ppo44", change)
        result = self.run_audit()
        self.assert_no_acceptance(result, "failed")
        self.assertIn("Final exported weights", result["error"])

    def test_missing_update_nonzero_exit_and_stop_do_not_publish(self):
        self.campaign["stages"][-1]["exit_code"] = 1
        write(self.campaign_path, self.campaign)
        self.assert_no_acceptance(self.run_audit(), "failed")
        self.campaign["stages"][-1]["exit_code"] = 0
        write(self.campaign_path, self.campaign)
        self.rewrite_progress("ppo44", lambda p: p.update(actual_optimizer_updates=2047))
        result = self.run_audit("partial-audit")
        self.assertEqual(result["status"], "failed")
        (self.fixture.output / "STOP").write_text("cooperative stop")
        result = self.run_audit("stopped-audit")
        self.assertEqual(result["status"], "failed")
        self.assertIn("STOP", result["error"])

    def test_historical_prerequisite_restoration_and_live_payload_integrity(self):
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        priority = Path(self.fixture.parent.plan["prerequisites"][0]["path"])
        priority.unlink()  # Same semantics as a legitimately restored external STOP.
        audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)
        accepted = json.loads(Path(result["accepted_models"]["path"]).read_text())
        self.assertNotIn(str(priority), accepted["evidence_files"])
        payload = Path(accepted["models"]["ppo44"]["artifact_dir"]) / "weights.pt"
        original = payload.read_bytes()
        payload.write_bytes(original + b"changed")
        with self.assertRaises(ValueError):
            audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)
        payload.unlink()
        with self.assertRaises(ValueError):
            audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)

    def test_thinned_evidence_and_missing_auditor_source_rejected(self):
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        accepted_path = Path(result["accepted_models"]["path"])
        original = json.loads(accepted_path.read_text())
        roster_path = Path(result["roster"]["path"])
        roster = json.loads(roster_path.read_text())
        for target in ("state.pt", "evaluation_acceptance.py"):
            value = copy.deepcopy(original)
            key = next(p for p in value["evidence_files"] if p.endswith(target))
            del value["evidence_files"][key]
            write(accepted_path, value)
            for row in roster["models"].values():
                row["acceptance_sha256"] = af.ss.sha(accepted_path)
            write(roster_path, roster)
            with self.assertRaisesRegex(ValueError, "evidence"):
                audit.verify_evaluation_roster(roster_path, af.ss.sha(roster_path), system=self.system)

    def test_verifier_has_no_torch_import_and_output_is_exclusive(self):
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        code = """
import sys, json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from alignment_lab import evaluation_acceptance as module
assert 'torch' not in sys.modules
module.ROOT = Path(sys.argv[2])
value = module.verify_evaluation_roster(sys.argv[3], sys.argv[4], system=SimpleNamespace(processes=lambda: []))
assert 'torch' not in sys.modules
print(json.dumps({'models':len(value['models']), 'torch_imported':False}))
"""
        run = subprocess.run([sys.executable, "-I", "-B", "-c", code,
            str(Path(__file__).resolve().parents[1]), str(self.root),
            result["roster"]["path"], result["roster"]["sha256"]],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"models": 9, "torch_imported": False})
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_readonly_reuse_allows_other_projects_but_rejects_same_run_writers(self):
        result = self.run_audit()
        self.assertTrue(result["models_verified"], result.get("error"))
        (self.root / "reports/.alignlab-gpu.lock").write_text("another admitted task owns the GPU")
        self.current = [{"pid": 987654321, "parent_pid": 1,
                         "created_at_utc": "2030-01-01T00:00:00+00:00",
                         "command_line": "python -m windows.train --project C:/other-project"}]
        audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)
        self.current = [self.campaign["stages"][-1]["known_processes"][-1]]
        with self.assertRaisesRegex(ValueError, "still alive"):
            audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)
        self.current = [{"pid": 987654321, "parent_pid": 1,
                         "created_at_utc": "2030-01-01T00:00:00+00:00",
                         "command_line": "python writer.py --output " + str(self.fixture.output / "ppo42")}]
        with self.assertRaisesRegex(ValueError, "writer"):
            audit.verify_evaluation_roster(result["roster"]["path"], result["roster"]["sha256"], system=self.system)


if __name__ == "__main__":
    unittest.main()
