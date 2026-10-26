"""CPU-only independent parent acceptance with actual safe tensor payloads."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import parent_acceptance as audit
from tests_v2 import test_study_supervisor as fixtures


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def tensor_bundle(directory, manifest, *, stage, cursor, final=False):
    import torch
    role = "sft" if stage == "sft" else "reward"
    update = cursor["optimizer_updates"]
    parameters = {f"base.layer.lora_A.{role}.weight": torch.tensor([float(update)])}
    if stage == "reward":
        parameters["scalar_heads.reward.weight"] = torch.tensor([float(update) / 2])
    if final:
        value = {("heads/reward/weight" if k.startswith("scalar_heads.") else
                  "adapter/" + k.replace(f".lora_A.{role}.", ".lora_A.")): v for k, v in parameters.items()}
        name = "weights.pt"
        manifest.update(source_role=role, heads=[] if stage == "sft" else ["reward"])
    else:
        value = {"parameters": parameters, "optimizer": {
            "param_groups": [{"params": list(range(len(parameters)))}],
            "state": {i: {"step": torch.tensor(float(update)), "exp_avg": torch.zeros_like(v),
                          "exp_avg_sq": torch.zeros_like(v)} for i, v in enumerate(parameters.values())}},
            "cursor": cursor, "rng": {"torch_cpu": torch.get_rng_state()}, "extra_state": None}
        name = "state.pt"
    torch.save(value, directory / name)
    manifest["artifacts"][name] = {"sha256": audit.sha(directory / name), "size_bytes": (directory / name).stat().st_size}
    manifest["identity_sha256"] = fixtures.hashlib.sha256(fixtures.ss._canonical(manifest["identity"])).hexdigest()
    write(directory / "manifest.json", manifest)
    return audit.sha(directory / "manifest.json")


class ParentAcceptanceTests(unittest.TestCase):
    def setUp(self):
        parent = fixtures.StudySupervisorTests("test_plan_only_no_output_lock_or_spawn")
        parent.setUp()
        self.addCleanup(parent.doCleanups)
        self.parent, self.root = parent, parent.root
        parent.run_campaign()
        self.campaign_path = parent.output / "campaign.json"
        self.campaign = json.loads(self.campaign_path.read_text())
        self.current = []
        self.system = SimpleNamespace(processes=lambda: copy.deepcopy(self.current))
        root_patch = patch.object(audit, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        auditor = self.root / "alignment_lab/parent_acceptance.py"
        auditor.write_text("# frozen audit fixture")
        source_patch = patch.object(audit, "AUDITOR_SOURCES", ("alignment_lab/parent_acceptance.py",))
        source_patch.start()
        self.addCleanup(source_patch.stop)
        snapshot_patch = patch.object(audit, "verify_snapshot", return_value={"repo_id": "Qwen/Qwen3-1.7B", "revision": "pin"})
        self.snapshot = snapshot_patch.start()
        self.addCleanup(snapshot_patch.stop)
        def verify_data(plan):
            manifest = json.loads((Path(plan["audit_dir"]) / "manifest.json").read_text())
            for name, digest in manifest["artifacts"].items():
                audit._pin(Path(plan["audit_dir"]) / name, digest)
            return {"fixture_only": True, "all_fixture_data_hashes_verified": True}
        data_patch = patch.object(audit, "_verify_data", side_effect=verify_data)
        data_patch.start()
        self.addCleanup(data_patch.stop)
        for process in self.campaign["supervisor_processes"]:
            process["command_line"] += " --plan-sha256 " + parent.plan_sha
        self.rewrite_real_payloads()

    def rewrite_real_payloads(self):
        sft_pin = None
        for record in self.campaign["stages"]:
            stage = record["stage"]
            directory = self.parent.output / stage
            progress = json.loads((directory / "progress.json").read_text())
            if stage == "reward":
                progress["identity"]["parents"]["sft"] = fixtures.file_record(
                    Path(sft_pin["artifact_dir"]) / "manifest.json")
                old_sha = record["request_sha256"]
                record["request"]["parents"]["sft"] = sft_pin
                progress["request"] = record["request"]
                request_path = self.parent.output / (stage + ".request.json")
                write(request_path, record["request"])
                write(directory / "request.json", record["request"])
                record["request_sha256"] = audit.sha(request_path)
                for process in record["known_processes"]:
                    process["command_line"] = process["command_line"].replace(old_sha, record["request_sha256"])
            count, updates = fixtures.ss.COUNTS[stage], fixtures.ss.UPDATES[stage]
            order = list(range(count))
            random.Random(42).shuffle(order)
            def cursor(update):
                offset = min(count, update * 16)
                return {"schema": 1, "config_sha256": "c" * 64, "data_sha256": "d" * 64,
                        "order": order, "offset": offset, "optimizer_updates": update,
                        "response_tokens": offset * (1 if stage == "sft" else 2),
                        "active_rollout": False, "completed_rollouts": 0}
            progress["cursor"] = cursor(updates)
            for receipt in progress["checkpoints"]:
                path = Path(receipt["checkpoint_path"])
                manifest = json.loads((path / "manifest.json").read_text())
                manifest["identity"] = progress["identity"]
                receipt["manifest_sha256"] = tensor_bundle(path, manifest, stage=stage,
                                                          cursor=cursor(receipt["optimizer_updates"]))
            progress["latest_safe_checkpoint"] = progress["checkpoints"][-1]
            artifact = progress["artifact"]
            path = Path(artifact["artifact_dir"])
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["identity"] = progress["identity"]
            artifact["manifest_sha256"] = tensor_bundle(path, manifest, stage=stage,
                                                       cursor=progress["cursor"], final=True)
            if stage == "sft":
                sft_pin = {k: artifact[k] for k in ("artifact_dir", "manifest_sha256")}
            required = {r["optimizer_updates"] for r in progress["checkpoints"]}
            metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            for item in metrics:
                if item.get("event") == "optimizer_boundary" and item["cursor"]["optimizer_updates"] in required:
                    item["cursor"] = cursor(item["cursor"]["optimizer_updates"])
            (directory / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in metrics))
            write(directory / "identity.json", progress["identity"])
            write(directory / "progress.json", progress)
            record.update(artifact=artifact, progress_sha256=audit.sha(directory / "progress.json"))
        write(self.campaign_path, self.campaign)

    def run_audit(self, name="audit.json"):
        return audit.audit_parents(self.parent.path, self.parent.plan_sha,
                                   output=self.root / "reports" / name, system=self.system)

    def update_progress(self, stage, mutate):
        record = next(s for s in self.campaign["stages"] if s["stage"] == stage)
        path = self.parent.output / stage / "progress.json"
        value = json.loads(path.read_text())
        mutate(value)
        write(path, value)
        record["progress_sha256"] = audit.sha(path)
        record["artifact"] = value["artifact"]
        write(self.campaign_path, self.campaign)

    def test_full_actual_payloads_and_repeat_readonly_acceptance(self):
        result = self.run_audit()
        self.assertEqual(result["status"], audit.ACCEPTED, result.get("error"))
        self.assertTrue(result["all_checks_passed"])
        self.assertTrue(result["processes_exited"])
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse(result["research_complete"])
        self.assertEqual((result["stages"]["sft"]["rows"], result["stages"]["reward"]["optimizer_updates"]), (4000, 822))
        self.assertTrue(result["stages"]["reward"]["payload_audit"]["checkpoints"][-1]["final_artifact_exact"])
        self.assertEqual(self.run_audit("audit-second.json")["status"], audit.ACCEPTED)

    def test_running_campaign_never_publishes_acceptance_or_hashes_model(self):
        self.campaign["status"], self.campaign["parents_verified"] = "running", False
        write(self.campaign_path, self.campaign)
        self.current = [self.campaign["supervisor_processes"][0]]
        result = self.run_audit()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["parents_verified"])
        self.assertIn("recorded_parent_process_still_alive", result["blockers"])
        self.snapshot.assert_not_called()

    def test_false_finished_status_with_real_process_alive_is_incomplete(self):
        self.current = [self.campaign["stages"][1]["known_processes"][1]]
        result = self.run_audit()
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["processes_exited"])

    def test_live_reused_pid_with_new_creation_is_not_the_historical_process(self):
        reused = copy.deepcopy(self.campaign["stages"][0]["known_processes"][0])
        reused["created_at_utc"] = "2030-01-01T00:00:00+00:00"
        reused["command_line"] = "python unrelated.py"
        self.current = [reused]
        result = self.run_audit()
        self.assertEqual(result["status"], audit.ACCEPTED, result.get("error"))
        self.assertIn("pid_reused_original_exited", [p["state"] for p in result["processes"]])

    def test_same_pid_creation_but_changed_command_is_not_attested_exited(self):
        reused = copy.deepcopy(self.campaign["stages"][0]["known_processes"][0])
        reused["command_line"] = "python altered.py"
        self.current = [reused]
        result = self.run_audit()
        self.assertEqual(result["status"], "failed")
        self.assertIn("different command", result["error"])

    def test_old_hash_valid_checkpoint_cannot_claim_final_cursor(self):
        def replace(progress):
            first, final = progress["checkpoints"][0], progress["checkpoints"][-1]
            for name in ("state.pt", "manifest.json"):
                Path(final["checkpoint_path"], name).write_bytes(Path(first["checkpoint_path"], name).read_bytes())
            final["manifest_sha256"] = first["manifest_sha256"]
            progress["latest_safe_checkpoint"] = final
        self.update_progress("sft", replace)
        result = self.run_audit()
        self.assertEqual(result["status"], "failed")
        self.assertIn("payload cursor", result["error"])

    def test_final_export_weights_must_equal_actual_final_checkpoint(self):
        def change(progress):
            import torch
            path = Path(progress["artifact"]["artifact_dir"])
            manifest = json.loads((path / "manifest.json").read_text())
            weights = audit._cpu_payload(path / "weights.pt", manifest["artifacts"]["weights.pt"]["sha256"])
            next(iter(weights.values())).add_(1)
            torch.save(weights, path / "weights.pt")
            manifest["artifacts"]["weights.pt"] = {"sha256": audit.sha(path / "weights.pt"),
                                                 "size_bytes": (path / "weights.pt").stat().st_size}
            write(path / "manifest.json", manifest)
            progress["artifact"]["manifest_sha256"] = audit.sha(path / "manifest.json")
        self.update_progress("reward", change)
        result = self.run_audit()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Final artifact weights differ", result["error"])

    def test_missing_real_updates_cannot_be_laundered_by_campaign_pass(self):
        self.update_progress("sft", lambda p: p.update(actual_optimizer_updates=249))
        result = self.run_audit()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["parents_verified"])

    def test_output_is_never_overwritten(self):
        path = self.root / "reports/audit.json"
        path.write_text("old evidence")
        with self.assertRaises(ValueError):
            self.run_audit()
        self.assertEqual(path.read_text(), "old evidence")


if __name__ == "__main__":
    unittest.main()
