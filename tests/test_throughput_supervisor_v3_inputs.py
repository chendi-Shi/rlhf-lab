"""Input closure retains historical evidence without treating it as live state."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import throughput_supervisor_v3_inputs as subject


class InputsTests(unittest.TestCase):
    def tearDown(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_original_dispatch_and_full_verifier_bytecode_local_namespace(self):
        for name in ("run_alignment", "validate_plan", "_metadata_plan"):
            clone, original = getattr(subject, name), getattr(subject.frozen, name)
            self.assertIs(clone.__code__, original.__code__)
            self.assertIsNot(clone.__globals__, original.__globals__)
        self.assertIs(subject.frozen.run_alignment.__globals__["_LiveIdentity"], subject.frozen._LiveIdentity)
        self.assertIs(subject.run_alignment.__globals__["_LiveIdentity"], subject._LiveIdentity)
        self.assertIs(subject.run_alignment.__globals__["_verify_stage"], subject.frozen._verify_stage)

    def test_new_sources_and_cpu_module_are_mandatory(self):
        plan = {"cpu_report": {}, "source_sha256": dict.fromkeys(subject.ADDED_SOURCES, "a" * 64)}
        with patch.object(subject.frozen, "_cpu_proof"), patch.object(subject.frozen, "_pin_record"), \
                patch.object(subject, "_parse", return_value={"modules": ["tests.test_throughput_supervisor_v3_inputs"]}):
            subject._cpu_proof(plan)
            for name in subject.ADDED_SOURCES:
                changed = copy.deepcopy(plan)
                del changed["source_sha256"][name]
                with self.assertRaises(ValueError): subject._cpu_proof(changed)
        with patch.object(subject.frozen, "_cpu_proof"), patch.object(subject.frozen, "_pin_record"), \
                patch.object(subject, "_parse", return_value={"modules": []}):
            with self.assertRaises(ValueError): subject._cpu_proof(plan)

    def test_exact_historical_pin_resolves_to_authenticated_leaf_and_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "project"
            root.mkdir()
            external = base / "other/STOP"
            snapshot = root / "saved.json"
            snapshot.write_text(json.dumps({"old_live_state": {"path": str(base / "unavailable"), "sha256": "a" * 64}}))
            document = root / "evidence.json"
            digest = subject.sha(snapshot)
            mapping = {(str(external), digest): snapshot}
            leaves = {str(snapshot): digest}
            document.write_text(json.dumps({"artifact": {"path": str(external), "sha256": digest}}))
            live = subject.frozen.legacy._LiveIdentity({str(document): subject.sha(document)})
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=(mapping, leaves)):
                subject._add_document_dependencies(live)
            self.assertEqual(set(live.files), {str(document), str(snapshot)})
            proof = subject._ContentIdentity(live)
            snapshot.write_text("{}")
            with self.assertRaises(ValueError): proof.check()
            document.write_text(json.dumps({"artifact": {"path": str(external), "sha256": "b" * 64}}))
            live = subject.frozen.legacy._LiveIdentity({str(document): subject.sha(document)})
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=(mapping, leaves)):
                with self.assertRaisesRegex(ValueError, "escaped"): subject._add_document_dependencies(live)

    def test_unknown_external_pin_is_not_whitelisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "evidence.json"
            document.write_text(json.dumps({"path": str(root.parent / "other.py"), "sha256": "a" * 64}))
            live = subject.frozen.legacy._LiveIdentity({str(document): subject.sha(document)})
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=({}, {})):
                with self.assertRaisesRegex(ValueError, "escaped"): subject._add_document_dependencies(live)

    def test_current_transitive_input_stays_fully_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report, plan = (root / name for name in ("code.py", "report.json", "plan.json"))
            code.write_text("original")
            report.write_text(json.dumps({"source": {"path": code.name, "sha256": subject.sha(code)}}))
            plan.write_text(json.dumps({"report": {"path": str(report), "sha256": subject.sha(report)}}))
            live = subject.frozen.legacy._LiveIdentity({str(plan): subject.sha(plan)})
            with patch.object(subject, "ROOT", root), patch.object(subject, "_historical_map", return_value=({}, {})):
                subject._add_document_dependencies(live)
            self.assertEqual(set(live.files), {str(code), str(report), str(plan)})
            proof = subject._ContentIdentity(live)
            code.write_text("modified")
            with self.assertRaises(ValueError): proof.check()

    def test_historical_verifier_requires_exact_document_pointer_hash_and_path(self):
        document = (subject.ROOT / subject.PRIORITY_REPORT).resolve()
        value = {"module_path": str((subject.ROOT.parent / "revla/windows/checkpointing.py").resolve()),
            "module_sha256": "6916f81a648d90fdff2dca8267f98d9f544c22654e1f57ce60e56580fd3899df",
            "training_state_loaded": False, "torch_imported": False}
        with patch.object(subject.frozen, "_pin") as pin:
            self.assertTrue(subject._historical_observation(document, ("revla", "verifier"), value))
            pin.assert_called_once_with(document, subject.PRIORITY_REPORT_SHA)
            self.assertFalse(subject._historical_observation(document, ("current", "verifier"), value))
            self.assertFalse(subject._historical_observation(document.parent / "unknown.json", ("revla", "verifier"), value))
            value["module_sha256"] = "a" * 64
            self.assertFalse(subject._historical_observation(document, ("revla", "verifier"), value))

    def test_exact_new_cli_path_required_even_with_same_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text("{}")
            command = f'python "{subject.ROOT / "scripts/throughput_inputs_v3/run_alignlab_throughput_campaign_v3.py"}" --plan "{plan}" --plan-sha256 {subject.sha(plan)} --execute'
            process = {"pid": 123, "parent_pid": 5, "created_at_utc": "2026-10-24T09:00:00.0000000Z", "command_line": command}
            with patch.object(subject.frozen.os, "getpid", return_value=123):
                self.assertEqual(subject._own_processes([process], plan), {123: process})
                process["command_line"] = command.replace("throughput_inputs_v3/", "").replace("throughput_inputs_v3\\", "")
                with self.assertRaises(RuntimeError): subject._own_processes([process], plan)


if __name__ == "__main__":
    unittest.main()
