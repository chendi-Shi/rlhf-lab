"""R5 dispatch routing only; actual parent proof has separate payload audits."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import alignment_stage as child
from alignment_lab import alignment_supervisor as supervisor
from alignment_lab import evaluation_acceptance as evaluation
from alignment_lab import recovered_parent_acceptance as old
from alignment_lab import recovered_parent_acceptance_r5 as r5


class RecoveredParentR5DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = self.write("campaign.json", {"status": "failed", "parents_verified": False})
        self.parent = self.write("parent.json", {"source_sha256": {}, "output": str(self.root)})
        self.accepted = {"kind": r5.KIND, "original_campaign": {"path": str(self.campaign)}}
        self.proof = self.write("accepted.json", self.accepted)
        self.request = {"parent_acceptance_path": str(self.proof), "parent_acceptance_sha256": self.sha(self.proof),
            "parent_plan_path": str(self.parent), "parent_plan_sha256": self.sha(self.parent)}

    def write(self, name, obj):
        path = self.root / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_stage_routes_only_r5_kind_to_r5_validator(self):
        with patch.object(r5, "validate_recovered_for_request", return_value=self.accepted) as verify, \
                patch.object(old, "validate_recovered_for_request", side_effect=AssertionError("wrong protocol")):
            self.assertIs(child.validate_parent_acceptance(self.request), self.accepted)
            verify.assert_called_once_with(self.request)

    def test_stage_propagates_actual_r5_proof_rejection(self):
        with patch.object(r5, "validate_recovered_for_request", side_effect=ValueError("bad complete chain")):
            with self.assertRaisesRegex(ValueError, "bad complete chain"):
                child.validate_parent_acceptance(self.request)

    def test_supervisor_preserves_failed_campaign_and_uses_r5_validator(self):
        before = self.campaign.read_bytes()
        with patch.object(r5, "validate_recovered_for_request", return_value=self.accepted) as verify, \
                patch.object(supervisor.parent_supervisor, "validate_plan", side_effect=AssertionError("old success route")):
            parent, campaign = supervisor._parent_acceptance(self.request)
            self.assertEqual(parent["source_sha256"], {})
            self.assertEqual(campaign["status"], "failed")
            self.assertFalse(campaign["parents_verified"])
            verify.assert_called_once_with(self.request)
        self.assertEqual(before, self.campaign.read_bytes())

    def test_bad_external_acceptance_hash_rejected_before_validator(self):
        request = {**self.request, "parent_acceptance_sha256": "0" * 64}
        with patch.object(r5, "validate_recovered_for_request") as verify:
            for function in (child.validate_parent_acceptance, supervisor._parent_acceptance):
                with self.assertRaises((ValueError, RuntimeError)):
                    function(request)
            verify.assert_not_called()

    def test_old_recovery_protocol_is_not_silently_redirected(self):
        self.write("accepted.json", {**self.accepted, "kind": old.KIND})
        self.request["parent_acceptance_sha256"] = self.sha(self.proof)
        with patch.object(old, "validate_recovered_for_request", return_value=self.accepted) as verify, \
                patch.object(r5, "validate_recovered_for_request", side_effect=AssertionError("wrong protocol")):
            child.validate_parent_acceptance(self.request)
            supervisor._parent_acceptance(self.request)
            self.assertEqual(verify.call_count, 2)

    def test_unknown_recovery_kind_is_rejected(self):
        self.write("accepted.json", {**self.accepted, "kind": r5.KIND + "-unknown"})
        self.request["parent_acceptance_sha256"] = self.sha(self.proof)
        with patch.object(r5, "validate_recovered_for_request") as verify:
            with self.assertRaises(ValueError):
                child.validate_parent_acceptance(self.request)
            verify.assert_not_called()
        self.assertIsNone(evaluation._recovered_auditor(r5.KIND + "-unknown"))

    def test_evaluation_explicit_protocol_route(self):
        self.assertIs(evaluation._recovered_auditor(r5.KIND), r5)
        self.assertIs(evaluation._recovered_auditor(old.KIND), old)
        self.assertIsNone(evaluation._recovered_auditor("alignlab-independent-formal-parents-acceptance-v1"))

    def test_historical_r5_process_requires_exact_exit_identity(self):
        identity = {"pid": 55, "parent_pid": 50, "created_at_utc": "2026-10-21T08:00:00Z",
                    "command_line": "python run_alignlab_parent_recovery_r5.py --execute"}
        self.write("accepted.json", {**self.accepted, "processes": [{"expected": identity}]})
        supervisor._parents_gone(self.request, [])
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            supervisor._parents_gone(self.request, [identity])
        supervisor._parents_gone(self.request, [{**identity, "created_at_utc": "2026-10-21T09:00:00Z"}])


if __name__ == "__main__":
    unittest.main()