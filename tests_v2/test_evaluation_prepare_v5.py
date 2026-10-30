"""New explicit service-pause recovery boundaries; no real payload access."""
import copy
import unittest
from unittest.mock import patch
from scripts import prepare_alignlab_evaluation_v5 as subject
from tests_v2.test_evaluation_prepare_v3 import prior_plan


class AuthorizedRecoveryPlanTests(unittest.TestCase):
    def setUp(self):
        self.prior = prior_plan()
        self.candidate = copy.deepcopy(self.prior)
        self.candidate["output"] = "fresh-v4-output"
        self.candidate["resource_profile"] = {"path": "new-v4-profile", "sha256": "a" * 64}
        self.profile = {"authorized_pause": {"baseline_profile": copy.deepcopy(self.prior["resource_profile"]),
                                           "deadline_epoch": self.prior["deadline_epoch"]}}

    def check(self):
        with patch.object(subject, "load_profile", return_value=self.profile) as reader:
            subject.validate_reused_inputs(self.prior, self.candidate)
            reader.assert_called_once_with(self.candidate["resource_profile"])

    def test_only_verified_pause_transition_is_allowed(self):
        self.check()

    def test_profile_validation_failure_is_not_bypassed(self):
        with patch.object(subject, "load_profile", side_effect=ValueError("No explicit pause permission")):
            with self.assertRaises(ValueError):
                subject.validate_reused_inputs(self.prior, self.candidate)

    def test_scope_protocol_allocation_or_original_authority_cannot_change(self):
        for field in ("kind", "schema_version", "started_at_epoch", "deadline_epoch", "wall_seconds",
                      "analysis_wall_seconds", "poll_seconds", "roster", "decoder",
                      "model_python_sha256", "scoring_python_sha256"):
            candidate = copy.deepcopy(self.candidate)
            candidate[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.validate_reused_inputs(self.prior, candidate)
        for field in ("protocol", "authorization"):
            candidate = copy.deepcopy(self.candidate)
            candidate[field]["sha256"] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.validate_reused_inputs(self.prior, candidate)

    def test_unrelated_baseline_or_extended_pause_are_rejected(self):
        for field, value in (("baseline_profile", {"path": "elsewhere", "sha256": "b" * 64}),
                             ("deadline_epoch", self.prior["deadline_epoch"] + 1)):
            with self.subTest(field=field):
                old = self.profile["authorized_pause"][field]
                self.profile["authorized_pause"][field] = value
                with self.assertRaises(ValueError):
                    self.check()
                self.profile["authorized_pause"][field] = old

    def test_old_output_must_never_be_reused(self):
        self.candidate["output"] = self.prior["output"]
        with self.assertRaises(ValueError):
            self.check()

    def test_old_cpu_proof_cannot_authorize_new_sources(self):
        sources = {"alignment_lab/evaluation_resources_v5.py": "a" * 64}
        proof = {"status": "cpu_software_checks_passed", "source_unchanged": True,
                 "device": "cpu", "cuda_initialized": False, "tests_run": 6,
                 "failures": 0, "errors": 0, "skipped": 0,
                 "test_modules": sorted(subject.RECOVERY_TEST_MODULES), "source_sha256": sources}
        subject.verify_cpu_proof(proof, sources)
        old = copy.deepcopy(proof)
        old["test_modules"] = ["tests_v2.test_model_loader_eval_v3"]
        with self.assertRaises(ValueError):
            subject.verify_cpu_proof(old, sources)
        old = copy.deepcopy(proof)
        old["source_sha256"] = {}
        with self.assertRaises(ValueError):
            subject.verify_cpu_proof(old, sources)


if __name__ == "__main__":
    unittest.main()
