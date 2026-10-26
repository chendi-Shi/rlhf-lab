"""Small recovery-plan fixtures; no accepted payload access or model execution."""
import copy
import unittest

from scripts.prepare_alignlab_evaluation_v3 import validate_reused_inputs, verify_cpu_proof, RECOVERY_TEST_MODULES


def prior_plan():
    return {"kind": "alignlab-primary-evaluation-campaign-v2", "schema_version": 1,
        "started_at_epoch": 1000, "deadline_epoch": 877600, "wall_seconds": 777600,
        "analysis_wall_seconds": 21600, "poll_seconds": 10,
        "roster": {"path": "same-roster", "sha256": "1" * 64}, "decoder": {"script": "same-decoder"},
        "model_python_sha256": "2" * 64, "scoring_python_sha256": "3" * 64,
        "authorization": {"path": "old-authority", "sha256": "4" * 64},
        "resource_profile": {"path": "old-profile", "sha256": "5" * 64},
        "protocol": {"path": "old-protocol", "sha256": "6" * 64}, "output": "failed-output"}


class RecoveryPlanTests(unittest.TestCase):
    def test_same_input_bytes_allow_new_location_without_new_budget(self):
        prior = prior_plan()
        candidate = copy.deepcopy(prior)
        candidate["output"] = "fresh-output"
        for field in ("authorization", "resource_profile", "protocol"):
            candidate[field]["path"] = "new-copy-" + field
        validate_reused_inputs(prior, candidate)

    def test_scope_budget_identity_changes_are_rejected(self):
        prior = prior_plan()
        for field in ("started_at_epoch", "deadline_epoch", "wall_seconds", "analysis_wall_seconds",
                      "poll_seconds", "roster", "decoder", "model_python_sha256", "scoring_python_sha256"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(prior)
                candidate["output"] = "fresh-output"
                candidate[field] = None
                with self.assertRaises(ValueError):
                    validate_reused_inputs(prior, candidate)

    def test_scientific_and_resource_byte_changes_are_rejected(self):
        prior = prior_plan()
        for field in ("authorization", "resource_profile", "protocol"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(prior)
                candidate["output"] = "fresh-output"
                candidate[field]["sha256"] = "f" * 64
                with self.assertRaises(ValueError):
                    validate_reused_inputs(prior, candidate)

    def test_failed_output_cannot_be_reused(self):
        with self.assertRaises(ValueError):
            validate_reused_inputs(prior_plan(), prior_plan())

    def test_proof_requires_actual_new_tests_and_matching_executable_sources(self):
        sources = {"alignment_lab/model_loader_eval_v3.py": "a" * 64,
                   "scripts/supervise_alignlab_evaluation_v3.py": "b" * 64}
        proof = {"status": "cpu_software_checks_passed", "source_unchanged": True,
            "device": "cpu", "cuda_initialized": False, "tests_run": 7,
            "failures": 0, "errors": 0, "skipped": 0,
            "source_sha256": dict(sources), "test_modules": sorted(RECOVERY_TEST_MODULES)}
        verify_cpu_proof(proof, sources)
        historical = copy.deepcopy(proof)
        historical["test_modules"] = ["tests_v2.test_evaluation_model_v2"]
        historical["tests_run"] = 151
        with self.assertRaises(ValueError):
            verify_cpu_proof(historical, sources)
        stale = copy.deepcopy(proof)
        stale["source_sha256"]["scripts/supervise_alignlab_evaluation_v3.py"] = "c" * 64
        with self.assertRaises(ValueError):
            verify_cpu_proof(stale, sources)


if __name__ == "__main__":
    unittest.main()
