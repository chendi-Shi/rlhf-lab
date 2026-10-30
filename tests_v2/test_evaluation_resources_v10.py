"""CPU metadata fixtures for the closed V9 and its exact restored successor.

Only immutable JSON/source bytes are read. Temporary new profile/authority
fixtures are never launch artifacts and never change a service or load weights.
"""
import copy
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_resources_v10 as resource
from alignment_lab.evaluation_protocol_v2 import canonical, parse
from tests_v2.test_evaluation_resources_v8 import process

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "reports/alignlab-quality-v9-launch-20261029T064300Z"
CLOSURE = LAUNCH / "failure-closure-20261029T1240319507741Z.json"


def pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    path.write_bytes(data)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def existing(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture():
    profile = parse((LAUNCH / "resource-profile.json").read_bytes())
    receipt = parse((LAUNCH / "restoration-complete.json").read_bytes())
    profile.update(schema_version=8, kind=resource.KIND)
    evidoc = next(s for s in profile["services"] if s["name"] == "evidoc")
    evidoc["processes"] = [{k: receipt["processes"][0][k] for k in resource.IDENTITY_FIELDS}]
    profile["authorized_pause"]["baseline_profile"] = existing(LAUNCH / "resource-profile.json")
    profile["service_identity_succession"] = {
        "restoration_report": existing(LAUNCH / "restoration-complete.json"),
        "restoration_manifest": existing(LAUNCH / "restoration-manifest.json"),
        "failure_closure": existing(CLOSURE),
    }
    live = copy.deepcopy(profile["background_processes"] + [p for s in profile["services"]
        if s["name"] != "evidoc" for p in s["processes"]])
    health = {s["name"]: copy.deepcopy(s["json_equals"]) for s in profile["services"]}
    gpu = [{"index": 0, "name": "NVIDIA GeForce RTX 3060 Ti", "driver": "472.80",
            "total_mb": 8192, "free_mb": 6000, "utilization_percent": 0}]
    return profile, live, health, gpu


def pin_profile(root, profile, health, **kwargs):
    if kwargs:
        raise ValueError("No synthetic replacement of original V9 proof is supported")
    baseline = parse((LAUNCH / "resource-profile.json").read_bytes())
    old_auth = resource._read(baseline["authorized_pause"]["authorization"])
    authority = {"kind": resource.PAUSE_KIND, "status": "authorized", "authorized": True,
        "user_authorization_quote": resource.PAUSE_QUOTE, "service": "evidoc", "port": 8083,
        "baseline_profile": profile["authorized_pause"]["baseline_profile"],
        "restoration_report": profile["restoration_report"], "deadline_epoch": resource.DEADLINE,
        "restore_required": True, "inherited_authorization": baseline["authorized_pause"]["authorization"],
        "service_identity_succession": profile["service_identity_succession"],
        "prior_failure_closure": profile["service_identity_succession"]["failure_closure"],
        "evaluation_authorization": old_auth["evaluation_authorization"]}
    profile["authorized_pause"]["authorization"] = pin(root / "fixture-pause-authorization.json", authority)
    return pin(root / "fixture-resource-profile.json", profile)


@contextmanager
def resource_test_anchors(profile):
    # Real frozen V9 trust anchors remain active; only the fixture wall clock
    # is fixed inside the original allocation. There is no production bypass.
    with patch.object(resource.time, "time", return_value=1793279000):
        yield


class V10ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile, self.live, self.health, self.gpu = fixture()
        self.binding = pin_profile(Path(self.temp.name), self.profile, self.health)

    def closure_inputs(self):
        succession = self.profile["service_identity_succession"]
        return (resource._read(self.profile["authorized_pause"]["baseline_profile"]),
                resource._read(succession["restoration_report"]),
                resource._read(succession["restoration_manifest"]),
                resource._read(succession["failure_closure"]))

    @contextmanager
    def changed_closure(self, changed):
        original = resource._read
        closure_pin = self.profile["service_identity_succession"]["failure_closure"]
        def read(bound):
            return copy.deepcopy(changed) if bound == closure_pin else original(bound)
        # Only the semantic input is fault-injected. Historical bytes/pins and
        # every other evidence reader stay intact; no fixture is authority.
        with patch.object(resource, "_read", side_effect=read):
            yield

    def test_complete_v9_succession_is_read_only(self):
        with resource_test_anchors(self.profile), patch.object(resource, "_health", side_effect=AssertionError), \
                patch.object(resource, "_port_8083_listeners", side_effect=AssertionError):
            self.assertEqual(resource.load_profile(self.binding), self.profile)

    def test_only_original_deadline_and_fixed_limits(self):
        for key in resource.LIMITS:
            candidate = copy.deepcopy(self.profile)
            candidate[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                resource.validate_profile(candidate)
        with self.assertRaises(ValueError):
            resource.load_profile(self.binding, now_fn=lambda: resource.DEADLINE)

    def test_wrong_succession_pin_never_becomes_authority(self):
        for key in self.profile["service_identity_succession"]:
            candidate = copy.deepcopy(self.profile)
            candidate["service_identity_succession"][key]["sha256"] = "f" * 64
            bound = pin(Path(self.temp.name) / (key + ".json"), candidate)
            with self.subTest(key=key), self.assertRaises(ValueError):
                resource.load_profile(bound)

    def test_restored_ticks_are_exact_strings(self):
        identity = parse((LAUNCH / "restoration-complete.json").read_bytes())["processes"][0]
        self.assertEqual(resource._created_ticks(identity), "639288735689153970")
        resource._exact_identity(identity)
        with self.assertRaises(ValueError):
            resource._exact_identity({**identity, "created_utc_ticks": 1.25})

    def test_real_historical_stage_four_fields_are_bound_without_rewriting_evidence(self):
        baseline, receipt, manifest, closure = self.closure_inputs()
        expected_ticks = {61168: "639288553760130810", 60156: "639288553760492180"}
        stage_identities = [c["expected"] for c in closure["owned_process_checks"]
                            if c["expected"]["pid"] in expected_ticks]
        self.assertEqual(len(stage_identities), 2)
        before = hashlib.sha256(CLOSURE.read_bytes()).hexdigest()
        for identity in stage_identities:
            self.assertEqual(set(identity), resource.IDENTITY_FIELDS)
            self.assertNotIn("created_utc_ticks", identity)
            self.assertEqual(resource._created_ticks(identity), expected_ticks[identity["pid"]])
        accepted = resource._verify_failure_closure(self.profile, baseline, receipt, manifest)
        self.assertEqual(accepted, closure)
        self.assertEqual(hashlib.sha256(CLOSURE.read_bytes()).hexdigest(), before)

    def test_explicit_wrong_ticks_cannot_be_ignored_for_historical_or_helper_identity(self):
        baseline, receipt, manifest, closure = self.closure_inputs()
        for pid in (61168, 60156, 27308, 49032, 41584, 62152):
            changed = copy.deepcopy(closure)
            identity = next(c["expected"] for c in changed["owned_process_checks"]
                            if c["expected"]["pid"] == pid)
            correct = resource._created_ticks(identity)
            identity["created_utc_ticks"] = correct[:-1] + ("1" if correct[-1] != "1" else "2")
            with self.subTest(pid=pid), self.changed_closure(changed), self.assertRaises(ValueError):
                resource._verify_failure_closure(self.profile, baseline, receipt, manifest)

    def test_all_six_owned_identities_must_crossbind_to_frozen_campaign_and_helpers(self):
        baseline, receipt, manifest, closure = self.closure_inputs()
        for pid in (27308, 49032, 61168, 60156, 41584, 62152):
            for field in ("parent_pid", "command_line"):
                changed = copy.deepcopy(closure)
                identity = next(c["expected"] for c in changed["owned_process_checks"]
                                if c["expected"]["pid"] == pid)
                if field == "parent_pid": identity[field] += 1
                else: identity[field] += " --synthetic-unowned"
                # Cardinality, uniqueness, valid creation time/ticks and exit
                # booleans still hold; rejection must come from provenance.
                self.assertEqual(len(changed["owned_process_checks"]), 6)
                self.assertEqual(len({resource.signature(c["expected"])
                                      for c in changed["owned_process_checks"]}), 6)
                with self.subTest(pid=pid, field=field), self.changed_closure(changed), self.assertRaises(ValueError):
                    resource._verify_failure_closure(self.profile, baseline, receipt, manifest)

    def test_failed_partial_or_restoration_claims_cannot_be_promoted(self):
        original = resource._read
        closure_pin = self.profile["service_identity_succession"]["failure_closure"]
        baseline = original(self.profile["authorized_pause"]["baseline_profile"])
        receipt = original(self.profile["service_identity_succession"]["restoration_report"])
        manifest = original(self.profile["service_identity_succession"]["restoration_manifest"])
        closure = original(closure_pin)
        changes = {"failed_partial_accepted": True, "manifest_present": True,
            "restoration_complete": False, "restore_pending": True, "source_count": 210,
            "all_owned_exact_identities_exited": False, "restore_launch_attempt_count": 2,
            "automatic_relaunch_allowed": True, "deadline_epoch": resource.DEADLINE + 1,
            "health_fields_checked": 27, "formal_generations_accepted": 4000}
        for key, value in changes.items():
            changed = {**closure, key: value}
            def read(bound):
                return copy.deepcopy(changed) if bound == closure_pin else original(bound)
            with self.subTest(key=key), patch.object(resource, "_read", side_effect=read), self.assertRaises(ValueError):
                resource._verify_failure_closure(self.profile, baseline, receipt, manifest)

    def test_source_audit_or_owned_identity_loss_is_rejected(self):
        original = resource._read
        succession = self.profile["service_identity_succession"]
        baseline = original(self.profile["authorized_pause"]["baseline_profile"])
        receipt, manifest = original(succession["restoration_report"]), original(succession["restoration_manifest"])
        closure = original(succession["failure_closure"])
        for field in ("source_audit", "owned_process_checks", "preserved_process_checks"):
            changed = copy.deepcopy(closure)
            changed[field].pop()
            def read(bound):
                return copy.deepcopy(changed) if bound == succession["failure_closure"] else original(bound)
            with self.subTest(field=field), patch.object(resource, "_read", side_effect=read), self.assertRaises(ValueError):
                resource._verify_failure_closure(self.profile, baseline, receipt, manifest)


if __name__ == "__main__":
    unittest.main()
