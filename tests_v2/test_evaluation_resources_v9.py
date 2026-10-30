"""CPU-only V9 fixtures reproduce the V8 prelaunch failure/restoration schema.

All fixture pins are temporary synthetic bytes. Production trust anchors are
patched only by an explicit test context; no verifier accepts a bypass flag.
"""
import copy
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_resources_v9 as resource
from alignment_lab.evaluation_protocol_v2 import canonical
from tests_v2.test_evaluation_resources_v8 import process
from tests_v2.test_evaluation_resources_v8 import resource_test_anchors as prior_test_anchors


def fixture():
    from tests_v2.test_evaluation_resources_v8 import fixture as old_fixture
    profile, live, health, gpu = old_fixture()
    profile.update(schema_version=7, kind=resource.KIND)
    profile["services"][0]["processes"][0].update(
        pid=60380, parent_pid=61972, created_at_utc="2026-10-29T06:30:17.9429660Z")
    profile["service_identity_succession"]["prelaunch_closure"] = {
        "path": str(Path.cwd() / "unused-closure.json"), "sha256": "a" * 64}
    return profile, live, health, gpu


def pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def observed(identity):
    return {**identity, "created_utc_ticks": resource._created_ticks(identity)}


def pin_profile(root, profile, health, *, cpu_sources_mutator=None):
    """Return a temporary full six-restoration chain with the actual V8 closure topology."""
    from tests_v2.test_evaluation_resources_v8 import fixture as old_fixture, pin_profile as pin_old
    root = Path(root)
    launch = root / "closed-v8"
    launch.mkdir()
    history = root / "prior-history"
    history.mkdir()
    prior, _, prior_health, _ = old_fixture()
    pin_old(history, prior, prior_health)
    prior_authority = resource._read(prior["authorized_pause"]["authorization"])
    prior["authorized_pause"]["authorization"] = pin(launch / "evidoc-pause-authorization.json", prior_authority)
    prior_binding = pin(launch / "resource-profile.json", prior)
    preserved = prior["background_processes"] + prior["services"][1]["processes"]
    restore = {"executable": "fixture-python.exe", "arguments": ["-u", "-m", "scripts.serve_adapter",
        "--bundle", "dist/evidoc-adapter", "--port", "8083"], "working_directory": str(root / "evidoc")}
    manifest = {"kind": "alignlab-evidoc-restoration-manifest-v2", "deadline_epoch": resource.DEADLINE,
        "baseline_profile": prior["authorized_pause"]["baseline_profile"],
        "original_restoration": prior["restoration_report"],
        "service_identity_succession": prior["service_identity_succession"],
        "inherited_authorization": prior_authority["inherited_authorization"],
        "target": prior["services"][0]["processes"][0], "preserved_processes": preserved,
        "preserved_services": [prior["services"][1]], "restore": restore,
        "service": {"name": "evidoc", "port": 8083, "url": prior["services"][0]["url"],
                    "json_equals": prior_health[prior["services"][0]["url"]]}}
    manifest_pin = pin(launch / "restoration-manifest.json", manifest)
    restored = observed(profile["services"][0]["processes"][0])
    restored["handle_start_utc_ticks"] = "639288522179429666"
    common = {"manifest_sha256": manifest_pin["sha256"],
              "pause_authorization_sha256": prior["authorized_pause"]["authorization"]["sha256"]}
    receipt = {**common, "kind": "alignlab-evidoc-restoration-complete-v1", "status": "restored_and_verified",
        "restoration_complete": True, "restore_pending": False, "all_28_fields_match": True,
        "deadline_epoch": resource.DEADLINE, "preserved_processes": preserved,
        "processes": [restored], "health": health[profile["services"][0]["url"]]}
    receipt_pin = pin(launch / "restoration-complete.json", receipt)
    evidence = {"restoration-manifest.json": manifest_pin, "restoration-complete.json": receipt_pin,
        "restoration-transaction.json": pin(launch / "restoration-transaction.json", receipt),
        "resource-profile.json": prior_binding,
        "evidoc-pause-authorization.json": prior["authorized_pause"]["authorization"]}
    guardian_identity = observed({**process(61972, "pwsh.exe -File frozen-guardian.ps1", 63212),
        "created_at_utc": "2026-10-29T06:29:33.8467810Z"})
    guardian = {**common, "kind": "alignlab-evidoc-restore-guardian-v1", "status": "complete",
        "reason": "explicit_restore_requested", "deadline_epoch": resource.DEADLINE, "supervisor": None,
        "plan": None, "restoration_evidence": receipt_pin["path"], "guardian": guardian_identity}
    evidence["guardian/status.json"] = pin(launch / "guardian/status.json", guardian)
    evidence["guardian/restore-launch-attempt.json"] = pin(launch / "guardian/restore-launch-attempt.json",
        {**common, **restore, "max_restore_launches": 1, "guardian": guardian_identity})
    evidence["guardian/RESTORE"] = pin(launch / "guardian/RESTORE", {"restore_required": True})
    evidence["pause-complete.json"] = pin(launch / "pause-complete.json", {**common,
        "kind": "alignlab-evidoc-pause-complete-v1", "status": "paused", "port_8083_absent": True,
        "restore_pending": True, "restoration_complete": False, "deadline_epoch": resource.DEADLINE,
        "target": observed(manifest["target"]), "guardian": guardian_identity, "preserved_processes": preserved})
    sources = {"fixture_source_" + str(i) + ".py": "a" * 64 for i in range(192)}
    # Real V8 evidence audited a superset: 348 files, including all 192 plan
    # sources. Mutators construct internally pinned negative fixture chains.
    audited = {**sources, **{"extra_audited_" + str(i) + ".py": "b" * 64 for i in range(156)}}
    if cpu_sources_mutator is not None:
        cpu_sources_mutator(audited)
    cpu = {"kind": "alignlab-independent-evaluation-v8-cpu-proof", "status": "cpu_software_checks_passed",
        "tests_run": 121, "failures": 0, "errors": 0, "skipped": 0, "source_unchanged": True,
        "device": "cpu", "torch_imported": False, "cuda_initialized": False,
        "pretrained_model_weights_used": False, "training_started": False, "model_acceptance_repeated": False,
        "capacity_verified": False, "source_sha256": audited}
    cpu_pin = pin(root / "v8-cpu.json", cpu)
    slots = {slot: {"fixture": slot} for slot in sorted(resource.INHERITED_SLOTS)}
    plan = {"kind": "alignlab-primary-evaluation-campaign-v2", "started_at_epoch": resource.STARTED,
        "deadline_epoch": resource.DEADLINE, "wall_seconds": resource.WALL_SECONDS,
        "resource_profile": prior_binding, "source_sha256": sources, "cpu_check_report": cpu_pin,
        "output": str(root / "absent-v8-output"), "inherited_inference": slots, "inherited_scores": slots,
        "inherited_capacity": {"original": "full-v5-v6"},
        "protocol": pin(root / "protocol.json", {"fixture": "protocol"}),
        "authorization": prior_authority["evaluation_authorization"]}
    plan_pin = pin(launch / "plan.json", plan)
    evidence["plan.json"] = plan_pin
    prepared = {"kind": "alignlab-v8-static-prepare-accepted",
        "status": "static_validation_passed_waiting_for_resource_change", "tool_exit_code": 0,
        "tool_session_collected_and_closed": True, "ready_for_launch": False,
        "plan": plan_pin, "candidate": pin(launch / "plan-candidate.json", plan),
        "source_count": 192, "prior_source_count": 173, "new_source_count": 19,
        "current_source_sha256_all_match": True, "actual_full_static_validation_passed": True,
        "prior_sources_unchanged": True, "output_created": False, "model_loaded": False,
        "cuda_used": False, "service_pause_performed": False, "model_acceptance_repeated": False,
        "inherited_result_chains_verified": 7, "inherited_complete_capacity_models_verified": 9,
        "inherited_complete_capacity_audits_verified": 18, "inherited_results_accepted": 3787,
        "ppo44_partial_records_excluded": 303, "cpu_tests_passed": 121,
        "cpu_errors": 0, "cpu_failures": 0, "cpu_skips": 0, "deadline_epoch": resource.DEADLINE,
        "inherited_inference_slots": sorted(slots), "inherited_score_slots": sorted(slots),
        "fresh_output": plan["output"], "capacity_binding": plan["inherited_capacity"],
        "cpu_proof": cpu_pin, "prepare_log": pin(root / "prepare.log", {"exit": 0})}
    evidence[resource.PREPARE_EVIDENCE] = pin(launch / resource.PREPARE_EVIDENCE, prepared)
    ram = 12.018177032470703
    admission = {"kind": "alignlab-evaluation-v8-prestart-admission", "status": "waiting_resources_before_start",
        "plan": plan_pin, "presence": {"process": False, "output": False, "campaign": False,
            "execution_launch": False, "plan_stop": False}, "admission_ram_headroom_gib": 12.5,
        "runtime_ram_floor_gib": 8, "sample_conditions_passed": False, "first_admission": None,
        "sample_separation_seconds": None, "required_sample_separation_seconds": 300,
        "unknown_python": [], "locks": [], "blockers": ["admission_ram_below_12_5_gib"],
        "deadline_epoch": resource.DEADLINE, "actual_capacity_started": False,
        "actual_inference_started": False, "model_or_python_started": False, "available_ram_gib": ram,
        "hashes": {"plan": plan_pin["sha256"], "protocol": plan["protocol"]["sha256"],
            "profile": prior_binding["sha256"], "authorization": plan["authorization"]["sha256"]},
        "gpu": {"index": 0, "total_mib": 8192, "used_mib": 1345,
                "free_mib": 6847, "utilization_percent": 7},
        "identities": [{"expected": p, "matches": True} for p in preserved]}
    evidence[resource.ADMISSION_EVIDENCE] = pin(launch / resource.ADMISSION_EVIDENCE, admission)
    direct_attempt = {"kind": "alignlab-v8-direct-user-single-launch-attempt",
        "direct_user_instruction": "那就启动啊", "preceding_direct_user_instruction": "那继续跑啊",
        "natural_resource_improvement_claimed": False, "automatic_retry_authorized": False,
        "maximum_attempts": 1, "plan": plan_pin, "static_acceptance": evidence[resource.PREPARE_EVIDENCE],
        "source_count_verified": 192, "helper_files_verified": 9,
        "limits_unchanged": {"gpu_free_mib": 3840, "sample_gap_max_seconds": 900,
            "sample_gap_min_seconds": 300, "admission_ram_gib": 12.5,
            "deadline_epoch": resource.DEADLINE, "gpu_utilization_max": 20},
        "pause_scope": "Only exact EviDoc 8083 adapter PID60124 with frozen full identity",
        "failure_action": "Immediately signal frozen guardian to restore original configuration and verify all28 fields; close this allocation permanently",
        "other_services_or_apps_operations_authorized": False, "frozen_plan_source_or_proofs_modified": False,
        "gpu_execution_requires_two_passed_actual_admissions": True,
        "historical_unknown_pid58104_origin_resolved": False}
    evidence[resource.DIRECT_ATTEMPT_EVIDENCE] = pin(launch / resource.DIRECT_ATTEMPT_EVIDENCE, direct_attempt)
    failed = {"status": "prelaunch_admission_failed", "plan": plan_pin,
        "admission": evidence[resource.ADMISSION_EVIDENCE], "available_ram_gib": ram, "required_ram_gib": 12.5,
        "blockers": admission["blockers"], "restore_required": True, "gpu_started": False,
        "retry_forbidden": True, "direct_user_attempt": evidence[resource.DIRECT_ATTEMPT_EVIDENCE]}
    evidence["launch-failed.json"] = pin(launch / "launch-failed.json", failed)
    closure = {"kind": "alignlab-v8-prelaunch-admission-failure-closure-v1",
        "status": "v8_prelaunch_admission_failed_closed_services_restored", "launch_directory": str(launch),
        "intended_output_root": plan["output"], "plan": plan_pin, "evidence": evidence,
        "failure_reason": "available_ram_below_frozen_admission_floor", "required_admission_ram_gib": 12.5,
        "failed_admission_available_ram_gib": ram, "failed_admission_blockers": admission["blockers"],
        "static_prepare_passed": True, "cpu_tests_passed": 121, "new_gpu_run_started": False,
        "evaluation_output_created": False, "evaluation_process_created": False, "execution_launch_created": False,
        "second_admission_attempted": False, "failed_plan_must_not_be_redispatched": True,
        "retry_authorized": False, "restore_pending": False, "restoration_complete": True,
        "restored_identity_matches": True, "restored_identity": restored, "restored_live_identity": restored,
        "port_8083_listener_owners": [60380], "health_fields_checked": 28, "all_28_fields_match": True,
        "health": receipt["health"], "unknown_python_processes": [], "guardian_status": "complete",
        "guardian_exact_identity_exited": True, "guardian_pid_current_occupant": None,
        "restoration_launch_attempt_count": 1, "accepted_policy_score_count": 7,
        "accepted_policy_generated_records": 3787, "excluded_ppo44_partial_records": 303,
        "deadline_epoch": resource.DEADLINE, "no_other_service_operations": True, "no_frozen_source_changes": True,
        "source_count": 192, "manual_single_attempt": True, "natural_resource_improvement_claimed": False,
        "direct_user_instruction": "那就启动啊", "original_budget_unchanged": True,
        "preserved_process_checks": [{"expected": p, "live": observed(p), "matches": True} for p in preserved],
        "python_processes": [*(observed(p) for p in preserved), restored]}
    profile["restoration_report"] = prior["restoration_report"]
    profile["service_identity_succession"] = {"restoration_report": receipt_pin,
        "restoration_manifest": manifest_pin, "prelaunch_closure": pin(launch / "closure.json", closure)}
    profile["authorized_pause"]["baseline_profile"] = prior_binding
    authority = {"kind": resource.PAUSE_KIND, "status": "authorized", "authorized": True,
        "user_authorization_quote": resource.PAUSE_QUOTE, "service": "evidoc", "port": 8083,
        "baseline_profile": prior_binding, "restoration_report": profile["restoration_report"],
        "deadline_epoch": resource.DEADLINE, "restore_required": True,
        "evaluation_authorization": prior_authority["evaluation_authorization"],
        "inherited_authorization": prior["authorized_pause"]["authorization"],
        "service_identity_succession": profile["service_identity_succession"]}
    profile["authorized_pause"]["authorization"] = pin(root / "authority-v9.json", authority)
    return pin(root / "profile-v9.json", profile)


@contextmanager
def resource_test_anchors(profile):
    succession = profile["service_identity_succession"]
    closure = resource._read(succession["prelaunch_closure"])
    prior = resource._read(profile["authorized_pause"]["baseline_profile"])
    with prior_test_anchors(prior), patch.multiple(resource,
            V8_BASELINE_SHA256=profile["authorized_pause"]["baseline_profile"]["sha256"],
            V8_PLAN_SHA256=closure["plan"]["sha256"],
            SUCCESSION_SHA256={key: value["sha256"] for key, value in succession.items()}):
        yield


class V9ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.live, self.health, self.gpu = fixture()
        clock = patch.object(resource.time, "time", return_value=1793246400)
        clock.start()
        self.addCleanup(clock.stop)

    def observe(self):
        return resource.collect_resources(self.profile, inventory_reader=lambda: copy.deepcopy(self.gpu),
            ram_reader=lambda: 12.0, process_reader=lambda: copy.deepcopy(self.live),
            health_reader=lambda url, timeout: copy.deepcopy(self.health[url]), port_reader=lambda: [])

    def test_same_allocator_global_ram_and_preload_floors(self):
        self.assertEqual(self.observe()["blockers"], [])
        self.assertEqual(self.observe()["required_free_mib"], 3840)
        for key, value in (("allocator_limit_mib", 2561), ("min_available_ram_gib", 7),
                           ("minimum_global_free_mib", 1000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                resource.validate_profile({**self.profile, key: value})

    def test_unknown_python_keeps_full_identity_and_fails_closed(self):
        self.live.append(process(58104, "python unrelated.py"))
        result = self.observe()
        self.assertTrue(any("Unknown Python" in reason for reason in result["blockers"]))

    def test_actual_cim_ticks_are_not_rounded_or_handle_ticks(self):
        original = observed(self.profile["services"][0]["processes"][0])
        self.assertEqual(resource._created_ticks(original), "639288522179429660")
        resource._exact_identity({**original, "created_utc_ticks": 639288522179429660})
        for ticks in ("639288522179429666", "639288522179429661", float(639288522179429660), None, True):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                resource._exact_identity({**original, "created_utc_ticks": ticks})

    def test_full_six_restoration_chain_and_relation_without_live_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            with resource_test_anchors(self.profile), \
                 patch.object(resource, "_health", side_effect=AssertionError("No live health")), \
                 patch.object(resource, "_port_8083_listeners", side_effect=AssertionError("No live process")):
                self.assertEqual(resource.load_profile(binding), self.profile)
                self.assertEqual(resource.validate_authorized_profile_relation(
                    self.profile["authorized_pause"]["baseline_profile"], binding), self.profile)
                with self.assertRaises(ValueError):
                    resource.load_profile(binding, now_fn=lambda: resource.DEADLINE)

    def test_unanchored_self_reported_successor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            with self.assertRaisesRegex(ValueError, "exact closed V8"):
                resource.load_profile(binding)

    def test_cpu_audit_superset_retains_every_plan_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            closure = resource._read(self.profile["service_identity_succession"]["prelaunch_closure"])
            plan = resource._read(closure["plan"])
            cpu = resource._read(plan["cpu_check_report"])
            self.assertEqual(len(plan["source_sha256"]), 192)
            self.assertEqual(len(cpu["source_sha256"]), 348)
            with resource_test_anchors(self.profile):
                self.assertEqual(resource.load_profile(binding), self.profile)

    def test_cpu_audit_missing_required_source_is_rejected_after_full_pin_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health,
                cpu_sources_mutator=lambda sources: sources.pop("fixture_source_72.py"))
            with resource_test_anchors(self.profile), self.assertRaisesRegex(ValueError, "every required plan source"):
                resource.load_profile(binding)

    def test_cpu_audit_wrong_required_sha_is_rejected_after_full_pin_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health,
                cpu_sources_mutator=lambda sources: sources.update({"fixture_source_72.py": "f" * 64}))
            with resource_test_anchors(self.profile), self.assertRaisesRegex(ValueError, "every required plan source"):
                resource.load_profile(binding)

    def test_closed_allocation_cannot_gain_execution_or_output(self):
        for name in ("process.json", "execution-launch.json", "output"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile, _, health, _ = fixture()
                binding = pin_profile(root, profile, health)
                closure = resource._read(profile["service_identity_succession"]["prelaunch_closure"])
                target = (Path(closure["intended_output_root"]) if name == "output"
                          else Path(closure["launch_directory"]) / name)
                target.write_text("{}", encoding="utf-8")
                with resource_test_anchors(profile), self.assertRaisesRegex(ValueError, "remain unused"):
                    resource.load_profile(binding)

    def test_closure_negative_flags_cannot_be_rebound_as_safe(self):
        changes = {"new_gpu_run_started": True, "evaluation_process_created": True,
            "second_admission_attempted": True, "retry_authorized": True, "guardian_exact_identity_exited": False,
            "restoration_launch_attempt_count": 2, "deadline_epoch": resource.DEADLINE + 1,
            "unknown_python_processes": [{"pid": 58104}], "health_fields_checked": 27,
            "source_count": 173, "manual_single_attempt": False, "natural_resource_improvement_claimed": True,
            "original_budget_unchanged": False}
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile, _, health, _ = fixture()
                pin_profile(root, profile, health)
                succession = profile["service_identity_succession"]
                closure = resource._read(succession["prelaunch_closure"])
                closure[key] = value
                succession["prelaunch_closure"] = pin(Path(succession["prelaunch_closure"]["path"]), closure)
                authority = resource._read(profile["authorized_pause"]["authorization"])
                authority["service_identity_succession"] = copy.deepcopy(succession)
                profile["authorized_pause"]["authorization"] = pin(root / "changed-auth.json", authority)
                binding = pin(root / "changed-profile.json", profile)
                with resource_test_anchors(profile), self.assertRaises(ValueError):
                    resource.load_profile(binding)

    def test_rehashed_single_attempt_cannot_authorize_retry_or_wider_scope(self):
        changes = {"maximum_attempts": 2, "automatic_retry_authorized": True,
            "natural_resource_improvement_claimed": True, "source_count_verified": 173,
            "helper_files_verified": 8, "other_services_or_apps_operations_authorized": True,
            "gpu_execution_requires_two_passed_actual_admissions": False,
            "historical_unknown_pid58104_origin_resolved": True,
            "pause_scope": "Any service may pause", "failure_action": "Retry instead of restoring"}
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile, _, health, _ = fixture()
                pin_profile(root, profile, health)
                succession = profile["service_identity_succession"]
                closure = resource._read(succession["prelaunch_closure"])
                evidence = closure["evidence"]
                direct = resource._read(evidence[resource.DIRECT_ATTEMPT_EVIDENCE])
                direct[key] = value
                evidence[resource.DIRECT_ATTEMPT_EVIDENCE] = pin(
                    Path(evidence[resource.DIRECT_ATTEMPT_EVIDENCE]["path"]), direct)
                failed = resource._read(evidence["launch-failed.json"])
                failed["direct_user_attempt"] = evidence[resource.DIRECT_ATTEMPT_EVIDENCE]
                evidence["launch-failed.json"] = pin(Path(evidence["launch-failed.json"]["path"]), failed)
                succession["prelaunch_closure"] = pin(Path(succession["prelaunch_closure"]["path"]), closure)
                authority = resource._read(profile["authorized_pause"]["authorization"])
                authority["service_identity_succession"] = copy.deepcopy(succession)
                profile["authorized_pause"]["authorization"] = pin(root / "changed-authority.json", authority)
                binding = pin(root / "changed-profile.json", profile)
                with resource_test_anchors(profile), self.assertRaisesRegex(ValueError, "single human-directed"):
                    resource.load_profile(binding)

    def test_candidate_bytes_remain_bound_to_the_accepted_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            closure = resource._read(self.profile["service_identity_succession"]["prelaunch_closure"])
            prepared = resource._read(closure["evidence"][resource.PREPARE_EVIDENCE])
            candidate = Path(prepared["candidate"]["path"])
            candidate.write_bytes(candidate.read_bytes() + b" ")
            with resource_test_anchors(self.profile), self.assertRaises(ValueError):
                resource.load_profile(binding)

    def test_every_closure_evidence_pin_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            closure = resource._read(self.profile["service_identity_succession"]["prelaunch_closure"])
            with resource_test_anchors(self.profile):
                for name, item in closure["evidence"].items():
                    path = Path(item["path"])
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    try:
                        with self.subTest(name=name), self.assertRaises(ValueError):
                            resource.load_profile(binding)
                    finally:
                        path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
