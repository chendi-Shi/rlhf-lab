"""CPU-only resource fault injection; never a capacity or quality claim."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_resources_v7 as resource
from alignment_lab.evaluation_protocol_v2 import canonical


def process(pid, command, parent=1):
    return {"pid": pid, "parent_pid": parent, "created_at_utc": "2026-10-25T02:31:42.123456Z",
            "command_line": command}


def fixture():
    from tests_v2.test_evaluation_resources_v6 import fixture as old_fixture
    profile, live, health, gpu = old_fixture()
    profile.update(schema_version=5, kind=resource.KIND)
    profile["services"][0]["processes"][0].update(
        pid=55612, parent_pid=40508, created_at_utc="2026-10-29T02:34:38.4723310Z")
    dummy = {"path": str(Path.cwd() / "unused.json"), "sha256": "a" * 64}
    profile["service_identity_succession"] = {
        "restoration_report": copy.deepcopy(dummy), "restoration_manifest": copy.deepcopy(dummy)}
    return profile, live, health, gpu


def pin_profile(root, profile, health):
    """Create a complete four-restoration fixture chain without real evidence access."""
    from tests_v2.test_evaluation_resources_v6 import fixture as old_fixture, pin_profile as pin_old
    root = Path(root)
    def pin(name, value):
        path = root / name
        data = canonical(value)
        path.write_bytes(data)
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    prior, _, prior_health, _ = old_fixture()
    prior_root = root / "prior-v6"
    prior_root.mkdir()
    prior_binding = pin_old(prior_root, prior, prior_health)
    # Preserve the actual old profile at a distinct path before writing the new one.
    prior_binding = pin("fixture-prior-v6-profile.json", prior)
    preserved = prior["background_processes"] + prior["services"][1]["processes"]
    manifest = {"kind": "alignlab-evidoc-restoration-manifest-v2", "deadline_epoch": resource.DEADLINE,
        "baseline_profile": prior["authorized_pause"]["baseline_profile"],
        "original_restoration": prior["restoration_report"],
        "service_identity_succession": prior["service_identity_succession"],
        "inherited_authorization": resource._read(prior["authorized_pause"]["authorization"])["inherited_authorization"],
        "target": prior["services"][0]["processes"][0], "preserved_processes": preserved,
        "preserved_services": [prior["services"][1]],
        "service": {"name": "evidoc", "port": 8083, "url": prior["services"][0]["url"],
                    "json_equals": prior_health[prior["services"][0]["url"]]}}
    manifest_binding = pin("fixture-prior-restoration-manifest.json", manifest)
    receipt = {"kind": "alignlab-evidoc-restoration-complete-v1", "status": "restored_and_verified",
        "restoration_complete": True, "restore_pending": False, "all_28_fields_match": True,
        "deadline_epoch": resource.DEADLINE, "manifest_sha256": manifest_binding["sha256"],
        "pause_authorization_sha256": prior["authorized_pause"]["authorization"]["sha256"],
        "preserved_processes": preserved, "processes": [
            {**p, "created_utc_ticks": int(resource._created_ticks(p))}
            for p in profile["services"][0]["processes"]],
        "health": health[profile["services"][0]["url"]]}
    profile["restoration_report"] = prior["restoration_report"]
    profile["service_identity_succession"] = {
        "restoration_report": pin("fixture-prior-restoration-complete.json", receipt),
        "restoration_manifest": manifest_binding}
    profile["authorized_pause"]["baseline_profile"] = prior_binding
    prior_authority = resource._read(prior["authorized_pause"]["authorization"])
    authority = {"kind": resource.PAUSE_KIND, "status": "authorized", "authorized": True,
        "user_authorization_quote": resource.PAUSE_QUOTE, "service": "evidoc", "port": 8083,
        "baseline_profile": prior_binding, "restoration_report": profile["restoration_report"],
        "deadline_epoch": resource.DEADLINE, "restore_required": True,
        "evaluation_authorization": prior_authority["evaluation_authorization"],
        "inherited_authorization": prior["authorized_pause"]["authorization"],
        "service_identity_succession": profile["service_identity_succession"]}
    profile["authorized_pause"]["authorization"] = pin("fixture-continuation-authority.json", authority)
    return pin("fixture-resource-profile-v7.json", profile)

class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.live, self.health, self.gpu = fixture()
        clock = patch.object(resource.time, "time", return_value=1793240000)
        clock.start()
        self.addCleanup(clock.stop)

    def observe(self, **kwargs):
        return resource.collect_resources(self.profile, inventory_reader=lambda: copy.deepcopy(self.gpu),
            ram_reader=lambda: 12.0, process_reader=lambda: copy.deepcopy(self.live),
            health_reader=lambda url, timeout: copy.deepcopy(self.health[url]),
            port_reader=lambda: [], **kwargs)

    def test_authorized_pause_preserves_3840_preload_requirement(self):
        result = self.observe()
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["required_free_mib"], 3840)

    def test_lower_preload_free_and_high_utilization_block(self):
        for field, value in (("free_mb", 3839), ("utilization_percent", 21)):
            with self.subTest(field=field):
                previous = self.gpu["gpus"][0][field]
                self.gpu["gpus"][0][field] = value
                self.assertTrue(self.observe()["blockers"])
                self.gpu["gpus"][0][field] = previous

    def test_active_mode_preserves_global_floor_without_empty_card_threshold(self):
        self.gpu["gpus"][0].update(free_mb=1024, utilization_percent=100)
        self.health[self.profile["services"][0]["url"]]["busy"] = True
        self.assertEqual(self.observe(active=True)["blockers"], [])
        self.gpu["gpus"][0]["free_mb"] = 1023
        self.assertTrue(self.observe(active=True)["blockers"])

    def test_measured_peak_controls_formal_preload_but_cannot_raise_limit(self):
        self.assertEqual(self.observe(expected_peak_reserved_bytes=2000 * resource.MIB)["required_free_mib"], 3280)
        for peak in (0, 2560 * resource.MIB + 1, float("nan")):
            with self.assertRaises(ValueError):
                self.observe(expected_peak_reserved_bytes=peak)

    def test_restored_service_pid_reuse_or_disappearance_rejected(self):
        self.live[0] = {**self.live[0], "created_at_utc": "2026-10-25T03:00:00Z"}
        self.assertTrue(self.observe()["blockers"])
        self.live.pop(0)
        self.assertTrue(self.observe()["blockers"])

    def test_process_creation_keeps_windows_submicrosecond_precision(self):
        left = process(22, "python server.py")
        right = {**left, "created_at_utc": "2026-10-25T02:31:42.1234561Z"}
        self.assertNotEqual(resource.signature(left), resource.signature(right))
        right["created_at_utc"] = "2026-10-25T02:31:42.1234560Z"
        self.assertEqual(resource.signature(left), resource.signature(right))

    def test_health_identity_and_unknown_training_rejected(self):
        self.health[self.profile["services"][1]["url"]]["mode"] = "online"
        self.assertTrue(self.observe()["blockers"])
        self.health[self.profile["services"][1]["url"]]["mode"] = "offline"
        self.live.append(process(19, "python run_training.py"))
        self.assertTrue(self.observe()["blockers"])

    def test_exact_owned_evaluator_is_allowed_but_pid_only_is_not(self):
        child = process(25, "python run_alignlab_evaluation_stage_v2.py --request frozen.json")
        self.live.append(child)
        self.assertEqual(self.observe(allowed_processes=[child])["blockers"], [])
        self.live[-1] = {**child, "command_line": "python run_training.py"}
        self.assertTrue(self.observe(allowed_processes=[child])["blockers"])

    def test_unknown_benign_python_is_not_implicitly_allowed(self):
        self.live.append(process(31, "python benign.py"))
        self.assertTrue(any("Unknown Python" in item for item in self.observe()["blockers"]))

    def test_new_pid_cannot_inherit_a_frozen_service_command(self):
        self.live.append({**self.live[0], "pid": 32})
        self.assertTrue(any("Unknown Python" in item for item in self.observe()["blockers"]))

    def test_exact_owned_codecs_are_allowed_without_parent_or_command_whitelist(self):
        evaluator = process(40, "python run_alignlab_evaluation_stage_v2.py --request pinned")
        encoder = process(41, "python -I -B scripts/encode_alignlab_evaluation_capacity_v2.py", parent=40)
        decoder = process(42, "python -I -B scripts/decode_alignlab_evaluation_v2.py", parent=40)
        self.live.extend([evaluator, encoder, decoder])
        owned = [evaluator, encoder, decoder]
        self.assertEqual(self.observe(allowed_processes=owned)["blockers"], [])
        for extra in ({**encoder, "pid": 43}, {**decoder, "pid": 44}, process(45, "python benign.py", parent=40)):
            with self.subTest(pid=extra["pid"]):
                self.live.append(extra)
                self.assertTrue(any("Unknown Python" in item
                                    for item in self.observe(allowed_processes=owned)["blockers"]))
                self.live.pop()

    def test_unknown_uninspectable_python_is_not_whitelisted(self):
        self.live.append({"pid": 77, "parent_pid": 1, "created_at_utc": "2026-10-25T00:00:00Z", "command_line": None})
        self.assertTrue(self.observe()["blockers"])

    def test_profile_pin_requires_verified_restoration_and_all_stable_health(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = pin_profile(root, self.profile, self.health)
            self.assertEqual(resource.load_profile(binding), self.profile)
            self.profile["services"][0]["json_equals"].pop("status")
            with self.assertRaisesRegex(ValueError, "unchanged|All stable"):
                resource.load_profile(self.pin(root / "profile2.json", self.profile))

    @staticmethod
    def pin(path, value):
        data = canonical(value)
        path.write_bytes(data)
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}

    def test_allocator_is_actually_set_and_read_back_before_use(self):
        calls = []
        cuda = SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(total_memory=8192 * resource.MIB),
            set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)),
            get_per_process_memory_fraction=lambda index: calls[-1][0])
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            receipt = resource.configure_cuda_allocator(self.profile)
        self.assertEqual(calls, [(0.3125, 0)])
        resource.verify_allocator_receipt(receipt, self.profile)
        receipt["limit_bytes"] += 1
        with self.assertRaises(ValueError):
            resource.verify_allocator_receipt(receipt, self.profile)

    def test_existing_context_reload_keeps_full_allocator_and_service_reserve(self):
        receipt = {"kind": "alignlab-cuda-allocator-ceiling-v2", "device": "cuda:0",
                   "total_memory_bytes": 8192 * resource.MIB, "limit_bytes": 2560 * resource.MIB,
                   "fraction": 0.3125, "configured_before_model_load": True}
        self.gpu["gpus"][0]["free_mb"] = 3584
        self.assertTrue(self.observe()["blockers"])
        after = self.observe(allocator_receipt=receipt)
        self.assertEqual(after["blockers"], [])
        self.assertEqual(after["required_free_mib"], 3584)
        self.gpu["gpus"][0]["free_mb"] = 3583
        self.assertTrue(self.observe(allocator_receipt=receipt)["blockers"])

    def test_runtime_resource_and_deadline_faults_fail_closed(self):
        system = SimpleNamespace(now=lambda: 50)
        observations = self.observe(active=True)
        guard = resource.RuntimeResourceGuard(self.profile, system=system, deadline_epoch=100,
            allowed_processes=[], collector=lambda *a, **k: observations,
            memory_reader=lambda: {"reserved_bytes": 10, "max_reserved_bytes": 20})
        self.assertFalse(guard(force=True))
        observations["blockers"] = ["evidoc unavailable"]
        with self.assertRaisesRegex(RuntimeError, "evidoc unavailable"):
            guard(force=True)
        system.now = lambda: 100
        with self.assertRaises(TimeoutError):
            guard()

    def test_runtime_allocator_peak_exceedance_is_not_success(self):
        guard = resource.RuntimeResourceGuard(self.profile, system=SimpleNamespace(now=lambda: 50),
            deadline_epoch=100, allowed_processes=[], collector=lambda *a, **k: self.observe(active=True),
            memory_reader=lambda: {"reserved_bytes": 10, "max_reserved_bytes": 2560 * resource.MIB + 1})
        with self.assertRaisesRegex(RuntimeError, "allocator ceiling"):
            guard(force=True)

    def test_pause_does_not_whitelist_old_pid_or_a_replacement(self):
        old = self.profile["services"][0]["processes"][0]
        self.live.append(old)
        self.assertTrue(any("must be absent" in reason for reason in self.observe()["blockers"]))
        with self.assertRaisesRegex(ValueError, "cannot be granted"):
            self.observe(allowed_processes=[old])
        self.live[-1] = {**old, "pid": 999}
        self.assertTrue(any("Unknown Python" in reason for reason in self.observe()["blockers"]))
        self.assertNotIn(55612, {p["pid"] for p in resource.active_service_processes(self.profile)})

    def test_any_listener_including_non_python_replacement_blocks(self):
        def observe(reader):
            return resource.collect_resources(self.profile, inventory_reader=lambda: self.gpu,
                ram_reader=lambda: 12.0, process_reader=lambda: self.live,
                health_reader=lambda url, timeout: self.health[url], port_reader=reader)
        replacement = [{"local_address": "0.0.0.0", "local_port": 8083, "pid": 9999}]
        self.assertTrue(any("has a listener" in reason
                            for reason in observe(lambda: replacement)["blockers"]))
        def unavailable():
            raise OSError("TCP observation unavailable")
        self.assertTrue(any("unobservable" in reason
                            for reason in observe(unavailable)["blockers"]))
        self.assertTrue(any("unobservable" in reason
                            for reason in observe(lambda: None)["blockers"]))

    def test_paused_health_is_absence_evidence_not_fabricated_ready(self):
        calls = []
        def health(url, timeout):
            calls.append(url)
            return self.health[url]
        result = resource.collect_resources(self.profile, inventory_reader=lambda: self.gpu,
            ram_reader=lambda: 12.0, process_reader=lambda: self.live,
            health_reader=health, port_reader=lambda: [])
        self.assertEqual(calls, ["http://127.0.0.1:8765/api/health"])
        paused = result["service_health"]["evidoc"]
        self.assertTrue(paused["process_absent"])
        self.assertTrue(paused["port_8083_absent"])
        self.assertNotIn("observed", paused)
        self.assertNotIn("matched", paused)

    def test_no_pause_schema_or_changed_limits_are_not_admitted(self):
        for key in resource.LIMITS:
            profile = copy.deepcopy(self.profile)
            profile[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                resource.validate_profile(profile)
        for mutation in (lambda p: p.pop("authorized_pause"),
                         lambda p: p["authorized_pause"].update(service="shopassist"),
                         lambda p: p["authorized_pause"].update(deadline_epoch=resource.DEADLINE + 1),
                         lambda p: p["services"][0]["processes"][0].update(pid=0),
                         lambda p: p["background_processes"].pop(),
                         lambda p: p["services"][1]["processes"].pop()):
            profile = copy.deepcopy(self.profile)
            mutation(profile)
            with self.assertRaises(ValueError):
                resource.validate_profile(profile)

    def test_missing_revoked_or_wrong_scope_pause_authorization_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_profile(root, self.profile, self.health)
            path = Path(self.profile["authorized_pause"]["authorization"]["path"])
            original = resource.parse(path.read_bytes())
            changes = [("authorized", False), ("status", "proposed"), ("restore_required", False),
                       ("service", "shopassist"), ("port", 8765),
                       ("user_authorization_quote", "想办法开始"),
                       ("deadline_epoch", resource.DEADLINE + 1)]
            for key, value in changes:
                with self.subTest(key=key):
                    modified = {**original, key: value}
                    self.profile["authorized_pause"]["authorization"] = self.pin(path, modified)
                    binding = self.pin(root / "modified-profile.json", self.profile)
                    with self.assertRaisesRegex(ValueError, "Explicit user authorization"):
                        resource.load_profile(binding)
            self.profile["authorized_pause"]["authorization"] = {
                "path": str(root / "missing.json"), "sha256": "1" * 64}
            with self.assertRaises((OSError, ValueError)):
                resource.load_profile(self.pin(root / "missing-profile.json", self.profile))

    def test_baseline_other_services_and_stable_health_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_profile(root, self.profile, self.health)
            for mutation in (
                    lambda p: p["background_processes"][0].update(command_line="python changed.py"),
                    lambda p: p["services"][1]["processes"][0].update(parent_pid=777),
                    lambda p: p["services"][0]["json_equals"].update(status="changed")):
                changed = copy.deepcopy(self.profile)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, "must remain unchanged|stable restored"):
                    resource.load_profile(self.pin(root / "modified.json", changed))

    def test_prior_profile_and_authorization_evidence_pins_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = pin_profile(root, self.profile, self.health)
            prior = self.profile["authorized_pause"]["baseline_profile"]
            self.assertEqual(resource.validate_authorized_profile_relation(prior, binding), self.profile)
            with self.assertRaisesRegex(ValueError, "prior plan"):
                resource.validate_authorized_profile_relation({**prior, "sha256": "f" * 64}, binding)
            baseline_path = Path(prior["path"])
            baseline_path.write_bytes(baseline_path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                resource.load_profile(binding)

    def test_original_budget_cannot_be_rebased_or_extended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_profile(root, self.profile, self.health)
            auth_path = Path(self.profile["authorized_pause"]["authorization"]["path"])
            authority = resource.parse(auth_path.read_bytes())
            budget_path = Path(authority["evaluation_authorization"]["path"])
            budget = resource.parse(budget_path.read_bytes())
            for key, value in (("started_at_epoch", resource.STARTED + 1),
                               ("deadline_epoch", resource.DEADLINE + 1),
                               ("wall_seconds", resource.WALL_SECONDS + 1),
                               ("automatic_extension", True)):
                with self.subTest(key=key):
                    authority["evaluation_authorization"] = self.pin(budget_path, {**budget, key: value})
                    self.profile["authorized_pause"]["authorization"] = self.pin(auth_path, authority)
                    with self.assertRaises(ValueError):
                        resource.load_profile(self.pin(root / "modified-profile.json", self.profile))

    def test_expired_authorization_blocks_static_and_live_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            with self.assertRaisesRegex(ValueError, "outside the original"):
                resource.load_profile(binding, now_fn=lambda: resource.DEADLINE)
            with self.assertRaisesRegex(ValueError, "outside the original"):
                resource.load_profile(binding, now_fn=lambda: resource.STARTED - 1)
            result = self.observe(now_fn=lambda: resource.DEADLINE)
            self.assertTrue(any("authorization expired" in reason for reason in result["blockers"]))

    def test_static_freeze_does_not_require_service_already_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = pin_profile(Path(directory), self.profile, self.health)
            with patch.object(resource, "_port_8083_listeners", side_effect=AssertionError("No live query")), \
                 patch.object(resource, "_health", side_effect=AssertionError("No live query")):
                self.assertEqual(resource.load_profile(binding), self.profile)

    def test_successor_requires_the_exact_completed_restoration_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_profile(root, self.profile, self.health)
            for key, value in (("pid", 55613), ("parent_pid", 40509),
                               ("created_at_utc", "2026-10-29T02:34:38.4723311Z"),
                               ("command_line", "python -m scripts.serve_adapter --port 8083 --changed")):
                changed = copy.deepcopy(self.profile)
                changed["services"][0]["processes"][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact identity"):
                    resource.load_profile(self.pin(root / "changed-successor.json", changed))

    def test_old_adapter_cannot_inherit_new_pause_or_evaluator_ownership(self):
        retired = process(828, self.profile["services"][0]["processes"][0]["command_line"])
        self.live.append(retired)
        self.assertTrue(any("Unknown Python" in reason for reason in self.observe()["blockers"]))
        successor = self.profile["services"][0]["processes"][0]
        with self.assertRaisesRegex(ValueError, "cannot be granted"):
            self.observe(allowed_processes=[successor])

    def test_incomplete_or_unrelated_restoration_cannot_authorize_succession(self):
        mutations = (
            lambda r: r.update(restoration_complete=False),
            lambda r: r.update(restore_pending=True),
            lambda r: r.update(all_28_fields_match=False),
            lambda r: r.update(deadline_epoch=resource.DEADLINE + 1),
            lambda r: r.update(manifest_sha256="f" * 64),
            lambda r: r.update(pause_authorization_sha256="f" * 64),
            lambda r: r["preserved_processes"][0].update(parent_pid=999),
            lambda r: r["health"].update(dtype="fp32"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile, _, health, _ = fixture()
                pin_profile(root, profile, health)
                succession = profile["service_identity_succession"]
                receipt = resource._read(succession["restoration_report"])
                mutate(receipt)
                succession["restoration_report"] = self.pin(root / "changed-receipt.json", receipt)
                auth = resource._read(profile["authorized_pause"]["authorization"])
                auth["service_identity_succession"] = copy.deepcopy(succession)
                profile["authorized_pause"]["authorization"] = self.pin(root / "changed-auth.json", auth)
                with self.assertRaises(ValueError):
                    resource.load_profile(self.pin(root / "changed-profile.json", profile))

    def test_new_authority_must_retain_original_pause_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_profile(root, self.profile, self.health)
            authority = resource._read(self.profile["authorized_pause"]["authorization"])
            authority["inherited_authorization"] = {"path": str(root / "foreign.json"), "sha256": "f" * 64}
            self.profile["authorized_pause"]["authorization"] = self.pin(root / "changed-auth.json", authority)
            with self.assertRaisesRegex(ValueError, "Explicit user authorization"):
                resource.load_profile(self.pin(root / "changed-profile.json", self.profile))

    def test_v6_restoration_receipt_exact_ticks_survive_integer_and_string_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = pin_profile(root, self.profile, self.health)
            self.assertEqual(resource.load_profile(binding), self.profile)
            restored = self.profile["services"][0]["processes"][0]
            self.assertEqual(resource._created_ticks(restored), "639288380784723310")
            succession = self.profile["service_identity_succession"]
            original = resource._read(succession["restoration_report"])
            for ticks in ("639288380784723310", "639288380784723311",
                          639288380784723311, float(639288380784723310), None):
                receipt = copy.deepcopy(original)
                receipt["processes"][0]["created_utc_ticks"] = ticks
                succession["restoration_report"] = self.pin(root / "ticks-receipt.json", receipt)
                authority = resource._read(self.profile["authorized_pause"]["authorization"])
                authority["service_identity_succession"] = copy.deepcopy(succession)
                self.profile["authorized_pause"]["authorization"] = self.pin(root / "ticks-authority.json", authority)
                binding = self.pin(root / "ticks-profile.json", self.profile)
                with self.subTest(ticks=ticks):
                    if ticks == "639288380784723310":
                        self.assertEqual(resource.load_profile(binding), self.profile)
                    else:
                        with self.assertRaisesRegex(ValueError, "exact integer/string UTC"):
                            resource.load_profile(binding)

    def test_unknown_snapshot_preserves_complete_identity_without_granting_ownership(self):
        foreign = {"pid": 58104, "parent_pid": 12345,
            "created_at_utc": "2026-10-29T02:34:05.1234567Z",
            "command_line": '"C:\\fixture\\python.exe" -I script.py --value "two words"'}
        self.live.append(foreign)
        result = self.observe(active=True)
        self.assertEqual(len(result["process_identity_diagnostics"]), 1)
        detail = result["process_identity_diagnostics"][0]
        self.assertEqual(detail["reason"], "unknown_python")
        for field in resource.IDENTITY_FIELDS:
            self.assertEqual(detail["observed"][field], foreign[field])
        self.assertEqual(detail["observed"]["created_utc_ticks"], "639288380451234567")
        self.assertIs(type(detail["observed"]["created_utc_ticks"]), str)
        self.assertTrue(detail["observed"]["identity_complete"])
        self.assertIn('"parent_pid":12345', result["blockers"][0])
        foreign["command_line"] = "subsequent process-reader mutation"
        self.assertNotEqual(detail["observed"]["command_line"], foreign["command_line"])

    def test_uninspectable_unknown_snapshot_retains_missing_fields_and_fails_closed(self):
        unknown = {"pid": 58104, "parent_pid": 12345,
                   "created_at_utc": None, "command_line": None}
        self.live.append(unknown)
        result = self.observe(active=True)
        detail = result["process_identity_diagnostics"][0]
        self.assertEqual(detail["reason"], "uninspectable_python")
        self.assertFalse(detail["observed"]["identity_complete"])
        self.assertEqual(detail["observed"]["pid"], 58104)
        self.assertEqual(detail["observed"]["parent_pid"], 12345)
        self.assertIsNone(detail["observed"]["command_line"])
        self.assertIsNone(detail["observed"]["created_at_utc"])
        self.assertTrue(result["blockers"])

    def test_frozen_identity_mismatch_diagnostic_contains_expected_and_observed(self):
        original = copy.deepcopy(self.live[0])
        self.live[0] = {**self.live[0], "created_at_utc": "2026-10-25T02:31:42.1234567Z"}
        result = self.observe(active=True)
        detail = result["process_identity_diagnostics"][0]
        self.assertEqual(detail["reason"], "frozen_identity_changed")
        self.assertEqual(detail["expected"]["created_at_utc"], original["created_at_utc"])
        self.assertEqual(detail["observed"]["created_at_utc"], self.live[0]["created_at_utc"])
        self.assertTrue(result["blockers"])

    def test_runtime_unknown_fails_immediately_and_keeps_identity_after_process_disappears(self):
        unknown = process(58104, "python unknown.py --request actual.json", parent=12345)
        self.live.append(unknown)
        calls = []
        guard = resource.RuntimeResourceGuard(self.profile, system=SimpleNamespace(now=lambda: 50),
            deadline_epoch=100, allowed_processes=[],
            collector=lambda *args, **kwargs: self.observe(active=True),
            memory_reader=lambda: calls.append("memory") or {"reserved_bytes": 10, "max_reserved_bytes": 20})
        with self.assertRaisesRegex(RuntimeError, "Unknown Python"):
            guard(force=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(guard.observations), 1)
        self.live.pop()
        detail = guard.observations[0]["process_identity_diagnostics"][0]
        self.assertEqual(detail["observed"]["command_line"], unknown["command_line"])
        self.assertEqual(detail["observed"]["parent_pid"], 12345)
        self.assertTrue(guard.observations[0]["blockers"])

    def test_v6_manifest_must_retain_its_own_validated_predecessor_chain(self):
        changes = (
            lambda value: value.update(kind="alignlab-evidoc-restoration-manifest-v1"),
            lambda value: value.pop("service_identity_succession"),
            lambda value: value["service_identity_succession"]["restoration_report"].update(sha256="f" * 64),
            lambda value: value["inherited_authorization"].update(sha256="f" * 64),
            lambda value: value["target"].update(parent_pid=999),
        )
        for index, mutate in enumerate(changes):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile, _, health, _ = fixture()
                pin_profile(root, profile, health)
                succession = profile["service_identity_succession"]
                manifest = resource._read(succession["restoration_manifest"])
                mutate(manifest)
                succession["restoration_manifest"] = self.pin(root / "altered-manifest.json", manifest)
                receipt = resource._read(succession["restoration_report"])
                receipt["manifest_sha256"] = succession["restoration_manifest"]["sha256"]
                succession["restoration_report"] = self.pin(root / "altered-receipt.json", receipt)
                authority = resource._read(profile["authorized_pause"]["authorization"])
                authority["service_identity_succession"] = copy.deepcopy(succession)
                profile["authorized_pause"]["authorization"] = self.pin(root / "altered-authority.json", authority)
                with self.assertRaisesRegex(ValueError, "Restoration manifest"):
                    resource.load_profile(self.pin(root / "altered-profile.json", profile))


if __name__ == "__main__":
    unittest.main()
