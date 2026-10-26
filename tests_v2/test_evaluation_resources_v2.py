"""CPU-only resource fault injection; never a capacity or quality claim."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_resources_v2 as resource
from alignment_lab.evaluation_protocol_v2 import canonical


def process(pid, command, parent=1):
    return {"pid": pid, "parent_pid": parent, "created_at_utc": "2026-10-25T02:31:42.123456Z",
            "command_line": command}


def fixture():
    evidoc = {"status": "ok", "ready": True, "device": "cuda:0", "dtype": "bf16", "quantization": "none"}
    shop = {"status": "ok", "mode": "offline", "version": "0.2.0", "gpu_required": False}
    services = [
        {"name": "evidoc", "url": "http://127.0.0.1:8083/health", "processes": [process(11, "python -m scripts.serve_adapter --port 8083")], "json_equals": evidoc},
        {"name": "shopassist", "url": "http://127.0.0.1:8765/api/health", "processes": [process(12, "python -m shopassist --port 8765")], "json_equals": shop}]
    profile = {"schema_version": 1, "kind": resource.KIND, **resource.LIMITS, "services": services,
               "background_processes": [process(13, "python api.py --port 8777")],
               "restoration_report": {"path": str(Path.cwd() / "unused.json"), "sha256": "a" * 64}}
    live = [*services[0]["processes"], *services[1]["processes"], *profile["background_processes"]]
    health = {services[0]["url"]: {**evidoc, "busy": False}, services[1]["url"]: shop}
    gpu = {"gpus": [{"index": 0, "total_mb": 8192, "free_mb": 3953, "utilization_percent": 0}], "error": None}
    return profile, live, health, gpu


class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.live, self.health, self.gpu = fixture()

    def observe(self, **kwargs):
        return resource.collect_resources(self.profile, inventory_reader=lambda: copy.deepcopy(self.gpu),
            ram_reader=lambda: 12.0, process_reader=lambda: copy.deepcopy(self.live),
            health_reader=lambda url, timeout: copy.deepcopy(self.health[url]), **kwargs)

    def test_current_service_coexistence_has_3840_preload_requirement(self):
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
        self.health[self.profile["services"][0]["url"]]["dtype"] = "fp16"
        self.assertTrue(self.observe()["blockers"])
        self.health[self.profile["services"][0]["url"]]["dtype"] = "bf16"
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
            restoration = {"status": "restored_and_verified", "restoration_complete": True}
            for service in self.profile["services"]:
                restoration[service["name"]] = {"processes": service["processes"], "health": self.health[service["url"]]}
            def pin(name, value):
                path = root / name
                data = canonical(value)
                path.write_bytes(data)
                return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
            self.profile["restoration_report"] = pin("restoration.json", restoration)
            binding = pin("profile.json", self.profile)
            self.assertEqual(resource.load_profile(binding), self.profile)
            self.profile["services"][0]["json_equals"].pop("status")
            with self.assertRaisesRegex(ValueError, "All stable"):
                resource.load_profile(pin("profile2.json", self.profile))

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


if __name__ == "__main__":
    unittest.main()
