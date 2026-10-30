"""CPU-only lifecycle fixtures: no model, real subprocess, CUDA, or scoring.

The existing numerical/official-score validators have independent tests. Here
their calls are replaced with file-backed fixture validators while the new
source pins, request generation, process ownership, locks and clock run for real.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_supervisor_v10 as es
from alignment_lab import evaluation_analysis_v10 as analysis
from alignment_lab import evaluation_acceptance_v2 as acceptance
from alignment_lab import evaluation_execution_v10 as inference
from alignment_lab import evaluation_capacity_v6 as capacity
from tests_v2.test_evaluation_resources_v10 import fixture as resource_fixture
from scripts import prepare_alignlab_evaluation_v10 as preparation
from alignment_lab import evaluation_inheritance_v10 as inheritance
from alignment_lab import evaluation_capacity_inheritance_v10 as capacity_inheritance


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(es.canonical(value))


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class Child:
    def __init__(self, system, args, index):
        self.system, self.args, self.index = system, args, index
        self.pid = 70000 + index * 3
        self.started = system.now()
        self.done = False
        self.code = None

    def poll(self):
        if not self.done and self.system.now() >= self.started + 10:
            self.done = True
            self.code = 7 if self.index == self.system.fail_index else 0
            if self.code == 0:
                self.system.fixture.complete(self)
        return self.code


class System:
    def __init__(self, fixture):
        self.fixture = fixture
        self.clock = 2_000_000_000.0
        self.children = []
        self.fail_index = None
        self.initial_busy = False
        self.after_busy = False
        self.on_sleep = None
        self.on_spawn = None
        self.extra_processes = []
        self.wrapper_survives = False
        self.pid_drift = False
        self.hide_children = False
        self.decoder_child = False
        self.encoder_child = False
        self.capacity_encoder_child = False
        self.encoder_extra_arg = False
        self.spawn_assertions = []

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds
        if self.on_sleep:
            self.on_sleep(self)

    def resources(self):
        blocked = self.initial_busy if not self.children else self.after_busy
        return {"blockers": ["fixture busy"] if blocked else [], "available_ram_gb": 20}

    def processes(self):
        args = [str(self.fixture.root / es.MODEL_PYTHON), "-I", "-B", "-u", es.ENTRYPOINT,
                "--plan", str(self.fixture.plan_path), "--plan-sha256", self.fixture.pin, "--execute"]
        command = subprocess.list2cmdline(args)
        records = [
            {"pid": os.getpid(), "parent_pid": 69000, "created_at_utc": stamp(1_999_999_000),
             "command_line": command},
            {"pid": 69000, "parent_pid": 1, "created_at_utc": stamp(1_999_999_000),
             "command_line": command},
        ]
        if not self.hide_children:
            for child in self.children:
                if child.done and not self.wrapper_survives:
                    continue
                record = {"pid": child.pid, "parent_pid": os.getpid(),
                    "created_at_utc": stamp(child.started),
                    "command_line": subprocess.list2cmdline(child.args)}
                if self.pid_drift and self.now() > child.started:
                    record["created_at_utc"] = stamp(child.started + 1)
                records.append(record)
                if not child.done:
                    records.append({**record, "pid": child.pid + 1, "parent_pid": child.pid})
                    if self.decoder_child and child.args[4] in (es.SCORE_ENTRYPOINT, es.ANALYSIS_ENTRYPOINT):
                        records.append({**record, "pid": child.pid + 2, "parent_pid": child.pid + 1,
                            "command_line": subprocess.list2cmdline([
                                str(self.fixture.root / es.MODEL_PYTHON), "-I", "-B",
                                str(self.fixture.root / es.DECODER_ENTRYPOINT)])})
                    if ((self.encoder_child and child.args[4] == es.INFERENCE_ENTRYPOINT)
                            or (self.capacity_encoder_child and child.args[4] == "unused-capacity-entrypoint")):
                        encoder_args = [str(self.fixture.root / es.MODEL_PYTHON), "-I", "-B",
                                        str(self.fixture.root / es.ENCODER_ENTRYPOINT)]
                        if self.encoder_extra_arg:
                            encoder_args.append("--unexpected")
                        records.append({**record, "pid": child.pid + 2, "parent_pid": child.pid + 1,
                                        "command_line": subprocess.list2cmdline(encoder_args)})
        return records + copy.deepcopy(self.extra_processes)

    def spawn(self, args, stdout, stderr):
        self.spawn_assertions.append((stdout.name, stderr.name))
        if any(not child.done for child in self.children):
            raise AssertionError("Fixture detects concurrent child dispatch")
        child = Child(self, list(args), len(self.children))
        self.children.append(child)
        if self.on_spawn:
            self.on_spawn(self, child)
        return child


class EvaluationSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "reports").mkdir()
        (self.root / "runs").mkdir()
        self.patch = patch.object(es, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.source_names = (analysis.REQUIRED_SOURCES | set(acceptance.SOURCES) | es.NEW_SOURCES
                             | {"scripts/verify_alignlab_cpu.py"})
        for name in self.source_names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# tiny frozen lifecycle fixture\n", encoding="utf-8")
        for name in (es.MODEL_PYTHON, es.SCORE_PYTHON, "models/tokenizer.json"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not an executable; never launched")
        self.sources = {name: es._sha(self.root / name) for name in sorted(self.source_names)}
        self.cpu = self.root / "reports/cpu.json"
        write(self.cpu, {"status": "cpu_software_checks_passed", "device": "cpu",
            "source_unchanged": True, "cuda_initialized": False, "tests_run": 1, "torch_imported": False, "pretrained_model_weights_used": False,
            "test_modules": sorted(preparation.RECOVERY_TEST_MODULES),
            "failures": 0, "errors": 0, "skipped": 0, "source_sha256": self.sources})
        self.protocol, self.roster = self.root / "reports/protocol.json", self.root / "reports/roster.json"
        write(self.protocol, {"fixture": "unscored protocol"})
        write(self.roster, {"fixture": "unloaded roster"})
        prerequisite = self.root / "reports/other-project-STOP"
        prerequisite.write_text("owned waiting marker", encoding="utf-8")
        self.output = self.root / "runs/primary"
        self.plan = {"schema_version": 1, "kind": es.KIND, "output": str(self.output),
            "wall_seconds": 1000, "analysis_wall_seconds": 100, "poll_seconds": 10,
            "source_sha256": self.sources, "cpu_check_report": es._pin(self.cpu),
            "protocol": es._pin(self.protocol), "roster": es._pin(self.roster),
            "model_python_sha256": es._sha(self.root / es.MODEL_PYTHON),
            "scoring_python_sha256": es._sha(self.root / es.SCORE_PYTHON),
            "decoder": {"python": str(self.root / es.MODEL_PYTHON),
                "python_sha256": es._sha(self.root / es.MODEL_PYTHON),
                "script": str(self.root / es.DECODER_ENTRYPOINT),
                "script_sha256": self.sources[es.DECODER_ENTRYPOINT],
                "tokenizer_path": str(self.root / "models/tokenizer.json"),
                "tokenizer_sha256": es._sha(self.root / "models/tokenizer.json"),
                "tokenizers_version": "fixture"},
            "prerequisites": [{**es._pin(prerequisite), "json_equals": {}}], "inherited_inference": {}, "inherited_capacity": None}
        self.plan_path = self.root / "configs/plan.json"
        self.profile_path = self.root / "reports/resource-profile.json"
        self.authority_path = self.root / "reports/authorization.json"
        write(self.profile_path, {"fixture": "exact services tested by resource unit suite"})
        self.plan["resource_profile"] = es._pin(self.profile_path)
        pins = {}
        for field in ("prior_plan", "campaign", "closure"):
            path = self.root / "reports" / (field + ".json")
            write(path, {"fixture": field})
            pins[field] = es._pin(path)
        entries = {}
        for slot in sorted(es.INHERITED_SLOTS):
            entry = copy.deepcopy(pins)
            for field in ("request", "manifest", "score_request", "score"):
                path = self.root / "reports" / (slot.replace("/", "-") + "-" + field + ".json")
                write(path, {"fixture": field, "slot": slot})
                entry[field] = es._pin(path)
            entries[slot] = entry
        self.plan["inherited_inference"] = entries
        self.plan["inherited_scores"] = copy.deepcopy(entries)
        bridge = self.root / "reports/bridge.json"
        write(bridge, {"fixture": "full capacity bridge"})
        self.plan["inherited_capacity"] = {**pins, "bridge": es._pin(bridge)}
        for module, name, replacement in (
                (inheritance, "verify_inherited_inference", self.verify_inherited),
                (capacity_inheritance, "verify_inherited_capacity", self.verify_capacity_bridge)):
            change = patch.object(module, name, side_effect=replacement)
            change.start()
            self.addCleanup(change.stop)
        self.save_plan()
        self.system = System(self)
        self.resource_fixture, _, _, _ = resource_fixture()
        for module in (es, inference):
            change = patch.object(module, "load_profile", return_value=self.resource_fixture)
            change.start()
            self.addCleanup(change.stop)
        self.validations = []
        self.full_patch = patch.object(es, "_full_inputs", return_value=SimpleNamespace())
        self.full_patch.start()
        self.addCleanup(self.full_patch.stop)
        for target, replacement in (("_verify_inference", self.verify_inference),
                                    ("_verify_score", self.verify_score)):
            change = patch.object(es, target, side_effect=replacement)
            change.start()
            self.addCleanup(change.stop)
        change = patch.object(analysis, "verify_analysis", side_effect=self.verify_analysis)
        change.start()
        self.addCleanup(change.stop)
        module = SimpleNamespace(verify_evaluation_capacity=self.verify_capacity)
        change = patch.dict(sys.modules, {"alignment_lab.evaluation_capacity_v6": module})
        change.start()
        self.addCleanup(change.stop)

    def verify_inherited(self, binding, plan, *, system=None):
        slot = es._read(binding["request"])["slot"]
        self.assertEqual(binding, plan["inherited_inference"][slot])
        self.assertEqual(binding, plan["inherited_scores"][slot])
        self.validations.append("inherited")
        return {"slot": slot, "request": es._read(binding["request"]),
                "manifest_binding": binding["manifest"], "score_binding": binding["score"],
                "verification": {"official_scoring_complete": True}}

    def verify_capacity_bridge(self, binding, plan):
        self.assertEqual(binding, plan["inherited_capacity"])
        self.validations.append("capacity_bridge")
        return {"capacity_verified": True, "fixture": True}

    def save_plan(self):
        self.plan["started_at_epoch"] = 2_000_000_000
        self.plan["deadline_epoch"] = self.plan["started_at_epoch"] + self.plan["wall_seconds"]
        write(self.authority_path, {
            "kind": "alignlab-independent-quality-evaluation-authorization-v2",
            "started_at_epoch": self.plan["started_at_epoch"],
            "deadline_epoch": self.plan["deadline_epoch"],
            "wall_seconds": self.plan["wall_seconds"], "automatic_extension": False,
            "reuses_old_budget": False, "no_new_training": True})
        self.plan["authorization"] = es._pin(self.authority_path)
        write(self.plan_path, self.plan)
        self.pin = es._sha(self.plan_path)

    def run_campaign(self, execute=True):
        return es.run_evaluation(self.plan_path, self.pin, execute=execute, system=self.system)

    def test_preparation_elapsed_time_is_charged_to_frozen_window(self):
        self.system.clock += 1001
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.children, [])

    def test_paused_evidoc_identity_is_not_an_allowed_resident(self):
        paused = next(service for service in self.resource_fixture["services"]
                      if service["name"] == "evidoc")["processes"][0]
        self.system.extra_processes.append(copy.deepcopy(paused))
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIn(f"Unknown Python process identity: {paused['pid']}", result["blockers"])
        self.assertEqual(self.system.children, [])
        self.assertFalse(self.output.exists())

    def test_changed_resource_profile_and_authorization_are_rejected(self):
        self.profile_path.write_text("changed profile", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_campaign()
        self.assertEqual(self.system.children, [])

    def complete(self, child):
        request_path = Path(child.args[child.args.index("--request") + 1])
        expected = child.args[child.args.index("--request-sha256") + 1]
        request = es._read({"path": str(request_path), "sha256": expected})
        script = child.args[4]
        output = Path(request["output"]) if "--execute" in child.args else Path(
            child.args[child.args.index("--output") + 1])
        output.mkdir()
        value = {"request": request, "pid": child.pid + 1, "fixture": True}
        if script == "unused-capacity-entrypoint":
            value["process_identity"] = [{"pid": child.pid + 1, "parent_pid": child.pid,
                "created_at_utc": stamp(child.started), "command_line": subprocess.list2cmdline(child.args)}]
        if script == es.ANALYSIS_ENTRYPOINT:
            value.update(primary_evaluation_complete=True, research_complete=False)
            name = "run_manifest.json"
        elif script == es.SCORE_ENTRYPOINT:
            name = "scores.json"
        else:
            name = "manifest.json"
        write(output / name, value)

    def verify_capacity(self, output, *, expected_manifest_sha256, expected_request=None, deadline_epoch=None):
        result = es._read({"path": str(output / "manifest.json"), "sha256": expected_manifest_sha256})
        self.assertEqual(result["request"], expected_request)
        self.assertEqual(deadline_epoch, expected_request["deadline_epoch"])
        self.validations.append("capacity")
        return result

    def verify_inference(self, output, request, known):
        result = es._read(es._pin(output / "manifest.json"))
        self.assertEqual(result["request"], request)
        self.assertIn(result["pid"], known)
        # The actual frozen inference CLI's exact request validator is exercised,
        # without calling its context/model/resource paths.
        with patch.object(inference, "ROOT", self.root):
            inference._request(request)
        self.validations.append("inference")
        return es._pin(output / "manifest.json")

    def verify_score(self, output, request, plan, inference_request, *, plan_binding):
        result = es._read(es._pin(output / "scores.json"))
        self.assertEqual(result["request"], request)
        self.assertEqual(request["deadline_epoch"], inference_request["deadline_epoch"])
        self.assertEqual(request["decoder"] is not None, inference_request["task"] == "ifeval")
        self.assertEqual(es._read({"path": request["inference_manifest"],
            "sha256": request["inference_manifest_sha256"]})["request"], inference_request)
        self.validations.append("score")
        return es._pin(output / "scores.json")

    def verify_analysis(self, output, sha):
        result = es._read({"path": str(output / "run_manifest.json"), "sha256": sha})
        request = result["request"]
        self.assertEqual(set(request["scores"]), {a + "/" + b for a, b in es.TASKS})
        for binding in request["scores"].values():
            es._read(binding)
        self.validations.append("analysis")
        return {"primary_evaluation_complete": True, "research_complete": False}

    def test_inheritance_does_not_create_children_or_write_original_output(self):
        originals = {slot: Path(pin["score"]["path"]).read_bytes()
                     for slot, pin in self.plan["inherited_scores"].items()}
        result = self.run_campaign()
        self.assertEqual(result["status"], "primary_evaluation_passed")
        self.assertEqual([c["name"] for c in result["children"]], [
            "ifeval-ppo44", "ifeval-ppo44-score", "helpsteer_rm-reward",
            "helpsteer_rm-reward-score", "rm_bench-reward", "rm_bench-reward-score", "analysis"])
        self.assertEqual(set(result["inherited_inference"]), es.INHERITED_SLOTS)
        self.assertEqual(set(result["inherited_scores"]), es.INHERITED_SLOTS)
        self.assertEqual(self.validations.count("inherited"), 21)
        for slot, entry in self.plan["inherited_scores"].items():
            self.assertEqual(result["scores"][slot], entry["score"])
            self.assertEqual(Path(entry["score"]["path"]).read_bytes(), originals[slot])
            self.assertFalse((self.output / slot.replace("/", "-")).exists())

    def test_unverified_inheritance_fails_before_any_new_output_or_child(self):
        with patch.object(inheritance, "verify_inherited_inference", side_effect=ValueError("Unaccepted prior inference")):
            with self.assertRaisesRegex(ValueError, "Unaccepted"):
                self.run_campaign()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.children)

    def test_changed_external_inheritance_bytes_fail_before_dispatch(self):
        Path(self.plan["inherited_inference"]["ifeval/dpo44"]["score"]["path"]).write_text("changed")
        with self.assertRaises(ValueError):
            self.run_campaign()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.system.children)

    def test_partial_or_missing_or_differing_inheritance_map_is_rejected(self):
        original = copy.deepcopy(self.plan)
        for mutation in ("missing", "partial", "score_drift", "capacity_missing", "wrong_campaign"):
            self.plan = copy.deepcopy(original)
            if mutation == "missing":
                self.plan["inherited_scores"].pop("ifeval/base")
            elif mutation == "partial":
                self.plan["inherited_inference"]["ifeval/ppo44"] = self.plan["inherited_inference"]["ifeval/dpo44"]
            elif mutation == "score_drift":
                self.plan["inherited_scores"]["ifeval/dpo44"] = self.plan["inherited_scores"]["ifeval/base"]
            elif mutation == "capacity_missing":
                self.plan["inherited_capacity"] = None
            else:
                self.plan["inherited_capacity"]["campaign"] = self.plan["protocol"]
            self.save_plan()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_campaign()
            self.assertFalse(self.output.exists())
            self.assertFalse(self.system.children)

    def test_full_serial_order_shared_deadline_and_separate_interpreters(self):
        result = self.run_campaign()
        self.assertEqual(result["status"], "primary_evaluation_passed")
        self.assertTrue(result["primary_evaluation_complete"])
        self.assertFalse(result["research_complete"])
        self.assertEqual(result["completed_tasks"], 10)
        self.assertEqual(len(self.system.children), 7)
        self.assertEqual([c["phase"] for c in result["children"]],
                         ["inference", "score"] * 3 + ["analysis"])
        self.assertEqual(self.validations.count("capacity_bridge"), 3)
        self.assertEqual(self.validations.count("score"), 3)
        self.assertEqual(self.validations.count("analysis"), 1)
        for record in result["children"]:
            request = record["request"]
            allocation = request.get("verification_allocation", {})
            self.assertEqual(request.get("deadline_epoch", allocation.get("deadline_epoch")), result["deadline_epoch"])
            self.assertEqual(Path(record["output"]).parent, self.output)
            self.assertGreaterEqual(len(record["known_processes"]), 2)
            self.assertEqual(record["active_processes"], [])
            expected = es.MODEL_PYTHON if record["phase"] in ("capacity", "inference") else es.SCORE_PYTHON
            self.assertEqual(record["command"][0], str(self.root / expected))
            self.assertTrue(Path(record["stdout_path"]).is_file())
            self.assertTrue(Path(record["stderr_path"]).is_file())
        self.assertTrue(result["historical_prerequisites"])
        self.assertFalse((self.root / "reports/.alignlab-evaluation.lock").exists())

    def test_plan_only_never_creates_output_or_launches(self):
        result = self.run_campaign(execute=False)
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.children, [])
        self.assertEqual(self.system.clock, 2_000_000_000)

    def test_initial_busy_no_allocation_output_or_wait(self):
        self.system.initial_busy = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertIsNone(result["started_at_epoch"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.clock, 2_000_000_000)

    def test_new_duplicate_supervisor_is_not_owned(self):
        self.system.extra_processes = [{"pid": 912345, "parent_pid": 1,
            "created_at_utc": stamp(self.system.now()), "command_line":
                subprocess.list2cmdline([str(self.root / es.MODEL_PYTHON), es.ENTRYPOINT,
                                        "--plan", str(self.plan_path)])}]
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertEqual(self.system.children, [])

    def test_unrecognized_python_without_training_keywords_is_blocked(self):
        self.system.extra_processes = [{"pid": 912346, "parent_pid": 1,
            "created_at_utc": stamp(self.system.now()), "command_line": "python benign.py"}]
        result = self.run_campaign()
        self.assertEqual(result["status"], "blocked_before_start")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.children, [])

    def test_existing_output_unknown_lock_and_bad_plan_sha_never_overwritten(self):
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_text("foreign lock", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        self.assertEqual(lock.read_text(), "foreign lock")
        lock.unlink()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_campaign()
        with self.assertRaises(ValueError):
            es.run_evaluation(self.plan_path, "0" * 64, system=self.system)

    def test_cpu_proof_missing_source_or_skipped_checks_rejected(self):
        cpu = json.loads(self.cpu.read_text())
        cpu["source_sha256"].pop("alignment_lab/evaluation_supervisor_v10.py")
        write(self.cpu, cpu)
        self.plan["cpu_check_report"] = es._pin(self.cpu)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "does not pin"):
            self.run_campaign(execute=False)
        cpu["source_sha256"] = self.sources
        cpu["skipped"] = 1
        write(self.cpu, cpu)
        self.plan["cpu_check_report"] = es._pin(self.cpu)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            self.run_campaign(execute=False)

    def test_invalid_budget_and_changed_tokenizer_rejected(self):
        self.plan["analysis_wall_seconds"] = self.plan["wall_seconds"] + 1
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "fit within"):
            self.run_campaign(execute=False)
        self.plan["analysis_wall_seconds"] = 100
        self.save_plan()
        (self.root / "models/tokenizer.json").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.run_campaign(execute=False)

    def test_preexisting_plan_stop_creates_no_campaign(self):
        Path(str(self.plan_path) + ".stop").write_text("user stop")
        result = self.run_campaign()
        self.assertEqual(result["status"], "stopped_before_start")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.children, [])

    def test_stop_during_child_waits_for_exit_and_never_dispatches_next(self):
        self.system.on_sleep = lambda system: (self.output / "STOP").write_text("user stop")
        result = self.run_campaign()
        self.assertEqual(result["status"], "paused")
        self.assertFalse(result["primary_evaluation_complete"])
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue(self.system.children[0].done)
        self.assertEqual((self.output / "STOP").read_text(), "user stop")

    def test_deadline_during_child_is_not_reset_or_success(self):
        self.plan["wall_seconds"] = 15
        self.plan["analysis_wall_seconds"] = 10
        self.save_plan()
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["deadline_epoch"], result["started_at_epoch"] + 15)
        self.assertTrue(self.system.children[0].done)
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse(result["primary_evaluation_complete"])

    def test_resource_wait_after_admission_consumes_original_budget(self):
        self.plan["wall_seconds"], self.plan["analysis_wall_seconds"] = 45, 10
        self.save_plan()
        self.system.after_busy = True
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(self.system.children), 2)
        self.assertEqual(result["elapsed_seconds"], 45)

    def test_changed_source_signals_current_child_then_fails_without_retry(self):
        def change(system, child):
            (self.root / "alignment_lab/evaluation_supervisor_v10.py").write_text("# changed")
        self.system.on_spawn = change
        with self.assertRaises(RuntimeError):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertTrue((self.output / "STOP").exists())
        self.assertTrue(self.system.children[0].done)
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse((self.root / "reports/.alignlab-evaluation.lock").exists())

    def test_nonzero_child_exit_is_failure_and_existing_run_cannot_retry(self):
        self.system.fail_index = 2
        with self.assertRaisesRegex(RuntimeError, "Child failed"):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(len(self.system.children), 3)
        with self.assertRaises(FileExistsError):
            self.run_campaign()

    def test_process_identity_drift_rejected(self):
        self.system.pid_drift = True
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertFalse(state["primary_evaluation_complete"])

    def test_exit_without_any_observed_child_rejected(self):
        self.system.hide_children = True
        with self.assertRaisesRegex(RuntimeError, "without observed"):
            self.run_campaign()
        self.assertEqual(len(self.system.children), 1)

    def test_wrapper_exit_does_not_allow_dispatch_while_observed_process_remains(self):
        self.system.wrapper_survives = True
        self.system.fail_index = 1
        def release(system):
            if system.clock >= 2_000_000_040:
                system.wrapper_survives = False
        self.system.on_sleep = release
        with self.assertRaisesRegex(RuntimeError, "Child failed"):
            self.run_campaign()
        self.assertGreaterEqual(self.system.children[1].started, 2_000_000_040)

    def test_cpu_decoder_descendant_is_recorded_without_becoming_foreign_worker(self):
        self.system.decoder_child = True
        self.system.fail_index = 3
        with self.assertRaisesRegex(RuntimeError, "Child failed"):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        score = state["children"][1]
        self.assertEqual(score["status"], "accepted")
        self.assertEqual(len(score["known_processes"]), 3)
        self.assertTrue(any(es.DECODER_ENTRYPOINT.replace("/", "\\") in p["command_line"]
            or es.DECODER_ENTRYPOINT in p["command_line"] for p in score["known_processes"]))

    def test_last_analysis_crossing_deadline_never_publishes_success(self):
        def advance(system, child):
            if child.args[4] == es.ANALYSIS_ENTRYPOINT:
                system.clock = 2_000_000_995
        self.system.on_spawn = advance
        result = self.run_campaign()
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertFalse(result["primary_evaluation_complete"])
        self.assertEqual(len(self.system.children), 7)
        self.assertTrue(self.system.children[-1].done)

    def test_gpu_inference_encoder_descendant_is_exactly_owned(self):
        self.system.encoder_child = True
        self.system.fail_index = 2
        with self.assertRaisesRegex(RuntimeError, "Child failed"):
            self.run_campaign()
        state = json.loads((self.output / "campaign.json").read_text())
        self.assertEqual(state["children"][0]["status"], "accepted")
        self.assertEqual(len(state["children"][0]["known_processes"]), 3)

    def test_gpu_encoder_with_extra_arguments_is_rejected(self):
        self.system.encoder_child = True
        self.system.encoder_extra_arg = True
        with self.assertRaisesRegex(RuntimeError, "Child creation/command/ancestry mismatch"):
            self.run_campaign()
        self.assertEqual(len(self.system.children), 1)

    def test_production_spawn_is_hidden_and_cpu_children_have_no_cuda(self):
        with patch.object(es.os, "name", "nt"), patch.object(es.subprocess, "Popen") as launch, \
                patch.object(es.subprocess, "CREATE_NO_WINDOW", 0x8000000, create=True):
            system = es.EvaluationSystem()
            for script, visible in ((es.INFERENCE_ENTRYPOINT, "0"), (es.SCORE_ENTRYPOINT, ""),
                                    (es.ANALYSIS_ENTRYPOINT, "")):
                args = ["fixed-python", "-I", "-B", "-u", script]
                system.spawn(args, "fixture-out", "fixture-err")
                kwargs = launch.call_args.kwargs
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], visible)
                self.assertEqual(kwargs["creationflags"], 0x8000000)
                self.assertEqual(kwargs["cwd"], self.root)
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_stop_drain_observation_does_not_start_terminal_publication_budget(self):
        original = es.AtomicStatusPublisher
        instances, publications = [], []
        fixture = self

        class Publisher(original):
            def __init__(self, path, **kwargs):
                instances.append((path, kwargs))
                super().__init__(path, **kwargs)

            def publish(self, value, *, terminal=False):
                publications.append({"terminal": terminal, "status": value["status"],
                    "clock": fixture.system.now(),
                    "all_children_done": all(child.done for child in fixture.system.children)})
                return super().publish(value, terminal=terminal)

        self.system.on_sleep = lambda system: (self.output / "STOP").write_text("fixture user stop")
        with patch.object(es, "AtomicStatusPublisher", Publisher):
            result = self.run_campaign()
        self.assertEqual(result["status"], "paused")
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0][1]["deadline_epoch"], self.plan["deadline_epoch"])
        self.assertTrue(any(row["terminal"] for row in publications))
        self.assertFalse(any(row["status"] == "stopping" for row in publications))
        self.assertTrue(all(row["all_children_done"] for row in publications if row["terminal"]))
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse((self.root / "reports/.alignlab-evaluation.lock").exists())

    def test_status_failure_signals_owned_child_without_retry_or_replacing_first_error(self):
        original = es.AtomicStatusPublisher
        first = OSError("campaign replacement failed")
        instances, calls = [], []

        class Publisher(original):
            def __init__(self, path, **kwargs):
                instances.append(self)
                self.failed = False
                super().__init__(path, **kwargs)

            def publish(self, value, *, terminal=False):
                calls.append(terminal)
                if value["children"]:
                    self.failed = True
                    raise first
                return super().publish(value, terminal=terminal)

        with patch.object(es, "AtomicStatusPublisher", Publisher):
            with self.assertRaises(OSError) as raised:
                self.run_campaign()
        self.assertIs(raised.exception, first)
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue((self.output / "STOP").exists())
        self.assertTrue((self.root / "reports/.alignlab-evaluation.lock").exists())
        self.assertEqual(calls[-2:], [True, True])
        self.assertFalse(self.system.children[0].done)
        # The retained lock and STOP leave exact-owned cleanup to the guardian;
        # this CPU fixture never terminates a live process or retries the run.

    def test_publisher_constructor_failure_never_leaves_an_evaluation_lock(self):
        with patch.object(es, "AtomicStatusPublisher", side_effect=OSError("clock construction failed")):
            with self.assertRaisesRegex(OSError, "construction failed"):
                self.run_campaign()
        self.assertFalse((self.root / "reports/.alignlab-evaluation.lock").exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(self.system.children, [])


if __name__ == "__main__":
    unittest.main()
