"""CPU fake lifecycle evidence only: no real model, GPU or quality result."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_capacity as cap
from alignment_lab import evaluation_execution as ex
from alignment_lab.evaluation_protocol import canonical, digest, parse
from tests_v2 import test_evaluation_execution as execution_fixtures

FakeModel, dataset, write = execution_fixtures.FakeModel, execution_fixtures.dataset, execution_fixtures.write


class Codec:
    def encode_prompt(self, context):
        text = context[-1]["content"]
        count = int(text.split(":")[1]) if text.startswith("tokens:") else 2
        return SimpleNamespace(input_ids=(10,) * count)

    def encode_completion(self, prompt, text):
        count = int(text.split(":")[1]) if text.startswith("tokens:") else 2
        return SimpleNamespace(completion_ids=(30,) * (count - 1) + (cap.EOS,))


def generation(prompt, *, forced):
    actions = (33,) * cap.HORIZON if forced else (cap.EOS,)
    value = {"action_ids": actions, "prompt_ids": tuple(prompt), "prompt_tokens": len(prompt),
             "generated_tokens": len(actions), "generated_eos_count": actions.count(cap.EOS),
             "termination": "forced_max_new_tokens" if forced else "eos",
             "mode": "forced_continuation_kv_stress" if forced else "eos_aware_greedy",
             "max_new_tokens": cap.HORIZON, "eos_id": cap.EOS, "ignore_eos": forced,
             "attention_all_true": True, "synthetic_eos_added": False,
             "final_action_flushed_to_kv": True, "max_cache_tokens": len(prompt) + len(actions),
             "decoder_calls": len(actions) + 1,
             "tokens_sha256": hashlib.sha256(json.dumps([list(prompt), list(actions)], separators=(",", ":")).encode()).hexdigest(),
             "capacity_only": True, "quality_evaluation": False}
    if not forced:
        value.update(generated_token_ids=actions, complete=True, prompt_truncated=False)
    return value


class Model(FakeModel):
    def greedy_generate(self, prompt_ids, **kwargs):
        kwargs["stop_requested"]()
        self._called()
        return generation(prompt_ids, forced=False)


class Memory:
    def reset(self):
        pass

    def release(self):
        pass

    def sample(self):
        return {"device": "cuda:0", "cuda_initialized": True, "device_name": "explicit CPU fixture",
                "total_memory_bytes": 8000, "allocated_bytes": 1000, "reserved_bytes": 2000,
                "max_allocated_bytes": 3000, "max_reserved_bytes": 4000}


class System:
    def __init__(self):
        self.clock = 100.0
        self.live, self.busy = [], False

    def now(self):
        return self.clock

    def resources(self):
        return {"blockers": ["busy"] if self.busy else [], "gpus": [{"index": 0, "free_mb": 8000}]}

    def processes(self):
        return copy.deepcopy(self.live)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing real protocol/complete-roster fixture validator,
        # not a parallel hand-written copy of its acceptance formula.
        self.fixture = execution_fixtures.ExecutionTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(cap, "ROOT", self.root))
        for name in cap.REQUIRED_SOURCES:
            path = self.root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"# Explicit CPU capacity fixture dependency\n")
        ifrows, ilengths = list(self.fixture.ifeval.rows), self.fixture.ifeval.lengths
        for index in (1, 2):
            ifrows[index]["prompt"], ilengths[str(index)] = "tokens:7", 7
        self.fixture.ifeval = dataset("ifeval", "key", ifrows, ilengths)
        heldrows, hlengths = list(self.fixture.held.rows), self.fixture.held.lengths
        heldrows[5]["response2"], hlengths["v5"][1] = "tokens:5479", 5481
        self.fixture.held = dataset("helpsteer3_final_holdout", "evaluation_id", heldrows, hlengths)
        rmrows, rlengths = list(self.fixture.rm.rows), self.fixture.rm.lengths
        rmrows[3]["chosen"][2], rlengths["r3"]["chosen"][2] = "tokens:2361", 2363
        self.fixture.rm = dataset("rm_bench", "id", rmrows, rlengths)
        self.fixture.assets = SimpleNamespace(ifeval=self.fixture.ifeval, rm_bench=self.fixture.rm)
        self.stack.enter_context(patch.object(ex, "load_evaluation_assets", return_value=self.fixture.assets))
        self.stack.enter_context(patch.object(ex, "load_final_holdout", return_value=self.fixture.held))
        self.fixture.protocol["dataset_identities"] = {
            d.name: d.identity for d in (self.fixture.ifeval, self.fixture.rm, self.fixture.held)}
        self.fixture.protocol_sha = write(self.fixture.protocol_path, self.fixture.protocol)
        self.system, self.models = System(), []
        self.stack.enter_context(patch.object(cap, "_CudaMemory", Memory))
        self.stack.enter_context(patch.object(cap, "_verify_roster", return_value=self.fixture.roster))
        self.stack.enter_context(patch.object(cap, "_forced",
            side_effect=lambda model, prompt, request, stop: generation(prompt, forced=True)))
        self.stack.enter_context(patch.object(cap, "_invoke_encoder", side_effect=self.fake_encoder))
        self.sources = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                        for name in cap.REQUIRED_SOURCES}
        token_dir = Path(self.fixture.roster["tokenizer_dir"])
        token_dir.mkdir()
        token_files = {}
        for name in cap.TOKENIZER_FILES:
            payload = ("opaque CPU fixture " + name).encode()
            (token_dir / name).write_bytes(payload)
            token_files[name] = {"sha256": hashlib.sha256(payload).hexdigest()}
        snapshot_path = Path(self.fixture.roster["model_snapshot"]) / "snapshot-manifest.json"
        snapshot_sha = write(snapshot_path, {"files": token_files, "fixture_only": True})
        self.fixture.roster["model_manifest_sha256"] = snapshot_sha
        self.fixture.protocol["model_manifests"]["base"]["sha256"] = snapshot_sha
        for record in self.fixture.roster["models"].values():
            path = Path(record["acceptance_path"])
            accepted = parse(path.read_bytes())
            accepted["snapshot_manifest_sha256"] = snapshot_sha
            record["acceptance_sha256"] = write(path, accepted)
        self.fixture.roster_sha = write(self.fixture.roster_path, self.fixture.roster)
        self.fixture.protocol_sha = write(self.fixture.protocol_path, self.fixture.protocol)
        python = self.root / cap.MODEL_PYTHON
        python.parent.mkdir(parents=True)
        python.write_bytes(b"explicit fake interpreter; never executed")
        python_sha = hashlib.sha256(python.read_bytes()).hexdigest()
        self.campaign = self.root / "runs/campaign"
        self.campaign.mkdir()
        self.plan_path = self.root / "plan.json"
        self.plan = {"schema_version": 1, "kind": cap.CAMPAIGN_KIND, "output": str(self.campaign),
                     "wall_seconds": 900, "poll_seconds": 10, "source_sha256": self.sources,
                     "model_python_sha256": python_sha,
                     "decoder": {"tokenizer_path": str(token_dir / "tokenizer.json"),
                                 "tokenizer_sha256": token_files["tokenizer.json"]["sha256"],
                                 "tokenizers_version": "fixture"},
                     "protocol": {"path": str(self.fixture.protocol_path), "sha256": self.fixture.protocol_sha},
                     "roster": {"path": str(self.fixture.roster_path), "sha256": self.fixture.roster_sha}}
        self.plan_sha = write(self.plan_path, self.plan)
        self.request_path = self.root / "request.json"
        self.request = {"schema_version": 1, "kind": cap.KIND, "output": str(self.campaign / "capacity"),
                        "deadline_epoch": 1000.0, "evaluation_plan_path": str(self.plan_path),
                        "evaluation_plan_sha256": self.plan_sha, "source_sha256": self.sources,
                        "protocol_path": str(self.fixture.protocol_path), "protocol_sha256": self.fixture.protocol_sha,
                        "roster_path": str(self.fixture.roster_path), "roster_sha256": self.fixture.roster_sha}
        self.request_sha = write(self.request_path, self.request)
        command = lambda script, args: subprocess.list2cmdline(
            ["python.exe", "-I", "-B", str(self.root / "scripts" / script)] + args)
        child = command("run_alignlab_evaluation_capacity.py",
                        ["--request", str(self.request_path), "--request-sha256", self.request_sha, "--execute"])
        parent = command("supervise_alignlab_evaluation.py",
                         ["--plan", str(self.plan_path), "--plan-sha256", self.plan_sha, "--execute"])
        def process(pid, ppid, cmd):
            return {"pid": pid, "parent_pid": ppid, "created_at_utc": "2026-10-20T00:00:00Z", "command_line": cmd}
        self.system.live = [process(os.getpid(), 900003, child), process(900003, 900001, child),
                            process(900001, 900002, parent), process(900002, 1, parent)]
        self.lock = {"owner": cap.CAMPAIGN_KIND, "pid": 900001, "plan_path": str(self.plan_path),
                     "plan_sha256": self.plan_sha, "output": str(self.campaign), "started_at_epoch": 100.0,
                     "deadline_epoch": 1000.0, "processes": copy.deepcopy(self.system.live[2:]), "token": "test-owner"}
        self.lock_path = self.root / "reports/.alignlab-evaluation.lock"
        write(self.lock_path, self.lock)
        self.hook = None

    def fake_encoder(self, payload, *, deadline_epoch=None):
        encoded = {}
        codec = Codec()
        for task, case in payload["cases"].items():
            prompt = codec.encode_prompt(case["context"])
            completion = () if task == "ifeval" else codec.encode_completion(prompt, case["response"]).completion_ids
            pair = [list(prompt.input_ids), list(completion)]
            encoded[task] = {"prompt_ids": pair[0], "completion_ids": pair[1],
                             "prompt_tokens": len(pair[0]), "completion_tokens": len(pair[1]),
                             "tokens_sha256": digest(pair)}
        return {"schema_version": 1, "kind": "alignlab-capacity-input-encoding-result-v1",
                "input_sha256": digest(payload), "source_sha256": payload["source_sha256"],
                "tokenizer_files": payload["tokenizer_files"], "python_sha256": payload["python_sha256"],
                "tokenizers_version": payload["tokenizers_version"], "transformers_version": payload["transformers_version"],
                "encoded_inputs": encoded, "torch_imported": False, "gpu_operations_performed": False,
                "process_identity": {"pid": 999991, "parent_pid": os.getpid(),
                                     "created_at_utc": "2026-10-20T00:00:00Z", "command_line": "explicit CPU fake encoder"}}

    def tearDown(self):
        self.stack.close()
        self.fixture.tearDown()

    def factory(self, roster, model_id):
        model = Model(model_id, hook=self.hook)
        self.models.append(model)
        return model

    def run_capacity(self):
        return cap.run_evaluation_capacity(self.request_path, self.request_sha, system=self.system,
                                          model_factory=self.factory, codec_factory=lambda _: Codec())

    def verify(self, receipt):
        return cap.verify_evaluation_capacity(self.request["output"],
            expected_manifest_sha256=receipt["manifest_sha256"], expected_request=self.request)

    def test_full_maxima_serial_nine_models_and_exact_contexts(self):
        receipt = self.run_capacity()
        value = self.verify(receipt)
        self.assertEqual([m.model_id for m in self.models], list(cap.MODEL_IDS))
        cases = value["selection"]["cases"]
        self.assertEqual((cases["ifeval"]["record_id"], cases["ifeval"]["input_tokens"]), (1, 7))
        self.assertEqual((cases["helpsteer_rm"]["record_id"], cases["helpsteer_rm"]["response_index"],
                          cases["helpsteer_rm"]["input_tokens"]), ("v5", 1, 5481))
        self.assertEqual((cases["rm_bench"]["record_id"], cases["rm_bench"]["response_index"],
                          cases["rm_bench"]["input_tokens"]), ("r3", 2, 2363))
        for model_id in cap.MODEL_IDS:
            tests = value["models"][model_id]["tests"]
            if model_id == "reward":
                self.assertEqual(tests["helpsteer_rm"]["input_tokens"], 5481)
            else:
                self.assertEqual(tests["natural_eos"]["receipt"]["generated_tokens"], 1)
                self.assertEqual(tests["forced_2048"]["receipt"]["max_cache_tokens"], 2055)
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())
        self.assertEqual(parse(self.lock_path.read_bytes()), self.lock)
        with self.assertRaises(FileExistsError):
            self.run_capacity()

    def test_stop_and_deadline_leave_no_success_manifest(self):
        self.hook = lambda _: (self.campaign / "STOP").write_text("test", encoding="utf-8")
        with self.assertRaises(InterruptedError):
            self.run_capacity()
        output = Path(self.request["output"])
        self.assertFalse((output / "manifest.json").exists())
        self.assertEqual(parse((output / "progress.json").read_bytes())["status"], "paused")
        with self.assertRaisesRegex(ValueError, "regular file"):
            cap.verify_evaluation_capacity(output, expected_manifest_sha256="a" * 64)

    def test_deadline_during_inference_preserves_failure(self):
        self.hook = lambda _: setattr(self.system, "clock", 1000.0)
        with self.assertRaises(TimeoutError):
            self.run_capacity()
        status = parse((Path(self.request["output"]) / "progress.json").read_bytes())
        self.assertEqual(status["status"], "budget_exhausted")
        self.assertFalse(status["capacity_complete"])

    def test_wrong_ancestry_reused_pid_and_foreign_training_rejected(self):
        original = copy.deepcopy(self.system.live)
        for change in ("ancestor", "creation", "foreign"):
            with self.subTest(change=change):
                self.system.live = copy.deepcopy(original)
                if change == "ancestor":
                    self.system.live[1]["parent_pid"] = 1
                elif change == "creation":
                    self.system.live[2]["created_at_utc"] = "2026-10-20T00:00:01Z"
                else:
                    self.system.live.append({"pid": 999999, "parent_pid": 1,
                        "created_at_utc": "2026-10-20T00:00:00Z", "command_line": "python windows.train"})
                with self.assertRaises(RuntimeError):
                    self.run_capacity()
                self.assertFalse(Path(self.request["output"]).exists())
                self.assertEqual(self.models, [])

    def test_frozen_source_and_parent_deadline_mismatch_block_before_model(self):
        self.lock["deadline_epoch"] += 1
        write(self.lock_path, self.lock)
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.run_capacity()
        self.lock["deadline_epoch"] -= 1
        write(self.lock_path, self.lock)
        (self.root / cap.SOURCES[0]).write_bytes(b"# changed source")
        with self.assertRaises(ValueError):
            self.run_capacity()
        self.assertEqual(self.models, [])

    def test_wrong_codec_lengths_and_changed_weights_rejected(self):
        with patch.object(Codec, "encode_prompt", return_value=SimpleNamespace(input_ids=(10,))):
            with self.assertRaisesRegex(ValueError, "lengths"):
                self.run_capacity()
        self.assertEqual(self.models, [])

    def test_changed_model_full_audit_refuses_success(self):
        self.hook = lambda model: setattr(model, "mutation", True)
        with self.assertRaisesRegex(ValueError, "audit"):
            self.run_capacity()
        self.assertFalse((Path(self.request["output"]) / "manifest.json").exists())

    def test_partial_forced_kv_receipt_and_zero_cuda_peak_rejected(self):
        context = cap._context(self.request)
        selected = cap.select_maximum_inputs(context)
        self.assertEqual(selected["cases"]["helpsteer_rm"]["input_tokens"], 5481)
        receipt = generation((10, 10), forced=True)
        receipt["max_cache_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "KV"):
            cap._generation(receipt, (10, 10), forced=True)
        memory = Memory().sample()
        memory["max_allocated_bytes"] = 0
        with self.assertRaises(ValueError):
            cap._memory(memory)

    def test_result_payload_tampering_and_full_before_after_anchor_mismatch(self):
        receipt = self.run_capacity()
        output = Path(self.request["output"])
        manifest = parse((output / "manifest.json").read_bytes())
        manifest["models"]["sft"]["frozen_audits"][1]["base_sha256"] = "b" * 64
        # Rehash progress as well, so this exercises semantic audit validation.
        progress = parse((output / "progress.json").read_bytes())
        progress["models"] = manifest["models"]
        progress_sha = write(output / "progress.json", progress)
        manifest["files"]["progress.json"] = {"sha256": progress_sha, "bytes": (output / "progress.json").stat().st_size}
        receipt["manifest_sha256"] = write(output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "frozen audits"):
            self.verify(receipt)

    def test_same_length_token_forgery_with_consistent_hashes_is_rejected(self):
        receipt = self.run_capacity()
        output = Path(self.request["output"])
        manifest = parse((output / "manifest.json").read_bytes())
        item = manifest["encoded_inputs"]["helpsteer_rm"]
        item["completion_ids"][0] = 31
        item["tokens_sha256"] = digest([item["prompt_ids"], item["completion_ids"]])
        manifest["models"]["reward"]["tests"]["helpsteer_rm"]["input_tokens_sha256"] = item["tokens_sha256"]
        progress = parse((output / "progress.json").read_bytes())
        progress["encoded_inputs"], progress["models"] = manifest["encoded_inputs"], manifest["models"]
        progress_sha = write(output / "progress.json", progress)
        manifest["files"]["progress.json"] = {"sha256": progress_sha, "bytes": (output / "progress.json").stat().st_size}
        receipt["manifest_sha256"] = write(output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "original-source token re-encoding"):
            self.verify(receipt)

    def test_encoder_identity_mismatch_refused_before_subprocess(self):
        context = cap._context(self.request)
        selection = cap.select_maximum_inputs(context)
        encoded = cap._encode_selection(context, selection, Codec())
        (self.root / cap.ENCODER_SOURCE).write_bytes(b"# changed encoder")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            cap._reencode_inputs(context, self.request, self.plan, selection, encoded)

    def test_expired_shared_deadline_blocks_cpu_encoder(self):
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            cap.verify_evaluation_capacity(self.request["output"], expected_manifest_sha256="a" * 64,
                                           deadline_epoch=time.time() - 1)

    def test_no_torch_import_for_import_or_invalid_receipt(self):
        real_root = Path(__file__).resolve().parents[1]
        code = ("import sys; sys.path.insert(0, " + repr(str(real_root)) + "); "
                "from alignment_lab.evaluation_capacity import verify_evaluation_capacity; "
                "assert 'torch' not in sys.modules; "
                "print('capacity import is stdlib-only')")
        child = subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True,
                               capture_output=True, text=True)
        self.assertIn("stdlib-only", child.stdout)


class RealEncoderTests(unittest.TestCase):
    """Actual pinned tokenizer on three frozen raw inputs; no model or GPU."""
    evidence = None

    def test_actual_three_maximum_inputs_without_torch(self):
        from alignment_lab.evaluation_protocol import load_evaluation_assets, load_final_holdout
        root = Path(__file__).resolve().parents[1]
        assets = load_evaluation_assets(root / "data/evaluation-assets-20261020T103505Z/ASSET_MANIFEST.json",
                                       "6950abb56fc6ae6044b0283c53a7b5fc3e64d57a43ea8ba863621e2274505720")
        holdout = load_final_holdout(root / "reports/alignlab-v2-data-audit-v3-p1024-r768-20261020",
                                    "84d059ba111c37d8f89d70cec6db6bd6c4fd7aef8d4cf933343d67556a204e0b")
        context = SimpleNamespace(assets=assets, dataset=holdout, roster={
            "model_snapshot": str(root / "models/qwen3-1.7b-70d244c"),
            "model_manifest_sha256": "745814c6165a1d3d9445c26dd65a5b6392227132e6ac4b7ea28c7effb120a02a",
            "tokenizer_dir": str(root / "models/qwen3-1.7b-tokenizer-70d244c")})
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        request = {"source_sha256": {name: sha(root / name) for name in cap.ENCODER_SOURCES}}
        plan = {"model_python_sha256": sha(root / cap.MODEL_PYTHON),
                "decoder": {"tokenizer_path": str(Path(context.roster["tokenizer_dir"]) / "tokenizer.json"),
                            "tokenizer_sha256": sha(Path(context.roster["tokenizer_dir"]) / "tokenizer.json"),
                            "tokenizers_version": importlib.metadata.version("tokenizers")}}
        selection = cap.select_maximum_inputs(context)
        payload = cap._encoder_payload(context, request, plan, selection)
        receipt = cap._invoke_encoder(payload, deadline_epoch=time.time() + 120)
        for task, maximum in {"ifeval": 373, "helpsteer_rm": 5481, "rm_bench": 2363}.items():
            encoded = receipt["encoded_inputs"][task]
            self.assertEqual(encoded["prompt_tokens"] + encoded["completion_tokens"], maximum)
            self.assertEqual(selection["cases"][task]["input_tokens"], maximum)
        self.assertFalse(receipt["torch_imported"])
        self.assertFalse(receipt["gpu_operations_performed"])
        self.__class__.evidence = {"fixture_only_not_model_acceptance": True, "selection": selection,
                                  "encoder_receipt": receipt}


if __name__ == "__main__":
    unittest.main()
