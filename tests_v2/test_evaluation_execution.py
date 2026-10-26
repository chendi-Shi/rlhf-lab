"""CPU-only fake-model fixtures; none of these outputs is a quality measurement."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_execution as ex
from alignment_lab.evaluation_protocol import EvaluationDataset, canonical, digest, parse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def dataset(name, field, rows, lengths):
    return EvaluationDataset(name, field, canonical(rows),
        canonical({"dataset": name, "record_count": len(rows), "records_sha256": digest(rows),
                   "ordered_ids_sha256": digest([row[field] for row in rows])}), canonical(lengths))


class FakeCodec:
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(chr(65 + token % 26) for token in ids)
    tokenizer = Tokenizer()

    def encode_prompt(self, context):
        return SimpleNamespace(input_ids=(10, 20))

    def encode_completion(self, prompt, text):
        return SimpleNamespace(completion_ids=(31 if text.endswith("2") else 30, 151645))


class FakeSystem:
    def __init__(self):
        self.clock, self.busy, self.other = 100.0, False, []

    def now(self):
        return self.clock

    def resources(self):
        return {"blockers": ["busy"] if self.busy else [],
                "gpus": [{"index": 0, "free_mb": 8000, "utilization_percent": 0}]}

    def processes(self):
        return [{"pid": os.getpid(), "parent_pid": 1,
                 "created_at_utc": "2026-10-20T00:00:00+00:00",
                 "command_line": "python run_alignlab_evaluation_stage.py --fake"}] + self.other


class FakeModel:
    def __init__(self, model_id, hook=None):
        self.model_id, self.hook, self.calls = model_id, hook, 0
        self.metadata = {"device": "cuda:0", "cpu_fixture": False, "inference_only": True,
                         "model_id": model_id, "stage": ex.MODEL_STAGES[model_id],
                         "quantization": ex.QUANTIZATION, "quantized_linear_count": 196}
        self.mutation = False
        self.bad_receipt = None
        self.length_first = False

    def frozen_audit(self, *, full):
        return {"unchanged": not self.mutation, "full": full,
                "type": "alignlab-evaluation-frozen-audit-v1", "model_id": self.model_id,
                "file_contents_rehashed": True, "untracked_content_writes_checked": True,
                **{key: "a" * 64 for key in
                   ("source_sha256", "adapter_sha256", "base_sha256", "head_sha256")}}

    def _called(self):
        self.calls += 1
        if self.hook:
            self.hook(self)

    def greedy_generate(self, prompt_ids, **kwargs):
        self._called()
        actions = (33,) * 2048 if self.length_first and self.calls == 1 else (151645,)
        result = {"generated_token_ids": actions, "termination": "length" if len(actions) == 2048 else "eos",
                  "complete": True, "prompt_truncated": False, "prompt_ids": prompt_ids,
                  "generated_tokens": len(actions), "ignore_eos": False,
                  "synthetic_eos_added": False, "attention_all_true": True}
        if self.bad_receipt:
            result.update(self.bad_receipt)
        return result

    def reward_score(self, prompt_ids, completion_ids):
        self._called()
        return float(completion_ids[0] / 10)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "runs").mkdir()
        (self.root / "reports").mkdir()
        for name in ex.REQUIRED_SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# CPU fixture source\n")
        ifrows = [{"key": i, "prompt": f"prompt {i}",
                   "instruction_id_list": ["a", "b"] if i < 293 else ["a"]}
                  for i in range(541)]
        rmrows = [{"id": f"r{i}", "prompt": f"prompt {i}", "domain": "chat",
                   "chosen": ["c0", "c1", "c2"], "rejected": ["r0", "r1", "r2"]}
                  for i in range(1327)]
        heldrows = [{"evaluation_id": f"v{i}", "record_id": f"content{i // 2}", "source_index": i,
                     "context": [{"role": "user", "content": f"prompt {i}"}],
                     "response1": "r1", "response2": "r2", "overall_preference": 0 if i < 97 else -1}
                    for i in range(2017)]
        self.ifeval = dataset("ifeval", "key", ifrows, {str(i): 2 for i in range(541)})
        self.rm = dataset("rm_bench", "id", rmrows,
                          {r["id"]: {"chosen": [4] * 3, "rejected": [4] * 3} for r in rmrows})
        self.held = dataset("helpsteer3_final_holdout", "evaluation_id", heldrows,
                            {r["evaluation_id"]: [4, 4] for r in heldrows})
        self.assets = SimpleNamespace(ifeval=self.ifeval, rm_bench=self.rm)
        self.roster_path, self.protocol_path = self.root / "roster.json", self.root / "protocol.json"
        snapshot = self.root / "snapshot"
        snapshot_sha = write(snapshot / "snapshot-manifest.json", {"fixture": "no real model"})
        self.roster = {"schema_version": 1, "kind": "alignlab-accepted-evaluation-roster-v1",
                       "model_snapshot": str(snapshot), "model_manifest_sha256": snapshot_sha,
                       "tokenizer_dir": str(self.root / "tokenizer"), "models": {}}
        bindings, acceptances = {}, {}
        for model_id, stage in ex.MODEL_STAGES.items():
            if stage == "base":
                directory, sha, progress_path, progress_sha = None, None, None, None
                bindings[model_id] = {"path": str(snapshot / "snapshot-manifest.json"), "sha256": snapshot_sha}
            else:
                directory = self.root / "models" / model_id
                directory.mkdir(parents=True)
                weights = b"synthetic opaque bytes; never loaded"
                (directory / "weights.pt").write_bytes(weights)
                required = {"sft": (), "reward": ("sft",), "dpo": ("sft",), "ppo": ("sft", "reward")}[stage]
                parents = {p: {"path": bindings[p]["path"], "sha256": bindings[p]["sha256"],
                               "size_bytes": Path(bindings[p]["path"]).stat().st_size} for p in required}
                full_scope = "full_accepted_real_training_partition"
                scope = "registered_unique_real_prompt_subset" if stage == "ppo" else full_scope
                row_count = {"sft": 4000, "reward": 13139, "dpo": 13139, "ppo": 1024}[stage]
                updates = {"sft": 250, "reward": 822, "dpo": 822, "ppo": 2048}[stage]
                config = {"purpose": "alignlab-formal-parent-stage-v1" if stage in {"sft", "reward"}
                          else "alignlab-formal-alignment-stage-v1",
                          "scope": scope, "expected_rows": row_count, "expected_updates": updates,
                          "registered_workload": {"rows": row_count, "optimizer_updates": updates, "scope": scope}}
                identity = {"config": {**config, "stage": stage, "formal_training": True, "synthetic_rows": 0,
                                      "work": {"seed": int(model_id[-2:]) if stage in {"dpo", "ppo"} else 42}},
                            "parents": parents}
                completion = {"synthetic_rows": 0, "stop_reason": "data_exhausted",
                              "data_pass_complete": True, "protocol_complete": True, "optimizer_updates": updates,
                              "rows_processed": row_count, "scope": scope,
                              "full_accepted_dataset_pass": stage != "ppo", "full_selected_dataset_pass": True}
                manifest = {"format": "alignlab-completed-stage-v1", "status": "complete", "stage": stage,
                            "identity": identity, "identity_sha256": digest(identity), "completion": completion,
                            "artifacts": {"weights.pt": {"sha256": hashlib.sha256(weights).hexdigest(),
                                                         "size_bytes": len(weights)}}}
                sha = write(directory / "manifest.json", manifest)
                bindings[model_id] = {"path": str(directory / "manifest.json"), "sha256": sha}
                progress_path = self.root / f"{model_id}-progress.json"
                progress_sha = write(progress_path, {"status": "formal_parent_stage_passed"
                    if stage in {"sft", "reward"} else "formal_alignment_stage_passed", "formal_training": True,
                    "research_complete": False, "identity": identity, "completion": completion,
                    "actual_optimizer_updates": updates, "artifact": {"artifact_dir": str(directory),
                    "manifest_path": str(directory / "manifest.json"), "manifest_sha256": sha}})
            acceptance_path = self.root / f"{model_id}-acceptance.json"
            row = {"stage": stage, "artifact_dir": str(directory) if directory else None,
                   "manifest_sha256": sha, "progress_path": str(progress_path) if progress_path else None,
                   "progress_sha256": progress_sha}
            acceptance_sha = write(acceptance_path, {"schema_version": 1,
                "kind": "alignlab-independent-evaluation-model-acceptance-v1", "status": "accepted_for_evaluation",
                "all_checks_passed": True, "processes_exited": True, "snapshot_manifest_sha256": snapshot_sha,
                "models": {model_id: row}})
            acceptances[model_id] = row
            self.roster["models"][model_id] = {"model_id": model_id, "stage": stage,
                "artifact_dir": row["artifact_dir"], "manifest_sha256": sha,
                "acceptance_path": str(acceptance_path), "acceptance_sha256": acceptance_sha}
        self.roster_sha = write(self.roster_path, self.roster)
        asset_path, held_path = self.root / "assets.json", self.root / "holdout/manifest.json"
        asset_sha, held_sha = write(asset_path, {}), write(held_path, {})
        self.protocol = {"kind": "alignlab-independent-evaluation-protocol-v1", "schema_version": 1,
            "serialization_version": ex.SERIALIZATION, "denominators": ex.DENOMINATORS,
            "generation": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 2048,
                           "eos_token_id": 151645, "pad_token_id": 151643, "vocab_size": 151936,
                           "input_truncation": False}, "scorer_seed": 0, "training_use_allowed": False,
            "quality_selection_allowed": False, "model_manifests": bindings, "source_files": {},
            "nltk_resource_files": {}, "asset_manifest": {"path": str(asset_path), "sha256": asset_sha},
            "holdout_manifest": {"path": str(held_path), "sha256": held_sha},
            "dataset_identities": {d.name: d.identity for d in (self.ifeval, self.rm, self.held)}}
        self.protocol_sha = write(self.protocol_path, self.protocol)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(ex, "ROOT", self.root))
        self.stack.enter_context(patch.object(ex, "load_evaluation_assets", return_value=self.assets))
        self.stack.enter_context(patch.object(ex, "load_final_holdout", return_value=self.held))
        self.system, self.codec, self.output_number = FakeSystem(), FakeCodec(), 0

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def request(self, task="ifeval"):
        self.output_number += 1
        sources = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in self.root.rglob("*.py")}
        self.value = {"schema_version": 1, "kind": ex.KIND, "task": task,
            "model_id": "base" if task == "ifeval" else "reward",
            "output": str(self.root / "runs" / f"out{self.output_number}"), "deadline_epoch": 1000.0,
            "protocol_path": str(self.protocol_path), "protocol_sha256": self.protocol_sha,
            "roster_path": str(self.roster_path), "roster_sha256": self.roster_sha, "source_sha256": sources}
        self.request_path = self.root / f"request{self.output_number}.json"
        self.request_sha = write(self.request_path, self.value)
        self.model = FakeModel(self.value["model_id"])
        return self.value

    def run_stage(self):
        return ex.run_evaluation_stage(self.request_path, self.request_sha, system=self.system,
            model_factory=lambda *_: self.model, codec_factory=lambda *_: self.codec)

    def verify(self, receipt):
        return ex.verify_evaluation_result(self.value["output"],
            expected_manifest_sha256=receipt["manifest_sha256"], expected_request=self.value,
            decode_tokens=lambda ids: self.codec.tokenizer.decode(ids), codec=self.codec)

    def test_full_ifeval_eos_and_length_output_and_no_overwrite(self):
        self.request()
        self.model.length_first = True
        receipt = self.run_stage()
        verified = self.verify(receipt)
        self.assertEqual(verified["counts"], {"records": 541, "prompts": 541, "instructions": 834})
        self.assertEqual(self.model.calls, 541)
        rows = verified["ordered_results"]
        self.assertEqual(rows[0]["termination"], "length")
        self.assertEqual(len(rows[0]["generated_token_ids"]), 2048)
        self.assertEqual(rows[1]["generated_token_ids"], [151645])
        self.assertEqual(rows[1]["response"], "")
        with self.assertRaises(FileExistsError):
            self.run_stage()
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())

    def test_full_rm_bench_six_answers_per_record(self):
        self.request("rm_bench")
        verified = self.verify(self.run_stage())
        self.assertEqual(self.model.calls, 7962)
        self.assertEqual(verified["counts"], {"records": 1327, "answer_scores": 7962, "comparisons": 11943})

    def test_full_holdout_duplicate_content_is_not_deduplicated(self):
        self.request("helpsteer_rm")
        verified = self.verify(self.run_stage())
        self.assertEqual(self.model.calls, 4034)
        self.assertEqual(verified["counts"]["records"], 2017)
        rows = verified["ordered_results"]
        self.assertEqual(rows[0]["record_id"], rows[1]["record_id"])
        self.assertNotEqual(rows[0]["evaluation_id"], rows[1]["evaluation_id"])

    def test_busy_and_foreign_lock_never_create_output(self):
        self.request()
        self.system.busy = True
        with self.assertRaisesRegex(RuntimeError, "resources unavailable"):
            self.run_stage()
        self.assertFalse(Path(self.value["output"]).exists())
        self.system.busy = False
        lock = self.root / "reports/.alignlab-gpu.lock"
        lock.write_bytes(b"unknown owner")
        with self.assertRaises(FileExistsError):
            self.run_stage()
        self.assertEqual(lock.read_bytes(), b"unknown owner")
        self.assertEqual(self.model.calls, 0)

    def test_other_alignment_process_blocks_without_gpu(self):
        self.request()
        self.system.other = [{"pid": 12345, "command_line": "python run_alignlab_alignment_stage.py",
                             "parent_pid": 1, "created_at_utc": "2026-10-20T01:00:00+00:00"}]
        with self.assertRaisesRegex(RuntimeError, "Other active"):
            self.run_stage()
        self.assertFalse(Path(self.value["output"]).exists())

    def test_stop_and_deadline_leave_partial_not_complete(self):
        for mode in ("stop", "deadline"):
            with self.subTest(mode=mode):
                self.request()
                self.system.clock = 100.0
                def stop(model):
                    if mode == "stop":
                        (Path(self.value["output"]) / "STOP").write_text("stop", encoding="utf-8")
                    else:
                        self.system.clock = 1001.0
                self.model.hook = stop
                with self.assertRaises((InterruptedError, TimeoutError)):
                    self.run_stage()
                output = Path(self.value["output"])
                self.assertFalse((output / "manifest.json").exists())
                status = parse((output / "progress.json").read_bytes())
                self.assertEqual(status["status"], "paused" if mode == "stop" else "budget_exhausted")
                self.assertFalse(status["inference_complete"])

    def test_expired_before_start_never_creates_output_or_lock(self):
        self.request()
        self.system.clock = 1000
        with self.assertRaises(TimeoutError):
            self.run_stage()
        self.assertFalse(Path(self.value["output"]).exists())
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())

    def test_source_changes_during_run_are_rejected_at_final_boundary(self):
        self.request()
        def change(model):
            if model.calls == 1:
                (self.root / "alignment_lab/evaluation_model.py").write_bytes(b"changed")
        self.model.hook = change
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_stage()
        self.assertFalse((Path(self.value["output"]) / "manifest.json").exists())

    def test_changed_weights_and_bad_generation_rejected(self):
        for bad in ({"complete": False}, {"prompt_truncated": True}, {"synthetic_eos_added": True},
                    {"generated_token_ids": (3,), "termination": "length"}):
            with self.subTest(bad=bad):
                self.request()
                self.model.bad_receipt = bad
                with self.assertRaises(ValueError):
                    self.run_stage()
                self.assertFalse((Path(self.value["output"]) / "manifest.json").exists())
        self.request()
        self.model.hook = lambda model: setattr(model, "mutation", True)
        with self.assertRaisesRegex(ValueError, "unchanged-weight"):
            self.run_stage()

    def test_file_tamper_and_extra_file_rejected(self):
        self.request()
        receipt = self.run_stage()
        output = Path(self.value["output"])
        path = output / "outputs.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b"{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.verify(receipt)
        path.write_bytes(original)
        (output / "extra.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.verify(receipt)

    def test_complete_id_validator_reorders_but_rejects_missing_duplicate_and_truncation(self):
        self.request()
        receipt = self.run_stage()
        context = ex._context(self.value)
        rows = list(self.verify(receipt)["ordered_results"])
        decode = lambda ids: self.codec.tokenizer.decode(ids)
        valid = ex._validate_rows(context, self.value, list(reversed(rows)),
                                  decode_tokens=decode, codec=self.codec)
        self.assertEqual(valid[0]["key"], 0)
        for invalid in (rows[:-1], rows + rows[:1],
                        [{**rows[0], "input_truncated": True}] + rows[1:],
                        [{**rows[0], "prompt_token_ids": [11, 20]}] + rows[1:],
                        [{**rows[0], "protocol_sha256": "b" * 64}] + rows[1:]):
            with self.assertRaises(ValueError):
                ex._validate_rows(context, self.value, invalid, decode_tokens=decode, codec=self.codec)

    def test_live_roster_and_protocol_tamper_prevent_start(self):
        self.request()
        self.roster["models"].pop("ppo44")
        self.value["roster_sha256"] = write(self.roster_path, self.roster)
        self.request_sha = write(self.request_path, self.value)
        with self.assertRaisesRegex(ValueError, "nine-model"):
            self.run_stage()
        self.assertEqual(self.model.calls, 0)

    def test_nonfinite_reward_and_short_context_never_complete(self):
        self.request("helpsteer_rm")
        self.model.reward_score = lambda *_: float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_stage()
        self.request("rm_bench")
        self.codec.encode_prompt = lambda *_: SimpleNamespace(input_ids=(10,))
        with self.assertRaisesRegex(ValueError, "length differs"):
            self.run_stage()
        self.assertEqual(self.model.calls, 0)

    def test_all_model_workloads_are_checked_before_base_generation(self):
        self.request()
        # All other accepted bindings are coherent, but a non-selected seed has
        # falsely labeled a short run complete. It must block even base scoring.
        record = self.roster["models"]["dpo44"]
        path = Path(record["artifact_dir"]) / "manifest.json"
        manifest = parse(path.read_bytes())
        manifest["completion"]["rows_processed"] = 16
        bad_sha = write(path, manifest)
        with self.assertRaisesRegex(ValueError, "row/update"):
            ex._formal(manifest)
        # Rebinding cannot make the incomplete formal workload acceptable.
        self.protocol["model_manifests"]["dpo44"]["sha256"] = bad_sha
        self.value["protocol_sha256"] = write(self.protocol_path, self.protocol)
        acceptance_path = Path(record["acceptance_path"])
        acceptance = parse(acceptance_path.read_bytes())
        acceptance["models"]["dpo44"]["manifest_sha256"] = bad_sha
        record["manifest_sha256"] = bad_sha
        record["acceptance_sha256"] = write(acceptance_path, acceptance)
        self.value["roster_sha256"] = write(self.roster_path, self.roster)
        self.request_sha = write(self.request_path, self.value)
        with self.assertRaisesRegex(ValueError, "row/update"):
            self.run_stage()
        self.assertEqual(self.model.calls, 0)

    def test_full_audit_flags_and_digests_are_required(self):
        self.request()
        audit = self.model.frozen_audit(full=True)
        for key, value in (("full", False), ("type", "wrong"), ("model_id", "sft"),
                           ("file_contents_rehashed", False), ("untracked_content_writes_checked", False),
                           ("base_sha256", "invalid"), ("adapter_sha256", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ex._check_audit({**audit, key: value}, "base")

    def test_process_identity_change_after_lock_rejects_load(self):
        self.request()
        original = self.system.processes
        calls = [0]
        def changed():
            calls[0] += 1
            rows = original()
            if calls[0] > 1:
                rows[0]["created_at_utc"] = "2026-10-20T01:00:00+00:00"
            return rows
        self.system.processes = changed
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.run_stage()
        self.assertEqual(self.model.calls, 0)
        self.assertFalse((self.root / "reports/.alignlab-gpu.lock").exists())

    def test_source_requirements_do_not_scan_for_future_unrelated_modules(self):
        self.request()
        with patch.object(Path, "glob", side_effect=AssertionError("Dynamic source scan forbidden")):
            ex._sources(self.value["source_sha256"])


if __name__ == "__main__":
    unittest.main()
