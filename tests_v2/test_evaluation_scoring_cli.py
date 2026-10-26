"""Artificial saved-output fixtures; these tests are not model quality results."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_protocol as ep
from scripts import score_alignlab_evaluation as cli

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/evaluation-assets-20261020T103505Z/ASSET_MANIFEST.json"
AUDIT = ROOT / "reports/alignlab-v2-data-audit-v3-p1024-r768-20261020"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_bytes(ep.canonical(value))
    return {"path": str(path), "sha256": sha(path)}


class SavedScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = ep.load_evaluation_assets(ASSETS, ep.ASSET_MANIFEST_SHA256)
        cls.holdout = ep.load_final_holdout(AUDIT, ep.HOLDOUT_MANIFEST_SHA256)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.models = {}
        for model_id in ("base", "sft", "reward", "dpo42", "ppo42", "dpo43", "ppo43", "dpo44", "ppo44"):
            model_dir = self.directory / model_id
            model_dir.mkdir()
            self.models[model_id] = save(model_dir / (
                "snapshot-manifest.json" if model_id == "base" else "manifest.json"),
                {"synthetic_fixture": True, "model_id": model_id})
        self.model_id = "reward"
        self.model = self.models[self.model_id]
        self.bind_roster()
        self.protocol = ep.build_evaluation_protocol(
            self.assets, self.holdout, model_manifests={
                name: (item["path"], item["sha256"]) for name, item in self.models.items()},
            scorer_seed=0, bootstrap_seed=73)
        self.protocol_file = save(self.directory / "protocol.json", self.protocol.identity)
        self.deadline = time.time() + 3600
        self.sources = {name: sha(ROOT / name) for name in (
            "scripts/score_alignlab_evaluation.py", "scripts/decode_alignlab_evaluation.py",
            "alignment_lab/evaluation_protocol.py", "alignment_lab/official_evaluation.py")}
        self.inference = {"model_id": self.model_id, "task": "helpsteer_rm",
                          "deadline_epoch": self.deadline, "source_sha256": self.sources,
                          "protocol_path": self.protocol_file["path"],
                          "protocol_sha256": self.protocol_file["sha256"],
                          "roster_path": self.roster["path"], "roster_sha256": self.roster["sha256"]}
        self.rows = []
        for row in self.holdout.rows:
            self.rows.append({
                "evaluation_id": row["evaluation_id"], "record_id": row["record_id"],
                "source_index": row["source_index"],
                "input_lengths": self.holdout.lengths[row["evaluation_id"]],
                "response1_score": -1.0 if row["overall_preference"] > 0 else 1.0,
                "response2_score": 0.0, "model_id": self.model_id,
                "model_manifest_sha256": self.model["sha256"],
                "protocol_sha256": self.protocol.sha256, "input_truncated": False})
        self.refresh()

    def bind_roster(self):
        accepted_rows, records = {}, {}
        for model_id, manifest in self.models.items():
            stage = model_id if model_id in ("base", "sft", "reward") else model_id[:3]
            if stage == "base":
                accepted_rows[model_id] = {"stage": stage, "artifact_dir": None,
                    "manifest_sha256": None, "progress_path": None, "progress_sha256": None}
            else:
                progress = save(self.directory / model_id / "progress.json", {
                    "synthetic_fixture": True, "formal_training": True,
                    "status": "formal_parent_stage_passed" if stage in ("sft", "reward")
                        else "formal_alignment_stage_passed",
                    "artifact": {"manifest_sha256": manifest["sha256"]}})
                accepted_rows[model_id] = {"stage": stage,
                    "artifact_dir": str(Path(manifest["path"]).parent),
                    "manifest_sha256": manifest["sha256"],
                    "progress_path": progress["path"], "progress_sha256": progress["sha256"]}
        accepted = save(self.directory / "fake-acceptance.json", {
            "kind": "alignlab-independent-evaluation-model-acceptance-v1",
            "status": "accepted_for_evaluation", "all_checks_passed": True,
            "processes_exited": True, "snapshot_manifest_sha256": self.models["base"]["sha256"],
            "models": accepted_rows})
        for model_id, row in accepted_rows.items():
            records[model_id] = {key: row[key] for key in ("stage", "artifact_dir", "manifest_sha256")}
            records[model_id].update(model_id=model_id, acceptance_path=accepted["path"],
                                     acceptance_sha256=accepted["sha256"])
        self.roster = save(self.directory / "fake-roster.json", {
            "schema_version": 1, "kind": "alignlab-accepted-evaluation-roster-v1",
            "model_snapshot": str(self.directory / "base"),
            "model_manifest_sha256": self.models["base"]["sha256"],
            "tokenizer_dir": str(self.directory), "models": records})

    def refresh(self, *, rows=None, changes=None):
        rows = self.rows if rows is None else rows
        files = {
            "request.snapshot.json": ep.canonical(self.inference),
            "protocol.snapshot.json": self.protocol._payload,
            "roster.snapshot.json": Path(self.roster["path"]).read_bytes(),
            "outputs.jsonl": b"\n".join(ep.canonical(row) for row in rows) + b"\n"}
        for name, payload in files.items():
            (self.directory / name).write_bytes(payload)
        self.manifest = {
            "schema_version": 1, "kind": "alignlab-evaluation-inference-result-v1",
            "status": "evaluation_inference_passed", "task": "helpsteer_rm",
            "model_id": self.model_id, "request": self.inference,
            "protocol": {**self.protocol_file, "identity_sha256": self.protocol.sha256},
            "accepted_roster": self.roster, "model_manifest": self.model,
            "dataset_identity": self.holdout.identity, "deadline_epoch": self.deadline,
            "inference_complete": True, "scoring_complete": False,
            "quality_evaluation_complete": False, "research_complete": False,
            "model_runtime": {"device": "cuda:0", "cuda": True, "weights_unchanged": True},
            "frozen_audits": [
                {"type": "alignlab-evaluation-frozen-audit-v1",
                 "unchanged": True, "full": True, "file_contents_rehashed": True,
                 "untracked_content_writes_checked": True, "model_id": self.model_id,
                 **{key: "1" * 64 for key in ("source_sha256", "base_sha256", "adapter_sha256", "head_sha256")}}
                for _ in range(2)],
            "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                      for name, payload in files.items()}}
        self.manifest.update(changes or {})
        info = save(self.directory / "manifest.json", self.manifest)
        self.request = {"schema_version": 1, "kind": "alignlab-official-cpu-scoring-request-v1",
                        "inference_manifest": info["path"], "inference_manifest_sha256": info["sha256"],
                        "source_sha256": self.sources, "decoder": None,
                        "deadline_epoch": self.deadline}
        self.request_path = self.directory / "score-request.json"
        save(self.request_path, self.request)

    def run_score(self):
        return cli.score_request(self.request_path, sha(self.request_path),
                                 output=self.directory / "scores")

    def test_full_existing_holdout_observations_including_duplicate_content(self):
        result = self.run_score()
        self.assertEqual(result["scores"]["denominators"],
                         {"all": 2017, "non_ties": 1920, "ties": 97})
        self.assertEqual(result["scores"]["pair_accuracy"], 1.0)  # Artificial fixture.
        self.assertIsNone(result["human_win_rate"])
        self.assertFalse(result["research_complete"])
        with self.assertRaises(FileExistsError):
            self.run_score()

    def test_missing_or_duplicate_observations_never_publish_scores(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            self.refresh(rows=rows)
            with self.assertRaises(ValueError):
                self.run_score()
            self.assertFalse((self.directory / "scores").exists())

    def test_truncation_or_changed_observation_identity_rejected(self):
        for field, value in (("input_truncated", True), ("source_index", -1),
                             ("model_manifest_sha256", "0" * 64)):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            self.refresh(rows=rows)
            with self.assertRaises(ValueError):
                self.run_score()

    def test_bad_bytes_or_incomplete_inference_rejected(self):
        (self.directory / "outputs.jsonl").write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_score()
        self.refresh(changes={"inference_complete": False})
        with self.assertRaises(ValueError):
            self.run_score()

    def test_mutable_weights_or_cpu_inference_receipt_rejected(self):
        for change in ({"frozen_audits": [{"unchanged": False}, {"unchanged": True}]},
                       {"model_runtime": {"device": "cpu", "cuda": False, "weights_unchanged": True}}):
            self.refresh(changes=change)
            with self.assertRaises(ValueError):
                self.run_score()

    def test_deadline_cannot_be_extended_and_expiration_is_not_success(self):
        self.request["deadline_epoch"] += 1
        save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "reset or extend"):
            self.run_score()
        self.refresh()
        with patch.object(cli.time, "time", return_value=self.deadline + 1):
            with self.assertRaises(TimeoutError):
                self.run_score()
        self.assertFalse((self.directory / "scores").exists())

    def test_source_or_protocol_changes_rejected_before_scoring(self):
        self.request["source_sha256"] = dict(self.sources)
        self.request["source_sha256"]["scripts/decode_alignlab_evaluation.py"] = "0" * 64
        save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_score()
        self.refresh()
        Path(self.protocol_file["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_score()

    def test_complete_ifeval_cli_uses_bound_real_cpu_decoder(self):
        tokenizer = ROOT / "models/qwen3-1.7b-tokenizer-70d244c/tokenizer.json"
        self.model_id = "sft"
        self.model = save(Path(self.models["sft"]["path"]), {"synthetic_fixture": True,
            "identity": {"inputs": {"tokenizer/tokenizer.json": {"sha256": sha(tokenizer)}}}})
        self.models["sft"] = self.model
        self.bind_roster()
        self.protocol = ep.build_evaluation_protocol(
            self.assets, self.holdout,
            model_manifests={name: (item["path"], item["sha256"]) for name, item in self.models.items()},
            scorer_seed=0, bootstrap_seed=73)
        self.protocol_file = save(Path(self.protocol_file["path"]), self.protocol.identity)
        self.inference.update(task="ifeval", protocol_sha256=self.protocol_file["sha256"],
                              model_id=self.model_id, roster_sha256=self.roster["sha256"])
        rows = [{"key": source["key"], "prompt": source["prompt"], "response": "(",
                 "prompt_tokens": self.assets.ifeval.lengths[str(source["key"])],
                 "generation_seed": 0, "generated_token_ids": [7, 151645], "termination": "eos",
                 "input_truncated": False, "model_id": self.model_id,
                 "model_manifest_sha256": self.model["sha256"], "protocol_sha256": self.protocol.sha256}
                for source in self.assets.ifeval.rows]
        self.refresh(rows=rows, changes={"task": "ifeval", "dataset_identity": self.assets.ifeval.identity})
        python = ROOT / ".runtime/alignlab-v2/Scripts/python.exe"
        script = ROOT / "scripts/decode_alignlab_evaluation.py"
        self.request["decoder"] = {
            "python": str(python), "python_sha256": sha(python),
            "script": str(script), "script_sha256": sha(script),
            "tokenizer_path": str(tokenizer), "tokenizer_sha256": sha(tokenizer),
            "tokenizers_version": "0.22.2"}
        save(self.request_path, self.request)
        def validate_without_claiming_quality(assets, outputs, *, protocol, model_id, decode_tokens):
            ordered = ep.validate_generation_records(
                assets.ifeval, outputs, protocol=protocol,
                model_id=model_id, decode_tokens=decode_tokens)
            return {"fixture_only": True, "validated_prompts": len(ordered)}
        with patch.object(cli, "score_ifeval", side_effect=validate_without_claiming_quality):
            result = self.run_score()
        self.assertEqual(result["scores"]["validated_prompts"], 541)
        self.assertFalse(result["decoder_receipt"]["torch_imported"])
        self.assertEqual(result["decoder_receipt"]["texts"], ["("] * 541)

    def test_partial_frozen_audit_and_unaccepted_roster_rejected(self):
        self.manifest["frozen_audits"][0]["full"] = False
        info = save(self.directory / "manifest.json", self.manifest)
        self.request["inference_manifest_sha256"] = info["sha256"]
        save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "Full before/after"):
            self.run_score()
        self.refresh()
        accepted_path = self.roster["path"]
        bad = ep.parse(Path(accepted_path).read_bytes())
        bad["models"].pop("ppo44")
        self.roster = save(Path(accepted_path), bad)
        self.inference["roster_sha256"] = self.roster["sha256"]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "complete preregistered"):
            self.run_score()


class CPUDecoderTests(unittest.TestCase):
    def test_real_tokenizer_exact_unicode_whitespace_special_ids_without_torch(self):
        from tokenizers import Tokenizer
        tokenizer_path = ROOT / "models/qwen3-1.7b-tokenizer-70d244c/tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        texts = ["  hello\nworld  ", "中文测试", "<|im_end|>", ""]
        sequences = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
        request = {"schema_version": 1, "tokenizer_path": str(tokenizer_path),
                   "tokenizer_sha256": sha(tokenizer_path), "tokenizers_version": "0.22.2",
                   "sequences": sequences}
        result = subprocess.run(
            [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B",
             str(ROOT / "scripts/decode_alignlab_evaluation.py")],
            input=ep.canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = json.loads(result.stdout)
        self.assertEqual(output["texts"], texts)
        self.assertFalse(output["torch_imported"])
        self.assertFalse(output["gpu_operations_performed"])

    def test_hash_and_out_of_vocabulary_rejected_by_real_subprocess(self):
        path = ROOT / "models/qwen3-1.7b-tokenizer-70d244c/tokenizer.json"
        request = {"schema_version": 1, "tokenizer_path": str(path),
                   "tokenizer_sha256": sha(path), "tokenizers_version": "0.22.2",
                   "sequences": [[999999999]]}
        for changes in ({}, {"tokenizer_sha256": "0" * 64}):
            result = subprocess.run(
                [str(ROOT / ".runtime/alignlab-v2/Scripts/python.exe"), "-I", "-B",
                 str(ROOT / "scripts/decode_alignlab_evaluation.py")],
                input=ep.canonical({**request, **changes}), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, check=False, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
