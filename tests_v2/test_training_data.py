"""Small local-tokenizer fixtures; no downloaded weights, CUDA or full audit."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment_lab import training_data as td
from alignment_lab.chat_codec import Qwen3ChatCodec, TERMINAL_EOS_TOKEN_ID
from alignment_lab.data_audit import parse_row


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer_dir = td.ROOT / "models/qwen3-1.7b-tokenizer-70d244c"
        if not (cls.tokenizer_dir / "tokenizer.json").exists():
            raise unittest.SkipTest("Pinned local tokenizer unavailable; test never downloads")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_dir, local_files_only=True,
                                                     trust_remote_code=False)
        cls.codec = Qwen3ChatCodec(cls.tokenizer)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audit_dir = self.root / "accepted-audit"
        self.audit_dir.mkdir()
        rows = []
        for index, (question, a, b, score) in enumerate((
            ("First distinct question?", "\n\nA full response.", "", -2),
            ("Second independent question?", "Incorrect alternative.", "中文和 cafe\u0301。", 3),
        )):
            row = parse_row({
                "context": [{"role": "user", "content": question}],
                "response1": a, "response2": b, "overall_preference": score,
                "domain": "general", "language": "english",
                "individual_preference": [{"score": score}],
            }, "train", index)
            row = {k: v for k, v in row.items() if not k.startswith("_")}
            row["cluster_id"] = row["prompt_id"]
            row["local_split"] = "train"
            prompt = self.codec.encode_prompt(row["context"])
            row["tokens"] = {
                "prompt": len(prompt.input_ids),
                "response1": len(self.codec.encode_completion(prompt, a).completion_ids),
                "response2": len(self.codec.encode_completion(prompt, b).completion_ids),
            }
            rows.append(row)
        self.rows = rows
        self.write_rows("preference_train", rows)
        self.write_rows("sft_train", rows)
        self.write_rows("ppo_prompts", [{k: r[k] for k in td.PPO_FIELDS} for r in rows])
        for name in td.ARTIFACTS - {"preference_train.jsonl", "sft_train.jsonl", "ppo_prompts.jsonl", "audit.json"}:
            # Invalid JSON deliberately proves holdout rows are never parsed.
            (self.audit_dir / name).write_bytes(b"Integrity bytes only, not training JSON.\n")
        self.audit = {
            "data_ready": True, "readiness_blockers": [], "training_started": False,
            "partition_statistics": {
                name: {"rows": 2, "independent_prompt_clusters": 2, "unique_normalized_contexts": 2}
                for name in td.PARTITIONS
            },
        }
        (self.audit_dir / "audit.json").write_bytes(canonical(self.audit))
        inputs = {}
        for name in ("train", "validation"):
            path = self.root / f"{name}.jsonl"
            path.write_bytes(b'{"fixture":true}\n')
            inputs[name] = {"path": str(path), "sha256": sha(path)}
        self.identity = {
            "version": "alignlab-data-audit-v3-native-nfc",
            "serialization_version": td.SERIALIZATION_VERSION,
            "config": {"prompt_limit": 1024, "response_limit": 768, "minimum_pairs": 2,
                       "minimum_clusters": 2, "minimum_sft": 2},
            "source_revision": "a" * 40, "model_tokenizer_revision": "b" * 40,
            "inputs": inputs,
            "tokenizer": {"path": str(self.tokenizer_dir), "files": {
                p.relative_to(self.tokenizer_dir).as_posix(): sha(p)
                for p in self.tokenizer_dir.rglob("*")
                if p.is_file() and not any(s.startswith(".") for s in p.relative_to(self.tokenizer_dir).parts)
            }},
            "tokenizer_library_versions": {
                name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")
            },
            "source_sha256": {name: sha(td.ROOT / name) for name in td.SOURCES},
            "protocol_sha256": sha(td.ROOT / "docs/ALIGNLAB_V2.md"),
        }
        self.manifest = {"identity": self.identity, "identity_sha256": "", "artifacts": {},
                         "data_ready": True, "readiness_blockers": [], "training_started": False,
                         "status": "audit_complete"}
        self.seal()

    def write_rows(self, name, rows):
        (self.audit_dir / f"{name}.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in rows))

    def seal(self):
        self.manifest["identity_sha256"] = hashlib.sha256(canonical(self.identity)).hexdigest()
        self.manifest["artifacts"] = {name: sha(self.audit_dir / name) for name in td.ARTIFACTS}
        (self.audit_dir / "manifest.json").write_bytes(canonical(self.manifest))
        self.anchor = sha(self.audit_dir / "manifest.json")

    def load(self, partition="preference_train", codec=None):
        return td.load_training_partition(self.audit_dir, partition, self.anchor, codec)

    def test_three_partitions_are_immutable_and_preserve_exact_eos_and_nfc(self):
        before = {p.name: (sha(p), p.stat().st_mtime_ns) for p in self.audit_dir.iterdir()}
        preference = self.load(codec=self.codec)
        first = preference.rows[0]
        self.assertIsInstance(preference.rows, tuple)
        self.assertEqual(first.rejected_ids, (TERMINAL_EOS_TOKEN_ID,))
        self.assertTrue(self.tokenizer.decode(first.chosen_ids).startswith("\n\n"))
        self.assertTrue(self.tokenizer.decode(preference.rows[1].chosen_ids).startswith("中文和 café"))
        for row in preference.rows:
            self.assertEqual(row.chosen_ids[-1], TERMINAL_EOS_TOKEN_ID)
            self.assertEqual(row.chosen_ids.count(TERMINAL_EOS_TOKEN_ID), 1)
        with self.assertRaises(FrozenInstanceError):
            first.record_id = "changed"
        detached = preference.dataset_identity
        detached["config"]["prompt_limit"] = 1
        self.assertEqual(preference.dataset_identity["config"]["prompt_limit"], 1024)
        sft = self.load("sft_train")
        ppo = self.load("ppo_prompts")
        self.assertEqual(sft.rows[0].chosen_ids, first.chosen_ids)
        self.assertIsNone(sft.rows[0].rejected_ids)
        self.assertIsNone(ppo.rows[0].chosen_ids)
        self.assertEqual(ppo.rows[0].prompt_ids, first.prompt_ids)
        self.assertEqual(before, {p.name: (sha(p), p.stat().st_mtime_ns) for p in self.audit_dir.iterdir()})

    def test_holdout_paths_rejected_before_any_file_read(self):
        for partition in ("final_holdout", "tuning_dev", "diagnostic_holdout",
                          "preference_train.jsonl", "../preference_train"):
            with self.subTest(partition=partition), patch.object(td, "_file_hash", side_effect=AssertionError("read")):
                with self.assertRaisesRegex(ValueError, "Only"):
                    self.load(partition)

    def test_wrong_manifest_anchor_and_artifact_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash differs"):
            td.load_training_partition(self.audit_dir, "preference_train", "0" * 64)
        (self.audit_dir / "final_holdout.jsonl").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.load()

    def test_identity_self_hash_is_verified_even_with_new_manifest_anchor(self):
        self.manifest["identity_sha256"] = "0" * 64
        (self.audit_dir / "manifest.json").write_bytes(canonical(self.manifest))
        self.anchor = sha(self.audit_dir / "manifest.json")
        with self.assertRaisesRegex(ValueError, "self-hash"):
            self.load()

    def test_manifest_and_hashed_report_readiness_both_required(self):
        self.audit["readiness_blockers"] = ["fixture blocker"]
        (self.audit_dir / "audit.json").write_bytes(canonical(self.audit))
        self.seal()
        with self.assertRaisesRegex(ValueError, "no blockers"):
            self.load()
        self.manifest["status"] = "running"
        self.seal()
        with self.assertRaisesRegex(ValueError, "audit_complete"):
            self.load()

    def test_real_input_source_tokenizer_protocol_and_library_identity_checked(self):
        original = copy.deepcopy(self.identity)
        for group in ("input", "source", "tokenizer", "protocol", "library"):
            self.identity = copy.deepcopy(original)
            self.manifest["identity"] = self.identity
            if group == "input":
                self.identity["inputs"]["train"]["sha256"] = "0" * 64
            elif group == "source":
                self.identity["source_sha256"]["alignment_lab/chat_codec.py"] = "0" * 64
            elif group == "tokenizer":
                self.identity["tokenizer"]["files"]["tokenizer.json"] = "0" * 64
            elif group == "protocol":
                self.identity["protocol_sha256"] = "0" * 64
            else:
                self.identity["tokenizer_library_versions"]["tokenizers"] = "0.0.0"
            self.seal()
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "hash differs|version drift"):
                self.load()

    def test_wrong_codec_backend_and_template_are_not_trusted(self):
        for kind in ("vocabulary", "template"):
            tokenizer = copy.deepcopy(self.tokenizer)
            if kind == "vocabulary":
                tokenizer.add_tokens(["never-audited-fixture-token-1234567"])
            else:
                tokenizer.chat_template += "\n"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Supplied codec differs"):
                self.load(codec=Qwen3ChatCodec(tokenizer))
        class Impostor(Qwen3ChatCodec):
            def encode_prompt(self, context):
                raise AssertionError("Impostor must never encode")
        with self.assertRaisesRegex(ValueError, "Supplied codec differs"):
            self.load(codec=Impostor(self.tokenizer))

    def test_wrong_local_split_and_preference_sign_rejected(self):
        for key, value in (("local_split", "tuning_dev"), ("source_split", "validation"), ("chosen", "forged")):
            changed = copy.deepcopy(self.rows)
            changed[0][key] = value
            self.write_rows("preference_train", changed)
            self.seal()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Only source train|preference sign"):
                self.load()

    def test_token_length_drift_and_accepted_limits_rejected(self):
        for key in ("prompt", "response1", "response2"):
            changed = copy.deepcopy(self.rows)
            changed[0]["tokens"][key] += 1
            self.write_rows("preference_train", changed)
            self.seal()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "length drift"):
                self.load()
        self.write_rows("preference_train", self.rows)
        self.identity["config"]["response_limit"] = 1
        self.seal()
        with self.assertRaisesRegex(ValueError, "length limits"):
            self.load()

    def test_ppo_is_verified_projection_and_sft_is_verified_subset(self):
        for partition in ("ppo_prompts", "sft_train"):
            rows = ([{k: r[k] for k in td.PPO_FIELDS} for r in self.rows]
                    if partition == "ppo_prompts" else copy.deepcopy(self.rows))
            rows[0]["source_split"] = "validation"
            self.write_rows(partition, rows)
            self.seal()
            with self.subTest(partition=partition), self.assertRaisesRegex(ValueError, "exact member/projection"):
                self.load(partition)

    def test_explicit_reserved_controls_and_minimum_scale_are_not_silently_accepted(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["response2"] = "<|endoftext|>"
        changed[0]["rejected"] = changed[0]["response2"]
        payload = {k: changed[0][k] for k in ("context", "response1", "response2", "overall_preference",
                                            "domain", "language", "individual_scores")}
        changed[0]["record_id"] = hashlib.sha256(canonical(payload)).hexdigest()
        self.write_rows("preference_train", changed)
        self.seal()
        with self.assertRaisesRegex(ValueError, "reserved_tokenizer"):
            self.load()
        self.write_rows("preference_train", self.rows)
        self.identity["config"]["minimum_pairs"] = 3
        self.seal()
        with self.assertRaisesRegex(ValueError, "minimum scale"):
            self.load()

    def test_mid_encoding_source_change_rejected_without_returning_partial_rows(self):
        original = td.Qwen3ChatCodec.encode_completion
        changed = False
        def mutate(codec, prompt, response):
            nonlocal changed
            result = original(codec, prompt, response)
            if not changed:
                Path(self.identity["inputs"]["train"]["path"]).write_bytes(b"changed during encoding")
                changed = True
            return result
        with patch.object(td.Qwen3ChatCodec, "encode_completion", mutate):
            with self.assertRaisesRegex(ValueError, "hash differs"):
                self.load()


if __name__ == "__main__":
    unittest.main()
