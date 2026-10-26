"""Real tiny CPU scoring for prefix migration; no production data/GPU training."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_lab import alignment_stage as stage, gpu_reference as gpu
from alignment_lab import reference_cache_recovery as recovery
from alignment_lab import reference_cache as storage
from alignment_lab.checkpointing import build_identity
from test_shared_qwen import fixture
from test_reference_bridge import identity_inputs, paired_batches


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def pin(path):
    return {"path": str(path), "sha256": sha(path)}


class ReferenceCacheRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / "old"
        self.old.mkdir()
        self.parts = fixture()
        self.backend = self.parts[0]
        self.pairs = [paired_batches(self.parts)]
        raw = identity_inputs(self.root / "input", self.backend)
        raw.pop("scoring_config")
        accepted_sha = raw.pop("accepted_dataset_manifest_sha256")
        manifest = self.root / "parent" / "manifest.json"
        write(manifest, {"fixture": "actual tiny reference weights, no production acceptance"})
        shutil.copyfile(next(iter(raw["adapter_files"].values())), manifest.parent / "weights.pt")
        raw["adapter_files"] = {"manifest": manifest, "weights": manifest.parent / "weights.pt"}
        for group, prefix in (("base_files", "base"), ("tokenizer_files", "tokenizer"), ("source_files", "source")):
            raw[group] = {prefix + "/" + key: path for key, path in raw[group].items()}
        audit = self.root / "audit" / "manifest.json"
        audit.parent.mkdir()
        shutil.copyfile(raw["dataset_manifest"], audit)
        raw["dataset_manifest"] = audit
        self.artifacts = raw
        self.receipt = gpu.capture_frozen_reference(self.backend, "reference", artifact_files=raw,
            accepted_data_manifest_sha256=accepted_sha, precision="float32", quantization="none", hash_chunk_bytes=128)
        self.config = stage.scoring_config(self.receipt, "tiny-fixture-v1")
        self.deadline = time.time() + 300
        self.request = {"stage": "dpo", "seed": 42, "deadline_epoch": self.deadline, "output": str(self.old),
            "source_sha256": {key[len("source/"):]: sha(path) for key, path in raw["source_files"].items()},
            "audit_dir": str(audit.parent), "audit_manifest_sha256": accepted_sha,
            "parents": {"sft": {"artifact_dir": str(manifest.parent), "manifest_sha256": sha(manifest)}},
            "reference_cache": {"mode": "create", "path": str(self.old / "reference-cache")}}
        self.model = {"fixture": "actual tiny unquantized CPU"}
        self.selection = {"serialization_version": self.config["template"]}
        inputs = {**raw["base_files"], **raw["tokenizer_files"], **raw["source_files"], "data/manifest": audit}
        self.import_identity = build_identity(inputs, {"stage": "dpo", "synthetic_rows": 0, "formal_training": True,
            "work": {"deadline_epoch": self.deadline, "chunk_tokens": 16}, "model": self.model,
            "selection": self.selection}, {"sft": manifest})
        self.journal = self.old / "reference-cache-records.jsonl"
        self.boundary = self.old / "cache-safe-boundary-000002-000002.json"
        self.chain = "0" * 64
        self.rows = []
        def commit(**values):
            row = {"schema_version": 1, "previous_sha256": self.chain, **values}
            self.chain = hashlib.sha256(recovery._canonical(row)).hexdigest()
            row["record_sha256"] = self.chain
            self.rows.append(json.loads(recovery._canonical(row)))
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if values["scored_sequences"] == 2:
                write(self.boundary, {"kind": "alignlab-safe-cache-record-boundary-v1", "complete": True,
                    "optimizer_updates": 0, "cache_journal": str(self.journal), "committed_record": row,
                    "identity": self.import_identity})
                raise InterruptedError("tiny safe scoring pause")
        with self.assertRaisesRegex(InterruptedError, "tiny safe"):
            stage.create_bounded_cache(self.receipt, self.old / "reference-cache", self.pairs,
                config=self.config, deadline_epoch=self.deadline, record_boundary=commit)
        self.total = self.rows[-1]["expected_unique_sequences"]
        self.progress = {"status": "paused", "stage": "dpo", "actual_optimizer_updates": 0,
            "request": self.request, "latest_safe_checkpoint": None, "checkpoints": [],
            "model": self.model, "selection": self.selection, "reference_anchor": self.receipt.anchor,
            "cache_scoring_progress": {"scored_sequences": 2, "expected_unique_sequences": self.total, "complete": False}}
        write(self.old / "request.json", self.request)
        write(self.old / "progress.json", self.progress)
        self.spec = {"journal": pin(self.journal), "boundary_receipt": pin(self.boundary),
            "request": pin(self.old / "request.json"), "progress": pin(self.old / "progress.json"),
            "scored_sequences": 2, "expected_unique_sequences": self.total}
        added = self.root / "added-wrapper.py"
        added.write_text("# New provenance, identical numerical scoring.\n")
        new_artifacts = {**raw, "source_files": {**raw["source_files"], "source/new-wrapper.py": added}}
        self.new_receipt = gpu.capture_frozen_reference(self.backend, "reference", artifact_files=new_artifacts,
            accepted_data_manifest_sha256=accepted_sha, precision="float32", quantization="none", hash_chunk_bytes=128)
        self.unique = stage.expected_cache_sequences(self.new_receipt, gpu._cache_identity(self.new_receipt, self.config, 16), self.pairs)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def refresh(self, name):
        self.spec[name] = pin(self.spec[name]["path"])

    def test_actual_suffix_only_scores_equal_uninterrupted_cache_and_preserve_old_bytes(self):
        old_bytes = {p: p.read_bytes() for p in self.old.iterdir()}
        reports, events = [], []
        original = gpu._sequence_scores
        with patch.object(gpu, "_sequence_scores", wraps=original) as score:
            continued = recovery.create_bounded_cache_from_prefix(self.new_receipt, self.root / "new-cache", self.pairs,
                prefix_spec=self.spec, config=self.config, deadline_epoch=self.deadline,
                record_boundary=lambda **values: events.append(values), migration_callback=reports.append)
            self.assertEqual(score.call_count, self.total - 2)
        continuous = stage.create_bounded_cache(self.new_receipt, self.root / "continuous-cache", self.pairs,
            config=self.config, deadline_epoch=self.deadline)
        self.assertEqual(dict(continued._cache.entries), dict(continuous._cache.entries))
        self.assertEqual([e["scored_sequences"] for e in events], list(range(self.total + 1)))
        self.assertEqual(len({e["record"]["key"] for e in events[1:]}), self.total)
        self.assertEqual({p: p.read_bytes() for p in self.old.iterdir()}, old_bytes)
        self.assertEqual(reports[0]["new_sequences_to_score"], self.total - 2)
        self.assertTrue(reports[0]["all_original_sources_unchanged"])
        self.assertNotEqual(reports[0]["old_identity_sha256"], reports[0]["new_identity_sha256"])
        self.assertFalse(reports[0]["optimizer_checkpoint"])
        # Explicitly migrated keys differ, while exact values/tokens remain unchanged.
        imported = {tuple(r["prompt_token_ids"] + r["completion_token_ids"]): r["sum_logprob"] for r in recovery.audit_prefix(self.spec).records}
        for entry in continued._cache.entries.values():
            key = tuple(entry.prompt_token_ids + entry.completion_token_ids)
            if key in imported:
                self.assertEqual(entry.sum_logprob, imported[key])

    def test_external_pins_reject_mutated_journal_or_receipt_before_scoring(self):
        for name in ("journal", "boundary_receipt", "progress", "request"):
            with self.subTest(name=name):
                path = Path(self.spec[name]["path"])
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                try:
                    with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                        recovery.audit_prefix(self.spec)
                finally:
                    path.write_bytes(before)

    def test_rehashed_chain_cannot_admit_gap_duplicate_wrong_key_or_invalid_score(self):
        baseline = copy.deepcopy(self.rows)
        boundary = json.loads(self.boundary.read_text())
        for mutation in ("gap", "duplicate", "key", "positive", "tokens", "count"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(baseline)
                if mutation == "gap": rows[1]["scored_sequences"] = 3
                if mutation == "duplicate": rows[2]["record"] = copy.deepcopy(rows[1]["record"])
                if mutation == "key": rows[1]["record"]["key"] = "f" * 64
                if mutation == "positive": rows[1]["record"]["sum_logprob"] = 1.0
                if mutation == "tokens": rows[1]["record"]["completion_token_ids"].append(12)
                if mutation == "count": rows[1]["record"]["token_count"] += 1
                chain = "0" * 64
                for row in rows:
                    row["previous_sha256"] = chain
                    row.pop("record_sha256", None)
                    chain = hashlib.sha256(recovery._canonical(row)).hexdigest()
                    row["record_sha256"] = chain
                self.journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
                write(self.boundary, {**boundary, "committed_record": rows[-1]})
                self.refresh("journal"); self.refresh("boundary_receipt")
                with self.assertRaises(ValueError):
                    recovery.audit_prefix(self.spec)

    def test_valid_subset_that_is_not_initial_sorted_prefix_is_rejected(self):
        wrong = dict(self.unique)
        first_key = next(iter(wrong))
        prompt, completion = wrong.pop(first_key)
        altered = (*completion[:-1], completion[-1] + 1)
        identity = gpu._cache_identity(self.new_receipt, self.config, 16)
        wrong[storage.cache_key(identity, prompt, altered)] = (prompt, altered)
        # Remove one actual imported sequence rather than relying on sort luck.
        old_record = recovery.audit_prefix(self.spec).records[0]
        imported_key = storage.cache_key(identity, old_record["prompt_token_ids"], old_record["completion_token_ids"])
        wrong = dict(self.unique)
        prompt, completion = wrong.pop(imported_key)
        altered = (*completion[:-1], completion[-1] + 1)
        wrong[storage.cache_key(identity, prompt, altered)] = (prompt, altered)
        with self.assertRaisesRegex(ValueError, "exact initial prefix"):
            recovery.verified_prefix_records(self.spec, self.new_receipt, wrong, config=self.config)

    def test_actual_reference_anchor_or_protocol_difference_rejected(self):
        wrong = dict(self.config, serialization="different serialization")
        with self.assertRaisesRegex(ValueError, "Actual frozen"):
            recovery.verified_prefix_records(self.spec, self.new_receipt, self.unique, config=wrong)
        with torch.no_grad():
            self.backend._parameters["reference"][0].add_(0.01)
        with self.assertRaises(RuntimeError):
            recovery.verified_prefix_records(self.spec, self.new_receipt, self.unique, config=self.config)

    def test_original_source_change_is_rejected_not_treated_as_new_wrapper(self):
        source = next(iter(self.artifacts["source_files"].values()))
        # Only fixture source is touched; all repository sources remain unchanged.
        self.assertTrue(Path(source).is_relative_to(self.root))
        Path(source).write_text("# changed historical scorer\n")
        with self.assertRaises(ValueError):
            recovery.audit_prefix(self.spec)

    def test_old_completion_or_nonzero_optimizer_cannot_be_reused_as_prefix(self):
        for update in ({"actual_optimizer_updates": 1}, {"status": "completed"}, {"latest_safe_checkpoint": {"fake": True}}):
            write(self.old / "progress.json", {**self.progress, **update})
            self.refresh("progress")
            with self.assertRaisesRegex(ValueError, "paused zero-optimizer"):
                recovery.audit_prefix(self.spec)
        write(self.old / "progress.json", self.progress)
        self.refresh("progress")
        (self.old / "reference-cache").mkdir()
        with self.assertRaisesRegex(ValueError, "existing final/partial"):
            recovery.audit_prefix(self.spec)

    def test_new_output_nooverwrite_and_original_deadline_enforced_before_replay(self):
        target = self.root / "existing"
        target.mkdir()
        marker = target / "untouched"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            recovery.create_bounded_cache_from_prefix(self.new_receipt, target, self.pairs,
                prefix_spec=self.spec, config=self.config, deadline_epoch=self.deadline)
        self.assertEqual(marker.read_text(), "keep")
        events = []
        with self.assertRaisesRegex(ValueError, "exact original absolute deadline"):
            recovery.create_bounded_cache_from_prefix(self.new_receipt, self.root / "wrong-budget", self.pairs,
                prefix_spec=self.spec, config=self.config, deadline_epoch=self.deadline + 1,
                record_boundary=lambda **v: events.append(v))
        self.assertFalse(events)
        self.assertFalse((self.root / "wrong-budget").exists())
        with self.assertRaisesRegex(ValueError, "outside the historical run"):
            recovery.create_bounded_cache_from_prefix(self.new_receipt, self.old / "new-cache", self.pairs,
                prefix_spec=self.spec, config=self.config, deadline_epoch=self.deadline)
        self.assertFalse((self.old / "new-cache").exists())

    def test_stop_or_deadline_during_import_prevents_all_new_scoring(self):
        for stop_mode in (True, False):
            with self.subTest(stop=stop_mode):
                state = {"count": 0}
                def callback(**values): state["count"] = values["scored_sequences"]
                def stopped(): return stop_mode and state["count"] >= 1
                def clock(): return self.deadline if not stop_mode and state["count"] >= 1 else self.deadline - 1
                with patch.object(gpu, "_sequence_scores") as score:
                    with self.assertRaises((InterruptedError, TimeoutError)):
                        recovery.create_bounded_cache_from_prefix(self.new_receipt, self.root / ("stopped" + str(stop_mode)), self.pairs,
                            prefix_spec=self.spec, config=self.config, deadline_epoch=self.deadline,
                            stop_requested=stopped, clock=clock, record_boundary=callback)
                    score.assert_not_called()
                self.assertEqual(state["count"], 1)


if __name__ == "__main__":
    unittest.main()

