"""Pure stdlib fixtures: integrity checks only, never real reference scoring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.reference_cache import (
    build_identity, cache_key, identity_sha256, load_cache, write_or_reuse_cache,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "new-cache"
        for name in ("base.bin", "adapter.bin", "tokenizer.json", "scorer.py"):
            (self.root / name).write_bytes(("Fixture only: " + name).encode())
        self.dataset = self.root / "accepted-data-manifest.json"
        self.dataset.write_text(json.dumps({"status": "audit_complete", "data_ready": True,
                                           "readiness_blockers": [], "training_started": False}))
        self.arguments = {
            "base_files": {"weights": self.root / "base.bin"},
            "adapter_files": {"reference": self.root / "adapter.bin"},
            "tokenizer_files": {"tokenizer": self.root / "tokenizer.json"},
            "source_files": {"scorer": self.root / "scorer.py"},
            "dataset_manifest": self.dataset,
            "accepted_dataset_manifest_sha256": sha(self.dataset),
            "scoring_config": {"template": "fixture-template", "serialization": "fixture-serialization-v1",
                               "precision": "float32", "quantization": {"mode": "none"},
                               "library_versions": {"python": sys.version.split()[0], "fictional_scorer": "fixture-v1"},
                               "truncation": "none", "eos_is_scored": True},
        }
        self.identity = build_identity(**self.arguments)
        self.rows = [self.row(), self.row(completion_token_ids=[13, 14, 2], token_count=3, sum_logprob=-7.25)]

    def row(self, **changes):
        return dict({"prompt_token_ids": [1, 8, 9], "completion_token_ids": [11, 2],
                     "sum_logprob": -4.5, "token_count": 2}, **changes)

    def create(self):
        return write_or_reuse_cache(self.output, self.identity, self.rows)

    def snapshot(self):
        return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.output.iterdir()}

    def replace_scores_and_rehash(self, payload):
        scores = self.output / "scores.jsonl"
        scores.write_bytes(payload)
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["scores.jsonl"] = {"sha256": sha(scores), "size_bytes": len(payload)}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_round_trip_and_exact_token_lookup(self):
        cache = self.create()
        self.assertFalse(cache.reused)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(cache.lookup([1, 8, 9], [11, 2]).sum_logprob, -4.5)
        self.assertEqual(cache.lookup([1, 8, 9], [11, 2]).token_count, 2)
        self.assertEqual(cache.identity, self.identity)
        self.assertEqual(cache.manifest_sha256, sha(self.output / "manifest.json"))
        with self.assertRaises(KeyError):
            cache.lookup([1, 8, 10], [11, 2])
        with self.assertRaises(TypeError):
            cache.entries["invented"] = None
        detached = cache.identity
        detached["scoring_config"]["precision"] = "changed"
        self.assertNotEqual(detached, cache.identity)

    def test_existing_cache_is_verified_without_touching_files(self):
        first = self.create()
        before = self.snapshot()
        second = write_or_reuse_cache(self.output, self.identity, expected_manifest_sha256=first.manifest_sha256)
        third = write_or_reuse_cache(self.output, self.identity, reversed(self.rows))
        self.assertTrue(second.reused and third.reused)
        self.assertEqual(before, self.snapshot())

    def test_conflicting_scores_never_overwrite_existing_cache(self):
        self.create()
        before = self.snapshot()
        changed = [dict(self.rows[0], sum_logprob=-99), self.rows[1]]
        with self.assertRaisesRegex(ValueError, "scores differ"):
            write_or_reuse_cache(self.output, self.identity, changed)
        self.assertEqual(before, self.snapshot())

    def test_missing_new_scores_cannot_create_or_claim_computation(self):
        with self.assertRaisesRegex(ValueError, "does not compute"):
            write_or_reuse_cache(self.output, self.identity)
        self.assertFalse(self.output.exists())

    def test_empty_scores_and_duplicate_token_keys_rejected_before_writing(self):
        for rows in ([], [self.row(), self.row()], [dict(self.row(), key="0" * 64)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                write_or_reuse_cache(self.output, self.identity, rows)
            self.assertFalse(self.output.exists())

    def test_illegal_scores_rejected(self):
        for value in (None, "", "-1", True, [], float("nan"), float("inf"), -float("inf"), 0.01, 10 ** 1000):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                write_or_reuse_cache(self.output, self.identity, [self.row(sum_logprob=value)])
        self.assertFalse(self.output.exists())

    def test_zero_logprob_is_finite_and_allowed(self):
        cache = write_or_reuse_cache(self.output, self.identity, [self.row(sum_logprob=0)])
        self.assertEqual(cache.lookup([1, 8, 9], [11, 2]).sum_logprob, 0)

    def test_invalid_ids_or_token_counts_rejected(self):
        invalid = [{"prompt_token_ids": []}, {"completion_token_ids": []}, {"prompt_token_ids": "123"},
                   {"completion_token_ids": [1, True]}, {"completion_token_ids": [1, -2]},
                   {"prompt_token_ids": [1.0]}, {"token_count": 0}, {"token_count": 1},
                   {"token_count": 2.0}, {"token_count": True}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                write_or_reuse_cache(self.output, self.identity, [self.row(**changes)])
        self.assertFalse(self.output.exists())

    def test_key_covers_token_values_order_and_prompt_completion_boundary(self):
        key = cache_key(self.identity, [1, 2], [3, 4])
        for prompt, completion in (([1, 9], [3, 4]), ([1, 2], [3, 5]), ([2, 1], [3, 4]), ([1], [2, 3, 4])):
            self.assertNotEqual(key, cache_key(self.identity, prompt, completion))

    def test_complete_scoring_config_changes_key_and_prevents_reuse(self):
        self.create()
        key = cache_key(self.identity, [1], [2])
        changes = {"template": "new-template", "serialization": "new-serialization", "precision": "bfloat16",
                   "quantization": {"mode": "nf4"}, "library_versions": {"scorer": "changed"},
                   "truncation": "left", "eos_is_scored": False}
        before = self.snapshot()
        for name, value in changes.items():
            config = dict(self.arguments["scoring_config"], **{name: value})
            updated = build_identity(**dict(self.arguments, scoring_config=config))
            self.assertNotEqual(key, cache_key(updated, [1], [2]))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "different scoring identity"):
                load_cache(self.output, updated)
        self.assertEqual(before, self.snapshot())

    def test_actual_model_adapter_tokenizer_and_source_changes_are_rejected(self):
        self.create()
        for name in ("base.bin", "adapter.bin", "tokenizer.json", "scorer.py"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"modified")
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Frozen input file changed"):
                    load_cache(self.output, self.identity)
                updated = build_identity(**self.arguments)
                self.assertNotEqual(identity_sha256(updated), identity_sha256(self.identity))
            finally:
                path.write_bytes(original)

    def test_acceptance_requires_explicit_matching_digest_and_ready_manifest(self):
        with self.assertRaisesRegex(ValueError, "explicitly accepted digest"):
            build_identity(**dict(self.arguments, accepted_dataset_manifest_sha256="0" * 64))
        for changes in ({"data_ready": False}, {"readiness_blockers": ["not accepted"]},
                        {"status": "in_progress"}, {"training_started": True}):
            manifest = {"status": "audit_complete", "data_ready": True,
                        "readiness_blockers": [], "training_started": False, **changes}
            self.dataset.write_text(json.dumps(manifest))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "accepted, ready audit"):
                build_identity(**dict(self.arguments, accepted_dataset_manifest_sha256=sha(self.dataset)))

    def test_changed_dataset_manifest_rejected_with_old_identity(self):
        self.create()
        self.dataset.write_bytes(self.dataset.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen input file changed"):
            load_cache(self.output, self.identity)

    def test_explicit_base_only_reference_allowed_but_missing_other_groups_rejected(self):
        identity = build_identity(**dict(self.arguments, adapter_files={}))
        self.assertEqual(identity["files"]["adapter"], {})
        for group in ("base_files", "tokenizer_files", "source_files"):
            with self.subTest(group=group), self.assertRaises(ValueError):
                build_identity(**dict(self.arguments, **{group: {}}))

    def test_incomplete_nonfinite_or_ambiguous_configuration_rejected(self):
        invalid = [dict(self.arguments["scoring_config"], temperature=float("nan")),
                   dict(self.arguments["scoring_config"], precision=" "),
                   dict(self.arguments["scoring_config"], quantization=None),
                   dict(self.arguments["scoring_config"], library_versions={}),
                   dict(self.arguments["scoring_config"], library_versions={"scorer": ""}),
                   dict(self.arguments["scoring_config"], additional={1: "non-string-key"})]
        for key in ("template", "serialization", "precision", "quantization", "library_versions"):
            removed = dict(self.arguments["scoring_config"])
            del removed[key]
            invalid.append(removed)
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_identity(**dict(self.arguments, scoring_config=config))

    def test_artifact_tampering_is_detected(self):
        self.create()
        for name in ("identity.json", "scores.jsonl"):
            path = self.output / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                load_cache(self.output, self.identity)
            path.write_bytes(original)

    def test_manifest_wrong_count_and_artifact_inventory_rejected(self):
        self.create()
        path = self.output / "manifest.json"
        original = path.read_bytes()
        for changes in ({"record_count": 1}, {"record_count": True}, {"artifacts": {}},
                        {"identity_sha256": "0" * 64}, {"status": "running"}):
            manifest = dict(json.loads(original), **changes)
            path.write_text(json.dumps(manifest))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_cache(self.output, self.identity)
        path.write_bytes(original)

    def test_pinned_manifest_rejects_coordinated_score_and_manifest_rewrite(self):
        cache = self.create()
        rows = [json.loads(line) for line in (self.output / "scores.jsonl").read_text().splitlines()]
        rows[0]["sum_logprob"] = -123.0
        self.replace_scores_and_rehash(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            load_cache(self.output, self.identity, expected_manifest_sha256=cache.manifest_sha256)

    def test_rehashed_invalid_tokens_key_duplicate_and_nonfinite_rows_still_fail(self):
        self.create()
        rows = [json.loads(line) for line in (self.output / "scores.jsonl").read_text().splitlines()]
        variants = [[dict(rows[0], token_count=99), rows[1]], [dict(rows[0], key="0" * 64), rows[1]],
                    [rows[0], rows[0]], [dict(rows[0], sum_logprob=float("nan")), rows[1]],
                    [{key: value for key, value in rows[0].items() if key != "key"}, rows[1]]]
        for changed in variants:
            self.replace_scores_and_rehash(b"".join(json.dumps(row).encode() + b"\n" for row in changed))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                load_cache(self.output, self.identity)

    def test_duplicate_json_fields_rejected_even_if_rehashed(self):
        self.create()
        scores = (self.output / "scores.jsonl").read_bytes()
        scores = scores.replace(b'"token_count":', b'"token_count":999,"token_count":', 1)
        self.replace_scores_and_rehash(scores)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            load_cache(self.output, self.identity)

    def test_untracked_files_directories_and_partial_output_never_reused_or_repaired(self):
        self.output.mkdir()
        for extra in ("untracked.txt", "manifest.json"):
            (self.output / extra).write_bytes(b"untracked fixture")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "incomplete, untracked"):
            write_or_reuse_cache(self.output, self.identity, self.rows)
        self.assertEqual(before, self.snapshot())
        for path in self.output.iterdir():
            path.unlink()
        self.output.rmdir()
        self.create()
        extra = self.output / "unexpected-directory"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "incomplete, untracked"):
            load_cache(self.output, self.identity)
        extra.rmdir()
        (self.output / "scores.jsonl").unlink()
        with self.assertRaises(ValueError):
            write_or_reuse_cache(self.output, self.identity, self.rows)
        self.assertFalse((self.output / "scores.jsonl").exists())

    def test_nonexistent_inputs_and_unknown_record_fields_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            build_identity(**dict(self.arguments, source_files={"missing": self.root / "does-not-exist"}))
        with self.assertRaises(ValueError):
            write_or_reuse_cache(self.output, self.identity, [dict(self.row(), hidden_override=True)])


if __name__ == "__main__":
    unittest.main()
