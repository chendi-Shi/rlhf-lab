"""CPU data-isolation, label, boundary and immutable-output regression tests."""
from array import array
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment_lab.data_audit import (
    AuditConfig, LocalTokenCounter, audit_records, build_clusters, grouped_split,
    digest, file_hash, jaccard_at_least, parse_row, run_audit, shingles, verify_existing, training_length_grid,
)


def sample(question="Explain a distinct initial question with sufficient contextual detail.", score=-2, **changes):
    result = {"domain": "general", "language": "English", "context": [{"role": "user", "content": question}],
              "response1": "First candidate answer.", "response2": "A different candidate answer.",
              "overall_preference": score,
              "individual_preference": [{"score": score}, {"score": score}, {"score": score}]}
    result.update(changes)
    return result


def config(**changes):
    return replace(AuditConfig(), expected_train=None, expected_validation=None, minimum_pairs=0,
                   minimum_clusters=0, minimum_sft=0, diagnostic_clusters=0, **changes)


class LabelsTest(unittest.TestCase):
    def test_negative_selects_response1_and_no_vote_override(self):
        raw = sample(score=-1, individual_preference=[{"score": 3}, {"score": 3}, {"score": -3}])
        row = parse_row(raw, "train", 0)
        self.assertEqual(row["chosen"], raw["response1"])
        self.assertEqual(row["annotation_summary"]["positive_votes"], 2)
        self.assertEqual(parse_row(sample(score=3), "train", 0)["chosen"], raw["response2"])

    def test_ties_preserved_without_binary_labels(self):
        row = parse_row(sample(score=0), "train", 0)
        self.assertIsNone(row["chosen"])
        self.assertIsNone(row["rejected"])
        parts, report = audit_records([sample(score=0)], [], config())
        self.assertEqual(len(parts["ties"]), 1)
        self.assertEqual(parts["preference_train"], [])
        self.assertFalse(report["data_ready"])

    def test_invalid_labels_and_unsupported_template_input_rejected(self):
        for value in (True, -4, 4, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_row(sample(overall_preference=value), "train", 0)
        for raw in (sample(response1="<think>hidden</think>Answer"),
                    sample(response1=None),
                    sample(context=[{"role": "tool", "content": "x"}]),
                    sample(context=[{"role": "user", "content": [{"type": "image"}]}]),
                    sample(context=[{"role": "user", "content": "x", "tool_calls": []}])):
            with self.assertRaises(ValueError):
                parse_row(raw, "train", 0)


class IsolationTest(unittest.TestCase):
    def test_official_validation_wins_exact_and_user_prefix_conflicts(self):
        initial = "Please give a complete explanation of this particular historical situation in detail."
        followup = sample(context=[{"role": "user", "content": initial},
                                   {"role": "assistant", "content": "A previous answer."},
                                   {"role": "user", "content": "Continue with examples."}])
        parts, report = audit_records([sample(initial), followup], [sample(initial)], config())
        self.assertEqual(len(parts["final_holdout"]), 1)
        self.assertFalse(parts["eligible_train_pool"])
        self.assertEqual(report["exclusion_reasons"]["cross_official_split_cluster"], 2)
        self.assertTrue(all(value == 0 for value in report["cluster_overlaps"].values()))

    def test_near_duplicate_across_official_splits_quarantines_train(self):
        words = [f"topic{i}" for i in range(100)]
        modified = words.copy()
        modified[50] = "substitution"
        parts, report = audit_records([sample(" ".join(words))], [sample(" ".join(modified))], config())
        self.assertEqual(report["near_duplicates"]["successful_unions"]["near_jaccard"], 1)
        self.assertEqual(len(parts["final_holdout"]), 1)
        self.assertFalse(parts["preference_train"])

    def test_shared_system_prompt_does_not_merge_unrelated_users(self):
        first = sample(context=[{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "Explain mitochondria."}])
        second = sample(context=[{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": "Solve this algebra."}])
        rows = [parse_row(first, "train", 0), parse_row(second, "validation", 0)]
        build_clusters(rows, config())
        self.assertNotEqual(rows[0]["cluster_id"], rows[1]["cluster_id"])

    def test_largest_clusters_report_counts_without_text(self):
        rows = [parse_row(sample(), "train", 0), parse_row(sample(), "validation", 0)]
        report = build_clusters(rows, config())
        largest = report["largest_ten_clusters"][0]
        self.assertEqual(largest["rows"], 2)
        self.assertEqual(largest["source_splits"], {"train": 1, "validation": 1})
        self.assertNotIn("context", largest)

    def test_conflicting_duplicate_pair_is_quarantined(self):
        parts, report = audit_records([sample(score=-2), sample(score=2)], [], config())
        self.assertEqual(report["exclusion_reasons"]["contradictory_duplicate_pair"], 2)
        self.assertFalse(parts["preference_train"])

    def test_split_stable_under_input_order_and_diagnostic_clusters_disjoint(self):
        records = [sample(f"Unique question token{i} subject{i} scenario{i} explanation{i}") for i in range(80)]
        cfg = replace(config(near_min_chars=10000), diagnostic_clusters=3)
        first, _ = audit_records(records, [], cfg)
        shuffled = records.copy()
        random.Random(42).shuffle(shuffled)
        second, _ = audit_records(shuffled, [], cfg)
        for name in ("preference_train", "tuning_dev", "diagnostic_holdout"):
            self.assertEqual({r["record_id"] for r in first[name]}, {r["record_id"] for r in second[name]})
        self.assertEqual(len(first["diagnostic_holdout"]), 3)
        sets = [{r["cluster_id"] for r in first[name]} for name in ("preference_train", "tuning_dev", "diagnostic_holdout")]
        self.assertTrue(all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:]))


class NearDuplicateTest(unittest.TestCase):
    def test_jaccard_exact_boundary(self):
        self.assertTrue(jaccard_at_least(array("Q", range(10)), array("Q", range(9)), 0.9))
        self.assertFalse(jaccard_at_least(array("Q", range(10)), array("Q", range(8)), 0.9))

    def test_prefix_candidate_join_matches_bruteforce_components(self):
        from alignment_lab.data_audit import UnionFind
        texts = []
        rng = random.Random(101)
        for family in range(5):
            base = [f"w{family}x{i}" for i in range(60)]
            for variant in range(5):
                changed = base.copy()
                for index in rng.sample(range(60), variant):
                    changed[index] = f"replacement{variant}z{index}"
                texts.append(" ".join(changed))
        rows = [parse_row(sample(text), "train", i) for i, text in enumerate(texts)]
        cfg = config(near_threshold=0.8)
        report = build_clusters(rows, cfg)
        expected = UnionFind(len(rows))
        values = [shingles(text.casefold()) for text in texts]
        for i in range(len(rows)):
            for j in range(i):
                intersection = len(set(values[i]) & set(values[j]))
                union = len(set(values[i]) | set(values[j]))
                if intersection / union >= cfg.near_threshold:
                    expected.union(i, j)
        self.assertTrue(report["complete_for_declared_representation"])
        for i in range(len(rows)):
            for j in range(len(rows)):
                self.assertEqual(rows[i]["cluster_id"] == rows[j]["cluster_id"], expected.find(i) == expected.find(j))

    def test_work_budget_exhaustion_is_readiness_blocker(self):
        text = " ".join(f"word{i}" for i in range(100))
        variants = [sample(text.replace("word50", f"variant{i}")) for i in range(6)]
        _, report = audit_records(variants, [], config(posting_budget=1))
        self.assertFalse(report["near_duplicates"]["complete_for_declared_representation"])
        self.assertTrue(any("near_duplicate_budget" in reason for reason in report["readiness_blockers"]))


class LimitsAndReuseTest(unittest.TestCase):
    def test_training_length_grid_counts_boundaries_and_excludes_heldouts(self):
        records = [sample(f"Different unique subject {i} with keyword {i*13}") for i in range(6)]
        rows = [parse_row(raw, "train", i) for i, raw in enumerate(records)]
        for i, row in enumerate(rows):
            row.update(local_split="train", cluster_id=f"cluster-{i}",
                       tokens={"prompt": 1025 if i == 5 else 1024,
                               "response1": [512, 640, 768, 1024, 1025, 10][i], "response2": 2})
        metadata, grid = training_length_grid(rows, config())
        self.assertEqual([g["pairs_before_pair_cap"] for g in grid["grid"]], [1, 2, 3, 4])
        self.assertEqual([g["independent_prompt_clusters"] for g in grid["grid"]], [1, 2, 3, 4])
        self.assertEqual(len(metadata), 6)
        self.assertNotIn("context", metadata[0])
        self.assertNotIn("response1", metadata[0])
        for change in ({"source_split": "validation"}, {"local_split": "tuning_dev"},
                       {"local_split": "diagnostic_holdout"}, {"overall_preference": 0}):
            with self.assertRaisesRegex(ValueError, "assigned_non_tie_train"):
                training_length_grid([{**rows[0], **change}], config())

    def test_invalid_final_row_and_quarantined_token_error_block_readiness(self):
        def lengths(row):
            if row["source_split"] == "train":
                raise ValueError("test_template_prefix_failure")
            return {"prompt": 10, "response1": 10, "response2": 10}
        parts, report = audit_records([sample()], [sample(), sample(overall_preference=4)], config(), lengths)
        self.assertEqual(report["raw_counts"]["validation"], 2)
        self.assertEqual(len(parts["final_holdout"]), 1)
        self.assertFalse(report["data_ready"])
        self.assertTrue(any("tokenization_errors" in reason for reason in report["readiness_blockers"]))
        self.assertTrue(any("invalid_official_validation_rows" in reason for reason in report["readiness_blockers"]))

    def test_length_boundaries_inclusive_and_no_truncation(self):
        records = [sample(f"Length scenario{i} unique{i} phrase{i} ending{i}") for i in range(4)]
        def lengths(row):
            i = row["source_index"]
            return {"prompt": 1025 if i == 1 else 1024, "response1": 513 if i == 2 else 512, "response2": 513 if i == 3 else 512}
        parts, report = audit_records(records, [], config(dev_fraction=0.01), lengths)
        self.assertEqual(len(parts["preference_train"]), 1)
        self.assertEqual(parts["preference_train"][0]["response1"], records[0]["response1"])
        self.assertEqual(report["exclusion_reasons"], {"prompt_too_long": 1, "response_too_long": 2})

    def test_existing_output_reused_readonly_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, validation, output = directory / "train.json", directory / "val.json", directory / "audit"
            train.write_text(json.dumps([sample()]), encoding="utf-8")
            validation.write_text("[]", encoding="utf-8")
            first = run_audit(train, validation, output, config=config(), progress=lambda _: None)
            before = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            second = run_audit(train, validation, output, config=config(), progress=lambda _: None)
            self.assertEqual(first, second)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
            (output / "preference_train.jsonl").write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                run_audit(train, validation, output, config=config(), progress=lambda _: None)

    def test_incomplete_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train, validation, output = directory / "train.json", directory / "val.json", directory / "audit"
            train.write_text("[]", encoding="utf-8")
            validation.write_text("[]", encoding="utf-8")
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "without a complete manifest"):
                run_audit(train, validation, output, config=config(), progress=lambda _: None)


class ReuseHardeningTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.train = directory / "train.json"
        self.validation = directory / "validation.json"
        self.output = directory / "audit"
        self.train.write_text(json.dumps([sample()]), encoding="utf-8")
        self.validation.write_text("[]", encoding="utf-8")
        self.original = run_audit(self.train, self.validation, self.output,
                                  config=config(), progress=lambda _: None)
        self.identity = self.original["identity"]

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, manifest):
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def rewrite_report_and_hash(self, change):
        path = self.output / "audit.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        change(report)
        path.write_text(json.dumps(report), encoding="utf-8")
        manifest = self.manifest()
        manifest["artifacts"]["audit.json"] = file_hash(path)
        self.write_manifest(manifest)

    def test_omitted_required_artifact_rejected_even_when_file_also_deleted(self):
        manifest = self.manifest()
        manifest["artifacts"].pop("final_holdout.jsonl")
        (self.output / "final_holdout.jsonl").unlink()
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "fixed complete set"):
            verify_existing(self.output, self.identity)

    def test_extra_inventory_artifact_rejected_even_when_hash_is_valid(self):
        path = self.output / "additional.jsonl"
        path.write_text("[]", encoding="utf-8")
        manifest = self.manifest()
        manifest["artifacts"][path.name] = file_hash(path)
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "fixed complete set"):
            verify_existing(self.output, self.identity)

    def test_missing_or_incomplete_status_rejected(self):
        for status in (None, "running", "completed"):
            with self.subTest(status=status):
                manifest = self.manifest()
                if status is None:
                    manifest.pop("status", None)
                else:
                    manifest["status"] = status
                self.write_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "not a completed audit"):
                    verify_existing(self.output, self.identity)

    def test_identity_selfhash_cannot_be_skipped(self):
        manifest = self.manifest()
        manifest["identity_sha256"] = "0" * 64
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "self-hash"):
            verify_existing(self.output, self.identity)

    def test_new_source_identity_rejects_coherently_hashed_old_manifest(self):
        manifest = self.manifest()
        manifest["identity"]["source_sha256"]["alignment_lab/data_audit.py"] = "0" * 64
        manifest["identity_sha256"] = digest(manifest["identity"])
        self.write_manifest(manifest)
        before = (self.output / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            run_audit(self.train, self.validation, self.output, config=config(), progress=lambda _: None)
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)

    def test_manifest_cannot_promote_failed_audit_to_ready(self):
        manifest = self.manifest()
        manifest.update(data_ready=True, readiness_blockers=[])
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "differs from hashed audit.json"):
            verify_existing(self.output, self.identity)

    def test_manifest_blocker_text_must_match_hashed_report(self):
        manifest = self.manifest()
        manifest["readiness_blockers"] = ["substituted explanation"]
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "readiness_blockers differs"):
            verify_existing(self.output, self.identity)

    def test_report_cannot_promote_itself_even_with_refreshed_artifact_hash(self):
        self.rewrite_report_and_hash(lambda report: report.update(data_ready=True, readiness_blockers=[]))
        with self.assertRaisesRegex(ValueError, "differs from hashed audit.json"):
            verify_existing(self.output, self.identity)

    def test_report_ready_flag_must_agree_with_its_blockers(self):
        self.rewrite_report_and_hash(lambda report: report.update(data_ready=True))
        with self.assertRaisesRegex(ValueError, "audit.json readiness flags/blockers are inconsistent"):
            verify_existing(self.output, self.identity)

    def test_flags_are_strict_booleans_and_training_never_started(self):
        for key, value in (("data_ready", 0), ("data_ready", "false"),
                           ("training_started", 0), ("training_started", True), ("training_started", None)):
            with self.subTest(key=key, value=value):
                manifest = json.loads(json.dumps(self.original))
                manifest[key] = value
                self.write_manifest(manifest)
                with self.assertRaises(ValueError):
                    verify_existing(self.output, self.identity)

    def test_hashed_report_training_flag_is_checked_independently(self):
        self.rewrite_report_and_hash(lambda report: report.update(training_started=True))
        with self.assertRaisesRegex(ValueError, "audit.json training_started must be false"):
            verify_existing(self.output, self.identity)


class RealTokenizerBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[1] / "models/qwen3-1.7b-tokenizer-70d244c"
        if not (directory / "tokenizer.json").exists():
            raise unittest.SkipTest("local pinned Qwen3 tokenizer not available; never downloads")
        cls.counter = LocalTokenCounter(directory)

    def test_raw_whitespace_empty_answer_and_empty_history_roundtrip(self):
        context = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""},
                   {"role": "user", "content": "Return the next answer."}]
        codec = self.counter.codec
        for answer in ("", "\n\nAnswer", "\r\n leading\ttext\n", "  ", "中文回答。", "🙂"):
            with self.subTest(answer=answer):
                row = parse_row(sample(context=context, response1=answer), "train", 0)
                encoded = codec.encode(row["context"], row["response1"])
                tok = self.counter.tokenizer
                self.assertEqual(tok.decode(list(encoded.input_ids), skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False),
                                 encoded.prompt.rendered + answer + tok.eos_token)
                self.assertEqual(encoded.completion_ids.count(tok.eos_token_id), 1)
                self.assertEqual(encoded.labels[:len(encoded.prompt.input_ids)], (-100,) * len(encoded.prompt.input_ids))
                self.assertEqual(sum(encoded.completion_mask), len(encoded.completion_ids))
                self.assertEqual(self.counter(row)["response1"], len(encoded.completion_ids))
                if not answer:
                    self.assertEqual(encoded.completion_ids, (151645,))
                padded = encoded.pad_to(len(encoded.input_ids) + 3)
                self.assertEqual(padded["input_ids"][-3:], (151643,) * 3)
                self.assertEqual(padded["attention_mask"][-3:], (False,) * 3)
                self.assertEqual(padded["labels"][-3:], (-100,) * 3)
                self.assertEqual(sum(padded["attention_mask"]), len(encoded.input_ids))
                with self.assertRaises(ValueError):
                    encoded.pad_to(len(encoded.input_ids) - 1)

    def test_special_tokens_rejected_in_raw_context_and_response(self):
        codec = self.counter.codec
        for special in self.counter.tokenizer.all_special_tokens:
            with self.subTest(special=special):
                with self.assertRaises(ValueError):
                    codec.encode([{"role": "user", "content": "Question"}], "prefix" + special)
                with self.assertRaises(ValueError):
                    codec.encode([{"role": "user", "content": "prefix" + special}], "Answer")
        for marker in ("<think>", "</think>", "<tool_call>", "<tool_response>"):
            with self.assertRaises(ValueError):
                codec.encode([{"role": "user", "content": "Question"}], marker)

    def test_native_nfc_normalization_is_explicit_and_raw_text_is_retained(self):
        context = [{"role": "user", "content": "Describe O\u0303lk and cafe\u0301."}]
        answer = "\n\ncafe\u0301 O\u0303"
        row = parse_row(sample(context=context, response1=answer), "train", 0)
        encoded = self.counter.codec.encode(context, answer)
        tok = self.counter.tokenizer
        self.assertEqual(encoded.raw_response, answer)
        self.assertIn("O\u0303", encoded.prompt.rendered)
        self.assertIn("Õ", encoded.prompt.normalized_rendered)
        decoded = tok.decode(list(encoded.input_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertEqual(decoded, encoded.prompt.normalized_rendered + "\n\ncafé Õ" + tok.eos_token)
        lengths = self.counter(row)
        self.assertTrue(lengths["prompt_nfc_changed"])
        self.assertTrue(lengths["response1_nfc_changed"])
        self.assertFalse(lengths["response2_nfc_changed"])
        self.assertEqual(row["context"], context)
        self.assertEqual(row["response1"], answer)

    def test_raw_control_rejected_before_clustering_without_blocking_clean_training(self):
        bad = sample("Quoted control token", response1="Literal <|endoftext|> token")
        good = sample("Unique clean unrelated example")
        parts, report = audit_records([bad, good], [], config(dev_fraction=0.01), self.counter)
        self.assertEqual(report["exclusion_reasons"], {"reserved_tokenizer_special_token_in_raw_text": 1})
        self.assertEqual(report["tokenization_errors"], {})
        self.assertEqual(len(parts["assigned_train_lengths"]), 1)
        self.assertTrue(report["data_ready"])
        _, final_report = audit_records([good], [bad], config(dev_fraction=0.01), self.counter)
        self.assertTrue(any("invalid_official_validation_rows" in x for x in final_report["readiness_blockers"]))

    def test_newline_encoding_preserves_raw_text_instead_of_final_template_rewrite(self):
        tok = self.counter.tokenizer
        context = [{"role": "user", "content": "Continue."}]
        answer = "\n\nExact original answer."
        encoded = self.counter.codec.encode(context, answer)
        official = tok.apply_chat_template(context + [{"role": "assistant", "content": answer}],
                                           tokenize=False, add_generation_prompt=False, enable_thinking=False)
        self.assertNotEqual(official, encoded.prompt.rendered + answer + tok.eos_token + "\n")
        self.assertEqual(tok.decode(list(encoded.completion_ids), clean_up_tokenization_spaces=False),
                         answer + tok.eos_token)

    def test_nonthinking_prompt_and_exact_completion_eos_boundaries(self):
        for answer in ("A brief answer.", "中文回答。", "  Leading spaces are preserved.", "print('hello')\n"):
            raw = sample(response1=answer)
            row = parse_row(raw, "train", 0)
            lengths = self.counter(row)
            tok = self.counter.tokenizer
            prompt = tok.apply_chat_template(row["context"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = tok.apply_chat_template(row["context"] + [{"role": "assistant", "content": answer}], tokenize=False,
                                           add_generation_prompt=False, enable_thinking=False)
            self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"))
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            template_ids = tok.apply_chat_template(row["context"], tokenize=True, return_dict=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            self.assertEqual(template_ids, prompt_ids)
            ids = tok.encode(full[:-1], add_special_tokens=False)
            self.assertEqual(ids[:len(prompt_ids)], prompt_ids)
            completion = ids[len(prompt_ids):]
            self.assertEqual(completion[-1], tok.eos_token_id)
            self.assertEqual(lengths["response1"], len(completion))
            self.assertEqual(tok.decode(completion), answer + tok.eos_token)

    def test_history_assistant_is_prompt_only(self):
        row = parse_row(sample(context=[{"role": "user", "content": "Earlier question?"},
                                        {"role": "assistant", "content": "Earlier answer in the context."},
                                        {"role": "user", "content": "Please elaborate."}]), "train", 0)
        lengths = self.counter(row)
        self.assertGreater(lengths["prompt"], lengths["response1"])
        self.assertEqual(lengths["response1"], len(self.counter.tokenizer.encode(row["response1"] + self.counter.tokenizer.eos_token, add_special_tokens=False)))


if __name__ == "__main__":
    unittest.main()
