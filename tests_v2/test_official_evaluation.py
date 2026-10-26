"""CPU fixtures only: no real model scores, generations or benchmark results."""
import ast
import contextlib
import copy
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_protocol as ep
from alignment_lab import official_evaluation as oe

ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = ROOT / "data/evaluation-assets-20261020T103505Z/ASSET_MANIFEST.json"
HOLDOUT_DIR = ROOT / "reports/alignlab-v2-data-audit-v3-p1024-r768-20261020"
NLTK_DIR = ROOT / ".runtime/alignlab-eval-nltk-data-20261020"


def dataset(name, rows, field, lengths):
    identity = {"dataset": name, "record_count": len(rows), "records_sha256": ep.digest(rows),
                "ordered_ids_sha256": ep.digest([r[field] for r in rows])}
    return ep.EvaluationDataset(name, field, ep.canonical(rows), ep.canonical(identity), ep.canonical(lengths))


class SyntheticFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor = self.root / "anchor.json"
        self.anchor.write_text('{"synthetic_fixture":true}\n', encoding="utf-8")
        self.sha = hashlib.sha256(self.anchor.read_bytes()).hexdigest()

    def protocol(self, *datasets):
        anchor = {"path": str(self.anchor), "sha256": self.sha}
        return ep.EvaluationProtocol(ep.canonical({
            "model_manifests": {"fixture": anchor}, "source_files": {},
            "nltk_resource_files": {}, "asset_manifest": anchor, "holdout_manifest": anchor,
            "dataset_identities": {data.name: data.identity for data in datasets},
            "scorer_seed": 42, "bootstrap_seed": 73,
            "generation": {"max_new_tokens": 2048, "eos_token_id": 151645, "vocab_size": 151936},
        }))

    def identity_fields(self, protocol):
        return {"model_id": "fixture", "model_manifest_sha256": self.sha,
                "protocol_sha256": protocol.sha256, "input_truncated": False}

    def generation_fixture(self):
        data = dataset("ifeval", [{"key": 1, "prompt": "one"}, {"key": 2, "prompt": "two"}],
                       "key", {"1": 3, "2": 4})
        protocol = self.protocol(data)
        rows = [{"key": key, "prompt": prompt, "response": response,
                 "generated_token_ids": ids, "termination": "eos", "prompt_tokens": length,
                 "generation_seed": 42, **self.identity_fields(protocol)}
                for key, prompt, response, ids, length in
                [(1, "one", "7", [7, 151645], 3), (2, "two", "", [151645], 4)]]
        return data, protocol, rows

    @staticmethod
    def decode(ids):
        return "".join(str(token) for token in ids)

    def validate(self, data, protocol, rows):
        return ep.validate_generation_records(data, rows, protocol=protocol,
                                               model_id="fixture", decode_tokens=self.decode)


class GenerationSchemaTests(SyntheticFixture):
    def test_reorder_is_aligned_by_all_ids_and_empty_eos_is_retained(self):
        data, protocol, rows = self.generation_fixture()
        result = self.validate(data, protocol, rows[::-1])
        self.assertEqual([r["key"] for r in result], [1, 2])
        self.assertEqual(result[1]["response"], "")
        result[0]["generated_token_ids"].append(11)
        self.assertEqual(rows[0]["generated_token_ids"], [7, 151645])

    def test_missing_duplicate_extra_ids_are_rejected(self):
        data, protocol, rows = self.generation_fixture()
        for bad in (rows[:1], [rows[0], rows[0]], rows + [{**rows[0], "key": 3}]):
            with self.subTest(ids=[r["key"] for r in bad]), self.assertRaises(ValueError):
                self.validate(data, protocol, bad)

    def test_exact_prompt_decoding_identity_and_finite_metadata_required(self):
        data, protocol, rows = self.generation_fixture()
        changes = ({"prompt": "one "}, {"response": "wrong"}, {"protocol_sha256": "b" * 64},
                   {"model_manifest_sha256": "b" * 64}, {"generation_seed": False},
                   {"elapsed": float("nan")}, {"input_truncated": True}, {"prompt_tokens": 2})
        for change in changes:
            bad = copy.deepcopy(rows)
            bad[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(data, protocol, bad)

    def test_full_length_without_eos_is_explicit_not_fabricated_eos(self):
        data, protocol, rows = self.generation_fixture()
        rows[0].update(generated_token_ids=[7] * 2048, response="7" * 2048, termination="length")
        self.validate(data, protocol, rows)
        for ids, reason in (([7] * 2047, "length"), ([7, 151645, 7], "eos"),
                            ([7], "eos"), ([7] * 2049, "length"), ([7], "stopped")):
            bad = copy.deepcopy(rows)
            bad[0].update(generated_token_ids=ids, termination=reason)
            with self.subTest(reason=reason, length=len(ids)), self.assertRaises(ValueError):
                self.validate(data, protocol, bad)
        rows[0].update(generated_token_ids=[7] * 2047 + [151645],
                       response="7" * 2047, termination="eos")
        self.validate(data, protocol, rows)

    def test_dataset_copy_cannot_change_identity_and_protocol_binding_rechecks(self):
        data, protocol, rows = self.generation_fixture()
        data.rows[0]["prompt"] = "caller mutation"
        self.assertEqual(data.rows[0]["prompt"], "one")
        altered = dataclasses.replace(data, _rows=ep.canonical([{"key": 1, "prompt": "changed"}]))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate(altered, protocol, rows)
        self.anchor.write_text('{"changed":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate(data, protocol, rows)


class RMArithmeticTests(SyntheticFixture):
    @staticmethod
    def uneven():
        return ([{"domain": "chat", "score_chosen": [0, 0, 0], "score_rejected": [0, 0, 0]} for _ in range(4)]
                + [{"domain": domain, "score_chosen": [1, 1, 1], "score_rejected": [0, 0, 0]}
                   for domain in ("math", "code", "safety-refuse")])

    def test_official_macro_differs_from_micro_and_ties_are_not_half_credit(self):
        result = oe._rm_metrics(self.uneven())
        self.assertEqual(result["metrics"]["total_avg_acc"], .75)
        self.assertEqual(result["metrics"]["chat"], 0)
        self.assertNotEqual(result["metrics"]["total_avg_acc"], 3 / 7)

    def test_style_triangles_independent_hand_numeric(self):
        rows = [{"domain": d, "score_chosen": [1, 3, 5], "score_rejected": [2, 3, 4]} for d in ep.DOMAINS]
        result = oe._rm_metrics(rows)
        self.assertEqual(result["metrics"]["hard_acc"], 0)
        self.assertEqual(result["metrics"]["normal_acc"], 1 / 3)
        self.assertEqual(result["metrics"]["easy_acc"], 1)
        self.assertAlmostEqual(result["metrics"]["total_avg_acc"], 4 / 9)

    def test_matches_actual_official_ast_oracle_without_importing_gpu_runner(self):
        import numpy as np
        raw = (ASSET_PATH.parent / "rm_bench/scripts/utils.py").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "330f1eedbada021b715982785a69d0e39dcb9e19e66e11b14574be838f514a12")
        tree = ast.parse(raw)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("split_dataset_by_domain", "compute_accuracy")]
        self.assertEqual(len(functions), 2)
        namespace = {"np": np, "List": list, "Dict": dict, "Any": object}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<pinned-official-aggregation>", "exec"), namespace)
        rng = random.Random(83)
        rows = self.uneven() + [{"domain": d, "score_chosen": [rng.randrange(3) for _ in range(3)],
                                "score_rejected": [rng.randrange(3) for _ in range(3)]}
                               for d in ("safety-response", "chat", "code", "math") for _ in range(3)]
        before = copy.deepcopy(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = namespace["compute_accuracy"](copy.deepcopy(rows))
        actual = oe._rm_metrics(rows)["metrics"]
        for key in expected:
            self.assertAlmostEqual(actual[key], float(expected[key]), places=15)
        self.assertEqual(rows, before)

    def test_missing_domain_missing_style_nonfinite_and_unknown_domain_rejected(self):
        for rows in (self.uneven()[:4], [{"domain": "other", "score_chosen": [1]*3, "score_rejected": [0]*3}],
                     [{**r, "score_chosen": [1, 2]} for r in self.uneven()],
                     [{**r, "score_rejected": [0, float("inf"), 0]} for r in self.uneven()]):
            with self.assertRaises(ValueError):
                oe._rm_metrics(rows)

    def test_full_input_lengths_and_ids_are_required_before_rm_aggregation(self):
        sources = [{"id": str(i), "domain": domain} for i, domain in enumerate(ep.DOMAINS)]
        lengths = {r["id"]: {"chosen": [2100, 2300, 2363], "rejected": [40, 50, 60]} for r in sources}
        data = dataset("rm_bench", sources, "id", lengths)
        protocol = self.protocol(data)
        assets = types.SimpleNamespace(rm_bench=data, verify=lambda: None)
        rows = [{"id": r["id"], "score_chosen": [1, 2, 3], "score_rejected": [0, 0, 0],
                 "input_lengths_chosen": lengths[r["id"]]["chosen"],
                 "input_lengths_rejected": lengths[r["id"]]["rejected"],
                 **self.identity_fields(protocol)} for r in sources]
        result = oe.aggregate_rm_bench(assets, rows[::-1], protocol=protocol, model_id="fixture")
        self.assertEqual(result["denominators"], {"records": 4, "answer_scores": 24, "comparisons": 36})
        for truncated, badlength in ((True, False), (False, True)):
            changed = copy.deepcopy(rows)
            changed[0]["input_truncated"] = truncated
            if badlength:
                changed[0]["input_lengths_chosen"][0] = 2048
            with self.assertRaises(ValueError):
                oe.aggregate_rm_bench(assets, changed, protocol=protocol, model_id="fixture")

    def test_helpsteer_sign_ties_and_long_inputs_are_not_filtered(self):
        sources = [{"evaluation_id": "a", "record_id": "repeated-content", "source_index": 0, "overall_preference": -2},
                   {"evaluation_id": "b", "record_id": "repeated-content", "source_index": 1, "overall_preference": 1},
                   {"evaluation_id": "c", "record_id": "other-content", "source_index": 2, "overall_preference": 0}]
        data = dataset("helpsteer3_final_holdout", sources, "evaluation_id", {r["evaluation_id"]: [6000, 7000] for r in sources})
        protocol = self.protocol(data)
        rows = [{"evaluation_id": source["evaluation_id"], "record_id": source["record_id"],
                 "source_index": source["source_index"], "response1_score": one, "response2_score": two,
                 "input_lengths": [6000, 7000], **self.identity_fields(protocol)}
                for source, (one, two) in zip(sources, ((2, 0), (0, 1), (2, 1)))]
        result = oe.aggregate_helpsteer_rm(data, rows[::-1], protocol=protocol, model_id="fixture")
        self.assertEqual(result["denominators"], {"all": 3, "non_ties": 2, "ties": 1})
        self.assertEqual(result["pair_accuracy"], 1)
        self.assertEqual(result["tie_mean_absolute_margin"], 1)
        self.assertFalse(result["ties_assigned_binary_labels"])
        with self.assertRaises(ValueError):
            oe.aggregate_helpsteer_rm(data, rows[:2], protocol=protocol, model_id="fixture")


class ClusterBootstrapTests(unittest.TestCase):
    def test_cluster_multiplicity_and_record_weighted_point_estimate(self):
        left, right = {"a": 1, "b": 1, "c": 0}, {"a": 0, "b": 0, "c": 1}
        clusters = {"a": "same", "b": "same", "c": "other"}
        state = random.getstate()
        result = oe.paired_cluster_bootstrap(left, right, clusters, seed=7)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(result["estimate"], 1 / 3)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["confidence_interval"], [-1, 1])
        self.assertEqual(result, oe.paired_cluster_bootstrap(dict(reversed(list(left.items()))), right, clusters, seed=7))
        self.assertFalse(result["across_seed_pooling"])

    def test_one_cluster_all_members_stay_together(self):
        result = oe.paired_cluster_bootstrap({"a": 1, "b": -1}, {"a": 0, "b": 0},
                                            {"a": "cluster", "b": "cluster"}, seed=3)
        self.assertEqual(result["confidence_interval"], [0, 0])
        self.assertEqual(result["clusters"], 1)

    def test_mismatched_ids_nonfinite_bad_cluster_and_too_few_replicates(self):
        with self.assertRaises(ValueError):
            oe.paired_cluster_bootstrap({"a": 1}, {"b": 1}, {"a": "c"}, seed=0)
        for value, cluster in ((float("nan"), "c"), (1, ""), (float("inf"), "c")):
            with self.assertRaises(ValueError):
                oe.paired_cluster_bootstrap({"a": value}, {"a": 0}, {"a": cluster}, seed=0)
        with self.assertRaises(ValueError):
            oe.paired_cluster_bootstrap({"a": 1}, {"a": 0}, {"a": "c"}, seed=0, replicates=100)


class ScorerIsolationTests(unittest.TestCase):
    def test_python_and_langdetect_rng_restored_even_on_checker_failure(self):
        # Exercise import/RNG orchestration only; this stub is never an IFEval
        # checker and is never returned as an official benchmark result.
        class Factory:
            seed = 13
        language = types.ModuleType("langdetect")
        language.DetectorFactory = Factory
        factory_module = types.ModuleType("langdetect.detector_factory")
        assets = types.SimpleNamespace(manifest_path="/synthetic-fixture/manifest.json",
            read=lambda _: b"import random\nfixture_draw = random.random()\n")
        for preexisting in (False, True):
            instance = Factory() if preexisting else None
            if instance is not None:
                instance.seed = 17
            factory_module._factory = instance
            before = random.getstate()
            with patch.dict(sys.modules, {"langdetect": language, "langdetect.detector_factory": factory_module}):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    with oe._official_ifeval(assets, 42):
                        self.assertEqual(Factory.seed, 42)
                        if instance is not None:
                            self.assertEqual(instance.seed, 42)
                        else:
                            factory_module._factory = Factory()
                        random.random()
                        raise RuntimeError("fixture failure")
            self.assertEqual(random.getstate(), before)
            self.assertEqual(Factory.seed, 13)
            self.assertIs(factory_module._factory, instance)
            if instance is not None:
                self.assertEqual(instance.seed, 17)
            self.assertFalse(any(k == "instruction_following_eval" or k.startswith("instruction_following_eval.")
                                 for k in sys.modules))


class AcceptedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = ep.load_evaluation_assets(ASSET_PATH, ep.ASSET_MANIFEST_SHA256)
        cls.holdout = ep.load_final_holdout(HOLDOUT_DIR, ep.HOLDOUT_MANIFEST_SHA256)

    def test_all_official_denominators_and_no_holdout_length_filter(self):
        self.assertEqual(len(self.assets.ifeval.rows), 541)
        self.assertEqual(sum(len(r["instruction_id_list"]) for r in self.assets.ifeval.rows), 834)
        self.assertEqual(len(self.assets.rm_bench.rows), 1327)
        self.assertEqual(sum(len(r["chosen"]) + len(r["rejected"]) for r in self.assets.rm_bench.rows), 7962)
        self.assertEqual(len(self.holdout.rows), 2017)
        self.assertEqual(len(set(self.holdout.ids)), 2017)
        self.assertEqual(self.holdout.identity["unique_content_record_ids"], 1965)
        self.assertEqual(len(self.holdout.identity["non_tie_ids"]), 1920)
        self.assertEqual(len(self.holdout.identity["tie_ids"]), 97)
        self.assertEqual(self.holdout.identity["split_counts"], {"validation": 2017, "train": 0})
        self.assertGreater(max(max(v) for v in self.holdout.lengths.values()), 3072)
        self.holdout.rows[0]["context"].clear()
        self.assertTrue(self.holdout.rows[0]["context"])

    def test_missing_official_dependencies_raise_without_fallback(self):
        diagnostic = oe.ifeval_dependency_diagnostics(self.assets)
        self.assertIn("requirements.txt", diagnostic["requirements_source"]["path"])
        self.assertFalse(diagnostic["fallback_available"])
        forced = {**diagnostic, "ready": False, "missing_packages": ["nltk"]}
        with patch.object(oe, "ifeval_dependency_diagnostics", return_value=forced):
            with self.assertRaisesRegex(RuntimeError, "Official IFEval dependencies"):
                oe.score_ifeval(self.assets, [], protocol=None, model_id="fixture", decode_tokens=lambda _: "")

    def test_duplicate_json_keys_and_nonfinite_identity_cannot_be_silently_normalized(self):
        for raw in (b'{"a":1,"a":2}', b'{"value":NaN}', b'{"value":Infinity}'):
            with self.assertRaises(ValueError):
                ep.parse(raw)

    def test_actual_official_runtime_or_explicit_missing_dependency_error(self):
        diagnostic = oe.ifeval_dependency_diagnostics(self.assets, nltk_data_dir=str(NLTK_DIR))
        if not diagnostic["ready"]:
            # The training environment intentionally lacks scorer dependencies.
            # This branch verifies the failure, never substitutes any checker.
            with self.assertRaisesRegex(RuntimeError, "Official IFEval dependencies"):
                oe.score_ifeval(self.assets, [], protocol=None, model_id="fixture", decode_tokens=lambda _: "")
            return
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "synthetic_not_model.json"
            manifest.write_text('{"synthetic_test_fixture":true,"model_inference":false}', encoding="utf-8")
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            protocol = ep.build_evaluation_protocol(
                self.assets, self.holdout, model_manifests={"fixture": (str(manifest), sha)},
                scorer_seed=0, bootstrap_seed=0, nltk_data_dir=str(NLTK_DIR),
                nltk_resource_files={path: (path, checksum) for path, checksum in diagnostic["resource_files"].items()})
            text = "This is a synthetic unit test fixture, not an answer generated by a model."
            outputs = [{"key": row["key"], "prompt": row["prompt"], "response": text,
                        "generated_token_ids": [7, 151645], "termination": "eos",
                        "prompt_tokens": self.assets.ifeval.lengths[str(row["key"])],
                        "generation_seed": 0, "model_id": "fixture",
                        "model_manifest_sha256": sha, "protocol_sha256": protocol.sha256,
                        "input_truncated": False} for row in self.assets.ifeval.rows]
            result = oe.score_ifeval(self.assets, outputs[::-1], protocol=protocol,
                                     model_id="fixture", decode_tokens=lambda ids: text if ids == (7,) else "")
            self.assertEqual(len(result["per_prompt"]), 541)
            self.assertEqual(result["denominators"], {"prompts": 541, "instructions": 834})
            response_map = {r["prompt"]: text for r in self.assets.ifeval.rows}
            with oe._official_ifeval(self.assets, 0, str(NLTK_DIR)) as official:
                for mode in ("strict", "loose"):
                    flags = []
                    for raw, actual in zip(self.assets.ifeval.rows, result["per_prompt"]):
                        expected = getattr(official, "test_instruction_following_" + mode)(
                            official.InputExample(**raw), response_map)
                        self.assertEqual(expected.follow_instruction_list, actual[mode]["follow_instruction_list"])
                        flags.append(expected.follow_instruction_list)
                    self.assertEqual(result["metrics"][mode + "_prompt_accuracy"], sum(all(f) for f in flags) / 541)
                    self.assertEqual(result["metrics"][mode + "_instruction_accuracy"], sum(map(sum, flags)) / 834)
            self.assertNotIn("torch", sys.modules)

    def test_wrong_pin_missing_file_or_extra_asset_file_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unaccepted"):
            ep.load_evaluation_assets(ASSET_PATH, "a" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "assets"
            shutil.copytree(ASSET_PATH.parent, copied)
            (copied / "unregistered.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra"):
                ep.load_evaluation_assets(copied / ASSET_PATH.name, ep.ASSET_MANIFEST_SHA256)
            (copied / "unregistered.txt").unlink()
            path = copied / "rm_bench/scripts/utils.py"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ep.load_evaluation_assets(copied / ASSET_PATH.name, ep.ASSET_MANIFEST_SHA256)

    def test_protocol_requires_real_existing_manifest_bytes_and_is_detached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_model_manifest.json"
            with self.assertRaises(ValueError):
                ep.build_evaluation_protocol(self.assets, self.holdout,
                    model_manifests={"fixture": (str(path), "a" * 64)}, scorer_seed=42, bootstrap_seed=73)
            path.write_text('{"synthetic_fixture":true}', encoding="utf-8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            protocol = ep.build_evaluation_protocol(self.assets, self.holdout,
                model_manifests={"fixture": (str(path), sha)}, scorer_seed=42, bootstrap_seed=73)
            protocol.identity["generation"]["max_new_tokens"] = 1
            self.assertEqual(protocol.identity["generation"]["max_new_tokens"], 2048)
            self.assertEqual(protocol.sha256, hashlib.sha256(protocol._payload).hexdigest())
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                protocol.verify_bindings()


if __name__ == "__main__":
    unittest.main()
