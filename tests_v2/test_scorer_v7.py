"""Actual V6 CPU scorer over artificial V7 receipts; never model measurements.

Only immutable original datasets/tokenizer/checker assets are read. Generated
text, reward values, accepted-model receipts and CUDA audit claims below are
explicit synthetic fixtures. No model, Torch, CUDA or live inference is used.
"""
import copy
from pathlib import Path
import sys
import unittest

from alignment_lab import evaluation_protocol_v2 as ep
from alignment_lab.evaluation_analysis_v7 import REQUIRED_SOURCES
from alignment_lab.evaluation_acceptance_v2 import SOURCES
from alignment_lab.evaluation_supervisor_v7 import NEW_SOURCES
from scripts import prepare_alignlab_evaluation_v7 as preparation
from scripts import score_alignlab_evaluation_v6 as scorer
from tests_v2 import test_scorer_v6 as fixture

ROOT = Path(__file__).resolve().parents[1]
save = fixture.legacy.save
sha = fixture.legacy.sha


class V7ScorerCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ScorerV6Tests.setUpClass()

    def setUp(self):
        self.f = fixture.ScorerV6Tests(methodName="test_current_version_inference_and_scoring_closures_are_also_supported")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        prior = ep.parse(ep.verified_bytes(
            ROOT / "reports/alignlab-quality-v6-launch-20261026T085000Z/plan.json",
            preparation.PRIOR_PLAN_SHA256))
        names = set(prior["source_sha256"]) | REQUIRED_SOURCES | set(SOURCES) | NEW_SOURCES | {
            "scripts/prepare_alignlab_evaluation_v7.py", "scripts/verify_alignlab_evaluation_v7.py",
            "tests_v2/test_evaluation_prepare_v7.py", "tests_v2/test_scorer_v7.py"}
        # This is the complete current continuation closure, including every
        # immutable historical source; it is not a reduced scorer-only closure.
        for name, digest in prior["source_sha256"].items():
            ep.verified_bytes(ROOT / name, digest)
        self.sources = {name: sha(ROOT / name) for name in sorted(names)}
        self.f.scoring_sources = copy.deepcopy(self.sources)
        self.f.inference["source_sha256"] = copy.deepcopy(self.sources)
        self.f.refresh()

    def score(self):
        before = {path: path.read_bytes() for path in self.f.directory.rglob("*") if path.is_file()}
        result = scorer.score_request(self.f.request_path, sha(self.f.request_path), output=self.f.directory / "scores")
        self.assertEqual(result["status"], "official_scoring_passed")
        self.assertEqual(result["deadline_epoch"], self.f.deadline)
        self.assertEqual(result["inference_manifest"], {
            "path": self.f.request["inference_manifest"], "sha256": self.f.request["inference_manifest_sha256"]})
        self.assertFalse(result["gpu_operations_performed"])
        self.assertFalse(result["research_complete"])
        self.assertEqual(result["human_labels"], 0)
        self.assertIsNone(result["human_win_rate"])
        self.assertNotIn("torch", sys.modules)
        for path, payload in before.items():
            self.assertEqual(path.read_bytes(), payload, str(path))
        return result

    def test_full_helpsteer_v7_closure_uses_actual_v6_scorer(self):
        self.assertIn("alignment_lab/evaluation_execution_v7.py", self.f.inference["source_sha256"])
        self.assertIn("alignment_lab/evaluation_execution_v6.py", self.f.inference["source_sha256"])
        self.assertEqual(self.f.request["source_sha256"], self.sources)
        result = self.score()
        self.assertEqual(result["scores"]["denominators"], {"all": 2017, "non_ties": 1920, "ties": 97})

    def test_full_rm_bench_v7_closure_uses_actual_v6_scorer(self):
        f = self.f
        f.inference["task"] = "rm_bench"
        rows = [{"id": row["id"], "score_chosen": [1.0, 2.0, 3.0], "score_rejected": [0.0, 0.0, 0.0],
                 "input_lengths_chosen": f.assets.rm_bench.lengths[row["id"]]["chosen"],
                 "input_lengths_rejected": f.assets.rm_bench.lengths[row["id"]]["rejected"],
                 "model_id": "reward", "model_manifest_sha256": f.model["sha256"],
                 "protocol_sha256": f.protocol.sha256, "input_truncated": False}
                for row in f.assets.rm_bench.rows]
        f.refresh(rows=rows, changes={"task": "rm_bench", "dataset_identity": f.assets.rm_bench.identity})
        result = self.score()
        self.assertEqual(result["scores"]["denominators"], {"records": 1327, "answer_scores": 7962, "comparisons": 11943})

    def test_full_ppo44_v7_closure_actual_official_scorer_and_cpu_decoder(self):
        f = self.f
        tokenizer = ROOT / "models/qwen3-1.7b-tokenizer-70d244c/tokenizer.json"
        f.model_id = "ppo44"
        f.model = save(Path(f.models[f.model_id]["path"]), {"synthetic_fixture": True,
            "identity": {"inputs": {"tokenizer/tokenizer.json": {"sha256": sha(tokenizer)}}}})
        f.models[f.model_id] = f.model
        f.bind_roster()
        nltk = ROOT / ".runtime/alignlab-eval-nltk-data-20261020"
        resources = {str(path.resolve()): (str(path.resolve()), sha(path))
                     for path in (nltk / "tokenizers/punkt_tab").rglob("*") if path.is_file()}
        self.assertEqual(len(resources), 77)
        f.protocol = ep.build_evaluation_protocol(f.assets, f.holdout,
            model_manifests={name: (item["path"], item["sha256"]) for name, item in f.models.items()},
            scorer_seed=0, bootstrap_seed=73, nltk_data_dir=str(nltk), nltk_resource_files=resources)
        f.protocol_file = save(Path(f.protocol_file["path"]), f.protocol.identity)
        f.inference.update(task="ifeval", protocol_sha256=f.protocol_file["sha256"], model_id=f.model_id,
                           roster_sha256=f.roster["sha256"])
        synthetic_text = "This is a synthetic test response."
        # Verified with the pinned tokenizer; actual decoder checks it again.
        synthetic_ids = [1986, 374, 264, 27268, 1273, 2033, 13, 151645]
        rows = [{"key": source["key"], "prompt": source["prompt"], "response": synthetic_text,
                 "prompt_tokens": f.assets.ifeval.lengths[str(source["key"])],
                 "generation_seed": 0, "generated_token_ids": synthetic_ids, "termination": "eos",
                 "input_truncated": False, "model_id": f.model_id,
                 "model_manifest_sha256": f.model["sha256"], "protocol_sha256": f.protocol.sha256}
                for source in f.assets.ifeval.rows]
        f.refresh(rows=rows, changes={"task": "ifeval", "dataset_identity": f.assets.ifeval.identity})
        python = ROOT / ".runtime/alignlab-v2/Scripts/python.exe"
        script = ROOT / "scripts/decode_alignlab_evaluation_v2.py"
        f.request["decoder"] = {"python": str(python), "python_sha256": sha(python),
            "script": str(script), "script_sha256": sha(script), "tokenizer_path": str(tokenizer),
            "tokenizer_sha256": sha(tokenizer), "tokenizers_version": "0.22.2"}
        save(f.request_path, f.request)
        result = self.score()
        self.assertEqual(result["scores"]["denominators"], {"prompts": 541, "instructions": 834})
        self.assertTrue(result["scores"]["official_evaluator"])
        self.assertEqual(len(result["scores"]["per_prompt"]), 541)
        self.assertEqual(result["decoder_receipt"]["texts"], [synthetic_text] * 541)
        self.assertFalse(result["decoder_receipt"]["torch_imported"])
        self.assertFalse(result["decoder_receipt"]["gpu_operations_performed"])

    def test_v7_source_sha_drift_rejected_in_both_inference_and_scoring(self):
        f = self.f
        for boundary in ("inference", "scoring"):
            with self.subTest(boundary=boundary):
                f.inference["source_sha256"] = copy.deepcopy(self.sources)
                f.scoring_sources = copy.deepcopy(self.sources)
                target = f.inference["source_sha256"] if boundary == "inference" else f.scoring_sources
                target["alignment_lab/evaluation_execution_v7.py"] = "0" * 64
                f.refresh()
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    scorer.score_request(f.request_path, sha(f.request_path), output=f.directory / "scores")
                self.assertFalse((f.directory / "scores").exists())
                self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
