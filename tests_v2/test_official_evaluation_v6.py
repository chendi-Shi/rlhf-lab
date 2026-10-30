"""V6 resource-pin regressions; all answers here are synthetic CPU fixtures."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alignment_lab import evaluation_protocol_v2 as ep
from alignment_lab import official_evaluation_v6 as oe

ROOT = Path(__file__).resolve().parents[1]
NLTK_DIR = ROOT / ".runtime/alignlab-eval-nltk-data-20261020"


class NLTKPinSubsetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = {}
        for index in range(77):
            relative = (f"tokenizers/punkt_tab/english/{index}.tab" if index < 4
                        else f"tokenizers/punkt_tab/other/{index}.tab")
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic resource {index}\n".encode())
            self.files[relative] = {"path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.identity = {"nltk_data_dir": str(self.root),
                         "nltk_resource_files": copy.deepcopy(self.files)}
        self.diagnostics = {"resource_files": {
            record["path"]: record["sha256"] for record in list(self.files.values())[:4]}}

    def verify(self):
        protocol = ep.EvaluationProtocol(ep.canonical(self.identity))
        return oe.verify_nltk_resource_pins(protocol, self.diagnostics)

    def test_all_77_pins_verified_while_required_four_are_subset(self):
        before = ep.canonical(self.identity)
        with patch.object(oe, "verified_bytes", wraps=ep.verified_bytes) as read:
            self.assertEqual(self.verify(), {"all_declared_files_verified": True,
                                           "declared_files": 77, "required_files": 4})
        self.assertEqual(read.call_count, 77)
        self.assertEqual({str(call.args[0]) for call in read.call_args_list},
                         {item["path"] for item in self.files.values()})
        self.assertEqual(ep.canonical(self.identity), before)

    def test_tampered_unused_resource_is_rejected(self):
        path = Path(list(self.files.values())[-1]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.verify()

    def test_missing_unused_resource_is_rejected(self):
        Path(list(self.files.values())[-1]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Expected regular file"):
            self.verify()

    def test_tampered_required_resource_is_rejected(self):
        Path(list(self.files.values())[0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.verify()

    def test_missing_required_pin_is_rejected(self):
        self.identity["nltk_resource_files"].pop(next(iter(self.files)))
        with self.assertRaisesRegex(ValueError, "absent from the frozen pins"):
            self.verify()

    def test_diagnostic_digest_mismatch_and_unpinned_extra_are_rejected(self):
        original = copy.deepcopy(self.diagnostics)
        name = next(iter(self.diagnostics["resource_files"]))
        self.diagnostics["resource_files"][name] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            self.verify()
        self.diagnostics = original
        extra = self.root / "unexpected.tab"
        extra.write_bytes(b"unexpected")
        self.diagnostics["resource_files"][str(extra)] = hashlib.sha256(extra.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "absent from the frozen pins"):
            self.verify()

    def test_empty_diagnostics_or_pins_are_not_success(self):
        self.diagnostics["resource_files"] = {}
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.verify()
        self.diagnostics["resource_files"] = {"irrelevant": "0" * 64}
        self.identity["nltk_resource_files"] = {}
        with self.assertRaisesRegex(ValueError, "explicitly pinned"):
            self.verify()

    def test_duplicate_binding_or_escape_is_rejected(self):
        self.identity["nltk_resource_files"]["alias"] = copy.deepcopy(next(iter(self.files.values())))
        with self.assertRaisesRegex(ValueError, "Duplicate frozen"):
            self.verify()
        self.identity["nltk_resource_files"].pop("alias")
        self.identity["nltk_resource_files"]["outside"] = {
            "path": str(self.root.parent / "outside.tab"), "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "escapes its fixed directory"):
            self.verify()


class Official77PinRuntimeTests(unittest.TestCase):
    def test_full_541_official_scoring_with_77_pins_and_four_required_files(self):
        # This runs the pinned CPU checker on artificial text, never model inference.
        assets = ep.load_evaluation_assets(
            ROOT / "data/evaluation-assets-20261020T103505Z/ASSET_MANIFEST.json",
            ep.ASSET_MANIFEST_SHA256)
        holdout = ep.load_final_holdout(
            ROOT / "reports/alignlab-v2-data-audit-v3-p1024-r768-20261020",
            ep.HOLDOUT_MANIFEST_SHA256)
        diagnostic = oe.ifeval_dependency_diagnostics(assets, nltk_data_dir=str(NLTK_DIR))
        self.assertTrue(diagnostic["ready"], "Run this regression in the accepted official scoring environment")
        self.assertEqual(len(diagnostic["resource_files"]), 4)
        resources = {str(path.resolve()): (str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in (NLTK_DIR / "tokenizers/punkt_tab").rglob("*") if path.is_file()}
        self.assertEqual(len(resources), 77)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "synthetic_not_model.json"
            manifest.write_bytes(b'{"synthetic_test_fixture":true,"model_inference":false}')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            protocol = ep.build_evaluation_protocol(
                assets, holdout, model_manifests={"fixture": (str(manifest), digest)},
                scorer_seed=0, bootstrap_seed=0, nltk_data_dir=str(NLTK_DIR),
                nltk_resource_files=resources)
            original_protocol = protocol._payload
            text = "This is a synthetic CPU regression fixture, not a model answer."
            outputs = [{"key": row["key"], "prompt": row["prompt"], "response": text,
                        "generated_token_ids": [7, 151645], "termination": "eos",
                        "prompt_tokens": assets.ifeval.lengths[str(row["key"])],
                        "generation_seed": 0, "model_id": "fixture",
                        "model_manifest_sha256": digest, "protocol_sha256": protocol.sha256,
                        "input_truncated": False} for row in assets.ifeval.rows]
            result = oe.score_ifeval(assets, outputs[::-1], protocol=protocol,
                                     model_id="fixture", decode_tokens=lambda ids: text if ids == (7,) else "")
            self.assertEqual(protocol._payload, original_protocol)
            self.assertEqual(len(protocol.identity["nltk_resource_files"]), 77)
            self.assertEqual(result["denominators"], {"prompts": 541, "instructions": 834})
            self.assertEqual(len(result["per_prompt"]), 541)
            response_map = {row["prompt"]: text for row in assets.ifeval.rows}
            with oe._official_ifeval(assets, 0, str(NLTK_DIR)) as official:
                for mode in ("strict", "loose"):
                    flags = []
                    for source, actual in zip(assets.ifeval.rows, result["per_prompt"]):
                        expected = getattr(official, "test_instruction_following_" + mode)(
                            official.InputExample(**source), response_map)
                        self.assertEqual(expected.follow_instruction_list,
                                         actual[mode]["follow_instruction_list"])
                        flags.append(expected.follow_instruction_list)
                    self.assertEqual(result["metrics"][mode + "_prompt_accuracy"],
                                     sum(all(row) for row in flags) / 541)
                    self.assertEqual(result["metrics"][mode + "_instruction_accuracy"],
                                     sum(map(sum, flags)) / 834)


if __name__ == "__main__":
    unittest.main()
