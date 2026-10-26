"""Tiny random CPU serializer/inference fixtures, never real training acceptance."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]

import torch

from alignment_lab import checkpointing as ckpt
from alignment_lab import evaluation_model_v2 as evaluation
from alignment_lab import stage_artifacts as stages
from alignment_lab.model_loader import LoadedStage
import test_stage_artifacts as artifact_fixture
from test_stage_artifacts import backend_for, snapshot


class EvaluationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.fixture = artifact_fixture.StageArtifactTests()
        self.fixture.setUp()
        self.root, self.files = self.fixture.root, self.fixture.files
        self.snapshot_sha = "a" * 64  # Explicit fixture identifier; no official model load.
        self.records, self.sources, self.receipts, self.accepted = {}, {}, {}, {}
        self.acceptance_path = self.root / "fixture-acceptance.json"
        self._make_stage("sft")
        self._make_stage("reward")
        self._make_stage("dpo")
        self._make_stage("ppo")
        self.accepted["base"] = dict(stage="base", artifact_dir=None, manifest_sha256=None,
                                     progress_path=None, progress_sha256=None)
        self._write_acceptance()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.fixture.tearDown()

    def _make_stage(self, stage):
        backend, _ = backend_for(stage)
        role = "reward" if stage == "reward" else "policy"
        rows, updates = (4000, 250) if stage == "sft" else (1024, 2048) if stage == "ppo" else (13139, 822)
        scope = evaluation.SUBSET_SCOPE if stage == "ppo" else evaluation.FULL_SCOPE
        config = dict(stage=stage, fixture_only="tiny-random-CPU-not-training",
                      formal_training=True, synthetic_rows=0, scope=scope)
        complete = dict(stop_reason="data_exhausted", data_pass_complete=True, protocol_complete=True,
                        full_accepted_dataset_pass=stage != "ppo", scope=scope, synthetic_rows=0,
                        rows_processed=rows, optimizer_updates=updates)
        if stage in {"sft", "reward"}:
            config.update(purpose=evaluation.PARENT_KIND, expected_rows=rows, expected_updates=updates)
        else:
            config.update(purpose=evaluation.ALIGNMENT_KIND, work={"seed": 42},
                          registered_workload=dict(rows=rows, optimizer_updates=updates, scope=scope))
            complete["full_selected_dataset_pass"] = True
        parents = {} if stage == "sft" else {"sft": self.receipts["sft"]}
        if stage == "ppo":
            parents["reward"] = self.receipts["reward"]
        identity = ckpt.build_identity(self.files, config, {n: r["manifest_path"] for n, r in parents.items()})
        for target, parent, heads in ([] if stage == "sft" else
                [("reward" if stage == "reward" else "policy", "sft", ())]) + (
                [("reference", "sft", ())] if stage in {"dpo", "ppo"} else []) + (
                [("reward", "reward", ("reward",))] if stage == "ppo" else []):
            item = self.receipts[parent]
            stages.import_stage(item["artifact_dir"], backend, target,
                expected_manifest_sha256=item["manifest_sha256"], expected_identity=identity, heads=heads)
        heads = ("reward",) if stage == "reward" else ("value",) if stage == "ppo" else ()
        with torch.no_grad():
            for parameter in backend._parameters[role]:
                parameter.add_({"sft": .011, "reward": .022, "dpo": .033, "ppo": .044}[stage])
            for head in heads:
                backend._heads[head].weight.add_(.17)
        receipt = stages.export_stage(self.root / f"{stage}-artifact", backend, role,
            stage=stage, identity=identity, heads=heads, completion=complete)
        progress = dict(status="formal_parent_stage_passed" if stage in {"sft", "reward"} else "formal_alignment_stage_passed",
            formal_training=True, research_complete=False, identity=identity, completion=complete,
            actual_optimizer_updates=updates,
            artifact={k: receipt[k] for k in ("artifact_dir", "manifest_path", "manifest_sha256")})
        progress_path = self.root / f"{stage}-progress.json"
        progress_path.write_bytes(ckpt._canonical(progress))
        self.accepted[stage] = dict(stage=stage, artifact_dir=receipt["artifact_dir"],
            manifest_sha256=receipt["manifest_sha256"], progress_path=str(progress_path),
            progress_sha256=ckpt._file(progress_path)["sha256"])
        self.sources[stage], self.receipts[stage] = backend, receipt

    def _write_acceptance(self):
        self.acceptance_path.write_bytes(ckpt._canonical(dict(
            schema_version=2, kind=evaluation.ACCEPTANCE_KIND, status="accepted_for_evaluation",
            all_checks_passed=True, processes_exited=True, fixture_only=True,
            snapshot_manifest_sha256=self.snapshot_sha,
            models={self._model_id(stage): row for stage, row in self.accepted.items()})))
        for stage, item in self.accepted.items():
            self.records[stage] = dict(model_id=self._model_id(stage), stage=stage,
                artifact_dir=item["artifact_dir"], manifest_sha256=item["manifest_sha256"],
                acceptance_path=str(self.acceptance_path),
                acceptance_sha256=ckpt._file(self.acceptance_path)["sha256"])

    @staticmethod
    def _model_id(stage):
        return stage + "42" if stage in {"dpo", "ppo"} else stage

    def _loaded(self, stage):
        backend, _ = backend_for("reward" if stage == "reward" else "sft")
        role = "reward" if stage == "reward" else "policy"
        return LoadedStage(backend, role, ("reward",) if stage == "reward" else (),
            dict(snapshot_manifest_sha256=self.snapshot_sha, quantization="none", fixture_only=True),
            torch.device("cpu"))

    def _bind(self, stage, loaded=None, **kwargs):
        return evaluation.bind_evaluation_model(loaded or self._loaded(stage), self.records[stage],
            input_files=kwargs.get("input_files", self.files))

    def test_final_sft_dpo_ppo_copy_exact_policy_only_and_never_value_head(self):
        for stage in ("sft", "dpo", "ppo"):
            with self.subTest(stage=stage):
                loaded = self._loaded(stage)
                before = snapshot(loaded.backend)
                target = self._bind(stage, loaded)
                source = stages._parameters(self.sources[stage], "policy", ())
                actual = stages._parameters(target._backend, "policy", ())
                for name in source:
                    torch.testing.assert_close(actual[name], source[name], rtol=0, atol=0)
                for name, parameter in loaded.backend._model.named_parameters():
                    if ".policy." not in name:
                        self.assertTrue(torch.equal(parameter, before[name]), name)
                for head, module in loaded.backend._heads.items():
                    for name, parameter in module.named_parameters():
                        self.assertTrue(torch.equal(parameter, before[f"head.{head}.{name}"]))
                self.assertTrue(target.frozen_audit(full=True)["unchanged"])
                self.assertFalse(target.metadata["ppo_value_head_used"])

    def test_base_actually_disables_lora_and_matches_dense_base(self):
        loaded = self._loaded("base")
        model = self._bind("base", loaded)
        seen = []
        hook = model._backend._base.model.register_forward_pre_hook(
            lambda *_: seen.append(all(m.disable_adapters for m in model._backend._model.modules() if hasattr(m, "lora_A"))))
        try:
            result = model.greedy_generate((4, 8, 9), eos_id=2, max_new_tokens=4)
        finally:
            hook.remove()
        self.assertTrue(all(seen))
        tokens, expected = (4, 8, 9), []
        with torch.no_grad(), loaded.backend._model.disable_adapter():
            for _ in range(4):
                ids = torch.tensor([tokens])
                output = loaded.backend._base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                token = int(output.logits[0, -1].argmax())
                expected.append(token)
                tokens += (token,)
                if token == 2:
                    break
        self.assertEqual(result["generated_token_ids"], tuple(expected))
        self.assertTrue(model.frozen_audit(full=True)["unchanged"])

    def test_reward_import_scalar_head_and_pool_real_eos_and_pad_actions(self):
        model = self._bind("reward")
        source = self.sources["reward"]
        for name, parameter in source._heads["reward"].named_parameters():
            torch.testing.assert_close(parameter, getattr(model._backend._heads["reward"], name), rtol=0, atol=0)
        for completion in ((11, 2), (11, 0), (11, 0, 2)):
            prompt = (4, 8, 9)
            score = model.reward_score(prompt, completion)
            ids = torch.tensor([prompt + completion + (0, 0)])
            attention = torch.tensor([[1] * (len(prompt) + len(completion)) + [0, 0]])
            labels = ids.clone()
            labels[:, :len(prompt)] = -100
            labels[:, -2:] = -100
            expected = source.read("reward", lambda view: view.rewards(ids, attention, labels)).item()
            self.assertAlmostEqual(score, expected, places=6)
        self.assertTrue(model.frozen_audit(full=True)["unchanged"])

    def test_normal_eos_length_real_pad_ids_no_forced_stress_or_truncation(self):
        model = self._bind("sft")
        for token, reason, expected in ((2, "eos", (2,)), (0, "length", (0, 0, 0))):
            logits = torch.zeros(1, 97); logits[0, token] = 9
            with patch.object(model._backend._base.lm_head, "forward", return_value=logits):
                result = model.greedy_generate((4, 8, 9), eos_id=2, max_new_tokens=3)
            self.assertEqual(result["termination"], reason)
            self.assertEqual(result["generated_token_ids"], expected)
            self.assertFalse(result["synthetic_eos_added"])
            self.assertFalse(result["prompt_truncated"])
            self.assertNotIn("capacity_only", result)
        with self.assertRaises(ValueError):
            model.greedy_generate((4,) * 127, eos_id=2, max_new_tokens=3)
        with self.assertRaises(ValueError):
            model.greedy_generate((4,), eos_id=2, max_new_tokens=2049)
        with self.assertRaises(TypeError):
            model.greedy_generate((4,), eos_id=2, ignore_eos=True)
        with self.assertRaises(ValueError):
            model.greedy_generate((4,), eos_id=3, max_new_tokens=3)

    def test_stop_returns_incomplete_and_does_not_forward(self):
        model = self._bind("sft")
        with patch.object(model._backend._base.model, "forward", wraps=model._backend._base.model.forward) as call:
            result = model.greedy_generate((4, 8), eos_id=2, max_new_tokens=3, stop_requested=lambda: True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["termination"], "stop_requested")
        self.assertEqual(call.call_count, 0)

    def test_read_only_no_training_no_grad_no_optimizer_no_rng_changes(self):
        model = self._bind("ppo")
        before, rng = snapshot(model._backend), torch.get_rng_state().clone()
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No optimizer allowed")):
            model.greedy_generate((4, 8), eos_id=2, max_new_tokens=4)
        for name, parameter in snapshot(model._backend).items():
            self.assertTrue(torch.equal(before[name], parameter))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model._backend._all_parameters()))
        with self.assertRaises(RuntimeError):
            model._backend.backward(None)
        with self.assertRaises(ValueError):
            model._backend.parameters_for("policy")
        with self.assertRaises(ValueError):
            model.reward_score((4,), (2,))

    def test_wrong_tokenizer_and_base_input_identity_rejected_before_copy(self):
        for label in ("tokenizer/tokenizer.json", "base/weights.pt"):
            altered = self.root / ("different-" + Path(self.files[label]).name)
            altered.write_bytes(b"different")
            files = dict(self.files); files[label] = altered
            loaded = self._loaded("sft")
            before = snapshot(loaded.backend)
            with self.assertRaisesRegex(ValueError, "inventories"):
                self._bind("sft", loaded, input_files=files)
            for name, parameter in snapshot(loaded.backend).items():
                self.assertTrue(torch.equal(before[name], parameter))

    def test_changed_acceptance_progress_or_weights_rejected(self):
        for stage, target in (("base", self.acceptance_path),
                              ("sft", Path(self.accepted["sft"]["progress_path"])),
                              ("dpo", Path(self.receipts["dpo"]["artifact_dir"]) / "weights.pt")):
            original = target.read_bytes()
            try:
                target.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    self._bind(stage)
            finally:
                target.write_bytes(original)

    def test_capacity_or_partial_artifact_rejected_even_with_reanchored_acceptance(self):
        for purpose in ("alignlab-capacity-stage-v1", evaluation.PARENT_KIND):
            receipt = self.receipts["sft"]
            path = Path(receipt["manifest_path"])
            original = path.read_bytes()
            manifest = ckpt._parse(original)
            manifest["identity"]["config"]["purpose"] = purpose
            if purpose == evaluation.PARENT_KIND:
                manifest["completion"]["rows_processed"] = 1
            manifest["identity_sha256"] = ckpt._digest(ckpt._canonical(manifest["identity"]))
            path.write_bytes(ckpt._canonical(manifest))
            self.accepted["sft"]["manifest_sha256"] = ckpt._file(path)["sha256"]
            self._write_acceptance()
            with self.assertRaises(ValueError):
                self._bind("sft")
            path.write_bytes(original)
            self.accepted["sft"]["manifest_sha256"] = receipt["manifest_sha256"]
            self._write_acceptance()

    def test_parent_tamper_and_wrong_lineage_rejected(self):
        path = Path(self.receipts["sft"]["artifact_dir"]) / "weights.pt"
        raw = path.read_bytes(); path.write_bytes(raw + b"broken")
        with self.assertRaises(ValueError):
            self._bind("ppo")
        path.write_bytes(raw)
        wrong = copy.deepcopy(self.records["ppo"])
        wrong["artifact_dir"] = self.receipts["dpo"]["artifact_dir"]
        with self.assertRaises(ValueError):
            evaluation.bind_evaluation_model(self._loaded("ppo"), wrong, input_files=self.files)

    def test_ppo_value_head_payload_must_be_valid_even_when_not_used(self):
        path = Path(self.receipts["ppo"]["artifact_dir"])
        payload = torch.load(path / "weights.pt", weights_only=True)
        payload["heads/value/weight"][0, 0] = float("nan")
        torch.save(payload, path / "weights.pt")
        manifest = ckpt._parse((path / "manifest.json").read_bytes())
        record = ckpt._file(path / "weights.pt")
        manifest["artifacts"]["weights.pt"] = {k: record[k] for k in ("sha256", "size_bytes")}
        (path / "manifest.json").write_bytes(ckpt._canonical(manifest))
        digest = ckpt._file(path / "manifest.json")["sha256"]
        self.accepted["ppo"]["manifest_sha256"] = digest
        progress_path = Path(self.accepted["ppo"]["progress_path"])
        progress = ckpt._parse(progress_path.read_bytes())
        progress["artifact"]["manifest_sha256"] = digest
        progress_path.write_bytes(ckpt._canonical(progress))
        self.accepted["ppo"]["progress_sha256"] = ckpt._file(progress_path)["sha256"]
        self._write_acceptance()
        with self.assertRaises((ValueError, FloatingPointError)):
            self._bind("ppo")

    def test_full_audit_detects_untracked_tensor_write_and_quick_audit_tracked_write(self):
        model = self._bind("sft")
        parameter = next(model._backend._heads["value"].parameters())
        parameter.data.add_(.1)  # Intentional hostile fixture: version guard cannot catch this.
        self.assertTrue(model.frozen_audit(full=False)["unchanged"])
        with self.assertRaisesRegex(RuntimeError, "contents"):
            model.frozen_audit(full=True)
        another = self._bind("sft")
        with torch.no_grad():
            another._backend._parameters["policy"][0].add_(.1)
        with self.assertRaisesRegex(RuntimeError, "version"):
            another.frozen_audit(full=False)

    def test_model_metadata_and_adapter_structure_mutation_rejected(self):
        model = self._bind("sft")
        model._backend._base.config.rms_norm_eps *= 2
        with self.assertRaises(RuntimeError):
            model.frozen_audit(full=False)
        other = self._bind("sft")
        layer = next(m for m in other._backend._model.modules() if hasattr(m, "lora_A"))
        layer.scaling["policy"] += 1
        with self.assertRaises(RuntimeError):
            other.frozen_audit(full=False)

    def test_production_rejects_bad_acceptance_before_cuda_loader(self):
        with patch.object(evaluation, "load_quantized_stage", side_effect=AssertionError("CUDA must not load")), \
                patch("alignment_lab.model_snapshot.verify_snapshot", return_value={"files": {}}):
            tokenizer = self.root / "tokenizer"; tokenizer.mkdir()
            (tokenizer / "source-manifest.json").write_text('{"artifacts":{}}')
            with self.assertRaises((ValueError, FileNotFoundError)):
                evaluation.load_evaluation_model(self.records["base"], model_snapshot=self.root,
                    model_manifest_sha256="b" * 64, tokenizer_dir=tokenizer)

    def test_pending_grad_or_active_callback_cannot_be_rebound_as_base(self):
        loaded = self._loaded("base")
        parameter = loaded.backend._parameters["policy"][0]
        parameter.grad = torch.zeros_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "gradients"):
            self._bind("base", loaded)
        self.assertIsNotNone(parameter.grad)
        parameter.grad = None
        loaded.backend._phase = "read"
        with self.assertRaisesRegex(RuntimeError, "idle"):
            self._bind("base", loaded)

    def test_wrong_lora_structure_and_reanchored_false_progress_rejected(self):
        loaded = self._loaded("sft")
        layer = next(m for m in loaded.backend._model.modules() if hasattr(m, "lora_A"))
        layer.scaling["policy"] *= 2
        with self.assertRaisesRegex(ValueError, "LoRA/base"):
            self._bind("sft", loaded)
        progress_path = Path(self.accepted["sft"]["progress_path"])
        progress = ckpt._parse(progress_path.read_bytes())
        progress["actual_optimizer_updates"] = 1
        progress_path.write_bytes(ckpt._canonical(progress))
        self.accepted["sft"]["progress_sha256"] = ckpt._file(progress_path)["sha256"]
        self._write_acceptance()
        with self.assertRaisesRegex(ValueError, "progress.*artifact"):
            self._bind("sft")


if __name__ == "__main__":
    unittest.main()
