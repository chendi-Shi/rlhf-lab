"""Tiny random CPU Qwen/PEFT artifact tests, not completed research training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from alignment_lab import checkpointing as ckpt
from alignment_lab import stage_artifacts as stages
from alignment_lab.shared_qwen import SharedQwen3
from test_shared_qwen import fixture


COMPLETE = {"stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": True,
            "optimizer_updates": 1, "rows_processed": 2}


def backend_for(stage):
    _, model, heads, ids, attention, labels = fixture()
    role = "reward" if stage == "reward" else "policy"
    owned_heads = ("reward",) if stage == "reward" else ("value",) if stage == "ppo" else ()
    backend = SharedQwen3(model, trainable_roles={role}, heads=heads,
                          trainable_heads={role: owned_heads} if owned_heads else {})
    return backend, dict(input_ids=ids, attention_mask=attention, labels=labels)


def snapshot(backend):
    return {name: p.detach().clone() for name, p in backend._model.named_parameters()} | {
        f"head.{head}.{name}": p.detach().clone() for head, module in backend._heads.items() for name, p in module.named_parameters()}


class StageArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        backend, _ = backend_for("sft")
        adapter_ids = {id(p) for params in backend._parameters.values() for p in params}
        base = {name: p.detach().clone() for name, p in backend._model.named_parameters() if id(p) not in adapter_ids}
        torch.save(base, self.root / "tiny-base.pt")
        (self.root / "tokenizer.json").write_text('{"fixture_vocab_size":97,"eos_id":2}', encoding="utf-8")
        self.files = {"base/weights.pt": self.root / "tiny-base.pt", "tokenizer/tokenizer.json": self.root / "tokenizer.json",
                      "source/stage_artifacts.py": Path(stages.__file__)}
        self.serial = 0

    def tearDown(self):
        self.temp.cleanup()

    def identity(self, stage, parents=None, *, files=None):
        return ckpt.build_identity(self.files if files is None else files, {"stage": stage, "fixture": "tiny-random-cpu"},
                                   {name: receipt["manifest_path"] for name, receipt in (parents or {}).items()})

    def directory(self, label):
        self.serial += 1
        return self.root / f"{label}-{self.serial}"

    def sft(self, *, role="policy", perturb=0.1):
        backend, batch = backend_for("sft")
        if role != "policy":
            backend._model.add_adapter(role, copy.deepcopy(backend._model.peft_config["policy"]))
            backend = SharedQwen3(backend._model, trainable_roles={role}, heads=backend._heads)
        # Distinct deterministic fixture weights make role remapping observable.
        # COMPLETE is serializer-test metadata, not a claim of a research run.
        with torch.no_grad():
            for p in backend._parameters[role]:
                p.add_(perturb)
        identity = self.identity("sft")
        receipt = stages.export_stage(self.directory("sft"), backend, role, stage="sft", identity=identity, completion=COMPLETE)
        return backend, batch, receipt

    def import_(self, source, backend, target_role, identity, heads=()):
        return stages.import_stage(source["artifact_dir"], backend, target_role,
                                    expected_manifest_sha256=source["manifest_sha256"], expected_identity=identity, heads=heads)

    def reward(self, sft):
        backend, _ = backend_for("reward")
        identity = self.identity("reward", {"sft": sft})
        self.import_(sft, backend, "reward", identity)
        with torch.no_grad():
            for p in backend._parameters["reward"]:
                p.add_(0.03)
            backend._heads["reward"].weight.add_(0.2)
        receipt = stages.export_stage(self.directory("reward"), backend, "reward", stage="reward", identity=identity,
                                      heads=("reward",), completion=COMPLETE)
        return backend, receipt

    def assert_unchanged(self, before, backend):
        after = snapshot(backend)
        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def assert_adapter(self, source, source_role, target, target_role):
        left = stages._parameters(source, source_role, ())
        right = stages._parameters(target, target_role, ())
        self.assertEqual(list(left), list(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_role_normalization_sft_to_policy_reference_reward_without_switching(self):
        source, batch, artifact = self.sft(role="sft_source")
        for stage, role in (("dpo", "policy"), ("dpo", "reference"), ("reward", "reward")):
            with self.subTest(stage=stage, role=role):
                target, _ = backend_for(stage)
                before = snapshot(target)
                mode = target._model.training
                flags = [p.requires_grad for p in target._all_parameters()]
                identity = self.identity(stage, {"sft": artifact})
                receipt = self.import_(artifact, target, role, identity)
                self.assert_adapter(source, "sft_source", target, role)
                self.assertIsNone(target._active_role)
                self.assertEqual(target._model.training, mode)
                self.assertEqual([p.requires_grad for p in target._all_parameters()], flags)
                self.assertEqual(receipt["manifest_sha256"], artifact["manifest_sha256"])
                selected = {id(p) for p in target._parameters[role]}
                for name, p in target._model.named_parameters():
                    if id(p) not in selected:
                        self.assertTrue(torch.equal(p, before[name]), name)
                for name, module in target._heads.items():
                    for local, p in module.named_parameters():
                        self.assertTrue(torch.equal(p, before[f"head.{name}.{local}"]))
                actual = target.read(role, lambda view: view.sequence_scores(**batch).sums)
                expected = source.read("sft_source", lambda view: view.sequence_scores(**batch).sums)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_complete_reward_and_ppo_lineage_copies_only_reward_head(self):
        source, _, sft = self.sft()
        reward_backend, reward = self.reward(sft)
        target, _ = backend_for("ppo")
        identity = self.identity("ppo", {"sft": sft, "reward": reward})
        value_before = copy.deepcopy(target._heads["value"].state_dict())
        self.import_(sft, target, "policy", identity)
        self.import_(sft, target, "reference", identity)
        self.import_(reward, target, "reward", identity, heads=("reward",))
        self.assert_adapter(source, "policy", target, "policy")
        self.assert_adapter(source, "policy", target, "reference")
        self.assert_adapter(reward_backend, "reward", target, "reward")
        for key, value in reward_backend._heads["reward"].state_dict().items():
            self.assertTrue(torch.equal(value, target._heads["reward"].state_dict()[key]))
        for key, value in value_before.items():
            self.assertTrue(torch.equal(value, target._heads["value"].state_dict()[key]))
        final = stages.export_stage(self.directory("ppo"), target, "policy", stage="ppo", identity=identity,
                                    heads=("value",), completion=COMPLETE)
        payload = torch.load(Path(final["artifact_dir"]) / "weights.pt", weights_only=True, map_location="cpu")
        self.assertTrue(all(key.startswith("adapter/") or key.startswith("heads/value/") for key in payload))
        self.assertFalse(any("heads/reward" in key for key in payload))

    def test_ppo_cannot_start_from_dpo_even_with_an_authentic_completed_dpo(self):
        _, _, sft = self.sft()
        dpo_backend, _ = backend_for("dpo")
        dpo_identity = self.identity("dpo", {"sft": sft})
        for role in ("policy", "reference"):
            self.import_(sft, dpo_backend, role, dpo_identity)
        dpo = stages.export_stage(self.directory("dpo"), dpo_backend, "policy", stage="dpo", identity=dpo_identity, completion=COMPLETE)
        _, reward = self.reward(sft)
        ppo, _ = backend_for("ppo")
        before = snapshot(ppo)
        with self.assertRaisesRegex(ValueError, "never DPO"):
            self.import_(dpo, ppo, "policy", self.identity("ppo", {"sft": sft, "reward": reward}))
        self.assert_unchanged(before, ppo)

    def test_completion_flags_and_unsupported_dose_cannot_publish_partial_weights(self):
        backend, _ = backend_for("sft")
        for bad in (
            {"stop_reason": "budget_exhausted", "data_pass_complete": False, "protocol_complete": False},
            {"stop_reason": "paused", "data_pass_complete": True, "protocol_complete": True},
            {"stop_reason": "failed", "data_pass_complete": True, "protocol_complete": True},
            {"stop_reason": "data_exhausted", "data_pass_complete": True, "protocol_complete": False},
            {"stop_reason": "dose_complete", "data_pass_complete": False, "protocol_complete": True},
            {"stop_reason": "data_exhausted", "data_pass_complete": 1, "protocol_complete": True},
        ):
            path = self.directory("incomplete")
            with self.subTest(completion=bad), self.assertRaises(ValueError):
                stages.export_stage(path, backend, "policy", stage="sft", identity=self.identity("sft"), completion=bad)
            self.assertFalse(path.exists())

    def test_matching_parent_strings_without_verified_import_receipts_are_insufficient(self):
        _, _, sft = self.sft()
        target, _ = backend_for("dpo")
        identity = self.identity("dpo", {"sft": sft})
        with self.assertRaisesRegex(ValueError, "lacks verified initialization"):
            stages.export_stage(self.directory("false-dpo"), target, "policy", stage="dpo", identity=identity, completion=COMPLETE)
        self.import_(sft, target, "policy", identity)
        with self.assertRaisesRegex(ValueError, "reference lacks"):
            stages.export_stage(self.directory("missing-reference"), target, "policy", stage="dpo", identity=identity, completion=COMPLETE)

    def test_wrong_sft_ancestor_in_reward_lineage_is_rejected_before_copy(self):
        _, _, sft_a = self.sft(perturb=0.1)
        _, _, sft_b = self.sft(perturb=0.2)
        _, reward_b = self.reward(sft_b)
        target, _ = backend_for("ppo")
        before = snapshot(target)
        with self.assertRaisesRegex(ValueError, "same SFT"):
            self.import_(sft_a, target, "policy", self.identity("ppo", {"sft": sft_a, "reward": reward_b}))
        self.assert_unchanged(before, target)

    def test_base_tokenizer_hash_and_lora_configuration_mismatch_reject_before_copy(self):
        _, _, sft = self.sft()
        for kind in ("base", "tokenizer", "lora", "runtime-scaling", "dtype"):
            with self.subTest(kind=kind):
                target, _ = backend_for("dpo")
                files = dict(self.files)
                if kind in {"base", "tokenizer"}:
                    name = "base/weights.pt" if kind == "base" else "tokenizer/tokenizer.json"
                    modified = self.root / f"different-{kind}"
                    modified.write_bytes(Path(files[name]).read_bytes() + b"changed")
                    files[name] = modified
                elif kind == "lora":
                    target._model.peft_config["policy"].lora_alpha += 1
                elif kind == "runtime-scaling":
                    module = next(module for module in target._model.modules() if hasattr(module, "lora_A"))
                    module.scaling["policy"] *= 2
                else:
                    target._parameters["policy"][0].data = target._parameters["policy"][0].data.double()
                before = snapshot(target)
                identity = self.identity("dpo", {"sft": sft}, files=files)
                with self.assertRaises(ValueError):
                    self.import_(sft, target, "policy", identity)
                self.assert_unchanged(before, target)

    def test_identical_base_files_at_new_paths_remain_compatible(self):
        source, _, sft = self.sft()
        relocated = self.root / "relocated-base.pt"
        relocated.write_bytes(Path(self.files["base/weights.pt"]).read_bytes())
        files = {**self.files, "base/weights.pt": relocated}
        target, _ = backend_for("dpo")
        self.import_(sft, target, "policy", self.identity("dpo", {"sft": sft}, files=files))
        self.assert_adapter(source, "policy", target, "policy")

    def test_pending_gradients_active_callbacks_and_wrong_reference_trainability_rejected(self):
        source, _, sft = self.sft()
        target, _ = backend_for("dpo")
        identity = self.identity("dpo", {"sft": sft})
        target._heads["value"].weight.grad = torch.zeros_like(target._heads["value"].weight)
        before = snapshot(target)
        with self.assertRaisesRegex(RuntimeError, "Pending gradients"):
            self.import_(sft, target, "policy", identity)
        self.assert_unchanged(before, target)
        target._heads["value"].weight.grad = None
        with self.assertRaisesRegex(RuntimeError, "idle backend"):
            target.read("policy", lambda view: self.import_(sft, target, "reference", identity))
        with self.assertRaisesRegex(RuntimeError, "idle backend"):
            source.read("policy", lambda view: stages.export_stage(self.directory("inside-read"), source, "policy",
                stage="sft", identity=self.identity("sft"), completion=COMPLETE))
        target._trainable_roles.add("reference")
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.import_(sft, target, "reference", identity)

    def test_scalar_head_storage_alias_cannot_modify_an_unselected_head(self):
        _, _, sft = self.sft()
        _, reward = self.reward(sft)
        target, _ = backend_for("ppo")
        target._heads["value"].weight = torch.nn.Parameter(target._heads["reward"].weight.detach())
        before = snapshot(target)
        with self.assertRaisesRegex(ValueError, "share storage"):
            self.import_(reward, target, "reward", self.identity("ppo", {"sft": sft, "reward": reward}), heads=("reward",))
        self.assert_unchanged(before, target)

    def test_manifest_anchor_artifact_tamper_extra_files_and_wrong_payload_dtype(self):
        _, _, sft = self.sft()
        target, _ = backend_for("dpo")
        before = snapshot(target)
        identity = self.identity("dpo", {"sft": sft})
        path = Path(sft["artifact_dir"])
        with self.assertRaisesRegex(ValueError, "anchor"):
            stages.import_stage(path, target, "policy", expected_manifest_sha256="0" * 64, expected_identity=identity)
        original = (path / "weights.pt").read_bytes()
        (path / "weights.pt").write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "weights hash"):
            self.import_(sft, target, "policy", identity)
        (path / "weights.pt").write_bytes(original)
        (path / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra files"):
            self.import_(sft, target, "policy", identity)
        (path / "extra.txt").unlink()
        payload = torch.load(path / "weights.pt", weights_only=True, map_location="cpu")
        name = next(iter(payload))
        payload[name] = payload[name].double()
        torch.save(payload, path / "weights.pt")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        changed = (path / "weights.pt").read_bytes()
        manifest["artifacts"]["weights.pt"] = {"sha256": hashlib.sha256(changed).hexdigest(), "size_bytes": len(changed)}
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        forged = {**sft, "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "payload shape/dtype"):
            self.import_(forged, target, "policy", self.identity("dpo", {"sft": forged}))
        self.assert_unchanged(before, target)

    def test_export_new_directory_only_and_publication_failure_keeps_evidence(self):
        source, _, receipt = self.sft()
        path = Path(receipt["artifact_dir"])
        original = (path / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            stages.export_stage(path, source, "policy", stage="sft", identity=self.identity("sft"), completion=COMPLETE)
        self.assertEqual((path / "manifest.json").read_bytes(), original)
        partial = self.directory("partial")
        with patch.object(ckpt, "_write_bytes", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                stages.export_stage(partial, source, "policy", stage="sft", identity=self.identity("sft"), completion=COMPLETE)
        self.assertTrue((partial / "weights.pt").is_file())
        self.assertFalse((partial / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
