"""CPU optimizer/RNG boundary tests; no pretrained training or GPU validation."""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from alignment_lab import checkpointing as ckpt


def fixture():
    parameters = OrderedDict([
        ("policy.weight", nn.Parameter(torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64))),
        ("value.bias", nn.Parameter(torch.tensor([0.05, -0.02], dtype=torch.float64))),
    ])
    optimizer = torch.optim.AdamW([
        {"params": [parameters["policy.weight"]], "lr": 0.005},
        {"params": [parameters["value.bias"]], "lr": 0.003},
    ], betas=(0.8, 0.95), weight_decay=0.02, amsgrad=True, foreach=False)
    generators = {"sampling": torch.Generator(device="cpu").manual_seed(709)}
    return parameters, optimizer, generators


def seed():
    random.seed(301)
    np.random.seed(907)
    torch.manual_seed(411)


def step(parameters, optimizer, generators):
    uniform, gaussian = random.random(), random.gauss(0, 1)
    numpy_noise = np.random.normal(size=3)
    explicit = torch.randn((3, 2), generator=generators["sampling"], dtype=torch.float64)
    global_torch = torch.randn((3, 2), dtype=torch.float64)
    x = explicit + global_torch + uniform
    target = torch.from_numpy(numpy_noise[:, None]) + gaussian
    loss = ((x @ parameters["policy.weight"] + parameters["value.bias"]) - target).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"uniform": uniform, "gaussian": gaussian, "numpy": numpy_noise.tolist(),
            "explicit": explicit, "global_torch": global_torch, "loss": loss.item()}


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.original_python = random.getstate()
        self.original_numpy = np.random.get_state()
        self.original_torch = torch.get_rng_state()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.py"
        self.source.write_text("actual fixture source\n", encoding="utf-8")
        self.data = self.root / "preferences.jsonl"
        self.data.write_text('{"id":"a","chosen":"yes","rejected":"no"}\n', encoding="utf-8")
        self.parent = self.root / "parent.json"
        self.parent.write_text('{"stage":"sft","status":"complete"}', encoding="utf-8")
        self.identity = ckpt.build_identity({"source": self.source, "data": self.data},
                                           {"stage": "ppo", "seed": 301}, {"sft": self.parent})
        seed()

    def tearDown(self):
        random.setstate(self.original_python)
        np.random.set_state(self.original_numpy)
        torch.set_rng_state(self.original_torch)
        self.temp.cleanup()

    def assert_tree(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right), "Tensor differs bit-for-bit")
        elif isinstance(left, dict):
            self.assertEqual(list(left), list(right))
            for key in left:
                self.assert_tree(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree(a, b)
        else:
            self.assertEqual(left, right)

    def snapshot(self, parameters, optimizer, generators):
        return {"parameters": {key: p.detach().clone() for key, p in parameters.items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": ckpt._capture_rng(generators)}

    def saved(self, *, extra=None):
        parameters, optimizer, generators = fixture()
        step(parameters, optimizer, generators)
        destination = self.root / "checkpoint-1"
        saved = ckpt.save_checkpoint(destination, identity=self.identity, parameters=parameters,
                                     optimizer=optimizer, cursor={"step": 1}, generators=generators, extra_state=extra)
        return destination, saved

    def load(self, path, saved, parameters, optimizer, generators, *, anchored=True):
        return ckpt.load_checkpoint(path, expected_identity=self.identity, parameters=parameters,
                                    optimizer=optimizer, generators=generators,
                                    expected_manifest_sha256=saved["manifest_sha256"] if anchored else None)

    def rewrite_payload(self, directory, mutation):
        state_file = directory / "state.pt"
        payload = torch.load(state_file, map_location="cpu", weights_only=True)
        mutation(payload)
        torch.save(payload, state_file)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["state.pt"] = {"sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
                                                  "size_bytes": state_file.stat().st_size}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_adamw_resume_is_bitwise_equal_including_all_random_sources(self):
        continuous, optimizer, generators = fixture()
        for _ in range(3):
            step(continuous, optimizer, generators)
        destination = self.root / "step-3"
        saved = ckpt.save_checkpoint(destination, identity=self.identity, parameters=continuous,
                                     optimizer=optimizer, cursor={"step": 3, "epoch": 1, "offset": 5}, generators=generators)
        expected_trace = [step(continuous, optimizer, generators) for _ in range(5)]
        expected = self.snapshot(continuous, optimizer, generators)
        resumed, resumed_optimizer, resumed_generators = fixture()
        random.seed(888)
        np.random.seed(777)
        torch.manual_seed(666)
        resumed_generators["sampling"].manual_seed(555)
        loaded = self.load(destination, saved, resumed, resumed_optimizer, resumed_generators)
        self.assertEqual(loaded["cursor"], {"step": 3, "epoch": 1, "offset": 5})
        actual_trace = [step(resumed, resumed_optimizer, resumed_generators) for _ in range(5)]
        self.assert_tree(expected_trace, actual_trace)
        self.assert_tree(expected, self.snapshot(resumed, resumed_optimizer, resumed_generators))

    def test_ppo_rollout_extra_state_roundtrips_without_frozen_base_copy(self):
        extra = {"rollout": {"responses": [(11, 3, 2), (6, 9)], "record_ids": ["a", "b"],
                             "old_logprobs": torch.tensor([[-0.3, -0.8, -1.0], [-0.9, -0.4, 0.0]]),
                             "reference_logprobs": torch.tensor([[-0.4, -0.7, -1.1], [-0.8, -0.6, 0.0]]),
                             "old_values": torch.tensor([[0.3, 0.7, 0.4], [0.1, 0.2, 0.0]]),
                             "mask": torch.tensor([[True, True, True], [True, True, False]]),
                             "advantages": torch.tensor([[1.1, 0.5, -0.4], [-0.2, 0.8, 0.0]]),
                             "returns": torch.tensor([[1.4, 1.2, 0.0], [-0.1, 1.0, 0.0]]),
                             "reward_scores": torch.tensor([0.8, -0.2]), "epoch": 1, "offset": 1,
                             "termination": ("eos", "max_new_tokens")}}
        frozen_base = nn.Parameter(torch.ones(200000), requires_grad=False)
        before_base = frozen_base.clone()
        directory, saved = self.saved(extra=extra)
        parameters, optimizer, generators = fixture()
        with patch.object(torch, "load", wraps=torch.load) as reader:
            result = self.load(directory, saved, parameters, optimizer, generators)
        self.assertTrue(reader.call_args.kwargs["weights_only"])
        self.assertEqual(reader.call_args.kwargs["map_location"], "cpu")
        self.assert_tree(result["extra_state"], extra)
        self.assertEqual([item["name"] for item in saved["parameter_specs"]], ["policy.weight", "value.bias"])
        self.assertLess((directory / "state.pt").stat().st_size, frozen_base.numel() * frozen_base.element_size())
        self.assertTrue(torch.equal(frozen_base, before_base))

    def test_inputs_config_and_parent_identity_are_actual_file_bound(self):
        self.assertEqual(self.identity["inputs"]["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        directory, saved = self.saved()
        parameters, optimizer, generators = fixture()
        before = self.snapshot(parameters, optimizer, generators)
        changed_config = ckpt.build_identity({"source": self.source, "data": self.data}, {"stage": "dpo"}, {"sft": self.parent})
        with self.assertRaisesRegex(ValueError, "identity/config/parent"):
            ckpt.load_checkpoint(directory, expected_identity=changed_config, parameters=parameters,
                                  optimizer=optimizer, generators=generators)
        self.assert_tree(before, self.snapshot(parameters, optimizer, generators))
        for file in (self.source, self.data, self.parent):
            original = file.read_bytes()
            file.write_bytes(original + b" ")
            with self.subTest(file=file.name), self.assertRaisesRegex(ValueError, "Input identity changed"):
                self.load(directory, saved, parameters, optimizer, generators)
            self.assert_tree(before, self.snapshot(parameters, optimizer, generators))
            file.write_bytes(original)
        with self.assertRaises((ValueError, FileNotFoundError)):
            ckpt.build_identity({"source": "a" * 64}, {})
        with self.assertRaises(ValueError):
            ckpt.build_identity({}, {})

    def test_artifact_tamper_and_manifest_anchor_reject_without_any_target_mutation(self):
        directory, saved = self.saved()
        parameters, optimizer, generators = fixture()
        before = self.snapshot(parameters, optimizer, generators)
        artifact = directory / "state.pt"
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.load(directory, saved, parameters, optimizer, generators)
        self.assert_tree(before, self.snapshot(parameters, optimizer, generators))
        artifact.write_bytes(original)
        manifest = directory / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA anchor"):
            self.load(directory, saved, parameters, optimizer, generators)
        self.assert_tree(before, self.snapshot(parameters, optimizer, generators))

    def test_wrong_parameter_shape_dtype_optimizer_state_and_rng_reject_before_commit(self):
        mutations = {
            "parameter shape": lambda p: p["parameters"].__setitem__("policy.weight", torch.zeros(4, dtype=torch.float64)),
            "parameter dtype": lambda p: p["parameters"].__setitem__("policy.weight", p["parameters"]["policy.weight"].float()),
            "moment dtype": lambda p: p["optimizer"]["state"][0].__setitem__("exp_avg", p["optimizer"]["state"][0]["exp_avg"].float()),
            "missing moment": lambda p: p["optimizer"]["state"][0].pop("exp_avg_sq"),
            "nonfinite moment": lambda p: p["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")),
            "invalid RNG": lambda p: p["rng"].__setitem__("torch_cpu", torch.zeros(3, dtype=torch.uint8)),
            "invalid NumPy RNG": lambda p: p["rng"]["numpy"].__setitem__("position", 999),
            "extra payload": lambda p: p.__setitem__("unrecognized", True),
        }
        for index, (description, mutation) in enumerate(mutations.items()):
            with self.subTest(description=description):
                parameters, optimizer, generators = fixture()
                step(parameters, optimizer, generators)
                directory = self.root / f"case-{index}"
                saved = ckpt.save_checkpoint(directory, identity=self.identity, parameters=parameters,
                                             optimizer=optimizer, cursor={"step": 1}, generators=generators)
                self.rewrite_payload(directory, mutation)
                target, target_optimizer, target_generators = fixture()
                before = self.snapshot(target, target_optimizer, target_generators)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.load(directory, saved, target, target_optimizer, target_generators, anchored=False)
                self.assert_tree(before, self.snapshot(target, target_optimizer, target_generators))

    def test_optimizer_parameter_order_groups_and_hyperparameters_must_match(self):
        directory, saved = self.saved()
        for corruption in ("mapping-order", "optimizer-order", "lr", "betas", "groups"):
            with self.subTest(corruption=corruption):
                parameters, optimizer, generators = fixture()
                if corruption == "mapping-order":
                    parameters = OrderedDict(reversed(list(parameters.items())))
                elif corruption == "optimizer-order":
                    optimizer.param_groups.reverse()
                    parameters = OrderedDict(reversed(list(parameters.items())))
                elif corruption == "lr":
                    optimizer.param_groups[0]["lr"] = 0.5
                elif corruption == "betas":
                    optimizer.param_groups[0]["betas"] = (0.9, 0.999)
                else:
                    optimizer = torch.optim.AdamW(parameters.values(), lr=0.005, betas=(0.8, 0.95),
                                                  weight_decay=0.02, amsgrad=True, foreach=False)
                before = self.snapshot(parameters, optimizer, generators)
                with self.assertRaises(ValueError):
                    self.load(directory, saved, parameters, optimizer, generators)
                self.assert_tree(before, self.snapshot(parameters, optimizer, generators))

    def test_pending_gradients_reject_save_and_load_even_when_zero(self):
        directory, saved = self.saved()
        parameters, optimizer, generators = fixture()
        parameters["policy.weight"].grad = torch.zeros_like(parameters["policy.weight"])
        before = self.snapshot(parameters, optimizer, generators)
        with self.assertRaisesRegex(ValueError, "Pending gradients"):
            ckpt.save_checkpoint(self.root / "bad-save", identity=self.identity, parameters=parameters,
                                  optimizer=optimizer, cursor={"step": 1}, generators=generators)
        self.assertFalse((self.root / "bad-save").exists())
        with self.assertRaisesRegex(ValueError, "Pending gradients"):
            self.load(directory, saved, parameters, optimizer, generators)
        self.assert_tree(before, self.snapshot(parameters, optimizer, generators))
        self.assertIsNotNone(parameters["policy.weight"].grad)

    def test_save_refuses_nonfinite_parameter_optimizer_and_unsupported_extra_state(self):
        for kind in ("parameter", "optimizer", "custom", "gradient-extra", "nonfinite-json"):
            with self.subTest(kind=kind):
                parameters, optimizer, generators = fixture()
                step(parameters, optimizer, generators)
                extra = None
                if kind == "parameter":
                    with torch.no_grad():
                        parameters["policy.weight"][0, 0] = float("inf")
                elif kind == "optimizer":
                    optimizer.state[parameters["policy.weight"]]["exp_avg"][0, 0] = float("nan")
                elif kind == "custom":
                    extra = {"not_plain": object()}
                elif kind == "gradient-extra":
                    extra = torch.ones(2, requires_grad=True)
                else:
                    extra = {"bad": float("nan")}
                with self.assertRaises(ValueError):
                    ckpt.save_checkpoint(self.root / kind, identity=self.identity, parameters=parameters,
                                          optimizer=optimizer, cursor={"step": 1}, generators=generators, extra_state=extra)
                self.assertFalse((self.root / kind).exists())

    def test_immutable_directory_extra_files_and_incomplete_save_are_rejected(self):
        directory, saved = self.saved()
        parameters, optimizer, generators = fixture()
        original_manifest = (directory / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            ckpt.save_checkpoint(directory, identity=self.identity, parameters=parameters, optimizer=optimizer,
                                  cursor={"step": 99}, generators=generators)
        self.assertEqual((directory / "manifest.json").read_bytes(), original_manifest)
        (directory / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "extra files"):
            self.load(directory, saved, parameters, optimizer, generators)
        partial = self.root / "partial"
        with patch.object(ckpt, "_write_bytes", side_effect=OSError("simulated publish failure")):
            with self.assertRaisesRegex(OSError, "publish failure"):
                ckpt.save_checkpoint(partial, identity=self.identity, parameters=parameters, optimizer=optimizer,
                                      cursor={"step": 0}, generators=generators)
        self.assertTrue((partial / "state.pt").is_file())
        self.assertFalse((partial / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.load(partial, saved, parameters, optimizer, generators, anchored=False)

    def test_parameter_update_during_save_leaves_incomplete_evidence(self):
        parameters, optimizer, generators = fixture()
        original_save = torch.save
        def mutate(*args, **kwargs):
            result = original_save(*args, **kwargs)
            with torch.no_grad():
                parameters["policy.weight"].add_(0.01)
            return result
        directory = self.root / "interleaved-update"
        with patch.object(torch, "save", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                ckpt.save_checkpoint(directory, identity=self.identity, parameters=parameters,
                                      optimizer=optimizer, cursor={"step": 0}, generators=generators)
        self.assertTrue((directory / "state.pt").is_file())
        self.assertFalse((directory / "manifest.json").exists())

    def test_explicit_generator_binding_and_parameter_aliases_are_checked(self):
        directory, saved = self.saved()
        parameters, optimizer, generators = fixture()
        before = self.snapshot(parameters, optimizer, generators)
        with self.assertRaisesRegex(ValueError, "generator/runtime"):
            self.load(directory, saved, parameters, optimizer, {"renamed": generators["sampling"]})
        self.assert_tree(before, self.snapshot(parameters, optimizer, generators))
        shared = nn.Parameter(torch.ones(4))
        alias = nn.Parameter(shared.view(2, 2))
        aliased = OrderedDict(a=shared, b=alias)
        aliased_optimizer = torch.optim.AdamW(aliased.values())
        with self.assertRaisesRegex(ValueError, "share storage"):
            ckpt.save_checkpoint(self.root / "alias", identity=self.identity, parameters=aliased,
                                  optimizer=aliased_optimizer, cursor={}, generators={})

    def test_sgd_and_adam_supported_state_schemas_roundtrip(self):
        for cls in (torch.optim.SGD, torch.optim.Adam):
            with self.subTest(optimizer=cls.__name__):
                parameters, _, generators = fixture()
                options = {"momentum": 0.8} if cls is torch.optim.SGD else {"amsgrad": False}
                optimizer = cls(parameters.values(), lr=0.01, **options)
                step(parameters, optimizer, generators)
                directory = self.root / cls.__name__
                saved = ckpt.save_checkpoint(directory, identity=self.identity, parameters=parameters,
                                             optimizer=optimizer, cursor={"step": 1}, generators=generators)
                expected = self.snapshot(parameters, optimizer, generators)
                target, _, target_generators = fixture()
                target_optimizer = cls(target.values(), lr=0.01, **options)
                self.load(directory, saved, target, target_optimizer, target_generators)
                self.assert_tree(expected, self.snapshot(target, target_optimizer, target_generators))


if __name__ == "__main__":
    unittest.main()
