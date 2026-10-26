"""CPU-only candidate checks. These are not actual CUDA speed/equality proof."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from alignment_lab.rollout import sample_rollout
from alignment_lab.shared_qwen import SharedQwen3
from alignment_lab.throughput_sampling_v1 import (
    GUARD_MODES, ThroughputModelView, benchmark_sampling_equivalence,
    make_sampling_backend,
)


def fixture():
    torch.manual_seed(131)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, pad_token_id=0,
        eos_token_id=2, attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = get_peft_model(Qwen3ForCausalLM(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"), adapter_name="policy")
    for role in ("reference", "reward"):
        model.add_adapter(role, copy.deepcopy(model.peft_config["policy"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.policy." in name:
                parameter.normal_(0, 0.03)
        parameters = dict(model.named_parameters())
        for name, parameter in parameters.items():
            for role in ("reference", "reward"):
                if f".{role}." in name:
                    parameter.copy_(parameters[name.replace(f".{role}.", ".policy.")])
    heads = {"value": nn.Linear(32, 1), "reward": nn.Linear(32, 1)}
    backend = SharedQwen3(model, trainable_roles={"policy"}, heads=heads,
                         trainable_heads={"policy": ("value",)})
    return backend, model, heads


def sample(backend, *, generator=None, limit=5, prompts=((4, 8, 9), (5, 7))):
    generator = generator if generator is not None else torch.Generator().manual_seed(77)
    return sample_rollout(backend, "policy", prompts, eos_id=2, pad_id=0,
                         max_new_tokens=limit, generator=generator, device="cpu")


def response_bytes(rollout):
    # JSON distinguishes signed zero as well as every Python scalar value.
    return json.dumps([response.__dict__ for response in rollout.responses],
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class SamplingCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU candidate tests require CUDA to remain uninitialized")

    def test_real_tiny_qwen_matches_actions_statistics_and_rng_bitwise(self):
        for guard_mode in GUARD_MODES:
            for seed in (7, 77):
                with self.subTest(guard_mode=guard_mode, seed=seed):
                    old, _, _ = fixture()
                    new, _, _ = fixture()
                    new = make_sampling_backend(new, guard_mode=guard_mode)
                    expected = sample(old, generator=torch.Generator().manual_seed(seed))
                    actual = sample(new, generator=torch.Generator().manual_seed(seed))
                    self.assertEqual(response_bytes(actual), response_bytes(expected))
                    self.assertEqual(actual.rng_before, expected.rng_before)
                    self.assertEqual(actual.rng_after, expected.rng_after)
                    self.assertFalse(torch.cuda.is_initialized())

    def test_adoption_preserves_parameter_objects_weights_rng_and_optimizer_references(self):
        old, model, _ = fixture()
        parameters = tuple(old._all_parameters())
        snapshots = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.AdamW(old.parameters_for("policy", heads=("value",)), lr=1e-3)
        before_rng = torch.get_rng_state().clone()
        new = make_sampling_backend(old)
        self.assertIs(new._model, model)
        self.assertEqual(tuple(map(id, new._all_parameters())), tuple(map(id, parameters)))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                         set(map(id, new.parameters_for("policy", heads=("value",)))))
        for actual, expected in zip(new._all_parameters(), snapshots):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pad_eos_and_horizon_keep_exact_legacy_semantics(self):
        results = []
        for mode in (None, *GUARD_MODES):
            backend, _, _ = fixture()
            if mode is not None:
                backend = make_sampling_backend(backend, guard_mode=mode)
            with patch("torch.multinomial", side_effect=[torch.tensor([[0]]), torch.tensor([[2]]),
                                                        torch.tensor([[13]]), torch.tensor([[14]])]):
                result = sample(backend, limit=2, prompts=((4, 2, 9), (5, 7)))
            self.assertEqual(result.responses[0].action_ids, (0, 2))
            self.assertEqual(result.responses[0].termination, "eos")
            self.assertEqual(result.responses[1].action_ids, (13, 14))
            self.assertEqual(result.responses[1].termination, "max_new_tokens")
            results.append(response_bytes(result))
        self.assertEqual(results, [results[0]] * len(results))

    def test_nonfinite_logits_and_value_fail_before_random_draw(self):
        for mode in (None, *GUARD_MODES):
            for bad in ("logits", "value"):
                with self.subTest(mode=mode, bad=bad):
                    backend, model, heads = fixture()
                    if mode is not None:
                        backend = make_sampling_backend(backend, guard_mode=mode)
                    generator = torch.Generator().manual_seed(77)
                    before = generator.get_state().clone()
                    hook = None
                    if bad == "logits":
                        hook = model.get_base_model().lm_head.register_forward_hook(
                            lambda module, args, result: torch.full_like(result, float("nan")))
                    else:
                        with torch.no_grad():
                            heads["value"].bias.fill_(float("nan"))
                    try:
                        with self.assertRaises(FloatingPointError):
                            sample(backend, generator=generator, prompts=((4, 8),))
                    finally:
                        if hook is not None:
                            hook.remove()
                    self.assertTrue(torch.equal(before, generator.get_state()))
                    self.assertEqual(backend._phase, "idle")

    def test_immediate_eos_returns_one_scalar_and_no_extra_random_draw(self):
        for mode in GUARD_MODES:
            old, _, _ = fixture()
            backend = make_sampling_backend(old, guard_mode=mode)
            with patch("torch.multinomial", return_value=torch.tensor([[2]])) as draws:
                result = sample(backend, limit=5, prompts=((4, 8),))
            response = result.responses[0]
            self.assertEqual(response.action_ids, (2,))
            self.assertEqual(response.termination, "eos")
            self.assertEqual(len(response.old_logprobs), 1)
            self.assertEqual(len(response.old_values), 1)
            self.assertEqual(draws.call_count, 1)

    def test_non_scalar_value_fails_and_does_not_silently_flatten(self):
        states = []
        for mode in (None, *GUARD_MODES):
            backend, _, heads = fixture()
            if mode is not None:
                backend = make_sampling_backend(backend, guard_mode=mode)
            generator = torch.Generator().manual_seed(77)
            hook = heads["value"].register_forward_hook(lambda module, args, result: result.expand(1, 2))
            try:
                with self.assertRaises(RuntimeError):
                    sample(backend, generator=generator, prompts=((4, 8),))
            finally:
                hook.remove()
            states.append(generator.get_state())
        for state in states[1:]:
            self.assertTrue(torch.equal(states[0], state))

    def test_state_protection_is_retained_in_both_modes(self):
        for mode in GUARD_MODES:
            for mutation in ("mode", "adapter", "grad"):
                with self.subTest(mode=mode, mutation=mutation):
                    old, model, _ = fixture()
                    backend = make_sampling_backend(old, guard_mode=mode)
                    def corrupt(view):
                        if mutation == "mode":
                            model.get_base_model().model.embed_tokens.training = True
                        elif mutation == "adapter":
                            model.set_adapter("reference")
                        else:
                            backend.parameters_for("policy")[0].requires_grad_(True)
                        return view.sample_response(torch.tensor([4, 8]), eos_id=2,
                            max_new_tokens=2, generator=torch.Generator())
                    with self.assertRaises(RuntimeError):
                        backend.read("policy", corrupt)
                    self.assertEqual(backend._phase, "idle")

    def test_cached_guard_rejects_added_and_replaced_children_parameters_and_heads(self):
        for mutation in ("add_child", "replace_child", "replace_parameter", "replace_head"):
            with self.subTest(mutation=mutation):
                old, model, _ = fixture()
                backend = make_sampling_backend(old, guard_mode="cached_topology")
                def corrupt(view):
                    decoder = model.get_base_model().model
                    if mutation == "add_child":
                        decoder.add_module("unexpected_child", nn.Identity().eval())
                    elif mutation == "replace_child":
                        decoder.embed_tokens = copy.deepcopy(decoder.embed_tokens)
                    elif mutation == "replace_parameter":
                        decoder.embed_tokens.weight = nn.Parameter(
                            decoder.embed_tokens.weight.detach().clone(), requires_grad=False)
                    else:
                        backend._heads["value"] = copy.deepcopy(backend._heads["value"])
                    return view.sample_response(torch.tensor([4, 8]), eos_id=2,
                        max_new_tokens=2, generator=torch.Generator())
                with self.assertRaisesRegex(RuntimeError, "topology"):
                    backend.read("policy", corrupt)

    def test_cached_guard_detects_topology_mutation_during_forward(self):
        old, model, _ = fixture()
        backend = make_sampling_backend(old, guard_mode="cached_topology")
        decoder = model.get_base_model().model
        def mutate(module, args, result):
            decoder.add_module("during_forward", nn.Identity().eval())
            return result
        hook = decoder.register_forward_hook(mutate)
        try:
            with self.assertRaisesRegex(RuntimeError, "topology"):
                sample(backend, prompts=((4, 8),))
        finally:
            hook.remove()

    def test_expired_views_nested_reads_and_pending_gradient_adoption_are_rejected(self):
        old, _, _ = fixture()
        backend = make_sampling_backend(old, guard_mode="cached_topology")
        view = backend.read("policy", lambda current: current)
        with self.assertRaises(RuntimeError):
            view.sample_response(torch.tensor([4]), eos_id=2, max_new_tokens=1,
                                 generator=torch.Generator())
        with self.assertRaises(RuntimeError):
            backend.read("policy", lambda current: backend.read("policy", lambda other: None))
        old.parameters_for("policy")[0].grad = torch.zeros_like(old.parameters_for("policy")[0])
        with self.assertRaisesRegex(RuntimeError, "pending"):
            make_sampling_backend(old)

    def test_benchmark_is_equivalence_measurement_not_dispatch_or_speed_acceptance(self):
        backend, _, _ = fixture()
        state = torch.Generator().manual_seed(77).get_state()
        result = benchmark_sampling_equivalence(backend, ((4, 8), (5, 7)), role="policy",
            eos_id=2, pad_id=0, max_new_tokens=2, generator_state=state,
            device="cpu", warmups=0, repeats=2)
        self.assertTrue(result["all_exactly_equivalent"])
        self.assertEqual([row["order"] for row in result["trials"]],
                         [["legacy", "candidate"], ["candidate", "legacy"]])
        self.assertFalse(result["training_dispatched"])
        self.assertEqual(result["performance_acceptance"], "not_decided_by_this_function")
        self.assertTrue(torch.equal(state, torch.Generator().manual_seed(77).get_state()))
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(ValueError, "admission"):
            benchmark_sampling_equivalence(backend, ((4, 8),), role="policy", eos_id=2,
                pad_id=0, max_new_tokens=1, generator_state=state, device="cuda:0")


if __name__ == "__main__":
    unittest.main()
