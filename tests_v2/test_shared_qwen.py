"""Tiny random CPU Qwen3/PEFT fixtures; never pretrained-model or CUDA evidence."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

from alignment_lab.shared_qwen import SharedQwen3


def fixture():
    torch.manual_seed(131)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=128, pad_token_id=0,
                         eos_token_id=2, attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    base = Qwen3ForCausalLM(config)
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora, adapter_name="policy")
    model.add_adapter("reference", copy.deepcopy(lora))
    model.add_adapter("reward", copy.deepcopy(lora))
    # Nonzero B makes gradient checks exercise both LoRA factors, not just B.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B.policy." in name:
                param.normal_(0, 0.03)
        params = dict(model.named_parameters())
        for name, param in params.items():
            for role in ("reference", "reward"):
                if f".{role}." in name:
                    param.copy_(params[name.replace(f".{role}.", ".policy.")])
    heads = {"value": nn.Linear(32, 1), "reward": nn.Linear(32, 1)}
    backend = SharedQwen3(model, trainable_roles={"policy"}, heads=heads,
                         trainable_heads={"policy": ("value",)})
    ids = torch.tensor([[4, 8, 9, 11, 2, 0, 0], [5, 7, 6, 10, 13, 14, 2]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    labels = ids.clone()
    labels[:, :3] = -100
    labels[~attention.bool()] = -100
    return backend, model, heads, ids, attention, labels


class SharedQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def test_real_qwen_chunked_sft_gradients_match_dense_and_checkpoint_recomputes(self):
        backend, model, _, ids, attention, labels = fixture()
        dense = copy.deepcopy(model)
        dense.gradient_checkpointing_disable()
        dense.set_adapter("policy")
        dense.requires_grad_(False)
        for name, param in dense.named_parameters():
            if ".policy." in name:
                param.requires_grad_(True)
        dense.train()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        observed_rows, layer_calls = [], []
        handle = model.get_base_model().lm_head.register_forward_pre_hook(
            lambda module, args: observed_rows.append(args[0].shape[0]))
        hooks = [layer.register_forward_pre_hook(lambda module, args: layer_calls.append(True))
                 for layer in model.get_base_model().model.layers]
        try:
            actual = backend.backward("policy", optimizer, lambda view:
                -view.sequence_scores(ids, attention, labels, chunk_tokens=2).sums.sum() / labels.ne(-100).sum())
        finally:
            handle.remove()
            for hook in hooks:
                hook.remove()
        full_logits = dense(input_ids=ids, attention_mask=attention, use_cache=False).logits
        expected = F.cross_entropy(full_logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
        expected.backward()
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
        other_params = dict(dense.named_parameters())
        grads = []
        for name, param in model.named_parameters():
            if ".policy." in name:
                self.assertIsNotNone(param.grad)
                torch.testing.assert_close(param.grad, other_params[name].grad, rtol=3e-4, atol=1e-7)
                grads.append(param.grad.abs().sum())
            else:
                self.assertIsNone(param.grad)
        self.assertGreater(sum(grads).item(), 0)
        self.assertGreaterEqual(len(layer_calls), 4)
        self.assertTrue(observed_rows and max(observed_rows) <= 2)
        self.assertTrue(model.training)
        self.assertTrue(model.get_base_model().model.is_gradient_checkpointing)

    def test_optimizer_update_changes_only_policy_and_reference_output_stays_fixed(self):
        backend, model, _, ids, attention, labels = fixture()
        reference_before = backend.read("reference", lambda view: view.sequence_scores(ids, attention, labels).sums.clone())
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if ".policy." not in name}
        policy_before = [p.detach().clone() for p in backend.parameters_for("policy")]
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        backend.backward("policy", optimizer, lambda view: -view.sequence_scores(ids, attention, labels).sums.mean())
        with self.assertRaisesRegex(RuntimeError, "Clear"):
            backend.read("reference", lambda view: None)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_after = backend.read("reference", lambda view: view.sequence_scores(ids, attention, labels).sums)
        torch.testing.assert_close(reference_before, reference_after, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(policy_before, backend.parameters_for("policy"))))
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(p, frozen[name]))

    def test_nested_adapter_switch_rejected_and_failed_graph_cannot_be_reused(self):
        backend, _, _, ids, attention, labels = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        def unsafe(view):
            loss = -view.sequence_scores(ids, attention, labels).sums.mean()
            backend.read("reference", lambda other: None)
            return loss
        with self.assertRaisesRegex(RuntimeError, "Cannot switch"):
            backend.backward("policy", optimizer, unsafe)
        with self.assertRaisesRegex(RuntimeError, "Cannot switch"):
            backend.read("reference", lambda view: None)

    def test_optimizer_cannot_include_frozen_reference_or_omit_value_head(self):
        backend, model, heads, _, _, _ = fixture()
        wrong = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "exactly"):
            backend.backward("policy", wrong, lambda view: torch.tensor(0.0, requires_grad=True))
        missing = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        with self.assertRaisesRegex(ValueError, "exactly"):
            backend.backward("policy", missing, lambda view: heads["value"].weight.sum(), heads=("value",))
        with self.assertRaisesRegex(ValueError, "Frozen"):
            backend.parameters_for("reference")
        with self.assertRaisesRegex(ValueError, "not trainable"):
            backend.parameters_for("policy", heads=("reward",))

    def test_reward_and_value_head_alias_is_rejected(self):
        _, model, heads, _, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            SharedQwen3(model, trainable_roles={"policy"},
                        heads={"value": heads["reward"], "reward": heads["reward"]},
                        trainable_heads={"policy": ("value",)})

    def test_direct_external_adapter_switch_is_detected_before_checkpoint_backward(self):
        backend, model, _, ids, attention, labels = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        def unsafe(view):
            loss = -view.sequence_scores(ids, attention, labels).sums.mean()
            model.set_adapter("reference")
            return loss
        with self.assertRaisesRegex(RuntimeError, "Adapter changed"):
            backend.backward("policy", optimizer, unsafe)

    def test_direct_backbone_eval_is_detected_before_backward(self):
        backend, model, _, ids, attention, labels = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        def unsafe(view):
            loss = -view.sequence_scores(ids, attention, labels).sums.mean()
            model.get_base_model().model.eval()
            return loss
        with self.assertRaisesRegex(RuntimeError, "Submodule mode"):
            backend.backward("policy", optimizer, unsafe)

    def test_disconnected_leaf_loss_is_rejected_even_with_accumulated_gradients(self):
        backend, _, _, ids, attention, labels = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy"), lr=1e-3)
        backend.backward("policy", optimizer, lambda view: -view.sequence_scores(ids, attention, labels).sums.mean())
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            backend.backward("policy", optimizer, lambda view: torch.tensor(1.0, requires_grad=True))

    def test_reward_uses_final_answer_state_and_values_use_pre_action_states(self):
        backend, _, heads, ids, attention, labels = fixture()
        # One EOS-only completion verifies reward and action-state indexing differ.
        eos_labels = labels.clone()
        eos_labels[0] = -100
        eos_labels[0, 4] = 2
        def check(view):
            hidden = view.hidden(ids, attention)
            reward = view.rewards(ids, attention, eos_labels)
            expected_reward = heads["reward"](hidden[torch.arange(2), torch.tensor([4, 6])]).squeeze(-1)
            torch.testing.assert_close(reward, expected_reward)
            scores, values = view.token_scores_and_values(ids, attention, eos_labels, chunk_tokens=2)
            expected_values = heads["value"](hidden[:, :-1]).squeeze(-1).masked_fill(~scores.mask, 0)
            torch.testing.assert_close(values, expected_values)
            self.assertEqual(scores.mask[0].tolist(), [False, False, False, True, False, False])
            self.assertEqual(values[0, 4:].abs().sum().item(), 0)
        backend.read("policy", check)

    def test_ppo_value_head_and_policy_get_gradients_but_reward_head_does_not(self):
        backend, _, heads, ids, attention, labels = fixture()
        optimizer = torch.optim.AdamW(backend.parameters_for("policy", heads=("value",)), lr=1e-3)
        def loss(view):
            scores, values = view.token_scores_and_values(ids, attention, labels)
            return -scores.logprobs[scores.mask].mean() + (values[scores.mask] - 1.0).square().mean()
        backend.backward("policy", optimizer, loss, heads=("value",))
        self.assertGreater(heads["value"].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(heads["reward"].weight.grad)

    def test_scoring_rejects_retokenized_labels_and_mask_holes(self):
        for corruption in ("target", "hole", "pad"):
            backend, _, _, ids, attention, labels = fixture()
            if corruption == "target":
                labels[0, 3] = 15
            elif corruption == "hole":
                labels[1, 4] = -100
            else:
                labels[0, 5] = 0
            with self.assertRaises(ValueError):
                backend.read("policy", lambda view: view.sequence_scores(ids, attention, labels))

    def test_model_view_expires_and_cpu_run_never_initializes_cuda(self):
        backend, _, _, ids, attention, _ = fixture()
        view = backend.read("reference", lambda view: view)
        with self.assertRaisesRegex(RuntimeError, "owning callback"):
            view.hidden(ids, attention)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
