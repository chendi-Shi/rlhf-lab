"""Tiny randomly initialized GPT2 SOFTWARE fixtures, never model-quality evidence."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def fixture_config(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from rlhf_lab import data
    torch.set_num_threads(1)
    vocabulary = {word: index for index, word in enumerate(
        ["<unk>", "<eos>", "article", "good", "helpful", "bad", "wrong", "summary", "TL", ";", "DR", ":"]
        + [f"topic{i}" for i in range(20)])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")
    assets = root/"assets"
    assets.mkdir()
    fast.save_pretrained(assets)
    torch.manual_seed(100)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=32, n_embd=8,
                                      n_layer=1, n_head=2, eos_token_id=1, pad_token_id=1,
                                      resid_pdrop=0., attn_pdrop=0., embd_pdrop=0.))
    model.save_pretrained(assets, safe_serialization=True)
    source = root/"synthetic.jsonl"
    source.write_text("".join(json.dumps({"prompt": f"article topic{i}",
                                        "chosen": f"good helpful summary topic{i}",
                                        "rejected": f"bad wrong summary topic{i}"})+"\n" for i in range(20)), encoding="utf-8")
    Path(str(source)+".provenance.json").write_text(json.dumps({"synthetic": True, "human_preferences": False}), encoding="utf-8")
    data.prepare(source, root/"data", train_limit=16, valid_limit=2, test_limit=2)
    return {"model_path": str(assets), "data_dir": str(root/"data"), "output_dir": str(root/"runs"),
            "device": "cpu", "software_fixture": True, "seed": 7, "threads": 1,
            "max_prompt_tokens": 8, "max_response_tokens": 6, "completion_eos_policy": "natural",
            "sft_max_steps": 2, "reward_max_steps": 2, "dpo_max_steps": 2,
            "sft_learning_rate": .001, "reward_learning_rate": .001, "dpo_learning_rate": .001,
            "ppo_learning_rate": .001, "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 1, "ppo_updates": 1, "ppo_rollout_batch_size": 4,
            "ppo_response_length": 3, "ppo_epochs": 2, "ppo_minibatch_size": 2,
            "gradient_checkpointing": False, "checkpoint_every": 1,
            "min_free_ram_gb": .01, "eval_limit": 2, "eval_preference_limit": 2}


def _assert_nested_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_all_stages_true_updates_and_ppo_resume_retains_old_policy(tmp_path, monkeypatch):
    import torch
    engine = importlib.import_module("rlhf_lab.local_training")
    monkeypatch.setattr("rlhf_lab.local_runtime.assert_local_resources", lambda c: {"device": "cpu", "software_fixture": True})
    config = fixture_config(tmp_path)
    manifests = {}
    for stage in ("sft", "reward", "dpo"):
        manifests[stage] = engine.execute_local(config, stage)
        assert manifests[stage]["status"] == "completed"
        assert manifests[stage]["weights_changed"]
        assert manifests[stage]["completed_optimizer_steps"] == 2
        assert manifests[stage]["software_fixture"] is True
        assert manifests[stage]["test_labels_decoded"] is False
    sft_weights = engine.sha256_file(Path(config["output_dir"])/"sft"/"model.safetensors")
    paused = engine.execute_local(dict(config, max_updates_this_run=1), "ppo")
    assert paused["status"] == "paused"
    assert paused["completed_rollout_batches"] == 0
    state_path = Path(config["output_dir"])/"ppo"/"latest.pt"
    partial = torch.load(state_path, map_location="cpu", weights_only=True)
    assert partial["pending"]["cursor"] == 1
    old = partial["pending"]["tensors"]["old_logprobs"].clone()
    actual_rollout = engine._rollout

    def no_regeneration(*a, **k):
        raise AssertionError("Pending on-policy rollouts must not be regenerated on resume")

    monkeypatch.setattr(engine, "_rollout", no_regeneration)
    completed = engine.execute_local(config, "ppo", resume=True)
    assert completed["status"] == "completed"
    assert completed["completed_rollout_batches"] == 1
    assert completed["completed_optimizer_steps"] == 4
    assert completed["weights_changed"] and completed["value_head_weights_changed"]
    final = torch.load(state_path, map_location="cpu", weights_only=True)
    assert final["pending"] is None
    assert all("clipfrac" in entry and "value_explained_variance" in entry for entry in final["history"])
    assert engine.sha256_file(Path(config["output_dir"])/"sft"/"model.safetensors") == sft_weights
    assert not torch.equal(old, torch.zeros_like(old))
    monkeypatch.setattr(engine, "_rollout", actual_rollout)
    # Fresh branch starts from the same SFT/RM via an independent output root.
    import shutil
    second_root = tmp_path/"continuous"
    second_root.mkdir()
    for stage in ("sft", "reward"):
        shutil.copytree(Path(config["output_dir"])/stage, second_root/stage)
    continuous_config = dict(config, output_dir=str(second_root))
    continuous = engine.execute_local(continuous_config, "ppo")
    full = torch.load(second_root/"ppo"/"latest.pt", map_location="cpu", weights_only=True)
    for key in ("model", "value_head", "optimizer", "rng"):
        _assert_nested_equal(final[key], full[key])
    assert continuous["weight_fingerprint_after"] == completed["weight_fingerprint_after"]
    with pytest.raises(ValueError, match="identical"):
        engine.execute_local(dict(config, ppo_learning_rate=.01), "ppo", resume=True)


def test_reward_pooling_uses_attention_even_when_eos_equals_padding():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from rlhf_lab.local_training import pad_batch, reward_scores

    class Transformer(nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

    model = nn.Module()
    model.transformer = Transformer()
    model.score = nn.Identity()
    batch = pad_batch([[8, 4, 1], [5, 1]], pad_id=1)
    torch.testing.assert_close(reward_scores(model, batch), torch.tensor([1., 1.]))
    batch = pad_batch([[8, 4, 1], [5, 7]], pad_id=1)
    torch.testing.assert_close(reward_scores(model, batch), torch.tensor([1., 7.]))


def test_synthetic_data_cannot_masquerade_as_real_training(tmp_path, monkeypatch):
    from rlhf_lab.local_training import execute_local
    monkeypatch.setattr("rlhf_lab.local_runtime.assert_local_resources", lambda c: {"device": "cpu"})
    config = fixture_config(tmp_path)
    with pytest.raises(ValueError, match="human-preference provenance"):
        execute_local(dict(config, software_fixture=False), "sft")
    assert not (Path(config["output_dir"])/"sft").exists()
