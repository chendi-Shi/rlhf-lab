"""Validation-only evaluation of real local base/SFT/DPO/PPO checkpoints.

Model imports and resource use occur only inside evaluate_local, after the
resource guard. Blank human ballots remain blank; reward scores are diagnostics
from the trained reward model, not independent judgments of response quality.
"""

from __future__ import annotations

import gc
import hashlib
import html
import json
from pathlib import Path
import time

from .data import normalize_prompt, read_jsonl
from .evaluate import export_blind_pairs, summarize_generations
from .local_runtime import assert_local_resources
from .runtime import sha256_file, utc_now, write_json
from .tokenization import encode_pair, encode_prompt, encode_response


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _select_validation_rows(rows, limit):
    """Select the first unique normalized prompts; never choose by model score."""
    if type(limit) is not int or limit < 1:
        raise ValueError("evaluation limit must be a positive integer")
    selected, seen_prompts, seen_ids = [], set(), set()
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("id", "prompt", "chosen", "rejected")):
            raise ValueError("Invalid validation preference record")
        if row["id"] in seen_ids:
            raise ValueError("Duplicate validation ID")
        seen_ids.add(row["id"])
        key = normalize_prompt(row["prompt"])
        if key not in seen_prompts and len(selected) < limit:
            selected.append(dict(row))
            seen_prompts.add(key)
    if not selected:
        raise ValueError("No validation preferences available")
    return selected


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _render_report(path, summary, generations):
    esc = lambda value: html.escape(str(value), quote=True)
    metric_rows = []
    for label, metrics in summary["policies"].items():
        metric_rows.append("<tr><th>" + esc(label) + "</th><td>" + f"{metrics['chosen_token_nll']:.5f}" +
                           "</td><td>" + str(metrics["nll_examples"]) + "</td><td>" +
                           str(metrics["nll_tokens"]) + "</td><td>" +
                           f"{metrics['mean_generated_tokens']:.1f}" + "</td><td>" +
                           f"{metrics['mean_reward_model_score']:.5f}" + "</td></tr>")
    examples = []
    labels = list(generations)
    for index, first in enumerate(generations[labels[0]]):
        panes = []
        for label in labels:
            row = generations[label][index]
            text = row["response"] if row["response"].strip() else "（模型返回空文本）"
            panes.append(f"<article><h4>{esc(label)}</h4><pre>{esc(text)}</pre><small>生成 {row['generated_tokens']} tokens；"
                         f"EOS结束：{row['stopped_with_eos']}；RM分数 {row['reward_model_score']:.4f}</small></article>")
        examples.append(f"<section><h3>验证样本 {index+1} · {esc(first['id'])}</h3>"
                        f"<details><summary>原始文章与实际模型输入</summary><h4>原始文章</h4><pre>{esc(first['prompt'])}</pre>"
                        f"<h4>模型实际看到的截断输入</h4><pre>{esc(first['model_input_text'])}</pre></details>"
                        f"<h4>人类偏好中的 chosen 文本</h4><pre>{esc(first['reference'])}</pre>"
                        f"<div class='grid'>{''.join(panes)}</div></section>")
    reward = summary["reward_model"]
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLHF 本地验证报告</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#182339;margin:0;padding:32px;max-width:1300px;margin:auto}h1,h2,h3{line-height:1.2}.note,section{background:white;padding:22px;border-radius:12px;margin:20px 0;border:1px solid #dfe5ef}.note{border-left:5px solid #c78628}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #dfe5ef}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{border:1px solid #dfe5ef;border-radius:8px;padding:16px}pre{font:14px/1.7 system-ui;white-space:pre-wrap;overflow-wrap:anywhere}small{color:#53627a}a{color:#195bc6}@media(max-width:760px){body{padding:16px}.grid{grid-template-columns:1fr}table{font-size:12px}}</style>
<h1>RLHF 本地验证报告</h1>"""
    document += f"<p>设备：{esc(summary['device'])} · 验证输入：{summary['generation_examples']} 条 · 解码：固定预算 greedy</p>"
    document += f"<p>实验范围：{esc(summary['experiment_scope'])}</p>"
    document += "<div class='note'><strong>这是执行与数值诊断，不是质量提升证明。</strong> 本轮训练与验证结果不足以证明 DPO 或 PPO 优于 SFT。"
    document += "本报告没有读取最终测试标签。RM 分数来自本项目训练的奖励模型，尤其对 PPO 不构成独立评估；所有人工盲评单均留空。"
    document += "NLL 是所选 chosen 文本的逐 token 负对数似然；更低 NLL 也不等同于更好的摘要。</div>"
    document += "<h2>同一批验证样本</h2><table><tr><th>模型</th><th>Chosen token NLL</th><th>NLL 样本</th><th>NLL tokens</th><th>平均生成 tokens</th><th>训练 RM 分数</th></tr>" + "".join(metric_rows) + "</table>"
    document += f"<p>奖励模型偏好判断：{reward['correct']}/{reward['pairs']}，平分 {reward['ties']}；"
    document += f"截断后相同而跳过的偏好对 {reward['skipped_identical_after_tokenization']}。这是 validation 子集诊断。</p>"
    document += "<p>人工盲评：<a href='blind-dpo-vs-sft.csv'>DPO vs SFT</a> · <a href='blind-ppo-vs-sft.csv'>PPO vs SFT</a>。"
    document += "仅填写 A、B 或 tie；空白表示尚未评价。<a href='summary.json'>机器可读结果</a></p>"
    document += "".join(examples) + "</html>"
    Path(path).write_text(document, encoding="utf-8")


def evaluate_local(config):
    """Evaluate completed local stages once; preserve any prior output directory."""
    if config.get("eval_split", "valid") != "valid":
        raise ValueError("local evaluation is restricted to validation; final test labels remain sealed")
    if config.get("software_fixture", False):
        raise ValueError("This evaluation entry point requires real human-preference validation data")
    hardware = assert_local_resources(config)  # Before torch import/model loading.
    from .local_training import (_configuration, _load_model, checkpoint_ready_local,
                                 pad_batch, reward_scores)
    c, _ = _configuration(config, "sft")
    root, data_root = Path(c["output_dir"]), Path(c["data_dir"])
    out = Path(config.get("eval_output_dir", root / "evaluation")).resolve()
    protected = [data_root, Path(c["model_path"])] + [root / name for name in ("sft", "reward", "dpo", "ppo")]
    if any(out == path.resolve() or out in path.resolve().parents or path.resolve() in out.parents for path in protected):
        raise ValueError("evaluation output must be separate from data/model/stage directories")
    if out.exists():
        raise FileExistsError(f"Evaluation output already exists: {out}; use a new eval_output_dir")
    manifest_path = data_root / "manifest.json"
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset.get("source", {}).get("human_preferences") is not True or dataset.get("source", {}).get("synthetic") is True:
        raise ValueError("Evaluation requires documented real human preferences")
    dataset_hash = sha256_file(manifest_path)
    valid_path = data_root / "preferences_valid.jsonl"
    if dataset.get("files", {}).get(valid_path.name, {}).get("sha256") != sha256_file(valid_path):
        raise ValueError("Validation file no longer matches the dataset manifest")
    base_manifest_path = Path(c["model_path"]) / "local-manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("model_id") != c.get("model_id") or base_manifest.get("revision") != c.get("model_revision"):
        raise ValueError("Base model manifest does not match the pinned model identity")
    if "model.safetensors" not in base_manifest.get("files", {}):
        raise ValueError("Base model manifest must include safetensors weights")
    for name, identity in base_manifest["files"].items():
        if Path(name).name != name or sha256_file(Path(c["model_path"]) / name) != identity.get("sha256"):
            raise ValueError(f"Base model artifact mismatch: {name}")
    stage_manifests = {stage: checkpoint_ready_local(root / stage, c, stage, dataset_hash)
                       for stage in ("sft", "reward", "dpo", "ppo")}
    raw = read_jsonl(valid_path)  # This is the only decoded dataset file here.
    generation_rows = _select_validation_rows(raw, config.get("eval_limit", 4))
    preference_rows = _select_validation_rows(raw, config.get("eval_preference_limit", 16))
    selection = [{"id": row["id"], "prompt": row["prompt"], "reference": row["chosen"]} for row in generation_rows]
    prompt_hash = _canonical_hash(selection)

    import torch
    from transformers import AutoTokenizer
    from .objectives import completion_logprobs, response_token_mask

    torch.set_num_threads(c["threads"])
    torch.manual_seed(c["seed"])
    if c["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no CPU fallback")
    if c["device"] == "cuda":
        torch.cuda.set_device(c.get("gpu_index", 0))
    tokenizer = AutoTokenizer.from_pretrained(c["model_path"], local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    decoding = {"do_sample": False, "num_beams": 1, "max_new_tokens": c["max_response_tokens"], "use_cache": True}
    manifest = {
        "schema_version": 1, "status": "running", "eval_split": "valid", "config": c,
        "model_id": c.get("model_id"), "model_revision": c.get("model_revision"),
        "dataset_manifest_sha256": dataset_hash, "validation_file_sha256": sha256_file(valid_path),
        "prompts_sha256": prompt_hash, "decoding": decoding,
        "inference_dtype": "float32" if c["device"] == "cpu" else "bfloat16",
        "source_sha256": {"local_evaluate.py": sha256_file(Path(__file__))},
        "stage_manifest_sha256": {stage: sha256_file(root / stage / "run_manifest.json") for stage in stage_manifests},
        "base_model_manifest_sha256": sha256_file(base_manifest_path),
        "hardware_before": hardware, "started_at_utc": utc_now(), "test_labels_decoded": False,
        "experiment_scope": c.get("experiment_scope", "local pilot"),
        "scope": f"validation diagnostics on {c['device']}; quality improvement is not established; RM scores are not independent human judgments",
    }
    write_json(out / "run_manifest.json", manifest)
    generations, policies = {}, {}
    try:
        for label in ("base", "sft", "dpo", "ppo"):
            source = Path(c["model_path"]) if label == "base" else root / label
            model = _load_model(source, c, frozen=True)
            model.config.pad_token_id = tokenizer.pad_token_id
            output_rows, nll_rows = [], []
            with torch.no_grad():
                for row in generation_rows:
                    prompt_ids = encode_prompt(tokenizer, row["prompt"], c)
                    chosen_ids = encode_response(tokenizer, row["chosen"], c)
                    batch = pad_batch([prompt_ids + chosen_ids], tokenizer.pad_token_id,
                                      [len(prompt_ids)], device=c["device"])
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False, return_dict=True).logits
                    token_logps = completion_logprobs(logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
                    token_mask = response_token_mask(batch["input_ids"], batch["response_mask"], batch["attention_mask"])
                    nll_rows.append({"id": row["id"], "negative_logprob_sum": float(-token_logps.sum().item()),
                                     "tokens": int(token_mask.sum().item()), "reference_has_eos": chosen_ids[-1] == tokenizer.eos_token_id})
                    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=c["device"])
                    full = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **decoding)
                    generated_ids = full[0, len(prompt_ids):].tolist()
                    eos = bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id)
                    output_rows.append({
                        "id": row["id"], "prompt": row["prompt"], "reference": row["chosen"],
                        "response": tokenizer.decode(generated_ids, skip_special_tokens=True),
                        "model_label": label, "model_path": str(source.resolve()), "seed": c["seed"],
                        "generated_token_ids": generated_ids, "generated_tokens": len(generated_ids),
                        "stopped_with_eos": eos, "hit_length_cap": len(generated_ids) >= c["max_response_tokens"] and not eos,
                        "model_input_token_ids": prompt_ids, "model_input_text": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                    })
            total_tokens = sum(row["tokens"] for row in nll_rows)
            if total_tokens < 1:
                raise ValueError("NLL evaluation contains no completion tokens")
            policies[label] = {"chosen_token_nll": sum(row["negative_logprob_sum"] for row in nll_rows) / total_tokens,
                               "nll_examples": len(nll_rows), "nll_tokens": total_tokens, "nll_rows": nll_rows,
                               "mean_generated_tokens": sum(row["generated_tokens"] for row in output_rows) / len(output_rows)}
            generations[label] = output_rows
            del model, logits, token_logps, batch, full, inputs
            gc.collect()

        reward = _load_model(root / "reward", c, reward=True, frozen=True)
        pair_results, skipped = [], 0
        with torch.no_grad():
            for row in preference_rows:
                encoded = encode_pair(tokenizer, row, c)
                if encoded is None:
                    skipped += 1
                    continue
                batch = pad_batch([encoded["chosen_ids"], encoded["rejected_ids"]], tokenizer.pad_token_id, device=c["device"])
                scores = reward_scores(reward, batch).tolist()
                pair_results.append({"id": row["id"], "chosen_reward": scores[0], "rejected_reward": scores[1],
                                     "correct": scores[0] > scores[1], "tie": scores[0] == scores[1]})
            for label, rows in generations.items():
                for row in rows:
                    sequence = row["model_input_token_ids"] + row["generated_token_ids"]
                    batch = pad_batch([sequence], tokenizer.pad_token_id, device=c["device"])
                    row["reward_model_score"] = float(reward_scores(reward, batch).item())
                policies[label]["mean_reward_model_score"] = sum(row["reward_model_score"] for row in rows) / len(rows)
        del reward
        gc.collect()
        if not pair_results:
            raise ValueError("No distinct tokenized validation preference pairs")
        for label, rows in generations.items():
            destination = out / f"{label}.jsonl"
            _write_jsonl(destination, rows)
            sidecar = {key: manifest[key] for key in ("config", "model_id", "model_revision", "eval_split",
                        "prompts_sha256", "dataset_manifest_sha256", "decoding", "inference_dtype")}
            sidecar.update(status="completed", model_label=label, model_path=rows[0]["model_path"],
                           output_sha256=sha256_file(destination), examples=len(rows), test_labels_decoded=False)
            write_json(Path(str(destination) + ".manifest.json"), sidecar)
            policies[label]["text_diagnostics"] = summarize_generations(destination)
        ballots = {}
        for label in ("dpo", "ppo"):
            ballots[label] = export_blind_pairs(out / f"{label}.jsonl", out / "sft.jsonl",
                                                out / f"blind-{label}-vs-sft.csv", out / f"private-{label}-vs-sft.json", c["seed"])
        correct, ties = sum(row["correct"] for row in pair_results), sum(row["tie"] for row in pair_results)
        summary = {"schema_version": 1, "status": "completed", "split": "valid", "device": c["device"],
                   "generation_examples": len(generation_rows), "policies": policies,
                   "reward_model": {"pairs": len(pair_results), "correct": correct, "ties": ties,
                                    "accuracy": correct / len(pair_results), "skipped_identical_after_tokenization": skipped,
                                    "rows": pair_results, "scope": "trained reward model; not an independent judge"},
                   "blind_ballots": ballots, "human_judgments": 0, "human_win_rate": None,
                   "test_labels_decoded": False, "elapsed_seconds": time.perf_counter() - start,
                   "experiment_scope": manifest["experiment_scope"], "scope": manifest["scope"]}
        write_json(out / "summary.json", summary)
        _render_report(out / "report.html", summary, generations)
        manifest.update(status="completed", finished_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - start,
                        artifacts={path.name: sha256_file(path) for path in sorted(out.iterdir())
                                   if path.is_file() and path.name != "run_manifest.json"})
        write_json(out / "run_manifest.json", manifest)
        return {"status": "completed", "output": str(out), "report": str(out / "report.html"),
                "summary": str(out / "summary.json"), "generation_examples": len(generation_rows),
                "reward_pairs": len(pair_results), "test_labels_decoded": False, "human_judgments": 0}
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=utc_now())
        write_json(out / "run_manifest.json", manifest)
        raise
