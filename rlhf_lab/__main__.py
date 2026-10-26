"""Command-line interface; training and generation default to a printed plan."""

import argparse
import json
from pathlib import Path
import sys

from .runtime import doctor, load_config, plan, resolve_path


def parser():
    root = argparse.ArgumentParser(description="Real small-model RLHF. No automatic GPU work.")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("local-doctor", "local-plan", "local-train", "local-evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/local-smoke.json")
        if name == "local-train":
            command.add_argument("--stage", choices=("sft", "reward", "dpo", "ppo", "all"), default="all")
            command.add_argument("--execute", action="store_true")
            command.add_argument("--resume", action="store_true")
        if name == "local-evaluate":
            command.add_argument("--execute", action="store_true")
    for name in ("doctor", "plan", "prepare-data", "train", "generate", "evaluate-reward"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/pilot.json")
        if name == "prepare-data":
            command.add_argument("--input", help="Local JSONL with a provenance sidecar; omit to stream real Hub data")
        if name == "train":
            command.add_argument("--stage", required=True, choices=("sft", "reward", "ppo"))
            command.add_argument("--execute", action="store_true", help="Actually train, subject to readiness checks")
        if name == "generate":
            command.add_argument("--model", required=True)
            command.add_argument("--label", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--limit", type=int)
            command.add_argument("--split", choices=("valid", "test"), default="valid")
            command.add_argument("--execute", action="store_true")
        if name == "evaluate-reward":
            command.add_argument("--model", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--limit", type=int)
            command.add_argument("--split", choices=("valid", "test"), default="valid")
            command.add_argument("--execute", action="store_true")
    validate = commands.add_parser("validate-data")
    validate.add_argument("--directory", default="data/processed/pilot")
    blind = commands.add_parser("blind")
    blind.add_argument("--left", required=True)
    blind.add_argument("--right", required=True)
    blind.add_argument("--output-dir", required=True)
    blind.add_argument("--seed", type=int, default=42)
    score = commands.add_parser("score")
    score.add_argument("--judgments", required=True)
    score.add_argument("--mapping", required=True)
    score.add_argument("--output", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    return root


def dispatch(args):
    if args.command.startswith("local-"):
        from .local_runtime import load_local_config, local_doctor, local_plan, STAGES
        config = load_local_config(args.config)
        if args.command == "local-doctor":
            return local_doctor(config)
        if args.command == "local-plan" or not args.execute:
            return local_plan(config, getattr(args, "stage", None))
        if args.command == "local-evaluate":
            from .local_evaluate import evaluate_local
            return evaluate_local(config)
        from .local_training import execute_local
        stages = STAGES if args.stage == "all" else (args.stage,)
        results = []
        for stage in stages:
            stage_resume = args.resume and (args.stage != "all" or (Path(config["output_dir"]) / stage / "run_manifest.json").exists())
            result = execute_local(config, stage, resume=stage_resume)
            results.append(result)
            if result.get("status") != "completed":
                break
        return {"stages": results}
    config = load_config(args.config) if hasattr(args, "config") else None
    if args.command == "doctor":
        return doctor(config)
    if args.command == "plan":
        return plan(config)
    if args.command == "prepare-data":
        from .data import prepare, prepare_hub
        kwargs = {key: config[key] for key in ("seed", "train_limit", "valid_limit", "test_limit")}
        if args.input:
            result = prepare(resolve_path(args.input), config["data_dir"], **kwargs)
        else:
            result = prepare_hub(config["data_dir"], dataset_name=config["dataset_id"],
                                 revision=config["dataset_revision"], **kwargs)
        return {"status": "prepared", "manifest": str(Path(config["data_dir"]) / "manifest.json"),
                "samples": result["samples"], "split_policy": result["split_policy"],
                "human_preferences": result["source"].get("human_preferences"),
                "synthetic": result["source"].get("synthetic")}
    if args.command == "validate-data":
        from .data import validate_dataset
        result = validate_dataset(resolve_path(args.directory))
        result["source"] = {key: result["source"].get(key) for key in
                            ("kind", "human_preferences", "synthetic", "dataset_id", "sha256")}
        return result
    if args.command == "train":
        if not args.execute:
            return plan(config, args.stage)
        from .training import execute
        return execute(config, args.stage)
    if args.command == "generate":
        config["eval_split"] = args.split
        if not args.execute:
            return {"status": "plan_only", "generation_started": False,
                    "model": args.model, "split": args.split, "output": str(resolve_path(args.output)),
                    "limit": args.limit or config["eval_limit"]}
        from .generate import generate
        model = args.model if args.model == config["model_id"] else str(resolve_path(args.model))
        return generate(config, model, args.label, resolve_path(args.output), args.limit)
    if args.command == "evaluate-reward":
        if not args.execute:
            return {"status": "plan_only", "evaluation_started": False,
                    "model": args.model, "split": args.split, "output": str(resolve_path(args.output)),
                    "limit": args.limit or config["eval_limit"]}
        from .reward_eval import score_reward
        return score_reward(config, str(resolve_path(args.model)), args.split, resolve_path(args.output), args.limit)
    if args.command == "blind":
        from .evaluate import export_blind_pairs
        output = resolve_path(args.output_dir)
        if output.exists():
            raise FileExistsError(f"Will not overwrite blind-review directory: {output}")
        return export_blind_pairs(resolve_path(args.left), resolve_path(args.right),
                                  output / "ballot.csv", output / "mapping.json", args.seed)
    if args.command == "score":
        from .evaluate import score_judgments
        return score_judgments(resolve_path(args.judgments), resolve_path(args.mapping), resolve_path(args.output))
    if args.command == "summarize":
        from .evaluate import summarize_generations
        return summarize_generations(resolve_path(args.input), resolve_path(args.output))
    raise ValueError("Unhandled command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
