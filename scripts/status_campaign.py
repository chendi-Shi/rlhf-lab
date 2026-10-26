"""Read campaign progress and real process identity without loading models."""
from pathlib import Path
import argparse
import json
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def process_status(record):
    if not record:
        return None
    result = dict(record)
    try:
        proc = psutil.Process(record["pid"])
        result.update(alive=True, actual_created_at=proc.create_time(), actual_command=proc.cmdline())
        result["identity_matches"] = (record.get("created_at") is not None
                                      and abs(proc.create_time() - record["created_at"]) < 0.01
                                      and proc.cmdline() == record.get("command"))
    except psutil.NoSuchProcess:
        result.update(alive=False, identity_matches=False)
    except psutil.AccessDenied:
        result.update(alive=None, identity_matches=None, error="AccessDenied")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    path = Path(args.plan)
    path = path if path.is_absolute() else ROOT / path
    plan = read(path)
    output = Path(plan["output_dir"])
    output = output if output.is_absolute() else ROOT / output
    manifest = output / "campaign.json"
    if not manifest.exists():
        print(json.dumps({"status": "not_started", "expected_runs": len(plan["runs"]), "plan": str(path)}))
        return
    state = read(manifest)
    current = state.get("current") or {}
    supervisor = process_status(state.get("supervisor"))
    child = process_status(state.get("current_child"))
    latest = None
    entry = next((item for item in plan["runs"] if item["name"] == current.get("run")), None)
    if entry and current.get("stage") in ("sft", "reward", "dpo", "ppo"):
        config_path = Path(entry["config"])
        config = read(config_path if config_path.is_absolute() else ROOT / config_path)
        run_dir = Path(config["output_dir"])
        run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
        metrics = run_dir / current["stage"] / "metrics.jsonl"
        if metrics.exists():
            lines = metrics.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-2:]):
                try:
                    latest = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue  # The current process may still be writing the last line.
    completed = sum(item.get("status") == "completed" for item in state.get("runs", {}).values())
    report = {"status": state.get("status"), "expected_runs": len(plan["runs"]),
              "completed_runs": completed, "current": current, "latest_metric": latest,
              "supervisor": supervisor, "child": child, "updated_at": state.get("updated_at_utc"),
              "elapsed_wall_seconds": state.get("elapsed_wall_seconds"), "error": state.get("error"),
              "resource_check": state.get("resource_check") if state.get("status") == "waiting_for_resources" else None,
              "scope": "Progress only; completion still requires saved full artifact audits and analysis verification."}
    if state.get("status") in ("running", "waiting_for_resources") and not (supervisor or {}).get("alive"):
        report["attention"] = "Supervisor is absent; inspect any surviving child and logs before resume."
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
