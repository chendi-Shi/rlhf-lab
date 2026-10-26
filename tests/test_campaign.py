"""Supervisor control-plane tests: simulated children, clocks and resources only."""
import json
import os
from pathlib import Path

import pytest

from rlhf_lab import campaign as c


class Clock:
    def __init__(self):
        self.value = 1000.0

    def time(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    for directory in ("configs", "scripts", "rlhf_lab", "data/source", "models/base"):
        (tmp_path / directory).mkdir(parents=True)
    for name in ("local_cli.py", "run_campaign.py", "verify_local_run.py", "summary.py"):
        (tmp_path / "scripts" / name).write_text("# frozen test source\n")
    (tmp_path / "rlhf_lab/engine.py").write_text("# frozen test engine\n")
    (tmp_path / "data/source/data.jsonl").write_text('{"text":"test bytes, never decoded by runner"}\n')
    (tmp_path / "models/base/weights").write_bytes(b"model identity")
    config = {"output_dir": str(tmp_path / "runs/experiment/seed42"), "model_path": str(tmp_path / "models/base"),
              "data_dir": str(tmp_path / "data/source"), "device": "cuda"}
    config_path = tmp_path / "configs/run.json"
    c.atomic_json(config_path, config)
    monkeypatch.setattr(c, "load_local_config", lambda path: dict(c.read_json(path), config_path=str(path)))
    plan = {"schema_version": 1, "name": "experiment", "output_dir": "runs/experiment", "max_hours": 1,
            "poll_seconds": 1, "runs": [{"name": "seed42", "config": "configs/run.json"}],
            "summary_script": "scripts/summary.py"}
    plan_path = tmp_path / "configs/plan.json"
    c.atomic_json(plan_path, plan)
    clock = Clock()
    monkeypatch.setattr(c.time, "time", clock.time)
    monkeypatch.setattr(c.time, "monotonic", clock.time)
    monkeypatch.setattr(c.time, "sleep", clock.sleep)
    monkeypatch.setattr(c, "local_doctor", lambda config: {"blockers": [], "device": config["device"]})
    monkeypatch.setattr(c, "local_writers", lambda: [])
    monkeypatch.setattr(c, "recorded_process_alive", lambda record: False)
    monkeypatch.setattr(c, "process_record", lambda pid, command=None:
                        {"pid": pid, "created_at": clock.time(), "command": command or ["supervisor"], "workdir": str(tmp_path)})
    state = {"calls": [], "exit_codes": {}, "manifest_status": {}, "poll_hook": None,
             "root": tmp_path, "plan": plan, "plan_path": plan_path, "config": config,
             "config_path": config_path, "clock": clock}

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 300 + len(state["calls"])
            self.command, self.kwargs, self.polled = command, kwargs, False
            if "local-train" in command:
                self.stage = command[command.index("--stage") + 1]
            elif "local-evaluate" in command:
                self.stage = "evaluation"
            elif any(Path(arg).name == "verify_local_run.py" for arg in command):
                self.stage = "verify"
            else:
                self.stage = "summary"
            state["calls"].append(self)
            self.code = state["exit_codes"].get(self.stage, 0)
            if self.stage in c.STAGES or self.stage == "evaluation":
                folder = Path(config["output_dir"]) / self.stage
                c.atomic_json(folder / "run_manifest.json", {"stage": self.stage,
                              "status": state["manifest_status"].get(self.stage, "failed" if self.code else "completed"),
                              "completed_optimizer_steps": 10})
            if self.stage == "verify":
                json.dump({"status": "verified" if not self.code else "failed",
                           "stages": {stage: {"verified": True} for stage in c.STAGES},
                           "evaluation": {"verified": True}}, kwargs["stdout"])
                kwargs["stdout"].flush()

        def poll(self):
            if not self.polled:
                self.polled = True
                if state["poll_hook"]:
                    state["poll_hook"](self)
                return None
            return self.code

        def wait(self):
            return self.code

    monkeypatch.setattr(c.subprocess, "Popen", Child)
    return state


def execute(rig, resume=False):
    return c.run_campaign(rig["plan_path"], execute=True, resume=resume)


def test_plan_only_has_no_process_resource_or_output_side_effect(rig, monkeypatch):
    monkeypatch.setattr(c, "local_doctor", lambda config: pytest.fail("read-only plan queried hardware"))
    result = c.run_campaign(rig["plan_path"])
    assert result["status"] == "plan_only" and result["cuda_initialized"] is False
    assert not rig["calls"] and not (rig["root"] / "runs").exists()


def test_sequential_stages_evaluation_verifier_and_summary(rig):
    result = execute(rig)
    assert result["status"] == "completed"
    assert [child.stage for child in rig["calls"]] == [*c.STAGES, "evaluation", "verify", "summary"]
    assert result["runs"]["seed42"]["evaluation"]["verified"]
    assert result["current_child"] is None and result["elapsed_wall_seconds"] == 7
    assert all(attempt["process"]["created_at"] for attempt in result["attempts"])
    assert all(Path(attempt["stdout"]).is_file() and Path(attempt["stderr"]).is_file() for attempt in result["attempts"])
    assert all(child.kwargs["stdin"] == c.subprocess.DEVNULL for child in rig["calls"])
    if os.name == "nt":
        assert all(child.kwargs["creationflags"] == c.subprocess.CREATE_NO_WINDOW for child in rig["calls"])


def test_existing_stage_uses_explicit_cli_resume(rig):
    c.atomic_json(Path(rig["config"]["output_dir"]) / "sft/run_manifest.json", {"status": "completed"})
    # Permit pre-existing stage only by first registering the campaign as paused.
    Path(rig["config"]["output_dir"]).rename(rig["root"] / "seed42-prepared")
    (rig["root"] / "runs/experiment").rmdir()
    loaded = c.load_plan(rig["plan_path"])
    runner = c.Campaign(loaded)
    original = runner.wait_ready

    def ready(config):
        if not Path(config["output_dir"]).exists():
            (rig["root"] / "seed42-prepared").rename(Path(config["output_dir"]))
        original(config)

    runner.wait_ready = ready
    result = runner.execute()
    assert result["status"] == "completed"
    assert "--resume" in rig["calls"][0].command
    assert "--resume" not in rig["calls"][1].command


def test_resource_busy_waits_and_does_not_initialize_or_dispatch(rig, monkeypatch):
    seen = []

    def doctor(config):
        seen.append(len(rig["calls"]))
        return {"blockers": ["GPU busy"] if len(seen) <= 2 else []}

    monkeypatch.setattr(c, "local_doctor", doctor)
    result = execute(rig)
    assert result["status"] == "completed" and seen[:3] == [0, 0, 0]
    assert result["elapsed_wall_seconds"] == 9


def test_budget_expiry_during_child_lets_it_finish_then_blocks_next_stage(rig):
    rig["poll_hook"] = lambda child: rig["clock"].sleep(3600) if child.stage == "sft" else None
    result = execute(rig)
    assert result["status"] == "budget_exhausted"
    assert [child.stage for child in rig["calls"]] == ["sft"]
    assert result["attempts"][0]["exit_code"] == 0 and result["attempts"][0]["status"] == "completed"
    assert result["current_child"] is None


def test_stop_file_during_child_never_kills_child(rig):
    def stop(child):
        (rig["root"] / "runs/experiment/STOP").write_text("Pause after current stage")
    rig["poll_hook"] = stop
    result = execute(rig)
    assert result["status"] == "paused" and len(rig["calls"]) == 1
    assert result["runs"]["seed42"]["stages"]["sft"]["status"] == "completed"


def test_busy_writer_is_waited_for_until_budget_without_dispatch(rig, monkeypatch):
    monkeypatch.setattr(c, "local_writers", lambda: [{"pid": 123, "created_at": 999}])
    rig["plan"]["max_hours"] = 2 / 3600
    c.atomic_json(rig["plan_path"], rig["plan"])
    result = execute(rig)
    assert result["status"] == "budget_exhausted" and not rig["calls"]
    assert result["resource_check"]["active_writers"][0]["pid"] == 123


def test_child_failure_is_not_completion_and_has_logs(rig):
    rig["exit_codes"]["reward"] = 17
    result = execute(rig)
    assert result["status"] == "failed" and len(rig["calls"]) == 2
    assert "exited 17" in result["error"] and "stderr=" in result["error"]
    assert result["attempts"][-1]["exit_code"] == 17


def test_exit_zero_without_completed_manifest_fails(rig):
    rig["manifest_status"]["sft"] = "paused"
    result = execute(rig)
    assert result["status"] == "failed" and len(rig["calls"]) == 1
    assert "without a completed stage manifest" in result["error"]


def test_completed_evaluation_reused_only_through_verifier(rig, monkeypatch):
    assert execute(rig)["status"] == "completed"
    rig["calls"].clear()
    monkeypatch.setattr(c, "local_doctor", lambda config: pytest.fail("complete run should only verify"))
    result = execute(rig, resume=True)
    assert result["status"] == "completed"
    assert [child.stage for child in rig["calls"]] == ["verify", "summary"]


def test_incomplete_evaluation_not_overwritten(rig):
    assert execute(rig)["status"] == "completed"
    manifest = Path(rig["config"]["output_dir"]) / "evaluation/run_manifest.json"
    c.atomic_json(manifest, {"status": "failed"})
    before = manifest.read_bytes()
    rig["calls"].clear()
    result = execute(rig, resume=True)
    assert result["status"] == "failed" and not rig["calls"]
    assert manifest.read_bytes() == before and "will not overwrite" in result["error"]


def test_stop_with_existing_completed_evaluation_blocks_verification_and_summary(rig):
    assert execute(rig)["status"] == "completed"
    (rig["root"] / "runs/experiment/STOP").write_text("Keep paused")
    rig["calls"].clear()
    result = execute(rig, resume=True)
    assert result["status"] == "paused" and not rig["calls"]


def test_budget_expires_during_last_evaluation_blocks_verification_and_completion(rig):
    rig["poll_hook"] = lambda child: rig["clock"].sleep(3600) if child.stage == "evaluation" else None
    result = execute(rig)
    assert result["status"] == "budget_exhausted"
    assert [child.stage for child in rig["calls"]] == [*c.STAGES, "evaluation"]
    assert result["attempts"][-1]["status"] == "completed" and result["current_child"] is None


def test_stop_during_final_summary_does_not_mark_campaign_complete(rig):
    def stop(child):
        if child.stage == "summary":
            (rig["root"] / "runs/experiment/STOP").write_text("Pause even at final boundary")
    rig["poll_hook"] = stop
    result = execute(rig)
    assert result["status"] == "paused"
    assert result["attempts"][-1]["stage"] == "summary" and result["attempts"][-1]["status"] == "completed"


@pytest.mark.parametrize("target", ["configs/run.json", "rlhf_lab/engine.py", "data/source/data.jsonl", "models/base/weights"])
def test_changed_identity_refuses_resume_without_mutating_state(rig, target):
    assert execute(rig)["status"] == "completed"
    state_path = rig["root"] / "runs/experiment/campaign.json"
    before = state_path.read_bytes()
    with (rig["root"] / target).open("ab") as stream:
        stream.write(b"\n ")
    with pytest.raises(ValueError, match="identical plan/config/source/data/model/runtime"):
        execute(rig, resume=True)
    assert state_path.read_bytes() == before


def test_source_change_between_stages_prevents_new_child(rig):
    def edit(child):
        if child.stage == "sft":
            (rig["root"] / "rlhf_lab/engine.py").write_text("changed\n")
    rig["poll_hook"] = edit
    result = execute(rig)
    assert result["status"] == "failed" and len(rig["calls"]) == 1
    assert "identity changed" in result["error"]


def test_live_orphan_child_prevents_resume_without_state_change(rig, monkeypatch):
    result = execute(rig)
    path = rig["root"] / "runs/experiment/campaign.json"
    result["current_child"] = {"pid": 777, "created_at": 1000}
    c.atomic_json(path, result)
    before = path.read_bytes()
    monkeypatch.setattr(c, "recorded_process_alive", lambda record: bool(record))
    with pytest.raises(RuntimeError, match="recorded child is still alive"):
        execute(rig, resume=True)
    assert path.read_bytes() == before


def test_summary_failure_prevents_campaign_completion(rig):
    rig["exit_codes"]["summary"] = 9
    result = execute(rig)
    assert result["status"] == "failed" and result["runs"]["seed42"]["status"] == "completed"
    assert "summary exited 9" in result["error"]


def test_full_verifier_failure_prevents_summary(rig):
    rig["exit_codes"]["verify"] = 3
    result = execute(rig)
    assert result["status"] == "failed" and rig["calls"][-1].stage == "verify"


def test_failed_stage_only_retried_on_explicit_campaign_resume(rig):
    rig["exit_codes"]["reward"] = 5
    first = execute(rig)
    assert first["status"] == "failed" and len(rig["calls"]) == 2
    rig["exit_codes"].clear()
    rig["calls"].clear()
    second = execute(rig, resume=True)
    assert second["status"] == "completed"
    assert "--resume" in rig["calls"][0].command and "--resume" in rig["calls"][1].command
    assert "--resume" not in rig["calls"][2].command


def test_keyboard_interrupt_preserves_child_identity_for_safe_resume(rig):
    def interrupt(child):
        raise KeyboardInterrupt()
    rig["poll_hook"] = interrupt
    result = execute(rig)
    assert result["status"] == "interrupted" and len(rig["calls"]) == 1
    assert result["current_child"]["pid"] == rig["calls"][0].pid
    assert result["attempts"][0]["status"] == "running"


def test_project_os_lock_rejects_another_supervisor(rig):
    with c.campaign_lock():
        with pytest.raises(RuntimeError, match="holds the project OS lock"):
            with c.campaign_lock():
                pytest.fail("concurrent campaign lock acquired")


def test_existing_campaign_requires_explicit_resume(rig):
    assert execute(rig)["status"] == "completed"
    with pytest.raises(ValueError, match="explicit --resume"):
        execute(rig)


def test_duplicate_output_and_unsafe_path_are_rejected(rig):
    rig["plan"]["runs"].append({"name": "seed43", "config": "configs/run.json"})
    c.atomic_json(rig["plan_path"], rig["plan"])
    with pytest.raises(ValueError, match="must not overlap"):
        c.load_plan(rig["plan_path"])
    rig["plan"]["runs"].pop()
    rig["plan"]["output_dir"] = "../outside-project"
    c.atomic_json(rig["plan_path"], rig["plan"])
    with pytest.raises(ValueError, match="inside this project"):
        c.load_plan(rig["plan_path"])
