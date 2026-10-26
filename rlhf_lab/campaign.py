"""Sequential local experiment supervision. This module never imports torch.

The wall-clock budget gates dispatch, not an in-flight optimizer. STOP is a
cooperative request to finish the current child and dispatch no further work.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import psutil

from .local_runtime import STAGES, load_local_config, local_doctor
from .runtime import ROOT, sha256_file, utc_now


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project_path(value):
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError(f"Campaign paths must be inside this project: {path}")
    return path


def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def load_plan(path):
    """Resolve all paths against the project root, independently of caller cwd."""
    path = project_path(path)
    plan = read_json(path)
    if plan.get("schema_version") != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", plan.get("name", "")):
        raise ValueError("Plan requires schema_version=1 and a filesystem-safe name")
    hours = plan.get("max_hours")
    if type(hours) not in (float, int) or not math.isfinite(hours) or hours <= 0:
        raise ValueError("max_hours must be finite and positive")
    poll = plan.get("poll_seconds", 30)
    if type(poll) not in (float, int) or not math.isfinite(poll) or not 1 <= poll <= 60:
        raise ValueError("poll_seconds must be between 1 and 60")
    output = project_path(plan["output_dir"])
    if not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise ValueError("Plan requires a nonempty runs list")
    runs, names, outputs, protected = [], set(), [], [path, ROOT / "rlhf_lab", ROOT / "scripts", ROOT / "configs"]
    for item in plan["runs"]:
        name = item.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Run names must be unique and filesystem-safe")
        config_path = project_path(item["config"])
        config = load_local_config(config_path)
        run_output = project_path(config["output_dir"])
        evaluation = project_path(config.get("eval_output_dir", run_output / "evaluation"))
        if config.get("max_updates_this_run") is not None or config.get("software_fixture", False):
            raise ValueError("Campaign requires full budgets and real data, not fixtures/partial-step overrides")
        if any(overlaps(run_output, previous) or overlaps(evaluation, previous) for previous in outputs):
            raise ValueError("Run output/evaluation directories must not overlap other runs")
        if (output == run_output or run_output in output.parents or overlaps(output / "analysis", run_output)
                or overlaps(output / "analysis", evaluation)):
            raise ValueError("Run output must not contain campaign metadata or analysis")
        if evaluation == run_output or evaluation in run_output.parents:
            raise ValueError("Evaluation must not contain the training output")
        if any(overlaps(evaluation, run_output / stage) for stage in STAGES):
            raise ValueError("Evaluation must not overlap training stage directories")
        names.add(name)
        outputs.extend([run_output, evaluation])
        protected.extend([config_path, Path(config["model_path"]), Path(config["data_dir"])])
        runs.append({"name": name, "config_path": str(config_path), "config": config,
                     "output_dir": str(run_output), "evaluation_dir": str(evaluation)})
    for destination in [output] + outputs:
        if any(overlaps(destination, source.resolve()) for source in protected):
            raise ValueError(f"Campaign outputs must be separate from source/config/data/model paths: {destination}")
    summary = project_path(plan["summary_script"]) if plan.get("summary_script") else None
    if summary and (not summary.is_file() or summary.suffix != ".py"):
        raise ValueError("summary_script must be an existing project Python script")
    return {"path": str(path), "plan": plan, "output_dir": str(output), "runs": runs,
            "poll_seconds": float(poll), "max_seconds": float(hours) * 3600,
            "summary_script": str(summary) if summary else None}


def inventory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(directory).parts)
            and ".tmp-" not in path.name}


def freeze_identity(loaded):
    """Hash actual data/model bytes as well as plans, configs, and execution code."""
    sources = list((ROOT / "rlhf_lab").glob("*.py"))
    sources.extend(ROOT / "scripts" / name for name in ("local_cli.py", "run_campaign.py", "verify_local_run.py"))
    if loaded["summary_script"]:
        sources.append(Path(loaded["summary_script"]))
    directories = {run["config"][key] for run in loaded["runs"] for key in ("model_path", "data_dir")}
    return {"plan_path": loaded["path"], "plan_sha256": sha256_file(loaded["path"]),
            "config_sha256": {run["config_path"]: sha256_file(run["config_path"]) for run in loaded["runs"]},
            "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
                              for path in sorted(set(sources))},
            "input_sha256": {path: inventory(path) for path in sorted(directories)},
            "python": {"executable": str(Path(sys.executable).resolve()), "version": sys.version},
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "numpy", "safetensors", "psutil")}}


def process_record(pid, command=None):
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "created_at": process.create_time(), "command": process.cmdline(),
                "workdir": process.cwd()}
    except psutil.NoSuchProcess:
        return {"pid": pid, "created_at": None, "command": command,
                "exited_before_inventory": True}


def recorded_process_alive(record):
    if not record or record.get("pid") is None:
        return False
    try:
        process = psutil.Process(record["pid"])
        if record.get("created_at") is None:
            return True  # Cannot safely dismiss an unverified PID.
        return abs(process.create_time() - record["created_at"]) < 0.01
    except psutil.NoSuchProcess:
        return False


def local_writers():
    """Find direct CLI writers too; the OS campaign lock covers supervisors only."""
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            command = process.info["cmdline"] or []
            if process.pid == os.getpid() or not any(part in ("local-train", "local-evaluate") for part in command):
                continue
            if "--execute" not in command or not (any(Path(part).name == "local_cli.py" for part in command)
                                                 or "rlhf_lab" in command):
                continue
            cwd = Path(process.cwd()).resolve()
            script_paths = [(cwd / part).resolve() for part in command if Path(part).name == "local_cli.py"]
            if (cwd != ROOT.resolve() and not any(path.is_relative_to(ROOT.resolve()) for path in script_paths)
                    and not any(str(ROOT).lower() in part.lower() for part in command)):
                continue
            found.append({"pid": process.pid, "created_at": process.info["create_time"], "command": command})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            found.append({"pid": process.pid, "inspection_error": "AccessDenied; cannot exclude an active local writer"})
    return found


@contextmanager
def campaign_lock():
    """An OS lock releases on process exit; a leftover filename is not a lock."""
    path = ROOT / "reports" / ".campaign.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another campaign supervisor holds the project OS lock") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class DispatchStopped(Exception):
    def __init__(self, status):
        self.status = status


class Campaign:
    def __init__(self, loaded):
        self.loaded = loaded
        self.output = Path(loaded["output_dir"])
        self.path = self.output / "campaign.json"
        self.state = None

    def save(self):
        self.state["updated_at_utc"] = utc_now()
        self.state["elapsed_wall_seconds"] = max(0, time.time() - self.state["started_at_epoch"])
        atomic_json(self.path, self.state)

    def assert_identity(self):
        if freeze_identity(self.loaded) != self.state["identity"]:
            raise ValueError("Campaign plan/config/source/data/model/runtime identity changed; use a new output and identity")

    def stop_reason(self):
        if (self.output / "STOP").exists():
            return "paused"
        if time.time() - self.state["started_at_epoch"] >= self.loaded["max_seconds"]:
            return "budget_exhausted"
        return None

    def wait_ready(self, config):
        while True:
            reason = self.stop_reason()
            if reason:
                raise DispatchStopped(reason)
            info = local_doctor(config)
            writers = local_writers()
            self.state["resource_check"] = {"checked_at_utc": utc_now(), "doctor": info, "active_writers": writers}
            if not info["blockers"] and not writers:
                self.state["status"] = "running"
                self.save()
                self.assert_identity()
                reason = self.stop_reason()
                if reason:
                    raise DispatchStopped(reason)
                return
            self.state["status"] = "waiting_for_resources"
            self.save()
            time.sleep(min(self.loaded["poll_seconds"], max(0.1, self.loaded["max_seconds"] -
                                                           (time.time() - self.state["started_at_epoch"]))))

    def child(self, name, stage, arguments):
        """Wait naturally: neither STOP nor budget expiry terminates a child."""
        reason = self.stop_reason()
        if reason:
            raise DispatchStopped(reason)
        sequence = len(self.state["attempts"]) + 1
        stem = self.output / "logs" / f"{sequence:04d}-{name}-{stage}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stdout, stderr = Path(str(stem) + ".stdout.log"), Path(str(stem) + ".stderr.log")
        command = [sys.executable, "-u", *map(str, arguments)]
        attempt = {"run": name, "stage": stage, "status": "starting", "command": command,
                   "workdir": str(ROOT), "started_at_utc": utc_now(),
                   "stdout": str(stdout), "stderr": str(stderr)}
        self.state["attempts"].append(attempt)
        self.state["current"] = {"run": name, "stage": stage}
        self.save()
        started = time.monotonic()
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record = process_record(process.pid, command)
            attempt.update(status="running", process=record)
            self.state["current_child"] = record
            self.save()
            while process.poll() is None:
                reason = self.stop_reason()
                if reason:
                    self.state["stop_after_current_child"] = reason
                self.save()
                time.sleep(self.loaded["poll_seconds"])
            code = process.wait()
        attempt.update(exit_code=code, status="exited", finished_at_utc=utc_now(),
                       elapsed_wall_seconds=time.monotonic() - started)
        self.state["current_child"] = None
        self.save()
        if code:
            attempt["status"] = "failed"
            raise RuntimeError(f"{name}/{stage} exited {code}; stdout={stdout}; stderr={stderr}")
        return attempt

    def verify(self, run):
        self.assert_identity()
        writers = local_writers()
        if writers:
            raise RuntimeError(f"Cannot verify while local writers are active: {writers}")
        attempt = self.child(run["name"], "verify", [ROOT / "scripts/verify_local_run.py", "--config", run["config_path"]])
        audit = read_json(attempt["stdout"])
        if audit.get("status") != "verified":
            raise ValueError(f"Completion verifier did not confirm {run['name']}; see {attempt['stdout']}")
        attempt["status"] = "verified"
        return {"status": "completed", "verified_at_utc": utc_now(), "audit": attempt["stdout"],
                "stages": audit["stages"], "evaluation": audit["evaluation"]}

    def run_one(self, run):
        state = self.state["runs"].setdefault(run["name"], {"status": "pending", "stages": {}})
        evaluation = Path(run["evaluation_dir"])
        # A complete evaluation is reusable only after full independent verification.
        if evaluation.exists():
            manifest = evaluation / "run_manifest.json"
            if not manifest.is_file() or read_json(manifest).get("status") != "completed":
                raise ValueError(f"Existing evaluation is incomplete; will not overwrite: {evaluation}")
        else:
            for stage in STAGES:
                state["status"] = "running"
                self.state["current"] = {"run": run["name"], "stage": stage}
                self.wait_ready(run["config"])
                manifest_path = Path(run["output_dir"]) / stage / "run_manifest.json"
                args = [ROOT / "scripts/local_cli.py", "local-train", "--config", run["config_path"],
                        "--stage", stage, "--execute"]
                if manifest_path.exists():
                    args.append("--resume")
                attempt = self.child(run["name"], stage, args)
                manifest = read_json(manifest_path)
                if manifest.get("status") != "completed" or manifest.get("stage") != stage:
                    raise ValueError(f"{run['name']}/{stage} exited without a completed stage manifest")
                state["stages"][stage] = {"status": "completed", "manifest_sha256": sha256_file(manifest_path),
                                          "completed_optimizer_steps": manifest.get("completed_optimizer_steps")}
                attempt["status"] = "completed"
                self.save()
            self.state["current"] = {"run": run["name"], "stage": "evaluation"}
            self.wait_ready(run["config"])
            attempt = self.child(run["name"], "evaluation", [ROOT / "scripts/local_cli.py", "local-evaluate",
                                 "--config", run["config_path"], "--execute"])
            if read_json(evaluation / "run_manifest.json").get("status") != "completed":
                raise ValueError(f"{run['name']}/evaluation exited without completed artifacts")
            attempt["status"] = "completed"
        self.state["runs"][run["name"]] = self.verify(run)
        self.save()

    def execute(self, resume=False):
        identity = freeze_identity(self.loaded)
        with campaign_lock():
            if self.path.exists():
                if not resume:
                    raise ValueError("Existing campaign requires explicit --resume")
                self.state = read_json(self.path)
                if self.state.get("identity") != identity:
                    raise ValueError("Resume requires identical plan/config/source/data/model/runtime")
                if recorded_process_alive(self.state.get("current_child")):
                    raise RuntimeError("The recorded child is still alive; wait for it to exit before --resume")
            else:
                if resume:
                    raise FileNotFoundError("Cannot resume without campaign.json")
                if self.output.exists() and any(self.output.iterdir()):
                    raise FileExistsError(f"Refusing nonempty unregistered campaign output: {self.output}")
                self.state = {"schema_version": 1, "name": self.loaded["plan"]["name"], "identity": identity,
                              "plan": self.loaded["plan"], "started_at_utc": utc_now(), "started_at_epoch": time.time(),
                              "runs": {}, "attempts": [], "sessions": [], "current_child": None}
            self.state.update(status="running", error=None, stop_after_current_child=None,
                              supervisor=process_record(os.getpid()))
            session = {"started_at_utc": utc_now(), "supervisor": self.state["supervisor"]}
            self.state["sessions"].append(session)
            started = time.monotonic()
            atomic_json(ROOT / "reports/campaign-process.json", {"campaign": str(self.path), **self.state["supervisor"]})
            self.save()
            try:
                for run in self.loaded["runs"]:
                    self.run_one(run)
                if self.loaded["summary_script"]:
                    self.assert_identity()
                    attempt = self.child(self.loaded["plan"]["name"], "summary",
                                         [self.loaded["summary_script"], "--plan", self.loaded["path"],
                                          "--output", self.output / "analysis"])
                    attempt["status"] = "completed"
                    self.state["summary"] = {"status": "completed", "output_dir": str(self.output / "analysis"),
                                              "stdout": attempt["stdout"], "stderr": attempt["stderr"]}
                self.assert_identity()
                reason = self.stop_reason()
                if reason:
                    raise DispatchStopped(reason)
                self.state["status"] = "completed"
            except DispatchStopped as exc:
                self.state["status"] = exc.status
                current = self.state.get("current", {})
                if current.get("run") in self.state["runs"]:
                    self.state["runs"][current["run"]]["status"] = exc.status
            except BaseException as exc:
                self.state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                self.state["error"] = f"{type(exc).__name__}: {exc}"
                current = self.state.get("current", {})
                if self.state["attempts"]:
                    attempt = self.state["attempts"][-1]
                    if all(attempt.get(key) == current.get(key) for key in ("run", "stage")):
                        self.state["error"] += f"; stdout={attempt['stdout']}; stderr={attempt['stderr']}"
                        if not isinstance(exc, KeyboardInterrupt) and attempt["status"] not in ("completed", "verified"):
                            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if current.get("run") in self.state["runs"]:
                    self.state["runs"][current["run"]]["status"] = self.state["status"]
            finally:
                session.update(finished_at_utc=utc_now(), elapsed_wall_seconds=time.monotonic() - started,
                               status=self.state["status"])
                self.state["finished_at_utc"] = utc_now()
                self.save()
            return self.state


def run_campaign(plan_path, execute=False, resume=False):
    loaded = load_plan(plan_path)
    if not execute:
        return {"status": "plan_only", "training_started": False, "cuda_initialized": False,
                "name": loaded["plan"]["name"], "output_dir": loaded["output_dir"],
                "max_hours": loaded["plan"]["max_hours"],
                "runs": [{key: run[key] for key in ("name", "config_path", "output_dir", "evaluation_dir")}
                         for run in loaded["runs"]], "summary_script": loaded["summary_script"],
                "budget_policy": "STOP or the wall-clock limit stops all subsequent child dispatch, including verification/summary; let an active child finish"}
    return Campaign(loaded).execute(resume=resume)
