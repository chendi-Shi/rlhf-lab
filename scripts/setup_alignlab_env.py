"""Create and verify the isolated, CPU-only AlignLab v2 audit environment.

Run with a standard CPython (venv and ensurepip available), or pass
--base-python. No installs are made into that source interpreter. Its stdlib
is reused read-only by a normal venv; site-packages are local and independent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".runtime" / "alignlab-v2"
REQUIREMENTS = ROOT / "requirements-v2-audit.txt"
REPORT = ROOT / "reports" / "alignlab-v2-environment.json"
FORBIDDEN = {"torch", "torchvision", "torchaudio", "trl", "bitsandbytes", "peft",
             "accelerate", "tensorflow", "jax", "jaxlib", "flax", "triton"}

PROBE = r'''
import importlib.metadata as md, importlib.util, json, site, sys
from pathlib import Path
packages = {d.metadata["Name"]: {"version":d.version,"location":str(Path(d.locate_file("")).resolve())}
            for d in md.distributions() if d.metadata["Name"]}
print(json.dumps({"executable":sys.executable,"version":sys.version,
 "version_info":list(sys.version_info[:3]),"prefix":sys.prefix,"base_prefix":sys.base_prefix,
 "base_executable":getattr(sys,"_base_executable",None),"sys_path":sys.path,
 "enable_user_site":site.ENABLE_USER_SITE,"site_packages":site.getsitepackages(),
 "isolated_flag":sys.flags.isolated,"venv_available":importlib.util.find_spec("venv") is not None,
 "ensurepip_available":importlib.util.find_spec("ensurepip") is not None,"packages":packages},sort_keys=True))
'''

SMOKE = r'''
import importlib.util, json
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
backend=Tokenizer(models.WordLevel({"[UNK]":0,"hello":1,"world":2},unk_token="[UNK]"))
backend.pre_tokenizer=pre_tokenizers.Whitespace()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]")
tokenizer.chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}"
ids=tokenizer.apply_chat_template([{"role":"user","content":"hello world"}],tokenize=True,return_dict=False)
assert ids==[1,2],ids
absent={name:importlib.util.find_spec(name) is None for name in ("torch","trl","bitsandbytes","peft","accelerate")}
assert all(absent.values()),absent
print(json.dumps({"local_tokenizer_roundtrip":True,"chat_template_ids":ids,
 "training_imports_absent":absent,"network_used":False,"gpu_used":False}))
'''


def clean_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in {"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"} or key.upper().startswith("PIP_"):
            env.pop(key)
    env.update(PYTHONNOUSERSITE="1", USE_TORCH="0", USE_TF="0", USE_FLAX="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def run(command, *, capture=False, env=None):
    try:
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                env=clean_environment() if env is None else env,
                                text=True, encoding="utf-8", errors="replace",
                                capture_output=capture, check=True)
    except subprocess.CalledProcessError as exc:
        if capture:
            raise RuntimeError(f"Environment verification failed: {(exc.stderr or '')[-4000:]}") from exc
        raise
    return result.stdout if capture else None


def probe(python, *, isolated=True, env=None):
    return json.loads(run([python, "-B", *(["-I"] if isolated else []), "-c", PROBE], capture=True, env=env))


def validate_isolation(info):
    root = ENV.resolve()
    if Path(info["prefix"]).resolve() != root or Path(info["base_prefix"]).resolve() == root:
        raise RuntimeError("Not an independent venv at the expected path")
    if not Path(info["executable"]).resolve().is_relative_to(root):
        raise RuntimeError("Python executable is outside the requested venv")
    if info["enable_user_site"] is not False:
        raise RuntimeError("User site-packages must be disabled")
    for value in info["sys_path"] + info["site_packages"]:
        path = Path(value or ROOT).resolve()
        if "site-packages" in path.parts and not path.is_relative_to(root):
            raise RuntimeError(f"Foreign site-packages on sys.path: {path}")
    for name, package in info["packages"].items():
        if not Path(package["location"]).resolve().is_relative_to(root):
            raise RuntimeError(f"Package {name} inherited from outside the venv")
    for path in (ENV, ENV / "Lib", ENV / "Lib" / "site-packages"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise RuntimeError(f"Mutable runtime path must not be linked: {path}")
    config = (ENV / "pyvenv.cfg").read_text(encoding="utf-8")
    options = dict(line.split("=", 1) for line in config.splitlines() if "=" in line)
    options = {k.strip().lower(): v.strip().lower() for k, v in options.items()}
    if options.get("include-system-site-packages") != "false":
        raise RuntimeError("System site-packages inheritance is enabled")


def atomic_report(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT)


def setup(base_python, *, verify_only=False):
    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "preparing", "purpose": "CPU tokenizer/data audit; not training ready",
              "environment_root": str(ENV), "requirements": str(REQUIREMENTS),
              "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
              "setup_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu_operations_performed": False, "model_weights_downloaded": False,
              "source_environment_modified": False}
    try:
        python = ENV / "Scripts" / "python.exe"
        if ENV.exists() and not (ENV / "pyvenv.cfg").is_file():
            raise RuntimeError("Existing environment has no pyvenv.cfg; refusing to overwrite")
        if not ENV.exists():
            if verify_only:
                raise FileNotFoundError("Audit environment has not been created")
            source = probe(Path(base_python).resolve())
            if not source["venv_available"] or not source["ensurepip_available"]:
                raise RuntimeError("Use standard CPython with venv and ensurepip; embedded Python is unsuitable")
            if source["version_info"][:2] != [3, 12]:
                raise RuntimeError("This audit profile requires CPython 3.12")
            ENV.parent.mkdir(parents=True, exist_ok=True)
            run([base_python, "-I", "-B", "-m", "venv", "--copies", ENV])
            report["creation_action"] = "created_standard_venv"
        else:
            report["creation_action"] = "reused_verified_venv"
        initial = probe(python)
        validate_isolation(initial)
        if initial["version_info"][:2] != [3, 12]:
            raise RuntimeError("Existing environment is not CPython 3.12; refusing to replace")
        installed = {name.lower().replace("_", "-") for name in initial["packages"]}
        if installed & FORBIDDEN:
            raise RuntimeError("Existing environment contains training packages; refusing to mutate an expanded runtime")
        if not verify_only:
            pip = [python, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check"]
            args = ["install", "--no-input", "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                    "--cache-dir", ENV / ".pip-cache", "-r", REQUIREMENTS]
            plan_path = ENV / "audit-install-plan.json"
            run([*pip, *args, "--dry-run", "--report", plan_path])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            planned = {item["metadata"]["name"].lower().replace("_", "-") for item in plan.get("install", [])}
            if planned & FORBIDDEN or any(name.startswith(("nvidia-", "cuda-")) for name in planned):
                raise RuntimeError(f"Audit dependency plan includes training/CUDA packages: {sorted(planned)}")
            run([*pip, *args, "--report", ENV / "audit-install-result.json"])
        normal = probe(python, isolated=False)
        isolated = probe(python)
        validate_isolation(normal)
        validate_isolation(isolated)
        contaminated = clean_environment()
        contaminated["PYTHONPATH"] = str(Path(isolated["base_prefix"]) / "Lib" / "site-packages")
        challenged = probe(python, env=contaminated)
        validate_isolation(challenged)
        smoke = json.loads(run([python, "-I", "-B", "-c", SMOKE], capture=True))
        pip_check = run([python, "-I", "-B", "-m", "pip", "--isolated", "check"], capture=True).strip()
        report.update(status="audit_ready", python=isolated, normal_launch_sys_path=normal["sys_path"],
                      isolation={"normal_launch_with_sanitized_environment_passed": True, "isolated_launch_passed": True,
                                 "foreign_pythonpath_ignored_with_I": True, "system_site_packages": False,
                                 "user_site_packages": False, "site_packages_links": False,
                                 "stdlib_reused_read_only_from": isolated["base_prefix"]},
                      tokenizer_smoke=smoke, pip_check=pip_check,
                      missing_training_dependencies=sorted(FORBIDDEN & {"torch", "trl", "bitsandbytes", "peft", "accelerate"}),
                      recommended_invocation=f'"{python}" -I <audit-script>')
        install_report = ENV / "audit-install-result.json"
        if install_report.is_file():
            report["dependency_install_report"] = {
                "path": str(install_report),
                "sha256": hashlib.sha256(install_report.read_bytes()).hexdigest(),
                "contains": "PyPI wheel URLs and archive SHA256 for the actual installation",
            }
        atomic_report(report)
        print(json.dumps({"status": report["status"], "python": str(python), "report": str(REPORT),
                          "installed_package_count": len(isolated["packages"]), "isolation_passed": True}, indent=2))
        return report
    except BaseException as exc:
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        atomic_report(report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    setup(args.base_python, verify_only=args.verify_only)
