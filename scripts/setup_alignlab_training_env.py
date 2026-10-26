"""Extend only the isolated AlignLab venv with pinned training dependencies.

No model downloads, CUDA work, global driver changes, or training are performed.
Run after the CPU data audit exits. A new report directory is required each time.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".runtime/alignlab-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != VENV.resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("Use this project's independent AlignLab venv interpreter")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports"):
        raise ValueError("Installation reports must be inside this project reports directory")
    candidates = json.loads((ROOT / "reports/alignlab-v2-training-dependency-candidates.json").read_text())
    torch_candidate = json.loads((ROOT / "reports/alignlab-v2-torch-wheel-candidate.json").read_text())
    torch_url = torch_candidate["wheel_url_with_sha256"]
    if urlsplit(torch_url).hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
        raise ValueError("Unexpected Torch wheel host")
    specs = ["torch @ " + torch_url]
    for name in ("bitsandbytes", "peft", "accelerate"):
        wheels = candidates["candidates"][name]["wheel_candidates"]
        if len(wheels) != 1 or urlsplit(wheels[0]["url"]).hostname != "files.pythonhosted.org":
            raise ValueError("Expected a single verified PyPI wheel for " + name)
        specs.append(name + " @ " + wheels[0]["url"] + "#sha256=" + wheels[0]["sha256"])
    if not args.execute:
        print(json.dumps({"python": sys.executable, "venv": str(VENV), "specs": specs,
                          "execute": False, "gpu_operations": False}))
        return 0
    # A prior/partial installation is evidence, not permission to overwrite it.
    output.mkdir(parents=True, exist_ok=False)
    audit_pins = (ROOT / "requirements-v2-audit.txt").read_text(encoding="utf-8")
    # Freeze every installed dependency, not just the top-level tokenizer pins.
    installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    constraints = output / "existing-constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version}" for name, version in sorted(installed.items())) + "\n",
                           encoding="utf-8")
    (output / "audit-requirements.txt").write_text(audit_pins, encoding="utf-8")
    requirements = output / "training-requirements.txt"
    requirements.write_text("\n".join(specs) + "\n", encoding="utf-8")
    record = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.executable, "venv": str(VENV), "packages_before": installed,
              "status": "installing", "gpu_verified": False, "training_started": False}
    record_path = output / "status.json"

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # All import checks are explicitly CPU only.
    env.pop("USE_TORCH", None)
    env.pop("USE_TF", None)
    env["PIP_REQUIRE_VIRTUALENV"] = "true"
    command = [sys.executable, "-I", "-m", "pip", "install", "--no-input", "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
               "--constraint", str(constraints), "--requirement", str(requirements),
               "--report", str(output / "pip-install.json"), "--disable-pip-version-check"]
    print("Installing pinned training packages into the isolated project environment", flush=True)
    record["command"] = command
    with (output / "pip-stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "pip-stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    record["pip_exit_code"] = result.returncode
    if result.returncode:
        record["status"] = "installation_failed"
        save()
        return result.returncode
    check = subprocess.run([sys.executable, "-I", "-m", "pip", "check"], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    record["pip_check"] = {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
    smoke_code = """import json,torch,transformers,peft,accelerate,bitsandbytes
assert torch.__version__ == '2.7.1+cu118', torch.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
assert not torch.cuda.is_initialized()
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,
 'peft':peft.__version__,'accelerate':accelerate.__version__,'bitsandbytes':bitsandbytes.__version__,
 'cuda_initialized':torch.cuda.is_initialized(),'device':'cpu','training_started':False}))
"""
    smoke = subprocess.run([sys.executable, "-I", "-B", "-c", smoke_code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    record["cpu_import_smoke"] = {"exit_code": smoke.returncode, "stdout": smoke.stdout, "stderr": smoke.stderr}
    record["status"] = "cpu_imports_verified" if check.returncode == 0 and smoke.returncode == 0 else "verification_failed"
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps({"status": record["status"], "report": str(record_path), "gpu_verified": False}), flush=True)
    return 0 if record["status"] == "cpu_imports_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
