"""Run new AlignLab software fixtures offline and without initializing CUDA.

This is not the four-stage trainer, a pretrained model run, or GPU capacity.
Requires the completed project-local training environment and a new report path.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/alignlab-v2").resolve():
        raise ValueError("Use only the isolated project interpreter")
    environment = json.loads(args.environment_report.read_text(encoding="utf-8-sig"))
    if environment.get("status") != "cpu_imports_verified":
        raise ValueError("Training environment import verification must pass first")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports"):
        raise ValueError("Use a fresh output inside project reports")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.pop("USE_TORCH", None)
    os.environ.pop("USE_TF", None)
    sys.path.insert(0, str(ROOT))
    sources = list((ROOT / "alignment_lab").glob("*.py")) + list((ROOT / "tests_v2").glob("test_*.py"))
    sources += [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.perf_counter()
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests_v2"), pattern="test_*.py")
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    import torch
    unchanged = hashes == {str(p.relative_to(ROOT)): sha(p) for p in sources}
    passed = result.wasSuccessful() and not result.skipped and unchanged and not torch.cuda.is_initialized()
    report = {"status": "cpu_software_checks_passed" if passed else "cpu_software_checks_failed",
              "started_at_utc": started, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "elapsed_seconds": time.perf_counter() - t0, "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "source_sha256": hashes, "source_unchanged": unchanged,
              "python": sys.executable, "versions": {name: importlib.metadata.version(name) for name in
                   ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers")},
              "environment_report": str(args.environment_report.resolve()),
              "environment_report_sha256": sha(args.environment_report),
              "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
              "pretrained_model_weights_used": False, "new_project_training_started": False,
              "capacity_verified": False,
              "scope": "Data, codec, reference-cache, gradient objectives, tiny random Qwen3 adapter access and PPO sampling/update software checks; not a full trainer or GPU result"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "errors", "failures", "elapsed_seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
