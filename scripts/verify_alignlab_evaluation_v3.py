"""Current CPU proof for the new evaluator loader and V3 execution integration.

Runs new loader/model/capacity/execution/supervisor/analysis/plan regressions and
the unchanged acceptance/resource/scoring suites used by this execution. Source
hash observation does not imply that every historical training file was tested.
The old 151-test report is historical evidence, never this version's result.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "tests_v2.test_model_loader_eval_v3", "tests_v2.test_evaluation_model_v3",
    "tests_v2.test_evaluation_execution_v3", "tests_v2.test_evaluation_capacity_v3",
    "tests_v2.test_evaluation_supervisor_v3", "tests_v2.test_evaluation_analysis_v3",
    "tests_v2.test_evaluation_prepare_v3", "tests_v2.test_evaluation_acceptance_v2",
    "tests_v2.test_evaluation_resources_v2", "tests_v2.test_official_evaluation_v2",
    "tests_v2.test_evaluation_scoring_cli_v2",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-worker", action="store_true")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/alignlab-v2").resolve():
        raise ValueError("Use the pinned isolated CPU interpreter")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA must be hidden before interpreter startup")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports") or (not args.official_worker and output.parent != ROOT / "reports"):
        raise ValueError("Use a fresh report directory within this project")
    output.mkdir(exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = [str(ROOT), str(ROOT / "tests_v2")]
    sources = list((ROOT / "alignment_lab").glob("*.py")) + list((ROOT / "tests_v2").glob("test_*.py"))
    # Every executable source present in the reused plan also receives a current
    # hash observation; only MODULES above claim actual test execution coverage.
    sources += list((ROOT / "scripts").glob("*.py"))
    sources += [ROOT / "rlhf_lab/__init__.py", ROOT / "rlhf_lab/runtime.py"]
    sources = sorted(set(sources))
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    syntax_checked = []
    for source in sources:
        if source.name.endswith("_v3.py") and ("evaluation" in source.name or "model_loader_eval" in source.name):
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            syntax_checked.append(source.relative_to(ROOT).as_posix())
    started, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    official_name = "tests_v2.test_official_evaluation_v2"
    worker, worker_code = None, 0
    if not args.official_worker:
        worker_dir = output / "official-worker"
        with (output / "official-worker.log").open("w", encoding="utf-8") as worker_log:
            child = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                "--output", str(worker_dir), "--official-worker"], cwd=ROOT,
                stdout=worker_log, stderr=subprocess.STDOUT, timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        worker_code = child.returncode
        worker = json.loads((worker_dir / "report.json").read_text(encoding="utf-8"))
    modules = (official_name,) if args.official_worker else tuple(m for m in MODULES if m != official_name)
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not args.official_worker:
        import torch
    cuda_initialized = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
    unchanged = before == {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    worker_ok = worker is None or (worker_code == 0 and worker["status"] == "cpu_software_checks_passed"
                                  and worker["source_sha256"] == before and worker["torch_imported"] is False)
    passed = result.wasSuccessful() and not result.skipped and unchanged and not cuda_initialized and worker_ok
    report = {"kind": "alignlab-independent-evaluation-v3-cpu-proof", "schema_version": 1,
        "status": "cpu_software_checks_passed" if passed else "cpu_software_checks_failed",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - t0, "tests_run": result.testsRun + (worker or {}).get("tests_run", 0),
        "failures": len(result.failures) + (worker or {}).get("failures", 0),
        "errors": len(result.errors) + (worker or {}).get("errors", 0),
        "skipped": len(result.skipped) + (worker or {}).get("skipped", 0),
        "test_modules": modules if args.official_worker else MODULES,
        "source_sha256": before, "source_unchanged": unchanged, "syntax_checked": syntax_checked,
        "device": "cpu", "cuda_initialized": cuda_initialized, "torch_imported": "torch" in sys.modules,
        "python": sys.executable, "official_worker": worker,
        "pretrained_model_weights_used": False, "training_started": False, "capacity_verified": False,
        "model_acceptance_repeated": False, "historical_151_tests_reused_as_current_proof": False,
        "scope": __doc__, "direct_v4_training_cpu_tests_run": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "errors", "failures", "elapsed_seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
