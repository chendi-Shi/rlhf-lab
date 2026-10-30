"""CPU regression proof for V8 continuation, restoration succession and inheritance.

Changed orchestration and inference admission are exercised with stdlib fake
models. Real official CPU checker fixtures use the accepted scoring environment.
Unchanged V5 GPU kernels/loader/attention proofs remain historical pins. No
accepted pretrained weights, CUDA, model acceptance or training occurs.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "tests_v2.test_evaluation_resources_v8", "tests_v2.test_evaluation_prepare_v8",
    "tests_v2.test_evaluation_execution_v8",
    "tests_v2.test_evaluation_supervisor_v8", "tests_v2.test_evaluation_analysis_v8",
    "tests_v2.test_official_evaluation_v6", "tests_v2.test_scorer_v6", "tests_v2.test_scorer_v8",
    "tests_v2.test_evaluation_inheritance_v8", "tests_v2.test_evaluation_capacity_inheritance_v8",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".runtime/alignlab-eval-20261020").resolve():
        raise ValueError("Use the pinned official scoring CPU interpreter")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA must be hidden before interpreter startup")
    output = args.output.resolve()
    if output.parent != ROOT / "reports":
        raise ValueError("Use a fresh direct project report directory")
    output.mkdir(exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(ROOT), str(ROOT / "tests_v2")]
    sources = sorted(set(list((ROOT / "alignment_lab").glob("*.py"))
        + list((ROOT / "tests_v2").glob("test_*.py")) + list((ROOT / "scripts").glob("*.py"))
        + [ROOT / "rlhf_lab/__init__.py", ROOT / "rlhf_lab/runtime.py"]))
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    syntax_checked = []
    for source in sources:
        if source.name.endswith("_v8.py") and "evaluation" in source.name:
            ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            syntax_checked.append(source.relative_to(ROOT).as_posix())
    started, t0 = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    cuda_initialized = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
    unchanged = before == {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    passed = result.wasSuccessful() and not result.skipped and unchanged and not cuda_initialized and "torch" not in sys.modules
    report = {"kind": "alignlab-independent-evaluation-v8-cpu-proof", "schema_version": 1,
        "status": "cpu_software_checks_passed" if passed else "cpu_software_checks_failed",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - t0, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "test_modules": MODULES, "source_sha256": before, "source_unchanged": unchanged,
        "syntax_checked": syntax_checked, "device": "cpu", "cuda_initialized": cuda_initialized,
        "torch_imported": "torch" in sys.modules, "python": sys.executable,
        "pretrained_model_weights_used": False, "model_acceptance_repeated": False,
        "training_started": False, "capacity_verified": False,
        "historical_166_tests_reused_as_current_proof": False,
        "historical_77_tests_reused_as_current_proof": False, "scope": __doc__}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "errors", "failures", "elapsed_seconds")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
