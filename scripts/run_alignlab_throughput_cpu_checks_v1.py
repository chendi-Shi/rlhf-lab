"""CPU-only targeted continuity checks; never production CUDA speed evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sources():
    plan = json.loads((ROOT / "configs/alignlab-v2-primary-r3-20261022T064456409883Z/plan.json").read_text(encoding="utf-8"))
    names = set(plan["source_sha256"])
    names.update(str(p.relative_to(ROOT)).replace("\\", "/") for pattern in
        ("alignment_lab/throughput_*_v1.py", "scripts/run_alignlab_throughput*_v1.py", "tests_throughput_v1/*.py")
        for p in ROOT.glob(pattern))
    actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    if any(actual[name] != expected for name, expected in plan["source_sha256"].items()):
        raise RuntimeError("Frozen R3 dependency changed")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modules", nargs="*")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh CPU report path required")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    before = sources()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only checks must not initialize CUDA")
    modules = args.modules or ["tests_throughput_v1." + p.stem for p in sorted((ROOT / "tests_throughput_v1").glob("test_*.py"))]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    started = time.time()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    after = sources()
    passed = result.wasSuccessful() and not result.skipped and before == after and not torch.cuda.is_initialized()
    report = {"kind": "alignlab-throughput-cpu-validation-v1", "status": "passed" if passed else "failed",
        "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(), "tests_run": result.testsRun,
        "errors": len(result.errors), "failures": len(result.failures), "skipped": len(result.skipped),
        "source_before": before, "source_after": after, "modules": modules,
        "started_at_epoch": started, "finished_at_epoch": time.time(),
        "errors_detail": [(str(t), text) for t, text in result.errors],
        "failures_detail": [(str(t), text) for t, text in result.failures],
        "production_cuda_equivalence_proven": False, "production_speedup_proven": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
