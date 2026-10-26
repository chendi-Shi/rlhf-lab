"""Exact-source CPU regression proof for the authorized PPO43 recovery."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRIOR = ROOT / "configs/alignlab-v2-throughput-resume609-20261023T080559Z/plan.json"
PRIOR_SHA = "adea484fc1b052d5c19eda58e8b5aba8dd18e4b13ee988e53034181cbea89086"
NEW = {
    "alignment_lab/ppo43_supplemental_v3.py",
    "alignment_lab/throughput_fresh_v3.py",
    "alignment_lab/throughput_supervisor_v3.py",
    "scripts/run_alignlab_throughput_fresh_v3.py",
    "scripts/run_alignlab_throughput_campaign_v3.py",
    "scripts/run_alignlab_throughput_cpu_checks_v3.py",
    "scripts/run_alignlab_throughput_prepare_v3.py",
    "tests/test_ppo43_supplemental_v3.py",
    "tests/test_throughput_fresh_v3.py",
    "tests/test_throughput_supervisor_v3.py",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sources():
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError("Frozen prior plan changed")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    actual = {name: sha(ROOT / name) for name in sorted(set(prior["source_sha256"]) | NEW)}
    if any(actual[name] != expected for name, expected in prior["source_sha256"].items()):
        raise ValueError("Frozen prior source changed")
    return actual

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("CPU evidence must have a fresh path")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    before = sources()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU regression must not initialize CUDA")
    modules = [p + "." + f.stem for p in ("tests_throughput_v1", "tests_throughput_v2")
               for f in sorted((ROOT / p).glob("test_*.py"))]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    for name in sorted(n for n in NEW if n.startswith("tests/test_")):
        label = name[:-3].replace("/", ".")
        spec = importlib.util.spec_from_file_location(label, ROOT / name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[label] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        modules.append(label)
    started = time.time()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    after = sources()
    passed = result.wasSuccessful() and not result.skipped and before == after and not torch.cuda.is_initialized()
    report = {"kind": "alignlab-throughput-cpu-validation-v3", "status": "passed" if passed else "failed",
              "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(), "tests_run": result.testsRun,
              "errors": len(result.errors), "failures": len(result.failures), "skipped": len(result.skipped),
              "source_before": before, "source_after": after, "modules": modules,
              "started_at_epoch": started, "finished_at_epoch": time.time(),
              "errors_detail": [(str(t), detail) for t, detail in result.errors],
              "failures_detail": [(str(t), detail) for t, detail in result.failures],
              "production_cuda_equivalence_proven": False, "production_speedup_proven": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
