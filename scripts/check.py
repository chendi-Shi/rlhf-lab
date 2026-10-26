"""Run only dependency-free preparation tests; no model loading or GPU initialization."""

import ast
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
files = list((root / "rlhf_lab").glob("*.py")) + list((root / "scripts").glob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
suite = unittest.TestSuite()
for pattern in ("test_data.py", "test_runtime.py", "test_evaluate.py", "test_reward_eval.py", "test_local_runtime.py"):
    suite.addTests(unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "syntax_checked_files": len(files), "tests_run": result.testsRun,
          "failures": len(result.failures), "errors": len(result.errors),
          "tests_passed": result.wasSuccessful(),
          "gpu_training_tested": False, "model_quality_measured": False,
          "scope": "Standard-library unit/integration tests with synthetic fixtures; no training dependencies loaded."}
(root / "reports").mkdir(exist_ok=True)
(root / "reports" / "preparation-check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
raise SystemExit(0 if result.wasSuccessful() else 1)
