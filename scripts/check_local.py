"""Run software checks and save an honest report (no pretrained quality claims)."""
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import pytest


class Results:
    def __init__(self):
        self.counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.counts[report.outcome] += 1
        elif report.failed:
            self.counts["errors"] += 1


result = Results()
code = pytest.main([str(root / "tests"), "-q"], plugins=[result])
summary = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "pytest_exit_code": int(code),
           **result.counts, "scope": "Software tests including tiny random GPT-2 fixtures; not pretrained model quality evidence", "gpu_training_tested": False}
(root / "reports").mkdir(exist_ok=True)
(root / "reports" / "local-check.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
print(json.dumps(summary))
raise SystemExit(code)
