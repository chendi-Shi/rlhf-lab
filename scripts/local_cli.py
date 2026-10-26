"""CLI bootstrap for a Python executable whose isolated mode excludes the working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlhf_lab.__main__ import main

raise SystemExit(main())
