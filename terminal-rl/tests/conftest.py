from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
TERMINAL_RL_ROOT = TESTS_DIR.parents[1]
REPO_ROOT = TERMINAL_RL_ROOT.parent
SLIME_ROOT = REPO_ROOT / "slime"

for path in (str(TERMINAL_RL_ROOT), str(REPO_ROOT), str(SLIME_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
