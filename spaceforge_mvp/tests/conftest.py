"""Pytest configuration ensuring spaceforge_mvp package discovery."""

import sys
from pathlib import Path

# Insert package directory into sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
