"""
conftest.py — pytest configuration for BengaluruFlow tests.

Adds the project root to sys.path so that `import src.xxx` works
when pytest is run from the project root directory.
"""

import sys
from pathlib import Path

# Insert project root at the beginning of sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
