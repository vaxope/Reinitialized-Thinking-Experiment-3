"""Minimal test runner (pytest is not installed locally; the tests are also pytest-compatible).

Runs every `test_*` function in tests/test_*.py. A test that returns a string starting
with "SKIP" is reported as skipped. Exit code 1 if anything fails.
Usage: python3 tests/run_all.py
"""
import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import src...` works without installing the package

passed, failed, skipped = 0, [], []
for path in sorted((ROOT / "tests").glob("test_*.py")):
    mod = importlib.import_module(f"tests.{path.stem}")
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            result = getattr(mod, name)()
        except Exception:
            failed.append(f"{path.stem}.{name}\n{traceback.format_exc()}")
            continue
        if isinstance(result, str) and result.startswith("SKIP"):
            skipped.append(f"{path.stem}.{name}: {result}")
        else:
            passed += 1

for s in skipped:
    print("SKIPPED", s)
for f in failed:
    print("FAILED", f)
print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
sys.exit(1 if failed else 0)
