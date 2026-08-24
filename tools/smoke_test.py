from __future__ import annotations
import compileall
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ok = compileall.compile_dir(root / "pipelines", quiet=1)
if not ok:
    raise SystemExit("Python syntax check failed")
print("Python syntax check: PASS")
