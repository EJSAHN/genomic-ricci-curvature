"""Entry points for release checking, derived-input replay, and raw-read analysis."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'replay', 'analyze', 'export-tables'])
    args, rest = parser.parse_known_args()
    scripts = {'verify': ROOT/'tools/verify_release.py', 'replay': ROOT/'workflows/replay.py',
               'analyze': ROOT/'workflows/analyze.py', 'export-tables': ROOT/'tools/export_reporting_tables.py'}
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # Integrity is checked before numerical replay or explicit raw-read work.
    if args.command in {'replay','analyze'} and '--help' not in rest and '-h' not in rest:
        subprocess.run([sys.executable, str(scripts['verify'])], check=True, env=env)
    return subprocess.call([sys.executable, '-u', str(scripts[args.command]), *rest], env=env)


if __name__ == '__main__': raise SystemExit(main())
