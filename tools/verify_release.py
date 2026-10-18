"""Check the file manifest, public-package boundary, and current reporting tables."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'PUBLIC_RELEASE_MANIFEST.tsv'
IGNORED_TOP = {'.git', 'outputs', 'runs', '.pytest_cache', '.venv'}
FORBIDDEN = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.docx', '.pptx', '.ttf', '.otf', '.bam', '.sam', '.cram', '.fastq', '.fq', '.fasta', '.fna', '.bt2', '.zip', '.pyc'}
PLOT_MODULES = {'matplotlib', 'seaborn', 'plotly', 'bokeh', 'svgwrite', 'reportlab'}
PERSONAL = re.compile(r'(?<![\w])[A-Za-z]:[\\/](?:Users|projects)[\\/]|(?<![\w.])/(?:home|Users|mnt)/[\w.-]+/')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files():
    return {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
            if p.is_file() and p.relative_to(ROOT).parts[0] not in IGNORED_TOP
            and '__pycache__' not in p.parts and not p.name.endswith('.pyc')}


def verify():
    with (ROOT / MANIFEST).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    expected = {r['path']: r for r in rows}
    actual = files()
    if len(expected) != len(rows) or set(actual) != set(expected) | {MANIFEST}:
        raise ValueError('File inventory mismatch: missing=' + repr(sorted(set(expected)-set(actual))) + '; unexpected=' + repr(sorted(set(actual)-set(expected)-{MANIFEST})))
    n_python = 0
    for rel, row in expected.items():
        path = actual[rel]
        if path.is_symlink() or path.stat().st_size != int(row['bytes']) or sha(path) != row['sha256']:
            raise ValueError('Changed file: ' + rel)
        if path.suffix.lower() in FORBIDDEN or path.name.lower().endswith(('.fastq.gz', '.fq.gz', '.fasta.gz', '.fna.gz')):
            raise ValueError('Unexpected public artifact: ' + rel)
        if path.suffix in {'.py', '.tsv', '.csv', '.md', '.txt', '.json', '.yml', '.yaml', '.cff'}:
            text = path.read_text(encoding='utf-8-sig')
            # The boundary checker contains its own forbidden-pattern vocabulary.
            if rel != 'tools/verify_release.py' and PERSONAL.search(text):
                raise ValueError('Personal path/dialog text: ' + rel)
        if path.suffix == '.py':
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=rel)
            n_python += 1
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import): names = [x.name.split('.')[0] for x in node.names]
                elif isinstance(node, ast.ImportFrom): names = [(node.module or '').split('.')[0]]
                if PLOT_MODULES.intersection(names): raise ValueError('Plotting dependency in ' + rel)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'savefig':
                    raise ValueError('Plot writer in ' + rel)
    sys.path.insert(0, str(ROOT / 'workflows'))
    from common import verify_metadata, verify_design_contract
    from replay import validate_reporting, check_sensitivity, check_ablation
    return {'result': 'PASS', 'files_checked': len(expected), 'python_sources_parsed': n_python,
            'plotting_code_in_public_package': False, 'metadata': verify_metadata(), 'design': verify_design_contract(),
            'reporting': validate_reporting(), 'sensitivity_checkpoints': check_sensitivity(), 'ablation': check_ablation(),
            'raw_read_alignment_performed': False, 'git_remote_modified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    result = verify()
    if a.report:
        q = a.report.resolve()
        if q == ROOT or q.is_relative_to(ROOT) and q.relative_to(ROOT).parts[0] not in {'outputs','runs'}:
            raise ValueError('Write the check report outside the source tree or under outputs/.')
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
