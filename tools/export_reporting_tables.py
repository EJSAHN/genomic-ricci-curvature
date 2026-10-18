"""Export current numerical tables from the article's Appendix S1, without figures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workflows'))
from common import ROOT, make_workspace, read_rows, require, write_json
from xlsx_readonly import CachedWorkbook


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', type=Path, required=True, help='The integrated Appendix_S1.xlsx supplied with the article')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = make_workspace(a.output)
    workbook = CachedWorkbook(a.workbook)
    exported = []
    for distributed in sorted((ROOT / 'reporting').glob('*.tsv')):
        name = distributed.stem
        records = workbook.records(name, 4)
        fields = list(read_rows(distributed)[0])
        rows = [{key: r.get(key) for key in fields} for r in records]
        with (out / distributed.name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        exported.append({'sheet': name, 'rows': len(rows), 'file': distributed.name})
    write_json(out/'EXPORT_REPORT.json', {'exported': exported, 'original_workbook_modified': False})
    print('Exported numerical tables:', out)


if __name__ == '__main__': main()
