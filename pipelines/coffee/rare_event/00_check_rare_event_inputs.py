# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--read_level_root', required=True)
    p.add_argument('--outdir', required=True)
    args=p.parse_args()
    root=Path(args.read_level_root); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name, ok, detail):
        checks.append({'name':name,'ok':bool(ok),'detail':str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    check('read-level root', root.is_dir(), root)
    pass_file=root/'audit'/'READ_LEVEL_VALIDATION_PASS.txt'
    check('read-level validation audit', pass_file.is_file() and 'PASS' in pass_file.read_text(errors='ignore'), pass_file)
    truth_path=root/'generated'/'manifests'/'truth_manifest.tsv'
    check('truth manifest', truth_path.is_file(), truth_path)
    if truth_path.is_file():
        truth=pd.read_csv(truth_path,sep='\t')
        check('truth rows', len(truth)==365, len(truth))
        check('scenario set', set(truth['scenario'])=={'primary','conservative'}, sorted(set(truth['scenario'])))
        check('replicate set', set(truth['replicate'])==set(range(1,6)), sorted(set(truth['replicate'])))
    missing=[]
    for s in ['primary','conservative']:
        for r in range(1,6):
            for mode in ['r1','paired']:
                f=root/'analysis'/'runs'/s/f'rep_{r:02d}'/mode/'js_distance.csv'
                if not f.is_file(): missing.append(str(f))
    check('precomputed distance matrices', not missing, f"missing={len(missing)}")
    payload={'status':'PASS' if all(x['ok'] for x in checks) else 'FAIL','checks':checks,'missing_distance_matrices':missing}
    (out/'rare_event_preflight.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if payload['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
