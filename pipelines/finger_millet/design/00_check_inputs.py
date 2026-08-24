# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, parse_int, read_json, read_tsv, write_json, write_tsv

EXPECTED_POPULATIONS = {'Pop-1':12,'Pop-2':12,'Pop-3':12,'Pop-4':12,'Pop-5':9,'Pop-6':14,'Pop-7':12}
EXPECTED_HASH = '92a780376f8ca250402f6ad2330b27f70b39e433c4b2b85d7b33ea88ee002e1e'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_root', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--expected_hash', default=EXPECTED_HASH)
    args = parser.parse_args()

    root = Path(args.module_root)
    outdir = ensure_dir(args.outdir)
    paths = {
        'metadata_pass': root/'01_metadata_audit'/'METADATA_AUDIT_PASS.txt',
        'panel_pass': root/'02_subset_manifest'/'PANEL_SELECTION_PASS.txt',
        'download_pass': root/'06_download_audit'/'FASTQ_DOWNLOAD_COMPLETE.txt',
        'selection_summary': root/'02_subset_manifest'/'panel_selection_summary.json',
        'download_audit': root/'06_download_audit'/'fastq_download_audit.json',
        'samples': root/'02_subset_manifest'/'finger_millet_panel_83_samples.tsv',
        'runs': root/'02_subset_manifest'/'finger_millet_panel_83_runs.tsv',
        'inventory': root/'06_download_audit'/'fastq_inventory_verified.tsv',
    }
    checks: List[Dict] = []
    def check(name, observed, expected, ok=None):
        if ok is None: ok = observed == expected
        checks.append({'name':name,'observed':observed,'expected':expected,'ok':bool(ok)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: observed={observed}; expected={expected}")

    for key in ('metadata_pass','panel_pass','download_pass'):
        check(key, str(paths[key]), 'existing file', paths[key].exists())
    if not all(paths[key].exists() for key in paths):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise SystemExit('Missing required acquisition files:\n' + '\n'.join(missing))

    selection = read_json(paths['selection_summary'])
    download = read_json(paths['download_audit'])
    samples = read_tsv(paths['samples'])
    runs = read_tsv(paths['runs'])
    inventory = read_tsv(paths['inventory'])

    check('selection status', selection.get('status'), 'PASS')
    check('download audit status', download.get('status'), 'PASS')
    check('selection manifest SHA-256', selection.get('selection_manifest_sha256'), args.expected_hash)
    check('download manifest SHA-256', download.get('selection_manifest_sha256'), args.expected_hash)
    check('sample count', len(samples), 83)
    check('run count', len(runs), 83)
    check('FASTQ inventory rows', len(inventory), 166)
    population_counts = Counter(row.get('population','') for row in samples)
    check('population counts', dict(population_counts), EXPECTED_POPULATIONS)

    by_sample: Dict[str, List[Dict[str,str]]] = defaultdict(list)
    missing_files: List[str] = []
    size_failures: List[str] = []
    unverified: List[str] = []
    for row in inventory:
        by_sample[row['sample_accession']].append(row)
        path = Path(row['local_path'])
        if not path.exists():
            missing_files.append(str(path)); continue
        expected = parse_int(row.get('bytes'))
        if path.stat().st_size != expected:
            size_failures.append(f'{path}: {path.stat().st_size} != {expected}')
        if str(row.get('verified','')).lower() != 'true':
            unverified.append(str(path))
    check('missing local FASTQ files', len(missing_files), 0)
    check('FASTQ byte-size failures', len(size_failures), 0)
    check('inventory unverified rows', len(unverified), 0)

    run_index = {row['sample_accession']:row for row in runs}
    sample_index = {row['sample_accession']:row for row in samples}
    paired_rows: List[Dict] = []
    pairing_failures: List[Dict] = []
    for sample_id, sample in sample_index.items():
        files = sorted(by_sample.get(sample_id, []), key=lambda row: int(row.get('mate','0')))
        mates = {row.get('mate'):row for row in files}
        if set(mates) != {'1','2'}:
            pairing_failures.append({'sample_accession':sample_id,'reason':f'mates={sorted(mates)}'})
            continue
        run = run_index.get(sample_id, {})
        paired_rows.append({
            'panel_order':sample.get('panel_order',''),
            'population':sample.get('population',''),
            'sample_accession':sample_id,
            'sample_alias':sample.get('sample_alias',''),
            'run_accession':run.get('run_accession', sample.get('run_accessions','')),
            'read_count_archive':run.get('read_count',''),
            'base_count_archive':run.get('base_count',''),
            'r1_path':mates['1']['local_path'],
            'r2_path':mates['2']['local_path'],
            'r1_bytes':mates['1']['bytes'],
            'r2_bytes':mates['2']['bytes'],
            'r1_md5':mates['1']['md5'],
            'r2_md5':mates['2']['md5'],
            'selection_key_sha256':sample.get('selection_key_sha256',''),
        })
    paired_rows.sort(key=lambda row: int(row['panel_order']))
    check('paired sample rows', len(paired_rows), 83)
    check('pairing failures', len(pairing_failures), 0)
    write_tsv(outdir/'paired_input_manifest.tsv', paired_rows)
    write_tsv(outdir/'input_pairing_failures.tsv', pairing_failures)

    status = 'PASS' if all(item['ok'] for item in checks) else 'FAIL'
    report = {'status':status,'checks':checks,'expected_selection_hash':args.expected_hash}
    write_json(outdir/'input_preflight.json', report)
    (outdir/'input_preflight.txt').write_text('\n'.join([f"STATUS: {status}"] + [f"[{'PASS' if x['ok'] else 'FAIL'}] {x['name']}: observed={x['observed']}; expected={x['expected']}" for x in checks]) + '\n', encoding='utf-8')
    if status != 'PASS':
        raise SystemExit('[ERROR] External-data input preflight failed.')
    (outdir/'INPUT_PREFLIGHT_PASS.txt').write_text('PASS\n', encoding='utf-8')
    print('STATUS: PASS')

if __name__ == '__main__':
    main()
