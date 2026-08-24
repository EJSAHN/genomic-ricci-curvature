# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, parse_int, read_json, read_tsv, sha256_file, write_json, write_tsv


def add_tree(archive, root: Path, prefix: str, exclude_suffixes=('.npz',)):
    if not root.exists(): return
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() in exclude_suffixes: continue
        archive.write(path,str(Path(prefix)/path.relative_to(root)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module_root',required=True)
    parser.add_argument('--work_root',required=True)
    parser.add_argument('--outdir',required=True)
    args=parser.parse_args()
    module=Path(args.module_root); work=Path(args.work_root); outdir=ensure_dir(args.outdir)
    qc=read_tsv(work/'qc'/'fastq_qc_per_sample.tsv'); sources=read_tsv(work/'source_selection'/'benchmark_source_panel_28.tsv'); parents=read_tsv(work/'design_lock'/'locked_parent_sets.tsv'); definitions=read_tsv(work/'design_lock'/'locked_mixture_definitions_84.tsv'); design=read_tsv(work/'design_lock'/'locked_generated_library_design_560.tsv'); allocations=read_tsv(work/'design_lock'/'locked_read_allocations.tsv'); rare=read_tsv(work/'design_lock'/'locked_rare_event_schedule_735.tsv'); lock=read_json(work/'design_lock'/'analysis_design_lock.json')
    checks=[]
    def check(name,observed,expected,ok=None):
        if ok is None: ok=observed==expected
        checks.append({'name':name,'observed':observed,'expected':expected,'ok':bool(ok)}); print(f"[{'PASS' if ok else 'FAIL'}] {name}: observed={observed}; expected={expected}")
    check('QC rows',len(qc),83); check('structural QC failures',sum(str(row['structural_qc_pass']).lower()!='true' for row in qc),0)
    check('source rows',len(sources),28); check('source population counts',dict(Counter(row['population'] for row in sources)),{f'Pop-{i}':4 for i in range(1,8)})
    check('parent-set rows',len(parents),28); check('parent categories',dict(Counter(row['category'] for row in parents)),{'within_population':7,'between_population_moderate':7,'between_population_high':7,'three_population_high':7})
    check('mixture definitions',len(definitions),84); check('generated libraries',len(design),560); check('generated controls',sum(row['class_label']=='single_source_control' for row in design),140); check('generated mixtures',sum(row['class_label']=='synthetic_mixture' for row in design),420)
    by_lib=defaultdict(int)
    ranges=defaultdict(list)
    params={}
    for row in allocations:
        by_lib[row['sample_id']]+=parse_int(row['read_pairs']); source=row['source_sample_accession']; start=parse_int(row['allocation_ordinal_start']); stop=parse_int(row['allocation_ordinal_stop']); ranges[source].append((start,stop,row['sample_id'])); params[source]=(parse_int(row['permutation_modulus']),parse_int(row['eligible_physical_start']))
    check('libraries with allocation total !=6000',sum(value!=6000 for value in by_lib.values()),0)
    overlap_count=0; overflow_count=0
    for source,items in ranges.items():
        items.sort();
        for previous,current in zip(items,items[1:]):
            if current[0] < previous[1]: overlap_count+=1
        modulus,_=params[source]
        if max(stop for _,stop,_ in items)>modulus: overflow_count+=1
    check('source allocation overlaps',overlap_count,0); check('source allocation overflows',overflow_count,0)
    check('rare-event graph rows',len(rare),735); check('rare-event injection counts',dict(Counter(parse_int(row['injection_count']) for row in rare)),{1:420,2:210,4:105})
    check('lock status',lock.get('status'),'LOCKED'); check('primary kNN finite',lock.get('locked_knn_synthetic'),'integer 2..10',isinstance(lock.get('locked_knn_synthetic'),int) and 2<=lock['locked_knn_synthetic']<=10)
    canonical=[work/'preflight'/'paired_input_manifest.tsv',work/'qc'/'fastq_qc_per_sample.tsv',work/'source_selection'/'benchmark_source_panel_28.tsv',work/'sketches'/'full83_js_distance.csv',work/'sketches'/'source28_js_distance.csv',work/'design_lock'/'locked_parent_sets.tsv',work/'design_lock'/'locked_mixture_definitions_84.tsv',work/'design_lock'/'locked_generated_library_design_560.tsv',work/'design_lock'/'locked_read_allocations.tsv',work/'design_lock'/'locked_rare_event_schedule_735.tsv',work/'design_lock'/'analysis_design_lock.json']
    missing=[str(path) for path in canonical if not path.exists()]; check('missing canonical lock files',len(missing),0)
    hash_lines=[]
    for path in canonical:
        if path.exists(): hash_lines.append(f'{sha256_file(path)}  {path.relative_to(work)}')
    lock_hash=hashlib.sha256(('\n'.join(hash_lines)+'\n').encode('utf-8')).hexdigest()
    (outdir/'canonical_lock_file_sha256.txt').write_text('\n'.join(hash_lines)+f'\nMASTER_LOCK_SHA256  {lock_hash}\n',encoding='utf-8')
    status='PASS' if all(item['ok'] for item in checks) else 'FAIL'
    report={'status':status,'checks':checks,'master_lock_sha256':lock_hash,'canonical_files':[str(path.relative_to(work)) for path in canonical]}
    write_json(outdir/'preanalysis_lock_audit.json',report)
    (outdir/'PREANALYSIS_LOCK_AUDIT.txt').write_text('\n'.join([f'STATUS: {status}',f'MASTER LOCK SHA-256: {lock_hash}']+[f"[{'PASS' if x['ok'] else 'FAIL'}] {x['name']}: observed={x['observed']}; expected={x['expected']}" for x in checks])+'\n',encoding='utf-8')
    if status!='PASS': raise SystemExit('[ERROR] Pre-analysis QC/design-lock audit failed.')
    (outdir/'PREANALYSIS_QC_AND_DESIGN_LOCK_PASS.txt').write_text('PASS\n',encoding='utf-8')
    review=work/'finger_millet_preanalysis_qc_design_lock_results_archive.zip'; codezip=work/'finger_millet_preanalysis_qc_design_lock_source_archive.zip'
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for folder in ('preflight','qc','source_selection','sketches','design_lock','audit','logs'):
            add_tree(archive,work/folder,folder)
        add_tree(archive,work/'config','config'); add_tree(archive,work/'code','code'); add_tree(archive,work/'run','run')
    with zipfile.ZipFile(codezip,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        add_tree(archive,work/'config','config'); add_tree(archive,work/'code','code'); add_tree(archive,work/'run','run')
    write_tsv(work/'package_checksums.tsv',[{'file':review.name,'sha256':sha256_file(review),'bytes':review.stat().st_size},{'file':codezip.name,'sha256':sha256_file(codezip),'bytes':codezip.stat().st_size}])
    print(f'STATUS: PASS\nMASTER LOCK SHA-256: {lock_hash}\nResults archive: {review}\nSource archive: {codezip}')

if __name__=='__main__': main()
