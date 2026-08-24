# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, parse_int, read_tsv, stable_key, write_json, write_tsv

POPS=[f'Pop-{i}' for i in range(1,8)]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--paired_manifest',required=True)
    parser.add_argument('--panel_samples',required=True)
    parser.add_argument('--qc_table',required=True)
    parser.add_argument('--qc_pass',required=True)
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--seed',type=int,default=79152228)
    parser.add_argument('--per_population',type=int,default=4)
    parser.add_argument('--geometry_pairs',type=int,default=50000)
    parser.add_argument('--design_offset',type=int,default=50000)
    parser.add_argument('--design_pairs',type=int,default=30000)
    parser.add_argument('--synthetic_start',type=int,default=100000)
    parser.add_argument('--minimum_source_pairs',type=int,default=500000)
    args=parser.parse_args()
    if not Path(args.qc_pass).exists(): raise SystemExit('[ERROR] FASTQ QC PASS marker absent.')
    outdir=ensure_dir(args.outdir)
    paired={row['sample_accession']:row for row in read_tsv(args.paired_manifest)}
    samples=read_tsv(args.panel_samples)
    qc={row['sample_accession']:row for row in read_tsv(args.qc_table)}
    if len(samples)!=83 or len(qc)!=83: raise SystemExit('[ERROR] Expected 83 sample and QC rows.')
    failures=[row for row in qc.values() if str(row.get('structural_qc_pass','')).lower()!='true']
    if failures: raise SystemExit('[ERROR] Structural QC failures exist; source selection blocked.')

    full_geometry=[]
    for sample in samples:
        q=qc[sample['sample_accession']]; p=paired[sample['sample_accession']]
        if parse_int(q['pair_count']) < args.geometry_pairs:
            raise SystemExit(f"[ERROR] {sample['sample_accession']} has insufficient pairs for full geometry")
        full_geometry.append({**p,'geometry_offset':0,'geometry_pairs':args.geometry_pairs,'pair_count':q['pair_count']})
    full_geometry.sort(key=lambda row:int(row['panel_order']))
    write_tsv(outdir/'full_cohort_geometry_manifest_83.tsv',full_geometry)

    source_rows: List[Dict]=[]; ineligible=[]
    for pop in POPS:
        candidates=[]
        for sample in samples:
            if sample['population']!=pop: continue
            q=qc[sample['sample_accession']]
            eligible=parse_int(q['pair_count'])>=args.minimum_source_pairs
            key=stable_key(args.seed,pop,sample['sample_accession'])
            item={**sample,**paired[sample['sample_accession']], 'pair_count':q['pair_count'], 'source_selection_key':key, 'source_eligible':eligible}
            (candidates if eligible else ineligible).append(item)
        candidates.sort(key=lambda row:(row['source_selection_key'],row['sample_accession']))
        if len(candidates)<args.per_population:
            raise SystemExit(f'[ERROR] {pop}: only {len(candidates)} libraries meet the hard read-pair requirement; need {args.per_population}.')
        for rank,item in enumerate(candidates[:args.per_population],start=1):
            item.update({'source_rank_within_population':rank,'design_offset':args.design_offset,'design_pairs':args.design_pairs,'synthetic_eligible_start':args.synthetic_start})
            source_rows.append(item)
    source_rows.sort(key=lambda row:(int(row['population'].split('-')[1]),int(row['source_rank_within_population'])))
    for index,row in enumerate(source_rows,start=1): row['source_order']=index
    write_tsv(outdir/'benchmark_source_panel_28.tsv',source_rows)
    write_tsv(outdir/'source_ineligible_due_to_hard_pair_count.tsv',ineligible)
    counts=Counter(row['population'] for row in source_rows)
    status='PASS' if len(source_rows)==28 and all(counts[pop]==args.per_population for pop in POPS) else 'FAIL'
    summary={'status':status,'seed':args.seed,'source_count':len(source_rows),'per_population':dict(counts),'selection_uses_downstream_results':False,'hard_eligibility_only':'structural FASTQ PASS and pair_count >= minimum_source_pairs','minimum_source_pairs':args.minimum_source_pairs,'geometry_segment':[0,args.geometry_pairs],'design_segment':[args.design_offset,args.design_offset+args.design_pairs],'synthetic_eligible_start':args.synthetic_start}
    write_json(outdir/'source_selection_summary.json',summary)
    (outdir/'SOURCE_SELECTION_SUMMARY.txt').write_text('\n'.join(['Finger millet benchmark source selection','========================================','',f'Status: {status}',f'Sources: {len(source_rows)} (4 per population)',f'Seed: {args.seed}',f'Minimum hard pair count: {args.minimum_source_pairs:,}','', 'Selection used only population labels, a fixed SHA-256 ordering, structural FASTQ validity, and a prespecified minimum pair count. No TMS, PCA, graph, or performance result was used.'])+'\n',encoding='utf-8')
    if status!='PASS': raise SystemExit('[ERROR] Source selection failed.')
    (outdir/'SOURCE_SELECTION_PASS.txt').write_text('PASS\n',encoding='utf-8')
    print((outdir/'SOURCE_SELECTION_SUMMARY.txt').read_text(encoding='utf-8'))

if __name__=='__main__': main()
