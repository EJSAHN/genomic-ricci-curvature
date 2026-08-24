# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common import ensure_dir, iter_paired_fastq, normalize_read_id, parse_int, read_json, read_tsv, write_json, write_tsv

ADAPTER = 'AGATCGGAAGAGC'
MAX_POSITION = 200
PREFIX_LENGTH = 12


def scan_one(row: Dict[str,str], cache_dir: str, sample_pairs: int) -> Dict:
    cache_path = Path(cache_dir)/f"{row['run_accession']}.json"
    cache_key = {
        'r1_path':row['r1_path'],'r2_path':row['r2_path'],
        'r1_bytes':row['r1_bytes'],'r2_bytes':row['r2_bytes'],
        'r1_md5':row['r1_md5'],'r2_md5':row['r2_md5'],
        'sample_pairs_target':sample_pairs,
    }
    if cache_path.exists():
        try:
            cached = read_json(cache_path)
            if cached.get('cache_key') == cache_key and cached.get('structural_qc_pass') is True:
                cached['cache_reused'] = True
                return cached
        except Exception:
            pass

    expected_reads = parse_int(row.get('read_count_archive'))
    expected_pairs = max(1, expected_reads // 2)
    stride = max(1, expected_pairs // max(1, sample_pairs))

    pair_count = 0
    total_bases = [0,0]
    min_len = [10**9,10**9]
    max_len = [0,0]
    sampled_pairs = 0
    q_sum = [0,0]; q_count = [0,0]
    q_lt20 = [0,0]; q_lt30 = [0,0]
    gc_count = [0,0]; n_count = [0,0]; base_count = [0,0]
    adapter_reads = [0,0]
    prefixes = [Counter(), Counter()]
    length_sample = [Counter(), Counter()]
    pos_q_sum = np.zeros((2,MAX_POSITION), dtype=np.float64)
    pos_q_count = np.zeros((2,MAX_POSITION), dtype=np.int64)
    pos_base = {base:np.zeros((2,MAX_POSITION), dtype=np.int64) for base in 'ACGTN'}

    structural_pass = True
    error = ''
    try:
        for index, r1, r2 in iter_paired_fastq(row['r1_path'], row['r2_path']):
            pair_count += 1
            records = (r1,r2)
            for mate, record in enumerate(records):
                length = len(record.sequence)
                total_bases[mate] += length
                min_len[mate] = min(min_len[mate], length)
                max_len[mate] = max(max_len[mate], length)
            if index % stride != 0 or sampled_pairs >= sample_pairs:
                continue
            sampled_pairs += 1
            for mate, record in enumerate(records):
                seq = record.sequence.upper()
                qual = record.quality
                qs = [ord(ch)-33 for ch in qual]
                q_sum[mate] += sum(qs); q_count[mate] += len(qs)
                q_lt20[mate] += sum(value < 20 for value in qs)
                q_lt30[mate] += sum(value < 30 for value in qs)
                gc_count[mate] += seq.count('G') + seq.count('C')
                n_count[mate] += seq.count('N')
                base_count[mate] += len(seq)
                adapter_reads[mate] += int(ADAPTER in seq)
                prefixes[mate][seq[:PREFIX_LENGTH]] += 1
                length_sample[mate][str(len(seq))] += 1
                limit = min(len(seq), MAX_POSITION)
                for pos in range(limit):
                    base = seq[pos] if seq[pos] in 'ACGT' else 'N'
                    pos_base[base][mate,pos] += 1
                    pos_q_sum[mate,pos] += qs[pos]
                    pos_q_count[mate,pos] += 1
    except Exception as exc:
        structural_pass = False
        error = f'{type(exc).__name__}: {exc}'

    result = dict(row)
    result.update({
        'cache_key':cache_key,
        'cache_reused':False,
        'structural_qc_pass':bool(structural_pass),
        'structural_error':error,
        'pair_count':int(pair_count),
        'sampled_pairs':int(sampled_pairs),
        'sampling_stride':int(stride),
        'r1_total_bases':int(total_bases[0]),'r2_total_bases':int(total_bases[1]),
        'r1_min_length':0 if min_len[0] == 10**9 else int(min_len[0]),
        'r2_min_length':0 if min_len[1] == 10**9 else int(min_len[1]),
        'r1_max_length':int(max_len[0]),'r2_max_length':int(max_len[1]),
        'r1_mean_length':float(total_bases[0]/pair_count) if pair_count else float('nan'),
        'r2_mean_length':float(total_bases[1]/pair_count) if pair_count else float('nan'),
        'r1_mean_phred':float(q_sum[0]/q_count[0]) if q_count[0] else float('nan'),
        'r2_mean_phred':float(q_sum[1]/q_count[1]) if q_count[1] else float('nan'),
        'r1_q_lt20_fraction':float(q_lt20[0]/q_count[0]) if q_count[0] else float('nan'),
        'r2_q_lt20_fraction':float(q_lt20[1]/q_count[1]) if q_count[1] else float('nan'),
        'r1_q_lt30_fraction':float(q_lt30[0]/q_count[0]) if q_count[0] else float('nan'),
        'r2_q_lt30_fraction':float(q_lt30[1]/q_count[1]) if q_count[1] else float('nan'),
        'r1_gc_fraction':float(gc_count[0]/base_count[0]) if base_count[0] else float('nan'),
        'r2_gc_fraction':float(gc_count[1]/base_count[1]) if base_count[1] else float('nan'),
        'r1_n_fraction':float(n_count[0]/base_count[0]) if base_count[0] else float('nan'),
        'r2_n_fraction':float(n_count[1]/base_count[1]) if base_count[1] else float('nan'),
        'r1_adapter_read_fraction':float(adapter_reads[0]/sampled_pairs) if sampled_pairs else float('nan'),
        'r2_adapter_read_fraction':float(adapter_reads[1]/sampled_pairs) if sampled_pairs else float('nan'),
        'prefix_top':[
            [{'prefix':prefix,'count':count,'fraction':count/max(1,sampled_pairs)} for prefix,count in prefixes[mate].most_common(20)]
            for mate in (0,1)
        ],
        'length_sample': [dict(length_sample[0]),dict(length_sample[1])],
        'per_position': {
            'q_sum':pos_q_sum.tolist(),'q_count':pos_q_count.tolist(),
            **{f'base_{base}':array.tolist() for base,array in pos_base.items()},
        },
    })
    warnings = []
    if structural_pass:
        if min(result['r1_mean_phred'],result['r2_mean_phred']) < 25: warnings.append('mean_phred_below_25')
        if max(result['r1_n_fraction'],result['r2_n_fraction']) > 0.01: warnings.append('n_fraction_above_0.01')
        if max(result['r1_adapter_read_fraction'],result['r2_adapter_read_fraction']) > 0.05: warnings.append('adapter_motif_above_0.05')
        if pair_count < 50000: warnings.append('pair_count_below_geometry_requirement')
    result['reporting_warnings'] = warnings
    write_json(cache_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired_manifest', required=True)
    parser.add_argument('--input_pass', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--cache_dir', required=True)
    parser.add_argument('--sample_pairs', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not Path(args.input_pass).exists(): raise SystemExit('[ERROR] Input PASS marker absent.')
    outdir = ensure_dir(args.outdir); cache_dir = ensure_dir(args.cache_dir)
    rows = read_tsv(args.paired_manifest)
    results: List[Dict] = []
    print(f'[INFO] Full paired FASTQ scan: {len(rows)} libraries; workers={args.workers}')
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures = {pool.submit(scan_one,row,str(cache_dir),args.sample_pairs):row['run_accession'] for row in rows}
        done = 0
        for future in as_completed(futures):
            result = future.result(); results.append(result); done += 1
            print(f"[QC {done}/{len(rows)}] {result['run_accession']}: pairs={result['pair_count']:,}; pass={result['structural_qc_pass']}; warnings={','.join(result['reporting_warnings']) or '-'}")

    results.sort(key=lambda row:int(row['panel_order']))
    compact_fields = [
        'panel_order','population','sample_accession','sample_alias','run_accession','r1_path','r2_path',
        'structural_qc_pass','structural_error','pair_count','sampled_pairs','sampling_stride',
        'r1_total_bases','r2_total_bases','r1_min_length','r2_min_length','r1_max_length','r2_max_length',
        'r1_mean_length','r2_mean_length','r1_mean_phred','r2_mean_phred',
        'r1_q_lt20_fraction','r2_q_lt20_fraction','r1_q_lt30_fraction','r2_q_lt30_fraction',
        'r1_gc_fraction','r2_gc_fraction','r1_n_fraction','r2_n_fraction',
        'r1_adapter_read_fraction','r2_adapter_read_fraction','reporting_warnings','cache_reused'
    ]
    compact = []
    prefix_rows = []; position_rows = []
    for result in results:
        row = {key:result.get(key,'') for key in compact_fields}
        row['reporting_warnings'] = ';'.join(result.get('reporting_warnings',[]))
        compact.append(row)
        for mate in (0,1):
            for rank,item in enumerate(result['prefix_top'][mate],start=1):
                prefix_rows.append({'run_accession':result['run_accession'],'population':result['population'],'mate':mate+1,'rank':rank,**item})
            pp = result['per_position']; qsum=np.asarray(pp['q_sum'])[mate]; qcount=np.asarray(pp['q_count'])[mate]
            bases={base:np.asarray(pp[f'base_{base}'])[mate] for base in 'ACGTN'}
            for pos in range(MAX_POSITION):
                count=int(qcount[pos])
                if count <= 0: continue
                position_rows.append({
                    'run_accession':result['run_accession'],'population':result['population'],'mate':mate+1,'position':pos+1,
                    'n_bases':count,'mean_phred':float(qsum[pos]/count),
                    **{f'{base}_fraction':float(bases[base][pos]/count) for base in 'ACGTN'}
                })
    write_tsv(outdir/'fastq_qc_per_sample.tsv', compact, compact_fields)
    write_tsv(outdir/'prefix_profile_top20.tsv', prefix_rows)
    write_tsv(outdir/'per_position_quality_and_bases.tsv', position_rows)
    failures=[row for row in compact if str(row['structural_qc_pass']).lower() != 'true']
    write_tsv(outdir/'structural_qc_failures.tsv', failures, compact_fields)
    warnings=[row for row in compact if row['reporting_warnings']]
    write_tsv(outdir/'qc_reporting_warnings.tsv', warnings, compact_fields)
    status='PASS' if len(compact)==83 and not failures and min(int(row['pair_count']) for row in compact)>=50000 else 'FAIL'
    summary={
        'status':status,'library_count':len(compact),'structural_failure_count':len(failures),
        'warning_library_count':len(warnings),'minimum_pair_count':min(int(row['pair_count']) for row in compact),
        'maximum_pair_count':max(int(row['pair_count']) for row in compact),
        'total_pair_count':sum(int(row['pair_count']) for row in compact),
        'qc_sample_pairs_target':args.sample_pairs,
        'warning_thresholds':{'mean_phred':25,'n_fraction':0.01,'adapter_read_fraction':0.05},
        'warnings_trigger_exclusion':False,
    }
    write_json(outdir/'fastq_qc_summary.json', summary)
    lines=['Finger millet paired FASTQ pre-analysis QC','==========================================','',f"Status: {status}",f"Libraries: {len(compact)}",f"Structural failures: {len(failures)}",f"Libraries with reporting warnings: {len(warnings)}",f"Pair-count range: {summary['minimum_pair_count']:,}–{summary['maximum_pair_count']:,}",f"Total read pairs: {summary['total_pair_count']:,}",'','Reporting warnings do not exclude libraries. Only malformed/truncated/unsynchronized FASTQ or inadequate read-pair counts are hard failures.']
    (outdir/'FASTQ_QC_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    if status!='PASS': raise SystemExit('[ERROR] FASTQ structural QC failed.')
    (outdir/'FASTQ_QC_PASS.txt').write_text('PASS\n',encoding='utf-8')

if __name__=='__main__': main()
