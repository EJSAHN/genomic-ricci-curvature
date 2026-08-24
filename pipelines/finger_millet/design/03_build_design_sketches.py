# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common import ensure_dir, paired_kmer_sketch, read_tsv, write_json, write_tsv


def sketch_task(row: Dict[str,str], offset: int, count: int, k: int, dim: int, cache_path: str) -> Tuple[str,str]:
    path=Path(cache_path)
    cache_key=f"{row['r1_md5']}|{row['r2_md5']}|{offset}|{count}|{k}|{dim}"
    if path.exists():
        try:
            data=np.load(path,allow_pickle=False)
            if str(data['cache_key'].item())==cache_key:
                return row['sample_accession'],str(path)
        except Exception: pass
    signature=paired_kmer_sketch(row['r1_path'],row['r2_path'],offset,count,k,dim)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,signature=signature,cache_key=np.array(cache_key),sample_accession=np.array(row['sample_accession']),offset=np.array(offset),count=np.array(count),k=np.array(k),dim=np.array(dim))
    tmp.replace(path)
    return row['sample_accession'],str(path)


def build(manifest_path: str, cache_dir: Path, label: str, offset_field: str, count_field: str, k: int, dim: int, workers: int, outdir: Path):
    rows=read_tsv(manifest_path)
    results={}
    with ProcessPoolExecutor(max_workers=max(1,workers)) as pool:
        futures={}
        for row in rows:
            offset=int(row[offset_field]); count=int(row[count_field])
            cache=cache_dir/f"{label}_{row['run_accession']}_k{k}_d{dim}_o{offset}_n{count}.npz"
            futures[pool.submit(sketch_task,row,offset,count,k,dim,str(cache))]=row['run_accession']
        done=0
        for future in as_completed(futures):
            sample,path=future.result(); results[sample]=path; done+=1
            print(f'[{label} {done}/{len(rows)}] {futures[future]}')
    names=[row['sample_accession'] for row in rows]
    signatures=np.stack([np.load(results[name])['signature'] for name in names])
    from common import pairwise_js
    distance=pairwise_js(signatures,names)
    np.savez_compressed(outdir/f'{label}_sketches.npz',names=np.asarray(names),signatures=signatures,k=np.array(k),dim=np.array(dim))
    pd.DataFrame(distance,index=names,columns=names).to_csv(outdir/f'{label}_js_distance.csv')
    metadata=[]
    for row,name in zip(rows,names): metadata.append({**row,'sketch_cache':results[name],'kmer':k,'sketch_dimension':dim})
    write_tsv(outdir/f'{label}_sketch_manifest.tsv',metadata)
    return len(rows),float(distance[np.triu_indices(len(names),1)].min()),float(np.median(distance[np.triu_indices(len(names),1)])),float(distance[np.triu_indices(len(names),1)].max())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full_manifest',required=True)
    parser.add_argument('--source_manifest',required=True)
    parser.add_argument('--source_pass',required=True)
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--cache_dir',required=True)
    parser.add_argument('--kmer',type=int,default=17)
    parser.add_argument('--sketch',type=int,default=16384)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not Path(args.source_pass).exists(): raise SystemExit('[ERROR] Source selection PASS marker absent.')
    outdir=ensure_dir(args.outdir); cache=ensure_dir(args.cache_dir)
    full=build(args.full_manifest,cache,'full83','geometry_offset','geometry_pairs',args.kmer,args.sketch,args.workers,outdir)
    source=build(args.source_manifest,cache,'source28','design_offset','design_pairs',args.kmer,args.sketch,args.workers,outdir)
    summary={'status':'PASS','analysis_mode':'paired','kmer':args.kmer,'sketch':args.sketch,'full83':{'n':full[0],'js_min':full[1],'js_median':full[2],'js_max':full[3]},'source28':{'n':source[0],'js_min':source[1],'js_median':source[2],'js_max':source[3]},'segment_independence':'full83 geometry uses pairs 0:50000; source-design sketches use pairs 50000:80000; synthetic allocations start at 100000'}
    write_json(outdir/'design_sketch_summary.json',summary)
    (outdir/'DESIGN_SKETCH_PASS.txt').write_text('PASS\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
