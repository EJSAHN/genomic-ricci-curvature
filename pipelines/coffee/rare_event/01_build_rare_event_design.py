# -*- coding: utf-8 -*-
"""Build deterministic rare-event graph designs from completed paired-read mixtures."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from rare_event_common import stable_seed

def split_list(text): return [x for x in str(text).split(',') if x]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--truth_manifest',required=True); p.add_argument('--schedule',required=True); p.add_argument('--outdir',required=True); p.add_argument('--replicates',default='1,2,3,4,5'); p.add_argument('--seed',type=int,default=43000); args=p.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    truth=pd.read_csv(args.truth_manifest,sep='\t'); schedule=pd.read_csv(args.schedule,sep='\t'); reps=[int(x) for x in args.replicates.split(',') if x.strip()]
    rows=[]
    for sched in schedule.itertuples(index=False):
        scenario=str(sched.scenario); mode=str(sched.analysis_mode); context=str(sched.context); counts=[int(x) for x in str(sched.injection_counts).split(',') if x]
        if context not in {'all_controls','exclude_parents'}: raise SystemExit(f'[ERR] Unknown context: {context}')
        for rep in reps:
            group=truth[(truth['scenario']==scenario)&(truth['replicate']==rep)].copy(); controls=group[group['class_label']=='single_source_control'].sort_values('sample_id'); mixtures=group[group['class_label']=='synthetic_mixture'].sort_values('sample_id')
            if len(mixtures)!=24: raise SystemExit(f'[ERR] Expected 24 mixtures: {scenario} rep {rep}')
            source_to_control={str(r.parents):str(r.sample_id) for r in controls.itertuples(index=False)}; base_controls=controls['sample_id'].astype(str).tolist(); base_mixtures=mixtures['sample_id'].astype(str).tolist(); lookup=mixtures.set_index('sample_id')
            for m in counts:
                if len(base_mixtures)%m!=0: raise SystemExit(f'[ERR] Mixtures not divisible by {m}')
                rng=np.random.default_rng(stable_seed(args.seed,scenario,rep,m)); shuffled=np.array(base_mixtures,dtype=object); rng.shuffle(shuffled); groups=[shuffled[i:i+m].tolist() for i in range(0,len(shuffled),m)]
                for idx,injected in enumerate(groups,1):
                    parents=[]; patterns=[]
                    for sample in injected:
                        rec=lookup.loc[sample]; parents.extend(split_list(rec['parents'])); patterns.append(str(rec['pattern_id']))
                    removed=sorted({source_to_control[x] for x in parents if x in source_to_control}) if context=='exclude_parents' else []
                    controls_used=[x for x in base_controls if x not in set(removed)]; graph_id=f'{scenario}_{mode}_R{rep:02d}_M{m}_{context}_G{idx:02d}'; n=len(controls_used)+len(injected)
                    rows.append({'graph_id':graph_id,'scenario':scenario,'replicate':rep,'analysis_mode':mode,'injection_count':m,'context':context,'design_index':idx,'control_ids':';'.join(controls_used),'mixture_ids':';'.join(injected),'removed_parent_controls':';'.join(removed),'parent_sources':';'.join(sorted(set(parents))),'pattern_ids':';'.join(patterns),'n_controls':len(controls_used),'n_mixtures':len(injected),'n_nodes':n,'prevalence':len(injected)/n})
    design=pd.DataFrame(rows).sort_values(['scenario','analysis_mode','replicate','context','injection_count','design_index']); design.to_csv(out/'rare_event_design.tsv',sep='\t',index=False)
    coverage=[]
    for row in design.itertuples(index=False):
        for sample in str(row.mixture_ids).split(';'): coverage.append({'scenario':row.scenario,'analysis_mode':row.analysis_mode,'replicate':row.replicate,'context':row.context,'injection_count':row.injection_count,'sample_id':sample})
    pd.DataFrame(coverage).to_csv(out/'mixture_design_coverage.tsv',sep='\t',index=False)
    summary={'status':'COMPLETE','seed':args.seed,'n_designs':int(len(design)),'designs_by_schedule':design.groupby(['scenario','analysis_mode','context','injection_count']).size().rename('n').reset_index().to_dict('records')}
    (out/'design_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True),encoding='utf-8'); print(f'[DONE] Graph designs: {len(design)}')
if __name__=='__main__': main()
