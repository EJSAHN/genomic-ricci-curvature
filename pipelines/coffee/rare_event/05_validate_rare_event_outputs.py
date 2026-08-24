# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--design_dir',required=True); p.add_argument('--analysis_dir',required=True); p.add_argument('--summary_dir',required=True); p.add_argument('--outdir',required=True); args=p.parse_args()
    dd=Path(args.design_dir); ad=Path(args.analysis_dir); sd=Path(args.summary_dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    checks=[]
    def add(name,ok,obs,exp): checks.append({'name':name,'ok':bool(ok),'observed':obs,'expected':exp}); print(f"[{'PASS' if ok else 'FAIL'}] {name}: observed={obs}; expected={exp}")
    design=pd.read_csv(dd/'rare_event_design.tsv',sep='\t'); cov=pd.read_csv(dd/'mixture_design_coverage.tsv',sep='\t'); graph=pd.read_csv(ad/'graph_metrics.tsv',sep='\t'); mix=pd.read_csv(ad/'mixture_rank_metrics.tsv',sep='\t'); comp=pd.read_csv(ad/'comparator_metrics.tsv',sep='\t'); master=json.loads((sd/'rare_event_master_metrics.json').read_text(encoding='utf-8'))
    add('design row count',len(design)==210,len(design),210)
    add('graph run count',len(graph)==len(design),len(graph),len(design))
    add('analysis mode set',set(graph['analysis_mode'])=={'paired'},sorted(set(graph['analysis_mode'])),['paired'])
    add('scenario set',set(graph['scenario'])=={'primary'},sorted(set(graph['scenario'])),['primary'])
    add('injection counts',set(graph['injection_count'])=={1,2,4},sorted(set(graph['injection_count'])),[1,2,4])
    add('context set',set(graph['context'])=={'all_controls'},sorted(set(graph['context'])),['all_controls'])
    add('replicate set',set(graph['replicate'])==set(range(1,6)),sorted(set(graph['replicate'])),[1,2,3,4,5])
    add('graph connectivity',bool(graph['graph_connected'].all()),int(graph['graph_connected'].sum()),len(graph))
    finite_cols=['roc_auc','average_precision','ap_lift_over_prevalence','mean_mixture_rank','mean_mixture_percentile','negative_edge_fraction']
    missing={c:int(graph[c].isna().sum()) for c in finite_cols if graph[c].isna().any()}; add('finite graph metrics',not missing,missing,{})
    add('mixture row count',len(mix)==360,len(mix),360)
    add('comparator row count',len(comp)==len(graph)*7,len(comp),len(graph)*7)
    counts=cov.groupby(['scenario','analysis_mode','replicate','context','injection_count','sample_id']).size(); add('balanced mixture coverage',bool((counts==1).all()),sorted(counts.unique().tolist()),[1])
    add('master metrics complete',master.get('status')=='COMPLETE',master.get('status'),'COMPLETE')
    primary=master.get('primary_endpoint',{}); add('primary endpoint present',all(k in primary for k in ['roc_auc_mean','top1_capture_rate','top3_capture_rate','negative_edge_fraction_mean']),sorted(primary.keys()),'required keys')
    status='PASS' if all(x['ok'] for x in checks) else 'FAIL'; payload={'status':status,'checks':checks,'performance_status':master.get('rare_event_detection_status'),'note':'PASS indicates computational integrity only; performance is reported separately.'}
    (out/'rare_event_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); lines=[f'STATUS: {status}',f"PERFORMANCE STATUS: {payload['performance_status']}",'']+[f"[{'PASS' if x['ok'] else 'FAIL'}] {x['name']}: observed={x['observed']}; expected={x['expected']}" for x in checks]+['','PASS indicates computational integrity only; it does not imply successful detection.']; (out/'rare_event_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if status!='PASS': raise SystemExit(1)
    (out/'RARE_EVENT_AUDIT_PASS.txt').write_text('PASS\n',encoding='utf-8'); print(f'STATUS: {status}')
if __name__=='__main__': main()
