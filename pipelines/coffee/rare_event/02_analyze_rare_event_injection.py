# -*- coding: utf-8 -*-
"""Recompute graph geometry after injecting a small number of read-level mixtures."""
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from rare_event_common import compute_node_scores, evaluate_score

SCORES={
    'tms':'tms',
    'betweenness':'betweenness',
    'negative_orc':'negative_orc_incidence',
    'mean_incident_distance':'mean_incident_distance',
    'pca_distance':'pca_distance',
    'local_outlier_factor':'lof_score',
    'raw_betweenness_plus_negative_orc':'raw_sum',
}

def semis(text): return [x for x in str(text).split(';') if x]
def commas(text): return [x for x in str(text).split(',') if x]

def any_topk_chance(n:int,m:int,k:int)->float:
    k=min(k,n)
    if m<=0:return 0.0
    if n-m<k:return 1.0
    return 1.0-math.comb(n-m,k)/math.comb(n,k)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--design',required=True)
    p.add_argument('--truth_manifest',required=True)
    p.add_argument('--source_analysis_dir',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--knn',type=int,default=4)
    p.add_argument('--alpha',type=float,default=0.5)
    args=p.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    design=pd.read_csv(args.design,sep='\t')
    truth=pd.read_csv(args.truth_manifest,sep='\t')
    tlookup=truth.set_index(['scenario','replicate','sample_id'])
    source=Path(args.source_analysis_dir)
    dist_cache:Dict[tuple,pd.DataFrame]={}
    graph_rows=[]; mix_rows=[]; comparator_rows=[]; edge_diag_rows=[]
    for _idx,drow in enumerate(design.itertuples(index=False),1):
        if _idx == 1 or _idx % 10 == 0 or _idx == len(design):
            print(f'[PROGRESS] graph {_idx}/{len(design)}: {drow.graph_id}', flush=True)
        controls=semis(drow.control_ids); mixtures=semis(drow.mixture_ids); names=controls+mixtures
        mode=str(drow.analysis_mode)
        key=(drow.scenario,int(drow.replicate),mode)
        if key not in dist_cache:
            f=source/'runs'/drow.scenario/f'rep_{int(drow.replicate):02d}'/mode/'js_distance.csv'
            dist_cache[key]=pd.read_csv(f,index_col=0)
        full=dist_cache[key]
        missing=[x for x in names if x not in full.index]
        if missing: raise SystemExit(f'[ERR] Missing samples in {key}: {missing}')
        distance=full.loc[names,names]
        node,edge,graph=compute_node_scores(distance,k=args.knn,alpha=args.alpha)
        node['raw_sum']=node['betweenness']+node['negative_orc_incidence']
        node['is_mixture']=node['sample_id'].isin(mixtures).astype(int)
        y=node['is_mixture'].to_numpy(int)
        n=len(node); m=int(y.sum()); prevalence=m/n
        # Stable exact ranks and average ranks for ties.
        ordered=node.sort_values(['tms','sample_id'],ascending=[False,True]).reset_index(drop=True)
        exact_rank={s:i+1 for i,s in enumerate(ordered['sample_id'])}
        avg_ranks=rankdata(-node['tms'].to_numpy(float),method='average')
        avg_rank={s:float(r) for s,r in zip(node['sample_id'],avg_ranks)}
        mix_ranks=[exact_rank[s] for s in mixtures]
        mix_avg=[avg_rank[s] for s in mixtures]
        top1_ids=set(ordered.head(1)['sample_id']); top3_ids=set(ordered.head(min(3,n))['sample_id']); top5_ids=set(ordered.head(min(5,n))['sample_id'])
        hits1=len(top1_ids.intersection(mixtures)); hits3=len(top3_ids.intersection(mixtures)); hits5=len(top5_ids.intersection(mixtures))
        tms_metrics=evaluate_score(y,node['tms'].to_numpy(float))
        all_pos_f1=2*m/(2*m+(n-m))
        neg_count=int((edge['orc']<0).sum()) if len(edge) else 0
        neg_frac=neg_count/len(edge) if len(edge) else 0.0
        neg_var=float(node['negative_orc_incidence'].var(ddof=0))
        try:
            rho_tb=float(spearmanr(node['tms'],node['betweenness']).statistic)
        except Exception:
            rho_tb=float('nan')
        graph_rows.append({
            'graph_id':drow.graph_id,'scenario':drow.scenario,'replicate':int(drow.replicate),
            'analysis_mode':mode,'context':drow.context,'injection_count':int(drow.injection_count),
            'design_index':int(drow.design_index),'n_controls':int(drow.n_controls),'n_mixtures':m,
            'n_nodes':n,'prevalence':prevalence,'graph_connected':bool(__import__('networkx').is_connected(graph)),
            'graph_components':int(__import__('networkx').number_connected_components(graph)),'graph_edges':int(graph.number_of_edges()),
            'roc_auc':tms_metrics['roc_auc'],'average_precision':tms_metrics['average_precision'],
            'ap_lift_over_prevalence':tms_metrics['average_precision']/prevalence if prevalence>0 else float('nan'),
            'best_f1':tms_metrics['best_f1'],'all_positive_f1':all_pos_f1,
            'best_f1_gain_over_all_positive':tms_metrics['best_f1']-all_pos_f1,
            'mean_mixture_rank':float(np.mean(mix_ranks)),'median_mixture_rank':float(np.median(mix_ranks)),
            'mean_mixture_average_rank':float(np.mean(mix_avg)),
            'mean_mixture_percentile':float(np.mean([1-(r-1)/(n-1) for r in mix_avg])) if n>1 else 1.0,
            'top1_is_mixture':int(hits1>0),'any_mixture_top3':int(hits3>0),'any_mixture_top5':int(hits5>0),
            'mixture_recall_top3':hits3/m,'mixture_recall_top5':hits5/m,
            'mixture_precision_top3':hits3/min(3,n),'mixture_precision_top5':hits5/min(5,n),
            'chance_top1':prevalence,'chance_any_top3':any_topk_chance(n,m,3),'chance_any_top5':any_topk_chance(n,m,5),
            'chance_recall_top3':min(3,n)/n,'chance_recall_top5':min(5,n)/n,
            'negative_edge_count':neg_count,'negative_edge_fraction':neg_frac,
            'negative_orc_node_variance':neg_var,'negative_orc_nonzero_nodes':int((node['negative_orc_incidence']>0).sum()),
            'orc_component_informative':int(neg_var>1e-12),'tms_betweenness_spearman':rho_tb,
        })
        for score_name,col in SCORES.items():
            metrics=evaluate_score(y,node[col].to_numpy(float))
            comparator_rows.append({'graph_id':drow.graph_id,'scenario':drow.scenario,'replicate':int(drow.replicate),'analysis_mode':mode,'context':drow.context,'injection_count':int(drow.injection_count),'score_name':score_name,'prevalence':prevalence,**metrics,'ap_lift_over_prevalence':metrics['average_precision']/prevalence if prevalence>0 else float('nan')})
        # Parent-control map from the full truth group, including controls removed from a parent-excluded graph.
        tg=truth[(truth['scenario']==drow.scenario)&(truth['replicate']==int(drow.replicate))]
        source_to_control={str(r.parents):str(r.sample_id) for r in tg[tg['class_label']=='single_source_control'].itertuples(index=False)}
        for mix in mixtures:
            rec=tlookup.loc[(drow.scenario,int(drow.replicate),mix)]
            parent_sources=commas(rec['parents']); parent_controls=[source_to_control[x] for x in parent_sources if x in source_to_control]
            pair_d=[]
            for i in range(len(parent_controls)):
                for j in range(i+1,len(parent_controls)):
                    pair_d.append(float(full.loc[parent_controls[i],parent_controls[j]]))
            mix_to_parent=[float(full.loc[mix,x]) for x in parent_controls]
            nr=node.loc[node['sample_id']==mix].iloc[0]
            rank=exact_rank[mix]; arank=avg_rank[mix]
            mix_rows.append({
                'graph_id':drow.graph_id,'scenario':drow.scenario,'replicate':int(drow.replicate),'analysis_mode':mode,
                'context':drow.context,'injection_count':int(drow.injection_count),'sample_id':mix,
                'pattern_id':str(rec['pattern_id']),'n_parents':int(rec['n_parents']),'parents':str(rec['parents']),
                'actual_entropy_norm':float(rec['actual_entropy_norm']),'actual_minor_fraction':float(rec['actual_minor_fraction']),
                'rank_exact':rank,'rank_average':arank,'rank_percentile':1-(arank-1)/(n-1) if n>1 else 1.0,
                'top1':int(rank<=1),'top3':int(rank<=3),'top5':int(rank<=5),
                'tms':float(nr['tms']),'betweenness':float(nr['betweenness']),'negative_orc_incidence':float(nr['negative_orc_incidence']),
                'mean_incident_distance':float(nr['mean_incident_distance']),'pca_distance':float(nr['pca_distance']),'lof_score':float(nr['lof_score']),
                'parent_controls_present':sum(x in names for x in parent_controls),'parent_controls_total':len(parent_controls),
                'parent_distance_min':float(np.min(pair_d)) if pair_d else float('nan'),
                'parent_distance_mean':float(np.mean(pair_d)) if pair_d else float('nan'),
                'parent_distance_max':float(np.max(pair_d)) if pair_d else float('nan'),
                'mixture_to_parent_min':float(np.min(mix_to_parent)) if mix_to_parent else float('nan'),
                'mixture_to_parent_mean':float(np.mean(mix_to_parent)) if mix_to_parent else float('nan'),
                'mixture_to_parent_max':float(np.max(mix_to_parent)) if mix_to_parent else float('nan'),
            })
        edge_diag_rows.append({'graph_id':drow.graph_id,'scenario':drow.scenario,'replicate':int(drow.replicate),'analysis_mode':mode,'context':drow.context,'injection_count':int(drow.injection_count),'edge_count':len(edge),'negative_edge_count':neg_count,'negative_edge_fraction':neg_frac,'orc_min':float(edge['orc'].min()) if len(edge) else float('nan'),'orc_median':float(edge['orc'].median()) if len(edge) else float('nan'),'orc_max':float(edge['orc'].max()) if len(edge) else float('nan')})
    pd.DataFrame(graph_rows).to_csv(out/'graph_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(mix_rows).to_csv(out/'mixture_rank_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(comparator_rows).to_csv(out/'comparator_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(edge_diag_rows).to_csv(out/'edge_diagnostics.tsv',sep='\t',index=False)
    print(f'[DONE] Graph runs: {len(graph_rows)}')
    print(f'[DONE] Mixture-level rows: {len(mix_rows)}')

if __name__=='__main__': main()
