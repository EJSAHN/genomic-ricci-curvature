# -*- coding: utf-8 -*-
"""Summarize rare-event injection performance across independent read-level replicates."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def describe_by_replicate(df:pd.DataFrame, value_cols:list[str])->pd.DataFrame:
    keys=['scenario','analysis_mode','context','injection_count']
    rep=df.groupby(keys+['replicate'],as_index=False)[value_cols].mean(numeric_only=True)
    rows=[]
    for group,sub in rep.groupby(keys,sort=True):
        row=dict(zip(keys,group)); row['n_replicates']=int(len(sub))
        for col in value_cols:
            vals=sub[col].astype(float)
            row[f'{col}_mean']=float(vals.mean()); row[f'{col}_sd']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
            row[f'{col}_min']=float(vals.min()); row[f'{col}_max']=float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def qcut_safe(series:pd.Series)->pd.Series:
    try:
        return pd.qcut(series,3,labels=['low','medium','high'],duplicates='drop')
    except Exception:
        return pd.Series(['unavailable']*len(series),index=series.index,dtype='object')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis_dir',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--read_level_master_json',required=True)
    p.add_argument('--baseline_audit_json',required=True)
    args=p.parse_args()
    ad=Path(args.analysis_dir); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    graph=pd.read_csv(ad/'graph_metrics.tsv',sep='\t')
    mix=pd.read_csv(ad/'mixture_rank_metrics.tsv',sep='\t')
    comp=pd.read_csv(ad/'comparator_metrics.tsv',sep='\t')
    edge=pd.read_csv(ad/'edge_diagnostics.tsv',sep='\t')
    read_level=json.loads(Path(args.read_level_master_json).read_text(encoding='utf-8'))
    baseline=json.loads(Path(args.baseline_audit_json).read_text(encoding='utf-8'))
    metric_cols=['prevalence','roc_auc','average_precision','ap_lift_over_prevalence','best_f1_gain_over_all_positive','mean_mixture_rank','mean_mixture_percentile','top1_is_mixture','any_mixture_top3','any_mixture_top5','mixture_recall_top3','mixture_recall_top5','mixture_precision_top3','mixture_precision_top5','chance_top1','chance_any_top3','chance_any_top5','negative_edge_fraction','orc_component_informative','tms_betweenness_spearman']
    group_summary=describe_by_replicate(graph,metric_cols)
    # Comparator summary based on replicate means to avoid pseudo-replication.
    comp_rep=comp.groupby(['scenario','analysis_mode','context','injection_count','score_name','replicate'],as_index=False)[['roc_auc','average_precision','ap_lift_over_prevalence']].mean()
    comp_rows=[]
    for keys,sub in comp_rep.groupby(['scenario','analysis_mode','context','injection_count','score_name'],sort=True):
        row=dict(zip(['scenario','analysis_mode','context','injection_count','score_name'],keys)); row['n_replicates']=int(len(sub))
        for c in ['roc_auc','average_precision','ap_lift_over_prevalence']:
            row[f'{c}_mean']=float(sub[c].mean()); row[f'{c}_sd']=float(sub[c].std(ddof=1)) if len(sub)>1 else 0.0
        comp_rows.append(row)
    comparator_summary=pd.DataFrame(comp_rows)
    # Single-injection pattern summary.
    single=mix[(mix['context']=='all_controls')&(mix['injection_count']==1)].copy()
    pattern_rep=single.groupby(['scenario','analysis_mode','pattern_id','replicate'],as_index=False)[['rank_percentile','top1','top3','top5','parent_distance_mean','negative_orc_incidence']].mean()
    pattern_rows=[]
    for keys,sub in pattern_rep.groupby(['scenario','analysis_mode','pattern_id'],sort=True):
        row=dict(zip(['scenario','analysis_mode','pattern_id'],keys)); row['n_replicates']=int(len(sub))
        for c in ['rank_percentile','top1','top3','top5','parent_distance_mean','negative_orc_incidence']:
            row[f'{c}_mean']=float(sub[c].mean()); row[f'{c}_sd']=float(sub[c].std(ddof=1)) if len(sub)>1 else 0.0
        pattern_rows.append(row)
    pattern_summary=pd.DataFrame(pattern_rows)
    # Parent distance strata, defined separately per scenario and mode.
    strata_parts=[]
    for (scenario,mode),sub in single.groupby(['scenario','analysis_mode'],sort=True):
        sub=sub.copy(); sub['parent_distance_stratum']=qcut_safe(sub['parent_distance_mean'])
        strata_parts.append(sub)
    strata=pd.concat(strata_parts,ignore_index=True) if strata_parts else pd.DataFrame()
    strata_summary=(strata.groupby(['scenario','analysis_mode','parent_distance_stratum'],observed=True,as_index=False)
                    .agg(n=('sample_id','size'),rank_percentile_mean=('rank_percentile','mean'),top1_rate=('top1','mean'),top3_rate=('top3','mean'),top5_rate=('top5','mean'),parent_distance_mean=('parent_distance_mean','mean'))) if len(strata) else pd.DataFrame()
    # Correlations between detectability and design attributes for single injections.
    corr_rows=[]
    for (scenario,mode),sub in single.groupby(['scenario','analysis_mode'],sort=True):
        for x in ['parent_distance_mean','actual_minor_fraction','actual_entropy_norm','mixture_to_parent_mean']:
            valid=sub[[x,'rank_percentile']].dropna()
            if len(valid)>=3 and valid[x].nunique()>1 and valid['rank_percentile'].nunique()>1:
                rho,pv=spearmanr(valid[x],valid['rank_percentile'])
            else: rho,pv=np.nan,np.nan
            corr_rows.append({'scenario':scenario,'analysis_mode':mode,'predictor':x,'n':len(valid),'spearman_rho':rho,'spearman_p':pv})
    correlation_summary=pd.DataFrame(corr_rows)
    # Parent context paired comparison for single injection.
    context_key=['scenario','replicate','analysis_mode','sample_id']
    included=single[['scenario','replicate','analysis_mode','sample_id','rank_percentile','top1','top3','top5']].rename(columns={c:f'{c}_all_controls' for c in ['rank_percentile','top1','top3','top5']})
    excluded=mix[(mix['context']=='exclude_parents')&(mix['injection_count']==1)][context_key+['rank_percentile','top1','top3','top5']].rename(columns={c:f'{c}_exclude_parents' for c in ['rank_percentile','top1','top3','top5']})
    context_pairs=included.merge(excluded,on=context_key,how='inner',validate='one_to_one')
    for c in ['rank_percentile','top1','top3','top5']:
        context_pairs[f'{c}_difference_exclude_minus_all']=context_pairs[f'{c}_exclude_parents']-context_pairs[f'{c}_all_controls']
    context_summary=context_pairs.groupby(['scenario','analysis_mode'],as_index=False).agg(n=('sample_id','size'),rank_percentile_difference_mean=('rank_percentile_difference_exclude_minus_all','mean'),top1_difference=('top1_difference_exclude_minus_all','mean'),top3_difference=('top3_difference_exclude_minus_all','mean'),top5_difference=('top5_difference_exclude_minus_all','mean'))
    # Main interpretation target.
    target=group_summary[(group_summary['scenario']=='primary')&(group_summary['analysis_mode']=='paired')&(group_summary['context']=='all_controls')&(group_summary['injection_count']==1)]
    if len(target)!=1: raise SystemExit('[ERR] Missing primary paired single-injection summary')
    t=target.iloc[0]
    auc=float(t['roc_auc_mean']); orc=float(t['orc_component_informative_mean']); top3=float(t['any_mixture_top3_mean']); chance3=float(t['chance_any_top3_mean'])
    if auc>=0.70 and orc>=0.50:
        status='SUPPORTED'
    elif auc>=0.60:
        status='LIMITED'
    else:
        status='NOT_SUPPORTED'
    baseline_obs=baseline.get('observed',baseline)
    payload={
        'status':'COMPLETE','rare_event_detection_status':status,
        'primary_endpoint':{
            'scenario':'primary','analysis_mode':'paired','context':'all_controls','injection_count':1,
            'roc_auc_mean':auc,'roc_auc_sd':float(t['roc_auc_sd']),
            'top1_capture_rate':float(t['top1_is_mixture_mean']),
            'top3_capture_rate':top3,'top5_capture_rate':float(t['any_mixture_top5_mean']),
            'chance_top1':float(t['chance_top1_mean']),'chance_any_top3':chance3,'chance_any_top5':float(t['chance_any_top5_mean']),
            'mean_rank_percentile':float(t['mean_mixture_percentile_mean']),
            'negative_edge_fraction_mean':float(t['negative_edge_fraction_mean']),
            'orc_informative_graph_fraction':orc,
            'tms_betweenness_spearman_mean':float(t['tms_betweenness_spearman_mean']),
        },
        'batch_read_level_primary':read_level.get('primary_metrics',{}),
        'submission_baseline_synthetic':baseline_obs.get('synthetic',{}),
        'n_graph_runs':int(len(graph)),'n_mixture_rows':int(len(mix)),
    }
    (out/'rare_event_master_metrics.json').write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    group_summary.to_csv(out/'rare_event_group_summary.tsv',sep='\t',index=False)
    comparator_summary.to_csv(out/'rare_event_comparator_summary.tsv',sep='\t',index=False)
    pattern_summary.to_csv(out/'rare_event_pattern_summary.tsv',sep='\t',index=False)
    strata_summary.to_csv(out/'rare_event_parent_distance_strata.tsv',sep='\t',index=False)
    correlation_summary.to_csv(out/'rare_event_correlations.tsv',sep='\t',index=False)
    context_pairs.to_csv(out/'rare_event_parent_context_pairs.tsv',sep='\t',index=False)
    context_summary.to_csv(out/'rare_event_parent_context_summary.tsv',sep='\t',index=False)
    with pd.ExcelWriter(out/'rare_event_master_metrics.xlsx',engine='openpyxl') as writer:
        group_summary.to_excel(writer,sheet_name='group_summary',index=False)
        comparator_summary.to_excel(writer,sheet_name='comparators',index=False)
        pattern_summary.to_excel(writer,sheet_name='patterns',index=False)
        strata_summary.to_excel(writer,sheet_name='parent_distance',index=False)
        correlation_summary.to_excel(writer,sheet_name='correlations',index=False)
        context_summary.to_excel(writer,sheet_name='parent_context',index=False)
        graph.to_excel(writer,sheet_name='graph_metrics',index=False)
        mix.to_excel(writer,sheet_name='mixture_ranks',index=False)
    text=f'''Rare-event paired-read injection analysis\n========================================\n\nPrimary endpoint: primary scenario, paired mode, all controls, one injected mixture\nReplicates: 5\n\nROC AUC: {auc:.3f} +/- {float(t['roc_auc_sd']):.3f}\nTop-1 capture: {float(t['top1_is_mixture_mean']):.3f} (chance {float(t['chance_top1_mean']):.3f})\nAny mixture in Top-3: {top3:.3f} (chance {chance3:.3f})\nAny mixture in Top-5: {float(t['any_mixture_top5_mean']):.3f} (chance {float(t['chance_any_top5_mean']):.3f})\nMean mixture rank percentile: {float(t['mean_mixture_percentile_mean']):.3f}\nNegative-edge fraction: {float(t['negative_edge_fraction_mean']):.6f}\nGraphs with informative negative-ORC variation: {orc:.3f}\nMean TMS-betweenness Spearman rho: {float(t['tms_betweenness_spearman_mean']):.3f}\n\nRead-level rare-event detection status: {status}\n\nPASS/COMPLETE refers to computational integrity, not evidence of detection performance.\n'''
    (out/'RARE_EVENT_RESULTS_SUMMARY.txt').write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__': main()
