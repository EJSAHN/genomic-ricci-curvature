# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from common import coprime_multiplier, ensure_dir, largest_remainder_counts, normalized_entropy, read_tsv, stable_key, stable_seed, write_json, write_tsv

POPS=[f'Pop-{i}' for i in range(1,8)]
PAIR_PATTERNS=[('P2_50_50',[0.50,0.50]),('P2_75_25',[0.75,0.25]),('P2_90_10',[0.90,0.10])]
TRIPLE_PATTERNS=[('P3_34_33_33',[0.34,0.33,0.33]),('P3_50_25_25',[0.50,0.25,0.25]),('P3_60_30_10',[0.60,0.30,0.10])]


def symmetric_knn_graph(distance: pd.DataFrame, k: int) -> nx.Graph:
    names=distance.index.tolist(); graph=nx.Graph(); graph.add_nodes_from(names)
    values=distance.to_numpy(float)
    for i,name in enumerate(names):
        order=np.argsort(values[i],kind='stable')
        neighbors=[j for j in order if j!=i][:k]
        for j in neighbors:
            other=names[j]; graph.add_edge(name,other,weight=float(values[i,j]))
    return graph


def lock_knn(distance: pd.DataFrame, candidates: Sequence[int]) -> Tuple[int,List[Dict]]:
    rows=[]; chosen=None
    for k in candidates:
        graph=symmetric_knn_graph(distance,k)
        connected=nx.is_connected(graph)
        rows.append({'k':k,'connected':connected,'components':nx.number_connected_components(graph),'edges':graph.number_of_edges(),'mean_degree':float(np.mean([degree for _,degree in graph.degree()]))})
        if connected and chosen is None: chosen=k
    if chosen is None: raise SystemExit(f'[ERROR] No connected graph in candidate k range {list(candidates)}')
    return chosen,rows


def pair_distance(pair, distance): return float(distance.loc[pair[0],pair[1]])

def choose_anchor_pairs(source_rows, distance, mode, seed, usage, chosen_pairs):
    by_id={row['sample_accession']:row for row in source_rows}
    all_pairs=[]
    ids=list(by_id)
    for a,b in itertools.combinations(ids,2):
        if by_id[a]['population']==by_id[b]['population']: continue
        all_pairs.append((a,b,pair_distance((a,b),distance)))
    d=np.asarray([item[2] for item in all_pairs]); median=float(np.median(d)); q35=float(np.quantile(d,0.35)); q65=float(np.quantile(d,0.65)); q85=float(np.quantile(d,0.85))
    result=[]
    for pop in POPS:
        candidates=[item for item in all_pairs if pop in {by_id[item[0]]['population'],by_id[item[1]]['population']} and tuple(sorted(item[:2])) not in chosen_pairs]
        fallback=False
        if mode=='moderate':
            filtered=[item for item in candidates if q35<=item[2]<=q65]
            if not filtered: filtered=candidates; fallback=True
            def metric(item): return abs(item[2]-median)
        else:
            filtered=[item for item in candidates if item[2]>=q85]
            if not filtered: filtered=candidates; fallback=True
            def metric(item): return -item[2]
        filtered.sort(key=lambda item:(usage[item[0]]+usage[item[1]],max(usage[item[0]],usage[item[1]]),metric(item),stable_key(seed,mode,pop,*sorted(item[:2]))))
        if not filtered: raise SystemExit(f'[ERROR] No {mode} pair candidate for {pop}')
        a,b,dist=filtered[0]; pair=tuple(sorted((a,b))); chosen_pairs.add(pair); usage[a]+=1; usage[b]+=1
        result.append({'anchor_population':pop,'parents':pair,'distance':dist,'selection_fallback':fallback,'distance_band':{'median':median,'q35':q35,'q65':q65,'q85':q85}})
    return result


def choose_high_triples(source_rows,distance,seed,usage):
    by_id={row['sample_accession']:row for row in source_rows}; ids=list(by_id)
    triples=[]
    for triple in itertools.combinations(ids,3):
        if len({by_id[x]['population'] for x in triple})<3: continue
        ds=[float(distance.loc[a,b]) for a,b in itertools.combinations(triple,2)]
        triples.append((tuple(sorted(triple)),float(np.mean(ds)),float(np.min(ds))))
    means=np.asarray([x[1] for x in triples]); threshold=float(np.quantile(means,0.85)); chosen=set(); result=[]
    for pop in POPS:
        candidates=[item for item in triples if pop in {by_id[x]['population'] for x in item[0]} and item[0] not in chosen]
        filtered=[item for item in candidates if item[1]>=threshold]
        fallback=False
        if not filtered: filtered=candidates; fallback=True
        filtered.sort(key=lambda item:(sum(usage[x] for x in item[0]),max(usage[x] for x in item[0]),-item[1],-item[2],stable_key(seed,'triple',pop,*item[0])))
        triple,mean_d,min_d=filtered[0]; chosen.add(triple)
        for x in triple: usage[x]+=1
        result.append({'anchor_population':pop,'parents':triple,'mean_distance':mean_d,'min_distance':min_d,'selection_fallback':fallback,'top15_mean_threshold':threshold})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full_distance',required=True)
    parser.add_argument('--source_distance',required=True)
    parser.add_argument('--source_manifest',required=True)
    parser.add_argument('--qc_table',required=True)
    parser.add_argument('--sketch_pass',required=True)
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--seed',type=int,default=79152284)
    parser.add_argument('--replicates',type=int,default=5)
    parser.add_argument('--read_pairs',type=int,default=6000)
    parser.add_argument('--synthetic_start',type=int,default=100000)
    parser.add_argument('--alpha',type=float,default=0.5)
    args=parser.parse_args()
    if not Path(args.sketch_pass).exists(): raise SystemExit('[ERROR] Design sketch PASS marker absent.')
    outdir=ensure_dir(args.outdir)
    full=pd.read_csv(args.full_distance,index_col=0); source_dist=pd.read_csv(args.source_distance,index_col=0)
    sources=read_tsv(args.source_manifest); by_id={row['sample_accession']:row for row in sources}; qc={row['sample_accession']:row for row in read_tsv(args.qc_table)}
    full_k,full_sweep=lock_knn(full,range(3,13)); source_k,source_sweep=lock_knn(source_dist,range(2,11))
    graph_rows=[{'graph_context':'full_cohort_83',**row} for row in full_sweep]+[{'graph_context':'synthetic_control_panel_28',**row} for row in source_sweep]
    write_tsv(outdir/'knn_connectivity_lock_sweep.tsv',graph_rows)

    usage=Counter(); chosen_pairs=set(); base_sets=[]
    for pop in POPS:
        ids=sorted([row['sample_accession'] for row in sources if row['population']==pop])
        pairs=list(itertools.combinations(ids,2)); pairs.sort(key=lambda pair:stable_key(args.seed,'within',pop,*pair)); pair=tuple(sorted(pairs[0])); chosen_pairs.add(pair)
        usage[pair[0]]+=1; usage[pair[1]]+=1
        base_sets.append({'base_set_id':f'W_{pop.replace("Pop-","")}','category':'within_population','anchor_population':pop,'parents':pair,'distance_summary':pair_distance(pair,source_dist),'selection_fallback':False})
    moderate=choose_anchor_pairs(sources,source_dist,'moderate',args.seed,usage,chosen_pairs)
    for index,item in enumerate(moderate,1): base_sets.append({'base_set_id':f'M_{index:02d}','category':'between_population_moderate','anchor_population':item['anchor_population'],'parents':item['parents'],'distance_summary':item['distance'],'selection_fallback':item['selection_fallback']})
    high=choose_anchor_pairs(sources,source_dist,'high',args.seed,usage,chosen_pairs)
    for index,item in enumerate(high,1): base_sets.append({'base_set_id':f'H_{index:02d}','category':'between_population_high','anchor_population':item['anchor_population'],'parents':item['parents'],'distance_summary':item['distance'],'selection_fallback':item['selection_fallback']})
    triples=choose_high_triples(sources,source_dist,args.seed,usage)
    for index,item in enumerate(triples,1): base_sets.append({'base_set_id':f'T_{index:02d}','category':'three_population_high','anchor_population':item['anchor_population'],'parents':item['parents'],'distance_summary':item['mean_distance'],'min_pair_distance':item['min_distance'],'selection_fallback':item['selection_fallback']})
    base_rows=[]
    for item in base_sets:
        parents=list(item['parents']); base_rows.append({**{k:v for k,v in item.items() if k!='parents'},'parents':';'.join(parents),'parent_populations':';'.join(by_id[x]['population'] for x in parents),'parent_runs':';'.join(by_id[x]['run_accession'] for x in parents)})
    write_tsv(outdir/'locked_parent_sets.tsv',base_rows)

    definitions=[]
    for base in base_sets:
        patterns=TRIPLE_PATTERNS if len(base['parents'])==3 else PAIR_PATTERNS
        for pattern_id,weights in patterns:
            definitions.append({'mixture_definition_id':f"{base['base_set_id']}_{pattern_id}",'base_set_id':base['base_set_id'],'category':base['category'],'anchor_population':base['anchor_population'],'n_parents':len(base['parents']),'parents_canonical':';'.join(base['parents']),'pattern_id':pattern_id,'weights_canonical':';'.join(f'{x:.6f}' for x in weights),'target_entropy_norm':normalized_entropy(weights),'target_minor_fraction':min(weights),'distance_summary':base['distance_summary']})
    write_tsv(outdir/'locked_mixture_definitions_84.tsv',definitions)

    design_rows=[]; allocation_rows=[]; source_cursors=Counter(); source_params={}
    for sample_id,row in by_id.items():
        total_pairs=int(qc[sample_id]['pair_count']); modulus=total_pairs-args.synthetic_start
        if modulus<=0: raise SystemExit(f'[ERROR] {sample_id}: no synthetic eligible read pairs')
        seed=stable_seed(args.seed,'permutation',sample_id); a=coprime_multiplier(modulus,seed); b=stable_seed(args.seed,'offset',sample_id)%modulus
        source_params[sample_id]={'eligible_start':args.synthetic_start,'modulus':modulus,'a':a,'b':b,'pair_count':total_pairs}
    for replicate in range(1,args.replicates+1):
        for sample_id in sorted(by_id,key=lambda x:(int(by_id[x]['source_order']),x)):
            sample_id_out=f"CTRL_R{replicate:02d}_{by_id[sample_id]['run_accession']}"
            design_rows.append({'replicate':replicate,'sample_id':sample_id_out,'class_label':'single_source_control','mixture_definition_id':'CONTROL','category':'control','pattern_id':'CONTROL','n_parents':1,'parents':sample_id,'parent_runs':by_id[sample_id]['run_accession'],'weights':'1.000000','read_pairs':args.read_pairs,'entropy_norm':0.0,'minor_fraction':0.0})
            count=args.read_pairs; start=source_cursors[sample_id]; stop=start+count; p=source_params[sample_id]
            allocation_rows.append({'replicate':replicate,'sample_id':sample_id_out,'source_sample_accession':sample_id,'source_run_accession':by_id[sample_id]['run_accession'],'target_weight':1.0,'read_pairs':count,'allocation_ordinal_start':start,'allocation_ordinal_stop':stop,'eligible_physical_start':p['eligible_start'],'permutation_modulus':p['modulus'],'permutation_a':p['a'],'permutation_b':p['b']})
            source_cursors[sample_id]=stop
        for definition in definitions:
            parents=definition['parents_canonical'].split(';'); weights=[float(x) for x in definition['weights_canonical'].split(';')]
            rng=np.random.default_rng(stable_seed(args.seed,'parent_order',replicate,definition['mixture_definition_id']))
            order=rng.permutation(len(parents)).tolist(); ordered_parents=[parents[i] for i in order]; ordered_weights=[weights[i] for i in range(len(weights))]
            counts=largest_remainder_counts(ordered_weights,args.read_pairs)
            sample_id_out=f"MIX_R{replicate:02d}_{definition['mixture_definition_id']}"
            design_rows.append({'replicate':replicate,'sample_id':sample_id_out,'class_label':'synthetic_mixture','mixture_definition_id':definition['mixture_definition_id'],'category':definition['category'],'pattern_id':definition['pattern_id'],'n_parents':definition['n_parents'],'parents':';'.join(ordered_parents),'parent_runs':';'.join(by_id[x]['run_accession'] for x in ordered_parents),'weights':';'.join(f'{x:.6f}' for x in ordered_weights),'read_pairs':args.read_pairs,'entropy_norm':normalized_entropy(ordered_weights),'minor_fraction':min(ordered_weights)})
            for parent,weight,count in zip(ordered_parents,ordered_weights,counts):
                start=source_cursors[parent]; stop=start+count; p=source_params[parent]
                allocation_rows.append({'replicate':replicate,'sample_id':sample_id_out,'source_sample_accession':parent,'source_run_accession':by_id[parent]['run_accession'],'target_weight':weight,'read_pairs':count,'allocation_ordinal_start':start,'allocation_ordinal_stop':stop,'eligible_physical_start':p['eligible_start'],'permutation_modulus':p['modulus'],'permutation_a':p['a'],'permutation_b':p['b']})
                source_cursors[parent]=stop
    write_tsv(outdir/'locked_generated_library_design_560.tsv',design_rows)
    write_tsv(outdir/'locked_read_allocations.tsv',allocation_rows)
    allocation_summary=[]
    for sample_id,p in source_params.items():
        used=source_cursors[sample_id]; ok=used<=p['modulus']
        allocation_summary.append({'source_sample_accession':sample_id,'source_run_accession':by_id[sample_id]['run_accession'],'population':by_id[sample_id]['population'],'pair_count':p['pair_count'],'eligible_start':p['eligible_start'],'eligible_pairs':p['modulus'],'allocated_pairs':used,'remaining_eligible_pairs':p['modulus']-used,'allocation_fits':ok,'permutation_a':p['a'],'permutation_b':p['b']})
        if not ok: raise SystemExit(f'[ERROR] Allocation exceeds available pairs for {sample_id}: {used}>{p["modulus"]}')
    write_tsv(outdir/'source_read_allocation_summary.tsv',allocation_summary)

    rare_rows=[]
    controls_by_rep={rep:[row['sample_id'] for row in design_rows if row['replicate']==rep and row['class_label']=='single_source_control'] for rep in range(1,args.replicates+1)}
    mixtures_by_rep={rep:[row['sample_id'] for row in design_rows if row['replicate']==rep and row['class_label']=='synthetic_mixture'] for rep in range(1,args.replicates+1)}
    for rep in range(1,args.replicates+1):
        mix=mixtures_by_rep[rep]
        rng=np.random.default_rng(stable_seed(args.seed,'rare_schedule',rep)); shuffled=[mix[i] for i in rng.permutation(len(mix))]
        for injection in (1,2,4):
            for group_index in range(0,len(shuffled),injection):
                chosen=shuffled[group_index:group_index+injection]
                if len(chosen)!=injection: raise AssertionError('Mixture count must divide 84')
                graph_n=len(controls_by_rep[rep])+injection
                rare_rows.append({'replicate':rep,'injection_count':injection,'graph_id':f'R{rep:02d}_I{injection}_{group_index//injection+1:03d}','control_sample_ids':';'.join(controls_by_rep[rep]),'mixture_sample_ids':';'.join(chosen),'graph_n':graph_n,'mixture_prevalence':injection/graph_n})
    write_tsv(outdir/'locked_rare_event_schedule_735.tsv',rare_rows)

    lock={
        'status':'LOCKED','dataset':'finger_millet_PRJNA791522','panel_samples':83,'benchmark_sources':28,
        'primary_analysis_mode':'paired','secondary_analysis_mode':'r1','kmer':17,'sketch_dimension':16384,
        'full_cohort_geometry_pairs':50000,'source_design_pairs':30000,'synthetic_read_pairs_per_library':args.read_pairs,
        'replicates':args.replicates,'generated_controls_per_replicate':28,'generated_mixtures_per_replicate':84,
        'generated_libraries_total':len(design_rows),'rare_event_graphs_total':len(rare_rows),
        'locked_knn_full_cohort':full_k,'locked_knn_synthetic':source_k,'orc_alpha':args.alpha,
        'knn_rule':'smallest candidate k yielding a connected symmetric-union kNN graph on control-only design sketches',
        'source_selection_seed':79152228,'design_seed':args.seed,
        'primary_endpoint':'mean TMS rank/AUC for one true read-level mixture injected among 28 controls across five replicates',
        'primary_metrics':['ROC AUC','Average Precision with prevalence lift','Top-1/Top-3/Top-5 capture vs exact chance','negative-edge fraction','fraction of graphs with informative negative-ORC variation'],
        'secondary_endpoints':['batch mixture discrimination','2- and 4-mixture rare-event injections','within vs moderate-between vs high-between vs three-population mixtures','ratio-stratified performance','TMS ablation and outlier comparators','R1-only secondary mode'],
        'performance_interpretation':{'SUPPORTED':'mean primary AUC >= 0.70','WEAK_TO_MODERATE':'0.60 <= mean primary AUC < 0.70','NOT_SUPPORTED':'mean primary AUC < 0.60'},
        'parameter_tuning_on_mixture_labels':False,
        'geometry_design_synthetic_read_segments_disjoint':True,
    }
    write_json(outdir/'analysis_design_lock.json',lock)
    text=['Finger millet external-validation analysis design lock','================================================','',f"Primary mode: paired",f"Source controls: 28 (4 per population)",f"Generated design: 28 controls + 84 mixtures per replicate x {args.replicates} replicates = {len(design_rows)} libraries",f"Primary rare-event endpoint: one mixture + 28 controls ({1/29:.3%} prevalence)",f"Rare-event schedule: {len(rare_rows)} graphs",f"Locked kNN: full cohort={full_k}; synthetic benchmark={source_k}",f"k-mer={17}; sketch dimension={16384}; ORC alpha={args.alpha}",'','No mixture label or downstream performance result was used to select sources, parent sets, kNN, or score parameters. Geometry, design, and synthetic read allocations use disjoint source-read segments.']
    (outdir/'ANALYSIS_DESIGN_LOCK.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (outdir/'DESIGN_LOCK_CREATED.txt').write_text('LOCKED\n',encoding='utf-8')
    print('\n'.join(text))

if __name__=='__main__': main()
