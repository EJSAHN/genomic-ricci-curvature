"""Execution and validation helpers; no scientific score formula is redefined here."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCES=[f'SRR170376{i}' for i in list(range(10,18))+list(range(24,28))]
POOLS=[f'SRR170376{i}' for i in range(18,24)]
REFS=['ET39','Cara1']
PARAMS={'source_pairs':100000,'real_skip_pairs':100000,'real_pairs':50000,'generated_pairs':6000,'marker_seed':52001,'feature_seed':53001,'threads':4,'min_mapq':20,'min_baseq':20,'min_alt_total':4,'min_alt_samples':2,'min_alt_in_one_sample':4,'max_markers':50000,'min_control_call_rate':.30,'max_features':5000,'max_pca_components':5,'min_overlap':10}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def safe_json(x):
 if isinstance(x,dict):return {str(k):safe_json(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe_json(v) for v in x]
 if isinstance(x,np.ndarray):return safe_json(x.tolist())
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,(np.floating,float)):return float(x) if math.isfinite(float(x)) else None
 if isinstance(x,np.bool_):return bool(x)
 if isinstance(x,Path):return str(x)
 return x

def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(safe_json(x),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read_tsv(p):return pd.read_csv(p,sep='\t',encoding='utf-8-sig')
def write_tsv(p,df):
 Path(p).parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,sep='\t',index=False)

def module(path,name):
 path=Path(path);sys.path.insert(0,str(path.parent));sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def require(ok,message):
 if not bool(ok):raise RuntimeError(message)

def compare_frames(observed,expected,keys,columns=None,atol=1e-8):
 require(not observed.duplicated(keys).any(),'Observed duplicate keys: '+str(keys))
 require(not expected.duplicated(keys).any(),'Expected duplicate keys: '+str(keys))
 a=observed.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
 require(len(a)==len(b),f'Row count differs: {len(a)} / {len(b)}')
 require(a[keys].astype(str).equals(b[keys].astype(str)),'Row identities differ')
 columns=columns or [c for c in b if c not in keys and c in a and pd.api.types.is_numeric_dtype(b[c])]
 maxdiff=0.;checks=0
 for c in columns:
  require(c in a and c in b,'Missing comparison column: '+c)
  x=pd.to_numeric(a[c],errors='raise').to_numpy(dtype=float);y=pd.to_numeric(b[c],errors='raise').to_numpy(dtype=float)
  require(np.array_equal(np.isnan(x),np.isnan(y)),'NaN pattern differs: '+c)
  require(np.allclose(x,y,rtol=0,atol=atol,equal_nan=True),'Replay mismatch in '+c)
  valid=np.isfinite(x)&np.isfinite(y)
  if valid.any():maxdiff=max(maxdiff,float(np.max(np.abs(x[valid]-y[valid]))))
  checks+=int(valid.sum())
 return {'rows':len(a),'numeric_cells':checks,'max_abs_difference':maxdiff,'tolerance':atol}

def stage_complete(folder,identity):
 folder=Path(folder);p=folder/'STAGE_COMPLETE.json'
 if not p.exists():return False
 rec=read_json(p);require(rec['identity']==identity,'Completed-stage input/config mismatch: '+str(folder))
 for rel,dig in rec['outputs'].items():
  q=folder/rel;require(q.is_file() and sha(q)==dig,'Completed output missing/modified: '+str(q))
 return True

def seal_stage(folder,identity):
 folder=Path(folder)
 files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='STAGE_COMPLETE.json' and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
 require(bool(files),'Cannot mark empty stage complete')
 write_json(folder/'STAGE_COMPLETE.json',{'identity':identity,'outputs':files,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

def capture_step(work,name,args,timeout=21600):
 log=work/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 print('\n[RUN]',name,flush=True);start=time.monotonic()
 env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with open(log,'w',encoding='utf-8',newline='\n') as f:
  f.write('COMMAND: '+json.dumps([str(v) for v in args])+'\n');f.flush()
  proc=subprocess.Popen([str(v) for v in args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
  try:
   for line in proc.stdout:
    print(line.rstrip(),flush=True);f.write(line);f.flush()
   rc=proc.wait(timeout=timeout)
  except BaseException:
   if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   else:proc.kill()
   proc.wait();raise
 elapsed=time.monotonic()-start
 with open(work/'TIMINGS.jsonl','a',encoding='utf-8') as f:f.write(json.dumps({'step':name,'seconds':elapsed,'exit':rc})+'\n')
 print(f'[TIME] {name}: {elapsed:.1f}s; exit={rc}',flush=True)
 require(rc==0,f'{name} failed (exit={rc}); see {log}')

def validate_feature_root(root,scenario,ncontrols):
 root=Path(root);counts=[]
 for ref in REFS:
  for kind,nrows in [('generated',5*(24+ncontrols)),('real',18)]:
   a=np.load(root/ref/(kind+'_matrix.npz'),allow_pickle=False)
   ids=a['sample_ids'].astype(str).tolist();x=a['alt_fraction'];depth=a['depth']
   samples=read_tsv(root/ref/(kind+'_samples.tsv'));mapping=read_tsv(root/ref/(kind+'_mapping_and_marker_metrics.tsv'))
   require(len(ids)==nrows and len(set(ids))==nrows,'Wrong matrix row count: '+ref+' '+kind)
   require(ids==samples.sample_id.astype(str).tolist(),'Matrix order mismatch: '+ref+' '+kind)
   require(set(ids)==set(mapping.sample_id.astype(str)),'Mapping row mismatch: '+ref+' '+kind)
   require(x.shape==depth.shape and x.shape[0]==nrows,'Matrix shape mismatch')
   require(x.shape[1]>=20 and np.isfinite(x).any(),'Empty or undersized feature matrix')
   require(not np.isinf(x).any(),'Infinite allele fraction')
   finite=np.isfinite(x);require(((x[finite]>=0)&(x[finite]<=1)).all(),'Allele fractions out of bounds')
   require(np.array_equal(finite,depth>=1),'Depth/callability mismatch')
   pairs=6000 if kind=='generated' else 50000
   require((mapping.total_pairs==pairs).all() and (mapping.primary_records==2*pairs).all(),'SAM did not contain all requested paired reads')
   if kind=='generated':
    require(set(samples.scenario)=={scenario},'Mixed scenario in feature matrix')
    for r,g in samples.groupby('replicate'):
     require(r in range(1,6) and len(g)==24+ncontrols,'Wrong replicate dimensions')
     control=g[g.class_label=='single_source_control'];require(len(control)==ncontrols,'Wrong control count')
     if scenario=='conservative':
      require(set(control.parents)==set(SOURCES),'Incorrect corrected control sources')
      for pp in g.parents.astype(str):require(set(pp.replace(';',',').split(','))<=set(SOURCES),'Pooled source in corrected features')
   counts.append({'reference':ref,'kind':kind,'n':nrows,'markers':int(x.shape[1]),'finite_entries':int(finite.sum())})
 return counts

def validate_crossfit(folder,scenario,ncontrols):
 folder=Path(folder);pr=read_tsv(folder/'crossfit_generated_fold_provenance.tsv');sc=read_tsv(folder/'crossfit_generated_reference_scores.tsv')
 require(len(pr)==10 and set(pr.reference_id)==set(REFS),'Wrong outer-fold count')
 require(len(sc)==2*5*(ncontrols+24),'Wrong score row count')
 require(set(sc.scenario)=={scenario},'Score scenario differs')
 for r in pr.to_dict('records'):
  require(int(r['outer_test_control_count'])==ncontrols and int(r['outer_test_mixture_count'])==24,'Outer test dimensions differ')
  require(int(r['outer_training_control_count'])==4*ncontrols,'Outer training dimensions differ')
  require(int(r['outer_train_test_overlap_count'])==0,'Outer train/test overlap')
  require(int(r['outer_selected_marker_count'])>=10,'Insufficient selected markers')
  for col in ['marker_selection_used_outer_test_rows','pca_fit_used_outer_test_rows','scaling_used_outer_test_rows']:require(int(r[col])==0,'Leakage flag '+col)
  inner=json.loads(r['inner_split_audit']);require(len(inner)==4,'Wrong inner-fold count')
  for v in inner:
   require(v['inner_train_test_overlap_count']==0,'Inner train/test overlap')
   require(v['inner_fit_control_count']==3*ncontrols and v['inner_calibration_control_count']==ncontrols,'Inner split dimensions')
  testrep=int(r['outer_test_replicate']);require(str(testrep) not in str(r['outer_training_replicates']).split(','),'Held-out replicate in outer training')
 for c in ['reference_qc_crossfit','reference_qc_crossfit_no_pca','reference_mapping_crossfit']:require(np.isfinite(sc[c]).all(),'Nonfinite final score '+c)
 return {'outer_folds':len(pr),'score_rows':len(sc),'test_controls_per_fold':ncontrols,'train_controls_per_fold':4*ncontrols,'reported_leakage_flags_zero':True}

def make_consensus(scores_dir,outdir,scenario,tms_scores=None):
 cc=module(ROOT/'crossfit/crossfit_common.py','crossfit_common')
 co=module(ROOT/'crossfit/02_build_crossfit_consensus.py','consensus_module')
 scores_dir=Path(scores_dir);outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
 gen=read_tsv(scores_dir/'crossfit_generated_reference_scores.tsv');real=read_tsv(scores_dir/'crossfit_real_reference_scores.tsv')
 for frame in [gen,real]:
  require(not frame.duplicated(['reference_id','sample_id']).any(),'Duplicate per-reference scores')
  require((frame.groupby('sample_id').reference_id.nunique()==2).all(),'A reference missing from consensus')
  for c in cc.SCORE_VARIANTS:require(np.isfinite(frame[c]).all(),'Nonfinite reference score')
 g=co.consensus_table(gen,co.GENERATED_META,cc.SCORE_VARIANTS);r=co.consensus_table(real,co.REAL_META,cc.SCORE_VARIANTS)
 write_tsv(outdir/'crossfit_generated_consensus_scores.tsv',g);write_tsv(outdir/'crossfit_real_consensus_scores.tsv',r)
 rows=[]
 for rep,group in g.groupby('replicate'):
  for variant in cc.SCORE_VARIANTS:rows.append({'replicate':int(rep),'score_variant':variant,**cc.evaluate_binary(group.is_mixture.to_numpy(),group[variant+'_consensus'].to_numpy())})
 metrics=pd.DataFrame(rows);write_tsv(outdir/'crossfit_generated_consensus_metrics.tsv',metrics)
 summary=[]
 for variant,group in metrics.groupby('score_variant'):
  summary.append({'score_variant':variant,'roc_auc_mean':group.roc_auc.mean(),'roc_auc_sd':group.roc_auc.std(ddof=0),'average_precision_mean':group.average_precision.mean(),'average_precision_sd':group.average_precision.std(ddof=0),'prevalence':float(group.prevalence.mean()),'ap_lift':float(group.average_precision.mean()/group.prevalence.mean()),'n_replicates':len(group)})
 ss=pd.DataFrame(summary);write_tsv(outdir/'method_summary.tsv',ss)
 target=ss[ss.score_variant=='reference_qc_crossfit'].iloc[0].to_dict();target['scenario']=scenario;target['sd_ddof']=0
 target['interpretation']='Read-disjoint generated-replicate evaluation; not unseen-genotype validation'
 write_json(outdir/'REFERENCE_RESULTS.json',target)
 if scenario=='conservative' and tms_scores is not None:
  tms=read_tsv(tms_scores);tms=tms[(tms.scenario==scenario)&(tms.analysis_mode=='paired')].copy()
  require(len(tms)==len(g),'Corrected TMS/reference library count mismatch')
  # Preserve the exact published TMS field name in the joined output.
  merged=g.merge(tms,on=['sample_id','replicate'],validate='one_to_one',suffixes=('_reference','_TMS'))
  write_tsv(outdir/'reference_and_paired_TMS_same_libraries.tsv',merged)
  rr=[]
  y=r['is_metadata_pool_candidate'].astype(int).to_numpy();require(int(y.sum())==6,'Wrong number of metadata-identified pool records')
  for v in cc.SCORE_VARIANTS:rr.append({'score_variant':v,'scope':'descriptive_technical_replicates_not_independent_biological_validation',**cc.evaluate_binary(y,r[v+'_consensus'].to_numpy())})
  write_tsv(outdir/'real_metadata_descriptive_metrics.tsv',pd.DataFrame(rr))
 return target
