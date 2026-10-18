"""Run the marker and feature algorithms with strict paired-SAM and cache guards."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
import numpy as np
from reference_support import ROOT, module, sha, write_json, read_json, require

common=module(ROOT/'features/reference_qc_common.py','reference_qc_common')

def guarded_stream(*,launcher,index_prefix,r1_path,r2_path,threads,seed,upto_pairs=0,skip_pairs=0,stderr_log=None):
 # Fixed Bowtie arguments; stderr goes directly to a file to avoid pipe deadlock.
 command=common.build_bowtie_command(launcher,index_prefix,r1_path,r2_path,threads,seed,upto_pairs,skip_pairs)
 if os.name!='nt':
  # POSIX invokes the executable directly; Windows uses the checked command wrapper.
  command=[launcher,'--very-sensitive','--reorder','--seed',str(seed),'-p',str(threads),'-x',index_prefix,'-1',r1_path,'-2',r2_path]
  if skip_pairs:command+=['-s',str(skip_pairs)]
  if upto_pairs:command+=['-u',str(upto_pairs)]
 p=Path(stderr_log);p.parent.mkdir(parents=True,exist_ok=True)
 timed_out=threading.Event();primary=0;pairs=0
 with open(p,'w',encoding='utf-8') as err:
  proc=subprocess.Popen(command,shell=os.name=='nt',stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',errors='replace',bufsize=1)
  def kill():
   timed_out.set()
   if proc.poll() is None:
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:proc.kill()
  timer=threading.Timer(3600,kill);timer.daemon=True;timer.start()
  try:
   for line in proc.stdout:
    if not line or line.startswith('@'):continue
    r=common.parse_sam_line(line)
    if r.is_primary:
     primary+=1
     if r.is_read1:pairs+=1
    yield r
   rc=proc.wait()
  finally:
   timer.cancel()
   if proc.poll() is None:kill();proc.wait()
   if proc.stdout is not None:proc.stdout.close()
 require(not timed_out.is_set(),'Bowtie per-library timeout; see '+str(p))
 require(rc==0,'Bowtie failed; see '+str(p)+'\n'+p.read_text(encoding='utf-8',errors='replace')[-4000:])
 require(primary>0 and pairs>0,'Bowtie emitted no primary paired records')
 if upto_pairs:require(pairs==upto_pairs and primary==2*upto_pairs,f'Bowtie pair count mismatch {pairs}/{upto_pairs}; see {p}')

common.stream_bowtie_sam=guarded_stream

def install_cache_guard(mod):
 original=mod.process_sample
 def process_sample(**kw):
  cache=Path(kw['cache_path']);metrics=cache.with_suffix('.metrics.json');seal=cache.with_suffix('.validated.json')
  key={k:str(kw[k]) if isinstance(kw[k],Path) else kw[k] for k in ['sample_id','r1_path','r2_path','marker_panel_sha256','index_prefix','threads','seed','upto_pairs','skip_pairs','min_mapq','min_baseq']}
  key['n_markers']=len(kw['marker_index'].marker_table)
  if seal.is_file():
   rec=read_json(seal);require(rec['identity']==key,'Cache configuration mismatch: '+str(cache))
   require(cache.is_file() and metrics.is_file() and sha(cache)==rec['cache_sha256'] and sha(metrics)==rec['metrics_sha256'],'Completed cache altered: '+str(cache))
   return read_json(metrics)
  # Never reuse unsealed partial or historical caches. Move them within this new run only.
  for item in [cache,metrics]:
   if item.exists():
    trash=item.parent/'incomplete';trash.mkdir(exist_ok=True)
    item.replace(trash/(item.name+'.'+str(time.time_ns())))
  result=original(**kw)
  require(int(result['total_pairs'])==kw['upto_pairs'],'Incomplete paired feature counts')
  require(int(result['primary_records'])==2*kw['upto_pairs'],'Incomplete primary SAM count')
  with np.load(cache,allow_pickle=False) as a:
   require(int(a['n_markers'])==key['n_markers'],'Bad feature cache marker count')
   for col in ['ref_counts','alt_counts','other_counts']:require(a[col].shape==(key['n_markers'],),'Bad feature cache dimension')
  write_json(seal,{'identity':key,'cache_sha256':sha(cache),'metrics_sha256':sha(metrics)})
  return result
 mod.process_sample=process_sample

if __name__=='__main__':
 kind=sys.argv[1];sys.argv=[sys.argv[0]]+sys.argv[2:]
 if kind=='markers':
  mod=module(ROOT/'features/02_build_marker_panels.py','marker_stage')
 elif kind=='features':
  mod=module(ROOT/'features/03_extract_reference_features.py','feature_stage');install_cache_guard(mod)
 else:raise SystemExit('Unknown guarded stage: '+kind)
 mod.main()
