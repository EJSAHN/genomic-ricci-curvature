"""Tests of the retained marker, matrix, and paired-SAM guards with artificial inputs."""
from pathlib import Path
import tempfile, sys, json, shutil, contextlib, io, gzip
import unittest
import numpy as np
import pandas as pd
REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY/'pipelines/coffee/reference_based'))
from reference_support import *
ROOT=REPOSITORY
from guarded_reference import install_cache_guard
def rejects(fn):
 try:fn()
 except (RuntimeError,ValueError,FileNotFoundError):return
 raise AssertionError('Expected rejection')


def frame_test():
 a=pd.DataFrame({'sample_id':['x','y'],'v':[1.,np.nan]});assert compare_frames(a,a,['sample_id'])['rows']==2
 b=a.copy();b.loc[0,'v']=1.001;rejects(lambda:compare_frames(a,b,['sample_id']))
 b=a.copy();b.loc[1,'v']=0.;rejects(lambda:compare_frames(a,b,['sample_id']))
 b=a.copy();b.loc[1,'sample_id']='x';rejects(lambda:compare_frames(a,b,['sample_id']))


def source_sampler():
 m=module(ROOT/'pipelines/coffee/reference_based/features/02_build_marker_panels.py','test_sampler')
 with tempfile.TemporaryDirectory() as td:
  p=Path(td);f1=p/'a_1.fastq';f2=p/'a_2.fastq'
  for mate,f in [(1,f1),(2,f2)]:f.write_text(''.join(f'@r{i}/{mate}\nACGT\n+\nIIII\n' for i in range(100)))
  mask=bytearray(100)
  for i in range(30):mask[i]=1
  a,e,n=m.sample_independent_pairs(f1,f2,mask,10,42);b,e2,n2=m.sample_independent_pairs(f1,f2,mask,10,42)
  assert a==b and e==70 and n==100
  ids=[int(v[0][0].split('/')[0][2:]) for v in a];assert len(set(ids))==10 and min(ids)>=30
  rejects(lambda:m.sample_independent_pairs(f1,f2,mask,80,42))


def cache_test():
 from types import SimpleNamespace
 import guarded_reference as guard
 with tempfile.TemporaryDirectory() as td:
  calls=[];p=Path(td)/'a.npz';n=30
  def original(**kw):
   calls.append(1);np.savez_compressed(p,n_markers=np.array(n),ref_counts=np.ones(n,dtype=np.uint32),alt_counts=np.zeros(n,dtype=np.uint32),other_counts=np.zeros(n,dtype=np.uint32))
   v={'total_pairs':2,'primary_records':4};write_json(p.with_suffix('.metrics.json'),v);return v
  mod=SimpleNamespace(process_sample=original);install_cache_guard(mod)
  kw=dict(sample_id='x',r1_path='r1',r2_path='r2',marker_panel_sha256='abc',index_prefix='idx',threads=4,seed=1,upto_pairs=2,skip_pairs=0,min_mapq=20,min_baseq=20,marker_index=SimpleNamespace(marker_table=list(range(n))),cache_path=p)
  p.write_bytes(b'incomplete')
  mod.process_sample(**kw);assert len(calls)==1 and (p.parent/'incomplete').is_dir()
  mod.process_sample(**kw);assert len(calls)==1
  altered=dict(kw,seed=2);rejects(lambda:mod.process_sample(**altered))
  p.write_bytes(b'tampered');rejects(lambda:mod.process_sample(**kw))


def numpy_input_rejection():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td);np.savez_compressed(str(p/'x')+'_matrix.npz',sample_ids=np.array(['a','b']),alt_fraction=np.ones((2,3)),depth=np.ones((2,3)))
  write_tsv(str(p/'x')+'_samples.tsv',pd.DataFrame({'sample_id':['b','a']}))
  cf=module(ROOT/'pipelines/coffee/reference_based/crossfit/crossfit_common.py','crossfit_common');rejects(lambda:cf.load_matrix(p/'x'))


def sam_stream_guards():
 import guarded_reference as guard,os
 if os.name=='nt':return  # Production Windows is gated with real Bowtie, not this POSIX fixture.
 with tempfile.TemporaryDirectory(prefix='sam stream ') as td:
  d=Path(td);tool=d/'fakealign';tool.write_text('#!/usr/bin/env python3\nprint("r\\t77\\t*\\t0\\t0\\t*\\t*\\t0\\t0\\tACGT\\tIIII")\nprint("r\\t141\\t*\\t0\\t0\\t*\\t*\\t0\\t0\\tACGT\\tIIII")\n');tool.chmod(0o755)
  kw=dict(launcher=str(tool),index_prefix='i',r1_path='a',r2_path='b',threads=4,seed=1,upto_pairs=1,skip_pairs=0,stderr_log=str(d/'stderr.log'))
  records=list(guard.guarded_stream(**kw));assert len(records)==2
  rejects(lambda:list(guard.guarded_stream(**dict(kw,upto_pairs=2))))
  tool.write_text('#!/usr/bin/env python3\n');rejects(lambda:list(guard.guarded_stream(**kw)))
  tool.write_text('#!/usr/bin/env python3\nraise SystemExit(3)\n');rejects(lambda:list(guard.guarded_stream(**kw)))


def marker_builder_cli():
 m=module(ROOT/'pipelines/coffee/reference_based/features/02_build_marker_panels.py','test_marker_main')
 common=sys.modules['reference_qc_common']
 with tempfile.TemporaryDirectory(prefix='marker count ') as td:
  d=Path(td);raw=d/'raw';raw.mkdir();out=d/'out';gen=d/'gen';gen.mkdir();ss=[];gm=[]
  for sid in SOURCES:
   for mate in [1,2]:
    (raw/f'{sid}_{mate}.fastq').write_text(''.join(f'@{sid}_{i}/{mate}\n'+('A'*100)+'\n+\n'+('I'*100)+'\n' for i in range(12)))
   g=gen/(sid+'.fastq');g.write_text('@generated|source='+sid+'|source_pair=0/1\nAAAA\n+\nIIII\n')
   gm.append({'r1_path':str(g)});ss.append({'parent_id':sid,'read_pairs_available':12})
  write_tsv(d/'samples.tsv',pd.DataFrame({'sample_id':SOURCES,'include_primary':1}));write_tsv(d/'source.tsv',pd.DataFrame(ss));write_tsv(d/'gen.tsv',pd.DataFrame(gm));write_tsv(d/'refs.tsv',pd.DataFrame({'reference_id':REFS,'accession':['test','test'],'align_launcher':['mock','mock'],'index_prefix':['mock','mock']}))
  def stream(**kw):
   for i in range(kw['upto_pairs']):
    for flag in [99,147]:
     line='\t'.join([f'read{i}',str(flag),'chr1','1','60','100M','=','1','100','C'*30+'A'*70,'I'*100,'MD:Z:'+''.join(['0A']*30)+'70','NM:i:30'])
     yield common.parse_sam_line(line)
  m.stream_bowtie_sam=stream
  oldargs=sys.argv
  try:
   sys.argv=['marker','--reference_manifest',str(d/'refs.tsv'),'--sample_manifest',str(d/'samples.tsv'),'--expected_source_count','12','--source_fastq_root',str(raw),'--generated_fastq_manifest',str(d/'gen.tsv'),'--source_sampling_summary',str(d/'source.tsv'),'--outdir',str(out),'--logs_dir',str(d/'logs'),'--source_pairs','4','--reserve_real_skip_pairs','1','--reserve_real_pairs','1']
   with contextlib.redirect_stdout(io.StringIO()):m.main()
  finally:sys.argv=oldargs
  for ref in REFS:
   p=read_tsv(out/ref/'marker_panel.tsv');assert len(p)==30
   summary=read_json(out/ref/'marker_panel_summary.json');assert summary['n_source_libraries']==12
  indep=read_tsv(out/'marker_discovery_independence.tsv');assert len(indep)==12 and (indep.panel_pairs_selected==4).all() and (indep.overlap_with_generated_pairs==0).all()


class ReferenceGuardTests(unittest.TestCase):
    def test_changed_table_values_fail(self): frame_test()
    def test_deterministic_disjoint_sampling(self): source_sampler()
    def test_cache_resume_and_mutation(self): cache_test()
    def test_matrix_sample_order_fails(self): numpy_input_rejection()
    def test_sam_counts_and_failed_process(self): sam_stream_guards()
    def test_marker_builder_twelve_source_cli(self): marker_builder_cli()

if __name__=='__main__': unittest.main()
