"""Execute the preserved numerical calibration with strict failures and atomic outputs."""
from __future__ import annotations
import argparse,importlib.util,json,math,os,sys,time,uuid
from pathlib import Path
from audit_support import validate_result,sha,dump,require,COMPLETION,REQUIRED
ROOT=Path(__file__).resolve().parent
CORE=ROOT/'calibration_core.py'

def checked_transport(solver):
    def solve(*a,**kw):
        r=solver(*a,**kw)
        if not bool(r.success) or r.fun is None or not math.isfinite(float(r.fun)):
            raise RuntimeError(f'Transport solver failed; no zero-value fallback is allowed. status={getattr(r,"status",None)} message={getattr(r,"message",None)}')
        solve.calls+=1
        return r
    solve.calls=0
    return solve

def checked_fastq(path,max_reads):
    import gzip
    opener=gzip.open if str(path).lower().endswith('.gz') else open
    with opener(path,'rt',encoding='ascii',errors='strict') as f:
        for i in range(max_reads):
            h=f.readline();s=f.readline().strip();p=f.readline();q=f.readline().strip()
            require(h.startswith('@') and p.startswith('+') and s and len(s)==len(q),
                    f'Invalid/truncated FASTQ at record {i+1} in {path}')
            yield s

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fastq_dir',required=True);p.add_argument('--outdir',required=True)
    for k,d in [('reads_per_sample',6000),('kmer',17),('sketch',16384),('n_synth',40),('max_parents',3),('knn',4),('seed',42)]:p.add_argument('--'+k,type=int,default=d)
    p.add_argument('--min_minor',type=float,default=.10);p.add_argument('--alpha',type=float,default=.5)
    p.add_argument('--known_pools',required=True)
    a=p.parse_args();target=Path(a.outdir).resolve();fastq=Path(a.fastq_dir).resolve()
    require(target!=fastq,'Output cannot be the raw-input folder')
    args={k:getattr(a,k) for k in ('reads_per_sample','kmer','sketch','n_synth','max_parents','knn','seed','min_minor','alpha')}
    args['use_r2']=False;args['include_real_pools']=False
    args['known_pools']=sorted(x.strip() for x in a.known_pools.split(',') if x.strip())
    require(args['known_pools'] in [[f'SRR170376{i}' for i in range(18,23)],[f'SRR170376{i}' for i in range(18,24)]],'Unapproved exclusion set')
    expected=json.loads((ROOT/'kernel_identity.json').read_text())
    require(sha(CORE)==expected['original_canonical_calibration_sha256'],'Calibration-kernel checksum mismatch')
    spec=importlib.util.spec_from_file_location('frozen_calibration',CORE)
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    all_ids=sorted(core.discover_fastqs(str(fastq)))
    require(all_ids==[f'SRR170376{i}' for i in range(10,28)],'Raw source identity set is not the frozen 18-run cohort')
    source_ids=[s for s in all_ids if s not in args['known_pools']]
    if (target/COMPLETION).exists():
        validate_result(target,args,source_ids);print(f'[RESUME] {target}',flush=True);return 0
    require(not target.exists() or not any(target.iterdir()),f'Nonempty output without completion record: {target}')
    if target.exists():target.rmdir()
    attempt=target.parent/'_attempts'/f'{target.name}_{uuid.uuid4().hex[:8]}'
    attempt.mkdir(parents=True)
    transport=checked_transport(core.linprog);core.linprog=transport;core.read_fastq_seqs=checked_fastq
    original_sketch=core.kmer_sketch_prob
    def checked_sketch(seqs,k,sketch_size):
        seqs=list(seqs)
        require(any(len(s)>=k and 'N' not in s.upper() for s in seqs),'No eligible read for the original k-mer algorithm')
        return original_sketch(seqs,k,sketch_size)
    core.kmer_sketch_prob=checked_sketch
    argv=sys.argv[:];start=time.monotonic()
    try:
        call=sys.argv[1:];call[call.index('--outdir')+1]=str(attempt)
        sys.argv=[str(CORE)]+call
        core.main()
        validate_result(attempt,args,source_ids,need_marker=False)
        rec={'status':'COMPLETE','parameters':args,'source_ids':source_ids,
             'core_sha256':sha(CORE),'transport_calls':transport.calls,'transport_failures':0,
             'elapsed_seconds':time.monotonic()-start,
             'output_sha256':{n:sha(attempt/n) for n in REQUIRED}}
        dump(attempt/COMPLETION,rec)
        attempt.rename(target)
        print('[CALIBRATION_CHECKPOINT] '+json.dumps({'directory':str(target),'seconds':rec['elapsed_seconds'],'transport_calls':transport.calls}),flush=True)
    except BaseException as exc:
        dump(attempt/'ATTEMPT_FAILED.json',{'error':str(exc),'type':type(exc).__name__})
        raise
    finally:sys.argv=argv
    return 0
if __name__=='__main__':raise SystemExit(main())
