"""Read-only validation and atomic checkpoints for the frozen coffee calibration.

Does not change any graph construction, score, sampling, or metric formula.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, time, uuid, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
BOOK='Supplementary_Data_S4_SyntheticMixing.xlsx'
COMPLETION='RUN_COMPLETE.json'
REQUIRED=['synthetic_truth.csv','node_scores.csv','js_distance.csv',BOOK]

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.partial')
    q.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
    os.replace(q,p)

def read_tables(path, wanted):
    """Extract only the numeric/text tables needed to validate an existing XLSX."""
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, f'Corrupt workbook: {path}')
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rel={x.get('Id'):x.get('Target') for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        wb=ET.fromstring(z.read('xl/workbook.xml'));out={}
        for sh in wb.find('s:sheets',NS):
            name=sh.get('name')
            if name not in wanted: continue
            target=rel[sh.get('{'+R+'}id')].lstrip('/')
            if not target.startswith('xl/'):target='xl/'+target
            rows=[]
            for row in ET.fromstring(z.read(target)).findall('.//s:sheetData/s:row',NS):
                values={}
                for c in row:
                    address=c.get('r','');col=''.join(x for x in address if x.isalpha());idx=0
                    for ch in col:idx=idx*26+ord(ch)-64
                    v=c.find('s:v',NS);typ=c.get('t');text=v.text if v is not None else None
                    if typ=='s' and text is not None:value=strings[int(text)]
                    elif typ=='inlineStr':value=''.join(c.find('s:is',NS).itertext())
                    elif typ=='b':value=(text=='1')
                    elif text is None:value=None
                    elif typ=='e':raise RuntimeError(f'Excel error at {name}!{address}: {text}')
                    elif typ in ('str','d'):value=text
                    else:
                        try:value=float(text)
                        except ValueError:value=text
                    values[idx-1]=value
                if values:rows.append([values.get(i) for i in range(max(values)+1)])
            require(bool(rows),f'Empty worksheet {name}')
            header=rows[0];out[name]=[dict(zip(header,r+[None]*(len(header)-len(r)))) for r in rows[1:]]
        require(set(wanted)<=set(out),f'Missing worksheets {set(wanted)-set(out)}')
        return out

def expected_truth(source_ids, args):
    import numpy as np
    rng=np.random.default_rng(int(args['seed']));out=[]
    for i in range(int(args['n_synth'])):
        k=int(rng.integers(2,int(args['max_parents'])+1))
        parents=rng.choice(source_ids,size=k,replace=False).tolist()
        for _ in range(1000):
            w=rng.dirichlet(np.ones(k))
            if float(np.min(w))>=float(args['min_minor']): break
        w=w/w.sum()
        out.append({'sample':f'SYNTH_{i+1:03d}','parents':','.join(parents),
                    'weights':','.join(f'{v:.4f}' for v in w),'n_parents':str(k)})
    return out

def validate_result(folder, args, source_ids, need_marker=True):
    folder=Path(folder)
    for name in REQUIRED:require((folder/name).is_file(),f'Missing output {folder/name}')
    tables=read_tables(folder/BOOK,['metrics_summary','node_scores','synthetic_truth','synthetic_score_vs_truth','edge_curvature'])
    require(len(tables['metrics_summary'])==1,'Expected exactly one metric row')
    m=tables['metrics_summary'][0]
    for key,exp in [('n_individual',len(source_ids)),('n_synthetic',int(args['n_synth'])),('n_real_pools',0),
                    ('reads_per_sample',int(args['reads_per_sample'])),('kmer',int(args['kmer'])),
                    ('sketch',int(args['sketch'])),('knn',int(args['knn'])),('seed',int(args['seed']))]:
        require(m.get(key)==exp,f'Parameter mismatch {key}: {m.get(key)} != {exp} in {folder}')
    require(abs(float(m['orc_alpha'])-float(args['alpha']))<1e-14,'Curvature parameter mismatch')
    require(not bool(m['use_r2']) and not args.get('use_r2',False),'R1-only scoring expected for profile calibration')
    for key in ['roc_auc_synthetic','avg_precision_synthetic','best_f1_synthetic','best_threshold']:
        require(math.isfinite(float(m[key])),f'Nonfinite {key}')
    for key in ['roc_auc_synthetic','avg_precision_synthetic','best_f1_synthetic']:
        require(0<=float(m[key])<=1,f'Out of range {key}')
    nodes=tables['node_scores'];syn_ids={f'SYNTH_{i+1:03d}' for i in range(int(args['n_synth']))}
    require(len(nodes)==len(source_ids)+len(syn_ids),'Wrong node row count')
    require({r['sample'] for r in nodes}==set(source_ids)|syn_ids,'Wrong node identities')
    require({r['sample'] for r in nodes if r['label']=='individual'}==set(source_ids),'Wrong source/control set')
    require({r['sample'] for r in nodes if r['label']=='synthetic_pool'}==syn_ids,'Wrong synthetic labels')
    for r in nodes:
        for k in ('betweenness','neg_orc_incidence','mixture_score_ot'):
            require(math.isfinite(float(r[k])),f'Nonfinite node metric {r["sample"]}/{k}')
    with (folder/'synthetic_truth.csv').open(encoding='utf-8-sig',newline='') as f:truth=list(csv.DictReader(f))
    expected=expected_truth(list(source_ids),args)
    require(len(truth)==len(expected),'Incomplete synthetic truth')
    for actual,exp in zip(truth,expected):
        for k in ('sample','parents','weights','n_parents'):
            require(str(actual.get(k))==exp[k],f'Wrong generated design/seed: {k} at {exp["sample"]}')
    # Independently recalculate headline metrics from saved scores, not just a summary cell.
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=[int(r['label']=='synthetic_pool') for r in nodes];s=[float(r['mixture_score_ot']) for r in nodes]
    require(abs(float(roc_auc_score(y,s))-float(m['roc_auc_synthetic']))<1e-10,'AUC vs saved scores mismatch')
    require(abs(float(average_precision_score(y,s))-float(m['avg_precision_synthetic']))<1e-10,'AP vs saved scores mismatch')
    for row in tables['edge_curvature']:
        require(math.isfinite(float(row['orc'])) and math.isfinite(float(row['distance'])),'Nonfinite edge')
    if need_marker:
        marker=json.loads((folder/COMPLETION).read_text(encoding='utf-8'))
        require(marker.get('status')=='COMPLETE','Missing successful completion marker')
        require(marker['parameters']==args and marker['source_ids']==list(source_ids),'Completion design mismatch')
        for name,h in marker['output_sha256'].items():require(sha(folder/name)==h,f'Completed output changed: {name}')
        require(marker.get('transport_failures')==0,'Invalid LP completion record')
    return {'metrics':m,'n_sources':len(source_ids),'source_ids':list(source_ids),'n_synthetic':len(truth)}

def quarantine_partial(folder):
    folder=Path(folder)
    if not folder.exists() or not any(folder.iterdir()):return None
    require(not (folder/COMPLETION).exists(),f'Cannot quarantine completed results: {folder}')
    dest=folder.parent/'_interrupted_attempts'/f'{folder.name}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
    dest.parent.mkdir(parents=True,exist_ok=True);folder.rename(dest)
    return str(dest)
