"""Within-replicate reduction from the archived rare-event analysis."""
import numpy as np
import pandas as pd

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
