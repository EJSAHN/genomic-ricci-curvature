"""Fast checks for source identity, deterministic design, tied metrics, and guards."""
from pathlib import Path
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workflows'))
from common import load_module, metrics, verify_metadata, SOURCES, POOLS


class ReleaseTests(unittest.TestCase):
    def test_tied_metrics(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        y=[1,0,1,0,0,1];s=[.5,.5,.9,.1,.9,.5]
        a=metrics(y,s)
        self.assertAlmostEqual(a['auc'],roc_auc_score(y,s),15)
        self.assertAlmostEqual(a['ap'],average_precision_score(y,s),15)

    def test_nonfinite_metrics_fail(self):
        with self.assertRaises(ValueError): metrics([0,1],[0,float('nan')])
        with self.assertRaises(ValueError): metrics([1,1],[0,1])

    def test_detailed_metadata(self):
        d=verify_metadata();self.assertEqual(len(d['individual_cultivar_sources']),12)
        self.assertNotIn('SRR17037623',d['individual_cultivar_sources'])

    def test_solver_failure_is_not_zero_curvature(self):
        m=load_module(ROOT/'pipelines/coffee/calibration/run_calibration.py','test_cal_guard')
        solver=lambda *a,**kw:SimpleNamespace(success=False,fun=None,status=2,message='test infeasible')
        with self.assertRaises(RuntimeError):m.checked_transport(solver)(None)
        solver=lambda *a,**kw:SimpleNamespace(success=True,fun=float('nan'))
        with self.assertRaises(RuntimeError):m.checked_transport(solver)(None)

    def test_historical_sampling_design(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable,str(ROOT/'pipelines/coffee/read_level/01_design_read_level_mixtures.py'),'--sample_manifest',str(ROOT/'history/coffee_source_manifest_13.tsv'),'--ratio_patterns',str(ROOT/'config/coffee/mixture_ratio_patterns.tsv'),'--outdir',d,'--seed','42000'],check=True,stdout=subprocess.DEVNULL)
            a=pd.read_csv(Path(d)/'design_manifest.tsv',sep='\t')
            b=pd.read_csv(ROOT/'examples/coffee/rare_input/truth_manifest.tsv',sep='\t')
            self.assertEqual(len(a),365)
            for c in ['sample_id','scenario','replicate','class_label','parents','target_counts']:
                x=a.set_index('sample_id').sort_index() if c!='sample_id' else None
                if c=='sample_id':self.assertEqual(set(a[c]),set(b[c]))
                else:pd.testing.assert_series_equal(x[c],b.set_index('sample_id').sort_index()[c],check_dtype=False)
            q=a[a.scenario=='conservative'];self.assertEqual(len(q),180)
            for parents in q.parents:self.assertTrue(set(parents.split(','))<=set(SOURCES))

    def test_completed_output_mutation_fails(self):
        m=load_module(ROOT/'workflows/analyze.py','test_raw_workflow')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'value.txt').write_text('one');m.seal(p,{'seed':42})
            self.assertTrue(m.reusable(p,{'seed':42}))
            with self.assertRaises(ValueError):m.reusable(p,{'seed':43})
            (p/'value.txt').write_text('two')
            with self.assertRaises(ValueError):m.reusable(p,{'seed':42})


if __name__=='__main__':unittest.main()
