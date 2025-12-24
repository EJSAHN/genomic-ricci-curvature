## Pipeline Execution Order

Please run the numbered scripts in the following order to reproduce the results.

1. **Preprocessing & Distance Matrix (Figures 1–2)**
   python 01_run_preprocessing.py --fastq_dir ./data --outdir ./results

2. **Geometry & Curvature Analysis (Figures 3–4)**
   python 02_run_geometry_analysis.py --results_dir ./results --outdir ./results/gauss_euler

3. **Synthetic Validation (Figure 5)**
   python 03_run_synthetic_validation.py --fastq_dir ./data --outdir ./results/synthetic --include_real_pools --known_pools "SRR17037621,SRR17037622,SRR17037623" --seed 42

4. **Robustness Sweeps (Figure 6)**
   python 04_run_sensitivity_sweeps.py --fastq_dir ./data --outdir ./results/sensitivity --n_boot 30 --seed 42

5. **Final Validation Check (Fig. 11 / Validation plot)**
   python 06_final_check.py --results_dir ./results --true_pools "SRR17037621,SRR17037622,SRR17037623"

6. **Optional: Metadata-based validation (if labels/runinfo are available)**
   python 05_check_metadata.py --results_dir ./results --metadata ./data/metadata.csv

## Notes
- Helper modules `utils_*.py` are included and used by the numbered scripts.
- Jensen–Shannon distance (sqrt of JS divergence) is used throughout the geometry steps.
- Ground truth pool IDs are passed explicitly via `--known_pools` (03) and `--true_pools` (06) to avoid ambiguity.
- FASTQ files and the `results/` directory are not committed to GitHub (see `.gitignore`).


