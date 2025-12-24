# Alignment-Free Geometry Analysis Code

This repository contains the source code for the paper "Alignment-Free Geometry Reveals Cryptic Admixture in Polyploid Genomes via Ollivier-Ricci Curvature".

## Requirements
- Python 3.8+
- Libraries: numpy, pandas, scipy, matplotlib, seaborn, networkx, scikit-learn, tqdm, openpyxl

## Pipeline Execution Order

Please run the numbered scripts in the following order to reproduce the results.

1. **Preprocessing & Distance Matrix (Figure 1-2)**
   python 01_run_preprocessing.py --fastq_dir ./data --outdir ./results

2. **Geometry & Curvature Analysis (Figure 3)**
   python 02_run_geometry_analysis.py --results_dir ./results

3. **Synthetic Validation (Figure 4)**
   python 03_run_synthetic_validation.py --fastq_dir ./data --outdir ./results/synthetic

4. **Robustness Sweeps (Figure 5)**
   python 04_run_sensitivity_sweeps.py --fastq_dir ./data --outdir ./results/sensitivity --n_boot 30

## Notes
- Scripts 01–06 are self-contained; no separate utils_*.py files are required.
- To avoid ambiguity, pass the known pool SRR IDs explicitly:
  - Synthetic calibration / mapping (03): use --known_pools "SRR...,SRR...,..."
  - Final validation check (06): use --true_pools "SRR...,SRR...,..."
- Jensen–Shannon distance (sqrt of JS divergence) is used throughout the geometry steps.
