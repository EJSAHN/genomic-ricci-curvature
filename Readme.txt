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

## Utility Scripts
- utils_kmer_matrix.py: Core logic for k-mer matrix construction.
- utils_fastq_qc.py: Helper for checking FASTQ quality.
- utils_visualization.py: Helper for plotting PCA/PCoA.