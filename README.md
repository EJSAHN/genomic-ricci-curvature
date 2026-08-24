# Genomic Ricci Curvature

Reference-free screening of sample heterogeneity in genotyping-by-sequencing data using hashed k-mer profiles, Jensen–Shannon distance, sample graphs, and discrete Ollivier–Ricci curvature.

The repository contains the analysis code for a coffee case study and an independent finger millet benchmark. The workflow is intended for upstream quality-control triage: it ranks libraries with unusual bridge-like geometry and does not assign a biological or technical cause to a flagged library.

## Main findings represented by the workflows

- The idealized coffee sketch-space calibration is reproducible and provides a useful controlled benchmark.
- True read-level mixture detection is more difficult than sketch interpolation and is strongly cohort-dependent.
- In the finger millet benchmark, the reference-free score provides modest enrichment under low-prevalence screening, while a leakage-controlled reference-based comparator performs better when a suitable genome assembly is available.
- Negative-curvature incidence is not informative in every graph. Component-level outputs are retained so that betweenness and curvature contributions can be assessed separately.

## Repository layout

```text
config/                         Analysis parameters and sample manifests
pipelines/coffee/core/          Coffee preprocessing, geometry, calibration, and sensitivity analyses
pipelines/coffee/read_level/    True read-level mixture construction and evaluation
pipelines/coffee/rare_event/    Low-prevalence mixture-injection analysis
pipelines/coffee/reference_based/
                                Reference-derived features and leakage-controlled cross-fitting
pipelines/finger_millet/acquisition/
                                Metadata reconciliation and resumable FASTQ download
pipelines/finger_millet/design/ Prespecified source, parent, and read-allocation design
pipelines/finger_millet/reference_free/
                                Full-cohort, batch, and rare-event reference-free analyses
pipelines/finger_millet/reference_based/
                                Independent marker discovery and cross-fitted reference comparator
tools/                          Release checks and Supplementary Data S1 builder
```

Plotting programs and figure files are not included. The analysis scripts write tabular outputs used by the accompanying study.

## Data

Coffee GBS data: NCBI SRA BioProject `PRJNA783534`  
Finger millet GBS data: NCBI SRA BioProject `PRJNA791522`  
Finger millet reference assembly: `GCA_032690845.1`

Raw reads and reference indexes are not stored in this repository.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
```

Bowtie 2 and the NCBI Datasets command-line tool are required only for the reference-based workflows.

## Reproducing the analyses

Each script provides `--help`. The principal stages are:

1. Run the coffee core workflow on paired FASTQ files.
2. Generate disjoint true read-level mixtures and evaluate batch and rare-event screening.
3. Build reference-derived features from independent read segments and evaluate them by nested cross-fitting.
4. Acquire and audit the finger millet panel.
5. Create the prespecified finger millet benchmark design before evaluating performance.
6. Run reference-free and reference-based finger millet benchmarks on the same generated libraries.
7. Build `Supplementary_Data_S1.xlsx` from the audited result directories or result archives.

Example coffee preprocessing command:

```bash
python pipelines/coffee/core/01_run_preprocessing.py \
  --fastq_dir data/coffee/fastq \
  --outdir results/coffee/core
```

Example Supplementary Data build:

```bash
python tools/build_supplementary_data_s1.py \
  --base_workbook templates/Supplementary_Data_S1_base.xlsx \
  --coffee_read_level results/coffee/read_level \
  --coffee_rare_event results/coffee/rare_event \
  --coffee_reference results/coffee/reference_crossfit \
  --finger_design results/finger_millet/design \
  --finger_reference_free results/finger_millet/reference_free \
  --finger_reference results/finger_millet/reference_based \
  --output Supplementary_Data_S1.xlsx
```

## Interpretation

A high screening score identifies unusual graph placement. Follow-up should use library metadata, technical replicates, read-level inspection, and reference-based analyses when appropriate. The score should not be interpreted as direct evidence of contamination, hybridization, or biological admixture.

## Reproducibility notes

- Random seeds and read allocations are written to output manifests.
- Generated read-level libraries use disjoint physical read pairs within each benchmark.
- Reference-based marker discovery uses read segments that do not overlap generated benchmark libraries.
- Cross-fitted evaluations exclude held-out rows from marker selection, PCA fitting, and score scaling.

## Disclaimer

Mention of trade names or commercial products is solely for the purpose of providing specific information and does not imply recommendation or endorsement by the U.S. Department of Agriculture. USDA is an equal opportunity provider and employer.
