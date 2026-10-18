# Genomic Ricci Curvature

Reference-free screening of genotyping-by-sequencing (GBS) libraries using hashed k-mer profiles, Jensen–Shannon distances, sample graphs, and Ollivier–Ricci curvature. The repository contains the analysis code for a coffee case study and a finger millet validation cohort, together with the reference-based comparators used in the associated article.

The Topological Mixture Score (TMS) ranks libraries for follow-up. It does not determine whether a flagged library reflects contamination, pooling, hybridization, or another cause.

## Data

| Cohort | Source | Libraries |
|---|---|---|
| Coffee (*Coffea arabica*) | NCBI SRA BioProject PRJNA783534 (Bawin et al., 2025) | 18 archived libraries |
| Finger millet (*Eleusine coracana*) | NCBI SRA BioProject PRJNA791522 (Brhane et al., 2022) | 83 archived libraries |

The coffee run metadata (`metadata/coffee/SraRunTable.csv`, an unmodified NCBI Run Selector export) describe 12 single-cultivar libraries and six technical replicates of one 12-cultivar pool (SRR17037618–SRR17037623; two ligation replicates × three PCR replicates). The 12 single-cultivar libraries are the sources for all coffee benchmarks. All 18 libraries are included in the descriptive graph. SRR17037623 is titled as an individual-cultivar library in the SRA, but its cultivar, library-name, and replicate fields place it in the pooled series.

Raw reads, reference assemblies, and alignment indexes are not redistributed. Reference accessions are listed in the configuration files.

## Repository layout

```text
pipelines/coffee/        preprocessing, graph analysis, calibration, read-level, rare-event,
                         and reference-based analyses
pipelines/finger_millet/ acquisition, design, reference-free, and reference-based analyses
workflows/               command-line runners (verify, replay, analyze)
config/                  analysis settings and sample manifests
metadata/coffee/         SRA run metadata and input-file checksums
examples/coffee/         derived inputs for numerical replay (distance matrices, scores,
                         synthetic-truth records, reference feature matrices)
reporting/               numerical tables underlying Appendix S1
history/                 read-allocation records needed for exact regeneration (see below)
tests/, tools/           unit tests, package checks, and table export
```

## Requirements

Python 3.10 or later with the packages in `requirements.txt`. `environment-tested.txt` lists the exact versions used for the reported analyses (Python 3.10.19). Bowtie 2 v2.5.5 is required only for the raw coffee reference workflow.

## Checking the package

From the repository root:

```bash
python run.py verify                          # checksums, metadata, endpoint tables (seconds)
python -m unittest discover -s tests -v       # unit tests
python run.py replay --output ../replay_check # numerical replay (a few minutes)
```

`replay` recomputes the calibration graph scores from the included distance matrices, re-evaluates the 210 rare-event graphs, and refits the dual-reference cross-fitting models from the included feature matrices, then compares the results with the reported values. It does not realign reads. The output directory must not already exist.

## Running the coffee analysis from raw data

1. Copy `examples/paths.example.json` outside the repository and enter the local paths to the raw FASTQ files, the generated FASTQ files, the Bowtie 2 executable, and both reference FASTA files and index prefixes.
2. Review the planned steps:

   ```bash
   python run.py analyze --config ../paths.json --output ../coffee_analysis --target all
   ```

3. Run them by adding `--execute`.

Targets are `calibration`, `sensitivity`, `restore-generated`, `batch`, `rare`, `reference`, and `all`. Settings are fixed in `config/coffee/correction_design.json`. Input files are checked against the recorded checksums before use, and completed stages are reused only when their inputs and settings match.

### Read allocation

The generated read-level libraries were sampled in a single allocation of 365 libraries, of which the 180 libraries built from the 12 single-cultivar sources are analyzed (scenario label `conservative` in file and sample names). To regenerate exactly the same reads and the same marker-discovery exclusions, the full allocation is required; `history/` contains the source manifest that defined it. `history/` is not the current source classification, which is `config/coffee/coffee_sample_manifest.tsv`.

### Reference-based comparator

Marker discovery uses 100,000 read pairs from each of the 12 sources and excludes every generated-library read and the read segment reserved for scoring the archived libraries. Candidate alleles require at least four alternate observations in total and either observations in at least two sources or at least four alternate observations in one source. Models are fitted with one generated replicate held out at a time; the held-out libraries are excluded from marker selection, PCA fitting, scaling, and calibration. Because the same source genotypes recur across replicates, this tests performance on unseen reads, not on unseen genotypes.

## Finger millet

The finger millet modules are in `pipelines/finger_millet/` with settings in `config/finger_millet/`. Each script lists its options with `--help`.

## Reporting conventions

- Standard deviations are sample standard deviations (n − 1) across five replicate summaries. Rare-event metrics are averaged within each replicate first.
- Sensitivity analyses use 30 seeded calibrations per condition (directories `boot_01`–`boot_30`); these are repeated calibrations with new synthetic parents and weights, not bootstrap resamples.
- Columns named `neg_orc_incidence` contain the cumulative magnitude of negative curvature, not a count of edges.

## Version history

See `CHANGELOG.md`. Version 1.0.0 corresponds to commit `8f9e3954a69d004e6fce9874030ab35eed37a5c3`.

## Citation and license

Please cite the associated article (see `CITATION.cff`). The software is distributed under the terms in `LICENSE`; `NOTICE` applies. Sequence data remain subject to the terms of their original repositories.
