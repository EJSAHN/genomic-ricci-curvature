# Changelog

## 1.1.0

- Coffee benchmark sources are now the 12 single-cultivar libraries defined by the SRA cultivar, library-name, and replicate fields. Version 1.0.0 also used SRR17037623, which belongs to the pooled series. All coffee calibration, sensitivity, read-level, rare-event, and reference-based results were recomputed.
- Added runners for verification, numerical replay from derived inputs, and raw-data analysis with checksum-checked inputs and resumable stages.
- The expected number of marker-discovery sources is now a parameter. The marker rule and model code are unchanged.
- Reported standard deviations use n − 1 throughout.
- Figure-generation code is no longer included.

## 1.0.0

Commit `8f9e3954a69d004e6fce9874030ab35eed37a5c3`.
