# R02 interpretation: missing references are not negative chemical evidence

## Observation

The experiment fixed the existing 90%-fit checkpoint and removed every held-out connectivity key from the spectral reference library. The structure-only catalog remained inclusive. All 10,000 previously examined keys were excluded before selecting 160 calibration and 320 audit keys. The declared calibration procedure selected the R01 neural-plus-mass ranker rather than any candidate spectral fusion.

On the same 320 audit molecules, the original production blend reached MRR@25 0.0774316, Top-1 0.03125 and Recall@25 0.2875. R01 reached MRR@25 0.6890581, Top-1 0.55 and Recall@25 0.965625. Full values and evidence identifiers are in R02-results.json. This is not a Kaggle score or a comparison on arbitrary absent molecular graphs.

## Mechanism

The original score averages the best library match for each query spectrum and then adds 0.75 times this value to 0.25 times the neural fingerprint similarity. A correct candidate without a reference spectrum receives spectral score zero. An incorrect candidate with an available spectrum can receive a positive value even from a weak accidental overlap. The global weight therefore confounds **reference availability** with **chemical support**. R02 deliberately exposes this problem by making exact references unavailable for all query structures.

This does not show that library evidence is useless when a genuine high-quality match exists. It also does not justify replacing the entire system with the ranker and claiming an unconditional hidden-test improvement. The real competition mixes regimes with different reference and structural coverage. A new combination must demonstrate that it does not destroy high-confidence library matches while improving reference-absent cases.

## Next experiment requirements

Use unused molecular keys and keep all spectra of a key out of neural fitting. Evaluate two separate conditions for each selected molecule: reference-present with independently held-out query spectra/measurement conditions, and reference-absent with all key-equivalent spectra removed from every library. Report each condition separately rather than choosing an arbitrary mixture weight and hiding regressions in an average.

Gate library influence using evidence available without test labels: matched peak count, explained intensity, score margin, agreement across ion modes/energies and observed reference availability. Keep the R01 neural-plus-mass score as a distinct fallback, not a score suppressed by missing library data. Fit any thresholds or calibrator only on the calibration partition, lock them before audit, and compare against both original blend and R01 alone. Do not reuse the spent R02 audit to select thresholds.

A separate catalog-excluded benchmark remains necessary for generation of molecular graphs absent from the catalog. Neither a gating change nor an inclusive-catalog MRR establishes de novo recovery.

## Separation from Kaggle

The first official submission uses the unchanged baseline notebook, version 1. R02 was run without reading the Kaggle test or scores and did not alter that notebook. Later submissions should carry different explicit versions and be justified by predeclared local measurements. The submission guard limits all account attempts to two per rolling 24 hours, below the five-per-day limit observed in the official rules. No repeated submissions or public-score probing are needed to establish this failure mechanism.
