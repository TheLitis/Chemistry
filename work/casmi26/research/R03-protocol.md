# R03: paired reference-present / reference-absent confidence gate

Predeclared before execution on 2026-09-16. No test spectra, hidden labels or
Kaggle scores are used to select configurations. Production stays unchanged.

Select 600 fresh 10%-holdout connectivity keys, excluding initial validation,
R01 and R02 keys. Each must have >=2 distinct normalized spectra. Partition
acquisitions by source/instrument if possible, then adduct/energy, otherwise
by deterministic normalized-spectrum signatures. Exact normalized duplicate
spectra never cross query/reference. Use 200 keys for calibration, 400 for audit.
The frozen holdout model has never fitted any query key.

Score each molecule twice: with its withheld reference acquisition available
and with all reference spectra of its connectivity key absent. Other reference
candidates are unchanged. Structural catalog includes the answer by design.
Aggregate equivalent-tautomer references by connectivity key.

Compare original 75% spectral + 25% Tanimoto blend, R01 neural/mass head,
and a single-candidate promotion gate: spectral threshold in
[.7,.8,.9,.95,.99], distinct-connectivity runner-up margin [0,.05,.1,.2].
A gated decision promotes only the best unambiguous match; remaining candidates
retain neural order. No negative penalty for absence of reference evidence.

Choose on calibration only, maximizing balanced mean MRR across regimes,
subject to absent-reference MRR >= R01 minus .005. Include no gate as an option.
Lock choice before audit. Audit compares selected, R01 and original blend.
Bootstrap molecular paired-average reciprocal ranks (not independent duplicate
regime rows). Promote only with positive paired 95% interval versus R01 and
no-reference audit degradation <= .01. A failed gate is a research result,
not permission to tune thresholds on the audit.

This is catalog retrieval and mixed-acquisition evaluation, not recovery of
new molecular graphs or proof of a hidden Kaggle improvement. Record exact
query keys, code/model hashes, missing data and acquisition-split composition.
