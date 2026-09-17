# R08: frozen forward-spectrum evidence against the complete R07 recipe

Do not alter the working R07 bundle or any Kaggle submission. The user-reported
improvement is independently confirmed as public score 0.188, ref 56292572,
without a scoring error (metadata did not identify the kernel; provenance is
reconciled separately). The old automatic R07 submit is disabled to avoid a
manual/automatic duplicate.

## Question and bounded experiment

Can a pinned public FIORA model supply useful extra evidence for mass-compatible
structures, including external COCONUT structures, without sacrificing the
spectral-library advantage? No new generator or formula oracle is assumed.

Prepare exact R07 scores using its independent from-scratch NP-excluded model.
Replay all 170 old audit rows exactly before analyzing any new method. Preserve
the original 80 calibration keys; the 170 old NP audit keys are explicitly reused
diagnostics, NOT a new untouched benchmark. Also fix 128 previously unused
hash-holdout enveda-180 timsTOF keys, excluded from fitting and prior experiments,
for a genuinely fresh transfer audit. They are not novel natural products.

Use the union of the best 32 candidates per available/absent/external-recovery
regime, with no answer insertion. Predict supported [M+H]+/[M-H]- spectra at
10,20,40,60 eV with FIORA e19ef82c9a6cb9dbac92bce23e914008f1aeb44e,
state SHA256 83221e187991f116a8aed1bf272dd656cf31721a177dcbb0239f414c8df31a0e.
The released model's instrument is HCD, not timsTOF; this is the hypothesis under
test, not an equivalent instrument mapping. Its MSnLib pretraining overlap is
not yet audited and must be reported as unknown. Do not remap other adducts.

Exclude the precursor region from matching. Test sqrt-intensity cosine,
unweighted Jensen-Shannon similarity, and explained query intensity, each under
nearest-grid mixed-energy spectra and maximum-over-grid scoring. Calibrate only
an additive weight in 0,0.1,0.25,0.5,1,2 on the 80 calibration molecules. Select
maximum minimum gain across the three regimes, then mean gain, with no-change
as an explicit option. Save this selection before evaluating the 128 new keys.
Use old 170-key results only as diagnostic confirmation, never for reselection.

Failures or unsupported modes retain the R07 candidate scores, not blank rows or
an invented chemical answer. Missing true candidates remain failures. Track
shortlist coverage, supported acquisitions, simulated candidates, failures, time,
and actual score deltas. Do not claim a hidden score from this experiment.

## Execution and verification

New code lives in isolated R08 modules/tasks and versioned artifacts. A
single-molecule GNN cache across energy covariates is allowed only after direct
numerical parity against the unchanged author implementation. Never unpickle
arbitrary model objects; use the reviewed weights_only=True state loader. Do not
change the CASMI environment or old weights: extra pure-python dependencies use
an isolated directory. Test alignment, precursor exclusion, scale/order invariance,
missing evidence, shortlist limits, exact baseline replay, and model parity.

A new submission requires independent evidence and a later explicit quota check.
This research pass itself makes no Kaggle writes.
