# V3: implemented candidate, not the official champion

The source is in `main`. Trained assets stay outside Git, under
`C:\ProgramData\ChemistryRunner\artifacts\casmi26\highres-v3\bundle`.
V1 remains the best officially evaluated version (0.147 MRR@25). V2 returned
0.112. V3 has no official score yet. Do not promote it solely on local results.

## Implemented components

`features_v3.py` keeps the original 4104 inputs and adds 4096 intensity-weighted
fractional-mass moments for fragments and neutral losses. These add information
but are not a lossless encoding of the peak list. `model_v3.py` trains/exports
three complementary targets: Morgan radius 2 / 2048 bits, Morgan radius 3 /
4096 bits, and atom pairs / 8192 bits. `cache_v3.py` builds hash-checked arrays
from training data only. `pipeline_v3.py` separates fitting, calibration, audit,
and full-data refit. `inference_v3.py` groups all supplied spectra by the actual
compound ID and returns up to 25 tautomer-distinct structures per compound.

The root `predict.py` now dispatches V1/V2 and V3 from the bundle manifest,
including `--bundle=...`. Ambiguous manifests, missing feature definitions,
invalid confidence settings, and identical train/test inputs are rejected.
Original checkpoints and the measured V1 champion are not overwritten.

## Measured training result

The experiment excluded 11,894 previously examined keys. Fixed 10-epoch updates
were selected on 800 new calibration molecules and audited on 1600 other
molecules. Their spectra did not enter holdout-model training; their structures
DID remain in the candidate catalog. This measures retrieval, not de novo.

| Variant | Audit MRR@25 |
|---|---:|
| Frozen old-input model with R01 ranking | 0.6497576962 |
| Old-input model plus 10 epochs | 0.6576373713 |
| Fractional-mass input, Morgan target | 0.6665814648 |
| Multitarget training, Morgan-only ranking | 0.6736809397 |
| Selected multitarget ranking (0.5/0.25/0.25) | 0.6802474876 |

The selected model reached 55.3125% Top-1 and 96.9375% Recall@25, with paired
MRR improvement 0.03048979 and bootstrap 95% interval [0.02100214,0.04018395].
A slightly higher audit value for other weights was NOT used to replace the
calibration winner. A separate full-data refit used 276,241 molecular records,
10 epochs, CUDA mixed precision, and 11,553,280 parameters.

## Reduced-spectrum diagnostic

The frozen checkpoints were tested with one, at most three, and all available
spectra. These are spent audit keys used for diagnosis, not new model selection.

| Budget | Frozen MRR | V3 MRR |
|---|---:|---:|
| One spectrum | 0.5228460010 | 0.5504800021 |
| At most three | 0.6178260490 | 0.6446302819 |
| All | 0.6500119462 | 0.6794392538 |

The all-spectrum diagnostic uses median observed mass; the original audit uses
a cached mean. Do not confuse those slightly different scores. The Bruker-only
cohort has just 30 molecules and 61 spectra; its improvement interval includes
zero, so hidden-instrument transfer is unproven.

## Unified prediction

With the existing environment, actual extracted data directory and bundle:

```powershell
$repo = "$HOME\Chemistry"
$python = 'C:\ProgramData\ChemistryRunner\envs\casmi26\python.exe'
$data = 'C:\ProgramData\ChemistryRunner\data\external\enveda-CASMI26-molecule-id-mass-spectra'
$bundle = 'C:\ProgramData\ChemistryRunner\artifacts\casmi26\highres-v3\bundle'
& $python "$repo\predict.py" --test "$data\test.parquet" --train "$data\train.parquet" --bundle $bundle --sample-submission "$data\sample_submission.csv" --output "$repo\submission-v3.csv"
```

Use the actual extracted directory, not the ZIP. The acceptance task discovers
it rather than assuming a location. A matching template preserves ID order;
changed hidden-rerun IDs come from the current test mount. No test IDs or
predictions are embedded in model assets. The notebook embeds source and uses
pinned offline Linux wheels; a V3 Kaggle/Linux execution still needs verification.

`tasks/casmi_v3_acceptance.py` runs this root entry point, compares CSV/model/input
hashes with the previously verified notebook output, and creates the release in
`highres-v3\release-verified`. Acceptance results are stored separately from
training results. Read-only Kaggle history/quota checks do not create submissions.

## Remaining scientific limitations

V3 is still catalog retrieval: a ranker cannot return a graph absent from its
candidate universe. External coverage, independent instrument transfer and de
novo generation remain separate research problems. The R03 confidence gate
(0.95/0.05) was preserved, not revalidated by the neural-only audit. Software
tests and valid CSV rows do not establish a high hidden score or the 1.0 goal.

Keep the rolling 24-hour cap of two submissions, even if Kaggle reports more
available official slots. Each candidate needs a separate at-most-once journal.
Numerical evidence: `research/V3-results.json`, `research/V3-spectrum-budget.json`;
raw ranks remain in the cited Actions artifacts and versioned PC directories.
