# R08B: prospective fixed-score confirmation

## Question and unchanged baseline

Does the exact, bounded FIORA increment improve full R07 on previously unexamined
molecules, both on a target-instrument source and when external-catalog recovery
is possible? Preserve the officially scored R07 (0.188); no model/weight changes,
new hyperparameter search, Kaggle upload, or submission is part of this task.

The original R08 promotion gate remains FAILED. Its fresh timsTOF cohort had no
correct structures available in the external recovery catalog, so neither ranker
could succeed in that regime. This does not license relaxing the old gate on the
same outcomes. R08B uses new keys and a prospectively declared, separate test.

## Populations and exclusions

Use 128 previously unexamined hash-holdout molecular keys with enveda-180 spectra,
then 96 different previously unexamined keys also present in the fixed public
COCONUT slice produced before this experiment. The second cohort is deliberately
conditional on external coverage, NOT a representative sample of all molecules
or all of COCONUT. It uses one deterministically selected source per key and can
contain mixed instruments. Exclude every previously used calibration/screen/audit
key recoverable from the project protocols/rank files and all 250 NP examples.

Use the original independent R07 model, trained from scratch outside the target
keys and hash holdout. Recompute its exact training-key digest before evaluation.
Use every valid measurement of the selected source. Other-source references may
be available; remove identical query spectra globally. For the absent regime,
remove references of all new evaluation keys. For external recovery, additionally
remove those keys from the original structural catalog; actual independent public
records may restore them. Never force the known answer into a candidate list.

Freeze the exact key lists, input hashes, query sources and scoring contract on
the PC before candidate scoring. Missing requested cohorts or usable observations
stop the experiment rather than silently resampling based on outcome.

## Fixed score and exact top-25

Use full R07's already selected spectral/neural/Student-t-mass/external-source
recipe, plus 0.25 * cosine_nearest from FIORA OS v1.0.0, source commit
`e19ef82c9a6cb9dbac92bce23e914008f1aeb44e`. No recalibration on R08B is permitted.
FIORA uses its supported [M+H]+ and [M-H]- measurements; otherwise the increment
is zero. Model prediction failures are recorded and contribute no evidence.

Remove the top-32 simulation cap. Because the increment lies in [0,0.25], evaluate
only candidates whose upper score can still enter top-25. Emit a per-query and
per-regime certificate verifying strict exclusion of the unknown tail. This
certifies fixed-score ranking, not chemical correctness. Reuse existing simulations
only for the same graph string and supported modes, and keep extensions in a
separate cache. No costly old computation needs to be repeated merely for logging.

## Separate prospective candidate gate

Require all of the following: paired molecule-bootstrap lower 95% MRR improvement
above zero for (1) fresh timsTOF/absent references and (2) external-covered/external
recovery; lower bound above -0.03 for external-covered/available references; true
mass-window external coverage at least 0.80. Bootstrap uses 2000 paired draws,
seed 26091603. Report every regime and all failures, not only qualifying results.

A pass means eligible for further candidate integration, not automatic replacement
of the official champion or permission to claim a hidden score. A failure leaves
the champion unchanged. Do not reselect weights after seeing this result.

## Limits of the evidence

FIORA pretraining membership is not fully known. Our predictor is held out, but
this does not establish strict full-system novel-molecule generalization. The
external-covered cohort is selection-conditioned and potentially on instruments
unlike hidden timsTOF. Retrieval cannot create a graph absent from its universe.
Intervals describe sampling of these fixed models and cohorts, not variability
across training seeds or uncertainty in the private leaderboard.

Executable task: `tasks/casmi_r08b_confirmation.py`. Validation checks:
`work/casmi26/tests/test_r08b_confirmation.py`. Test red phase: five expected
missing-implementation failures; green phase: five passing helper/protocol tests.
