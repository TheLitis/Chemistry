# R07 candidate: verified on Windows and offline Kaggle, not yet officially scored

## The actual candidate

R07 combines the complete spectral-reference branch, a fingerprint predictor,
a robust mass prior, and an offline COCONUT August 2026 structure catalog.
The calibration-selected fingerprint head was the simpler V1 Tanimoto variant,
not the largest network. The selected settings are spectral weight 1.0,
mass weight 0.5, and external-candidate penalty 0.2. A mass offset of
1.4060715323698454 ppm was estimated on calibration data; the mass prior is
Student-t (3 degrees of freedom), scale 5 ppm and 0.001 Da minimum.

No true query formula or test structure is used. This is still structure
retrieval, not de novo graph generation. V1 remains the officially evaluated
champion at 0.147 until an actual better public score is returned.

## Leakage-controlled target-domain study

The completed study used 250 common natural products and 1184 timsTOF spectra.
All 250 structural keys were removed from every fitting source. The V1 and V3
models were trained from scratch, without a pretrained checkpoint. Fitting
used 245319 unique molecules. Selection used 80 target molecules; the other
170 were reserved for audit. Twenty-two previously examined target keys were
placed in calibration rather than presented as new audit data.

| Regime | Independently refitted V1 recipe | R07 |
|---|---:|---:|
| Other-source reference spectra available | 0.9292013216 | 0.9359082421 |
| Correct reference spectra absent, graph in catalog | 0.1472045237 | 0.2753418234 |
| Graph excised from original catalog, actual external lookup allowed | 0 | 0.1311611695 |

These are MRR@25 values on the same 170 molecules in controlled scenarios,
not Kaggle results. The first regime's paired 95% interval includes zero;
do not claim a statistically established gain there. The other intervals
are [0.0988685530, 0.1623652759] and [0.1007615257, 0.1660226343].
The third regime demonstrates retrieval from actual external records, not
invention of an unseen graph. The correct candidate coverage there was
0.9941176471, while Top-1 was only 0.0470588235: ranking remains a large gap.

The source study and model selection predated this continuation. They were
reconciled from immutable artifacts, not rerun/reselected to improve numbers.
All 170 raw ranks, metrics and paired bootstrap intervals were independently
recomputed again in the September 17 continuation. Full-corpus refit weights
are separate from these holdout checkpoints.

## Actual deployment verification

Windows cold-archive inference: all 1213 example spectra, 400 molecules and
9965 valid guesses; no empty rows or mass-incompatible fallback. It took
1116.536 seconds. The separate Kaggle Linux preview was offline and took
2356.593 seconds. Its CSV was byte-identical to the Windows output.

Private notebook: `thelindortis/casmi26-r07-timstof-hybrid`, version 1.
Private model dataset: `thelindortis/casmi26-r07-assets-v1`.
The downloadable example data are NOT the scored hidden test. A successful
private preview must never be reported as an official score.

PC release directory:
`C:\ProgramData\ChemistryRunner\artifacts\casmi26\final-r07-v1`

It contains the bundle, `predict_r07.py`, `submission.csv`, its report,
`casmi26-r07.ipynb`, `casmi-r07-candidate.zip`, and acceptance records.
The archive uses the existing pinned CASMI Python environment; the Kaggle
notebook installs pinned dependencies from the existing private offline
wheel dataset. No credential is included in model assets.

## One future competition submission, not repeated leaderboard tuning

`tasks/casmi_r07_submission.py --stage check` passed 290 PC tests and verified
candidate hashes on 2026-09-17 at 02:30 UTC. Kaggle offered five daily slots,
but four historical attempts were still inside our rolling 24-hour window.
The first allowable internal window is 2026-09-17T12:35:06.900000Z, subject to
fresh quota/history checks and any intervening user submissions.

The task accepts exactly version 1 of the private R07 notebook. It journals
an attempt before issuing the request. An ambiguous timeout, absent returned
ref, or failed submission cannot trigger a blind retry. A later invocation
reconciles the existing attempt by exact identity and only reads its status.
`COMPLETE` with an error description or without a numeric score is not success.
Every version's failed and successful attempts count toward the same cap.

## Additional methods investigated, not merged blindly

FIORA OS v1.0.0 was executed in an isolated CPU environment using the pinned
author implementation and safe state-dict loader. All ten author examples
reproduced the expected spectra, and six additional positive/negative-mode
energy probes produced valid spectra. This is software/model execution
verification, not a CASMI identification benchmark. Its released instrument
vocabulary is HCD and does not establish timsTOF transfer. It remains outside
the frozen R07 candidate pending a proper supported-adduct/domain comparison.

FRIGID checkpoint metadata was retrieved from Zenodo. The weight record says
CC-BY-NC-4.0 whereas the code license is Apache-2.0; competition deployment
and redistribution eligibility must not be inferred from the code license.
The public checkpoint archive exists, but it was not downloaded or executed.
MS-GPT's sharing page was retrieved, not its weights. Neither generator has
been integrated or reported as benchmarked. Compatible radius-2 fingerprints
and predicted formulas remain explicit dependencies.

## Remaining work, not completed claims

PubChem-scale expansion, genuinely out-of-catalog generation, independent
formula prediction, and a validated forward-spectral reranker remain open
implementation/research steps. No score near 1.0, private-test success, or
exhaustive architecture optimum is claimed by this candidate release.

Evidence: `research/R07-target-domain-results.json`,
`evidence/2026-09-17-r07-final-candidate.json`, and
`research/2026-09-17-public-model-feasibility.json`.
