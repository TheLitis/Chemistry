# R01: predeclared frozen-ranker experiment

Question: can a better interpretation of existing fingerprint outputs improve
known-catalog identification without retraining? The old loss used weighted BCE
plus a cosine objective. Sigmoid outputs are not established bit probabilities.
The pure-BCE optimum q=w*p/(1-p+w*p) motivates the approximate correction
p=q/(w*(1-q)+q). Its validity for the combined loss must be measured, not assumed.

Fixed inputs: official TRAIN-derived cache, holdout-model.npz, and prior
validation.json. Never use model.npz (full refit) for this test. Never read
visible/hidden test spectra, answers, or Kaggle scores. Preserve current production.

Exclude the 4000 previously examined validation keys. From other members of the
same deterministic 10% holdout, choose 2000 calibration keys and 4000 audit keys
by SHA256(rank-research-r01:key). Select among 5 scoring functions crossed with
3 mass-prior weights using calibration MRR only. Write the selected configuration
and audit-key hash before evaluating the audit. Report only baseline and selected
configuration on audit, plus mass-only and mass-near shuffled-posterior controls.
Use 2000 paired bootstrap samples for the MRR difference interval. No claims
about unseen-graph generation, all-spectrum production ensemble, or Kaggle score.

Selection candidates: soft Tanimoto, cosine, independent-bit log likelihood,
BCE-unweighted log likelihood, BCE-unweighted Tanimoto. Prior weights 0, .25, 1;
fixed Gaussian sigma=max(.001 Da, 5 ppm). Normalize score across each candidate
set only when adding a prior. Deduplicate tautomer connectivity keys at ranking.

Promotion criterion: lower endpoint of paired 95% MRR difference interval >0.
A passing result qualifies only the ranking head; ensemble integration still
needs its own validation. Once inspected, this audit set is spent.

Local verification before publication: 8 tests failed due to absent implementation;
then 8 passed, plus compileall. The PC task runs the whole repository suite again.
