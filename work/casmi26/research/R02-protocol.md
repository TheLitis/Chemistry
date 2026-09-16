# R02 — spectral/neural fusion, fixed before scoring

Question: does the R01 fingerprint ranking improvement survive fusion with spectral-reference evidence?

Use the existing 90%-fit holdout checkpoint, never the full-refit inference checkpoint. Verify its hash against R01. Exclude the previous 4,000 evaluated keys and R01's 2,000 calibration + 4,000 audit keys. Order remaining eligible holdout keys by SHA256('ensemble-r02:' + key), choosing 160 calibration and then 320 audit keys. Keep all admitted spectra across raw representations of each selected connectivity key.

Remove **every** held-out connectivity key from spectral references across all libraries. The structure-only catalog remains inclusive. This is known-catalog retrieval without exact reference spectra, not de novo generation, not a Bruker-only domain evaluation and not a hidden Kaggle score. Read only training data and training-derived caches. No test inputs, Kaggle scores, submissions, retraining or production changes are used in R02.

Candidates: the fixed production mass window (20 ppm, minimum 0.005 Da), using the median inferred neutral mass of all admitted query spectra. Use the existing production spectral matching and frozen model. Missing candidates count as failures, not exclusions.

Prespecified scores:
- Old blend: 0.75 * spectral score + 0.25 * soft fingerprint Tanimoto.
- R01: standardized Bernoulli fingerprint log-odds score + fixed mass prior.
- Fusion: R01 + w * standardized spectral score for w in {0.5, 1, 2, 4}.

Choose one variant using calibration MRR@25 only. Save the chosen configuration and audit-key hash before computing audit ranks. Evaluate the selected variant, old blend and R01 on the same 320 audit keys. Report MRR@25, Top-1, Recall@25 and a paired bootstrap interval (2,000 repeats) for selected minus old blend. This audit is spent after use and must not be reused for selecting later variants.

Do not promote a variant solely because its label says fusion. A null or negative spectral result is useful evidence. No improvement on the hidden Kaggle test is implied by this experiment.
