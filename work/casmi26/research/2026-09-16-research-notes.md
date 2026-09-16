# CASMI26 research: measured ranking gain and the route beyond a closed catalog

Research date: 2026-09-16. This records completed R01 work, not a claim that the
competition target has been achieved. No new Kaggle submission was made in R01.

## What changed in our scientific understanding

The existing production system is a trained catalog-retrieval baseline, not a
competitive de novo decoder. Its validation includes the true structure in the
candidate catalog, while its holdout excludes that structure's spectra from
model fitting. This is a useful known-structure/no-reference-spectrum test, but
it cannot demonstrate recovery of molecules missing from the catalog.

Also, validation.json evaluated a neural head alone, not the production blend
of 0.75 spectral score plus 0.25 neural score. That blend has not been established
as optimal. The 400 downloadable test examples are not a hidden-score estimate.
The numeric features currently do not explicitly encode instrument identity;
calling the current network instrument-conditioned would be inaccurate.
Sources: ../casmi26/production.py, ../casmi26/learning.py, ../COMPETITION.md.

## Completed R01 experiment

Predeclared protocol: R01-protocol.md. Executable: ../../../tasks/casmi_rank_research.py.
Evidence: R01-results.json and [completed ChemistryPC run](https://github.com/TheLitis/Chemistry/actions/runs/35051249403).

The checkpoint was frozen. From the original 10% structure-disjoint holdout,
we excluded all 4000 previously examined keys, then allocated 2000 other keys
to calibration and 4000 different keys to audit. We compared five scoring
functions crossed with three mass-prior weights on calibration only. The
winning configuration and audit-key hash were written before audit scoring.

| Same 4000-key audit | Old soft Tanimoto | Selected log score + mass prior |
|---|---:|---:|
| MRR@25 | 0.5830135606 | 0.6642581387 |
| Top-1 exact key | 0.45425 | 0.53250 |
| Recall@25 | 0.91800 | 0.95750 |

The MRR increase is 0.08124458 (13.94% relative); 2000 paired bootstrap resamples
give a 95% interval [0.07219154, 0.09011659]. Top-1 rises by 313 correct molecular
keys out of 4000. This interval describes this audit population, not the hidden
natural-product domain. The prior result 0.5668 was measured on a different
4000-key set; do not subtract it from the new result as a paired comparison.

The winning score computes, for candidate fingerprint f and network outputs p,

    L(f,p) = sum_b f_b * log(p_b / (1-p_b))
    S = standardize_across_candidates(L) - 0.5 * ((mass - observed_mass)/sigma)^2
    sigma = max(0.001 Da, observed_mass * 5 ppm)

The omitted sum(log(1-p)) term is common to all candidate fingerprints. This is
an independent-bit pseudo-log-likelihood, not a validated chemical posterior:
fingerprint bits are dependent and p was trained with weighted BCE plus cosine
loss. Ties and equivalent tautomer keys are handled deterministically.

A proposed pure-BCE inverse-weight correction did NOT improve calibration:
unweighted Bernoulli plus the same mass prior had MRR 0.56468 versus 0.65687 for
the selected uncorrected variant. We reject that correction for this model.

A mass-near shuffled-posterior control scored 0.18176 MRR versus 0.66426 with
correct query posteriors. The mass-only baseline scored 0.25075. These controls
support spectrum-specific usefulness, not solely selection by precursor mass.

97 software tests passed on ChemistryPC. No retraining, production overwrite,
visible test read, external-model execution, or Kaggle score was involved.
The result qualifies a new rank head; it does NOT automatically validate a new
spectral/neural ensemble. The original production checkpoint and bundle remain
unchanged. R01 audit keys are now marked as examined in persisted artifacts.

## Primary-source research and integration constraints

### MS-GPT (2026-07-26 preprint)

[Paper](https://arxiv.org/abs/2607.23607), [code and asset instructions](https://github.com/VIKI623/MS-GPT).
The architecture conditions a molecular language model on a 4096-bit radius-2
Morgan fingerprint and a KNOWN molecular formula, and samples across calibrated
fingerprint-density thresholds. Its released code reports Apache-2.0; separately
verify downloaded checkpoint terms and provenance before redistribution. A
150.1M-parameter decoder is far smaller than a general LLM, but the authors'
recommended reproduction GPU is not evidence of memory usage on our RTX 3070.
Their reported MassSpecGym Top-1/Top-10 23.91/28.65% are known-formula de novo
results, NOT directly comparable to our inclusive-catalog MRR.

The promising transferable idea is robust conditioning on uncertain fingerprints,
not taking a single arbitrary 0.5 threshold. Our 2048-bit output cannot simply
be padded into the required 4096-bit semantics. Formula uncertainty also needs
its own tested stage; use several inferred formulas, never a held-out label.

### FRIGID (2026-04-17 preprint)

[Paper](https://arxiv.org/abs/2604.16648), [repository](https://github.com/coleygroup/FRIGID).
It uses a MIST fingerprint encoder, formula-conditioned masked molecular decoder,
and optional ICEBERG forward-fragmentation refinement. The code is Apache-2.0;
the README points to Zenodo record 19685145 for weights. That external asset page
was not fetched successfully during this review, so availability/license of
those bytes is not verified here. Do not mark a decoder integrated on that basis.
Its useful design is candidate -> predicted spectrum -> inconsistency-guided
refinement. It is not equivalent to our simple bond-cut heuristic. Known-formula
benchmarks and substantially different compute also prevent direct score transfer.

### MSFlow (2026-02-23 preprint)

[Paper](https://arxiv.org/abs/2602.19912), [repository](https://github.com/ghaith-mq/MSFlow).
It uses a MIST-like encoder to CDDD descriptors followed by SAFE flow-matching
molecular decoding. This is a different conditioning space from our Morgan bits.
The paper says non-commercial availability while the current repository says
MIT. Treat code and checkpoint permissions separately; the discrepancy is
unresolved, not a reason to claim the pretrained assets are competition-ready.

### MetGenX (Nature Communications, 2026-04-20)

[Primary article](https://www.nature.com/articles/s41467-026-72149-6).
Template-guided generation is relevant because it can exploit near structural
neighbors. But the prominent 55.9% NIST Top-1 figure is from DATABASE-RESTRICTED
mode on 1388 spectra that retrieved templates, rather than all 1500 test spectra.
Do not market that number as unconstrained recovery of arbitrary new molecules.
This suggests a separate analog-to-structure branch, not replacing honest
out-of-catalog evaluation with known-database success.

### MIST-CF and MSNovelist operational checks

[MIST-CF](https://github.com/samgoldman97/mist-cf) ranks formula/adduct candidates
from spectra without a spectrum database. Its public model currently covers
positive mode only; applying it to all negative CASMI26 spectra would be an
unsupported extrapolation. NIST-trained checkpoints are license-dependent;
prefer the public-data checkpoint for an initial reproducible baseline.

[Archived MSNovelist](https://github.com/zamboni-lab/MSNovelist) explicitly says
its old server-backed repository is no longer operative; the maintained
integration is in SIRIUS. Do not waste a local setup attempt on that archive
or assume a service-dependent tool will work in an offline Kaggle notebook.

## Ordered next experiments (not yet executed)

1. Domain-aware ensemble validation. Use enveda-np-examples (documented as the
   same measurement pipeline as hidden data) and exclude every corresponding
   connectivity key from training across ALL libraries. Evaluate actual spectra,
   not only mean features from arbitrary sources. Tune old/new neural heads and
   spectral evidence on separate calibration keys. Record polarity/adduct strata,
   spectrum counts and mass-error tails before selecting a blend.
2. Out-of-catalog evaluation and generation interface. Remove held-out molecular
   graphs from spectral references AND candidate catalogs. Evaluate predicted
   formula recall separately from oracle-formula diagnostics. Add a 4096-bit
   fingerprint head or compatible public encoder, validate semantics and assay
   one permitted molecular decoder on a bounded public-data subset.
3. Broader catalog and forward model. Separate library-known, catalog-only and
   generated candidates. Measure candidate recall before reranker accuracy.
   Evaluate ICEBERG-like forward scores with train-key exclusions and domain
   checks. Choose methods on local evidence, not repeated leaderboard probing.

First Kaggle notebook execution remains necessary for submission acceptance and
hidden evaluation. The current R01 result alone does not establish the default
ensemble should be changed or the official score will rise. Upload only a
private train-derived bundle and a notebook that reads the current hidden mount;
never upload a precomputed visible CSV as a substitute for inference.
