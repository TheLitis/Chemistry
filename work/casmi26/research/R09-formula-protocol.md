# R09-F: elemental evidence, fixed exploratory diagnostic

Question: can exact fragment masses improve R07 candidate ranking through an
independent elemental-composition signal, without a new pretrained model?
This is NOT a new untouched audit. The complete 224 R08B query records and their
candidate sets/base scores are reused. Source SHA256:
`217153e52d50112f47c51de90806691fd1c6dd67c2b45ed6625a9c5d7ef0c629`.

For every actual candidate, enumerate masses of elemental subcompositions under
its atom-count limits. Use only neutral, unlabelled CHNOPS/F/Cl/Br/I candidates.
Only [M+H]+ and [M-H]- query fragments are interpreted; other modes supply no
bonus. No answer formula or answer SMILES from query metadata enters the feature.
No connectivity, valence, fragmentation-tree, isotope-pattern or novel-graph
claim follows from a matching elemental mass.

Fixed settings: 5 ppm / 0.001 Da tolerance; remove peaks at or above precursor
minus 0.5 Da; relative-intensity floor 1% after precursor removal; square-root
intensity weights. Process all eligible observations independently and average
scores by acquisition. Compare (1) unchanged R07; (2) R07 + 0.25 explained mass
fraction; (3) R07 + 0.25 positive excess over three fixed shifted-mass controls.
Controls (-0.2345678,+0.1234567,+0.3456789 Da) are arithmetic controls, not valid
chemical decoys or correctness probabilities. Neither configuration is selected
for production in this experiment. All full candidate sets are preserved.

Maximum 2,000,000 subcomposition states per formula, checked before allocation.
Unsupported or over-budget cases add no evidence, are counted, and never remove
the baseline candidate. Compute once per candidate formula per query group.

Report molecular MRR@25, Top1, Recall25, coverage and paired descriptive bootstrap
intervals separately for both R08B cohorts and all three library regimes. Record
source/code hashes, numerical features and raw ranks. Do not compare these R07
variants to an incomplete R08B score where unsimulated FIORA evidence was replaced
by zero. No new Kaggle submission is authorized by this diagnostic alone.

Conceptual primary references (no external implementation copied or executed):
- Goldman et al., MIST-CF: https://arxiv.org/abs/2307.08240
- Goldman et al., Prefix-Tree Decoding: https://arxiv.org/abs/2303.06470
- SIRIUS background: https://bio.informatik.uni-jena.de/software/sirius/

This small original implementation does not invoke SIRIUS code/web services or
restricted model weights. In particular, it is not a reproduction of SIRIUS.
The existing R07/R08B submissions, datasets, notebooks and weights stay frozen.
