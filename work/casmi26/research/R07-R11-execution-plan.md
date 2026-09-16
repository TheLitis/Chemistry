# R07-R11: full-system research and candidate release

Approved objective: maximize official CASMI26 MRR@25 with all available spectra;
no claimed optimum or hidden score without measurement. Preserve V1 (0.147) as
current official champion. Work on main as requested; use versioned local
artifact/cache directories, never overwrite earlier checkpoints.

## Tasks and acceptance

1. R07: inventory the current source/assets and actual enveda-np-examples
   instrument/adduct cohorts. Read Kaggle history/quota without writes. Audit
   model lineage; previously seen NP molecules are not untouched validation.
2. Construct NP-only queries and genuinely independent structure-level partitions.
   Retrain any learned component whose weights saw the audit molecules, or label
   its outcomes as diagnostics only. Compare the full V1, mass-only, neural-only,
   and joint decisions on identical queries. Separate correct-reference-present,
   reference-excised/structure-present, and catalog-excised settings.
3. R08: preserve V1 library evidence. Calibrate only on the development partition;
   evaluate exactly the frozen configuration on the audit. Record candidate
   coverage, conditional ranking, MRR, Top1 and Recall25 per regime. No invented
   50/50 hidden mixture and no truth-mass selection of query candidates.
4. R09: use the already downloaded public COCONUT snapshot if hashes/license
   verify. Compare expanded versus original catalogs including distractor cost.
   Candidate recovery must arise from actual external records, not answer insertion.
   PubChem preparation is optional only after reproducible source/download review.
5. R10-R11: inspect publicly distributed generator/forward-model weights,
   dependencies, licenses and precise feature contracts. Attempt a bounded real
   smoke evaluation before integration. Reject models that cannot be reproduced
   or that degrade the predeclared validation objective; record the actual reason.
   Formula-conditioned generation must use predicted formula hypotheses in end-to-end
   accuracy (oracle formula is labeled separately). Same-length incompatible
   fingerprints are never silently connected.
6. Select a release using the full pipeline, not a standalone encoder. Report
   all components tested, rejected or still unimplemented honestly. A negative
   finding does not justify replacing V1 with an unproven ensemble.
7. Build a private/offline Notebook and versioned bundle, validate current input
   IDs, all required rows and 1-25 nonempty chemically valid distinct guesses.
   Confirm source/model/data hashes, package execution, peak-memory/runtime and
   fallback reporting. Do not claim nearest-mass guesses are chemically compatible.
8. Competition submission is separate from Notebook execution. Use at-most-once
   journals, direct error_description checks, the live official allowance and a
   unified rolling 24h cap of two submission attempts across all versions.
   No automated retries of ambiguous writes. Full public-score confirmation is
   required to promote an official champion.

## Evaluation discipline

Fixed random seeds; exclusions span all source libraries and frozen pretrained
components. Data from visible test are packaging checks, never a quality audit.
Future audit answers do not select preprocessing, mass tolerance, head weights,
architecture, formula count or final blend. Record exact publication hashes and
any unknown external-pretraining overlap. De novo means the correct graph is not
in the retrieval catalog; absence from generator pretraining is a distinct claim.

## Status

Plan accepted in conversation. Execution evidence is stored under versioned
`C:\ProgramData\ChemistryRunner\artifacts\casmi26\research-r07` and subsequent
release folders. Check evidence files and completed Actions jobs for progress;
this plan alone is not evidence of completed experiments.
