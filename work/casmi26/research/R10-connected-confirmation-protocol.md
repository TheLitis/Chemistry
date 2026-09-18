# R10-G: prospective confirmation on new molecular keys

The completed R10 screen used 224 spent R08B cases. Its results choose exactly
one candidate for a new confirmation: R08B followed by a 0.25 two-cut connected
fragment bonus, reordering only the complete, numerically certified R08B top25.
Do not rescore the new cohorts with alternative weights, one-cut-only rules or
same-formula-only policies after seeing their outcomes.

## Fixed policy

- Incumbent comparator: complete R08B (full independent R07 recipe plus 0.25
  FIORA `cosine_nearest`), with the same full candidate universe in both arms.
- Connected hypotheses: all components after at most two cuts, parent-original
  attached H, at most one H shift, up to 96 heavy atoms / 8192 cut scenarios.
- Evidence: tolerance max(5 ppm, 0.001 Da), 1% relative intensity floor,
  square-root intensity normalization, fixed 1/(cuts+absolute H shift) weight.
- Candidate set: preserve exact R08B top25; no answer insertion, no recall loss.
- No training or changes to the public-scored model or its delivery package.

## Cohorts and lineage

Use `R10G-prospective-20260918` to deterministically select 128 previously unused
hash-holdout molecular keys with spectra in enveda-180, and 96 other unused keys
from a fixed, pre-existing COCONUT snapshot (see the v2 source amendment). All previous research protocols and
rank files contribute to exclusions, including the entire 224-case R08B group
used in the exploratory graph screen. Save the exclusion list and its hash.

The independent R07 model fitting keys must not include any selected key. Its
original artifact hashes and fitting-key hash are revalidated. Query-source
spectra are excluded from the reference library, and byte-level spectrum
signatures deduplicate repeated reference/query measurements. All available
valid measurements in the selected source are used; no selection by prediction
quality. If a specified query has no valid observations, fail without resampling.

Freeze the feature code, preparation code, selection and gates BEFORE building
new candidate scores. Preserve three cases: available references, no references
for selected structures, and external recovery with selected true structures
removed from the original catalog. Missing correct candidates remain zero.
The externally covered cohort is intentionally conditional, not a random sample
of all unknown compounds. FIORA pretraining membership remains unknown.

## Preregistered pass/fail criteria

Calculate paired molecular bootstrap intervals (2000 resamples, seed 26091603)
for the fixed candidate minus the incumbent. Eligibility requires ALL of:

1. Target cohort, absent-reference MRR difference lower95 > 0.
2. External cohort, external-recovery MRR difference lower95 > 0.
3. External cohort, available-reference MRR difference lower95 > -0.02.
4. External-recovery mass-window coverage >= 0.80.
5. Identical Recall@25 for every query and unchanged candidate membership.

This is an intersection gate, not a tunable average of regimes. Report all six
case/cohort cells, Top1 gains and losses, coverage, rank certificates and graph
failures. The intervals quantify molecular sample uncertainty at fixed weights,
not training or pretraining uncertainty. A failed gate stays failed; changing
parameters requires another separately planned test. Passing is necessary for
considering a new candidate, not proof of higher Kaggle or final/private score.

## Operational controls

Run in a separate state directory. Use a read-only SQLite backup of existing
FIORA simulations; do not edit the incumbent cache or weights. Kaggle is queried
read-only for the existing R08B submission. This experiment cannot submit or
promote automatically. Independently reconstruct ranks, certificates, metric
values, intervals and gates from recorded arrays after completion.

## Pre-outcome source amendment (v2)

The v1 preflight, run 35378155922 (artifact SHA256
`b9b61bc932387d64f7e9f142b4dfc0755b720eec711ec6b847f9c9112fbdf829`),
found 6509 unused target keys but only FOUR unused external keys in the old
mass-selected slice. It stopped before creating candidate scores, computing
forward evidence or calculating any new ranks. The failed run and frozen v1
policy remain unchanged. No quality gate failed or was relaxed.

V2 therefore defines external membership from the SAME already-fixed complete
August-2026 COCONUT snapshot, mass-indexed against ALL unused holdout structures
with valid source metadata, and canonically keyed by the existing pipeline.
Membership is recorded before scoring, including input and exclusion hashes.
The source snapshot, seed, 128/96 cohort sizes, selected feature, weights, all
gates and model weights remain unchanged. Only the inadequate external pool
definition and state directory are amended. No query is resampled based on its
rank, no spectrum is added to model fitting, and no scoring parameter is retuned.
The v2 external cohort remains conditional on catalog membership, not an
estimate of coverage for arbitrary unknown compounds.
