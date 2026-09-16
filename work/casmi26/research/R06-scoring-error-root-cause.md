# R06 Kaggle scoring error — evidence and corrective contract

## Observed failure

Competition submission `56278642` (R06 notebook version 2) is shown by the Kaggle
Submissions UI as **Submission Scoring Error** with no public score. The older CLI
2.2.4 lists the same ref as `COMPLETE` with an empty `publicScore`; that state must
not be treated as successful scoring. The UI observation is recorded separately in
`../evidence/2026-09-16-r06-scoring-error-ui.json`.

Kaggle's documented meaning of Submission Scoring Error is a hidden-rerun output
that cannot be scored, including wrong row/column count, empty values, wrong types,
or invalid values. The exact server-side `error_description` is queried read-only by
`tasks/casmi_r06_scoring_diagnose.py` when the self-hosted runner is available.

## Concrete hidden-only defect in R06 v2

The submitted R06 inference selected candidates only with the strict mass window.
When that set was empty it produced `guesses=[]` and therefore wrote an empty
`smiles` cell. The visible 400-molecule example happened to have no empty candidate
rows, so its pre-submit validation could not expose this hidden-only branch.

This branch is not hypothetical in the model's validation domain. On the locked
2048-molecule R06 audit, candidate coverage was 0.98974609375 for the 3-spectrum
protocol and 0.97705078125 in the all-spectrum diagnostic. Thus the audit itself
contained 21 and 47 uncovered queries respectively. Offline MRR correctly counted
those as rank zero, but the old production writer converted the same condition into
an invalid empty submission value.

The successful V1 production path did not have this defect: it used strict mass
matching, then an expanded 50 ppm / 0.02 Da window, then up to 64 nearest-mass
structures. V1 received an official public score of 0.147.

## Corrective contract

`casmi26.r06_candidate.select_mass_candidates` now applies the same three-stage
non-empty policy:

1. original strict mass-compatible candidates;
2. expanded 50 ppm / 0.02 Da candidates if strict is empty;
3. up to 64 nearest catalog masses if both compatible windows are empty.

The first branch is unchanged, so all previously mass-compatible R06 predictions
retain their candidate set and ranking. Regression tests require the expanded and
nearest fallbacks to return at least one candidate, and full hosted CI passes on
Windows and Linux.

R06 v3 embeds the current source into the private notebook while reusing the exact
same private model weights and candidate bundle. Before any corrected competition
submission it must:

- finish its private Kaggle notebook run successfully;
- produce no empty candidate rows;
- pass schema and hash checks;
- reproduce the existing visible-test CSV byte-for-byte (`750e3410...508cffe`),
  proving that scorer safety did not alter visible mass-compatible rankings;
- recheck official submission quota and pending state;
- journal the corrected submission before the one allowed write.

No corrected competition submission is authorized merely by this document; the
at-most-once v3 gate remains enforced in `tasks/casmi_r06_kaggle_v3.py`.
