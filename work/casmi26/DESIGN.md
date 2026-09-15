# CASMI26 structure prediction: objective and implementation record

## Acceptance target

All spectra belonging to an unknown compound -> one predicted 2D molecular graph
-> competition-format submission -> maximum official held-out score. Internal
retrieval, generation and scoring are means, not the acceptance target. Neither
passing software tests nor matching a molecular fingerprint establishes success.
No test labels, answer hunting, private data or leaderboard probing are used.

## Verified state, 2026-09-15

The repository initially contains a file-only, fast-forward PC bridge. No data,
model weights, authenticated Kaggle session or official scoring implementation
is present. The Kaggle overview, evaluation, rules and public file-list requests
failed in the available browser. Consequently the official metric, row grouping,
file schema, external-data policy and submission execution limits are UNVERIFIED.
Do not replace these unknowns with inferred facts from unrelated CASMI papers.

## Executable architecture

Python 3.11+; RDKit for graph validation; NumPy for peaks; SQLite for an on-disk,
stream-built exact-mass candidate/reference index. PyArrow is required only for
Parquet. The inference entry point is the repository-root predict.py.

Stream reference spectra into the index without storing the whole library in
RAM. Test labels are never read as features. Explicit compound IDs group spectra;
precursor mass alone must never join compounds. Infer neutral mass using an
explicit supported adduct and verify agreement across the group. Retrieve
mass-compatible structures, score each with all query spectra (direct fragment
and neutral-loss matching, mode-compatible references), generate a bounded set
of valid formula-preserving graph rearrangements, and select a single structure.
Graph proposals can be absent from the reference library, but are not an
exhaustive de novo generator. Their bond-cut explanation score is a heuristic,
NOT a validated forward MS/MS model. It is opt-in until measured on real data.

Use the actual sample_submission header and IDs; reject unsupported schemas or
missing IDs rather than inventing a format. Output atomically, with a sidecar
recording provenance, all spectrum counts, candidate scores, unresolved errors
and official_score=null. An input, model or evaluation gap is never reported as
100%. No automatic submission, rule acceptance, shell execution or new PC
service is introduced.

## Implementation / verification plan

- [x] Write and run failing tests for graph identity, adducts, grouping, spectral
  matching, streaming readers, formula-preserving proposals, complete outputs,
  label non-interference and failure without fabricated structures.
- [x] Implement isolated chemistry, spectra, index/inference and CLI modules.
- [x] Execute full test suite, CLI end-to-end fixture and negative CLI cases.
- [x] Review source for input overwrites, external calls, secrets, molecule-level
  leakage, per-spectrum rather than per-compound decisions and false score claims.
- [ ] Publish a single fast-forward commit on main, preserving the PC bridge.

## Scientific evidence required beyond software tests

The actual competition files and scoring contract must be obtained through an
authorized environment. Structural holdouts must separate canonical 2D graphs;
any generative benchmark must also remove validation graphs from the candidate
catalog. An exact-graph diagnostic is not the official metric. No real-data
validation, training or Kaggle scoring has occurred in this implementation pass.
A competitive trained generative model and validated reranker are not supplied.
