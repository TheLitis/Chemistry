# R04: fixed public-catalog transfer audit

Predeclared 2026-09-16 after downloading/inspecting COCONUT August 2026, before
coverage or ranks are inspected. Snapshot SHA-256 is recorded by the download
job. No test data, hidden labels, leaderboard tuning or parameter search.

Choose 800 unused connectivity-key holdouts by fixed hash, excluding initial
validation and all R01/R02/R03 calibration+audit keys. Add remaining unused
held-out keys from enveda-np-examples and report that cohort separately even if
small. Use the highest-count admitted raw representation per key, observed
precursor-derived mass and spectral features only; no true formulas as input.

Mass-prefilter the complete public CSV using exact_molecular_weight, not average
molecular_weight. Recompute structures, tautomer keys, masses and fingerprints
with the official pinned RDKit. Use source annotations only for provenance,
not hidden-label inference. Record invalid/missing/mass-inconsistent records.

Evaluate four fixed conditions with the R01 head, no tuning:
1. Original inclusive structural catalog.
2. Original plus actual external structures (distractor cost).
3. Original catalog with ALL study query keys removed (must yield zero).
4. Excised original plus actual external structures (real-source recovery).

Deduplicate all conditions by evaluation key before scoring, selecting the mass
closest to observation without knowing the answer. Exact answers are never
manually inserted into external records. Include all measured molecules in MRR,
not only those with an external match. Report coverage ceilings, unconditional
MRR/Top-1/Recall@25, paired bootstrap, and NP/mixed cohort sizes.

This tests known public structures absent from the spectral catalog, NOT novel
graph generation or a hidden Kaggle score. It does not publish another Kaggle
submission, alter production weights, or redistribute source annotations.
