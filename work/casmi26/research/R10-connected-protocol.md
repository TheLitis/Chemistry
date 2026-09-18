# R10-G: structural cut evidence, fixed exploratory screen

## Hypothesis and scope

Replace unrestricted elemental-subcomposition matching with connected fragment
hypotheses from the candidate's molecular graph. This may distinguish isomers
that R09-F could not. The comparator is the complete R08B scoring recipe, NOT
R07 and NOT a mass-only score. No chemical accuracy is implied by a graph cut.

Use the 224 already-spent R08B query records and their exact top-25 numerical
certificates. Keep every candidate and all original scores in those top-25
lists. Only reorder those lists; Recall@25 must stay unchanged per molecule.
Do not impute uncomputed forward scores, even when the original FIORA trace
is partial. Reject invalid certificates or missing source cases.

This is NOT an untouched validation. The original historic cohort names in the
input are retained for traceability, not as a renewed freshness claim. There is
no training, parameter search, automatic promotion or Kaggle submission.

## Fixed graph feature

Canonicalize tautomers with the pinned RDKit 2026.03.3. Enumerate connected
heavy-atom components remaining after all one-edge and two-edge deletions,
including ring-opening two-edge cuts. Count original attached hydrogens before
cutting: do not sanitize a cut graph and silently add new capping hydrogens.
Use a fixed optional hydrogen shift -1/0/+1 within the parent H inventory.
A fragment's match weight is 1/(number_of_cuts + absolute_H_shift). This is a
heuristic complexity penalty, NOT a bond-energy or ionization model.

Accept only supported neutral unlabelled parent candidates; restrict observed
ions to [M+H]+ and [M-H]-. Exclude precursor and above-precursor peaks, keep peaks
at least 1% of the spectrum maximum, pool sqrt-intensity-normalized matches at
max(5 ppm, 0.001 Da), and average supported spectra. Unsupported chemistry or
a fixed work-budget overrun is explicitly missing evidence and adds no bonus.
Never use query formulas, answer SMILES, or hidden-test labels as feature inputs.

Declare work bounds before enumeration: at most 96 heavy atoms and 8192 cut
scenarios. Do not silently truncate an atom-order-dependent enumeration prefix.

## Comparisons fixed before full execution

1. Complete R08B top25 unchanged.
2. R08B + 0.25 * one-cut evidence, reordering its top25 globally.
3. R08B + 0.25 * up-to-two-cut evidence, reordering its top25 globally.
4. Same as 3 but reorder only slots occupied by the same elemental formula.

No configuration is selected by this screen. Report every configuration, MRR,
Top1, Recall25, paired molecular bootstrap (2000 resamples; seed 26091603), and
numbers of new/lost Top1 answers. Intervals are descriptive, unadjusted for the
multiple fixed comparisons. Keep cases with missing answers in the denominator.

## Verification and advancement

Unit-test enumeration against an independent small-graph all-edge-cut oracle,
including rings, aromatic graphs, original H bookkeeping, representation/order
invariance, budget rejection, missing evidence, and same-formula isomers.
Run real PC data with pinned RDKit. Independently recompute all stored rankings
and effects without importing the production feature or ranker. Preserve frozen
R08B model/package/CSV hashes. If promising, design a separate prospective test
on new keys; this screen alone can never replace the Kaggle champion.

## Primary-source context

- Ruttkies et al., MetFrag relaunched (2016), doi:10.1186/s13321-016-0115-9.
- Nowatzky et al., FIORA (2025), doi:10.1038/s41467-025-57422-4.
- RDKit Cookbook, Create Fragments (https://rdkit.org/docs/Cookbook.html).

This is our simple bounded graph feature, NOT a reproduction of MetFrag,
MS-FINDER, FIORA, or a mechanistically validated fragmentation predictor.
