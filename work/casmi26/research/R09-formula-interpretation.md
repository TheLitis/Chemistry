# R09-F: retain the measured negative result; do not change the champion

The independently checked R08B public result is **0.193**, up from R07's 0.188.
The comparison is one public aggregate; it does not establish private-test gain
or statistical significance. Submission 56331047 has COMPLETE status, a numeric
public score and an empty error. No new submission was made in this continuation.
See `../evidence/2026-09-18-r08b-official-score.json`.

## What actually ran

An original bounded elemental-subcomposition module was implemented, tested and
run on the already spent 224 R08B records, first locally and then on ChemistryPC.
The complete R07 candidate sets and score arrays remained unchanged. We evaluated
7,650 query/formula pairs (7,342 with supported spectra); unsupported cases add
no bonus rather than remove a candidate. All 28 new tests passed on the PC.

A separate verifier, with no production scorer imports, rebuilt 2,016 ranks,
54 metrics and 12 paired intervals. Recomputed metrics were identical. The PC
and Linux feature values matched within 3.4e-16 over 22,026 numerical values.
Raw features, candidate/formula associations and ranks are in the referenced
Actions artifact. It also contains the exact source snapshot.

The fixed .25 explained-mass bonus improved reused timsTOF no-reference MRR from
0.11695 to 0.12216. The mixed cohort rose only from 0.21722 to 0.22074, with an
interval crossing zero. External recovery fell from 0.07853 to 0.07669. None of
these listed regimes gained correct Top1 answers. The shifted-mass control did
not resolve this weakness. No parameter was chosen for production afterward.

## Why formula evidence alone is insufficient here

A diagnostic that knows the answer formula and merely promotes that formula's
group, preserving R07's within-formula ordering, reaches only 0.13992 MRR on the
128 timsTOF no-reference queries. The corresponding mixed result is 0.27018.
This is NOT an implementable system or a ceiling for a new structural scorer.
It isolates the many remaining same-formula ranking errors: the median group
size conditional on the truth being present is 32 structures in the timsTOF
cohort and 69 in the mixed cohort.

Do not infer that stronger formula inference, additional candidate discovery or
formula-constrained generation is useless. The tested enumeration is deliberately
permissive and contains no graph topology, bond-breaking or valence model. Its
failure limits this specific bonus, not the broader research direction.

## Consequence

Do not replace 0.193 R08B with this unconfirmed elemental bonus. Preserve the
bounded module and measured failures as diagnostics. The next quality experiment
must target structural discrimination within the same formula and retain the
R07/R08B full-system baseline, rather than merely score more atom combinations.
The numerically equivalent nearest-only backend is verified separately and is
not presented as a quality improvement.
