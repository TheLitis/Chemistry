# CASMI26 verified competition contract

Retrieved from the official public Kaggle pages on 2026-09-15 at 23:06 UTC,
using a fresh anonymous Edge profile on ChemistryPC. Evidence: GitHub Actions
run 35034137017, job 104599207158, artifact 10423170058. The artifact contains
public-evaluation.txt, public-rules.txt, public-data.txt and casmi-metadata.json.
No account cookies, test answers or authenticated competition files were used.

Sources:
- https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview/evaluation
- https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data
- https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
- https://github.com/TheLitis/Chemistry/actions/runs/35034137017

## Evaluation (replaces the earlier unverified single-guess assumption)

The metric is mean reciprocal rank at 25 (MRR@25). For each molecule the score
is 1/rank of the first correct prediction, or zero when none is correct. The
maximum 1.0 requires the correct structure at rank 1 for every scored molecule.
Submitting up to 25 guesses is not a replacement objective: additional guesses
recover partial credit while the target remains an exact first prediction.

A correct structure is determined after RDKit tautomer canonicalization, using
RDKit version 2026.03.3, then comparison of the first 14 characters of InChIKey.
Stereochemistry and tautomer forms do not have to match literally. A local
implementation of this description is not itself a score returned by Kaggle.

The CSV columns are molecule_id and smiles. Each molecule_id appears exactly
once. The smiles field contains 1-25 SMILES separated by semicolons, best first.
Nulls, empty submissions, duplicate molecule IDs, missing required columns and
more than 25 guesses cause rejection. Avoid equivalent-tautomer/stereo guesses
occupying multiple ranks. Invalid molecular strings never establish success.

## Execution and held-out test

This is a CODE COMPETITION, not a direct CSV-only competition. Submission must
be through a Kaggle Notebook, with Internet disabled and at most 9 hours runtime
for CPU or GPU. The output must be named submission.csv. Freely/publicly
available external data and pretrained models are permitted, subject to the
full rules and applicable licenses. No rules were accepted in this session.

The downloadable test.parquet is drawn from training examples. It is REPLACED
by a hidden test set when Kaggle reruns the notebook. Success on the visible
example file must not be represented as success on the private competition.

The hidden set is described as about 1500 spectra / 400 molecules, 1-16 spectra
per molecule, on Bruker timsTOF. It mixes library-known compounds, known structures
without public spectra, and genuinely novel structures absent from PubChem.
The class distribution and membership remain hidden.

## Input schema

Prediction grouping key: molecule_id (NOT spectrum_id).
Peak arrays: ms2_mzs and ms2_normalized_intensities.
Precursor: precursor_mz; ion type: adduct.
Training target: normalized_smiles.
Collision energy: collision_energy_ev, a list of one or more eV values; do not
silently coerce this to a scalar or confuse original NCE with eV.
Other shared metadata: spectrum_id, base_peak_intensity, ionization_mode,
instrument_type, collision_energy_orig, collision_energy_orig_units.

Ten documented test adducts:
[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+,
[M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-.
The training set has additional adducts and uncurated precursor-error values.

The official file set is train.parquet, test.parquet, sample_submission.csv,
about 3.04 GB in total. The training data are roughly 2.5 million spectra and
275,810 distinct structures, with overlapping structures across libraries.
The enveda-np-examples source has 1151 spectra of 250 common natural products
measured on the same instrument and pipeline as the hidden test.

## Confirmed access limitation

Kaggle CLI 2.2.4 is installed on the PC. Its file-list action exits with a
missing-credentials message. Unauthenticated file API requests return HTTP 401.
The official data page explicitly requires sign-in and acceptance of competition
rules to access the files. The configured local competition data folders and
standard Kaggle credentials checked by the scoped audit were absent.

No training on CASMI26 data, official submission, public score or private score
has been obtained. This document records the contract, not achievement of it.
