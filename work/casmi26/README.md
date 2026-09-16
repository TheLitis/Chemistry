# CASMI26 — trained inference from all MS/MS spectra

The goal remains exact rank-one molecular connectivity for every unknown
compound, scored by the competition's MRR@25. A runnable file, an embedding,
a matching molecular formula, and a valid CSV are not evidence of that goal.

## Current implementation

The production model is trained on the supplied official training corpus.
Full-corpus preprocessing inspected 2,539,608 spectra and admitted 2,487,355;
24 rows had unsupported/invalid ion or peak information, and 52,229 did not
pass the precursor/structure mass-consistency check. The raw structure catalog
contains 277,566 normalized SMILES records, of which 276,241 have admitted
spectral features. Raw records are not the same as unique tautomer-equivalent
connectivity keys.

The 3,152,384-parameter fingerprint model uses 2,048 fragment-mass bins, 2,048
neutral-loss bins and eight precursor/polarity/energy features. It was trained
on ChemistryPC's CUDA device for 30 epochs. A distinct holdout model is retained
for evaluation; the final inference model was refitted on all admitted records.
Inference combines per-spectrum spectral matching and learned structural
fingerprint ranking. All available spectra of a molecule contribute.

## Measured validation, not a Kaggle score

A fixed 10% connectivity-key holdout excludes all spectra of held-out keys from
fitting. Up to 4,000 distinct held-out keys are evaluated. The candidate catalog
**includes their known molecular structures by design**. This measures
known-catalog retrieval, not de novo discovery or the hidden Kaggle test.

| Measure, 4,000 held-out molecules | Mass-error baseline | Learned fingerprint ranker |
|---|---:|---:|
| MRR@25 | 0.2470373 | 0.5668168 |
| Top-1 exact connectivity | 0.13875 | 0.42875 |
| Recall@25 | 0.68875 | 0.91800 |

Among 3,922 cases with multiple mass-compatible catalog records, MRR@25 was
0.2323174 for mass ranking and 0.5584567 for the learned model. The final
spectral-plus-neural blend is not the model used in this comparison; its
hidden-test performance has not been measured. No official Kaggle score is
claimed. Evidence is linked in `evidence/2026-09-16-official-training.json`.

## Paths on ChemistryPC

```text
C:\ProgramData\ChemistryRunner\data\external\enveda-CASMI26-molecule-id-mass-spectra
C:\ProgramData\ChemistryRunner\cache\casmi26\official-v1
C:\ProgramData\ChemistryRunner\artifacts\casmi26\official-v1
```

The GitHub Actions workspace is separate from the user's synchronized checkout;
large data and models stay outside that disposable workspace. The isolated
Python executable is `C:\ProgramData\ChemistryRunner\envs\casmi26\python.exe`.

Production inference can be invoked from either checked-out repository copy:

```powershell
& 'C:\ProgramData\ChemistryRunner\envs\casmi26\python.exe' predict.py --production --test 'C:\ProgramData\ChemistryRunner\data\external\enveda-CASMI26-molecule-id-mass-spectra\test.parquet' --output submission.csv
```

The production mode discovers the adjacent training file and the default
ProgramData model bundle. Other inputs and models can be specified:

```powershell
python predict.py --production --test data/test.parquet --train data/train.parquet --bundle artifacts/bundle --output submission.csv
```

`--bundle` also selects production mode. Without either `--production` or
`--bundle`, the original experimental CLI remains available for compatibility;
that old mode does not automatically use the trained production model.

## Actual competition contract

See `COMPETITION.md` and `casmi26/metric.py`. Use RDKit **2026.03.3**. Each
molecule has one ranked list of 1–25 SMILES, separated by semicolons, in the
`molecule_id,smiles` CSV. Matching canonicalizes tautomers and compares the
first 14 InChIKey characters. Multiple representations of one connectivity
must not waste candidate ranks.

CASMI26 is a **code competition**. Kaggle reruns a notebook without Internet
and replaces the visible example test file with a hidden test file. The
visible file comes from training examples: success on it is not a hidden
accuracy estimate. A previously saved CSV alone is not a valid submission
workflow. The notebook must compute predictions from the current input mount.

## Production modules

`production.py` streams the full official corpus, parses stoichiometric adducts,
constructs feature and fingerprint arrays, trains the holdout and full-refit
models, and records provenance. `learning.py` contains the neural model and
numeric-only checkpoint format. `portable.py` consumes train-derived assets,
selects fresh reference spectra for the current queries, and writes predictions
and a diagnostic report. `submission_notebook.py` embeds inspectable source,
loads the current Kaggle test mount and uses offline wheels.

The notebook asset bundle contains only `catalog.json`, `fingerprints.npy`,
`model.npz`, their manifest, and dependency wheels. It does not contain visible
test identifiers, test answers, precomputed test predictions, or the
query-specific reference cache. Training-data hashes and model hashes are
checked. An old visible sample template cannot override replacement test IDs.

## Reproduce the PC stages

`control/request.json` selects a committed reviewed script for ChemistryPC.
Use `tasks/casmi_official.py --stage prepare`, then `--stage fit --epochs 30`.
`tasks/casmi_deliver.py` builds a train-only bundle, executes all generated
notebook code cells against the actual visible files, validates every output
row, packages Linux CPython 3.12/3.13 wheels and rechecks existing CLI access.
This Windows code-cell run is distinct from a Jupyter-kernel execution and from
an actual Kaggle Linux run; the reports state exactly which was performed.

The verified delivery includes `submission.csv` for all 400 visible example
molecules and all 1,213 spectra, a source-embedded notebook, trained model assets,
and a 243,412,845-byte ZIP. No mass-incompatible fallback was needed for these
visible examples. The delivery job ran 88 passing tests. These are software and
output-conformance checks, not proof that 400 structures are correct.

Attach the `bundle` directory as a **private** Kaggle dataset. Keep assets
private to the team and respect competition redistribution terms. Attach the
original competition separately to the notebook. Disable Internet. Do not upload
a saved visible CSV as a substitute for a notebook that processes hidden input.

## Kaggle access under the service

Installing/authenticating the CLI under the Windows user does not provision a
token to NETWORK SERVICE automatically. The final delivery check still had no
authorized file listing from the service. Local inference is independent of
Kaggle authentication and is complete; an official notebook submission is not.

`runner/stage-kaggle-access.ps1` is a user-invoked helper, not an automatically
executed runner task. Run it under your signed-in Windows account (elevate if
needed). It reads only an existing `KAGGLE_API_TOKEN` or the provider's standard
`access_token` file; otherwise it asks for hidden console input. It stages that
one token into `C:\ProgramData\ChemistryRunner\kaggle\access_token`, protects
that dedicated directory, and grants NETWORK SERVICE read access. It does not
print the secret, read browser sessions, open your profile, change the runner
identity, accept competition rules, or submit anything. Do not send tokens in
chat. The credential operation itself cannot be tested without your user-side
authorization; syntax is checked separately.

## Limitations that remain material

This is a trained catalog-retrieval baseline. It does **not** yet include a
competitive pretrained de novo molecular decoder or an external natural-product
structure catalog. If the correct graph is absent from the catalog, this
version cannot select it. Such absence is expected for some hidden novelty
classes described by the organizer.

When no mass-compatible candidate exists, production inference returns an
explicitly flagged nearest-mass learned ranking rather than dropping the
molecule and invalidating the entire submission. Those last-resort guesses are
not mass-compatible recovery and are not reported as success. They appear in
`mass_incompatible_fallbacks` in the report. The old experimental pipeline's
stricter behavior (stop instead) is unchanged.

Model scores and margins are not calibrated correctness probabilities. All
usable query spectra are averaged, but the validation's number and domain of
spectra per molecule can differ from the hidden natural-product test.
Instrument/domain shift, incomplete catalog coverage and ranker accuracy still
need measured improvement. Passing tests or generating 400 valid rows does not
establish 100% recovery. Official scoring remains null until Kaggle returns an
actual scored submission result.

## Software verification

```powershell
python -m pytest -q work/casmi26/tests
```

The suite checks parsing, mass/adduct handling, graph equivalence, train/test
isolation, array shapes, one-to-one matching, fresh hidden IDs, input protection,
bundle integrity and notebook generation. The completed trained-notebook
delivery run had 88 passing tests on ChemistryPC. Do not mistake these counts
for molecular accuracy.
