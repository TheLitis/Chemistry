# CASMI26: all spectra -> one molecular structure -> submission

**Acceptance target:** exact 2D structures for every unknown compound and the
highest possible official competition score. Retrieval accuracy, fingerprints,
a running script and passing unit tests do not establish that target.

**Current status: executable research prototype, NOT a completed high-score
solution.** There are no trained generative weights, real CASMI26 validation
results or Kaggle scores in this repository. The official metric, exact schema
and execution/external-data rules could not be retrieved in the implementation
session. The code therefore does not claim official compatibility or 100%.

## Run

From `C:\Users\loval\Chemistry` (Python 3.13 with the dependencies installed):

```powershell
python predict.py --test data/test --output submission.csv
```

One-time isolated environment, with Python 3.13 available:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r work/casmi26/requirements.txt
.\.venv\Scripts\python.exe predict.py --test data/test --output submission.csv
```

The command automatically locates `train` and `sample_submission.csv` alongside
the test data. Explicit paths are also accepted:

```powershell
python predict.py --test data/test.parquet --train data/train.parquet --sample-submission data/sample_submission.csv --output submission.csv
```

The required reference spectra and template are **not bundled**. No account
credentials are requested or read by this program; it runs offline. It does not
download competition data, accept rules, upload predictions or launch services.
The existing PC bridge delivers files only and is not changed into a remote shell.

## Input contract: provisional, to map to the actual competition files

Files or directories containing Parquet, JSONL, CSV, TSV or MGF are accepted.
Required canonical fields are:

| Field | Meaning |
|---|---|
| `compound_id` | Explicit compound-level identifier; all its spectra are used together |
| `precursor_mz` | Observed ion mass-to-charge ratio, not neutral molecular mass |
| `adduct` | An explicitly supported ion convention such as `[M+H]+` or `[M-H]-` |
| `mz`, `intensity` | Equal-length peak arrays; alternatively `peaks: [[mz, intensity], ...]` |
| `smiles` | Required in reference data only; ignored in test |

CSV/TSV arrays must be JSON-encoded. MGF needs a compound-level metadata field
(e.g. `COMPOUND_ID`), `PEPMASS` and `ADDUCT`/`ION_TYPE`. A numeric `CHARGE` by
itself does not identify the adduct. Spectrum IDs must not be substituted for
compound IDs without checking the official grouping. No compounds are merged
merely because their precursor masses coincide.

For other column names, provide a JSON mapping with `--columns mapping.json`:

```json
{"compound_id": "actual_compound_key", "precursor_mz": "actual_precursor_column"}
```

This is an example mapping, not a claim about CASMI26 field names. Unsupported
schemas stop with an error; they are not silently guessed.

The current writer supports an actual two-column ID/SMILES template. It
preserves the header, row order and leading zeroes in IDs, ignores all template
prediction values, and requires exactly the same compound ID set in test.
Use `--id-column` and `--prediction-column` for explicitly verified names.
A rank-list, spectrum-level or multi-column official submission contract needs
an adapter before this version may be submitted. No fake official scorer is
included. `exact_match` is only a local canonical-graph diagnostic: stereochemistry
is removed, isotopes retained, and tautomer/charge normalization is not assumed.

## What the current inference actually does

It streams reference spectra into an on-disk SQLite exact-mass index, preserving
peak precision. For each compound it checks agreement of neutral masses across
all its spectra, retrieves mass-compatible structural candidates, and averages
the best mode-compatible reference match for **each query spectrum**. Matching
uses square-root-intensity one-to-one cosine and neutral losses. It selects one
canonical 2D structure, not a final top-10 list.

An optional permitted local structure catalog can be supplied with
`--candidates data/catalog.csv`. The program does not check a catalog's license
or competition eligibility; those must be verified before use.

`--isomer-budget 64` adds a bounded set of valid formula-preserving graph
rewirings, including structures absent from the reference library. It also
activates a bond-cut fragment-explanation heuristic. This is NOT exhaustive de
novo generation and NOT a learned or chemically complete forward MS/MS model.
It is off by default because its effect on real accuracy has not been measured.
Without a correct library candidate or a suitable generated proposal, this
version cannot select the correct structure. If there is no mass-compatible
seed/candidate at all, it exits rather than writing arbitrary carbon or a
chemically incompatible fallback. Missing references for an ion mode yield
zero evidence and are explicitly recorded; margins are not probabilities.

## Output and reproducibility

`submission.csv` contains the chosen structure per compound. Its companion
`submission.csv.report.json` records SHA-256 input/output fingerprints, settings,
software versions, index reuse, numbers of spectra actually used, structural
candidates and uncalibrated diagnostic scores. `official_score` is always `null`
until an external scoring result is actually obtained. The top-10 diagnostic
list in that report is not the submission or the acceptance target.

A failed prediction does not create a partial CSV or overwrite source data.
Existing output from a previous successful run is preserved if a new run fails;
check the process exit code and the output hash, not just file existence.
Reference data is streamed; only test spectra and a mass-compatible candidate
set are kept in memory. The first index build can require substantial disk/time;
this has not been benchmarked on the full competition library. SQLite caches are
reused only when source-content hashes, schema mapping and RDKit version match.
All data, environments, credentials and generated outputs are git-ignored so
that they do not pause the existing file bridge.

## Verification

```powershell
python -m pip install pytest
python -m pytest -q work/casmi26/tests
```

Tests cover exact graph identity, isotope preservation, supported adducts,
input adapters, peak matching, joint-spectrum decisions, leading-zero IDs,
label non-interference, new graph proposals, cache invalidation and safe
failures. The Parquet test is skipped when PyArrow is absent. The GitHub workflow
installs PyArrow and runs the suite on hosted Linux and Windows, not on the PC.

Before treating this as a competitive solution, obtain the actual rules/metric
and data through authorized access, adapt the verified schema, run molecule-level
holdouts, and measure exact structure accuracy and the official score. For a de
novo holdout, remove held-out graphs from both spectral references and the
candidate catalog. Do not report synthetic fixture success as molecular
identification performance. See `DESIGN.md` for the objective and evidence gaps.
