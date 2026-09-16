# Kaggle evaluation workflow

## Scope and authorization

The user explicitly requested uploading the current solution, inspecting its official score, and making occasional later submissions without exhausting the limit. This authorizes a private code-notebook submission for CASMI26; it is not permission to publicly redistribute the competition corpus or expose credentials.

The active Kaggle owner was resolved by authenticated CLI initialization as `thelindortis`. All project assets and notebooks stay private. Do not select final submissions, accept new rules, change account settings or share with other teams as part of routine evaluation.

## Budget

Official rules checked on 2026-09-16 in preflight run 35051877903 allow 5 submissions per day and 2 final selections. The initial submission history was empty. Recheck rules when appropriate; this is a dated observation, not a permanent API guarantee.

Our operational guard is intentionally stricter: at most **2 total account attempts in a rolling 24-hour window**, counting failed submissions and user-made entries too. It leaves capacity below the official daily maximum. Do not submit while another submission is pending. Unparseable history or timestamps stop a submission rather than assuming available quota.

Dataset uploads, notebook preparation and read-only status calls must not be reported as scored competition submissions. Nonetheless avoid needless notebook executions and polling. Do not automatically resubmit an unchanged version or retry a write whose outcome is unknown.

## Stages

`tasks/casmi_kaggle_publish.py --stage publish` verifies the existing train-only asset bundle and creates a private dataset/notebook. It uses the already validated production baseline, not unintegrated research scripts. A journal records write intent before each API mutation. The original asset payload excludes test rows, IDs, precomputed predictions and credentials.

`--stage submit` requires a known successful notebook version, a completed Kaggle run, a real submission.csv output and a fresh budget check. The single code submission references that notebook version; uploading the local visible CSV instead is not an equivalent operation. A prior submission-attempt journal blocks blind retries.

`--stage status` only reads kernel status, account submission history, scoring status and rank. A zero or low returned score must not be hidden or replaced with local validation numbers.

Durable journal/report folder on ChemistryPC:

`C:\ProgramData\ChemistryRunner\artifacts\casmi26\kaggle-v1`

## When to submit a later research version

Use a new separately identified version only after a meaningful predeclared local experiment and fresh software/output checks. Keep the prior notebook and its official score as the baseline. Choose one worthwhile experiment, not every hyperparameter combination. A local improvement is a reason to test on Kaggle, not a promise that the public score will improve. Reserve subsequent attempts for genuinely different or verified repaired versions.

Every scored entry should record model, code and asset hashes, notebook version, submission reference, server status, returned public score and the local evaluation protocol. Never label a catalog-inclusive local MRR as a hidden Kaggle score. An unavailable private score stays unavailable.

## Current research caution

R01 is a frozen-fingerprint ranking experiment with an inclusive candidate catalog. R02 evaluates reference/neural fusion after excluding all held-out structures' spectra from the reference library. Neither reconstructs molecular graphs absent from the catalog. Do not silently substitute a large local retrieval score for novel-structure recovery or a score of 1.0.
