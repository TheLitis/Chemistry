# PC Execution Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository-scoped Windows self-hosted runner bootstrap and a hardware health workflow so `TheLitis/Chemistry` can execute approved jobs on the user's PC instead of only delivering files.

**Architecture:** Preserve `bridge/` unchanged as the file synchronizer. Add `runner/` for installation and diagnostics, and add a manually-dispatched GitHub Actions workflow targeted only at `[self-hosted, Windows, X64, ChemistryPC]`. Runner state lives outside the Git checkout under `%LOCALAPPDATA%\ChemistryRunner`.

**Tech Stack:** Windows PowerShell 5.1+, GitHub CLI, GitHub Actions self-hosted runner, Python 3, optional PyTorch/CUDA.

**Spec:** `docs/superpowers/specs/2026-09-16-pc-execution-layer-design.md`

## Global Constraints

- Do not modify the semantics of `bridge/sync.ps1`.
- Do not commit runner registration tokens or credentials.
- Do not expose arbitrary command text inputs.
- Runner must be repository-scoped to `TheLitis/Chemistry` and labeled `ChemistryPC`.
- Normal workflows must not operate in the live synchronized checkout.

---

### Task 1: Restore repository tree and add parser coverage

**Files:**
- Preserve all files from commit `322a6b58f2cdab9c40f474407cd88ae1cd1ad9bc`
- Keep: `docs/superpowers/specs/2026-09-16-pc-execution-layer-design.md`
- Modify: `tests/Test-BridgeScripts.ps1`

- [ ] Restore the full tree from `322a6b5` while retaining the execution-layer spec.
- [ ] Extend parser coverage to every `runner/*.ps1` script once those files exist.
- [ ] Run hosted Windows parser CI and require success.

### Task 2: Runner bootstrap

**Files:**
- Create: `runner/install.ps1`
- Create: `runner/uninstall.ps1`
- Create: `runner/health.ps1`

**Interfaces:**
- `install.ps1` registers runner `ChemistryPC` for `https://github.com/TheLitis/Chemistry` using a short-lived token obtained from `gh api`.
- State root: `%LOCALAPPDATA%\ChemistryRunner`.
- Machine configuration: `%LOCALAPPDATA%\ChemistryRunner\machine.json`.

- [ ] Add syntax tests first for all three PowerShell scripts.
- [ ] Implement administrator check and prerequisite validation (`gh`, authenticated GitHub, repository checkout).
- [ ] Resolve latest Windows x64 runner release from GitHub API, download and extract it outside the repository.
- [ ] Request a repository registration token at runtime with `gh api -X POST repos/TheLitis/Chemistry/actions/runners/registration-token`.
- [ ] Configure runner with `--name ChemistryPC --labels ChemistryPC --unattended --replace` and install/start the supported Windows service.
- [ ] Write `machine.json` without secrets.
- [ ] Make rerunning installer repair/reconfigure the existing runner rather than create a duplicate.
- [ ] Implement health/uninstall scripts and verify parser CI.

### Task 3: Hardware diagnostics and compute smoke

**Files:**
- Create: `runner/diagnostics.ps1`
- Create: `runner/compute_smoke.py`
- Create: `runner/test_compute_smoke.py`

**Interfaces:**
- Diagnostics write `runner-health.json` to a caller-supplied output directory.
- Compute smoke writes `compute-smoke.json` and exits nonzero only when a required capability fails.

- [ ] Add Python tests for memory sizing and CUDA-optional behavior.
- [ ] Implement CPU, memory, disk, NVIDIA, Python, and PyTorch inventory reporting.
- [ ] Implement bounded CPU/RAM/GPU work without attempting to exhaust the host.
- [ ] Run hosted tests; GPU absence on hosted CI must be reported, not treated as a parser/test failure.

### Task 4: PC health workflow

**Files:**
- Create: `.github/workflows/pc-health.yml`
- Modify: `.github/workflows/bridge-tests.yml`

**Interfaces:**
- Manual dispatch only.
- `runs-on: [self-hosted, Windows, X64, ChemistryPC]`.
- Upload health reports with `if: always()`.

- [ ] Add workflow with timeout and single-PC concurrency group.
- [ ] Run diagnostics and compute smoke from the checked-out repository.
- [ ] Upload JSON/text artifacts and preserve failure exit status.
- [ ] Verify hosted CI still passes without trying to run `pc-health.yml` on GitHub-hosted machines.

### Task 5: Documentation and local bootstrap verification

**Files:**
- Modify: `README.md`

- [ ] Document the one-time elevated command.
- [ ] Document expected successful installer output and health checks.
- [ ] After local install, verify GitHub lists `ChemistryPC` online with expected labels.
- [ ] Dispatch `pc-health.yml` and verify it ran on the physical PC and reported real CPU/RAM/GPU inventory.
