# Chemistry PC Bridge + Execution Plane

`TheLitis/Chemistry` is a private GitHub repository used as both a delivery bridge and a controlled execution plane for a Windows PC.

There are two independent layers:

1. `bridge/` keeps `C:\Users\loval\Chemistry` synchronized with `origin/main` and **never executes received files**.
2. The repository-scoped GitHub Actions runner `ChemistryPC` executes only explicitly requested/reviewed jobs and returns logs/artifacts through GitHub.

No inbound SSH, RDP, WinRM, or public listener is required.

## Local paths

Live synchronized checkout:

```text
C:\Users\loval\Chemistry
```

Self-hosted runner:

```text
C:\actions-runner\ChemistryPC
```

Persistent execution state, runtime, caches and local task data:

```text
C:\ProgramData\ChemistryRunner
```

Important subpaths:

```text
C:\ProgramData\ChemistryRunner\machine.json
C:\ProgramData\ChemistryRunner\python\python.exe
C:\ProgramData\ChemistryRunner\data
C:\ProgramData\ChemistryRunner\cache
```

The GitHub Actions runner uses its own `_work` checkout. Jobs do not execute from the live synchronized checkout, so Actions cleanup cannot dirty or destroy the bridge workspace.

## File bridge

The bridge follows only `main`, applies only safe fast-forward updates, and pauses instead of overwriting a dirty/diverged local checkout.

Bridge log:

```powershell
Get-Content "$env:LOCALAPPDATA\ChemistryBridge\bridge.log" -Tail 50 -Wait
```

Reinstall bridge:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\bridge\install.ps1"
```

Remove bridge:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\bridge\uninstall.ps1"
```

## One-time ChemistryPC bootstrap

From a normal PowerShell window:

```powershell
git -C "$HOME\Chemistry" pull --ff-only origin main; Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList "-NoExit -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$HOME\Chemistry\runner\install.ps1`""
```

This one-time elevated bootstrap:

- validates the authenticated GitHub CLI session;
- obtains a short-lived repository runner registration token at runtime;
- downloads the official current Windows x64 GitHub Actions runner;
- registers `ChemistryPC` only for `TheLitis/Chemistry`;
- installs `actions.runner.TheLitis-Chemistry.ChemistryPC` as a Windows service;
- enables delayed automatic startup and service recovery;
- runs normal jobs as `NT AUTHORITY\NETWORK SERVICE`, not as a permanent administrator;
- stores machine configuration outside Git.

Re-running the installer is safe for an already healthy runner. Use `runner/install.ps1 -Repair` only for an explicitly damaged installation.

## Persistent Python and CUDA runtime

Normal jobs bootstrap their own execution runtime under `C:\ProgramData\ChemistryRunner`; no system-wide Python installation is required for the service account.

`runner/bootstrap-python.ps1` installs an isolated CPython runtime and pip. `runner/bootstrap-torch.ps1` installs/verifies NumPy and a CUDA-enabled PyTorch build from the official PyTorch CUDA wheel index. Both scripts are idempotent: later jobs reuse the existing runtime and package cache.

The runtime is accepted as CUDA-ready only when `torch.cuda.is_available()` is true and at least one CUDA device is visible. A driver-visible GPU alone is not treated as proof of working CUDA compute.

## Automatic task queue

The permanent ChatGPT/GitHub -> PC execution entry point is:

```text
control/request.json
```

Updating this file on `main` automatically triggers `.github/workflows/pc-task.yml`, which can run only on:

```yaml
runs-on: [self-hosted, Windows, X64, ChemistryPC]
```

Supported request types are deliberately bounded:

- `health` — CPU/RAM/disk/NVIDIA/Python/PyTorch diagnostics plus bounded compute smoke tests;
- `tests` — repository/CASMI software tests;
- `casmi26` — run the CASMI26 entry point against configured local data;
- `script` — execute a reviewed `.ps1` or `.py` file committed below `tasks/`.

There is no arbitrary shell-command text input. `script` requests reject absolute paths, `..` traversal, non-`tasks/` files and unsupported extensions.

Example CUDA health request:

```json
{
  "version": 1,
  "id": "health-example-001",
  "task": "health",
  "requireCuda": true
}
```

Example reviewed script request:

```json
{
  "version": 1,
  "id": "system-summary-001",
  "task": "script",
  "script": "tasks/system-summary.ps1",
  "arguments": [],
  "requireCuda": false
}
```

Every execution receives `CHEMISTRY_REQUEST_ID` and `CHEMISTRY_REQUEST_OUTPUT`. Results are written into the job's isolated output directory and uploaded as a `chemistry-pc-task-*` workflow artifact. A `request-result.json` records task status, exit code, timing, runner and machine.

All PC workloads share the `chemistry-pc-exclusive` concurrency group so two heavy GPU jobs do not run against the single RTX GPU at the same time.

## Health workflow

`.github/workflows/pc-health.yml` remains available as a **manual** diagnostic workflow. It is not triggered by arbitrary pushes.

It verifies:

- Windows/hostname;
- CPU model, cores and logical processors;
- total/available RAM;
- fixed disks/free space;
- NVIDIA GPU/VRAM/driver visibility;
- persistent Python/pip;
- PyTorch and CUDA visibility;
- bounded parallel CPU compute;
- bounded RAM allocation/touch;
- real CUDA matrix multiplication when CUDA is required.

## Verified physical-PC integration

The end-to-end automatic request path has been exercised on `ChemistryPC` (`DESKTOP-AH6DPCN`). The successful CUDA health request confirmed:

- Intel Core i9-10900KF: 10 cores / 20 logical processors;
- about 63.87 GiB physical RAM;
- NVIDIA GeForce RTX 3070 with 8192 MiB VRAM;
- NVIDIA driver 616.56;
- persistent Python 3.13.15 + pip;
- PyTorch 2.14.0+cu126 with CUDA 12.6;
- `torch.cuda.is_available() == true`;
- an actual 8192x8192 CUDA matrix-multiplication smoke workload (8 iterations);
- CPU smoke using 20 workers;
- RAM smoke allocating and touching 8 GiB;
- successful GitHub artifact return from the physical PC.

These are capability checks, not a promise that every workload should force CPU, GPU and RAM to 100% simultaneously. Individual tasks should tune batching, data-loader workers, precision and memory use for maximum throughput without paging/OOM.

## CASMI26 local data

`machine.json` defines the local data root. The controlled `casmi26` request expects:

```text
C:\ProgramData\ChemistryRunner\data\casmi26\test
C:\ProgramData\ChemistryRunner\data\casmi26\train
C:\ProgramData\ChemistryRunner\data\casmi26\sample_submission.csv
```

Local datasets, credentials, caches, model weights and generated large files are intentionally not committed to Git. A future reviewed task may download or prepare authorized data there without changing the execution architecture.

## Runner health / repair / removal

Local health check:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\health.ps1"
```

Remove runner (elevated PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\uninstall.ps1"
```

Persistent machine data is kept by default. Use `-RemoveData` only when intentionally deleting `C:\ProgramData\ChemistryRunner` as well.

## Trust model

Anyone who can write trusted executable code/workflow changes to this private repository can cause the repository-scoped runner to execute that reviewed code as the runner service account. Therefore:

- the runner is repository-scoped rather than organization-wide;
- PC jobs require the `ChemistryPC` label;
- untrusted pull-request events do not execute PC jobs;
- arbitrary shell text is not accepted as a request;
- normal jobs are non-admin;
- secrets/data are not committed or automatically uploaded;
- the bridge remains file-only and independent of execution.
