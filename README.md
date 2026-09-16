# Chemistry PC Bridge + SYSTEM Execution Plane

`TheLitis/Chemistry` is a private GitHub repository used as both a delivery bridge and a controlled execution plane for a Windows PC.

There are two independent layers:

1. `bridge/` keeps `C:\Users\loval\Chemistry` synchronized with `origin/main` and **never executes received files**.
2. The repository-scoped GitHub Actions runner `ChemistryPC` executes explicitly requested/reviewed jobs and returns logs/artifacts through GitHub.

The runner is configured for permanent **NT AUTHORITY\SYSTEM** execution. `SYSTEM` uses SID `S-1-5-18` and has full local-machine privileges, including administrator-level filesystem, registry, service, driver/install and system-configuration operations.

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

## Permanent SYSTEM mode

For an already installed `ChemistryPC`, the one-time migration is:

```powershell
git -C "$HOME\Chemistry" pull --ff-only origin main
Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList '-NoExit','-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',"$HOME\Chemistry\runner\enable-system.ps1"
```

`runner/enable-system.ps1` is idempotent. It:

- requires one elevated PowerShell launch;
- finds the existing repository-scoped `ChemistryPC` Windows service;
- stops it;
- changes its Windows service logon account to `LocalSystem` (`S-1-5-18`);
- grants SYSTEM full control to the runner/state directories;
- configures automatic startup;
- restarts the service;
- verifies that the service is running as `LocalSystem`;
- records `privilegeMode=SYSTEM` and `privilegeSid=S-1-5-18` in `machine.json`.

After that one migration, no per-task UAC approval is required. The Windows service starts automatically after reboot and future GitHub Actions jobs inherit SYSTEM privileges.

`runner/install.ps1` also enforces the same SYSTEM mode for fresh installs, re-runs and repairs. During initial runner registration it may temporarily use `NETWORK SERVICE`, then immediately migrates the installed service to `LocalSystem` before declaring installation complete.

## Privilege invariant

Permanent administrator access is not merely documented; it is enforced in code.

Before executing any non-validation request, `runner/run-request.ps1` reads the current Windows identity and requires:

```text
NT AUTHORITY\SYSTEM
SID S-1-5-18
```

If the runner ever starts under another account after an update, manual service edit or damaged repair, the task is rejected instead of silently running with reduced privileges.

`runner/health.ps1` also requires the Windows service account to be `LocalSystem`, and the manual `ChemistryPC health` workflow verifies `S-1-5-18` before doing diagnostics.

## Persistent Python and CUDA runtime

Normal jobs use the persistent execution runtime under `C:\ProgramData\ChemistryRunner`; no separate system-wide Python setup is required.

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

Supported request types:

- `health` — CPU/RAM/disk/NVIDIA/Python/PyTorch diagnostics plus bounded compute smoke tests;
- `tests` — repository/CASMI software tests;
- `casmi26` — run the CASMI26 entry point against configured local data;
- `script` — execute a reviewed `.ps1` or `.py` file committed below `tasks/`.

There is no arbitrary shell-command text field in `request.json`. This is not a privilege limitation: a reviewed task committed under `tasks/` may perform SYSTEM-level operations. Keeping executable code in Git preserves an auditable record of exactly what was sent to the PC.

Example SYSTEM/CUDA health request:

```json
{
  "version": 1,
  "id": "health-example-001",
  "task": "health",
  "requireCuda": true
}
```

Example reviewed SYSTEM script request:

```json
{
  "version": 1,
  "id": "admin-task-001",
  "task": "script",
  "script": "tasks/system-summary.ps1",
  "arguments": [],
  "requireCuda": false
}
```

Every execution receives `CHEMISTRY_REQUEST_ID` and `CHEMISTRY_REQUEST_OUTPUT`. Results are uploaded as a `chemistry-pc-task-*` workflow artifact. `request-result.json` includes task status, exit code, timing, machine, execution identity, execution SID and privilege mode.

All PC workloads share the `chemistry-pc-exclusive` concurrency group so two heavy jobs do not compete for the single GPU at the same time.

## Health workflow

`.github/workflows/pc-health.yml` is a manual diagnostic workflow. It is not triggered by arbitrary pushes.

It verifies SYSTEM identity and then checks Windows/hostname, CPU, RAM, disks, NVIDIA GPU/VRAM/driver, persistent Python/pip, PyTorch/CUDA, bounded CPU/RAM compute and real CUDA matrix multiplication.

## Verified physical-PC integration

Before SYSTEM migration, the end-to-end automatic request path was already exercised on `ChemistryPC` (`DESKTOP-AH6DPCN`) and confirmed:

- Intel Core i9-10900KF: 10 cores / 20 logical processors;
- about 63.87 GiB physical RAM;
- NVIDIA GeForce RTX 3070 with 8192 MiB VRAM;
- NVIDIA driver 616.56;
- persistent Python 3.13.15 + pip;
- PyTorch 2.14.0+cu126 with CUDA 12.6;
- `torch.cuda.is_available() == true`;
- actual 8192x8192 CUDA matrix multiplication, 8 iterations;
- CPU smoke using 20 workers;
- RAM smoke allocating/touching 8 GiB;
- successful artifact return from the physical PC.

After running `runner/enable-system.ps1`, the same execution plane runs those tasks with SYSTEM privileges.

## CASMI26 local data

`machine.json` defines the local data root. The controlled `casmi26` request expects:

```text
C:\ProgramData\ChemistryRunner\data\casmi26\test
C:\ProgramData\ChemistryRunner\data\casmi26\train
C:\ProgramData\ChemistryRunner\data\casmi26\sample_submission.csv
```

Local datasets, credentials, caches, model weights and generated large files are intentionally not committed to Git.

## Runner health / repair / removal

Local health check:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\health.ps1"
```

Re-running the elevated installer preserves/enforces permanent SYSTEM mode:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\install.ps1"
```

Remove runner (elevated PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\uninstall.ps1"
```

Persistent machine data is kept by default. Use `-RemoveData` only when intentionally deleting `C:\ProgramData\ChemistryRunner` as well.

## Trust model

Permanent SYSTEM mode intentionally makes this repository a privileged control plane. **Any identity that gains trusted write access to executable code/workflow/request paths in this private repository can potentially obtain full local control of the PC through the runner.** Treat GitHub account security and repository write access as equivalent to an administrator credential for this machine.

Current controls remain:

- the runner is repository-scoped rather than organization-wide;
- PC jobs require the `ChemistryPC` label;
- untrusted pull-request events do not execute PC jobs;
- task code is committed and reviewable instead of being hidden in free-form shell input;
- secrets/data are not intentionally committed or automatically uploaded;
- the bridge remains file-only and independent of execution;
- the dispatcher refuses execution unless the process identity is exactly `S-1-5-18`.

SYSTEM service execution is non-interactive Windows Session 0. It gives full local administrative authority for files, registry, services, installers, scheduled tasks, system configuration and background processes, but it does **not** by itself provide mouse/keyboard control of applications already open on the signed-in desktop. Interactive desktop automation would require a separate user-session component if that is ever needed.
