# Chemistry PC Bridge

`TheLitis/Chemistry` is a private GitHub repository used as a delivery bridge and controlled execution plane for a Windows PC.

The existing bridge keeps local `main` synchronized with GitHub without exposing an inbound port. A separate repository-scoped GitHub Actions self-hosted runner named `ChemistryPC` can execute reviewed workflows on the PC and return logs/artifacts to GitHub.

## Local repository

```text
%USERPROFILE%\Chemistry
```

## File bridge

The bridge follows only `main`, applies only safe fast-forward updates, and never evaluates or executes files it receives. If the local checkout is dirty, diverged, or on another branch, synchronization pauses rather than overwriting local work.

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

## One-time ChemistryPC execution bootstrap

The runner installer is intentionally separate from the file bridge. It installs the official GitHub Actions Windows x64 runner at:

```text
C:\actions-runner\ChemistryPC
```

Persistent machine state/data is kept outside Git at:

```text
C:\ProgramData\ChemistryRunner
```

From a normal PowerShell window, run this one command. It first fast-forwards the local repository, then opens one elevated PowerShell window for the service installation:

```powershell
git -C "$HOME\Chemistry" pull --ff-only origin main; Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList '-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',"$HOME\Chemistry\runner\install.ps1"
```

The installer:

- validates `gh` authentication;
- obtains short-lived runner download/registration information at runtime;
- downloads the official current Windows x64 GitHub Actions runner;
- registers only against `TheLitis/Chemistry` as `ChemistryPC`;
- adds the custom `ChemistryPC` label;
- runs it as a Windows service under `NT AUTHORITY\NETWORK SERVICE`;
- configures automatic service startup;
- stores no registration token in Git;
- creates `C:\ProgramData\ChemistryRunner\machine.json` and persistent data/cache directories.

Re-running the installer is safe when the existing runner service is healthy. For an explicitly broken installation, run the installer with `-Repair`.

## Runner health

After bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\health.ps1"
```

Healthy output requires local runner configuration, a running Windows service, and an online GitHub runner named `ChemistryPC`.

The manually dispatched `ChemistryPC health` workflow targets only:

```yaml
runs-on: [self-hosted, Windows, X64, ChemistryPC]
```

It provisions a job-local Python runtime, records CPU/RAM/disk/NVIDIA/Python/PyTorch visibility, runs bounded CPU/RAM compute checks and optional CUDA compute when a CUDA-capable PyTorch is available, then uploads JSON reports as workflow artifacts.

## Remove the execution runner

Run in an elevated PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\runner\uninstall.ps1"
```

Persistent machine data is kept by default. Add `-RemoveData` only if you intentionally want to delete `C:\ProgramData\ChemistryRunner` too.

## Security model

No SSH, RDP, WinRM, or other inbound remote-control port is opened. The private repository is the execution control plane. PC workflows use the `ChemistryPC` label and are not triggered by untrusted pull requests. The execution design deliberately avoids an arbitrary shell-text workflow input: executable tasks must exist as reviewed repository code/workflows.

The runner service is not intended to grant administrator privileges to normal jobs. The one-time bootstrap itself requires elevation because Windows service registration does.
