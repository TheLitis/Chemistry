# Chemistry PC Bridge

`TheLitis/Chemistry` is a private GitHub repository used as a delivery bridge to a Windows PC.

The PC runs a small local watcher that keeps the local `main` branch synchronized with `origin/main`. This means files committed to this repository can arrive on the PC automatically without exposing an inbound port, RDP, SSH, or a public listener.

## Local location

By default the repository is cloned to:

```text
%USERPROFILE%\Chemistry
```

Delivered work should normally be placed under:

```text
work\
```

## One-time setup on the PC

Open PowerShell and run:

```powershell
gh auth status
cd $HOME
if (-not (Test-Path "$HOME\Chemistry\.git")) {
    gh repo clone TheLitis/Chemistry "$HOME\Chemistry"
} else {
    git -C "$HOME\Chemistry" pull --ff-only origin main
}

powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\bridge\install.ps1"
```

After installation, `ChemistryBridge` starts automatically when you sign in to Windows and checks GitHub for updates every 10 seconds.

## How synchronization behaves

- Only the `main` branch is followed.
- Incoming changes are applied only with a fast-forward update.
- If the local repository has uncommitted changes, is on another branch, has local-only commits, or has diverged from `origin/main`, the bridge does **not** overwrite anything. It pauses synchronization and records the reason in the log.
- The bridge does not execute files received from GitHub. It only synchronizes the repository.

## Log

```powershell
Get-Content "$env:LOCALAPPDATA\ChemistryBridge\bridge.log" -Tail 50 -Wait
```

## Reinstall / update the bridge

After pulling the latest repository version, run:

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\bridge\install.ps1"
```

## Remove the bridge

```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\Chemistry\bridge\uninstall.ps1"
```

## Security model

This repository is private. The PC initiates outbound connections to GitHub; no inbound network port is opened. The default bridge intentionally does not provide remote shell execution. If remote builds/tests are needed later, add a separately labeled GitHub Actions self-hosted runner with restricted workflows rather than turning the sync process into an unrestricted command channel.
