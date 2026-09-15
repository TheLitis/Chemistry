$ErrorActionPreference = 'Stop'

$RepoPath = Join-Path $HOME 'Chemistry'
$StateDir = Join-Path $env:LOCALAPPDATA 'ChemistryBridge'
$LogPath = Join-Path $StateDir 'bridge.log'

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Write-BridgeLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -Path $LogPath -Value $line
}

function Sync-ChemistryRepo {
    if (-not (Test-Path (Join-Path $RepoPath '.git'))) {
        Write-BridgeLog "Repository not found at $RepoPath"
        return
    }

    Push-Location $RepoPath
    try {
        $branch = (git branch --show-current).Trim()
        if ($branch -ne 'main') {
            Write-BridgeLog "Skipped: current branch is '$branch', expected 'main'."
            return
        }

        $dirty = git status --porcelain
        if ($dirty) {
            Write-BridgeLog 'Skipped: local working tree has uncommitted changes.'
            return
        }

        git fetch origin main --quiet
        $local = (git rev-parse HEAD).Trim()
        $remote = (git rev-parse origin/main).Trim()

        if ($local -eq $remote) {
            return
        }

        git merge-base --is-ancestor $local $remote
        if ($LASTEXITCODE -ne 0) {
            Write-BridgeLog "Skipped: local main is not an ancestor of origin/main (diverged or local-only commits)."
            return
        }

        git merge --ff-only origin/main --quiet
        Write-BridgeLog "Updated $local -> $remote"
    }
    catch {
        Write-BridgeLog "ERROR: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
}

Write-BridgeLog 'ChemistryBridge started.'
while ($true) {
    Sync-ChemistryRepo
    Start-Sleep -Seconds 10
}
