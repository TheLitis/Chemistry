[CmdletBinding()]
param(
    [switch]$RemoveData
)

$ErrorActionPreference = 'Stop'

$Repository = 'TheLitis/Chemistry'
$RunnerName = 'ChemistryPC'
$RunnerRoot = 'C:\actions-runner\ChemistryPC'
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-GhJson {
    param([string[]]$Arguments)
    $output = & gh @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gh command failed: $($output -join [Environment]::NewLine)"
    }
    return ($output -join "`n") | ConvertFrom-Json
}

if (-not (Test-Administrator)) {
    throw 'Run this uninstaller from PowerShell opened with Run as administrator.'
}

$service = Get-CimInstance Win32_Service | Where-Object {
    ($_.Name -like 'actions.runner.*') -and (
        $_.PathName -like "*$RunnerRoot*" -or
        $_.Name -like "*$RunnerName*" -or
        $_.DisplayName -like "*$RunnerName*"
    )
} | Select-Object -First 1

if ($service -and $service.State -ne 'Stopped') {
    Stop-Service -Name $service.Name -Force -ErrorAction SilentlyContinue
}

$configCmd = Join-Path $RunnerRoot 'config.cmd'
if ((Test-Path (Join-Path $RunnerRoot '.runner')) -and (Test-Path $configCmd)) {
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        throw 'GitHub CLI is required to obtain a short-lived runner removal token.'
    }

    & gh auth status 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'GitHub CLI is not authenticated. Run gh auth login first.'
    }

    $removeToken = (Invoke-GhJson @('api', '--method', 'POST', "repos/$Repository/actions/runners/remove-token")).token
    if (-not $removeToken) {
        throw 'GitHub did not return a runner removal token.'
    }

    Push-Location $RunnerRoot
    try {
        & .\config.cmd remove --token $removeToken --unattended
        if ($LASTEXITCODE -ne 0) {
            throw "Runner removal failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
        $removeToken = $null
    }
}

if (Test-Path $RunnerRoot) {
    Remove-Item -LiteralPath $RunnerRoot -Recurse -Force
}

if ($RemoveData -and (Test-Path $StateRoot)) {
    Remove-Item -LiteralPath $StateRoot -Recurse -Force
}

Write-Host 'ChemistryPC runner removed.' -ForegroundColor Green
if (-not $RemoveData) {
    Write-Host "Persistent state/data was kept at $StateRoot"
}
