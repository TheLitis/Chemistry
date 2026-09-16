[CmdletBinding()]
param(
    [switch]$Repair
)

$ErrorActionPreference = 'Stop'

$Repository = 'TheLitis/Chemistry'
$RepositoryUrl = 'https://github.com/TheLitis/Chemistry'
$RunnerName = 'ChemistryPC'
$RunnerRoot = 'C:\actions-runner\ChemistryPC'
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'
$DataRoot = Join-Path $StateRoot 'data'
$CacheRoot = Join-Path $StateRoot 'cache'
$MachineConfig = Join-Path $StateRoot 'machine.json'
$RepoPath = Join-Path $env:USERPROFILE 'Chemistry'
$EnableSystemScript = Join-Path $PSScriptRoot 'enable-system.ps1'
$SystemSid = 'S-1-5-18'

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

function Get-RunnerService {
    $services = Get-CimInstance Win32_Service | Where-Object {
        ($_.Name -like 'actions.runner.*') -and (
            $_.PathName -like "*$RunnerRoot*" -or
            $_.Name -like "*$RunnerName*" -or
            $_.DisplayName -like "*$RunnerName*"
        )
    }
    return $services | Select-Object -First 1
}

function Write-MachineConfig {
    New-Item -ItemType Directory -Force -Path $StateRoot, $DataRoot, $CacheRoot | Out-Null

    $config = [ordered]@{
        repository = $Repository
        runnerName = $RunnerName
        runnerRoot = $RunnerRoot
        repoPath = $RepoPath
        dataRoot = $DataRoot
        cacheRoot = $CacheRoot
        maxRamSmokeGiB = 8
        desiredPrivilegeMode = 'SYSTEM'
        desiredPrivilegeSid = $SystemSid
        installedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $config | ConvertTo-Json -Depth 4 | Set-Content -Path $MachineConfig -Encoding UTF8

    # NETWORK SERVICE is used only as the bootstrap account before enable-system.ps1 switches the service to LocalSystem.
    & icacls.exe $StateRoot /grant '*S-1-5-20:(OI)(CI)M' /T /C | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to grant bootstrap NETWORK SERVICE access to $StateRoot"
    }
}

if (-not (Test-Administrator)) {
    throw 'Run this installer from PowerShell opened with Run as administrator.'
}
if (-not (Test-Path -LiteralPath $EnableSystemScript -PathType Leaf)) {
    throw "SYSTEM migration script is missing: $EnableSystemScript"
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is not available in PATH.'
}

& gh auth status 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub CLI is not authenticated. Run gh auth login first.'
}

if (-not (Test-Path (Join-Path $RepoPath '.git'))) {
    throw "Expected synchronized repository at $RepoPath."
}

New-Item -ItemType Directory -Force -Path $RunnerRoot, $StateRoot | Out-Null
Write-MachineConfig

$configured = Test-Path (Join-Path $RunnerRoot '.runner')
$service = Get-RunnerService

if ($configured -and $service -and -not $Repair) {
    & $EnableSystemScript
    $service = Get-RunnerService
    Write-Host "ChemistryPC is already configured in permanent SYSTEM mode. Service: $($service.Name)" -ForegroundColor Green
    Write-Host "Service account: $($service.StartName)"
    Write-Host "State: $StateRoot"
    Write-Host "Machine config: $MachineConfig"
    exit 0
}

if ($configured -and $Repair) {
    Write-Host 'Repair requested: removing the existing local runner configuration.' -ForegroundColor Yellow
    if ($service -and $service.State -ne 'Stopped') {
        Stop-Service -Name $service.Name -Force -ErrorAction SilentlyContinue
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
    }
    $configured = $false
    $service = $null
}

if ($configured -and -not $service) {
    throw 'Runner configuration exists but its Windows service is missing. Re-run this command with -Repair.'
}

$configCmd = Join-Path $RunnerRoot 'config.cmd'
if (-not (Test-Path $configCmd)) {
    Write-Host 'Resolving current GitHub Actions runner package...'
    $downloads = Invoke-GhJson @('api', "repos/$Repository/actions/runners/downloads")
    $package = $downloads | Where-Object { $_.os -eq 'win' -and $_.architecture -eq 'x64' } | Select-Object -First 1
    if (-not $package) {
        throw 'GitHub did not return a Windows x64 runner package.'
    }

    $zipPath = Join-Path $env:TEMP $package.filename
    Write-Host "Downloading $($package.filename)..."
    Invoke-WebRequest -UseBasicParsing -Uri $package.download_url -OutFile $zipPath

    if (Test-Path $RunnerRoot) {
        Get-ChildItem -LiteralPath $RunnerRoot -Force | Remove-Item -Recurse -Force
    }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $RunnerRoot -Force
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $configCmd)) {
    throw "Runner package is incomplete: $configCmd is missing."
}

$registration = Invoke-GhJson @('api', '--method', 'POST', "repos/$Repository/actions/runners/registration-token")
$registrationToken = $registration.token
if (-not $registrationToken) {
    throw 'GitHub did not return a runner registration token.'
}

Write-Host 'Registering ChemistryPC as a repository-scoped Windows service...'
Push-Location $RunnerRoot
try {
    # The runner installer supports NETWORK SERVICE reliably. Immediately after registration,
    # enable-system.ps1 changes the Windows service logon to LocalSystem (S-1-5-18).
    & .\config.cmd `
        --url $RepositoryUrl `
        --token $registrationToken `
        --name $RunnerName `
        --labels $RunnerName `
        --work '_work' `
        --unattended `
        --replace `
        --runasservice `
        --windowslogonaccount 'NT AUTHORITY\NETWORK SERVICE'

    if ($LASTEXITCODE -ne 0) {
        throw "Runner configuration failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $registrationToken = $null
}

$service = Get-RunnerService
if (-not $service) {
    throw 'Runner configuration completed but the Windows service was not found.'
}

& $EnableSystemScript
$service = Get-RunnerService
if (-not $service -or $service.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
    throw 'Runner registration succeeded, but permanent SYSTEM mode was not established.'
}

$remoteRunner = $null
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 2
    $remote = Invoke-GhJson @('api', "repos/$Repository/actions/runners")
    $remoteRunner = $remote.runners | Where-Object { $_.name -eq $RunnerName } | Select-Object -First 1
    if ($remoteRunner -and $remoteRunner.status -eq 'online') {
        break
    }
}

Write-Host ''
Write-Host 'ChemistryPC installation complete in permanent SYSTEM mode.' -ForegroundColor Green
Write-Host "Runner directory: $RunnerRoot"
Write-Host "State/data directory: $StateRoot"
Write-Host "Service: $($service.Name)"
Write-Host "Service account: $($service.StartName)"
Write-Host "Privilege SID: $SystemSid"
if ($remoteRunner) {
    Write-Host "GitHub status: $($remoteRunner.status), busy=$($remoteRunner.busy)"
} else {
    Write-Warning 'The service is installed, but GitHub has not returned the runner yet. Run runner\health.ps1 to re-check.'
}
