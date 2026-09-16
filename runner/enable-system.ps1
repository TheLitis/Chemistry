[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RunnerName = 'ChemistryPC'
$RunnerRoot = 'C:\actions-runner\ChemistryPC'
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'
$MachineConfig = Join-Path $StateRoot 'machine.json'
$SystemSid = 'S-1-5-18'
$SystemAccount = 'LocalSystem'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-RunnerService {
    Get-CimInstance Win32_Service | Where-Object {
        ($_.Name -like 'actions.runner.*') -and (
            $_.PathName -like "*$RunnerRoot*" -or
            $_.Name -like "*$RunnerName*" -or
            $_.DisplayName -like "*$RunnerName*"
        )
    } | Select-Object -First 1
}

function Write-SystemModeConfig {
    param($Service)

    New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null

    $config = if (Test-Path -LiteralPath $MachineConfig) {
        Get-Content -LiteralPath $MachineConfig -Raw | ConvertFrom-Json
    } else {
        [pscustomobject]@{}
    }

    $values = [ordered]@{
        runnerServiceAccount = [string]$Service.StartName
        privilegeMode = 'SYSTEM'
        privilegeSid = $SystemSid
        systemModeEnabledAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    foreach ($entry in $values.GetEnumerator()) {
        $config | Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
    }
    $config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $MachineConfig -Encoding UTF8
}

if (-not (Test-Administrator)) {
    throw 'This one-time migration must be run from an elevated PowerShell window.'
}

$service = Get-RunnerService
if (-not $service) {
    throw 'ChemistryPC Windows service was not found.'
}

Write-Host "Service: $($service.Name)"
Write-Host "Current account: $($service.StartName)"

if ($service.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
    if ($service.State -ne 'Stopped') {
        Write-Host 'Stopping ChemistryPC service...'
        Stop-Service -Name $service.Name -Force
    }

    Write-Host 'Changing ChemistryPC service account to LocalSystem (S-1-5-18)...'
    $scOutput = & sc.exe config $service.Name obj= $SystemAccount 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "sc.exe failed to switch the runner to LocalSystem: $($scOutput -join [Environment]::NewLine)"
    }
} elseif ($service.State -ne 'Stopped') {
    Stop-Service -Name $service.Name -Force
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
foreach ($path in @($RunnerRoot, $StateRoot)) {
    $aclOutput = & icacls.exe $path /grant "*$SystemSid:(OI)(CI)F" /T /C 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to grant SYSTEM full control to ${path}: $($aclOutput -join [Environment]::NewLine)"
    }
}

Set-Service -Name $service.Name -StartupType Automatic
Start-Service -Name $service.Name

$deadline = (Get-Date).AddSeconds(30)
do {
    Start-Sleep -Milliseconds 500
    $service = Get-RunnerService
} while ($service -and $service.State -ne 'Running' -and (Get-Date) -lt $deadline)

if (-not $service -or $service.State -ne 'Running') {
    throw 'ChemistryPC service did not return to Running state after SYSTEM migration.'
}
if ($service.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
    throw "ChemistryPC service account is still '$($service.StartName)', expected LocalSystem."
}

Write-SystemModeConfig -Service $service

Write-Host ''
Write-Host 'ChemistryPC permanent SYSTEM mode enabled.' -ForegroundColor Green
Write-Host "Service account: $($service.StartName)"
Write-Host "Privilege SID: $SystemSid"
Write-Host "State: $($service.State)"
Write-Host 'All future ChemistryPC jobs now run with NT AUTHORITY\SYSTEM privileges.'
