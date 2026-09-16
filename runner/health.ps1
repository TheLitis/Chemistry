$ErrorActionPreference = 'Stop'

$Repository = 'TheLitis/Chemistry'
$RunnerName = 'ChemistryPC'
$RunnerRoot = 'C:\actions-runner\ChemistryPC'
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'
$MachineConfig = Join-Path $StateRoot 'machine.json'
$SystemSid = 'S-1-5-18'

function Invoke-GhJson {
    param([string[]]$Arguments)
    $output = & gh @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gh command failed: $($output -join [Environment]::NewLine)"
    }
    return ($output -join "`n") | ConvertFrom-Json
}

$ok = $true

Write-Host '=== Local runner files ==='
$configured = Test-Path (Join-Path $RunnerRoot '.runner')
Write-Host "Configured: $configured"
Write-Host "Machine config: $(Test-Path $MachineConfig) ($MachineConfig)"
if (-not $configured) { $ok = $false }

Write-Host "`n=== Windows service ==="
$service = Get-CimInstance Win32_Service | Where-Object {
    ($_.Name -like 'actions.runner.*') -and (
        $_.PathName -like "*$RunnerRoot*" -or
        $_.Name -like "*$RunnerName*" -or
        $_.DisplayName -like "*$RunnerName*"
    )
} | Select-Object -First 1

if ($service) {
    Write-Host "Name: $($service.Name)"
    Write-Host "State: $($service.State)"
    Write-Host "Start mode: $($service.StartMode)"
    Write-Host "Service account: $($service.StartName)"
    Write-Host "Required privilege SID: $SystemSid"
    if ($service.State -ne 'Running') { $ok = $false }
    if ($service.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
        Write-Host 'Permanent SYSTEM mode: NOT ENABLED' -ForegroundColor Red
        $ok = $false
    } else {
        Write-Host 'Permanent SYSTEM mode: ENABLED' -ForegroundColor Green
    }
} else {
    Write-Host 'Service: NOT FOUND' -ForegroundColor Red
    $ok = $false
}

if (Test-Path -LiteralPath $MachineConfig) {
    try {
        $machine = Get-Content -LiteralPath $MachineConfig -Raw | ConvertFrom-Json
        Write-Host "Configured privilege mode: $($machine.privilegeMode)"
        Write-Host "Configured privilege SID: $($machine.privilegeSid)"
        if ($machine.privilegeMode -ne 'SYSTEM' -or $machine.privilegeSid -ne $SystemSid) {
            $ok = $false
        }
    } catch {
        Write-Host "Machine config read failed: $($_.Exception.Message)" -ForegroundColor Red
        $ok = $false
    }
}

Write-Host "`n=== GitHub registration ==="
if (Get-Command gh -ErrorAction SilentlyContinue) {
    & gh auth status 1>$null 2>$null
    if ($LASTEXITCODE -eq 0) {
        $remote = Invoke-GhJson @('api', "repos/$Repository/actions/runners")
        $runner = $remote.runners | Where-Object { $_.name -eq $RunnerName } | Select-Object -First 1
        if ($runner) {
            Write-Host "Name: $($runner.name)"
            Write-Host "Status: $($runner.status)"
            Write-Host "Busy: $($runner.busy)"
            Write-Host "Labels: $(($runner.labels | ForEach-Object { $_.name }) -join ', ')"
            if ($runner.status -ne 'online') { $ok = $false }
        } else {
            Write-Host 'GitHub runner: NOT REGISTERED' -ForegroundColor Red
            $ok = $false
        }
    } else {
        Write-Host 'GitHub CLI is not authenticated; remote status unavailable.' -ForegroundColor Yellow
        $ok = $false
    }
} else {
    Write-Host 'GitHub CLI is not available; remote status unavailable.' -ForegroundColor Yellow
    $ok = $false
}

if ($ok) {
    Write-Host "`nChemistryPC health: OK (permanent SYSTEM mode)" -ForegroundColor Green
    exit 0
}

Write-Host "`nChemistryPC health: NOT READY" -ForegroundColor Red
exit 1
