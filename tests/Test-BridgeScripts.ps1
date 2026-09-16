$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$files = @(
    (Join-Path $repoRoot 'bridge\install.ps1'),
    (Join-Path $repoRoot 'bridge\sync.ps1'),
    (Join-Path $repoRoot 'bridge\uninstall.ps1'),
    (Join-Path $repoRoot 'runner\install.ps1'),
    (Join-Path $repoRoot 'runner\enable-system.ps1'),
    (Join-Path $repoRoot 'runner\health.ps1'),
    (Join-Path $repoRoot 'runner\uninstall.ps1'),
    (Join-Path $repoRoot 'runner\diagnostics.ps1'),
    (Join-Path $repoRoot 'runner\resolve-python.ps1'),
    (Join-Path $repoRoot 'runner\bootstrap-python.ps1'),
    (Join-Path $repoRoot 'runner\bootstrap-torch.ps1'),
    (Join-Path $repoRoot 'runner\run-request.ps1')
)

$regressionSource = @'
$cmd = "@echo off`r`nstart \"ChemistryBridge\" /min powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"$BridgeScript\"`r`n"
'@
$regressionTokens = $null
$regressionErrors = $null
[System.Management.Automation.Language.Parser]::ParseInput(
    $regressionSource,
    [ref]$regressionTokens,
    [ref]$regressionErrors
) | Out-Null

if ($regressionErrors.Count -eq 0) {
    Write-Host 'Regression check failed: the original C-style quote escaping was not rejected.' -ForegroundColor Red
    exit 1
}

$failed = $false

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        $failed = $true
        Write-Host "Required PowerShell script is missing: $file" -ForegroundColor Red
        continue
    }

    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors) | Out-Null

    if ($errors.Count -gt 0) {
        $failed = $true
        Write-Host "Parse errors in ${file}:" -ForegroundColor Red
        foreach ($error in $errors) {
            Write-Host "  $($error.Message) at $($error.Extent.StartLineNumber):$($error.Extent.StartColumnNumber)" -ForegroundColor Red
        }
    }
}

if ($failed) {
    exit 1
}

Write-Host 'Regression fixture is rejected and all bridge/runner PowerShell scripts parse successfully.'
