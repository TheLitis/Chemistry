$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$files = @(
    (Join-Path $repoRoot 'bridge\install.ps1'),
    (Join-Path $repoRoot 'bridge\sync.ps1')
)

$failed = $false

foreach ($file in $files) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors) | Out-Null

    if ($errors.Count -gt 0) {
        $failed = $true
        Write-Host "Parse errors in $file:" -ForegroundColor Red
        foreach ($error in $errors) {
            Write-Host "  $($error.Message) at $($error.Extent.StartLineNumber):$($error.Extent.StartColumnNumber)" -ForegroundColor Red
        }
    }
}

if ($failed) {
    exit 1
}

Write-Host 'Bridge PowerShell scripts parse successfully.'
