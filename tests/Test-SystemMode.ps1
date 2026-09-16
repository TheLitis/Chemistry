$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$installPath = Join-Path $repoRoot 'runner\install.ps1'
$enablePath = Join-Path $repoRoot 'runner\enable-system.ps1'
$requestPath = Join-Path $repoRoot 'runner\run-request.ps1'
$healthPath = Join-Path $repoRoot 'runner\health.ps1'

$failed = $false

if (-not (Test-Path -LiteralPath $enablePath -PathType Leaf)) {
    Write-Host 'Missing runner/enable-system.ps1 migration script.' -ForegroundColor Red
    $failed = $true
}

foreach ($item in @(
    @{ Path = $installPath; Name = 'runner/install.ps1' },
    @{ Path = $requestPath; Name = 'runner/run-request.ps1' },
    @{ Path = $healthPath; Name = 'runner/health.ps1' }
)) {
    if (-not (Test-Path -LiteralPath $item.Path -PathType Leaf)) {
        Write-Host "Missing $($item.Name)." -ForegroundColor Red
        $failed = $true
        continue
    }

    $text = Get-Content -LiteralPath $item.Path -Raw
    if ($text -notmatch 'S-1-5-18|LocalSystem|NT AUTHORITY\\SYSTEM') {
        Write-Host "$($item.Name) does not enforce/verify SYSTEM execution." -ForegroundColor Red
        $failed = $true
    }
}

if (Test-Path -LiteralPath $enablePath -PathType Leaf) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($enablePath, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count -gt 0) {
        Write-Host 'runner/enable-system.ps1 has parse errors.' -ForegroundColor Red
        $failed = $true
    }

    $enableText = Get-Content -LiteralPath $enablePath -Raw
    foreach ($required in @('LocalSystem', 'S-1-5-18', 'Start-Service', 'Stop-Service')) {
        if ($enableText -notmatch [regex]::Escape($required)) {
            Write-Host "runner/enable-system.ps1 is missing required behavior marker: $required" -ForegroundColor Red
            $failed = $true
        }
    }
}

if ($failed) { exit 1 }
Write-Host 'Permanent SYSTEM-mode invariants are present.'
