[CmdletBinding()]
param(
    [string]$MachineConfigPath = (Join-Path $env:ProgramData 'ChemistryRunner\machine.json')
)

$ErrorActionPreference = 'Stop'
$candidates = New-Object System.Collections.Generic.List[string]

function Add-Candidate([string]$Path) {
    if (-not $Path) { return }
    try { $full = [System.IO.Path]::GetFullPath($Path) } catch { return }
    if ((Test-Path -LiteralPath $full) -and -not $candidates.Contains($full)) {
        $candidates.Add($full)
    }
}

$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd -and $cmd.Source) { Add-Candidate $cmd.Source }
$cmd3 = Get-Command python3 -ErrorAction SilentlyContinue
if ($cmd3 -and $cmd3.Source) { Add-Candidate $cmd3.Source }

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    try {
        $resolved = (& $py.Source -3 -c 'import sys; print(sys.executable)' 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0) { Add-Candidate $resolved }
    } catch { }
}

if (Test-Path -LiteralPath $MachineConfigPath) {
    try {
        $cfg = Get-Content -LiteralPath $MachineConfigPath -Raw | ConvertFrom-Json
        if ($cfg.pythonPath) { Add-Candidate ([string]$cfg.pythonPath) }
        if ($cfg.repoPath) {
            $profileRoot = Split-Path -Parent ([string]$cfg.repoPath)
            Get-ChildItem -Path (Join-Path $profileRoot 'AppData\Local\Programs\Python\Python*\python.exe') -File -ErrorAction SilentlyContinue |
                ForEach-Object { Add-Candidate $_.FullName }
        }
    } catch { }
}

@($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ } | ForEach-Object {
    Get-ChildItem -Path (Join-Path $_ 'Python*\python.exe') -File -ErrorAction SilentlyContinue |
        ForEach-Object { Add-Candidate $_.FullName }
}

$valid = foreach ($candidate in $candidates) {
    try {
        $text = (& $candidate -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $text -match '^\d+\.\d+\.\d+$') {
            $version = [version]$text
            if ($version -ge [version]'3.11.0') {
                [pscustomobject]@{ Path = $candidate; Version = $version }
            }
        }
    } catch { }
}

$selected = $valid | Sort-Object Version -Descending | Select-Object -First 1
if (-not $selected) {
    throw 'No accessible Python 3.11+ interpreter found for the ChemistryPC service account.'
}

Write-Output $selected.Path
