[CmdletBinding()]
param(
    [string]$Version = '3.13.15',
    [string]$StateRoot = (Join-Path $env:ProgramData 'ChemistryRunner')
)

$ErrorActionPreference = 'Stop'
$PythonRoot = Join-Path $StateRoot 'python'
$PythonExe = Join-Path $PythonRoot 'python.exe'
$CacheRoot = Join-Path $StateRoot 'cache\python-bootstrap'

function Test-Python([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        $text = (& $Path -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>$null | Select-Object -First 1)
        return ($LASTEXITCODE -eq 0 -and $text -match '^\d+\.\d+\.\d+$' -and [version]$text -ge [version]'3.11.0')
    } catch {
        return $false
    }
}

if (Test-Python $PythonExe) {
    Write-Host "Chemistry Python already available: $PythonExe"
    & $PythonExe --version
    if (& $PythonExe -m pip --version 2>$null) {
        Write-Output $PythonExe
        exit 0
    }
}

New-Item -ItemType Directory -Force -Path $StateRoot, $CacheRoot | Out-Null
$archive = Join-Path $CacheRoot "python-$Version-embed-amd64.zip"
$archiveUrl = "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip"
$getPip = Join-Path $CacheRoot 'get-pip.py'

if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host "Downloading Python $Version embeddable package from python.org..."
    Invoke-WebRequest -UseBasicParsing -Uri $archiveUrl -OutFile $archive
}
if (-not (Test-Path -LiteralPath $getPip)) {
    Write-Host 'Downloading pip bootstrap from bootstrap.pypa.io...'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip
}

Write-Host "Extracting isolated Chemistry Python into $PythonRoot ..."
if (Test-Path -LiteralPath $PythonRoot) {
    Remove-Item -LiteralPath $PythonRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PythonRoot | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $PythonRoot -Force

if (-not (Test-Python $PythonExe)) {
    throw "Embedded Python extracted but $PythonExe is not usable."
}

$pth = Get-ChildItem -LiteralPath $PythonRoot -Filter 'python*._pth' -File | Select-Object -First 1
if (-not $pth) {
    throw 'Embedded Python _pth file was not found.'
}

$lines = Get-Content -LiteralPath $pth.FullName
$updated = New-Object System.Collections.Generic.List[string]
$hasLib = $false
$hasSitePackages = $false
$hasImportSite = $false
foreach ($line in $lines) {
    $trimmed = $line.Trim()
    if ($trimmed -ieq 'Lib') { $hasLib = $true }
    if ($trimmed -ieq 'Lib\site-packages') { $hasSitePackages = $true }
    if ($trimmed -match '^#?\s*import\s+site\s*$') {
        if (-not $hasImportSite) {
            $updated.Add('import site')
            $hasImportSite = $true
        }
        continue
    }
    $updated.Add($line)
}
if (-not $hasLib) { $updated.Add('Lib') }
if (-not $hasSitePackages) { $updated.Add('Lib\site-packages') }
if (-not $hasImportSite) { $updated.Add('import site') }
Set-Content -LiteralPath $pth.FullName -Value $updated -Encoding ASCII

Write-Host 'Bootstrapping pip inside isolated Chemistry Python...'
& $PythonExe $getPip --disable-pip-version-check --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    throw "get-pip.py failed with exit code $LASTEXITCODE."
}

& $PythonExe -m pip --version
if ($LASTEXITCODE -ne 0) {
    throw 'Python is available but pip bootstrap failed.'
}

Write-Host "Chemistry Python ready: $PythonExe" -ForegroundColor Green
& $PythonExe --version
Write-Output $PythonExe
