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
    Write-Output $PythonExe
    exit 0
}

New-Item -ItemType Directory -Force -Path $StateRoot, $CacheRoot | Out-Null
$installer = Join-Path $CacheRoot "python-$Version-amd64.exe"
$url = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"

if (-not (Test-Path -LiteralPath $installer)) {
    Write-Host "Downloading Python $Version from python.org..."
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
}

$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid') {
    throw "Python installer signature is not valid: $($signature.Status)"
}
if (-not $signature.SignerCertificate -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
    throw "Unexpected Python installer signer: $($signature.SignerCertificate.Subject)"
}

Write-Host "Installing isolated Chemistry Python into $PythonRoot ..."
$args = @(
    '/quiet',
    'InstallAllUsers=0',
    'Include_launcher=0',
    'Include_test=0',
    'Include_doc=0',
    'Include_tcltk=0',
    'Include_pip=1',
    'PrependPath=0',
    'Shortcuts=0',
    "TargetDir=$PythonRoot"
)
$process = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru -NoNewWindow
if ($process.ExitCode -ne 0) {
    throw "Python installer exited with code $($process.ExitCode)."
}

if (-not (Test-Python $PythonExe)) {
    throw "Python installation completed but $PythonExe is not usable."
}

& $PythonExe -m pip --version
if ($LASTEXITCODE -ne 0) {
    throw 'Python is installed but pip is unavailable.'
}

Write-Host "Chemistry Python ready: $PythonExe" -ForegroundColor Green
& $PythonExe --version
Write-Output $PythonExe
