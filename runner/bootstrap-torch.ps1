[CmdletBinding()]
param(
    [string]$PythonExe = (Join-Path $env:ProgramData 'ChemistryRunner\python\python.exe'),
    [string]$IndexUrl = 'https://download.pytorch.org/whl/cu126'
)

$ErrorActionPreference = 'Stop'
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'
$CacheRoot = Join-Path $StateRoot 'cache\pip'
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Chemistry Python is missing: $PythonExe"
}

function Test-Package {
    param([string]$Interpreter, [string]$Name)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Interpreter -m pip show $Name *> $null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Test-TorchCuda {
    param([string]$Interpreter)

    $probePath = Join-Path $env:TEMP ("chemistry-torch-probe-{0}.py" -f [guid]::NewGuid())
    $probe = @'
import json
try:
    import torch
    result = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
except Exception as exc:
    result = {"torch_available": False, "cuda_available": False, "error": repr(exc)}
print(json.dumps(result))
'@

    try {
        Set-Content -LiteralPath $probePath -Value $probe -Encoding UTF8
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $output = @(& $Interpreter $probePath 2>&1)
        $code = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
        $lines = @($output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
        $text = ($lines -join "`n").Trim()
        if ($code -ne 0) {
            return [pscustomobject]@{ ok = $false; error = $text }
        }
        $jsonLine = $lines | Where-Object { $_ -match '^\{.*\}$' } | Select-Object -Last 1
        if (-not $jsonLine) {
            return [pscustomobject]@{ ok = $false; error = "Torch probe produced no JSON object: $text" }
        }
        try {
            $obj = $jsonLine | ConvertFrom-Json
        } catch {
            return [pscustomobject]@{ ok = $false; error = "Invalid torch probe JSON: $jsonLine`nFull output:`n$text" }
        }
        return [pscustomobject]@{ ok = [bool]($obj.torch_available -and $obj.cuda_available); probe = $obj; output = $text }
    } finally {
        $ErrorActionPreference = 'Stop'
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
}

$env:PIP_CACHE_DIR = $CacheRoot
if (-not (Test-Package -Interpreter $PythonExe -Name 'numpy')) {
    Write-Host 'Installing NumPy into persistent Chemistry Python...'
    & $PythonExe -m pip install --disable-pip-version-check --no-warn-script-location numpy
    if ($LASTEXITCODE -ne 0) { throw "NumPy installation failed with exit code $LASTEXITCODE." }
}

$before = Test-TorchCuda -Interpreter $PythonExe
if ($before.ok) {
    Write-Host "PyTorch CUDA already ready: torch $($before.probe.torch_version), CUDA $($before.probe.torch_cuda_version)" -ForegroundColor Green
    Write-Host "Devices: $(($before.probe.devices) -join '; ')"
    exit 0
}

Write-Host "Installing/upgrading CUDA-enabled PyTorch from $IndexUrl ..."
& $PythonExe -m pip install --disable-pip-version-check --no-warn-script-location --upgrade --index-url $IndexUrl torch
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installation failed with exit code $LASTEXITCODE."
}

$after = Test-TorchCuda -Interpreter $PythonExe
if (-not $after.ok) {
    $details = if ($after.probe) { $after.probe | ConvertTo-Json -Depth 6 -Compress } else { $after.error }
    throw "PyTorch installed but CUDA verification failed: $details"
}

Write-Host "PyTorch CUDA ready: torch $($after.probe.torch_version), CUDA $($after.probe.torch_cuda_version)" -ForegroundColor Green
Write-Host "Devices: $(($after.probe.devices) -join '; ')"
