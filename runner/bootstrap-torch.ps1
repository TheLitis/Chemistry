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
        $text = (($output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
        if ($code -ne 0) {
            return [pscustomobject]@{ ok = $false; error = $text }
        }
        try {
            $obj = $text | ConvertFrom-Json
        } catch {
            return [pscustomobject]@{ ok = $false; error = "Invalid torch probe JSON: $text" }
        }
        return [pscustomobject]@{ ok = [bool]($obj.torch_available -and $obj.cuda_available); probe = $obj }
    } finally {
        $ErrorActionPreference = 'Stop'
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
}

$before = Test-TorchCuda -Interpreter $PythonExe
if ($before.ok) {
    Write-Host "PyTorch CUDA already ready: torch $($before.probe.torch_version), CUDA $($before.probe.torch_cuda_version)" -ForegroundColor Green
    Write-Host "Devices: $(($before.probe.devices) -join '; ')"
    exit 0
}

Write-Host "Installing/upgrading CUDA-enabled PyTorch from $IndexUrl ..."
$env:PIP_CACHE_DIR = $CacheRoot
& $PythonExe -m pip install --disable-pip-version-check --upgrade --index-url $IndexUrl torch
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
