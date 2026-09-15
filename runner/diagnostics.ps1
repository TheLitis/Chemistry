[CmdletBinding()]
param(
    [string]$OutputDirectory = $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { Join-Path $env:TEMP 'ChemistryRunnerHealth' })
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Invoke-PythonProbe {
    $probe = @'
import json
result = {"available": True}
try:
    import torch
    result.update({
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    })
except Exception as exc:
    result.update({"torch_available": False, "torch_error": repr(exc), "cuda_available": False, "cuda_device_count": 0, "cuda_devices": []})
print(json.dumps(result))
'@

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $version = (& $python.Source --version 2>&1) -join ' '
        $json = (& $python.Source -c $probe 2>&1) -join "`n"
        if ($LASTEXITCODE -eq 0) {
            $obj = $json | ConvertFrom-Json
            $obj | Add-Member -NotePropertyName executable -NotePropertyValue $python.Source -Force
            $obj | Add-Member -NotePropertyName version -NotePropertyValue $version -Force
            return $obj
        }
        return [pscustomobject]@{ available = $false; executable = $python.Source; version = $version; error = $json }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $version = (& $py.Source -3 --version 2>&1) -join ' '
        $json = (& $py.Source -3 -c $probe 2>&1) -join "`n"
        if ($LASTEXITCODE -eq 0) {
            $obj = $json | ConvertFrom-Json
            $obj | Add-Member -NotePropertyName executable -NotePropertyValue $py.Source -Force
            $obj | Add-Member -NotePropertyName launcher_args -NotePropertyValue '-3' -Force
            $obj | Add-Member -NotePropertyName version -NotePropertyValue $version -Force
            return $obj
        }
        return [pscustomobject]@{ available = $false; executable = $py.Source; launcher_args = '-3'; version = $version; error = $json }
    }

    return [pscustomobject]@{ available = $false; error = 'python/py not found in PATH' }
}

$cpuRows = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
$disks = Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | ForEach-Object {
    [ordered]@{
        device = $_.DeviceID
        size_gib = [math]::Round($_.Size / 1GB, 2)
        free_gib = [math]::Round($_.FreeSpace / 1GB, 2)
    }
}

$nvidia = [ordered]@{ available = $false }
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $banner = (& $nvidiaSmi.Source 2>&1) -join "`n"
    $query = (& $nvidiaSmi.Source --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -eq 0) {
        $gpus = @()
        foreach ($line in $query) {
            $parts = $line -split ',' | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 4) {
                $gpus += [ordered]@{
                    name = $parts[0]
                    memory_total_mib = [int]$parts[1]
                    memory_free_mib = [int]$parts[2]
                    driver_version = $parts[3]
                }
            }
        }
        $cudaVersion = $null
        if ($banner -match 'CUDA Version:\s*([0-9.]+)') {
            $cudaVersion = $Matches[1]
        }
        $nvidia = [ordered]@{
            available = $true
            executable = $nvidiaSmi.Source
            reported_cuda_version = $cudaVersion
            gpus = $gpus
        }
    } else {
        $nvidia = [ordered]@{ available = $false; executable = $nvidiaSmi.Source; error = ($query -join "`n") }
    }
}

$runnerService = Get-CimInstance Win32_Service | Where-Object {
    $_.Name -like 'actions.runner.*' -and ($_.Name -like '*ChemistryPC*' -or $_.DisplayName -like '*ChemistryPC*')
} | Select-Object -First 1

$report = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    hostname = $env:COMPUTERNAME
    windows = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build = $os.BuildNumber
    }
    cpu = [ordered]@{
        models = @($cpuRows | ForEach-Object { $_.Name.Trim() })
        sockets = @($cpuRows).Count
        cores = [int](($cpuRows | Measure-Object -Property NumberOfCores -Sum).Sum)
        logical_processors = [int](($cpuRows | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum)
    }
    memory = [ordered]@{
        total_gib = [math]::Round(($os.TotalVisibleMemorySize * 1KB) / 1GB, 2)
        available_gib = [math]::Round(($os.FreePhysicalMemory * 1KB) / 1GB, 2)
    }
    disks = @($disks)
    nvidia = $nvidia
    python = Invoke-PythonProbe
    runner_service = $(if ($runnerService) {
        [ordered]@{ found = $true; name = $runnerService.Name; state = $runnerService.State; start_mode = $runnerService.StartMode }
    } else {
        [ordered]@{ found = $false }
    })
}

$jsonPath = Join-Path $OutputDirectory 'runner-health.json'
$report | ConvertTo-Json -Depth 10 | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host "Hostname: $($report.hostname)"
Write-Host "CPU: $(($report.cpu.models) -join '; ')"
Write-Host "CPU cores/logical: $($report.cpu.cores)/$($report.cpu.logical_processors)"
Write-Host "RAM total/available GiB: $($report.memory.total_gib)/$($report.memory.available_gib)"
foreach ($disk in $report.disks) {
    Write-Host "Disk $($disk.device): $($disk.free_gib) GiB free / $($disk.size_gib) GiB"
}
if ($report.nvidia.available) {
    foreach ($gpu in $report.nvidia.gpus) {
        Write-Host "GPU: $($gpu.name), VRAM $($gpu.memory_free_mib)/$($gpu.memory_total_mib) MiB free, driver $($gpu.driver_version)"
    }
    Write-Host "nvidia-smi CUDA: $($report.nvidia.reported_cuda_version)"
} else {
    Write-Host 'NVIDIA GPU via nvidia-smi: unavailable' -ForegroundColor Yellow
}
Write-Host "Python available: $($report.python.available)"
if ($report.python.torch_available -ne $null) {
    Write-Host "PyTorch: $($report.python.torch_available); CUDA available: $($report.python.cuda_available)"
}
Write-Host "Report: $jsonPath"
