$ErrorActionPreference = 'Stop'

$outputRoot = $env:CHEMISTRY_REQUEST_OUTPUT
if (-not $outputRoot) {
    throw 'CHEMISTRY_REQUEST_OUTPUT is not set.'
}
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $row = & nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $row) {
        $parts = $row -split ',' | ForEach-Object { $_.Trim() }
        $gpu = [ordered]@{ name = $parts[0]; memory_total_mib = [int]$parts[1]; memory_free_mib = [int]$parts[2]; driver = $parts[3] }
    }
}

$summary = [ordered]@{
    request_id = $env:CHEMISTRY_REQUEST_ID
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    hostname = $env:COMPUTERNAME
    cpu = $cpu.Name.Trim()
    logical_processors = [Environment]::ProcessorCount
    memory_total_gib = [math]::Round(($os.TotalVisibleMemorySize * 1KB) / 1GB, 2)
    memory_available_gib = [math]::Round(($os.FreePhysicalMemory * 1KB) / 1GB, 2)
    gpu = $gpu
}

$path = Join-Path $outputRoot 'system-summary.json'
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
$summary | ConvertTo-Json -Depth 6
Write-Host "Summary written to $path"
