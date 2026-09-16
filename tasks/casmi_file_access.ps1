# Read-only access diagnostics, scoped to the explicitly supplied archive.
$ErrorActionPreference = 'Stop'
$report = [ordered]@{
    identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    runner = $env:RUNNER_NAME
    paths = @()
    services = @()
    permissions_changed = $false
}
try {
    $report.services = @(Get-CimInstance Win32_Service | Where-Object { $_.Name -like '*ChemistryPC*' } | Select-Object Name, StartName, State)
} catch { $report.service_query_error = $_.Exception.GetType().Name }
$paths = @(
    'C:\Users\loval\Downloads',
    'C:\Users\loval\Downloads\enveda-CASMI26-molecule-id-mass-spectra.zip',
    'C:\ProgramData\ChemistryRunner\data'
)
foreach ($path in $paths) {
    $item = [ordered]@{ path = $path }
    try {
        $info = Get-Item -LiteralPath $path -ErrorAction Stop
        $item.attributes = [string]$info.Attributes
        if (-not $info.PSIsContainer) { $item.bytes = $info.Length; $item.last_write_utc = $info.LastWriteTimeUtc.ToString('o') }
    } catch { $item.metadata_error = $_.Exception.GetType().Name }
    try {
        $acl = Get-Acl -LiteralPath $path -ErrorAction Stop
        $item.owner = $acl.Owner
        $item.rules = @($acl.Access | Select-Object IdentityReference, FileSystemRights, AccessControlType, IsInherited)
    } catch { $item.acl_error = $_.Exception.GetType().Name }
    if ($path.EndsWith('.zip')) {
        try {
            $handle = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
            try { $item.open_read = $true; $item.length = $handle.Length } finally { $handle.Dispose() }
        } catch {
            $item.open_read = $false
            $item.open_error = $_.Exception.GetType().Name
            $item.message = $_.Exception.Message
            $item.hresult = $_.Exception.HResult
            if ($_.Exception.InnerException) { $item.inner_hresult = $_.Exception.InnerException.HResult }
        }
    }
    $report.paths += $item
}
$text = $report | ConvertTo-Json -Depth 8
$text | Set-Content -LiteralPath (Join-Path $env:CHEMISTRY_REQUEST_OUTPUT 'file-access.json') -Encoding UTF8
Write-Host "FILE_ACCESS_BEGIN`n$text`nFILE_ACCESS_END"
$python = Join-Path $env:CHEMISTRY_STATE_ROOT 'envs\casmi26\python.exe'
$repo = Split-Path -Parent $PSScriptRoot
& $python (Join-Path $PSScriptRoot 'casmi_access_check.py')
exit $LASTEXITCODE
