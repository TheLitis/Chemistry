[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RequestPath,
    [string]$OutputDirectory = $(if ($env:RUNNER_TEMP) { Join-Path $env:RUNNER_TEMP 'chemistry-task' } else { Join-Path $env:TEMP 'chemistry-task' }),
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$StateRoot = Join-Path $env:ProgramData 'ChemistryRunner'
$MachineConfigPath = Join-Path $StateRoot 'machine.json'
$PythonExe = Join-Path $StateRoot 'python\python.exe'
$TasksRoot = Join-Path $RepoRoot 'tasks'

function Assert-Request {
    param($Request)

    if ($null -eq $Request) { throw 'Request JSON is empty.' }
    if ([int]$Request.version -ne 1) { throw "Unsupported request version: $($Request.version)" }
    if ([string]::IsNullOrWhiteSpace([string]$Request.id) -or [string]$Request.id -notmatch '^[A-Za-z0-9._-]{1,80}$') {
        throw 'Request id must match ^[A-Za-z0-9._-]{1,80}$.'
    }

    $allowed = @('health', 'tests', 'casmi26', 'script')
    if ($allowed -notcontains [string]$Request.task) {
        throw "Unsupported task '$($Request.task)'. Allowed: $($allowed -join ', ')."
    }

    if ([string]$Request.task -eq 'script') {
        if ([string]::IsNullOrWhiteSpace([string]$Request.script)) {
            throw 'script task requires the script field.'
        }
        $relative = ([string]$Request.script).Replace('/', '\')
        if ([System.IO.Path]::IsPathRooted($relative) -or $relative -match '(^|\\)\.\.(\\|$)') {
            throw 'Script path must be a relative path below tasks/ without .. segments.'
        }
        $candidate = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $relative))
        $tasksFull = [System.IO.Path]::GetFullPath($TasksRoot).TrimEnd('\') + '\'
        if (-not $candidate.StartsWith($tasksFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Script path must resolve below tasks/.'
        }
        $extension = [System.IO.Path]::GetExtension($candidate).ToLowerInvariant()
        if (@('.ps1', '.py') -notcontains $extension) {
            throw 'Only committed .ps1 or .py files below tasks/ may be executed.'
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Requested script does not exist: $relative"
        }
    }

    if ($null -ne $Request.arguments) {
        if ($Request.arguments -isnot [System.Array]) {
            throw 'arguments must be a JSON array of strings.'
        }
        foreach ($argument in $Request.arguments) {
            if ($argument -isnot [string]) { throw 'Every arguments entry must be a string.' }
        }
    }
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath |
            ForEach-Object { Write-Host $_ }
        $code = [int]$LASTEXITCODE
        return $code
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Ensure-Python {
    & (Join-Path $PSScriptRoot 'bootstrap-python.ps1')
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Chemistry Python is unavailable: $PythonExe" }
}

function Ensure-CudaTorch {
    Ensure-Python
    & (Join-Path $PSScriptRoot 'bootstrap-torch.ps1') -PythonExe $PythonExe
}

function Read-MachineConfig {
    if (-not (Test-Path -LiteralPath $MachineConfigPath)) {
        throw "Machine config is missing: $MachineConfigPath"
    }
    return Get-Content -LiteralPath $MachineConfigPath -Raw | ConvertFrom-Json
}

if (-not (Test-Path -LiteralPath $RequestPath -PathType Leaf)) {
    throw "Request file does not exist: $RequestPath"
}

$request = Get-Content -LiteralPath $RequestPath -Raw | ConvertFrom-Json
Assert-Request $request

if ($ValidateOnly) {
    Write-Host "Request '$($request.id)' is valid: task=$($request.task)"
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$resultPath = Join-Path $OutputDirectory 'request-result.json'
$started = (Get-Date).ToUniversalTime()
$status = 'failed'
$exitCode = 1
$errorText = $null

$env:CHEMISTRY_REQUEST_ID = [string]$request.id
$env:CHEMISTRY_REQUEST_OUTPUT = $OutputDirectory
$env:CHEMISTRY_STATE_ROOT = $StateRoot

try {
    if ([bool]$request.requireCuda) {
        Ensure-CudaTorch
    } else {
        Ensure-Python
    }

    switch ([string]$request.task) {
        'health' {
            $healthDir = Join-Path $OutputDirectory 'health'
            New-Item -ItemType Directory -Force -Path $healthDir | Out-Null
            $env:CHEMISTRY_PYTHON = $PythonExe
            & (Join-Path $PSScriptRoot 'diagnostics.ps1') -OutputDirectory $healthDir
            $args = @((Join-Path $PSScriptRoot 'compute_smoke.py'), '--output', (Join-Path $healthDir 'compute-smoke.json'), '--max-ram-gib', '8')
            if ([bool]$request.requireCuda) { $args += '--require-cuda' }
            $exitCode = Invoke-LoggedNative -Executable $PythonExe -Arguments $args -LogPath (Join-Path $OutputDirectory 'health.log')
        }
        'tests' {
            $requirements = Join-Path $RepoRoot 'work\casmi26\requirements.txt'
            $pipLog = Join-Path $OutputDirectory 'pip.log'
            if (Test-Path -LiteralPath $requirements) {
                $code = Invoke-LoggedNative -Executable $PythonExe -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirements, 'pytest') -LogPath $pipLog
            } else {
                $code = Invoke-LoggedNative -Executable $PythonExe -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', 'pytest') -LogPath $pipLog
            }
            if ($code -ne 0) { throw "Dependency installation failed with exit code $code." }
            $exitCode = Invoke-LoggedNative -Executable $PythonExe -Arguments @('-m', 'pytest', '-q', (Join-Path $RepoRoot 'work\casmi26\tests')) -LogPath (Join-Path $OutputDirectory 'tests.log')
        }
        'casmi26' {
            $config = Read-MachineConfig
            $dataRoot = Join-Path ([string]$config.dataRoot) 'casmi26'
            $testPath = Join-Path $dataRoot 'test'
            $trainPath = Join-Path $dataRoot 'train'
            $samplePath = Join-Path $dataRoot 'sample_submission.csv'
            foreach ($required in @($testPath, $trainPath, $samplePath)) {
                if (-not (Test-Path -LiteralPath $required)) { throw "CASMI26 local input is missing: $required" }
            }
            $requirements = Join-Path $RepoRoot 'work\casmi26\requirements.txt'
            $code = Invoke-LoggedNative -Executable $PythonExe -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirements) -LogPath (Join-Path $OutputDirectory 'pip.log')
            if ($code -ne 0) { throw "CASMI26 dependency installation failed with exit code $code." }
            $submission = Join-Path $OutputDirectory 'submission.csv'
            $exitCode = Invoke-LoggedNative -Executable $PythonExe -Arguments @((Join-Path $RepoRoot 'predict.py'), '--test', $testPath, '--train', $trainPath, '--sample-submission', $samplePath, '--output', $submission) -LogPath (Join-Path $OutputDirectory 'casmi26.log')
        }
        'script' {
            $relative = ([string]$request.script).Replace('/', '\')
            $scriptPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $relative))
            $arguments = if ($null -eq $request.arguments) { @() } else { @($request.arguments | ForEach-Object { [string]$_ }) }
            $extension = [System.IO.Path]::GetExtension($scriptPath).ToLowerInvariant()
            if ($extension -eq '.ps1') {
                $exitCode = Invoke-LoggedNative -Executable 'powershell.exe' -Arguments (@('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath) + $arguments) -LogPath (Join-Path $OutputDirectory 'script.log')
            } else {
                $exitCode = Invoke-LoggedNative -Executable $PythonExe -Arguments (@($scriptPath) + $arguments) -LogPath (Join-Path $OutputDirectory 'script.log')
            }
        }
    }

    if ($exitCode -ne 0) { throw "Task '$($request.task)' exited with code $exitCode." }
    $status = 'success'
} catch {
    $errorText = $_.Exception.Message
    if ($exitCode -eq 0) { $exitCode = 1 }
    throw
} finally {
    $finished = (Get-Date).ToUniversalTime()
    [ordered]@{
        version = 1
        request_id = [string]$request.id
        task = [string]$request.task
        status = $status
        exit_code = $exitCode
        error = $errorText
        started_utc = $started.ToString('o')
        finished_utc = $finished.ToString('o')
        duration_seconds = [math]::Round(($finished - $started).TotalSeconds, 3)
        runner = $env:RUNNER_NAME
        machine = $env:COMPUTERNAME
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding UTF8
}
