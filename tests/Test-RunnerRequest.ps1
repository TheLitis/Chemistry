$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dispatcher = Join-Path $repoRoot 'runner\run-request.ps1'
$tempRoot = Join-Path $env:TEMP ("ChemistryRunnerRequestTests-{0}" -f [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

function Write-Request([string]$Name, $Object) {
    $path = Join-Path $tempRoot $Name
    $Object | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
    return $path
}

try {
    $valid = Write-Request 'valid.json' ([ordered]@{ version = 1; id = 'ci-health'; task = 'health'; requireCuda = $false })
    & $dispatcher -RequestPath $valid -ValidateOnly
    if ($LASTEXITCODE -ne 0) { throw 'Valid health request did not validate.' }

    $invalidTask = Write-Request 'invalid-task.json' ([ordered]@{ version = 1; id = 'ci-invalid'; task = 'shell' })
    $caught = $false
    try { & $dispatcher -RequestPath $invalidTask -ValidateOnly } catch { $caught = $_.Exception.Message -like '*Unsupported task*' }
    if (-not $caught) { throw 'Unsupported task was not rejected.' }

    $traversal = Write-Request 'traversal.json' ([ordered]@{ version = 1; id = 'ci-traversal'; task = 'script'; script = 'tasks\..\bridge\sync.ps1' })
    $caught = $false
    try { & $dispatcher -RequestPath $traversal -ValidateOnly } catch { $caught = $_.Exception.Message -like '*without .. segments*' }
    if (-not $caught) { throw 'Path traversal request was not rejected.' }

    $absolute = Write-Request 'absolute.json' ([ordered]@{ version = 1; id = 'ci-absolute'; task = 'script'; script = 'C:\Windows\System32\whoami.exe' })
    $caught = $false
    try { & $dispatcher -RequestPath $absolute -ValidateOnly } catch { $caught = $_.Exception.Message -like '*relative path below tasks*' }
    if (-not $caught) { throw 'Absolute script path was not rejected.' }

    Write-Host 'Runner request validation tests passed.'
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
