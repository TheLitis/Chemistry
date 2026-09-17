[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$token = $env:AIMO_RUNNER_TOKEN
$sourceRoot = 'C:\actions-runner\ChemistryPC'
$runnerRoot = 'C:\actions-runner\AIMOPC'
$repositoryUrl = 'https://github.com/TheLitis/aimo-interpretability'

if ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne 'S-1-5-18') {
    throw 'AIMOPC bootstrap requires LocalSystem execution.'
}
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Existing runner package is unavailable: $sourceRoot"
}

try {
    if (-not (Test-Path -LiteralPath $runnerRoot)) {
        if ([string]::IsNullOrWhiteSpace($token)) { throw 'AIMO_RUNNER_TOKEN is unavailable.' }
        New-Item -ItemType Directory -Force -Path $runnerRoot | Out-Null
        foreach ($directory in @('bin', 'externals')) {
            Copy-Item -LiteralPath (Join-Path $sourceRoot $directory) -Destination $runnerRoot -Recurse -Force
        }
        foreach ($file in @('config.cmd', 'run.cmd', 'run-helper.cmd.template', 'run-helper.sh.template')) {
            Copy-Item -LiteralPath (Join-Path $sourceRoot $file) -Destination $runnerRoot -Force
        }
        Push-Location $runnerRoot
        try {
            & .\config.cmd --unattended --url $repositoryUrl --token $token --name AIMOPC --labels AIMOPC --work _work --replace --runasservice
            if ($LASTEXITCODE -ne 0) { throw "Runner configuration failed with exit code $LASTEXITCODE." }
        } finally {
            Pop-Location
        }
    }
    $serviceName = (Get-Content -LiteralPath (Join-Path $runnerRoot '.service') -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($serviceName)) { throw 'Runner service name was not created.' }
    & sc.exe config $serviceName obj= LocalSystem password= '""'
    if ($LASTEXITCODE -ne 0) { throw "Unable to configure $serviceName as LocalSystem." }
    & sc.exe config $serviceName 'start= auto'
    if ($LASTEXITCODE -ne 0) { throw "Unable to configure automatic startup for $serviceName." }
    Restart-Service -Name $serviceName
    $service = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
    if ($service.State -ne 'Running' -or $service.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
        throw "AIMOPC service verification failed: state=$($service.State), account=$($service.StartName)"
    }
    Write-Host "AIMOPC installed: service=$serviceName account=$($service.StartName)" -ForegroundColor Green
} finally {
    Remove-Item Env:AIMO_RUNNER_TOKEN -ErrorAction SilentlyContinue
}
