$ErrorActionPreference = 'Stop'

$RepoPath = Join-Path $HOME 'Chemistry'
$BridgeScript = Join-Path $RepoPath 'bridge\sync.ps1'
$StartupDir = [Environment]::GetFolderPath('Startup')
$Launcher = Join-Path $StartupDir 'ChemistryBridge.cmd'

if (-not (Test-Path $BridgeScript)) {
    throw "Bridge script not found: $BridgeScript"
}

$cmd = @(
    '@echo off'
    "start `"ChemistryBridge`" /min powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$BridgeScript`""
)
Set-Content -Path $Launcher -Value $cmd -Encoding ASCII

Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
    Where-Object { $_.CommandLine -like "*$BridgeScript*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
    '-NoLogo',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', "`"$BridgeScript`""
)

Write-Host "ChemistryBridge installed and started."
Write-Host "Repository: $RepoPath"
Write-Host "Startup launcher: $Launcher"
Write-Host "Log: $env:LOCALAPPDATA\ChemistryBridge\bridge.log"
