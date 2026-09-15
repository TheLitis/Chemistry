$ErrorActionPreference = 'SilentlyContinue'

$RepoPath = Join-Path $HOME 'Chemistry'
$BridgeScript = Join-Path $RepoPath 'bridge\sync.ps1'
$StartupDir = [Environment]::GetFolderPath('Startup')
$Launcher = Join-Path $StartupDir 'ChemistryBridge.cmd'

Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
    Where-Object { $_.CommandLine -like "*$BridgeScript*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Remove-Item -Force $Launcher
Write-Host 'ChemistryBridge removed from startup and stopped.'
