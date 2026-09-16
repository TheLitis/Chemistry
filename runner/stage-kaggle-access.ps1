# Run locally as the signed-in Kaggle user. Never run this to inspect another profile.
[CmdletBinding()]
param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
if ($ValidateOnly) {
    Write-Host 'Stages one existing Kaggle API access token for ChemistryPC. No browser access, rule acceptance, or submission.'
    exit 0
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if ($identity.User.Value -in @('S-1-5-18', 'S-1-5-19', 'S-1-5-20')) {
    throw 'Run this once under your signed-in Windows user, not the runner service.'
}
$token = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'Process')
if ([string]::IsNullOrWhiteSpace($token)) {
    $token = [Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN', 'User')
}
if ([string]::IsNullOrWhiteSpace($token)) {
    $roots = @()
    if ($env:KAGGLE_CONFIG_DIR) { $roots += $env:KAGGLE_CONFIG_DIR }
    $roots += (Join-Path $HOME '.kaggle')
    foreach ($root in $roots) {
        $file = Join-Path $root 'access_token'
        if (Test-Path -LiteralPath $file -PathType Leaf) {
            $token = [IO.File]::ReadAllText($file).Trim()
            if (-not [string]::IsNullOrWhiteSpace($token)) { break }
        }
    }
}
if ([string]::IsNullOrWhiteSpace($token)) {
    $secure = Read-Host 'Paste the Kaggle API access token here (hidden; never send it in chat)' -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr); $secure.Dispose() }
}
$token = $token.Trim()
if ($token.Length -lt 20 -or $token -match '\s') { throw 'Expected one nonempty API access token.' }
$destination = Join-Path $env:ProgramData 'ChemistryRunner\kaggle'
if (Test-Path -LiteralPath $destination) {
    $item = Get-Item -LiteralPath $destination -Force
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The dedicated Kaggle destination must be an ordinary directory.'
    }
} else { New-Item -ItemType Directory -Path $destination | Out-Null }
# Protect the dedicated provider directory BEFORE writing any credential.
$acl = New-Object Security.AccessControl.DirectorySecurity
$acl.SetAccessRuleProtection($true, $false)
$inherit = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
$none = [Security.AccessControl.PropagationFlags]::None
$allow = [Security.AccessControl.AccessControlType]::Allow
foreach ($sid in @($identity.User.Value, 'S-1-5-18', 'S-1-5-32-544') | Select-Object -Unique) {
    $principal = New-Object Security.Principal.SecurityIdentifier($sid)
    $rule = New-Object Security.AccessControl.FileSystemAccessRule($principal, 'FullControl', $inherit, $none, $allow)
    $acl.AddAccessRule($rule)
}
$service = New-Object Security.Principal.SecurityIdentifier('S-1-5-20')
$acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($service, 'ReadAndExecute', $inherit, $none, $allow)))
Set-Acl -LiteralPath $destination -AclObject $acl
$target = Join-Path $destination 'access_token'
if (Test-Path -LiteralPath $target) {
    if ((Get-Item -LiteralPath $target -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Refusing a credential-file reparse point.'
    }
    # Reset only this provider token file to the protected directory ACL.
    $fileAcl = Get-Acl -LiteralPath $target
    foreach ($rule in @($fileAcl.Access | Where-Object { -not $_.IsInherited })) {
        $fileAcl.RemoveAccessRuleSpecific($rule)
    }
    $fileAcl.SetAccessRuleProtection($false, $false)
    Set-Acl -LiteralPath $target -AclObject $fileAcl
}
[IO.File]::WriteAllText($target, $token, (New-Object Text.UTF8Encoding($false)))
$token = $null
Write-Host 'Kaggle access token staged in the dedicated ChemistryPC provider directory. Token value was not printed.'
Write-Host 'Only this directory ACL was changed; your profile permissions and runner identity were not changed.'
