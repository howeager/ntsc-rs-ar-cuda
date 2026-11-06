$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $PSCommandPath + '"')
    )
    exit
}

$plugin = Join-Path $PSScriptRoot "ntsc-rs-ae-cuda.aex"
if (-not (Test-Path -LiteralPath $plugin -PathType Leaf)) {
    throw "ntsc-rs-ae-cuda.aex was not found beside this installer."
}

$commonFiles = ${env:CommonProgramFiles}
if (-not $commonFiles) { throw "Windows CommonProgramFiles is unavailable." }
$targetRoot = Join-Path $commonFiles "Adobe\Plug-ins\7.0\MediaCore"
$target = Join-Path $targetRoot "ntsc-rs-ae-cuda.aex"

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
Copy-Item -LiteralPath $plugin -Destination $target -Force
Unblock-File -LiteralPath $target

Write-Host "ntsc-rs installed successfully." -ForegroundColor Green
Write-Host "Installed to: $target"
Write-Host "Premiere Pro can be installed before or after this step."
Read-Host "Press Enter to close"
