$ErrorActionPreference = 'Stop'
$cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
$cargoPath = if ($cargo) { $cargo.Source } else { Join-Path $env:USERPROFILE '.cargo\bin\cargo.exe' }
if (-not (Test-Path -LiteralPath $cargoPath)) { throw 'Install Rust with the MSVC toolchain first.' }
if (-not $env:CUDA_PATH) {
    $toolkits = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $toolkits) {
        $available = Get-ChildItem -LiteralPath $toolkits -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($available) { $env:CUDA_PATH = $available.FullName }
    }
}
$features = if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) { 'cuda,opencl' } else { 'opencl' }
Push-Location (Join-Path $PSScriptRoot 'source')
try {
    & $cargoPath build -p ntsc-rs-ae-plugin --release --features $features
    if ($LASTEXITCODE -ne 0) { throw 'NTSC build failed.' }
    Copy-Item -LiteralPath 'target\release\ntsc_rs_ae_plugin.dll' -Destination (Join-Path $PSScriptRoot 'ntsc-rs-ae-cuda.aex') -Force
} finally { Pop-Location }
Write-Output "Built NTSC plugin with $features. The existing filename is retained for installer compatibility."
