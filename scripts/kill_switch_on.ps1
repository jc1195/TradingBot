$ErrorActionPreference = "Stop"

if (-not (Test-Path "runtime")) {
    New-Item -ItemType Directory -Path "runtime" | Out-Null
}

$path = "runtime\kill_switch.flag"
"ACTIVE $(Get-Date -Format o)" | Set-Content -Path $path -Encoding UTF8
Write-Host "Kill switch enabled at $path"
