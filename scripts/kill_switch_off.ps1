$ErrorActionPreference = "Continue"

$path = "runtime\kill_switch.flag"
if (Test-Path $path) {
    Remove-Item $path -Force
    Write-Host "Kill switch disabled"
} else {
    Write-Host "Kill switch file not present"
}
