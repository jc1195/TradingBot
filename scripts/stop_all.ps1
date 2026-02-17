$ErrorActionPreference = "Continue"

$stateFile = "runtime\processes.json"
if (-not (Test-Path $stateFile)) {
    Write-Host "No tracked process file found at $stateFile"
    return
}

$state = Get-Content $stateFile | ConvertFrom-Json
$pids = @($state.bot_pid, $state.ui_pid) | Where-Object { $_ }

foreach ($pid in $pids) {
    try {
        Stop-Process -Id $pid -Force
        Write-Host "Stopped process PID $pid"
    } catch {
        Write-Host "Process PID $pid was not running or could not be stopped"
    }
}

Remove-Item $stateFile -ErrorAction SilentlyContinue
Write-Host "Cleanup complete."
