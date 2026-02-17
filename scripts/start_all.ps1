param(
    [switch]$InstallDeps,
    [switch]$StartDashboard,
    [switch]$SplitTerminals
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

if ($InstallDeps) {
    pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from template. Fill credentials/settings before live use."
}

if (-not (Test-Path "runtime")) {
    New-Item -ItemType Directory -Path "runtime" | Out-Null
}

$botProc = $null
$uiProc = $null

if ($SplitTerminals) {
    $botProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", ". .\.venv\Scripts\Activate.ps1; python -m src.trading_bot.main" -PassThru
    if ($StartDashboard) {
        $uiProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", ". .\.venv\Scripts\Activate.ps1; streamlit run ui/dashboard.py" -PassThru
    }
} else {
    if ($StartDashboard) {
        $uiProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", ". .\.venv\Scripts\Activate.ps1; streamlit run ui/dashboard.py" -PassThru
    }
}

if ($botProc -or $uiProc) {
    try {
        if ($botProc) { $botProc.PriorityClass = "BelowNormal" }
        if ($uiProc) { $uiProc.PriorityClass = "BelowNormal" }
    } catch {
        Write-Host "Could not change process priority. Continuing with default priority."
    }
}

$state = @{
    started_at = (Get-Date).ToString("o")
    bot_pid = if ($botProc) { $botProc.Id } else { $null }
    ui_pid = if ($uiProc) { $uiProc.Id } else { $null }
}
$state | ConvertTo-Json | Set-Content -Path "runtime\processes.json" -Encoding UTF8

if ($SplitTerminals -and $StartDashboard) {
    Write-Host "Launched bot service and dashboard in separate terminals."
    Write-Host "Tracked process IDs in runtime\\processes.json"
} elseif ($SplitTerminals) {
    Write-Host "Launched bot service in a separate terminal."
} elseif ($StartDashboard) {
    Write-Host "Bot service ran in current terminal. Start dashboard in another terminal if needed: streamlit run ui/dashboard.py"
} else {
    Write-Host "Bot service ran in current terminal (low-resource default)."
}

if (-not $SplitTerminals) {
    python -m src.trading_bot.main
}
