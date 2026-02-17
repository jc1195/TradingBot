param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    throw "Virtual environment .venv not found. Run setup first."
}

$listen = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listen) {
    try {
        Stop-Process -Id $listen.OwningProcess -Force
        Start-Sleep -Milliseconds 300
    } catch {
        Write-Host "Could not stop existing process on port 8501; continuing."
    }
}

if ($Foreground) {
    & .\.venv\Scripts\python.exe ui/run_dashboard.py
    exit $LASTEXITCODE
}

$pythonw = ".\.venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    $pythonw = ".\.venv\Scripts\python.exe"
}

$proc = Start-Process -FilePath $pythonw -ArgumentList "ui/run_dashboard.py" -WorkingDirectory (Resolve-Path ".").Path -WindowStyle Hidden -PassThru

if (-not (Test-Path "runtime")) {
    New-Item -ItemType Directory -Path "runtime" | Out-Null
}

$proc.Id | Set-Content -Path "runtime\dashboard.pid" -Encoding ascii
Write-Host "Dashboard started in background on http://localhost:8501 (PID $($proc.Id))"
