# ── run_backtest.ps1 ─────────────────────────────────────────────
# Quick-launch wrapper for the backtesting engine.
#
# Usage examples:
#   .\scripts\run_backtest.ps1                          # full watchlist, 6mo daily
#   .\scripts\run_backtest.ps1 -Symbols DOGE            # DOGE only
#   .\scripts\run_backtest.ps1 -Symbols DOGE,BTC -Period 1y
#   .\scripts\run_backtest.ps1 -Kinds crypto            # crypto only
#   .\scripts\run_backtest.ps1 -Kinds stock             # stocks only
# ─────────────────────────────────────────────────────────────────

param(
    [string[]]$Symbols,
    [string[]]$Kinds,
    [string]$Period = "6mo",
    [string]$Interval = "1d",
    [double]$Capital = 500,
    [int]$Lookback = 24,
    [int]$Hold = 6,
    [double]$MinConfidence = 0.55
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $project

try {
    $pyArgs = @("-m", "src.trading_bot.run_backtest")
    $pyArgs += "--period", $Period
    $pyArgs += "--interval", $Interval
    $pyArgs += "--capital", $Capital
    $pyArgs += "--lookback", $Lookback
    $pyArgs += "--hold", $Hold
    $pyArgs += "--min-confidence", $MinConfidence

    if ($Symbols) {
        $pyArgs += "--symbols"
        $pyArgs += $Symbols
    }
    if ($Kinds) {
        $pyArgs += "--kinds"
        $pyArgs += $Kinds
    }

    Write-Host "`n=== Starting Backtest ===" -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe @pyArgs
}
finally {
    Pop-Location
}
