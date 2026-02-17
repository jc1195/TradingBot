param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $OutputPath -or [string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $root "runtime\state_export_$timestamp.zip"
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$includeCandidates = @(
    "data\trading_bot.sqlite3",
    "runtime\special_circumstances.json",
    "runtime\shadow_model_status.json",
    "runtime\shadow_portfolio.json",
    "runtime\shadow_predictions.json",
    "runtime\training_recommendations.json"
)

$existing = @()
foreach ($relative in $includeCandidates) {
    $full = Join-Path $root $relative
    if (Test-Path $full) {
        $existing += $relative
    }
}

if ($existing.Count -eq 0) {
    throw "No exportable state files found. Expected at least one of: $($includeCandidates -join ', ')"
}

$tempRoot = Join-Path $root "runtime\_state_export_$timestamp"
if (Test-Path $tempRoot) {
    Remove-Item $tempRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

try {
    foreach ($relative in $existing) {
        $src = Join-Path $root $relative
        $dst = Join-Path $tempRoot $relative
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
        Copy-Item -Path $src -Destination $dst -Force
    }

    $manifest = [ordered]@{
        exported_at = (Get-Date).ToString("o")
        machine = $env:COMPUTERNAME
        files = $existing
        note = "Import with scripts/import_state.ps1 while bot/dashboard are stopped."
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $tempRoot "manifest.json") -Encoding UTF8

    if (Test-Path $OutputPath) {
        Remove-Item $OutputPath -Force
    }

    Compress-Archive -Path (Join-Path $tempRoot "*") -DestinationPath $OutputPath -Force

    Write-Host "State export created: $OutputPath"
    Write-Host "Included files:"
    $existing | ForEach-Object { Write-Host "  - $_" }
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item $tempRoot -Recurse -Force
    }
}
