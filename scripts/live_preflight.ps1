$ErrorActionPreference = "Continue"

$script:results = @()

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail
    )
    $script:results += [PSCustomObject]@{
        Check = $Name
        Passed = $Passed
        Detail = $Detail
    }
}

function Get-EnvMap {
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path $Path)) {
        return $map
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) { return }
        if ($line.StartsWith("#")) { return }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        $map[$key] = $value
    }
    return $map
}

function Is-Truthy {
    param([string]$Value)
    if (-not $Value) { return $false }
    return @("true", "1", "yes", "on") -contains $Value.ToLower()
}

$envPath = ".env"
$riskPath = "config\risk_policy.yaml"
$killSwitchPath = "runtime\kill_switch.flag"

$envMap = Get-EnvMap -Path $envPath

Add-Check "Env file exists" (Test-Path $envPath) ("Path: {0}" -f $envPath)
Add-Check "Risk policy exists" (Test-Path $riskPath) ("Path: {0}" -f $riskPath)

$botMode = if ($envMap.ContainsKey("BOT_MODE")) { $envMap["BOT_MODE"] } else { "" }
$unlock = if ($envMap.ContainsKey("LIVE_MODE_UNLOCK")) { $envMap["LIVE_MODE_UNLOCK"] } else { "" }

Add-Check "BOT_MODE is live" ($botMode -eq "live") ("BOT_MODE={0}" -f $botMode)
Add-Check "LIVE_MODE_UNLOCK is true" (Is-Truthy $unlock) ("LIVE_MODE_UNLOCK={0}" -f $unlock)

$username = if ($envMap.ContainsKey("ROBINHOOD_USERNAME")) { $envMap["ROBINHOOD_USERNAME"] } else { "" }
$password = if ($envMap.ContainsKey("ROBINHOOD_PASSWORD")) { $envMap["ROBINHOOD_PASSWORD"] } else { "" }
Add-Check "Robinhood credentials set" (-not [string]::IsNullOrWhiteSpace($username) -and -not [string]::IsNullOrWhiteSpace($password)) "ROBINHOOD_USERNAME/ROBINHOOD_PASSWORD"

$allowLive = $false
if (Test-Path $riskPath) {
    $riskText = Get-Content $riskPath -Raw
    $allowLive = $riskText -match "allow_live_orders:\s*true"
}
Add-Check "Risk policy allows live orders" $allowLive "trade_guards.allow_live_orders"

$killSwitchMissing = -not (Test-Path $killSwitchPath)
Add-Check "External kill switch is OFF" $killSwitchMissing ("Path: {0}" -f $killSwitchPath)

$orderCap = if ($envMap.ContainsKey("LIVE_MAX_ORDER_NOTIONAL_USD")) { [double]$envMap["LIVE_MAX_ORDER_NOTIONAL_USD"] } else { 0 }
$dayCap = if ($envMap.ContainsKey("LIVE_MAX_DAILY_NOTIONAL_USD")) { [double]$envMap["LIVE_MAX_DAILY_NOTIONAL_USD"] } else { 0 }
$ordersPerDay = if ($envMap.ContainsKey("LIVE_MAX_ORDERS_PER_DAY")) { [int]$envMap["LIVE_MAX_ORDERS_PER_DAY"] } else { 0 }

Add-Check "Live per-order cap configured" ($orderCap -gt 0) ("LIVE_MAX_ORDER_NOTIONAL_USD={0}" -f $orderCap)
Add-Check "Live daily cap configured" ($dayCap -gt 0) ("LIVE_MAX_DAILY_NOTIONAL_USD={0}" -f $dayCap)
Add-Check "Live daily order cap configured" ($ordersPerDay -gt 0) ("LIVE_MAX_ORDERS_PER_DAY={0}" -f $ordersPerDay)

$alertUrl = if ($envMap.ContainsKey("ALERT_WEBHOOK_URL")) { $envMap["ALERT_WEBHOOK_URL"] } else { "" }
Add-Check "Alert webhook configured" (-not [string]::IsNullOrWhiteSpace($alertUrl)) "ALERT_WEBHOOK_URL"

Write-Host "\n=== Live Preflight Checklist ==="
$script:results | Format-Table -AutoSize | Out-String | Write-Host

$failed = @($script:results | Where-Object { -not $_.Passed })
if ($failed.Count -gt 0) {
    Write-Host ("Preflight FAILED: {0} check(s) did not pass." -f $failed.Count) -ForegroundColor Red
    exit 1
}

Write-Host "Preflight PASSED. Live enablement gates are satisfied." -ForegroundColor Green
exit 0
