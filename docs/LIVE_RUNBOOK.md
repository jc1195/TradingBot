# Live Rollout Runbook

## Purpose
Operational checklist for controlled transition from paper mode to live mode.

## Preconditions
- Paper performance has been reviewed for the selected window.
- Risk policy limits are confirmed in `config/risk_policy.yaml`.
- You have emergency stop access to this machine.

## 1) Configure live mode (do not start yet)
1. Open `.env`
2. Set:
   - `BOT_MODE=live`
   - `LIVE_MODE_UNLOCK=true`
3. Confirm live caps are conservative:
   - `LIVE_MAX_ORDER_NOTIONAL_USD`
   - `LIVE_MAX_DAILY_NOTIONAL_USD`
   - `LIVE_MAX_ORDERS_PER_DAY`
4. Optional but recommended:
   - `ALERT_WEBHOOK_URL=<your webhook>`

## 2) Enable live trading in policy
In `config/risk_policy.yaml`:
- `trade_guards.allow_live_orders: true`

## 3) Verify kill switch is OFF
- Ensure `runtime/kill_switch.flag` does not exist
- If present:
  - run `./scripts/kill_switch_off.ps1`

## 4) Run preflight checklist
Run:

```powershell
./scripts/live_preflight.ps1
```

- If it fails, fix each failing check and rerun.
- Do not proceed until it passes.

## 5) Start service
- Low resource:

```powershell
./scripts/start_all.ps1
```

- Optional split with dashboard:

```powershell
./scripts/start_all.ps1 -SplitTerminals -StartDashboard
```

## 6) Monitor in first hour
- Watch `Alerts` panel in dashboard for:
  - `order_blocked_live_cap`
  - `drawdown_alert`
  - `quality_alert`
  - `pipeline_error`
- Verify proposed live orders remain within configured caps.

## 7) Emergency stop
- Immediate external kill switch:

```powershell
./scripts/kill_switch_on.ps1
```

- Optional process cleanup:

```powershell
./scripts/stop_all.ps1
```

## 8) Return to paper mode
1. Set `.env`:
   - `BOT_MODE=paper`
   - `LIVE_MODE_UNLOCK=false`
2. Set `trade_guards.allow_live_orders: false`
3. Restart service
