# TradingBot Memory

## Working Agreements (Operator Preferences)
- Always refresh the dashboard after UI edits so changes are visible immediately.
- Keep explanations plain-English and focused on trust/visibility:
  - Is it running?
  - Why did it decide this?
  - Is dynamic discovery working?
- Keep paper-first safety defaults unless explicitly changed.

## Post-Update Checklist
Use this after any UI/code change in this repo:
1. Run targeted compile check for edited files.
2. If bot logic changed, run one cycle (`run_bot.py --once`) to validate.
3. If dashboard changed, refresh/open `http://localhost:8501`.
4. Record key outcome in **Progress Log** below.

## Progress Log

### 2026-02-16
#### Status Snapshot
- Health score: **70/100**
- Health status: **watch**
- Dynamic discovery: **enabled** (`dynamic_discovery_enabled: true`)
- Latest cycle status: **ok**

#### Why Health Is 70/100 (from latest health report)
Primary negative factors:
- Win rate below threshold: **0.0%** vs required **45.0%** on **10 closed trades**.
- High-confidence trades underperforming on average (`avg_pnl_high_conf` is negative).

Supporting context:
- Realized PnL is currently negative (`-2.310097`).
- Open positions: **5** with open notional near current paper capital (`497.708123`).
- Council activity is healthy in volume (**30 recent decisions**, avg confidence **0.74**), but output quality metrics are still weak.

#### Latest Recommendations (from health report)
- Tighten confidence threshold or reduce max symbols per cycle.
- Recalibrate council prompts or confidence weighting.

#### Dynamic Discovery Evidence
- `trend_scan_mode` metadata includes `include_trending: true`.
- Latest cycle report shows bullish picks discovered (10) and per-symbol selection reasons.

---

## Notes Template (append each session)
### YYYY-MM-DD
- Health: `score/status`
- Dynamic discovery: `on/off`
- Runtime health: `ok/check`
- Key issue(s):
- Change(s) made:
- Verification performed:
- Next action:

### 2026-02-16T19:32:17.816795+00:00 - Retraining Session
- Realized PnL snapshot: $-44.42
- Win rate snapshot: 15.1% over 126 trades
- Biggest losing symbols: RIVN(-15.34), ZIM(-7.72), XRP(-6.73), HBAR(-6.40), RIG(-2.04)
- Best calibration: phase=daily, TP=5.0%, SL=2.0%, gate=10, conf=65%, PnL=+0.00, WR=0.0%, Sharpe=+0.00, trades=0
- Applied controls: min_conf=0.65, risk/trade=5.00%, max_positions=5, daytrade_min_score=50, min_volume_ratio=1.00
- Learning rule: Avoid symbols repeatedly showing negative expectancy until new backtest evidence improves.

### 2026-02-16T19:33:20.105503+00:00 - Retraining Session
- Realized PnL snapshot: $-44.42
- Win rate snapshot: 15.1% over 126 trades
- Biggest losing symbols: RIVN(-15.34), ZIM(-7.72), XRP(-6.73), HBAR(-6.40), RIG(-2.04)
- Best calibration: phase=intraday, TP=3.0%, SL=1.0%, gate=10, conf=40%, PnL=-3.03, WR=40.9%, Sharpe=-0.08, trades=171
- Applied controls: min_conf=0.65, risk/trade=5.00%, max_positions=5, daytrade_min_score=50, min_volume_ratio=1.00
- Learning rule: Avoid symbols repeatedly showing negative expectancy until new backtest evidence improves.

### 2026-02-16T19:34:31.234281+00:00 - Retraining Session
- Realized PnL snapshot: $-44.42
- Win rate snapshot: 15.1% over 126 trades
- Biggest losing symbols: RIVN(-15.34), ZIM(-7.72), XRP(-6.73), HBAR(-6.40), RIG(-2.04)
- Best calibration: phase=daily, TP=8.0%, SL=5.0%, gate=10, conf=45%, PnL=+1.66, WR=46.9%, Sharpe=+0.05, trades=32
- Applied controls: min_conf=0.65, risk/trade=5.00%, max_positions=5, daytrade_min_score=50, min_volume_ratio=1.00
- Learning rule: Avoid symbols repeatedly showing negative expectancy until new backtest evidence improves.

### 2026-02-16T19:43:56.953798+00:00 - Retraining Session
- Realized PnL snapshot: $-44.96
- Win rate snapshot: 15.0% over 127 trades
- Biggest losing symbols: RIVN(-15.34), ZIM(-7.72), XRP(-6.73), HBAR(-6.40), RIG(-2.59)
- Best calibration: phase=daily, TP=8.0%, SL=2.0%, gate=10, conf=45%, PnL=+1.50, WR=38.2%, Sharpe=+0.01, trades=34
- Applied controls: min_conf=0.65, risk/trade=5.00%, max_positions=5, daytrade_min_score=50, min_volume_ratio=1.00
- Learning rule: Avoid symbols repeatedly showing negative expectancy until new backtest evidence improves.
