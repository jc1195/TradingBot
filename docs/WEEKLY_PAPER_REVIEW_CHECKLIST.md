# Weekly Paper Trading Review Checklist

Use this at the end of each paper-trading week before changing settings.

## 1) Snapshot the week
- Confirm bot stayed in paper mode: `BOT_MODE=paper`, `LIVE_MODE_UNLOCK=false`.
- Record date range (Monday-Sunday) and total closed trades.
- Record realized PnL for the week.
- Record win rate for the week.
- Record max drawdown observed this week.

## 2) Compare against prior week
- Week-over-week change in realized PnL.
- Week-over-week change in win rate.
- Week-over-week change in max drawdown.
- Note if trade count changed significantly (more than +/-25%).

## 3) Diagnose by symbol and confidence
- Identify top 2 symbols by PnL and bottom 2 symbols by PnL.
- Check confidence calibration buckets (high confidence should outperform low confidence).
- If low-confidence trades underperform, raise `MIN_CONFIDENCE_TO_TRADE` slightly.
- If drawdown is elevated, reduce `MAX_RISK_PER_TRADE_PCT` or use `GOAL_PACING_PRESET=conservative`.

## 4) Decide next-week settings (small changes only)
- Change at most 1-2 parameters for the next week.
- Keep changes small (5-15% adjustments).
- Write the exact before/after values and rationale.
- Do not switch to live mode during this experiment week.

## 5) Log decisions
- Add summary to `EXECUTION_PLAN.md` session log.
- Save key metrics and decisions in your notes.
- Mark whether next week is:
  - `hold` (no changes)
  - `tune` (small setting changes)
  - `pause` (if quality alert/drawdown alert persists)
