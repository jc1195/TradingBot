"""Focused council backtest — lightweight test with reduced output noise.

Runs the 3-agent council on a small set of stocks already known to perform
well in the backtester (NVDA, RIVN, GME) to validate council mode works.
"""

import sys
import os
import warnings

# Suppress all non-critical warnings (yfinance pandas noise)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Force UTF-8 output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import time
from src.trading_bot.backtester import Backtester, BacktestConfig


def main():
    print("=" * 60)
    print("COUNCIL BACKTEST — FOCUSED TEST")
    print("=" * 60)

    cfg = BacktestConfig(
        symbols=["RIVN", "GME", "NIO"],
        kinds=["stock"],
        period="6mo",              # shorter period = faster test
        interval="1d",
        starting_capital=500.0,
        goal_equity=1000.0,
        profit_target_pct=8.0,
        stop_loss_pct=3.0,
        trailing_stop_pct=2.5,
        signal_score_gate=10,      # lowered from 15 → 10 for more opportunities
        min_confidence=0.55,       # lowered from 0.65
        use_council=True,
        max_concurrent_positions=2,
        lookback_window=20,        # slightly smaller window
    )

    print(f"Council: ENABLED (3 agents)")
    print(f"Symbols: {cfg.symbols}")
    print(f"Period: {cfg.period}")
    print(f"Capital: ${cfg.starting_capital} -> Goal: ${cfg.goal_equity}")
    print(f"Gates: score>={cfg.signal_score_gate}, conf>={cfg.min_confidence}")
    print()

    bt = Backtester(cfg)
    t0 = time.time()
    result = bt.run()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final equity:  ${result.final_equity:.2f}")
    print(f"Total PnL:     ${result.total_pnl:.2f}")
    print(f"Trades:        {len(result.trades)} (W:{result.win_count} / L:{result.loss_count})")
    print(f"Win rate:      {result.win_rate:.1%}")
    print(f"Max drawdown:  {result.max_drawdown_pct:.2f}%")
    print(f"AI calls:      {result.ai_calls}")
    print(f"Time:          {elapsed:.1f}s")

    if result.trades:
        print("\nTrades:")
        for t in result.trades:
            tag = "WIN " if t.net_pnl > 0 else "LOSS"
            print(f"  {tag} {t.symbol:6s} ${t.entry_price:.2f}→${t.exit_price:.2f}  "
                  f"pnl=${t.net_pnl:+.2f}  held={t.hold_candles}c  conf={t.confidence:.0%}")
    else:
        print("\nNo trades executed — council may be too conservative or all gates filtered.")

    # Now test single-agent mode for comparison
    print("\n" + "=" * 60)
    print("SINGLE AGENT COMPARISON")
    print("=" * 60)

    cfg2 = BacktestConfig(
        symbols=["RIVN", "GME", "NIO"],
        kinds=["stock"],
        period="6mo",
        interval="1d",
        starting_capital=500.0,
        goal_equity=1000.0,
        profit_target_pct=8.0,
        stop_loss_pct=3.0,
        trailing_stop_pct=2.5,
        signal_score_gate=10,
        min_confidence=0.55,
        use_council=False,         # single agent
        max_concurrent_positions=2,
        lookback_window=20,
    )

    bt2 = Backtester(cfg2)
    t0 = time.time()
    result2 = bt2.run()
    elapsed2 = time.time() - t0

    print(f"\nFinal equity:  ${result2.final_equity:.2f}")
    print(f"Total PnL:     ${result2.total_pnl:.2f}")
    print(f"Trades:        {len(result2.trades)} (W:{result2.win_count} / L:{result2.loss_count})")
    print(f"Win rate:      {result2.win_rate:.1%}")
    print(f"AI calls:      {result2.ai_calls}")
    print(f"Time:          {elapsed2:.1f}s")

    if result2.trades:
        print("\nTrades:")
        for t in result2.trades:
            tag = "WIN " if t.net_pnl > 0 else "LOSS"
            print(f"  {tag} {t.symbol:6s} ${t.entry_price:.2f}→${t.exit_price:.2f}  "
                  f"pnl=${t.net_pnl:+.2f}  held={t.hold_candles}c  conf={t.confidence:.0%}")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Council:  PnL=${result.total_pnl:+.2f}  WR={result.win_rate:.1%}  "
          f"Trades={len(result.trades)}  AI={result.ai_calls}  Time={elapsed:.0f}s")
    print(f"  Single:   PnL=${result2.total_pnl:+.2f}  WR={result2.win_rate:.1%}  "
          f"Trades={len(result2.trades)}  AI={result2.ai_calls}  Time={elapsed2:.0f}s")
    better = "COUNCIL" if result.total_pnl > result2.total_pnl else "SINGLE"
    print(f"  Winner:   {better}")


if __name__ == "__main__":
    main()
