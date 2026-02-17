"""Backtest runner — CLI entry point.

Usage:
    python -m src.trading_bot.run_backtest                # all watchlist, 6mo daily
    python -m src.trading_bot.run_backtest --symbols DOGE  # DOGE only
    python -m src.trading_bot.run_backtest --kinds crypto   # crypto only
    python -m src.trading_bot.run_backtest --period 1y --interval 1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_bot.backtester import Backtester, BacktestConfig
from src.trading_bot.db import initialize_database
from src.trading_bot.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a historical backtest")
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to test (e.g. DOGE BTC AMC)")
    parser.add_argument("--kinds", nargs="*", default=None, help="Asset kinds: crypto, stock, or both")
    parser.add_argument("--period", default="6mo", help="yfinance period: 1mo, 3mo, 6mo, 1y, 2y (default: 6mo)")
    parser.add_argument("--interval", default="1d", help="Candle interval: 1h, 1d, 1wk (default: 1d)")
    parser.add_argument("--capital", type=float, default=500.0, help="Starting capital (default: 500)")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback window in candles (default: 24)")
    parser.add_argument("--hold", type=int, default=6, help="Min hold candles before closing (default: 6)")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Min AI confidence to trade (default: 0.55)")
    args = parser.parse_args()

    initialize_database()

    config = BacktestConfig(
        starting_capital=args.capital,
        period=args.period,
        interval=args.interval,
        kinds=args.kinds,
        symbols=[s.upper() for s in args.symbols] if args.symbols else None,
        lookback_window=args.lookback,
        hold_candles=args.hold,
        min_confidence=args.min_confidence,
    )

    print(f"\n{'='*60}")
    print(f"  BACKTEST  —  run_id: {config.run_id}")
    print(f"  Capital: ${config.starting_capital:.2f}")
    print(f"  Period: {config.period}  Interval: {config.interval}")
    print(f"  Kinds: {config.kinds or 'all'}  Symbols: {config.symbols or 'all watchlist'}")
    print(f"{'='*60}\n")

    bt = Backtester(config)
    result = bt.run()

    print(f"\n{'='*60}")
    print(f"  RESULTS  —  run_id: {result.run_id}")
    print(f"  Starting Capital: ${result.config.starting_capital:.2f}")
    print(f"  Final Equity:     ${result.final_equity:.2f}")
    print(f"  Total PnL:        ${result.total_pnl:.2f}")
    print(f"  Win/Loss:         {result.win_count} / {result.loss_count}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.1%}")
    print(f"  Sharpe Estimate:  {result.sharpe_estimate:.4f}")
    print(f"  AI Calls:         {result.ai_calls}")
    print(f"  Candles Processed:{result.candles_processed}")
    print(f"  Duration:         {result.duration_seconds:.1f}s")
    print(f"  Symbols Tested:   {', '.join(result.symbols_tested)}")
    if result.errors:
        print(f"  Errors:           {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"    - {e}")
    print(f"{'='*60}\n")

    if result.trades:
        print("  Top trades by PnL:")
        sorted_trades = sorted(result.trades, key=lambda t: t.net_pnl, reverse=True)
        for t in sorted_trades[:10]:
            print(
                f"    {t.symbol:>6s} ({t.kind})  entry={t.entry_price:.4f}  "
                f"exit={t.exit_price:.4f}  pnl=${t.net_pnl:+.4f}  "
                f"conf={t.confidence:.2f}"
            )
        print()

    return


if __name__ == "__main__":
    main()
