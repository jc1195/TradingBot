"""Smart paper trading runner with trend-aware asset selection.

This script:
 1. Runs a trend scan to find bullish assets
 2. Feeds the best picks into the trading engine  
 3. Repeats on a configurable interval (default: every 30 minutes)
 4. Tracks performance and adapts position sizing

Usage:
    python scripts/run_paper_trading.py              # single cycle
    python scripts/run_paper_trading.py --continuous  # loop every 30 min
    python scripts/run_paper_trading.py --scan-only   # just show scan results
"""

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, ".")

from src.trading_bot.engine import TradingEngine
from src.trading_bot.risk import load_risk_policy
from src.trading_bot.trend_detector import (
    scan_for_bullish_opportunities,
    print_trend_report,
)
from src.trading_bot.settings import settings


def scan_and_report() -> list:
    """Run a trend scan and print results. Returns bullish picks."""
    print("=" * 70)
    print("TREND SCANNER -- Searching for bullish opportunities")
    print("=" * 70)

    t0 = time.time()
    bullish = scan_for_bullish_opportunities(
        include_penny=True,
        include_micro_crypto=True,
        include_trending=True,
        min_score=settings.scan_min_score,
        top_n=max(settings.scan_top_n, 15),
        min_avg_daily_move_stock_pct=settings.scan_min_avg_daily_move_stock_pct,
        min_avg_daily_move_crypto_pct=settings.scan_min_avg_daily_move_crypto_pct,
        min_last_day_move_pct=settings.scan_min_last_day_move_pct,
        min_volume_vs_avg=settings.scan_min_volume_vs_avg,
        volatility_weight=settings.scan_volatility_weight,
        activity_weight=settings.scan_activity_weight,
    )

    print_trend_report(bullish)
    elapsed = time.time() - t0
    print(f"\nScan completed in {elapsed:.0f}s -- found {len(bullish)} bullish assets")
    return bullish


def run_single_cycle():
    """Run one complete scan + trade cycle."""
    print()
    print("=" * 70)
    print(f"PAPER TRADING CYCLE -- Mode: {settings.bot_mode}")
    print(f"Capital: ${settings.paper_working_capital_usd:.0f}")
    print("=" * 70)

    # The engine now runs trend scanning internally
    engine = TradingEngine(load_risk_policy())
    engine.run_daily_cycle()

    print("\nCycle complete.")


def run_continuous(interval_minutes: int = 30):
    """Run scan + trade cycles continuously."""
    print(f"Starting continuous paper trading (cycle every {interval_minutes} min)")
    print(f"Mode: {settings.bot_mode} | Capital: ${settings.paper_working_capital_usd:.0f}")
    print("Press Ctrl+C to stop\n")

    engine = TradingEngine(load_risk_policy())
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n{'='*70}")
            print(f"CYCLE {cycle} -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            try:
                engine.run_daily_cycle()
            except Exception as exc:
                print(f"Cycle {cycle} failed: {exc}")

            print(f"\nNext cycle in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print(f"\n\nStopped after {cycle} cycles.")


def main():
    parser = argparse.ArgumentParser(description="Smart paper trading with trend scanning")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between cycles (default: 30)")
    parser.add_argument("--scan-only", action="store_true", help="Just run trend scan, no trading")
    args = parser.parse_args()

    if args.scan_only:
        scan_and_report()
    elif args.continuous:
        run_continuous(args.interval)
    else:
        # Single cycle: scan + report, then trade
        bullish = scan_and_report()
        if bullish:
            print("\n" + "-" * 70)
            print("Proceeding with paper trading cycle...")
            run_single_cycle()
        else:
            print("\nNo bullish assets found. Skipping trade cycle.")
            print("The bot is being smart -- not trading in a bearish market.")


if __name__ == "__main__":
    main()
