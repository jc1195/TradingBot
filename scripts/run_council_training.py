"""Master training script — runs all scanners and council backtest.

1. Scans penny stocks for bullish setups
2. Runs bullish trend scanner on all watchlist + discovered pennies
3. Runs 3-agent council backtest on the top picks
4. Reports full results

Usage:
    python scripts/run_council_training.py
"""

import sys
import time

sys.path.insert(0, ".")

from src.trading_bot.penny_stocks import PennyStockScreener, candidates_to_watchlist, print_penny_report
from src.trading_bot.trend_scanner import BullishTrendScanner, print_trend_report
from src.trading_bot.watchlist import get_watchlist
from src.trading_bot.backtester import Backtester, BacktestConfig


def main():
    overall_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Penny Stock Screening
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: PENNY STOCK & SMALL-CAP SCREENING")
    print("=" * 70)

    screener = PennyStockScreener(max_price=10.0)
    penny_result = screener.scan(top_n=10, include_small_caps=True)
    print_penny_report(penny_result)

    # Convert top penny picks to watchlist items
    penny_watchlist = candidates_to_watchlist(penny_result.top_picks)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Bullish Trend Scanning
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: BULLISH TREND ANALYSIS")
    print("=" * 70)

    # Scan existing watchlist
    existing_watchlist = get_watchlist()
    scanner = BullishTrendScanner()
    trend_result = scanner.scan_watchlist(existing_watchlist)
    print_trend_report(trend_result)

    # Also scan penny picks that aren't already in watchlist
    existing_symbols = {item.symbol for item in existing_watchlist}
    new_penny_symbols = [p.symbol for p in penny_result.top_picks if p.symbol not in existing_symbols]

    if new_penny_symbols:
        print(f"\nAlso scanning {len(new_penny_symbols)} newly discovered penny stocks...")
        penny_trends = scanner.scan_symbols(new_penny_symbols, kind="stock")
        print_trend_report(penny_trends)

        # Merge penny bulls into the results
        for sig in penny_trends.strong_bulls + penny_trends.emerging_bulls:
            trend_result.strong_bulls.append(sig) if sig.trend_type == "strong_bull" else None
            trend_result.emerging_bulls.append(sig) if sig.trend_type in ("bull", "emerging_bull") else None

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Select assets for council backtest
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: SELECTING BEST ASSETS FOR COUNCIL BACKTEST")
    print("=" * 70)

    # Combine strong + emerging bulls
    all_bulls = trend_result.strong_bulls + trend_result.emerging_bulls
    all_bulls.sort(key=lambda s: s.trend_strength, reverse=True)

    # Take top 8 (balance of speed vs coverage)
    top_for_trading = all_bulls[:8]

    if not top_for_trading:
        # Fallback: use top assets from full list
        all_sorted = sorted(trend_result.all_signals, key=lambda s: s.trend_strength, reverse=True)
        top_for_trading = all_sorted[:6]

    selected_symbols = [s.symbol for s in top_for_trading]
    selected_kinds = list(set(s.kind for s in top_for_trading))

    print(f"\nSelected {len(selected_symbols)} assets for council backtest:")
    for s in top_for_trading:
        print(f"  {s.symbol:6s}  Strength:{s.trend_strength:3d}  {s.trend_type:15s}  ${s.price:.4f}")
        for r in s.reasons[:2]:
            print(f"         → {r}")

    if not selected_symbols:
        print("\nNo bullish assets found. Market may be in a broad downturn.")
        print("Falling back to known performers...")
        selected_symbols = ["PLTR", "NVDA", "RIOT"]
        selected_kinds = ["stock"]

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: 3-Agent Council Backtest
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: 3-AGENT COUNCIL BACKTEST")
    print("=" * 70)

    cfg = BacktestConfig(
        symbols=selected_symbols,
        kinds=selected_kinds,
        period="1y",
        interval="1d",
        starting_capital=500.0,
        goal_equity=1000.0,
        profit_target_pct=8.0,
        stop_loss_pct=3.0,
        trailing_stop_pct=2.5,
        signal_score_gate=15,
        min_confidence=0.65,
        use_council=True,         # 3-agent parallel voting
        max_concurrent_positions=3,
    )

    print(f"\nRun ID: {cfg.run_id}")
    print(f"Capital: ${cfg.starting_capital} → Goal: ${cfg.goal_equity}")
    print(f"Council: {'ENABLED (3 agents)' if cfg.use_council else 'DISABLED'}")
    print(f"Assets: {cfg.symbols}")
    print(f"TP: {cfg.profit_target_pct}%  SL: {cfg.stop_loss_pct}%  Trail: {cfg.trailing_stop_pct}%")
    print()

    bt = Backtester(cfg)
    t0 = time.time()
    result = bt.run()
    backtest_elapsed = time.time() - t0

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COUNCIL BACKTEST RESULTS")
    print("=" * 70)
    print(f"Final equity:  ${result.final_equity:.2f}")
    print(f"Total PnL:     ${result.total_pnl:.2f}")
    print(f"Trades:        {len(result.trades)} (W:{result.win_count} / L:{result.loss_count})")
    print(f"Win rate:      {result.win_rate:.1%}")
    print(f"Max drawdown:  {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe:        {result.sharpe_estimate:.2f}")
    print(f"Backtest time: {backtest_elapsed:.1f}s")

    # Per-symbol breakdown
    symbols_seen = {}
    for t in result.trades:
        if t.symbol not in symbols_seen:
            symbols_seen[t.symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}
        if t.net_pnl > 0:
            symbols_seen[t.symbol]["wins"] += 1
        else:
            symbols_seen[t.symbol]["losses"] += 1
        symbols_seen[t.symbol]["pnl"] += t.net_pnl

    print("\nPer-symbol breakdown:")
    for sym, stats in sorted(symbols_seen.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        pnl_tag = "+" if stats["pnl"] >= 0 else ""
        print(f"  {sym:8s}  {total:2d} trades  W:{stats['wins']:2d} L:{stats['losses']:2d}  "
              f"WR:{wr:.0f}%  PnL: ${pnl_tag}{stats['pnl']:.2f}")

    print("\nAll trades:")
    for t in result.trades:
        tag = "WIN " if t.net_pnl > 0 else "LOSS"
        print(f"  {tag} {t.symbol:6s} ${t.entry_price:.4f} → ${t.exit_price:.4f}  "
              f"pnl=${t.net_pnl:+.2f}  held={t.hold_candles}c  conf={t.confidence:.0%}")

    total_time = time.time() - overall_start
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Penny stocks found: {penny_result.candidates_found}")
    print(f"Bullish assets: {trend_result.bullish_count}")


if __name__ == "__main__":
    main()
