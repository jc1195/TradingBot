"""Full bullish trend scan + automated backtest on top picks.

This script:
 1. Scans ALL assets (main watchlist + penny stocks + micro crypto)
 2. Detects bullish patterns (golden cross, breakout, momentum, etc.)
 3. Ranks everything by trend strength
 4. Backtests the top picks using FAST MODE (indicator-only, no AI)
 5. Optionally runs AI-powered council backtest on the best picks
 6. Outputs actionable trading recommendations
"""

import sys
import time
import warnings

# Suppress yfinance noise
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, ".")

from src.trading_bot.trend_detector import (
    scan_for_bullish_opportunities,
    print_trend_report,
    PENNY_STOCKS,
    MICRO_CRYPTO,
)
from src.trading_bot.scanner import AssetScanner, print_scan_report
from src.trading_bot.backtester import Backtester, BacktestConfig
from src.trading_bot.watchlist import WatchlistItem, get_watchlist
from src.trading_bot.settings import settings


def main():
    print("=" * 70)
    print("FULL MARKET SCAN -- Finding Bullish Opportunities")
    print("Scanning: Main watchlist + Penny stocks + Micro crypto + Trending")
    print("=" * 70)
    print()

    # ── Phase 1: Bullish Trend Scan ───────────────────────────────
    print("PHASE 1: Scanning for bullish patterns...")
    t0 = time.time()

    bullish = scan_for_bullish_opportunities(
        include_penny=True,
        include_micro_crypto=True,
        include_trending=True,   # try to find new trending tickers too
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
    phase1_time = time.time() - t0
    print(f"\nPhase 1 completed in {phase1_time:.0f}s")

    # ── Phase 2: Full Scanner (with news sentiment) ───────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Deep scan with news sentiment on top candidates...")

    if bullish:
        # Only deep-scan the bullish picks (sentiment is slow)
        top_symbols = [b.symbol for b in bullish[:10]]
        top_kinds = list(set(b.kind for b in bullish[:10]))

        scanner = AssetScanner(period="6mo", interval="1d")
        scan_result = scanner.scan(
            kinds=top_kinds,
            symbols=top_symbols,
            top_n=10,
        )
        print_scan_report(scan_result)
    else:
        print("No bullish opportunities found. Market is bearish across the board.")
        print("The bot will WAIT -- not forcing trades in a down market is a strategy too.")

    # ── Phase 3: FAST Backtest (indicator-only, no AI) ────────────
    if bullish:
        print("\n" + "=" * 70)
        print("PHASE 3: FAST Backtest (indicator-only, no GPU needed)...")

        # Pick the top 5 for backtest
        bt_symbols = [b.symbol for b in bullish[:5]]
        bt_kinds = list(set(b.kind for b in bullish[:5]))

        # Create ad-hoc WatchlistItems for dynamically discovered tickers
        known_symbols = {i.symbol for i in get_watchlist()}
        extra_items = []
        for b in bullish[:5]:
            if b.symbol not in known_symbols:
                yf_ticker = f"{b.symbol}-USD" if b.kind == "crypto" else b.symbol
                extra_items.append(WatchlistItem(
                    symbol=b.symbol,
                    kind=b.kind,
                    yf_ticker=yf_ticker,
                    description=f"Discovered by trend scanner (score={b.trend_score})",
                ))

        print(f"Backtesting: {bt_symbols}")
        if extra_items:
            print(f"  (includes {len(extra_items)} discovered tickers: {[e.symbol for e in extra_items]})")

        cfg = BacktestConfig(
            symbols=bt_symbols,
            kinds=bt_kinds if bt_kinds else None,
            extra_items=extra_items,
            period="1y",
            interval="1d",
            starting_capital=500.0,
            goal_equity=1000.0,
            profit_target_pct=8.0,
            stop_loss_pct=3.0,
            trailing_stop_pct=2.5,
            signal_score_gate=10,
            min_confidence=0.50,
            use_ai=False,          # FAST MODE: indicator-only
            use_council=False,
        )

        bt = Backtester(cfg)
        result = bt.run()

        print()
        print("=" * 50)
        print("FAST BACKTEST RESULTS -- Top Bullish Picks")
        print("=" * 50)
        print(f"Final equity:  ${result.final_equity:.2f}")
        print(f"Total PnL:     ${result.total_pnl:.2f}")
        print(f"Trades:        {len(result.trades)} (W:{result.win_count} / L:{result.loss_count})")
        print(f"Win rate:      {result.win_rate:.1%}")
        print(f"Max drawdown:  {result.max_drawdown_pct:.2f}%")
        print(f"Sharpe:        {result.sharpe_estimate:.2f}")
        print(f"Time:          {result.duration_seconds:.1f}s")

        # Per-symbol breakdown
        sym_stats: dict = {}
        for t in result.trades:
            if t.symbol not in sym_stats:
                sym_stats[t.symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t.net_pnl > 0:
                sym_stats[t.symbol]["wins"] += 1
            else:
                sym_stats[t.symbol]["losses"] += 1
            sym_stats[t.symbol]["pnl"] += t.net_pnl

        print("\nPer-symbol:")
        for sym, stats in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total * 100 if total > 0 else 0
            tag = "PROFITABLE" if stats["pnl"] > 0 else "LOSING"
            print(f"  {sym:6s}  {total:2d} trades  WR:{wr:.0f}%  PnL: ${stats['pnl']:+.2f}  [{tag}]")

    # ── Phase 4 (optional): AI Council Backtest on best pick ──────
    if bullish and sym_stats:
        best_sym = max(sym_stats.items(), key=lambda x: x[1]["pnl"])
        if best_sym[1]["pnl"] > 0:
            print(f"\n{'='*70}")
            print(f"PHASE 4: AI Council backtest on best performer: {best_sym[0]}")
            print("(This uses GPU and takes longer...)")

            best_kind = "stock"
            for b in bullish:
                if b.symbol == best_sym[0]:
                    best_kind = b.kind
                    break

            # Create extra item if needed
            best_extra = []
            if best_sym[0] not in known_symbols:
                yf_t = f"{best_sym[0]}-USD" if best_kind == "crypto" else best_sym[0]
                best_extra = [WatchlistItem(best_sym[0], best_kind, yf_t, "Best performer from fast backtest")]

            cfg_ai = BacktestConfig(
                symbols=[best_sym[0]],
                kinds=[best_kind],
                extra_items=best_extra,
                period="6mo",
                interval="1d",
                starting_capital=500.0,
                goal_equity=1000.0,
                profit_target_pct=8.0,
                stop_loss_pct=3.0,
                trailing_stop_pct=2.5,
                signal_score_gate=10,
                min_confidence=0.60,
                use_ai=True,
                use_council=True,
            )

            bt_ai = Backtester(cfg_ai)
            result_ai = bt_ai.run()

            print(f"\nAI COUNCIL RESULTS for {best_sym[0]}:")
            print(f"  PnL:      ${result_ai.total_pnl:.2f}")
            print(f"  Trades:   {len(result_ai.trades)} (W:{result_ai.win_count} / L:{result_ai.loss_count})")
            print(f"  Win rate: {result_ai.win_rate:.1%}")
            print(f"  Time:     {result_ai.duration_seconds:.1f}s")

    # ── Summary ───────────────────────────────────────────────────
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(f"FULL SCAN COMPLETE in {total_time:.0f}s")
    print("=" * 70)

    if bullish:
        print("\nTop bullish opportunities:")
        for i, b in enumerate(bullish[:5], 1):
            penny = " [PENNY]" if b.is_penny else ""
            gc = " [GOLDEN CROSS]" if b.golden_cross else ""
            bo = " [BREAKOUT]" if b.breakout_detected else ""
            print(f"  {i}. {b.symbol:6s}  ${b.price:>10.4f}  Score:{b.trend_score:+4d}  "
                  f"Move:{b.avg_daily_move_pct:.1f}%/day{penny}{gc}{bo}")
        print("\nRecommendation: Focus paper trading on these assets.")
    else:
        print("\nNo clear bullish opportunities right now.")
        print("Recommendation: Stay in cash. Patience IS a trading strategy.")


if __name__ == "__main__":
    main()
