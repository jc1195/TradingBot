"""Run the improved multi-asset backtest."""

import time
import sys
sys.path.insert(0, ".")

from src.trading_bot.backtester import Backtester, BacktestConfig


def main():
    cfg = BacktestConfig(
        symbols=["NVDA", "PLTR", "TSLA", "MARA", "RIOT"],
        kinds=["stock"],
        period="1y",
        interval="1d",
        starting_capital=500.0,
        goal_equity=1000.0,
        profit_target_pct=8.0,   # take profit at +8%
        stop_loss_pct=3.0,       # stop loss at -3%  (R:R = 2.67:1)
        trailing_stop_pct=2.5,   # trail 2.5% from peak
        signal_score_gate=15,    # only trade when technicals are decent
        min_confidence=0.70,     # raise AI confidence bar
    )
    bt = Backtester(cfg)
    print(f"Run ID: {cfg.run_id}")
    print(f"TP={cfg.profit_target_pct}%  SL={cfg.stop_loss_pct}%  Trail={cfg.trailing_stop_pct}%  Gate={cfg.signal_score_gate}  MinConf={cfg.min_confidence}")
    print(f"Assets: {cfg.symbols}")
    print()

    t0 = time.time()
    result = bt.run()
    elapsed = time.time() - t0

    print()
    print("=" * 50)
    print("IMPROVED BACKTEST RESULTS")
    print("=" * 50)
    print(f"Final equity:  ${result.final_equity:.2f}")
    print(f"Total PnL:     ${result.total_pnl:.2f}")
    print(f"Trades:        {len(result.trades)} (W:{result.win_count} / L:{result.loss_count})")
    print(f"Win rate:      {result.win_rate:.1%}")
    print(f"Max drawdown:  {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe:        {result.sharpe_estimate:.2f}")
    print(f"Time:          {elapsed:.1f}s")
    print()

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

    print("Per-symbol breakdown:")
    for sym, stats in sorted(symbols_seen.items()):
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        print(f"  {sym:8s}  {total:2d} trades  W:{stats['wins']:2d} L:{stats['losses']:2d}  WR:{wr:.0f}%  PnL: ${stats['pnl']:.2f}")

    print()
    print("All trades:")
    for t in result.trades:
        tag = "WIN " if t.net_pnl > 0 else "LOSS"
        print(f"  {tag} {t.symbol:5s} {t.entry_price:.4f} -> {t.exit_price:.4f}  pnl=${t.net_pnl:+.2f}  held={t.hold_candles}c  conf={t.confidence:.0%}")


if __name__ == "__main__":
    main()
