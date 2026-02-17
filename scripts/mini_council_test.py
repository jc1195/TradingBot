"""Minimal council test — writes results directly to file."""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, ".")

OUT = "council_results.txt"

def log(msg):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# Clear output file
open(OUT, "w").close()

from src.trading_bot.backtester import Backtester, BacktestConfig

log("=== COUNCIL BACKTEST TEST ===")
log(f"Time: {time.strftime('%H:%M:%S')}")

# Run Council mode
cfg = BacktestConfig(
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
    use_council=True,
    max_concurrent_positions=2,
    lookback_window=20,
)

log(f"Council: ON | Symbols: {cfg.symbols} | Period: {cfg.period}")
bt = Backtester(cfg)
t0 = time.time()
r1 = bt.run()
t1 = time.time() - t0

log(f"\n--- COUNCIL RESULTS ({t1:.0f}s) ---")
log(f"PnL: ${r1.total_pnl:+.2f} | Trades: {len(r1.trades)} | Win: {r1.win_count} Loss: {r1.loss_count} | WR: {r1.win_rate:.1%}")
log(f"Equity: ${r1.final_equity:.2f} | Drawdown: {r1.max_drawdown_pct:.2f}%")
for t in r1.trades:
    tag = "W" if t.net_pnl > 0 else "L"
    log(f"  {tag} {t.symbol:6s} ${t.entry_price:.2f}->${t.exit_price:.2f} pnl=${t.net_pnl:+.2f} held={t.hold_candles}c conf={t.confidence:.0%}")

# Run Single mode
log(f"\n=== SINGLE AGENT TEST ===")
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
    use_council=False,
    max_concurrent_positions=2,
    lookback_window=20,
)

log(f"Council: OFF | Symbols: {cfg2.symbols} | Period: {cfg2.period}")
bt2 = Backtester(cfg2)
t0 = time.time()
r2 = bt2.run()
t2 = time.time() - t0

log(f"\n--- SINGLE RESULTS ({t2:.0f}s) ---")
log(f"PnL: ${r2.total_pnl:+.2f} | Trades: {len(r2.trades)} | Win: {r2.win_count} Loss: {r2.loss_count} | WR: {r2.win_rate:.1%}")
log(f"Equity: ${r2.final_equity:.2f} | Drawdown: {r2.max_drawdown_pct:.2f}%")
for t in r2.trades:
    tag = "W" if t.net_pnl > 0 else "L"
    log(f"  {tag} {t.symbol:6s} ${t.entry_price:.2f}->${t.exit_price:.2f} pnl=${t.net_pnl:+.2f} held={t.hold_candles}c conf={t.confidence:.0%}")

log(f"\n=== COMPARISON ===")
log(f"Council: PnL=${r1.total_pnl:+.2f} WR={r1.win_rate:.1%} Trades={len(r1.trades)} Time={t1:.0f}s")
log(f"Single:  PnL=${r2.total_pnl:+.2f} WR={r2.win_rate:.1%} Trades={len(r2.trades)} Time={t2:.0f}s")
winner = "COUNCIL" if r1.total_pnl > r2.total_pnl else "SINGLE"
log(f"Winner:  {winner}")
log(f"Done at {time.strftime('%H:%M:%S')}")
