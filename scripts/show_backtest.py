"""Quick display of backtest results."""
import sqlite3

conn = sqlite3.connect("data/trading_bot.sqlite3")
conn.row_factory = sqlite3.Row

trades = conn.execute(
    "SELECT symbol, entry_price, exit_price, net_pnl, confidence "
    "FROM backtest_trades ORDER BY id ASC"
).fetchall()

if not trades:
    print("No backtest trades found.")
    raise SystemExit()

total = sum(float(t["net_pnl"]) for t in trades)
wins = sum(1 for t in trades if float(t["net_pnl"]) > 0)
losses = len(trades) - wins
wr = wins / len(trades) if trades else 0

print(f"  Trades: {len(trades)}  |  Wins: {wins}  |  Losses: {losses}")
print(f"  Win Rate: {wr:.1%}")
print(f"  Total PnL: ${total:+.2f}")
print(f"  Final Equity: ${500 + total:.2f}")
print()
for i, t in enumerate(trades, 1):
    pnl = float(t["net_pnl"])
    tag = " WIN" if pnl > 0 else "LOSS"
    print(
        f"  {i:>2}. [{tag}]  entry=${float(t['entry_price']):.4f}  "
        f"exit=${float(t['exit_price']):.4f}  "
        f"pnl=${pnl:+.4f}  conf={float(t['confidence']):.0%}"
    )

conn.close()
