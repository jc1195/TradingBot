import sqlite3

conn = sqlite3.connect('data/trading_bot.sqlite3')
cur = conn.cursor()

realized = cur.execute("SELECT COALESCE(SUM(pnl), 0) FROM trade_outcomes").fetchone()[0] or 0.0
closed_trades = cur.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
open_positions = cur.execute(
    "SELECT COUNT(*) FROM orders WHERE side='buy' AND status='paper_filled'"
).fetchone()[0]
open_notional = cur.execute(
    "SELECT COALESCE(SUM(quantity * price), 0) FROM orders WHERE side='buy' AND status='paper_filled'"
).fetchone()[0] or 0.0
total_orders = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

print(f"Realized PnL: ${realized:.2f}")
print(f"Closed trades: {closed_trades}")
print(f"Open positions: {open_positions}")
print(f"Open notional: ${open_notional:.2f}")
print(f"Total orders: {total_orders}")

print("Recent outcomes:")
for row in cur.execute(
    "SELECT order_id, pnl, hold_minutes, closed_at FROM trade_outcomes ORDER BY id DESC LIMIT 5"
):
    print(row)
