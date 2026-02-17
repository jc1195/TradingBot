"""Quick DB check."""
import sys
sys.path.insert(0, ".")
from src.trading_bot.db import get_connection, initialize_database

initialize_database()
conn = get_connection()

print("=== AI DECISIONS ===")
rows = conn.execute("SELECT symbol, model_name, score, confidence FROM ai_decisions ORDER BY created_at").fetchall()
for r in rows:
    action = "?"
    print(f"  {r['symbol']:8s} conf={r['confidence']:.0%} score={r['score']} via {r['model_name']}")
print(f"  Total: {len(rows)}")

print("\n=== ORDERS ===")
orders = conn.execute("SELECT symbol, side, quantity, price, status, mode FROM orders ORDER BY created_at").fetchall()
for o in orders:
    print(f"  {o['side']:5s} {o['symbol']:8s} qty={o['quantity']:.4f} ${o['price']:.4f} [{o['status']}] {o['mode']}")
print(f"  Total: {len(orders)}")

print("\n=== TRADE OUTCOMES ===")
outcomes = conn.execute("SELECT * FROM trade_outcomes").fetchall()
print(f"  Total: {len(outcomes)}")

print("\n=== SYSTEM EVENTS (last 5) ===")
events = conn.execute("SELECT event_type, message FROM system_events ORDER BY created_at DESC LIMIT 5").fetchall()
for e in events:
    print(f"  [{e['event_type']}] {e['message'][:80]}")

print("\n=== TLRY EVENTS ===")
tlry = conn.execute("SELECT event_type, message FROM system_events WHERE message LIKE '%TLRY%'").fetchall()
for e in tlry:
    print(f"  [{e['event_type']}] {e['message'][:100]}")
if not tlry:
    print("  No events found for TLRY")
