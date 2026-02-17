"""Quick DB check script for smoke test 2."""
import json
import sqlite3

conn = sqlite3.connect("data/trading_bot.sqlite3")
conn.row_factory = sqlite3.Row

print("=== ORDERS ===")
for r in conn.execute("SELECT symbol, side, quantity, price, status, mode FROM orders ORDER BY id"):
    notional = float(r["quantity"]) * float(r["price"])
    print(f"  {r['side']:4s} {r['symbol']:8s} qty={r['quantity']:.4f} @ ${r['price']:.4f} = ${notional:.2f} [{r['status']}]")

total_invested = sum(
    float(r["quantity"]) * float(r["price"])
    for r in conn.execute("SELECT quantity, price FROM orders WHERE status='paper_filled' AND side='buy'")
)
count = conn.execute("SELECT COUNT(*) FROM orders WHERE status='paper_filled' AND side='buy'").fetchone()[0]
print(f"\n  Total: {count} orders, ${total_invested:.2f} invested")

print("\n=== AI DECISIONS ===")
for r in conn.execute("SELECT symbol, model_name, decision_json FROM ai_decisions"):
    d = json.loads(r["decision_json"])
    dec = d.get("decision", d)
    action = dec.get("action", "?")
    conf = dec.get("confidence", 0)
    votes = dec.get("council_votes", "")
    print(f"  {r['symbol']:8s} {action:5s} conf={conf:.0%} via {r['model_name']}{' | ' + votes if votes else ''}")
