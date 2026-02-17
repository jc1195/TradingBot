import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading_bot.robinhood_client import RobinhoodClient
from src.trading_bot.watchlist import get_watchlist

conn = sqlite3.connect('data/trading_bot.sqlite3')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    """
    SELECT symbol, quantity, price
    FROM orders
    WHERE side='buy' AND status='paper_filled'
    ORDER BY id ASC
    """
).fetchall()

client = RobinhoodClient()
kind_map = {item.symbol.upper(): item.kind for item in get_watchlist()}

total_cost = 0.0
total_value = 0.0
print('Open positions mark-to-market:')
for row in rows:
    symbol = row['symbol']
    qty = float(row['quantity'] or 0.0)
    entry = float(row['price'] or 0.0)

    symbol_upper = symbol.upper().replace('-USD', '')
    kind = kind_map.get(symbol_upper, 'stock')
    quote = client.get_quote(symbol, kind)
    mark = float(quote.get('mark_price', 0.0) or 0.0)

    if mark <= 0 and kind == 'stock':
        quote = client.get_quote(symbol, 'crypto')
        mark = float(quote.get('mark_price', 0.0) or 0.0)
        kind = 'crypto'

    if mark <= 0:
        mark = entry
        kind = 'unknown'

    cost = qty * entry
    value = qty * mark
    pnl = value - cost

    total_cost += cost
    total_value += value

    print(f"- {symbol:8s} ({kind}) qty={qty:.4f} entry=${entry:.4f} mark=${mark:.4f} pnl=${pnl:+.2f}")

unrealized = total_value - total_cost
print(f"\nEstimated unrealized PnL: ${unrealized:+.2f}")
print(f"Current estimated equity in open positions: ${total_value:.2f}")
