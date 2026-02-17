import pathlib
import sqlite3
import subprocess

root = pathlib.Path('.')
pid_path = root / 'runtime' / 'bot.pid'

pid = None
alive = False
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        out = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}'],
            capture_output=True,
            text=True,
            check=False,
        )
        alive = str(pid) in out.stdout
    except Exception:
        pass

print('bot_pid:', pid)
print('bot_process_alive:', alive)

conn = sqlite3.connect('data/trading_bot.sqlite3')
conn.row_factory = sqlite3.Row

orders_total = conn.execute('SELECT COUNT(*) AS c FROM orders').fetchone()['c']
open_paper = conn.execute(
    "SELECT COUNT(*) AS c FROM orders WHERE mode='paper' AND side='buy' AND status='paper_filled'"
).fetchone()['c']
closed = conn.execute('SELECT COUNT(*) AS c FROM trade_outcomes').fetchone()['c']
realized = conn.execute('SELECT COALESCE(SUM(pnl), 0) AS p FROM trade_outcomes').fetchone()['p']
recent_fills = conn.execute(
    "SELECT COUNT(*) AS c FROM system_events WHERE event_type='paper_fill' AND created_at >= datetime('now','-2 hours')"
).fetchone()['c']

last_order = conn.execute(
    'SELECT symbol, status, created_at FROM orders ORDER BY id DESC LIMIT 1'
).fetchone()
last_decision = conn.execute(
    'SELECT symbol, model_name, created_at FROM ai_decisions ORDER BY id DESC LIMIT 1'
).fetchone()

print('orders_total:', orders_total)
print('open_paper_positions:', open_paper)
print('closed_trades:', closed)
print('realized_pnl:', round(float(realized or 0.0), 2))
print('paper_fills_last_2h:', recent_fills)
print('last_order:', dict(last_order) if last_order else None)
print('last_decision:', dict(last_decision) if last_decision else None)
