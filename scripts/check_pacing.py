"""Check pacing multiplier."""
import json
import sqlite3

conn = sqlite3.connect("data/trading_bot.sqlite3")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT metadata_json FROM system_events WHERE event_type='goal_progress'"
).fetchall()
for r in rows:
    data = json.loads(r["metadata_json"])
    print(json.dumps(data, indent=2))
