import sqlite3
from pathlib import Path

from .settings import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    snapshot_time TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    features_json TEXT
);

CREATE TABLE IF NOT EXISTS news_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    published_at TEXT,
    sentiment REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,
    score REAL,
    confidence REAL,
    rationale TEXT,
    decision_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL,
    price REAL,
    status TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    pnl REAL,
    hold_minutes INTEGER,
    max_drawdown REAL,
    closed_at TEXT,
    FOREIGN KEY(order_id) REFERENCES orders(id)
);

CREATE TABLE IF NOT EXISTS daily_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_date TEXT NOT NULL,
    summary TEXT NOT NULL,
    lessons TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    starting_capital REAL,
    final_equity REAL,
    total_pnl REAL,
    win_count INTEGER,
    loss_count INTEGER,
    win_rate REAL,
    max_drawdown_pct REAL,
    sharpe_estimate REAL,
    symbols_tested TEXT,
    candles_processed INTEGER,
    ai_calls INTEGER,
    duration_seconds REAL,
    errors_json TEXT
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    gross_pnl REAL,
    fees REAL,
    net_pnl REAL,
    hold_candles INTEGER,
    entry_timestamp TEXT,
    exit_timestamp TEXT,
    confidence REAL,
    rationale TEXT,
    FOREIGN KEY(run_id) REFERENCES backtest_runs(run_id)
);
"""



def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn



def initialize_database() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_column(conn, "orders", "decision_confidence", "REAL")
        _ensure_column(conn, "orders", "decision_score", "REAL")
        _ensure_column(conn, "orders", "ai_provider", "TEXT")
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in info}
    if column_name in existing:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
