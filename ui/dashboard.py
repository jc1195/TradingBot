from __future__ import annotations

import json
import os
import re
import subprocess
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trading_bot.settings import settings
from src.trading_bot.health_check import generate_health_report, save_health_report
from src.trading_bot.robinhood_client import RobinhoodClient
from src.trading_bot.watchlist import get_watchlist

DB_PATH = PROJECT_ROOT / "data" / "trading_bot.sqlite3"
ENV_PATH = PROJECT_ROOT / ".env"
RISK_PATH = PROJECT_ROOT / "config" / "risk_policy.yaml"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
BOT_PID_PATH = RUNTIME_DIR / "bot.pid"
BOT_LOG_PATH = RUNTIME_DIR / "bot.log"
DEBUG_SNAPSHOT_PATH = RUNTIME_DIR / "diagnostics_latest.json"
HEALTH_REPORT_PATH = RUNTIME_DIR / "health_report_latest.json"
CYCLE_REPORT_PATH = RUNTIME_DIR / "cycle_report_latest.json"
HOURLY_STRATEGY_REVIEW_PATH = RUNTIME_DIR / "hourly_strategy_review_latest.json"


def read_table(query: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(query, conn)
        except Exception:
            return pd.DataFrame()


def _env_line(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def update_env_values(changes: dict[str, float | int | str | bool]) -> None:
    existing = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    updated = existing[:]
    seen: set[str] = set()

    for idx, line in enumerate(updated):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in changes:
            updated[idx] = f"{key}={_env_line(changes[key])}"
            seen.add(key)

    for key, value in changes.items():
        if key not in seen:
            updated.append(f"{key}={_env_line(value)}")

    ENV_PATH.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def load_env_values() -> dict[str, str]:
    data: dict[str, str] = {}
    if not ENV_PATH.exists():
        return data
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def env_get_str(env: dict[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        return default
    return str(value)


def env_get_float(env: dict[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def env_get_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def env_get_bool(env: dict[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def update_risk_policy_values(
    risk_per_trade_pct: float,
    min_confidence: float,
    max_positions: int,
    allow_live_orders: bool,
) -> None:
    lines = RISK_PATH.read_text(encoding="utf-8").splitlines() if RISK_PATH.exists() else []
    if not lines:
        return

    def _replace(prefix: str, value: float | int) -> None:
        for i, line in enumerate(lines):
            if line.strip().startswith(prefix):
                indent = line[: len(line) - len(line.lstrip(" "))]
                lines[i] = f"{indent}{prefix} {value}"
                return

    _replace("max_risk_per_trade_pct:", round(risk_per_trade_pct, 4))
    _replace("min_confidence_to_trade:", round(min_confidence, 4))
    _replace("max_concurrent_positions:", int(max_positions))
    _replace("allow_live_orders:", "true" if allow_live_orders else "false")

    RISK_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_pid() -> int | None:
    pid: int | None = None
    if BOT_PID_PATH.exists():
        try:
            pid = int(BOT_PID_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None

    if pid and is_process_alive(pid):
        return pid

    discovered_pid = find_run_bot_pid()
    if discovered_pid:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        BOT_PID_PATH.write_text(str(discovered_pid), encoding="utf-8")
        return discovered_pid

    return None


def find_run_bot_pid() -> int | None:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$p=(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*run_bot.py*' } | Select-Object -First 1 -ExpandProperty ProcessId); if($p){$p}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (out.stdout or "").strip()
        return int(text) if text.isdigit() else None
    except Exception:
        return None


def is_process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in out.stdout
    except Exception:
        return False


def start_bot_process(interval_minutes: int) -> tuple[bool, str]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    pid = read_pid()
    if is_process_alive(pid):
        return False, f"Bot already running (PID {pid})."

    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        return False, "Python environment not found at .venv/Scripts/python.exe"

    with open(BOT_LOG_PATH, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [str(python_exe), "run_bot.py", "--interval", str(interval_minutes)],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    time.sleep(1.0)
    if proc.poll() is not None:
        return False, "Bot process exited immediately. Check Bot log tail for startup errors."

    BOT_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
    return True, f"Bot started in background (PID {proc.pid})."


def stop_bot_process() -> tuple[bool, str]:
    pid = read_pid()
    if not pid:
        pid = find_run_bot_pid()
    if not pid:
        return False, "No bot PID found."

    if not is_process_alive(pid):
        BOT_PID_PATH.unlink(missing_ok=True)
        return False, "Bot process is not running."

    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, check=False)
    BOT_PID_PATH.unlink(missing_ok=True)
    return True, f"Stopped bot process (PID {pid})."


def minutes_since(iso_value: str | None) -> float | None:
    if not iso_value:
        return None
    try:
        text = str(iso_value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None


def truncate_words(text: str, max_words: int = 200) -> str:
    parts = str(text or "").split()
    if len(parts) <= max_words:
        return " ".join(parts)
    return " ".join(parts[:max_words]) + " ..."


def extract_agent_reasons_from_summary(summary: str) -> dict[str, str]:
    text = str(summary or "")
    if not text:
        return {}

    patterns = {
        "TECHNICIAN": r"\[TECHNICIAN[^\]]*\]\s*(.*?)(?=\s*\|\s*\[|$)",
        "SENTINEL": r"\[SENTINEL[^\]]*\]\s*(.*?)(?=\s*\|\s*\[|$)",
        "STRATEGIST": r"\[STRATEGIST[^\]]*\]\s*(.*?)(?=\s*\|\s*\[|$)",
    }

    reasons: dict[str, str] = {}
    for agent, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            reasons[agent] = " ".join(match.group(1).split())
    return reasons


def get_symbol_kind_map() -> dict[str, str]:
    return {item.symbol.upper(): item.kind for item in get_watchlist()}


def load_open_positions_snapshot() -> pd.DataFrame:
    rows = read_table(
        """
        SELECT id, symbol, quantity, price, created_at
        FROM orders
        WHERE side = 'buy' AND status = 'paper_filled'
        ORDER BY id ASC
        """
    )
    if rows.empty:
        return rows

    kind_map = get_symbol_kind_map()
    rh = RobinhoodClient()
    enriched: list[dict[str, Any]] = []

    for _, row in rows.iterrows():
        symbol = str(row["symbol"])
        quantity = float(row["quantity"] or 0.0)
        entry_price = float(row["price"] or 0.0)
        symbol_key = symbol.upper().replace("-USD", "")
        kind = kind_map.get(symbol_key, "stock")

        quote = rh.get_quote(symbol, kind)
        mark_price = float(quote.get("mark_price", 0.0) or 0.0)
        if mark_price <= 0 and kind == "stock":
            quote = rh.get_quote(symbol, "crypto")
            mark_price = float(quote.get("mark_price", 0.0) or 0.0)
            kind = "crypto"

        safe_mark = mark_price if mark_price > 0 else entry_price
        notional_entry = quantity * entry_price
        notional_now = quantity * safe_mark
        pnl = notional_now - notional_entry
        pnl_pct = (pnl / notional_entry) if notional_entry > 0 else 0.0

        enriched.append(
            {
                "id": int(row["id"]),
                "symbol": symbol,
                "kind": kind,
                "quantity": quantity,
                "entry_price": entry_price,
                "mark_price": safe_mark,
                "entry_notional": notional_entry,
                "current_notional": notional_now,
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": pnl_pct,
                "created_at": str(row["created_at"]),
                "has_live_price": mark_price > 0,
            }
        )

    return pd.DataFrame(enriched)


def close_all_paper_positions_safe() -> tuple[int, int, float]:
    snapshot = load_open_positions_snapshot()
    if snapshot.empty:
        return 0, 0, 0.0

    now_iso = datetime.now(timezone.utc).isoformat()
    closed = 0
    skipped = 0
    total_pnl = 0.0

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for _, row in snapshot.iterrows():
            if not bool(row["has_live_price"]):
                skipped += 1
                continue

            order_id = int(row["id"])
            quantity = float(row["quantity"])
            entry_price = float(row["entry_price"])
            mark_price = float(row["mark_price"])
            created_at = str(row["created_at"])

            exit_price = mark_price * (1 - settings.paper_slippage_bps / 10000)
            gross_pnl = (exit_price - entry_price) * quantity
            traded_notional = (entry_price + exit_price) * quantity
            fees = traded_notional * (settings.paper_fee_bps / 10000)
            net_pnl = gross_pnl - fees
            hold_min = minutes_since(created_at) or 0.0
            max_drawdown = min(0.0, (mark_price - entry_price) / entry_price if entry_price > 0 else 0.0)

            conn.execute(
                "UPDATE orders SET status = 'paper_closed' WHERE id = ? AND status = 'paper_filled'",
                (order_id,),
            )
            conn.execute(
                """
                INSERT INTO trade_outcomes (order_id, pnl, hold_minutes, max_drawdown, closed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (order_id, net_pnl, int(hold_min), max_drawdown, now_iso),
            )
            closed += 1
            total_pnl += net_pnl

        conn.execute(
            """
            INSERT INTO system_events (event_type, message, metadata_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "manual_sell_all_safe",
                f"Manual safe sell-all closed {closed} positions (skipped {skipped})",
                json.dumps({"closed": closed, "skipped": skipped, "realized_pnl": round(total_pnl, 6)}),
                now_iso,
            ),
        )
        conn.commit()

    return closed, skipped, total_pnl


def close_all_live_positions_safe() -> tuple[int, int, int]:
    if not DB_PATH.exists():
        return 0, 0, 0

    rh = RobinhoodClient()
    kind_map = get_symbol_kind_map()
    submitted = 0
    cancelled = 0
    skipped = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, symbol, quantity, status
            FROM orders
            WHERE side = 'buy' AND status IN ('live_ready', 'live_submitted', 'live_filled')
            ORDER BY id ASC
            """
        ).fetchall()

        for row in rows:
            order_id = int(row["id"])
            symbol = str(row["symbol"])
            quantity = float(row["quantity"] or 0.0)
            status = str(row["status"])
            symbol_key = symbol.upper().replace("-USD", "")
            kind = kind_map.get(symbol_key, "stock")

            if status == "live_ready":
                conn.execute("UPDATE orders SET status = 'live_cancelled_safe' WHERE id = ?", (order_id,))
                cancelled += 1
                continue

            quote = rh.get_quote(symbol, kind)
            mark = float(quote.get("mark_price", 0.0) or 0.0)
            if mark <= 0:
                skipped += 1
                continue

            try:
                response = rh.place_sell_order(symbol=symbol, kind=kind, quantity=quantity)
                conn.execute("UPDATE orders SET status = 'live_exit_submitted' WHERE id = ?", (order_id,))
                conn.execute(
                    """
                    INSERT INTO system_events (event_type, message, metadata_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        "live_exit_submitted",
                        f"Manual safe sell submitted for {symbol}",
                        json.dumps({"order_id": order_id, "symbol": symbol, "quantity": quantity, "response": response}),
                        now_iso,
                    ),
                )
                submitted += 1
            except Exception:
                skipped += 1

        conn.execute(
            """
            INSERT INTO system_events (event_type, message, metadata_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "manual_sell_all_safe_live",
                f"Manual live safe sell-all: submitted {submitted}, cancelled {cancelled}, skipped {skipped}",
                json.dumps({"submitted": submitted, "cancelled": cancelled, "skipped": skipped}),
                now_iso,
            ),
        )
        conn.commit()

    return submitted, cancelled, skipped


def collect_debug_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bot_pid": read_pid(),
        "bot_alive": False,
        "events_24h": {},
        "table_counts": {},
        "last_errors": [],
        "last_log_lines": [],
    }

    snapshot["bot_alive"] = is_process_alive(snapshot.get("bot_pid"))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for table in ["orders", "ai_decisions", "trade_outcomes", "system_events", "daily_reviews"]:
            try:
                row = conn.execute(f"SELECT COUNT(1) AS n FROM {table}").fetchone()
                snapshot["table_counts"][table] = int(row["n"] if row else 0)
            except Exception:
                snapshot["table_counts"][table] = -1

        events = conn.execute(
            """
            SELECT event_type, COUNT(1) AS n
            FROM system_events
            WHERE created_at >= datetime('now', '-1 day')
            GROUP BY event_type
            ORDER BY n DESC
            LIMIT 25
            """
        ).fetchall()
        snapshot["events_24h"] = {str(r["event_type"]): int(r["n"]) for r in events}

        errs = conn.execute(
            """
            SELECT event_type, message, created_at
            FROM system_events
            WHERE event_type IN ('pipeline_error', 'kill_switch', 'drawdown_alert', 'quality_alert')
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()
        snapshot["last_errors"] = [dict(r) for r in errs]

    if BOT_LOG_PATH.exists():
        try:
            lines = BOT_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
            snapshot["last_log_lines"] = lines[-60:]
        except Exception:
            snapshot["last_log_lines"] = []

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def get_live_daily_notional_used() -> float:
    if not DB_PATH.exists():
        return 0.0
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(quantity * price), 0.0) AS notional
            FROM orders
            WHERE mode = 'live'
              AND side = 'buy'
              AND status IN ('live_ready', 'live_submitted', 'live_filled')
              AND created_at >= datetime('now', 'start of day')
            """
        ).fetchone()
        return float(row[0] if row and row[0] is not None else 0.0)

st.set_page_config(page_title="TradingBot Control Center", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .main {background: linear-gradient(180deg, #0B1020 0%, #121A30 100%); color: #E6EDF3;}
        .stApp header {background: transparent;}
        div[data-testid='stSidebar'] {background: #0E1529;}
        .card {
            padding: 16px; border-radius: 16px; background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 TradingBot Control Center")
st.caption("Local AI trading operations dashboard (paper mode first).")

env_values = load_env_values()
runtime_mode = env_get_str(env_values, "BOT_MODE", settings.bot_mode)
dynamic_discovery_enabled = env_get_bool(
    env_values,
    "DYNAMIC_DISCOVERY_ENABLED",
    bool(getattr(settings, "dynamic_discovery_enabled", False)),
)
runtime_use_council = env_get_bool(env_values, "USE_COUNCIL", bool(getattr(settings, "use_council", True)))
runtime_unanimous = env_get_bool(
    env_values,
    "COUNCIL_REQUIRE_UNANIMOUS_BUY",
    bool(getattr(settings, "council_require_unanimous_buy", False)),
)
runtime_daily_goal = env_get_float(env_values, "DAILY_PROFIT_GOAL_USD", float(settings.daily_profit_goal_usd))
runtime_weekly_goal = env_get_float(env_values, "WEEKLY_PROFIT_GOAL_USD", float(settings.weekly_profit_goal_usd))
runtime_paper_cap = env_get_float(env_values, "PAPER_WORKING_CAPITAL_USD", float(settings.paper_working_capital_usd))
runtime_paper_wiggle = env_get_float(
    env_values,
    "PAPER_EXTRA_PLAY_CASH_USD",
    float(getattr(settings, "paper_extra_play_cash_usd", 0.0)),
)
runtime_max_positions = env_get_int(env_values, "MAX_CONCURRENT_POSITIONS", int(settings.max_concurrent_positions))
runtime_min_conf = env_get_float(env_values, "MIN_CONFIDENCE_TO_TRADE", float(settings.min_confidence_to_trade))
runtime_risk_pct = env_get_float(env_values, "MAX_RISK_PER_TRADE_PCT", float(settings.max_risk_per_trade_pct))
runtime_cycle_interval_min = max(1, min(60, int(round(env_get_int(env_values, "SERVICE_HEARTBEAT_SECONDS", 900) / 60))))
runtime_prefer_gpu = env_get_bool(env_values, "PREFER_GPU", True)
runtime_ollama_model = env_get_str(env_values, "OLLAMA_MODEL", "")
runtime_force_paper_lock = env_get_bool(env_values, "UI_FORCE_PAPER_ONLY", runtime_mode == "paper")
runtime_show_advanced = env_get_bool(env_values, "UI_SHOW_ADVANCED", False)

with st.sidebar:
    st.subheader("Runtime Controls")
    mode = st.selectbox("Mode", ["paper", "live"], index=0 if runtime_mode == "paper" else 1)
    force_paper_only = st.toggle("Paper-only safety lock", value=runtime_force_paper_lock)
    show_advanced = st.toggle("Show advanced data tables", value=runtime_show_advanced)
    dynamic_discovery_ui = st.toggle("Dynamic discovery (find new trending tickers)", value=dynamic_discovery_enabled)
    live_confirm = st.checkbox("I understand live trading uses real money", value=False)
    cycle_interval_minutes = st.slider("Cycle interval (minutes)", 1, 60, runtime_cycle_interval_min)
    auto_refresh = st.slider("Auto-refresh hint (seconds)", 0, 120, 0)
    risk_per_trade = st.slider("Risk per trade (%)", 0.1, 25.0, float(runtime_risk_pct * 100), 0.1)
    min_conf = st.slider("Min confidence", 0.0, 1.0, float(runtime_min_conf), 0.01)
    max_positions = st.slider("Max concurrent positions", 1, 20, int(runtime_max_positions))
    daily_goal_ui = st.number_input("Daily goal ($)", min_value=0.0, value=float(runtime_daily_goal), step=10.0)
    weekly_goal_ui = st.number_input("Weekly goal ($)", min_value=0.0, value=float(runtime_weekly_goal), step=25.0)
    max_capital_ui = st.number_input("Max paper working capital ($)", min_value=0.0, value=float(runtime_paper_cap), step=25.0)
    extra_play_cash_ui = st.number_input(
        "Extra play cash wiggle room ($)",
        min_value=0.0,
        value=float(runtime_paper_wiggle),
        step=25.0,
    )

    st.subheader("Safety")
    pause_bot = st.toggle("Pause bot", value=False)
    kill_switch = st.toggle("Kill switch", value=False)

    st.subheader("Model")
    preferred_model = st.text_input("Preferred Ollama model", value=runtime_ollama_model)
    use_gpu = st.toggle("Use GPU acceleration (RTX 3070)", value=runtime_prefer_gpu)

    save_profile = st.button("Save runtime profile")
    start_bot = st.button("Start bot (background)")
    stop_bot = st.button("Stop bot")
    start_live = st.button("Arm + Start Live Trading")
    sell_all_safe = st.button("Sell All (safe, paper)")
    run_diagnostics = st.button("Run diagnostics snapshot")
    refresh_now = st.button("Refresh data now")

if save_profile:
    resolved_mode = "paper" if force_paper_only else mode
    allow_live = (resolved_mode == "live") and (not force_paper_only) and live_confirm
    env_changes: dict[str, float | int | str | bool] = {
        "BOT_MODE": resolved_mode,
        "LIVE_MODE_UNLOCK": allow_live,
        "REQUIRE_EXPLICIT_LIVE_CONFIRM": True,
        "MAX_RISK_PER_TRADE_PCT": round(risk_per_trade / 100.0, 4),
        "MIN_CONFIDENCE_TO_TRADE": round(min_conf, 4),
        "MAX_CONCURRENT_POSITIONS": int(max_positions),
        "DAILY_PROFIT_GOAL_USD": round(float(daily_goal_ui), 2),
        "WEEKLY_PROFIT_GOAL_USD": round(float(weekly_goal_ui), 2),
        "PAPER_WORKING_CAPITAL_USD": round(float(max_capital_ui), 2),
        "PAPER_EXTRA_PLAY_CASH_USD": round(float(extra_play_cash_ui), 2),
        "SERVICE_HEARTBEAT_SECONDS": int(cycle_interval_minutes * 60),
        "UI_FORCE_PAPER_ONLY": bool(force_paper_only),
        "UI_SHOW_ADVANCED": bool(show_advanced),
        "USE_COUNCIL": True,
        "COUNCIL_REQUIRE_UNANIMOUS_BUY": True,
        "DYNAMIC_DISCOVERY_ENABLED": bool(dynamic_discovery_ui),
    }
    if preferred_model.strip():
        env_changes["OLLAMA_MODEL"] = preferred_model.strip()
    env_changes["PREFER_GPU"] = bool(use_gpu)

    update_env_values(env_changes)
    update_risk_policy_values(risk_per_trade / 100.0, min_conf, max_positions, allow_live)
    st.success("Saved runtime profile to .env and config/risk_policy.yaml. Restart bot to apply.")
    st.rerun()

if start_live:
    if force_paper_only:
        st.warning("Paper-only safety lock is ON. Disable it first to arm live mode.")
    elif not live_confirm:
        st.warning("Confirm live trading checkbox before arming live mode.")
    else:
        update_env_values(
            {
                "BOT_MODE": "live",
                "LIVE_MODE_UNLOCK": True,
                "REQUIRE_EXPLICIT_LIVE_CONFIRM": True,
                "USE_COUNCIL": True,
            }
        )
        update_risk_policy_values(risk_per_trade / 100.0, min_conf, max_positions, True)
        ok, msg = start_bot_process(cycle_interval_minutes)
        (st.success if ok else st.warning)(f"Live profile armed. {msg}")

if start_bot:
    ok, msg = start_bot_process(cycle_interval_minutes)
    (st.success if ok else st.warning)(msg)

if stop_bot:
    ok, msg = stop_bot_process()
    (st.success if ok else st.warning)(msg)

if sell_all_safe:
    if settings.bot_mode == "live":
        submitted, cancelled, skipped = close_all_live_positions_safe()
        st.success(
            f"Live sell-all complete: submitted exits {submitted}, cancelled live-ready {cancelled}, skipped {skipped}"
        )
    else:
        closed, skipped, pnl = close_all_paper_positions_safe()
        st.success(f"Sell-all complete: closed {closed}, skipped {skipped}, realized PnL ${pnl:+.2f}")

if run_diagnostics:
    snap = collect_debug_snapshot()
    st.success(f"Diagnostics snapshot saved to runtime/diagnostics_latest.json ({len(snap.get('events_24h', {}))} event types).")

health_report = generate_health_report(DB_PATH)
save_health_report(health_report, HEALTH_REPORT_PATH)

current_pid = read_pid()
bot_alive = is_process_alive(current_pid)
if not bot_alive and current_pid:
    BOT_PID_PATH.unlink(missing_ok=True)
    current_pid = None

col1, col2, col3 = st.columns(3)
col1.metric("Current Mode", runtime_mode.upper())
col2.metric("GPU", "Enabled" if use_gpu else "Disabled")
col3.metric("Bot Status", f"Running (PID {current_pid})" if bot_alive else ("Paused" if pause_bot else "Stopped"))

st.caption(f"Paper-only lock: {'ON' if force_paper_only else 'OFF'} | Council: {'ON' if runtime_use_council else 'OFF'}")
st.caption(f"Dynamic discovery: {'ON' if dynamic_discovery_enabled else 'OFF'}")
st.caption(f"Unanimous council BUY required: {'ON' if runtime_unanimous else 'OFF'}")

latest_decision_row = read_table(
    """
    SELECT created_at
    FROM ai_decisions
    ORDER BY id DESC
    LIMIT 1
    """
)
latest_event_row = read_table(
    """
    SELECT event_type, created_at
    FROM system_events
    ORDER BY id DESC
    LIMIT 1
    """
)
latest_fill_row = read_table(
    """
    SELECT created_at
    FROM system_events
    WHERE event_type = 'paper_fill'
    ORDER BY id DESC
    LIMIT 1
    """
)

decision_age_min = None
event_age_min = None
fill_age_min = None
last_event_type = "none"
if not latest_decision_row.empty:
    decision_age_min = minutes_since(str(latest_decision_row.iloc[0]["created_at"] or ""))
if not latest_event_row.empty:
    event_age_min = minutes_since(str(latest_event_row.iloc[0]["created_at"] or ""))
    last_event_type = str(latest_event_row.iloc[0]["event_type"] or "none")
if not latest_fill_row.empty:
    fill_age_min = minutes_since(str(latest_fill_row.iloc[0]["created_at"] or ""))

fresh_window_min = max(5, cycle_interval_minutes * 3)
is_decision_fresh = decision_age_min is not None and decision_age_min <= fresh_window_min
is_event_fresh = event_age_min is not None and event_age_min <= fresh_window_min
runtime_ok = bool(bot_alive and is_event_fresh and is_decision_fresh)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Runtime Health", "OK" if runtime_ok else "Check")
h2.metric("Last Decision Age", f"{decision_age_min:.1f}m" if decision_age_min is not None else "n/a")
h3.metric("Last Event", f"{last_event_type}")
h4.metric("Last Paper Fill Age", f"{fill_age_min:.1f}m" if fill_age_min is not None else "n/a")

hc1, hc2, hc3 = st.columns(3)
hc1.metric("Bot Health Score", f"{int(health_report.get('score', 0))}/100")
hc2.metric("Bot Health Status", str(health_report.get("status", "unknown")))
hc3.metric("Council Decisions (recent)", int(health_report.get("council", {}).get("total_recent_decisions", 0)))

issues = health_report.get("issues", [])
if isinstance(issues, list) and issues:
    st.caption("Health issues: " + ", ".join(str(i) for i in issues[:4]))

if runtime_ok:
    st.success("Engine heartbeat is healthy: process is running and decision/event stream is fresh.")
else:
    st.warning(
        "Engine heartbeat needs attention. If this persists, start/restart the bot and check Debug Console + Bot log tail."
    )

hourly_review: dict[str, Any] = {}
if HOURLY_STRATEGY_REVIEW_PATH.exists():
    try:
        hourly_review = json.loads(HOURLY_STRATEGY_REVIEW_PATH.read_text(encoding="utf-8"))
    except Exception:
        hourly_review = {}

if hourly_review:
    st.markdown("### Hourly Strategy Review")
    meta = hourly_review.get("metadata", {}) if isinstance(hourly_review, dict) else {}
    hr1, hr2, hr3, hr4 = st.columns(4)
    hr1.metric("Strategy Score", f"{int(meta.get('strategy_score', 0) or 0)}/100")
    hr2.metric("Aggressiveness", f"{int(meta.get('aggressiveness_score', 0) or 0)}/100")
    hr3.metric("Informed", f"{int(meta.get('informed_score', 0) or 0)}/100")
    hr4.metric("Agree w/ Council", "YES" if bool(meta.get("agree_with_council", False)) else "NO")
    st.caption(str(hourly_review.get("message", "")))
    notes = meta.get("notes", [])
    if isinstance(notes, list) and notes:
        st.caption("Hourly notes: " + " | ".join(str(x) for x in notes[:4]))
    if show_advanced:
        with st.expander("Hourly strategy review JSON"):
            st.json(hourly_review)

latest_cycle_report: dict[str, Any] = {}
if CYCLE_REPORT_PATH.exists():
    try:
        latest_cycle_report = json.loads(CYCLE_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        latest_cycle_report = {}

st.markdown("### Latest Cycle Findings")
if not latest_cycle_report:
    st.info("No cycle findings report yet. Run the bot once to generate runtime/cycle_report_latest.json.")
else:
    cycle_summary = latest_cycle_report.get("summary", {}) if isinstance(latest_cycle_report, dict) else {}
    cycle_scan = latest_cycle_report.get("scan", {}) if isinstance(latest_cycle_report, dict) else {}
    cycle_status = str(latest_cycle_report.get("status", "unknown"))
    discovery_on = bool(latest_cycle_report.get("dynamic_discovery_enabled", False))
    cycle_decisions = latest_cycle_report.get("decisions", []) if isinstance(latest_cycle_report, dict) else []
    cycle_skipped = latest_cycle_report.get("skipped", []) if isinstance(latest_cycle_report, dict) else []
    scan_top_picks = cycle_scan.get("top_picks", []) if isinstance(cycle_scan, dict) else []

    decisions_total = int(cycle_summary.get("decisions_total", 0) or 0)
    if decisions_total <= 0 and isinstance(cycle_decisions, list):
        decisions_total = len(cycle_decisions)

    skipped_total = int(cycle_summary.get("skipped_total", 0) or 0)
    if skipped_total <= 0 and isinstance(cycle_skipped, list):
        skipped_total = len(cycle_skipped)

    lf1, lf2, lf3, lf4, lf5 = st.columns(5)
    lf1.metric("Cycle Status", cycle_status.upper())
    lf2.metric("Dynamic Discovery", "ON" if discovery_on else "OFF")
    lf3.metric("Bullish Picks Found", int(cycle_scan.get("bullish_pick_count", 0) or 0))
    lf4.metric("Decisioned This Cycle", decisions_total)
    lf5.metric("Skipped by Day-Trade Filter", skipped_total)

    if discovery_on:
        st.caption("Dynamic discovery is enabled for the latest cycle.")
    else:
        st.caption("Dynamic discovery is disabled for the latest cycle.")

    if isinstance(scan_top_picks, list) and scan_top_picks:
        scan_symbols = [str(p.get("symbol", "")).upper() for p in scan_top_picks if isinstance(p, dict)]
        scan_symbols = [s for s in scan_symbols if s]
        if scan_symbols:
            st.caption(
                f"Scanner discovered {len(scan_symbols)} strong movers this cycle: "
                + ", ".join(scan_symbols[:20])
                + (" ..." if len(scan_symbols) > 20 else "")
            )
        scanner_rows: list[dict[str, Any]] = []
        for pick in scan_top_picks[:30]:
            if not isinstance(pick, dict):
                continue
            scanner_rows.append(
                {
                    "symbol": str(pick.get("symbol", "") or "").upper(),
                    "kind": str(pick.get("kind", "") or ""),
                    "trend_score": int(pick.get("score", 0) or 0),
                    "trend_label": str(pick.get("label", "") or ""),
                    "avg_daily_move_pct": float(pick.get("avg_daily_move_pct", 0.0) or 0.0),
                    "last_day_move_pct": float(pick.get("last_day_move_pct", 0.0) or 0.0),
                    "volume_vs_avg": float(pick.get("volume_vs_avg", 0.0) or 0.0),
                }
            )
        if scanner_rows:
            st.markdown("#### Scanner Picks (Latest Cycle)")
            st.dataframe(pd.DataFrame(scanner_rows), use_container_width=True, hide_index=True)

    if skipped_total > 0:
        st.caption(
            "Some scanner picks were intentionally skipped by day-trade suitability rules "
            "(intraday movement/range/volume checks)."
        )

    if isinstance(cycle_decisions, list) and cycle_decisions:
        def _selection_label(value: str) -> str:
            mapping = {
                "scanner_bullish_pick": "Scanner found bullish setup",
                "watchlist_fallback": "Watchlist fallback (scanner had few picks)",
            }
            return mapping.get(value, value)

        findings_rows: list[dict[str, Any]] = []
        for item in cycle_decisions[:30]:
            if not isinstance(item, dict):
                continue
            findings_rows.append(
                {
                    "symbol": item.get("symbol", ""),
                    "kind": item.get("kind", ""),
                    "decision": item.get("action", ""),
                    "confidence": f"{float(item.get('confidence', 0.0) or 0.0):.0%}",
                    "analysis_mode": "Council (3-agent vote)" if bool(item.get("used_council", False)) else "Single/Fast (non-council path)",
                    "selection_reason": _selection_label(str(item.get("selected_by", ""))),
                    "why_preview": truncate_words(str(item.get("rationale", "")), 25),
                }
            )
        table_height = min(1100, 120 + len(findings_rows) * 35)
        st.dataframe(pd.DataFrame(findings_rows), use_container_width=True, hide_index=True, height=table_height)
        st.caption("Analysis mode: Council means full 3-agent vote; Single/Fast means indicator or single-model fallback path.")
        st.markdown("#### Full Why (Latest Cycle)")
        for item in cycle_decisions[:30]:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "") or "?")
            action = str(item.get("action", "") or "").upper()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            rationale = str(item.get("rationale", "") or "No rationale available.")
            with st.expander(f"{symbol} • {action} • {confidence:.0%}"):
                st.write(rationale)

    if show_advanced:
        with st.expander("Latest cycle report JSON"):
            st.json(latest_cycle_report)

if show_advanced:
    with st.expander("Latest health check report"):
        st.json(health_report)

st.markdown("### Capital & Funding")
fund_client = RobinhoodClient()
fund_snapshot = fund_client.get_account_snapshot()
portfolio_totals = fund_client.get_portfolio_totals()
live_bp = float(fund_snapshot.get("buying_power", 0.0) or 0.0)
live_cash = float(fund_snapshot.get("cash", 0.0) or 0.0)
live_withdrawable = float(fund_snapshot.get("cash_available_for_withdrawal", 0.0) or 0.0)
live_held = float(fund_snapshot.get("cash_held_for_orders", 0.0) or 0.0)
live_equity_total = float(portfolio_totals.get("equity", 0.0) or 0.0)
live_market_value = float(portfolio_totals.get("market_value", 0.0) or 0.0)
live_prev_close = float(portfolio_totals.get("portfolio_equity_previous_close", 0.0) or 0.0)
live_day_change = live_equity_total - live_prev_close if live_prev_close > 0 else 0.0

paper_base_cap = float(runtime_paper_cap or 0.0)
paper_wiggle = float(runtime_paper_wiggle or 0.0)
paper_cap = paper_base_cap + paper_wiggle
paper_open_row = read_table(
    """
    SELECT COALESCE(SUM(quantity * price), 0.0) AS open_notional
    FROM orders
    WHERE mode = 'paper' AND side = 'buy' AND status IN ('paper_proposed', 'paper_filled')
    """
)
paper_open = float(paper_open_row.iloc[0]["open_notional"] if not paper_open_row.empty else 0.0)
paper_available = max(0.0, paper_cap - paper_open) if paper_cap > 0 else 0.0
paper_realized_row_cap = read_table("SELECT COALESCE(SUM(pnl), 0.0) AS realized_pnl FROM trade_outcomes")
paper_realized_cap = float(paper_realized_row_cap.iloc[0]["realized_pnl"] if not paper_realized_row_cap.empty else 0.0)
paper_equity_sim = max(0.0, paper_cap + paper_realized_cap)
paper_buying_power_sim = max(0.0, paper_equity_sim - paper_open)

live_daily_used = get_live_daily_notional_used()
live_daily_remaining = max(0.0, float(settings.live_max_daily_notional_usd) - live_daily_used)
live_max_deployable_now = min(live_bp, live_daily_remaining)

f1, f2, f3, f4 = st.columns(4)
f1.metric("Investing Buying Power", f"${live_bp:.2f}")
f2.metric("Investing Cash", f"${live_cash:.2f}")
f3.metric("Withdrawable Cash", f"${live_withdrawable:.2f}")
f4.metric("Cash Held for Orders", f"${live_held:.2f}")

f9, f10, f11 = st.columns(3)
f9.metric("Investing Total Equity", f"${live_equity_total:.2f}", delta=f"${live_day_change:+.2f}")
f10.metric("Investing Market Value", f"${live_market_value:.2f}")
f11.metric("Prev Close Equity", f"${live_prev_close:.2f}")

f5, f6, f7, f8 = st.columns(4)
f5.metric("Paper Capital Cap (effective)", f"${paper_cap:.2f}")
f6.metric("Paper Buying Power (sim)", f"${paper_buying_power_sim:.2f}")
f7.metric("Live Daily Notional Remaining", f"${live_daily_remaining:.2f}")
f8.metric("Max Deployable Now (Live)", f"${live_max_deployable_now:.2f}")
st.caption(f"Paper cap breakdown: base ${paper_base_cap:.2f} + wiggle ${paper_wiggle:.2f} = effective ${paper_cap:.2f}")
st.caption(f"Paper simulated equity ${paper_equity_sim:.2f} = cap ${paper_cap:.2f} + realized PnL ${paper_realized_cap:+.2f}; open notional ${paper_open:.2f}.")

st.caption(
    "Max deployable now (live) is bounded by Robinhood buying power and your configured daily live notional cap."
)

transfers = fund_client.get_unified_transfers(limit=25)
if transfers:
    transfer_rows: list[dict[str, Any]] = []
    for item in transfers:
        transfer_rows.append(
            {
                "created_at": item.get("created_at", ""),
                "state": item.get("state", ""),
                "direction": item.get("direction", ""),
                "transfer_type": item.get("transfer_type", ""),
                "amount": item.get("amount", ""),
                "origin": item.get("originating_account_type", ""),
                "destination": item.get("receiving_account_type", ""),
                "description": item.get("description", ""),
            }
        )
    with st.expander("Recent transfer activity"):
        st.dataframe(pd.DataFrame(transfer_rows), use_container_width=True, hide_index=True)

st.markdown("### Council Activity")
council_recent = read_table(
    """
    SELECT symbol, decision_json, created_at
    FROM ai_decisions
    WHERE model_name LIKE '%council%'
    ORDER BY id DESC
    LIMIT 1000
    """
)

council_total_row = read_table(
    """
    SELECT COUNT(1) AS total
    FROM ai_decisions
    WHERE model_name LIKE '%council%'
    """
)
council_total = int(council_total_row.iloc[0]["total"] if not council_total_row.empty else 0)

council_24h_row = read_table(
    """
    SELECT COUNT(1) AS total
    FROM ai_decisions
    WHERE model_name LIKE '%council%'
      AND created_at >= datetime('now', '-1 day')
    """
)
council_24h = int(council_24h_row.iloc[0]["total"] if not council_24h_row.empty else 0)

if council_recent.empty:
    st.info("No council decisions yet.")
else:
    buy_votes = hold_votes = avoid_votes = 0
    council_rows: list[dict[str, Any]] = []
    voted_symbols: list[str] = []
    for _, row in council_recent.iterrows():
        try:
            payload = json.loads(str(row.get("decision_json") or "{}"))
            decision = payload.get("decision", payload)
            action = str(decision.get("action", "")).upper()
            symbol = str(row.get("symbol", "") or "").upper()
            voted_symbols.append(symbol)
            council_rows.append(
                {
                    "created_at": str(row.get("created_at", "") or ""),
                    "symbol": symbol,
                    "final_action": action,
                    "votes": str(decision.get("council_votes", "")),
                    "unanimous": bool(decision.get("council_unanimous", False)),
                    "confidence": f"{float(decision.get('confidence', 0.0) or 0.0):.0%}",
                }
            )
            if action == "BUY":
                buy_votes += 1
            elif action == "HOLD":
                hold_votes += 1
            else:
                avoid_votes += 1
        except Exception:
            continue

    last_council_ts = str(council_recent.iloc[0].get("created_at") or "")
    mins = minutes_since(last_council_ts)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Council Decisions (total)", council_total)
    c2.metric("Council Decisions (24h)", council_24h)
    c3.metric("BUY", buy_votes)
    c4.metric("HOLD", hold_votes)
    c5.metric("AVOID", avoid_votes)
    if mins is not None:
        st.caption(f"Last council vote: {mins:.1f} minutes ago")
    st.caption(f"Showing latest {len(council_recent.index)} council records in the table below.")

    unique_symbols = sorted({s for s in voted_symbols if s})
    st.markdown("#### Symbols Council Voted On (recent)")
    if unique_symbols:
        st.caption(
            f"{len(unique_symbols)} unique symbols in recent council votes: "
            + ", ".join(unique_symbols[:24])
            + (" ..." if len(unique_symbols) > 24 else "")
        )
    if council_rows:
        st.dataframe(pd.DataFrame(council_rows[:30]), use_container_width=True, hide_index=True)

    # Show latest council member votes + short rationale
    latest_payload = json.loads(str(council_recent.iloc[0].get("decision_json") or "{}"))
    latest_decision = latest_payload.get("decision", {}) if isinstance(latest_payload, dict) else {}
    latest_raw = latest_payload.get("raw", {}) if isinstance(latest_payload, dict) else {}
    latest_votes = latest_raw.get("votes", []) if isinstance(latest_raw, dict) else []
    fallback_reasons = extract_agent_reasons_from_summary(str(latest_decision.get("rationale", "")))
    if isinstance(latest_votes, list) and latest_votes:
        st.markdown("#### Last Council Vote Breakdown")
        vote_cols = st.columns(min(3, max(1, len(latest_votes))))
        agent_purpose = {
            "TECHNICIAN": "Technical indicators and price action only",
            "SENTINEL": "News, sentiment, and market mood only",
            "STRATEGIST": "Risk/reward and portfolio fit using both",
        }
        for idx, vote in enumerate(latest_votes[:3]):
            if not isinstance(vote, dict):
                continue
            agent = str(vote.get("agent", "AGENT"))
            action = str(vote.get("action", "")).upper()
            conf = float(vote.get("confidence", 0.0) or 0.0)
            agent_key = agent.upper()
            raw_reason = str(vote.get("rationale", "") or "").strip()
            fallback_reason = fallback_reasons.get(agent_key, "")
            reason = truncate_words(raw_reason or fallback_reason or "No rationale captured for this vote yet.", 200)
            purpose = agent_purpose.get(agent_key, "Specialist analysis")
            vote_cols[idx % len(vote_cols)].markdown(
                f"""
                <div class='card'>
                  <b>{agent}</b><br/>
                  <span style='opacity:0.85'><i>{purpose}</i></span><br/>
                  Vote: <b>{action}</b> | Confidence: {conf:.0%}<br/>
                  <span style='opacity:0.9'>{reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

log_tail = ""
if BOT_LOG_PATH.exists():
    try:
        log_lines = BOT_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        log_tail = "\n".join(log_lines[-12:])
    except Exception:
        log_tail = ""

if log_tail:
    with st.expander("Bot log tail"):
        st.code(log_tail, language="text")

st.markdown("### Open Positions (Live Mark-to-Market)")
open_positions = load_open_positions_snapshot()
paper_realized_row = read_table("SELECT COALESCE(SUM(pnl), 0.0) AS realized_pnl FROM trade_outcomes")
paper_realized_total = float(paper_realized_row.iloc[0]["realized_pnl"] if not paper_realized_row.empty else 0.0)
paper_start_value = float((paper_base_cap + paper_wiggle) if paper_base_cap > 0 else settings.paper_starting_equity)
paper_unreal_total = 0.0
paper_total_value = paper_start_value + paper_realized_total
if open_positions.empty:
    st.info("No open paper positions right now.")
else:
    total_entry = float(open_positions["entry_notional"].sum())
    total_now = float(open_positions["current_notional"].sum())
    total_unreal = float(open_positions["unrealized_pnl"].sum())
    paper_unreal_total = total_unreal
    paper_total_value = paper_start_value + paper_realized_total + paper_unreal_total
    total_pct = (total_unreal / total_entry) if total_entry > 0 else 0.0

    op1, op2, op3, op4 = st.columns(4)
    op1.metric("Open Positions", int(len(open_positions.index)))
    op2.metric("Entry Notional", f"${total_entry:.2f}")
    op3.metric("Current Value", f"${total_now:.2f}")
    op4.metric("Unrealized PnL", f"${total_unreal:+.2f}", delta=f"{total_pct:+.2%}")

    card_cols = st.columns(min(3, max(1, len(open_positions.index))))
    for idx, (_, row) in enumerate(open_positions.head(6).iterrows()):
        pnl = float(row["unrealized_pnl"])
        pnl_pct = float(row["unrealized_pnl_pct"])
        card_cols[idx % len(card_cols)].markdown(
            f"""
            <div class='card'>
              <b>{row['symbol']}</b> <span style='opacity:0.75'>({row['kind']})</span><br/>
              Qty: {float(row['quantity']):.4f}<br/>
              Entry: ${float(row['entry_price']):.4f} | Mark: ${float(row['mark_price']):.4f}<br/>
              <b>Unrealized: ${pnl:+.2f} ({pnl_pct:+.2%})</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    display_cols = [
        "symbol", "kind", "quantity", "entry_price", "mark_price",
        "entry_notional", "current_notional", "unrealized_pnl", "unrealized_pnl_pct", "has_live_price",
    ]
    st.dataframe(open_positions[display_cols], use_container_width=True, hide_index=True)

paper_growth_pct = ((paper_total_value / paper_start_value) - 1.0) if paper_start_value > 0 else 0.0
pv1, pv2, pv3, pv4 = st.columns(4)
pv1.metric("Paper Start Value", f"${paper_start_value:.2f}")
pv2.metric("Paper Realized PnL", f"${paper_realized_total:+.2f}")
pv3.metric("Paper Unrealized PnL", f"${paper_unreal_total:+.2f}")
pv4.metric("Paper Total Value", f"${paper_total_value:.2f}", delta=f"{paper_growth_pct:+.2%}")

perf_stats = read_table(
    """
    SELECT
      COUNT(1) AS total_closed,
      COALESCE(SUM(pnl), 0.0) AS realized_pnl,
      COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
    FROM trade_outcomes
    """
)

if not perf_stats.empty:
    total_closed = int(perf_stats.iloc[0]["total_closed"] or 0)
    realized_pnl = float(perf_stats.iloc[0]["realized_pnl"] or 0.0)
    wins = int(perf_stats.iloc[0]["wins"] or 0)
    win_rate = (wins / total_closed) if total_closed else 0.0
    p1, p2, p3 = st.columns(3)
    p1.metric("Closed Paper Trades", total_closed)
    p2.metric("Realized PnL", f"{realized_pnl:.4f}")
    p3.metric("Win Rate", f"{win_rate:.1%}")

if paper_base_cap > 0:
    paper_cap = float(paper_base_cap + paper_wiggle)
    bankroll_now = paper_cap + float(realized_pnl if "realized_pnl" in locals() else 0.0)
    b1, b2, b3 = st.columns(3)
    b1.metric("Paper Working Capital", f"${paper_cap:.2f}")
    b2.metric("Current Paper Bankroll", f"${bankroll_now:.2f}")
    b3.metric("Growth vs Start", f"{((bankroll_now / paper_cap) - 1.0):.1%}" if paper_cap > 0 else "0.0%")

weekly_card = read_table(
    """
    SELECT
      COUNT(1) AS trades_7d,
      COALESCE(SUM(pnl), 0.0) AS pnl_7d,
      COALESCE(AVG(hold_minutes), 0.0) AS avg_hold_minutes,
      COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins_7d
    FROM trade_outcomes
    WHERE closed_at >= datetime('now', '-7 days')
    """
)

if not weekly_card.empty:
    trades_7d = int(weekly_card.iloc[0]["trades_7d"] or 0)
    wins_7d = int(weekly_card.iloc[0]["wins_7d"] or 0)
    weekly_win_rate = (wins_7d / trades_7d) if trades_7d else 0.0
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Weekly Trades", trades_7d)
    w2.metric("Weekly PnL", f"{float(weekly_card.iloc[0]['pnl_7d'] or 0.0):.4f}")
    w3.metric("Weekly Win Rate", f"{weekly_win_rate:.1%}")
    w4.metric("Avg Hold (min)", f"{float(weekly_card.iloc[0]['avg_hold_minutes'] or 0.0):.1f}")

st.markdown("### Earnings Goals")
goal_daily_row = read_table(
    """
    SELECT COALESCE(SUM(pnl), 0.0) AS pnl
    FROM trade_outcomes
    WHERE closed_at >= datetime('now', 'start of day')
    """
)
goal_weekly_row = read_table(
    """
    SELECT COALESCE(SUM(pnl), 0.0) AS pnl
    FROM trade_outcomes
    WHERE closed_at >= datetime('now', '-7 days')
    """
)

daily_pnl = float(goal_daily_row.iloc[0]["pnl"] if not goal_daily_row.empty else 0.0)
weekly_pnl = float(goal_weekly_row.iloc[0]["pnl"] if not goal_weekly_row.empty else 0.0)
daily_goal = float(runtime_daily_goal)
weekly_goal = float(runtime_weekly_goal)
daily_ratio = (daily_pnl / daily_goal) if daily_goal > 0 else 0.0
weekly_ratio = (weekly_pnl / weekly_goal) if weekly_goal > 0 else 0.0

g1, g2 = st.columns(2)
g1.metric("Daily Goal", f"${daily_goal:.2f}", delta=f"PnL ${daily_pnl:.2f}")
g2.metric("Weekly Goal", f"${weekly_goal:.2f}", delta=f"PnL ${weekly_pnl:.2f}")

if daily_goal > 0:
    st.caption(f"Daily progress: {daily_ratio:.1%}")
    st.progress(min(max(daily_ratio, 0.0), 1.0))
if weekly_goal > 0:
    st.caption(f"Weekly progress: {weekly_ratio:.1%}")
    st.progress(min(max(weekly_ratio, 0.0), 1.0))

daily_met = daily_goal > 0 and daily_pnl >= daily_goal
weekly_met = weekly_goal > 0 and weekly_pnl >= weekly_goal
if daily_met or weekly_met:
    st.success("Goal target reached. New orders may be paused based on policy.")

latest_goal_event = read_table(
    """
    SELECT metadata_json
    FROM system_events
    WHERE event_type = 'goal_progress'
    ORDER BY id DESC
    LIMIT 1
    """
)
if not latest_goal_event.empty:
    try:
        payload = json.loads(str(latest_goal_event.iloc[0]["metadata_json"] or "{}"))
        rmult = float(payload.get("risk_multiplier", 1.0))
        dratio = float(payload.get("daily_ratio", 1.0))
        wratio = float(payload.get("weekly_ratio", 1.0))
        preset = str(payload.get("preset", "balanced")).lower()
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Pacing Multiplier", f"x{rmult:.2f}")
        p2.metric("Daily Pace Ratio", f"{dratio:.2f}")
        p3.metric("Weekly Pace Ratio", f"{wratio:.2f}")
        p4.metric("Pacing Preset", preset)
    except Exception:
        pass

if not show_advanced:
    st.caption("Advanced tables are hidden. Turn on 'Show advanced data tables' in the sidebar when you want deeper diagnostics.")
    if auto_refresh > 0:
        st.caption("Auto-refresh is disabled in low-resource mode. Use 'Refresh data now' to avoid unnecessary RAM/CPU usage.")
    if refresh_now:
        st.rerun()
    st.stop()

st.markdown("### Recent AI Decisions")


decisions = read_table(
    """
    SELECT symbol, model_name, score, confidence, rationale, created_at
    FROM ai_decisions
    ORDER BY id DESC
    LIMIT 50
    """
)

if decisions.empty:
    st.info("No AI decisions yet. Start the bot service and wait for the first cycle.")
else:
    st.dataframe(decisions, use_container_width=True, hide_index=True)

st.markdown("### Proposed Orders")
orders = read_table(
    """
    SELECT symbol, mode, side, quantity, price, status, rationale, created_at
    FROM orders
    ORDER BY id DESC
    LIMIT 50
    """
)

if orders.empty:
    st.info("No orders proposed yet.")
else:
    st.dataframe(orders, use_container_width=True, hide_index=True)

st.markdown("### Trade Outcomes")
outcomes = read_table(
    """
    SELECT order_id, pnl, hold_minutes, max_drawdown, closed_at
    FROM trade_outcomes
    ORDER BY id DESC
    LIMIT 50
    """
)

if outcomes.empty:
    st.info("No trade outcomes yet.")
else:
    st.dataframe(outcomes, use_container_width=True, hide_index=True)

st.markdown("### Symbol Hit Rate")
symbol_hit_rate = read_table(
    """
    SELECT o.symbol,
           COUNT(1) AS trades,
           COALESCE(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
           COALESCE(SUM(t.pnl), 0.0) AS pnl
    FROM trade_outcomes t
    JOIN orders o ON o.id = t.order_id
    GROUP BY o.symbol
    ORDER BY trades DESC, pnl DESC
    LIMIT 20
    """
)

if symbol_hit_rate.empty:
    st.info("No per-symbol outcomes yet.")
else:
    symbol_hit_rate["win_rate"] = symbol_hit_rate.apply(
        lambda row: (row["wins"] / row["trades"]) if row["trades"] else 0.0,
        axis=1,
    )
    st.dataframe(symbol_hit_rate, use_container_width=True, hide_index=True)

st.markdown("### Confidence Calibration")
confidence_calibration = read_table(
    """
    SELECT
      CASE
        WHEN o.decision_confidence < 0.50 THEN '0.00-0.49'
        WHEN o.decision_confidence < 0.60 THEN '0.50-0.59'
        WHEN o.decision_confidence < 0.70 THEN '0.60-0.69'
        WHEN o.decision_confidence < 0.80 THEN '0.70-0.79'
        ELSE '0.80-1.00'
      END AS bucket,
      COUNT(1) AS trades,
      COALESCE(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
      COALESCE(SUM(t.pnl), 0.0) AS pnl
    FROM trade_outcomes t
    JOIN orders o ON o.id = t.order_id
    WHERE o.decision_confidence IS NOT NULL
    GROUP BY bucket
    ORDER BY bucket
    """
)

if confidence_calibration.empty:
    st.info("No confidence calibration data yet.")
else:
    confidence_calibration["win_rate"] = confidence_calibration.apply(
        lambda row: (row["wins"] / row["trades"]) if row["trades"] else 0.0,
        axis=1,
    )
    st.dataframe(confidence_calibration, use_container_width=True, hide_index=True)

st.markdown("### Daily Reviews")
reviews = read_table(
    """
    SELECT review_date, summary, lessons, created_at
    FROM daily_reviews
    ORDER BY id DESC
    LIMIT 14
    """
)

if reviews.empty:
    st.info("No daily reviews generated yet.")
else:
    st.dataframe(reviews, use_container_width=True, hide_index=True)

st.markdown("### System Events")
events = read_table(
    """
    SELECT event_type, message, created_at
    FROM system_events
    ORDER BY id DESC
    LIMIT 25
    """
)

if events.empty:
    st.info("No events recorded yet.")
else:
    st.dataframe(events, use_container_width=True, hide_index=True)

st.markdown("### Inference Resilience")
fallback_events = read_table(
    """
    SELECT message, metadata_json, created_at
    FROM system_events
    WHERE event_type = 'inference_fallback'
    ORDER BY id DESC
    LIMIT 20
    """
)

col_a, col_b = st.columns(2)
col_a.metric("Fallback Events (last 20)", len(fallback_events.index))
col_b.metric("Configured Max Positions", max_positions)

if fallback_events.empty:
    st.success("No fallback events recorded recently.")
else:
    st.dataframe(fallback_events, use_container_width=True, hide_index=True)

st.markdown("### Alerts")
alerts = read_table(
    """
    SELECT event_type, message, metadata_json, created_at
    FROM system_events
        WHERE event_type IN (
            'drawdown_alert',
            'quality_alert',
            'order_blocked_live_cap',
            'order_blocked_goal',
            'kill_switch',
            'pipeline_error'
        )
    ORDER BY id DESC
    LIMIT 20
    """
)
if alerts.empty:
    st.success("No execution alerts in recent history.")
else:
    st.dataframe(alerts, use_container_width=True, hide_index=True)

st.markdown("### Debug Console")
debug_events = read_table(
    """
    SELECT event_type, COUNT(1) AS n
    FROM system_events
    WHERE created_at >= datetime('now', '-1 day')
    GROUP BY event_type
    ORDER BY n DESC
    LIMIT 30
    """
)

pipeline_errors_24h = 0
fallbacks_24h = 0
skipped_live_data_24h = 0
order_blocked_24h = 0
if not debug_events.empty:
    event_count_map = {str(r["event_type"]): int(r["n"]) for _, r in debug_events.iterrows()}
    pipeline_errors_24h = int(event_count_map.get("pipeline_error", 0))
    fallbacks_24h = int(event_count_map.get("inference_fallback", 0))
    skipped_live_data_24h = int(event_count_map.get("symbol_skipped_no_live_data", 0))
    order_blocked_24h = int(event_count_map.get("order_blocked", 0)) + int(event_count_map.get("order_blocked_live_cap", 0))

dc1, dc2, dc3, dc4 = st.columns(4)
dc1.metric("Pipeline Errors (24h)", pipeline_errors_24h)
dc2.metric("Inference Fallbacks (24h)", fallbacks_24h)
dc3.metric("Skipped No-Live-Data (24h)", skipped_live_data_24h)
dc4.metric("Order Blocks (24h)", order_blocked_24h)

dynamic_mode_events = int(event_count_map.get("trend_scan_mode", 0)) if not debug_events.empty else 0
dynamic_fail_events = int(event_count_map.get("trend_scan_failed", 0)) if not debug_events.empty else 0
ds1, ds2 = st.columns(2)
ds1.metric("Dynamic Scan Events (24h)", dynamic_mode_events)
ds2.metric("Dynamic Scan Failures (24h)", dynamic_fail_events)

# Health score (paper-to-live readiness heuristic)
health_score = 100
penalties: list[str] = []

if pipeline_errors_24h > 0:
    penalty = min(40, pipeline_errors_24h * 8)
    health_score -= penalty
    penalties.append(f"pipeline errors -{penalty}")

if fallbacks_24h > 20:
    health_score -= 20
    penalties.append("many inference fallbacks -20")
elif fallbacks_24h > 5:
    health_score -= 10
    penalties.append("elevated inference fallbacks -10")

if skipped_live_data_24h > 20:
    health_score -= 10
    penalties.append("too many no-live-data skips -10")

if order_blocked_24h > 25:
    health_score -= 10
    penalties.append("frequent order blocks -10")

last_decision = read_table(
    """
    SELECT created_at
    FROM ai_decisions
    ORDER BY id DESC
    LIMIT 1
    """
)
last_decision_minutes = None
if not last_decision.empty:
    last_decision_minutes = minutes_since(str(last_decision.iloc[0]["created_at"] or ""))

if not bot_alive:
    health_score -= 25
    penalties.append("bot not running -25")

if bot_alive and last_decision_minutes is not None and last_decision_minutes > (cycle_interval_minutes * 3):
    health_score -= 20
    penalties.append("decision stream stale -20")

if "total_closed" in locals() and "win_rate" in locals():
    if int(total_closed) >= 10 and float(win_rate) < 0.45:
        health_score -= 15
        penalties.append("win rate below quality threshold -15")

health_score = max(0, min(100, int(health_score)))
status_label = "Healthy" if health_score >= 85 else ("Watch" if health_score >= 65 else "Risky")

hs1, hs2 = st.columns([1, 2])
hs1.metric("Health Score", f"{health_score}/100", delta=status_label)
hs2.progress(health_score / 100)
if penalties:
    st.caption("Health penalties: " + ", ".join(penalties))
else:
    st.caption("No health penalties detected in the last 24h.")

if bot_alive and last_decision_minutes is not None and last_decision_minutes > (cycle_interval_minutes * 3):
    st.warning(f"Bot is running but AI decisions are stale ({last_decision_minutes:.1f} min old). Check logs and API connectivity.")

if DEBUG_SNAPSHOT_PATH.exists():
    try:
        snap_json = json.loads(DEBUG_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        with st.expander("Latest diagnostics snapshot"):
            st.json(snap_json)
    except Exception:
        st.caption("Diagnostics snapshot exists but could not be parsed.")

if not debug_events.empty:
    st.dataframe(debug_events, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# BACKTEST RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🧪 Backtesting Results")

backtest_runs = read_table(
    """
    SELECT run_id, started_at, starting_capital, final_equity, total_pnl,
           win_count, loss_count, win_rate, max_drawdown_pct, sharpe_estimate,
           symbols_tested, candles_processed, ai_calls, duration_seconds
    FROM backtest_runs
    ORDER BY id DESC
    LIMIT 20
    """
)

if backtest_runs.empty:
    st.info("No backtest runs yet. Run `scripts/run_backtest.ps1` to start one.")
else:
    # summary metrics for the latest run
    latest = backtest_runs.iloc[0]
    bt1, bt2, bt3, bt4 = st.columns(4)
    bt1.metric("Latest PnL", f"${float(latest['total_pnl'] or 0):.2f}")
    bt2.metric("Final Equity", f"${float(latest['final_equity'] or 0):.2f}")
    wr = float(latest["win_rate"] or 0)
    bt3.metric("Win Rate", f"{wr:.1%}")
    bt4.metric("Max Drawdown", f"{float(latest['max_drawdown_pct'] or 0):.1%}")

    bt5, bt6, bt7, bt8 = st.columns(4)
    bt5.metric("Wins / Losses", f"{int(latest['win_count'] or 0)} / {int(latest['loss_count'] or 0)}")
    bt6.metric("AI Calls", int(latest["ai_calls"] or 0))
    bt7.metric("Candles Processed", int(latest["candles_processed"] or 0))
    bt8.metric("Duration", f"{float(latest['duration_seconds'] or 0):.0f}s")

    st.markdown("#### Run History")
    display_cols = [
        "run_id", "started_at", "starting_capital", "final_equity",
        "total_pnl", "win_rate", "max_drawdown_pct", "sharpe_estimate",
        "duration_seconds",
    ]
    st.dataframe(
        backtest_runs[[c for c in display_cols if c in backtest_runs.columns]],
        use_container_width=True,
        hide_index=True,
    )

    # per-trade detail for latest run
    latest_run_id = str(latest["run_id"])
    backtest_trades = read_table(
        f"""
        SELECT symbol, kind, side, entry_price, exit_price, quantity,
               gross_pnl, fees, net_pnl, confidence,
               entry_timestamp, exit_timestamp, rationale
        FROM backtest_trades
        WHERE run_id = '{latest_run_id}'
        ORDER BY id ASC
        """
    )

    if not backtest_trades.empty:
        st.markdown(f"#### Trades — Run `{latest_run_id}`")
        st.dataframe(backtest_trades, use_container_width=True, hide_index=True)

        # per-symbol breakdown
        st.markdown("#### Per-Symbol Performance")
        symbol_summary = backtest_trades.groupby("symbol").agg(
            trades=("net_pnl", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            wins=("net_pnl", lambda x: (x > 0).sum()),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        symbol_summary["win_rate"] = symbol_summary.apply(
            lambda r: (r["wins"] / r["trades"]) if r["trades"] else 0.0, axis=1
        )
        symbol_summary = symbol_summary.sort_values("total_pnl", ascending=False)
        st.dataframe(symbol_summary, use_container_width=True, hide_index=True)

        # equity curve chart
        st.markdown("#### Equity Curve")
        equity_series = [float(latest["starting_capital"] or 500)]
        for _, row in backtest_trades.iterrows():
            equity_series.append(equity_series[-1] + float(row["net_pnl"] or 0))
        chart_df = pd.DataFrame({"Equity ($)": equity_series})
        st.line_chart(chart_df, use_container_width=True)
    else:
        st.info("No individual trades found for the latest run.")

if auto_refresh > 0:
    st.caption(
        "Auto-refresh is disabled in low-resource mode. Use 'Refresh data now' to avoid unnecessary RAM/CPU usage."
    )

if refresh_now:
    st.rerun()
