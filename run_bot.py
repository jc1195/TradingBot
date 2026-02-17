"""
  ============================================
   TRADING BOT -- One-Click Launcher
  ============================================

  This is the single entry point to start the bot.
  It runs paper trading continuously with the 3-agent AI council.

  Usage:
    python run_bot.py                # continuous paper trading
    python run_bot.py --once         # single cycle then exit
        python run_bot.py --interval 5   # custom cycle interval (minutes)

  The bot will:
    1. Scan the market for bullish opportunities
    2. Run the 3-agent AI council on the top picks
    3. Make fast indicator decisions on remaining picks
    4. Execute paper trades with smart exits (TP/SL)
    5. Track progress toward the $500 -> $1000 weekly goal
    6. Repeat every N minutes
"""

import argparse
import os
import signal
import sys
import time
import warnings

# ── Encoding & warnings ──────────────────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trading_bot.db import initialize_database, get_connection
from src.trading_bot.engine import TradingEngine
from src.trading_bot.risk import load_risk_policy
from src.trading_bot.settings import settings


# ── Safe shutdown ─────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n[SHUTDOWN] Graceful stop requested... finishing current cycle.\n")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def get_portfolio_status() -> dict:
    """Pull current portfolio state from the database."""
    cap = float(settings.paper_working_capital_usd) or 500.0
    with get_connection() as conn:
        # Realized PnL
        pnl_row = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) AS pnl FROM trade_outcomes"
        ).fetchone()
        realized_pnl = float(pnl_row["pnl"]) if pnl_row else 0.0

        # Open positions
        open_rows = conn.execute(
            """SELECT symbol, quantity, price,
                      ROUND(quantity * price, 2) AS notional,
                      created_at
               FROM orders
               WHERE mode = 'paper' AND status = 'paper_filled'
               ORDER BY created_at DESC"""
        ).fetchall()

        open_notional = sum(float(r["notional"]) for r in open_rows)

        # Trade count today
        today_row = conn.execute(
            """SELECT COUNT(1) AS cnt, COALESCE(SUM(pnl), 0.0) AS pnl
               FROM trade_outcomes
               WHERE closed_at >= date('now', 'start of day')"""
        ).fetchone()

        # Trade count this week
        week_row = conn.execute(
            """SELECT COUNT(1) AS cnt, COALESCE(SUM(pnl), 0.0) AS pnl
               FROM trade_outcomes
               WHERE closed_at >= date('now', 'weekday 0', '-7 days')"""
        ).fetchone()

        # Win rate
        stats = conn.execute(
            """SELECT COUNT(1) AS total,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
               FROM trade_outcomes"""
        ).fetchone()

    equity = max(0.0, cap + realized_pnl)
    total_trades = int(stats["total"]) if stats else 0
    wins = int(stats["wins"]) if stats else 0

    return {
        "capital": cap,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "open_notional": open_notional,
        "buying_power": max(0.0, equity - open_notional),
        "open_positions": open_rows,
        "today_trades": int(today_row["cnt"]) if today_row else 0,
        "today_pnl": float(today_row["pnl"]) if today_row else 0.0,
        "week_trades": int(week_row["cnt"]) if week_row else 0,
        "week_pnl": float(week_row["pnl"]) if week_row else 0.0,
        "total_trades": total_trades,
        "win_rate": wins / total_trades if total_trades > 0 else 0.0,
        "goal": 1000.0,
        "goal_progress": equity / 1000.0,
    }


def print_status(status: dict, cycle: int) -> None:
    """Print a compact portfolio status dashboard to terminal."""
    eq = status["equity"]
    goal = status["goal"]
    progress = status["goal_progress"]
    bar_len = 30
    filled = int(progress * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)

    print()
    print("=" * 62)
    print(f"  PORTFOLIO STATUS  (cycle {cycle})")
    print("=" * 62)
    print(f"  Capital:     ${status['capital']:.2f}")
    print(f"  Equity:      ${eq:.2f}  {'(UP)' if eq > status['capital'] else '(DOWN)' if eq < status['capital'] else ''}")
    print(f"  Open:        ${status['open_notional']:.2f} across {len(status['open_positions'])} positions")
    print(f"  Buying Power:${status['buying_power']:.2f}")
    print(f"  Total PnL:   ${status['realized_pnl']:+.2f}")
    print(f"  Win Rate:    {status['win_rate']:.0%} ({status['total_trades']} trades)")
    print(f"  Today:       ${status['today_pnl']:+.2f} ({status['today_trades']} trades)")
    print(f"  This Week:   ${status['week_pnl']:+.2f} ({status['week_trades']} trades)")
    print()
    print(f"  GOAL: $500 -> $1000")
    print(f"  [{bar}] {progress:.0%}")
    print(f"  Need ${max(0, goal - eq):.2f} more")
    print()

    if status["open_positions"]:
        print("  OPEN POSITIONS:")
        for pos in status["open_positions"]:
            sym = pos["symbol"]
            qty = float(pos["quantity"])
            price = float(pos["price"])
            notional = float(pos["notional"])
            print(f"    {sym:8s}  {qty:.4f} @ ${price:.4f}  (${notional:.2f})")
        print()

    print("=" * 62)


def run_bot(once: bool = False, interval_minutes: int = 5):
    """Main bot loop."""
    print()
    print("  ========================================")
    print("   TRADING BOT v2.0 -- AI Council Mode")
    print("  ========================================")
    print(f"  Mode:       {settings.bot_mode.upper()}")
    print(f"  Council:    {'ON' if settings.use_council else 'OFF'}")
    print(f"  Capital:    ${settings.paper_working_capital_usd:.0f}")
    print(f"  Goal:       $500 -> $1000 (weekly)")
    print(f"  Interval:   {interval_minutes} min between cycles")
    print(f"  Risk/trade: {settings.max_risk_per_trade_pct:.0%}")
    print(f"  Max pos:    {settings.max_concurrent_positions}")
    print(f"  AI model:   {settings.ollama_model or 'auto'}")
    print()

    initialize_database()
    engine = TradingEngine(load_risk_policy())
    cycle = 0

    while not _shutdown_requested:
        cycle += 1
        cycle_start = time.time()

        print(f"\n{'='*62}")
        print(f"  CYCLE {cycle}  --  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*62}")

        try:
            engine.run_daily_cycle()
        except Exception as exc:
            print(f"\n  [ERROR] Cycle {cycle} failed: {exc}")
            import traceback
            traceback.print_exc()

        # Show status
        try:
            status = get_portfolio_status()
            print_status(status, cycle)

            # Check if goal is hit
            if status["equity"] >= status["goal"]:
                print("  *** GOAL REACHED! $500 -> $1000 ***")
                print("  The bot has paused new orders.")
                print()
        except Exception:
            pass

        elapsed = time.time() - cycle_start
        print(f"  Cycle {cycle} took {elapsed:.0f}s")

        if once:
            print("\n  Single cycle mode -- exiting.")
            break

        if _shutdown_requested:
            break

        wait_seconds = interval_minutes * 60
        print(f"\n  Next cycle in {interval_minutes} min (at {time.strftime('%H:%M', time.localtime(time.time() + wait_seconds))})")
        print("  Press Ctrl+C to stop gracefully.\n")

        # Sleep in small chunks so we can respond to Ctrl+C quickly
        for _ in range(wait_seconds):
            if _shutdown_requested:
                break
            time.sleep(1)

    print("\n  Bot stopped cleanly.")
    try:
        status = get_portfolio_status()
        print(f"  Final equity: ${status['equity']:.2f}  PnL: ${status['realized_pnl']:+.2f}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Trading Bot -- One-Click Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_bot.py                # continuous (default 5 min intervals)
  python run_bot.py --once         # single cycle then exit
  python run_bot.py --interval 10  # run every 10 minutes
  python run_bot.py --interval 5   # aggressive: every 5 minutes
        """,
    )
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between cycles (default: 5)")
    args = parser.parse_args()

    run_bot(once=args.once, interval_minutes=args.interval)


if __name__ == "__main__":
    main()
