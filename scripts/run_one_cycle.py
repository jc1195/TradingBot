"""Run one trading cycle immediately (paper mode).

Tests the full pipeline:
  Robinhood auth -> trend scan -> quote -> council AI -> paper order
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")

from src.trading_bot.db import initialize_database
from src.trading_bot.engine import TradingEngine
from src.trading_bot.risk import load_risk_policy
from src.trading_bot.settings import settings


def main():
    print("=" * 60)
    print("LIVE ENGINE SMOKE TEST -- Paper Mode")
    print(f"Mode:    {settings.bot_mode}")
    print(f"Council: {settings.use_council}")
    print(f"Capital: ${settings.paper_working_capital_usd:.0f}")
    print(f"Max positions: {settings.max_concurrent_positions}")
    print("=" * 60)

    initialize_database()
    engine = TradingEngine(load_risk_policy())

    print(f"\nStarting cycle at {time.strftime('%H:%M:%S')}...")
    t0 = time.time()

    try:
        engine.run_daily_cycle()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print(f"\nCycle completed in {elapsed:.0f}s at {time.strftime('%H:%M:%S')}")

    # Show recent orders from DB
    from src.trading_bot.db import get_connection
    with get_connection() as conn:
        orders = conn.execute(
            "SELECT symbol, side, quantity, notional_usd, status, created_at "
            "FROM orders ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

        decisions = conn.execute(
            "SELECT symbol, provider, action, confidence, created_at "
            "FROM ai_decisions ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

    if decisions:
        print(f"\nRecent AI Decisions ({len(decisions)}):")
        for d in decisions:
            print(f"  {d['symbol']:8s} {d['action']:6s} conf={d['confidence']:.0%} via {d['provider']}")
    else:
        print("\nNo AI decisions recorded.")

    if orders:
        print(f"\nRecent Orders ({len(orders)}):")
        for o in orders:
            print(f"  {o['side']:5s} {o['symbol']:8s} qty={o['quantity']:.4f} ${o['notional_usd']:.2f} [{o['status']}]")
    else:
        print("\nNo orders placed.")


if __name__ == "__main__":
    main()
