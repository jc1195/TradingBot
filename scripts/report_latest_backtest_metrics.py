import json
import sqlite3
from collections import defaultdict
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Report metrics for latest backtest run")
    parser.add_argument("--require-use-ai", action="store_true", help="Require config_json to indicate use_ai=true")
    parser.add_argument("--require-period", type=str, default="", help="Require specific period, e.g. 1y")
    parser.add_argument("--require-interval", type=str, default="", help="Require specific interval, e.g. 1d")
    args = parser.parse_args()

    conn = sqlite3.connect("data/trading_bot.sqlite3")
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT run_id, started_at, config_json,
               starting_capital, final_equity, total_pnl,
               win_rate, max_drawdown_pct, sharpe_estimate
        FROM backtest_runs
    """
    clauses: list[str] = []
    params: list[str] = []

    if args.require_use_ai:
        clauses.append("config_json LIKE ?")
        params.append('%"use_ai": true%')
    if args.require_period:
        clauses.append("config_json LIKE ?")
        params.append(f'%"period": "{args.require_period}"%')
    if args.require_interval:
        clauses.append("config_json LIKE ?")
        params.append(f'%"interval": "{args.require_interval}"%')

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY started_at DESC LIMIT 1"

    row = conn.execute(sql, params).fetchone()

    if not row:
        print("NO_RUN")
        return

    cfg = json.loads(row["config_json"] or "{}")
    print(f"run_id={row['run_id']}")
    print(f"started_at={row['started_at']}")
    print(f"starting_capital={float(row['starting_capital']):.2f}")
    print(f"final_equity={float(row['final_equity']):.2f}")
    print(f"total_pnl={float(row['total_pnl']):+.2f}")
    print(f"trade_win_rate={float(row['win_rate']):.4f}")
    print(f"sharpe={float(row['sharpe_estimate']):+.4f}")
    print(f"max_drawdown_pct={float(row['max_drawdown_pct']):.4f}")
    print(f"use_ai={cfg.get('use_ai')}")
    print(f"period={cfg.get('period')}")
    print(f"interval={cfg.get('interval')}")
    print(f"symbols={cfg.get('symbols')}")

    trades = conn.execute(
        """
        SELECT exit_timestamp, net_pnl
        FROM backtest_trades
        WHERE run_id = ?
        """,
        (row["run_id"],),
    ).fetchall()

    pnl_by_day: dict[str, float] = defaultdict(float)
    count_by_day: dict[str, int] = defaultdict(int)

    for trade in trades:
        day = str(trade["exit_timestamp"] or "")[:10]
        if not day:
            continue
        pnl_by_day[day] += float(trade["net_pnl"] or 0.0)
        count_by_day[day] += 1

    days = sorted(pnl_by_day)
    positive_days = sum(1 for day in days if pnl_by_day[day] > 0)
    negative_days = sum(1 for day in days if pnl_by_day[day] < 0)
    flat_days = sum(1 for day in days if abs(pnl_by_day[day]) <= 1e-9)
    daily_positive_rate = (positive_days / len(days)) if days else 0.0

    print(f"trading_days_with_exits={len(days)}")
    print(f"positive_days={positive_days}")
    print(f"negative_days={negative_days}")
    print(f"flat_days={flat_days}")
    print(f"daily_positive_rate={daily_positive_rate:.4f}")

    top_days = sorted(days, key=lambda day: pnl_by_day[day], reverse=True)[:5]
    worst_days = sorted(days, key=lambda day: pnl_by_day[day])[:5]

    print("top_days=")
    for day in top_days:
        print(f"  {day} pnl={pnl_by_day[day]:+.2f} trades={count_by_day[day]}")

    print("worst_days=")
    for day in worst_days:
        print(f"  {day} pnl={pnl_by_day[day]:+.2f} trades={count_by_day[day]}")


if __name__ == "__main__":
    main()
