import json
from collections import defaultdict
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading_bot.backtester import Backtester, BacktestConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused council backtest evaluation")
    parser.add_argument("--symbols", type=str, default="NIO", help="Comma-separated symbols")
    parser.add_argument("--capital", type=float, default=2000.0, help="Starting capital")
    parser.add_argument("--period", type=str, default="1y", help="Backtest period")
    parser.add_argument("--interval", type=str, default="1d", help="Backtest candle interval")
    parser.add_argument("--gate", type=int, default=45, help="Signal score gate")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Minimum confidence")
    parser.add_argument("--tp", type=float, default=8.0, help="Take-profit percent")
    parser.add_argument("--sl", type=float, default=5.0, help="Stop-loss percent")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        symbols = ["NIO"]

    cfg = BacktestConfig(
        symbols=symbols,
        kinds=["crypto", "stock"],
        period=str(args.period),
        interval=str(args.interval),
        starting_capital=float(args.capital),
        goal_equity=float(args.capital) * 2.0,
        use_ai=True,
        use_council=True,
        signal_score_gate=int(args.gate),
        min_confidence=float(args.min_conf),
        profit_target_pct=float(args.tp),
        stop_loss_pct=float(args.sl),
        hold_candles_min=3,
        hold_candles_max=15,
    )

    bt = Backtester(cfg)
    result = bt.run()

    pnl_by_day: dict[str, float] = defaultdict(float)
    trades_by_day: dict[str, int] = defaultdict(int)

    for trade in result.trades:
        day = str(trade.exit_timestamp)[:10]
        pnl_by_day[day] += float(trade.net_pnl)
        trades_by_day[day] += 1

    days = sorted(pnl_by_day.keys())
    positive_days = sum(1 for d in days if pnl_by_day[d] > 0)
    negative_days = sum(1 for d in days if pnl_by_day[d] < 0)
    flat_days = sum(1 for d in days if abs(pnl_by_day[d]) <= 1e-9)
    daily_positive_rate = (positive_days / len(days)) if days else 0.0

    summary = {
        "run_id": result.run_id,
        "symbols": cfg.symbols,
        "starting_capital": cfg.starting_capital,
        "final_equity": result.final_equity,
        "total_pnl": result.total_pnl,
        "trade_count": len(result.trades),
        "trade_win_rate": result.win_rate,
        "max_drawdown_pct": result.max_drawdown_pct,
        "sharpe": result.sharpe_estimate,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "flat_days": flat_days,
        "trading_days_with_exits": len(days),
        "daily_positive_rate": round(daily_positive_rate, 4),
    }

    print(json.dumps(summary, indent=2))
    print("TOP_DAYS")
    for day in sorted(days, key=lambda d: pnl_by_day[d], reverse=True)[:5]:
        print(day, round(pnl_by_day[day], 4), trades_by_day[day])
    print("WORST_DAYS")
    for day in sorted(days, key=lambda d: pnl_by_day[d])[:5]:
        print(day, round(pnl_by_day[day], 4), trades_by_day[day])


if __name__ == "__main__":
    main()
