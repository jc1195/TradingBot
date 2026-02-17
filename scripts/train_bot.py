"""Day-trade retraining pipeline.

This script is designed to make the bot trade-ready for today by:
1) pulling one full year of historical data,
2) running day-trade oriented parameter calibration,
3) learning from current loss patterns,
4) writing recommendations to runtime + memory.md,
5) optionally applying safer settings to risk/env config.

Usage:
    python scripts/train_bot.py
    python scripts/train_bot.py --use-ai
    python scripts/train_bot.py --max-combos 24
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from src.trading_bot.data_cache import bulk_download, get_cache_stats
from src.trading_bot.optimizer import OptimizationResult, print_sweep_report, run_parameter_sweep
from src.trading_bot.scanner import AssetScanner, print_scan_report


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trading_bot.sqlite3"
ENV_PATH = ROOT / ".env"
RISK_PATH = ROOT / "config" / "risk_policy.yaml"
RUNTIME_RECOMMENDATION_PATH = ROOT / "runtime" / "training_recommendations.json"
MEMORY_PATH = ROOT / "memory.md"


def _read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    data: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _write_env(changes: dict[str, str | int | float | bool]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    updated = lines[:]
    seen: set[str] = set()

    def _to_text(value: str | int | float | bool) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    for i, line in enumerate(updated):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in changes:
            updated[i] = f"{key}={_to_text(changes[key])}"
            seen.add(key)

    for key, value in changes.items():
        if key not in seen:
            updated.append(f"{key}={_to_text(value)}")

    ENV_PATH.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def _collect_learning_snapshot() -> dict:
    if not DB_PATH.exists():
        return {
            "realized_pnl": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "worst_symbols": [],
            "recent_losses": [],
        }

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        totals = conn.execute(
            """
            SELECT COUNT(1) AS trades,
                   COALESCE(SUM(pnl), 0.0) AS realized,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
            FROM trade_outcomes
            """
        ).fetchone()

        worst_symbols = conn.execute(
            """
            SELECT o.symbol,
                   COUNT(1) AS n,
                   COALESCE(SUM(t.pnl), 0.0) AS pnl
            FROM trade_outcomes t
            JOIN orders o ON o.id = t.order_id
            GROUP BY o.symbol
            HAVING COUNT(1) >= 2
            ORDER BY pnl ASC
            LIMIT 5
            """
        ).fetchall()

        recent_losses = conn.execute(
            """
            SELECT o.symbol, t.pnl, t.hold_minutes, t.closed_at
            FROM trade_outcomes t
            JOIN orders o ON o.id = t.order_id
            WHERE t.pnl < 0
            ORDER BY t.id DESC
            LIMIT 8
            """
        ).fetchall()

    trades = int(totals["trades"] if totals else 0)
    wins = int(totals["wins"] if totals else 0)
    realized = float(totals["realized"] if totals else 0.0)
    win_rate = (wins / trades) if trades else 0.0

    return {
        "realized_pnl": round(realized, 4),
        "trades": trades,
        "win_rate": round(win_rate, 4),
        "worst_symbols": [
            {
                "symbol": str(row["symbol"]),
                "trades": int(row["n"]),
                "pnl": round(float(row["pnl"]), 4),
            }
            for row in worst_symbols
        ],
        "recent_losses": [
            {
                "symbol": str(row["symbol"]),
                "pnl": round(float(row["pnl"]), 4),
                "hold_minutes": int(row["hold_minutes"] or 0),
                "closed_at": str(row["closed_at"]),
            }
            for row in recent_losses
        ],
    }


def _best_result(results: list[OptimizationResult]) -> OptimizationResult | None:
    if not results:
        return None
    tradeful = [r for r in results if r.trades > 0]
    if tradeful:
        tradeful.sort(key=lambda r: (r.sharpe, r.pnl, r.win_rate), reverse=True)
        return tradeful[0]
    return results[0]


def _build_recommendations(best: OptimizationResult | None, learning: dict) -> dict:
    conf = float(best.config["conf"]) if best else 0.68
    gate = int(best.config["gate"]) if best else 20

    realized = float(learning.get("realized_pnl", 0.0))
    win_rate = float(learning.get("win_rate", 0.0))
    losing = realized < 0 or win_rate < 0.45

    return {
        "min_confidence_to_trade": round(max(0.65, conf), 2),
        "max_risk_per_trade_pct": 0.05 if losing else 0.07,
        "max_concurrent_positions": 5 if losing else 6,
        "daytrade_min_score": max(50, gate + 30),
        "daytrade_min_bar_volume_ratio": 1.0,
        "quality_min_win_rate": 0.45 if losing else 0.40,
        "quality_min_trades": 8,
    }


def _phase_score(result: OptimizationResult | None) -> float:
    if result is None:
        return -1e9
    trade_boost = min(result.trades, 100) / 100.0
    return (result.sharpe * 2.0) + (result.pnl / 10.0) + trade_boost


def _apply_recommendations_to_configs(reco: dict) -> None:
    raw = yaml.safe_load(RISK_PATH.read_text(encoding="utf-8")) if RISK_PATH.exists() else {}
    raw = raw if isinstance(raw, dict) else {}
    raw.setdefault("limits", {})

    raw["limits"]["max_risk_per_trade_pct"] = float(reco["max_risk_per_trade_pct"])
    raw["limits"]["max_concurrent_positions"] = int(reco["max_concurrent_positions"])
    raw["limits"]["min_confidence_to_trade"] = float(reco["min_confidence_to_trade"])

    RISK_PATH.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    _write_env(
        {
            "MIN_CONFIDENCE_TO_TRADE": reco["min_confidence_to_trade"],
            "MAX_RISK_PER_TRADE_PCT": reco["max_risk_per_trade_pct"],
            "MAX_CONCURRENT_POSITIONS": reco["max_concurrent_positions"],
            "DAYTRADE_MIN_SCORE": reco["daytrade_min_score"],
            "DAYTRADE_MIN_BAR_VOLUME_RATIO": reco["daytrade_min_bar_volume_ratio"],
            "ANALYTICS_MIN_WIN_RATE": reco["quality_min_win_rate"],
            "ANALYTICS_MIN_TRADES_FOR_QUALITY_ALERT": reco["quality_min_trades"],
        }
    )


def _append_memory_log(learning: dict, selected_phase: str, best: OptimizationResult | None, reco: dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    lines: list[str] = []
    lines.append("")
    lines.append(f"### {now} - Retraining Session")
    lines.append(f"- Realized PnL snapshot: ${float(learning.get('realized_pnl', 0.0)):+.2f}")
    lines.append(f"- Win rate snapshot: {float(learning.get('win_rate', 0.0)):.1%} over {int(learning.get('trades', 0))} trades")

    worst = learning.get("worst_symbols", [])
    if isinstance(worst, list) and worst:
        worst_txt = ", ".join(f"{w['symbol']}({float(w['pnl']):+.2f})" for w in worst[:5])
        lines.append(f"- Biggest losing symbols: {worst_txt}")

    if best is not None:
        lines.append(
            "- Best calibration: "
            f"phase={selected_phase}, TP={best.config['tp']}%, SL={best.config['sl']}%, "
            f"gate={best.config['gate']}, conf={best.config['conf']:.0%}, "
            f"PnL={best.pnl:+.2f}, WR={best.win_rate:.1%}, Sharpe={best.sharpe:+.2f}, trades={best.trades}"
        )

    lines.append(
        "- Applied controls: "
        f"min_conf={float(reco['min_confidence_to_trade']):.2f}, "
        f"risk/trade={float(reco['max_risk_per_trade_pct']):.2%}, "
        f"max_positions={int(reco['max_concurrent_positions'])}, "
        f"daytrade_min_score={int(reco['daytrade_min_score'])}, "
        f"min_volume_ratio={float(reco['daytrade_min_bar_volume_ratio']):.2f}"
    )
    lines.append("- Learning rule: Avoid symbols repeatedly showing negative expectancy until new backtest evidence improves.")

    existing = MEMORY_PATH.read_text(encoding="utf-8") if MEMORY_PATH.exists() else "# TradingBot Memory\n"
    MEMORY_PATH.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bot for day trading with 1-year history + intraday calibration")
    parser.add_argument("--max-combos", type=int, default=20, help="Max parameter combinations per sweep")
    parser.add_argument("--use-ai", action="store_true", help="Use AI inference during sweeps (slower)")
    parser.add_argument("--no-apply", action="store_true", help="Do not apply recommendations to risk/.env")
    parser.add_argument("--capital", type=float, default=2000.0, help="Training bankroll for backtests (default: 2000)")
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 70)
    print("DAY-TRADE RETRAINING PIPELINE")
    print("=" * 70)
    print(f"Training capital: ${float(args.capital):.2f}")

    print("\nSTEP 1: Downloading 1-year market history (daily) and intraday context (60d/1h)...")
    daily_dl = bulk_download(period="1y", interval="1d")
    intraday_dl = bulk_download(period="60d", interval="1h")
    stats = get_cache_stats()
    daily_ok = sum(1 for n in daily_dl.values() if n > 0)
    intraday_ok = sum(1 for n in intraday_dl.values() if n > 0)
    print(f"  Daily cache success: {daily_ok}/{len(daily_dl)}")
    print(f"  Intraday cache success: {intraday_ok}/{len(intraday_dl)}")
    print(f"  Cache footprint: {stats['files']} files, {stats['total_candles']} candles, {stats['total_size_mb']} MB")

    print("\nSTEP 2: Scanning for best current day-trade candidates...")
    scanner = AssetScanner(period="6mo", interval="1d")
    scan_result = scanner.scan(top_n=10)
    print_scan_report(scan_result)

    top_symbols = [s.symbol for s in scan_result.top_picks]
    top_kinds = sorted({s.kind for s in scan_result.top_picks})
    if not top_symbols:
        top_symbols = ["RIVN", "RIG", "ZIM", "FSLY", "ATOM", "XRP"]
        top_kinds = ["stock", "crypto"]

    print(f"\nSelected symbols: {top_symbols[:6]}")
    print(f"Selected kinds: {top_kinds}")

    print("\nSTEP 3: Running 1-year daily calibration sweep...")
    daily_results = run_parameter_sweep(
        symbols=top_symbols[:6],
        kinds=top_kinds,
        period="1y",
        interval="1d",
        starting_capital=float(args.capital),
        use_ai=args.use_ai,
        use_council=args.use_ai,
        hold_candles_min=3,
        hold_candles_max=15,
        tp_range=[5.0, 8.0, 12.0],
        sl_range=[2.0, 3.0, 5.0],
        gate_range=[5, 10, 15],
        conf_range=[0.45, 0.55, 0.65],
        max_combos=max(8, args.max_combos),
    )
    print_sweep_report(daily_results)

    print("\nSTEP 4: Running intraday day-trade sweep (60d, 1h candles)...")
    intraday_results = run_parameter_sweep(
        symbols=top_symbols[:6],
        kinds=top_kinds,
        period="60d",
        interval="1h",
        starting_capital=float(args.capital),
        use_ai=args.use_ai,
        use_council=args.use_ai,
        hold_candles_min=2,
        hold_candles_max=8,
        tp_range=[2.0, 3.0, 4.0, 5.0],
        sl_range=[1.0, 1.5, 2.0],
        gate_range=[5, 10, 15],
        conf_range=[0.40, 0.50, 0.60],
        max_combos=max(8, args.max_combos),
    )
    print_sweep_report(intraday_results)

    learning = _collect_learning_snapshot()
    best_intraday = _best_result(intraday_results)
    best_daily = _best_result(daily_results)

    selected_phase = "intraday"
    selected = best_intraday
    if selected is None and best_daily is not None:
        selected_phase = "daily"
        selected = best_daily
    elif best_daily is not None and _phase_score(best_daily) > _phase_score(best_intraday):
        selected_phase = "daily"
        selected = best_daily

    recommendations = _build_recommendations(selected, learning)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_phase": selected_phase,
        "selected_result": {
            "config": selected.config if selected else {},
            "pnl": selected.pnl if selected else 0.0,
            "win_rate": selected.win_rate if selected else 0.0,
            "sharpe": selected.sharpe if selected else 0.0,
            "trades": selected.trades if selected else 0,
            "run_id": selected.run_id if selected else "",
        },
        "learning_snapshot": learning,
        "recommendations": recommendations,
        "use_ai_in_training": bool(args.use_ai),
    }

    RUNTIME_RECOMMENDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_RECOMMENDATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.no_apply:
        _apply_recommendations_to_configs(recommendations)

    _append_memory_log(learning, selected_phase, selected, recommendations)

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)
    print(f"Duration: {elapsed / 60:.1f} minutes")
    print(f"Learning snapshot: realized=${float(learning['realized_pnl']):+.2f}, win_rate={float(learning['win_rate']):.1%}, trades={int(learning['trades'])}")
    if selected:
        print(
            "Selected config: "
            f"phase={selected_phase}, TP={selected.config['tp']}%, SL={selected.config['sl']}%, "
            f"gate={selected.config['gate']}, conf={selected.config['conf']:.0%}, trades={selected.trades}, "
            f"PnL={selected.pnl:+.2f}, Sharpe={selected.sharpe:+.2f}"
        )
    print(f"Recommendations saved to: {RUNTIME_RECOMMENDATION_PATH}")
    print(f"Memory updated: {MEMORY_PATH}")
    print(f"Applied to config: {'NO' if args.no_apply else 'YES'}")


if __name__ == "__main__":
    main()
