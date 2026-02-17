"""Parameter optimizer — batch backtests to find optimal strategy settings.

Runs a grid sweep over key parameters:
 • profit_target_pct: [5, 8, 10, 12]
 • stop_loss_pct: [2, 3, 4, 5]
 • signal_score_gate: [10, 15, 20, 25]
 • min_confidence: [0.60, 0.65, 0.70, 0.75]

For each combination, runs a backtest and records results.
Then picks the parameter set with highest Sharpe / best risk-adjusted return.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from loguru import logger

from .backtester import BacktestConfig, BacktestResult, Backtester


RESULTS_DIR = Path("data/optimization_results")


@dataclass
class OptimizationResult:
    config: dict
    pnl: float
    win_rate: float
    sharpe: float
    trades: int
    max_drawdown: float
    final_equity: float
    duration_seconds: float
    run_id: str


def run_parameter_sweep(
    symbols: list[str],
    kinds: list[str],
    period: str = "1y",
    interval: str = "1d",
    starting_capital: float = 500.0,
    use_ai: bool = True,
    use_council: bool = True,
    hold_candles_min: int = 3,
    hold_candles_max: int = 15,
    # Parameter ranges to sweep
    tp_range: list[float] | None = None,
    sl_range: list[float] | None = None,
    gate_range: list[int] | None = None,
    conf_range: list[float] | None = None,
    max_combos: int = 30,
) -> list[OptimizationResult]:
    """Run parameter sweep and return sorted results (best first)."""

    tp_range = tp_range or [5.0, 8.0, 12.0]
    sl_range = sl_range or [2.0, 3.0, 5.0]
    gate_range = gate_range or [10, 20]
    conf_range = conf_range or [0.65, 0.70]

    combos = list(product(tp_range, sl_range, gate_range, conf_range))
    if len(combos) > max_combos:
        # Sample evenly
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    logger.info("Parameter sweep: {} combinations to test", len(combos))
    results: list[OptimizationResult] = []

    for i, (tp, sl, gate, conf) in enumerate(combos):
        # Skip nonsense combos (TP must be > SL for positive R:R)
        if tp <= sl:
            continue

        logger.info(
            "Sweep {}/{}: TP={}% SL={}% gate={} conf={:.0%}",
            i + 1, len(combos), tp, sl, gate, conf,
        )

        cfg = BacktestConfig(
            symbols=symbols,
            kinds=kinds,
            period=period,
            interval=interval,
            starting_capital=starting_capital,
            goal_equity=starting_capital * 2,
            use_ai=use_ai,
            use_council=use_council and use_ai,
            hold_candles_min=hold_candles_min,
            hold_candles_max=hold_candles_max,
            profit_target_pct=tp,
            stop_loss_pct=sl,
            trailing_stop_pct=min(sl, 2.5),
            signal_score_gate=gate,
            min_confidence=conf,
        )

        t0 = time.time()
        try:
            bt = Backtester(cfg)
            result = bt.run()
            elapsed = time.time() - t0

            opt_result = OptimizationResult(
                config={
                    "tp": tp, "sl": sl, "gate": gate, "conf": conf,
                    "rr_ratio": round(tp / sl, 1),
                },
                pnl=result.total_pnl,
                win_rate=result.win_rate,
                sharpe=result.sharpe_estimate,
                trades=len(result.trades),
                max_drawdown=result.max_drawdown_pct,
                final_equity=result.final_equity,
                duration_seconds=round(elapsed, 1),
                run_id=result.run_id,
            )
            results.append(opt_result)

            logger.info(
                "  → PnL=${:.2f}  WR={:.1%}  Sharpe={:.2f}  Trades={}  Time={:.0f}s",
                result.total_pnl, result.win_rate, result.sharpe_estimate,
                len(result.trades), elapsed,
            )
        except Exception as e:
            logger.error("Sweep combo {} failed: {}", i + 1, e)

    # Sort by composite score: Sharpe * sqrt(trades) to reward both quality and volume
    import math
    results.sort(
        key=lambda r: r.sharpe * math.sqrt(max(r.trades, 1)) if r.trades > 5 else -999,
        reverse=True,
    )

    # Save results
    _save_results(results, symbols, kinds, period)

    return results


def _save_results(
    results: list[OptimizationResult],
    symbols: list[str],
    kinds: list[str],
    period: str,
) -> None:
    """Save optimization results to JSON for future reference."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sweep_{ts}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "kinds": kinds,
        "period": period,
        "results": [
            {
                "rank": i + 1,
                "config": r.config,
                "pnl": r.pnl,
                "win_rate": r.win_rate,
                "sharpe": r.sharpe,
                "trades": r.trades,
                "max_drawdown": r.max_drawdown,
                "final_equity": r.final_equity,
                "duration_seconds": r.duration_seconds,
                "run_id": r.run_id,
            }
            for i, r in enumerate(results)
        ],
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Optimization results saved to {}", path)


def print_sweep_report(results: list[OptimizationResult]) -> None:
    """Pretty-print optimization results."""
    print()
    print("=" * 70)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"{'Rank':>4}  {'TP%':>4}  {'SL%':>4}  {'R:R':>4}  {'Gate':>4}  "
          f"{'Conf':>5}  {'PnL':>8}  {'WR%':>5}  {'Sharpe':>6}  {'Trades':>6}")
    print("-" * 70)

    for i, r in enumerate(results):
        c = r.config
        pnl_color = "" if r.pnl >= 0 else ""
        print(f"{i + 1:4d}  {c['tp']:4.0f}  {c['sl']:4.0f}  {c['rr_ratio']:4.1f}  "
              f"{c['gate']:4d}  {c['conf']:5.0%}  ${r.pnl:+7.2f}  {r.win_rate:5.1%}  "
              f"{r.sharpe:+6.2f}  {r.trades:6d}")

    if results:
        best = results[0]
        print()
        print(f"BEST CONFIG: TP={best.config['tp']}%  SL={best.config['sl']}%  "
              f"R:R={best.config['rr_ratio']}  Gate={best.config['gate']}  "
              f"Conf={best.config['conf']:.0%}")
        print(f"  PnL: ${best.pnl:+.2f}  Win Rate: {best.win_rate:.1%}  "
              f"Sharpe: {best.sharpe:+.2f}  Trades: {best.trades}")
