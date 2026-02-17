from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class HealthThresholds:
    min_closed_trades: int = 8
    min_win_rate: float = 0.45
    max_pipeline_errors_24h: int = 2
    max_fallbacks_24h: int = 20


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def generate_health_report(
    db_path: Path,
    thresholds: HealthThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or HealthThresholds()

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score": 100,
        "status": "healthy",
        "summary": {},
        "council": {},
        "quality_checks": {},
        "issues": [],
        "recommendations": [],
        "latest": {},
    }

    if not db_path.exists():
        report["score"] = 0
        report["status"] = "no_data"
        report["issues"].append("Database not found")
        report["recommendations"].append("Start the bot to generate data before running health checks")
        return report

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
              COUNT(1) AS closed_trades,
              COALESCE(SUM(pnl), 0.0) AS realized_pnl,
              COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
            FROM trade_outcomes
            """
        ).fetchone()
        closed_trades = int(row["closed_trades"] if row else 0)
        wins = int(row["wins"] if row else 0)
        realized_pnl = float(row["realized_pnl"] if row else 0.0)
        win_rate = (wins / closed_trades) if closed_trades > 0 else 0.0

        open_row = conn.execute(
            """
            SELECT COUNT(1) AS open_positions,
                   COALESCE(SUM(quantity * price), 0.0) AS open_notional
            FROM orders
            WHERE side = 'buy' AND status IN ('paper_filled', 'live_ready', 'live_submitted', 'live_filled')
            """
        ).fetchone()
        open_positions = int(open_row["open_positions"] if open_row else 0)
        open_notional = float(open_row["open_notional"] if open_row else 0.0)

        events = conn.execute(
            """
            SELECT event_type, COUNT(1) AS n
            FROM system_events
            WHERE created_at >= datetime('now', '-1 day')
            GROUP BY event_type
            """
        ).fetchall()
        event_counts = {str(r["event_type"]): int(r["n"]) for r in events}

        decisions = conn.execute(
            """
            SELECT symbol, model_name, confidence, decision_json, created_at
            FROM ai_decisions
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()

        council_total = 0
        council_buy = 0
        council_hold = 0
        council_avoid = 0
        council_avg_conf = 0.0
        council_conf_values: list[float] = []
        recent_council_reasons: list[dict[str, Any]] = []

        for r in decisions:
            model_name = str(r["model_name"] or "")
            if "council" not in model_name:
                continue
            council_total += 1
            payload = _loads(str(r["decision_json"] or "{}"))
            decision = payload.get("decision", payload) if isinstance(payload, dict) else {}
            action = str(decision.get("action", "")).upper()
            conf = float(decision.get("confidence", r["confidence"] or 0.0) or 0.0)
            rationale = str(decision.get("rationale", ""))
            votes = str(decision.get("council_votes", ""))

            if action == "BUY":
                council_buy += 1
            elif action == "HOLD":
                council_hold += 1
            else:
                council_avoid += 1

            council_conf_values.append(conf)

            if len(recent_council_reasons) < 8:
                recent_council_reasons.append(
                    {
                        "symbol": str(r["symbol"]),
                        "created_at": str(r["created_at"]),
                        "action": action,
                        "confidence": round(conf, 4),
                        "votes": votes,
                        "rationale": rationale[:500],
                    }
                )

        if council_conf_values:
            council_avg_conf = sum(council_conf_values) / len(council_conf_values)

        # Confidence quality: are high-confidence trades actually working?
        conf_quality_rows = conn.execute(
            """
            SELECT
              COALESCE(AVG(CASE WHEN o.decision_confidence >= 0.75 THEN t.pnl END), 0.0) AS avg_pnl_high_conf,
              COALESCE(AVG(CASE WHEN o.decision_confidence < 0.60 THEN t.pnl END), 0.0) AS avg_pnl_low_conf,
              COALESCE(SUM(CASE WHEN o.decision_confidence >= 0.75 THEN 1 ELSE 0 END), 0) AS high_conf_count
            FROM trade_outcomes t
            JOIN orders o ON o.id = t.order_id
            """
        ).fetchone()

        avg_pnl_high_conf = float(conf_quality_rows["avg_pnl_high_conf"] if conf_quality_rows else 0.0)
        avg_pnl_low_conf = float(conf_quality_rows["avg_pnl_low_conf"] if conf_quality_rows else 0.0)
        high_conf_count = int(conf_quality_rows["high_conf_count"] if conf_quality_rows else 0)

        last_order = conn.execute(
            "SELECT symbol, status, created_at FROM orders ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_decision = conn.execute(
            "SELECT symbol, model_name, created_at FROM ai_decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()

    report["summary"] = {
        "closed_trades": closed_trades,
        "wins": wins,
        "win_rate": round(win_rate, 4),
        "realized_pnl": round(realized_pnl, 6),
        "open_positions": open_positions,
        "open_notional": round(open_notional, 6),
        "event_counts_24h": event_counts,
    }

    report["council"] = {
        "total_recent_decisions": council_total,
        "buy": council_buy,
        "hold": council_hold,
        "avoid": council_avoid,
        "avg_confidence": round(council_avg_conf, 4),
        "recent_reasons": recent_council_reasons,
    }

    report["quality_checks"] = {
        "avg_pnl_high_conf": round(avg_pnl_high_conf, 6),
        "avg_pnl_low_conf": round(avg_pnl_low_conf, 6),
        "high_conf_count": high_conf_count,
    }

    report["latest"] = {
        "last_order": dict(last_order) if last_order else None,
        "last_decision": dict(last_decision) if last_decision else None,
    }

    # Score and recommendations
    score = 100

    pipeline_errors = int(event_counts.get("pipeline_error", 0))
    if pipeline_errors > 0:
        penalty = min(35, pipeline_errors * 8)
        score -= penalty
        report["issues"].append(f"pipeline_error events in 24h: {pipeline_errors}")

    fallbacks = int(event_counts.get("inference_fallback", 0))
    if fallbacks > t.max_fallbacks_24h:
        score -= 12
        report["issues"].append(f"high inference fallbacks in 24h: {fallbacks}")

    if closed_trades >= t.min_closed_trades and win_rate < t.min_win_rate:
        score -= 20
        report["issues"].append(
            f"win rate below threshold: {win_rate:.1%} < {t.min_win_rate:.1%} (closed trades={closed_trades})"
        )

    if high_conf_count >= 5 and avg_pnl_high_conf < 0:
        score -= 10
        report["issues"].append("high-confidence trades are underperforming on average")

    if council_total < 3:
        score -= 8
        report["issues"].append("very few recent council decisions; monitor data freshness")

    score = max(0, min(100, int(score)))
    report["score"] = score
    report["status"] = "healthy" if score >= 80 else ("watch" if score >= 60 else "needs_attention")

    if pipeline_errors > t.max_pipeline_errors_24h:
        report["recommendations"].append("Investigate recent pipeline errors in system_events and bot logs")
    if fallbacks > t.max_fallbacks_24h:
        report["recommendations"].append("Reduce model latency/fallbacks (check Ollama health and timeout settings)")
    if closed_trades >= t.min_closed_trades and win_rate < t.min_win_rate:
        report["recommendations"].append("Tighten confidence threshold or reduce max symbols per cycle")
    if high_conf_count >= 5 and avg_pnl_high_conf < 0:
        report["recommendations"].append("Recalibrate council prompts or confidence weighting")
    if not report["recommendations"]:
        report["recommendations"].append("No urgent changes required; keep monitoring and gather more trades")

    return report


def save_health_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
