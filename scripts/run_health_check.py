from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading_bot.health_check import generate_health_report, save_health_report

DB_PATH = ROOT / "data" / "trading_bot.sqlite3"
OUT_PATH = ROOT / "runtime" / "health_report_latest.json"


def main() -> None:
    report = generate_health_report(DB_PATH)
    save_health_report(report, OUT_PATH)

    summary = report.get("summary", {})
    council = report.get("council", {})

    print(f"Health Score: {report.get('score', 0)}/100 ({report.get('status', 'unknown')})")
    print(
        f"Closed trades: {summary.get('closed_trades', 0)} | Win rate: {float(summary.get('win_rate', 0.0)):.1%} | "
        f"Realized PnL: ${float(summary.get('realized_pnl', 0.0)):+.2f}"
    )
    print(
        f"Council recent: total={council.get('total_recent_decisions', 0)} "
        f"BUY={council.get('buy', 0)} HOLD={council.get('hold', 0)} AVOID={council.get('avoid', 0)}"
    )

    issues = report.get("issues", [])
    if issues:
        print("Issues:")
        for item in issues[:8]:
            print(f"- {item}")

    recs = report.get("recommendations", [])
    if recs:
        print("Recommendations:")
        for item in recs[:8]:
            print(f"- {item}")

    print(f"Saved: {OUT_PATH}")
    print(json.dumps({"score": report.get("score"), "status": report.get("status")}, indent=2))


if __name__ == "__main__":
    main()
