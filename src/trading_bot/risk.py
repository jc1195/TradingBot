from dataclasses import dataclass
from pathlib import Path

import yaml

from .settings import settings


@dataclass
class RiskPolicy:
    max_risk_per_trade_pct: float
    max_daily_realized_loss_pct: float
    max_concurrent_positions: int
    min_confidence_to_trade: float
    max_position_pct: float
    allow_live_orders: bool
    cooldown_minutes: int
    kill_switch: bool



def load_risk_policy(file_path: Path = Path("config/risk_policy.yaml")) -> RiskPolicy:
    if not file_path.exists():
        return RiskPolicy(
            max_risk_per_trade_pct=settings.max_risk_per_trade_pct,
            max_daily_realized_loss_pct=settings.max_daily_realized_loss_pct,
            max_concurrent_positions=settings.max_concurrent_positions,
            min_confidence_to_trade=settings.min_confidence_to_trade,
            max_position_pct=0.35,
            allow_live_orders=False,
            cooldown_minutes=15,
            kill_switch=False,
        )

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    limits = raw.get("limits", {})
    trade_guards = raw.get("trade_guards", {})

    return RiskPolicy(
        max_risk_per_trade_pct=float(limits.get("max_risk_per_trade_pct", settings.max_risk_per_trade_pct)),
        max_daily_realized_loss_pct=float(
            limits.get("max_daily_realized_loss_pct", settings.max_daily_realized_loss_pct)
        ),
        max_concurrent_positions=int(limits.get("max_concurrent_positions", settings.max_concurrent_positions)),
        min_confidence_to_trade=float(limits.get("min_confidence_to_trade", settings.min_confidence_to_trade)),
        max_position_pct=float(limits.get("max_position_pct", 0.20)),
        allow_live_orders=bool(trade_guards.get("allow_live_orders", False)),
        cooldown_minutes=int(trade_guards.get("cooldown_minutes", 60)),
        kill_switch=bool(raw.get("kill_switch", False)),
    )
