from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from .db import get_connection, initialize_database
from .engine import TradingEngine
from .risk import load_risk_policy
from .robinhood_client import RobinhoodClient
from .settings import settings
from .shadow_ml import ShadowMLService
from .state import runtime_state
from .trend_detector import scan_for_bullish_opportunities
from .watchlist import get_watchlist


RUNTIME_DIR = Path("runtime")
CYCLE_REPORT_PATH = RUNTIME_DIR / "cycle_report_latest.json"
HEALTH_REPORT_PATH = RUNTIME_DIR / "health_report_latest.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _env_line(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _update_env_values(changes: dict[str, float | int | str | bool]) -> bool:
    try:
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
        return True
    except Exception:
        return False


class PaperSizePayload(BaseModel):
    direction: Literal["increase", "decrease"]
    pct: float = Field(..., ge=0.01, le=0.5)


class PaperCapitalPayload(BaseModel):
    max_capital: float = Field(..., ge=0.0, le=10_000_000)
    wiggle_capital: float = Field(..., ge=0.0, le=1_000_000)


class AdaptiveTogglesPayload(BaseModel):
    adaptive_special_circumstances_enabled: bool
    adaptive_special_circumstances_paper_enabled: bool
    adaptive_special_circumstances_live_enabled: bool
    adaptive_technician_enabled: bool
    adaptive_hourly_rule_generation_enabled: bool
    adaptive_exception_scope: Literal["symbol", "asset_class", "global"] = "asset_class"


class DashboardController:
    def __init__(self) -> None:
        self.engine = TradingEngine(load_risk_policy())
        self.robinhood = RobinhoodClient()
        self.shadow = ShadowMLService()
        self.telemetry_cache: dict[str, Any] = {
            "updated_at": "",
            "bot_state": "PAUSED",
            "paper": {},
            "live": {},
            "highest_mover_today": {},
            "learning_panels": {},
            "positions": {"paper": [], "live": []},
            "orders": [],
            "errors": [],
            "alerts": [],
            "scan": {},
            "watchlist": [],
            "adaptive": {},
        }
        self._threads_started = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._trade_timeframe = "1m"
        self._candle_delay_seconds = 2
        self._full_scan_seconds = 600
        self._watchlist_refresh_seconds = 60
        self._last_full_scan_at = 0.0
        self._last_watchlist_update_at = 0.0
        self._kind_map = self._build_kind_map()

    @staticmethod
    def _build_kind_map() -> dict[str, str]:
        mapping: dict[str, str] = {}
        try:
            for item in get_watchlist(include_penny=True, include_micro_crypto=True):
                sym = str(getattr(item, "symbol", "")).upper().replace("-USD", "")
                kind = str(getattr(item, "kind", "stock")).lower()
                if sym:
                    mapping[sym] = "crypto" if kind == "crypto" else "stock"
        except Exception:
            pass
        return mapping

    def _guess_kind(self, symbol: str) -> str:
        clean = str(symbol or "").upper().replace("-USD", "")
        if clean in self._kind_map:
            return self._kind_map[clean]
        if "-" in str(symbol or ""):
            return "crypto"
        return "stock"

    def start(self) -> None:
        if self._threads_started:
            return
        initialize_database()
        self._threads_started = True
        self._stop_event.clear()

        threading.Thread(target=self._telemetry_loop, name="telemetry-loop", daemon=True).start()
        threading.Thread(target=self._trading_loop, name="trading-loop", daemon=True).start()
        threading.Thread(target=self._scanner_loop, name="scanner-loop", daemon=True).start()
        threading.Thread(target=self._shadow_loop, name="shadow-loop", daemon=True).start()
        logger.info("Dashboard controller started")

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return fallback

    def _telemetry_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self.telemetry_cache["updated_at"] = self._now_iso()
                    self.telemetry_cache["paper"] = self._paper_account_snapshot()
                    self.telemetry_cache["live"] = self._live_account_snapshot()
                    self.telemetry_cache["positions"] = self._positions_snapshot()
                    self.telemetry_cache["orders"] = self._orders_snapshot(limit=100)
                    self.telemetry_cache["errors"] = self._latest_errors()
                    self.telemetry_cache["alerts"] = self._latest_alerts()
                    self.telemetry_cache["scan"] = self._latest_cycle_scan()
                    self.telemetry_cache["highest_mover_today"] = self._highest_mover_today()
                    self.telemetry_cache["learning_panels"] = self._learning_panels_snapshot()
                    self.telemetry_cache["adaptive"] = self._adaptive_snapshot()
                    self.telemetry_cache["bot_state"] = self._bot_state_value()
            except Exception as exc:
                logger.warning("Telemetry refresh failed: {}", exc)
            time.sleep(2.0)

    def _special_circumstances_snapshot(self) -> dict[str, Any]:
        path = Path(getattr(settings, "adaptive_special_circumstances_path", Path("runtime/special_circumstances.json")))
        payload: dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload = raw
            except Exception:
                payload = {}

        rules = payload.get("rules", []) if isinstance(payload.get("rules", []), list) else []
        pair_stats = payload.get("pair_stats", {}) if isinstance(payload.get("pair_stats", {}), dict) else {}
        history = payload.get("history", []) if isinstance(payload.get("history", []), list) else []

        enabled_rules = [r for r in rules if isinstance(r, dict) and bool(r.get("enabled", True))]
        buy_rules = [r for r in enabled_rules if str(r.get("type", "") or "").lower() == "daytrade_exception"]
        sell_rules = [r for r in enabled_rules if str(r.get("type", "") or "").lower() == "sell_exit_override"]

        blocked_technician = 0
        if not bool(getattr(settings, "adaptive_technician_enabled", True)):
            blocked_technician = sum(
                1
                for r in enabled_rules
                if str(r.get("source_role", "") or "").strip().lower() == "technician"
            )

        top_pairs: list[dict[str, Any]] = []
        for pair_id, stat in pair_stats.items():
            if not isinstance(stat, dict):
                continue
            top_pairs.append(
                {
                    "pair_id": str(pair_id),
                    "trades": int(stat.get("trades", 0) or 0),
                    "wins": int(stat.get("wins", 0) or 0),
                    "losses": int(stat.get("losses", 0) or 0),
                    "net_pnl": float(stat.get("net_pnl", 0.0) or 0.0),
                    "disabled": bool(stat.get("disabled", False)),
                }
            )
        top_pairs.sort(key=lambda x: (x["disabled"], -x["trades"], -x["net_pnl"]))

        return {
            "path": str(path),
            "file_exists": bool(path.exists()),
            "updated_at": str(payload.get("updated_at", "") or ""),
            "last_hourly_review_at": str(payload.get("last_hourly_review_at", "") or ""),
            "last_close_event_scan_at": str(payload.get("last_close_event_scan_at", "") or ""),
            "rules_total": len(rules),
            "rules_enabled": len(enabled_rules),
            "buy_rules_enabled": len(buy_rules),
            "sell_rules_enabled": len(sell_rules),
            "blocked_technician_rules": int(blocked_technician),
            "history_count": len(history),
            "top_pairs": top_pairs[:8],
            "recent_history": history[-8:] if isinstance(history, list) else [],
        }

    @staticmethod
    def _adaptive_settings_snapshot() -> dict[str, Any]:
        return {
            "adaptive_special_circumstances_enabled": bool(getattr(settings, "adaptive_special_circumstances_enabled", False)),
            "adaptive_special_circumstances_paper_enabled": bool(getattr(settings, "adaptive_special_circumstances_paper_enabled", False)),
            "adaptive_special_circumstances_live_enabled": bool(getattr(settings, "adaptive_special_circumstances_live_enabled", False)),
            "adaptive_technician_enabled": bool(getattr(settings, "adaptive_technician_enabled", True)),
            "adaptive_hourly_rule_generation_enabled": bool(getattr(settings, "adaptive_hourly_rule_generation_enabled", False)),
            "adaptive_exception_scope": str(getattr(settings, "adaptive_exception_scope", "asset_class") or "asset_class"),
            "adaptive_special_circumstances_path": str(getattr(settings, "adaptive_special_circumstances_path", Path("runtime/special_circumstances.json"))),
            "adaptive_max_new_rule_pairs_per_hour": int(getattr(settings, "adaptive_max_new_rule_pairs_per_hour", 2)),
            "adaptive_min_candidate_daytrade_score_ratio": float(getattr(settings, "adaptive_min_candidate_daytrade_score_ratio", 0.75)),
            "adaptive_rule_pair_max_losses": int(getattr(settings, "adaptive_rule_pair_max_losses", 2)),
            "adaptive_rule_pair_max_net_loss_usd": float(getattr(settings, "adaptive_rule_pair_max_net_loss_usd", 20.0)),
        }

    def _adaptive_snapshot(self) -> dict[str, Any]:
        return {
            "settings": self._adaptive_settings_snapshot(),
            "rules": self._special_circumstances_snapshot(),
        }

    def _latest_decision_payload_for_symbol(self, symbol: str) -> dict[str, Any]:
        sym = str(symbol or "").upper().replace("-USD", "")
        if not sym:
            return {}
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, symbol, decision_json, created_at
                FROM ai_decisions
                WHERE UPPER(REPLACE(symbol, '-USD', '')) = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (sym,),
            ).fetchone()
        if not row:
            return {}
        try:
            payload = json.loads(str(row["decision_json"] or "{}"))
        except Exception:
            payload = {}
        return {
            "id": int(row["id"]),
            "symbol": str(row["symbol"] or sym),
            "created_at": str(row["created_at"] or ""),
            "payload": payload,
        }

    @staticmethod
    def _build_daytrade_level_checks(daytrade_profile: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(daytrade_profile, dict):
            return []
        thresholds = daytrade_profile.get("thresholds", {}) if isinstance(daytrade_profile.get("thresholds", {}), dict) else {}

        def _f(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        checks: list[dict[str, Any]] = []
        check_specs = [
            ("Avg Bar Move", "avg_bar_move_pct", "min_avg_bar_move_pct"),
            ("Avg Bar Range", "avg_bar_range_pct", "min_avg_bar_range_pct"),
            ("Recent Bar Move", "recent_bar_move_pct", "min_recent_bar_move_pct"),
            ("Volume Ratio", "volume_ratio", "min_volume_ratio"),
            ("Daytrade Score", "score", "min_score"),
        ]
        for label, value_key, threshold_key in check_specs:
            value_num = _f(daytrade_profile.get(value_key, 0.0))
            threshold_num = _f(thresholds.get(threshold_key, 0.0))
            checks.append(
                {
                    "label": label,
                    "value": round(value_num, 3),
                    "threshold": round(threshold_num, 3),
                    "passed": bool(value_num >= threshold_num),
                }
            )

        early = daytrade_profile.get("early_climb", {}) if isinstance(daytrade_profile.get("early_climb", {}), dict) else {}
        checks.append(
            {
                "label": "Early Climb",
                "value": str("on" if bool(early.get("eligible", False)) else "off"),
                "threshold": "eligible",
                "passed": bool(early.get("eligible", False)),
            }
        )
        return checks

    def _top_held_symbol(self) -> str:
        with self._lock:
            positions = dict(self.telemetry_cache.get("positions", {"paper": [], "live": []}))
        best_symbol = ""
        best_move = -1e9
        for row in positions.get("paper", []):
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            entry = float(row.get("avg_entry", 0.0) or 0.0)
            current = float(row.get("current_price", 0.0) or 0.0)
            if not symbol or entry <= 0 or current <= 0:
                continue
            move = ((current - entry) / entry) * 100.0
            if move > best_move:
                best_move = move
                best_symbol = symbol
        return best_symbol

    def _latest_council_vote_summary(self) -> dict[str, Any]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, decision_json, created_at
                FROM ai_decisions
                ORDER BY id DESC
                LIMIT 80
                """
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["decision_json"] or "{}"))
            except Exception:
                payload = {}
            decision = payload.get("decision", payload)
            raw = payload.get("raw", {}) if isinstance(payload.get("raw", {}), dict) else {}
            votes = raw.get("votes", []) if isinstance(raw.get("votes", []), list) else []
            used_council = bool(payload.get("used_council", False)) or bool(raw.get("council", False)) or bool(votes)
            if not used_council:
                continue

            why = str(decision.get("rationale") or decision.get("reason") or "")
            member_votes: list[dict[str, Any]] = []
            vote_counts = {"buy": 0, "hold": 0, "avoid": 0}
            for v in votes[:3]:
                if not isinstance(v, dict):
                    continue
                agent = str(v.get("agent", "agent"))
                agent_l = agent.lower()
                if "technician" in agent_l:
                    role = "Technician"
                elif "sentinel" in agent_l:
                    role = "Sentinel"
                elif "strategist" in agent_l:
                    role = "Strategist"
                else:
                    role = "Council Member"
                action = str(v.get("action", "")).upper()
                rationale = str(v.get("rationale", ""))
                if action == "BUY":
                    vote_counts["buy"] += 1
                elif action == "HOLD":
                    vote_counts["hold"] += 1
                elif action == "AVOID":
                    vote_counts["avoid"] += 1
                member_votes.append(
                    {
                        "member": agent,
                        "role": role,
                        "action": action,
                        "confidence": float(v.get("confidence", 0.0) or 0.0),
                        "why": rationale[:1200],
                    }
                )

            return {
                "decision_id": int(row["id"]),
                "symbol": str(row["symbol"] or ""),
                "kind": self._guess_kind(str(row["symbol"] or "")),
                "action": str(decision.get("action", "")).upper(),
                "confidence": float(decision.get("confidence", 0.0) or 0.0),
                "why": why[:300],
                "votes": member_votes,
                "vote_counts": vote_counts,
                "created_at": str(row["created_at"] or ""),
            }

        return {
            "decision_id": 0,
            "symbol": "n/a",
            "kind": "n/a",
            "action": "N/A",
            "confidence": 0.0,
            "why": "No council vote captured yet.",
            "votes": [],
            "vote_counts": {"buy": 0, "hold": 0, "avoid": 0},
            "created_at": "",
        }

    def _learning_panels_snapshot(self) -> dict[str, Any]:
        council = self._latest_council_vote_summary()

        top_held_symbol = self._top_held_symbol()
        top_held_decision = self._latest_decision_payload_for_symbol(top_held_symbol)
        top_held_payload = top_held_decision.get("payload", {}) if isinstance(top_held_decision, dict) else {}
        top_held_decision_obj = top_held_payload.get("decision", top_held_payload) if isinstance(top_held_payload, dict) else {}
        top_held_profile = top_held_decision_obj.get("daytrade_profile", {}) if isinstance(top_held_decision_obj, dict) else {}

        with self._lock:
            mover = dict(self.telemetry_cache.get("highest_mover_today", {}))
        mover_symbol = str(mover.get("symbol", "")).upper().replace("-USD", "")
        mover_decision = self._latest_decision_payload_for_symbol(mover_symbol)
        mover_payload = mover_decision.get("payload", {}) if isinstance(mover_decision, dict) else {}
        mover_decision_obj = mover_payload.get("decision", mover_payload) if isinstance(mover_payload, dict) else {}
        mover_profile = mover_decision_obj.get("daytrade_profile", {}) if isinstance(mover_decision_obj, dict) else {}

        if not mover_profile:
            scan = self._latest_cycle_scan().get("scan", {}) if isinstance(self._latest_cycle_scan(), dict) else {}
            skipped = self._latest_cycle_scan().get("skipped", []) if isinstance(self._latest_cycle_scan(), dict) else []
            if isinstance(skipped, list):
                for row in skipped:
                    if not isinstance(row, dict):
                        continue
                    sym = str(row.get("symbol", "")).upper().replace("-USD", "")
                    if sym == mover_symbol and isinstance(row.get("daytrade_profile", {}), dict):
                        mover_profile = row.get("daytrade_profile", {})
                        break
            if not mover_profile and isinstance(scan, dict):
                picks = scan.get("top_picks", []) if isinstance(scan.get("top_picks", []), list) else []
                for pick in picks:
                    if not isinstance(pick, dict):
                        continue
                    sym = str(pick.get("symbol", "")).upper().replace("-USD", "")
                    if sym == mover_symbol:
                        mover_profile = {
                            "score": float(pick.get("score", 0.0) or 0.0),
                            "avg_bar_move_pct": float(pick.get("avg_daily_move_pct", 0.0) or 0.0),
                            "recent_bar_move_pct": float(pick.get("last_day_move_pct", 0.0) or 0.0),
                            "avg_bar_range_pct": 0.0,
                            "volume_ratio": float(pick.get("volume_vs_avg", 0.0) or 0.0),
                            "thresholds": {},
                            "early_climb": {},
                        }
                        break

        return {
            "latest_council_vote": council,
            "top_held": {
                "symbol": top_held_symbol or "n/a",
                "decision_id": int(top_held_decision.get("id", 0) or 0) if isinstance(top_held_decision, dict) else 0,
                "entry_path": str((top_held_profile or {}).get("eligible_path", "")),
                "checks": self._build_daytrade_level_checks(top_held_profile),
            },
            "highest_mover": {
                "symbol": mover_symbol or "n/a",
                "move_pct": float(mover.get("move_pct", 0.0) or 0.0),
                "entry_path": str((mover_profile or {}).get("eligible_path", "")),
                "checks": self._build_daytrade_level_checks(mover_profile),
            },
        }

    def _highest_mover_today(self) -> dict[str, Any]:
        latest_scan_report = self._latest_cycle_scan()
        scan_ctx = latest_scan_report.get("scan", {}) if isinstance(latest_scan_report, dict) else {}
        stocks_tradable_now = bool(scan_ctx.get("stocks_tradable_now", True)) if isinstance(scan_ctx, dict) else True

        tz_name = str(getattr(settings, "timezone", "America/New_York") or "America/New_York")
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = timezone.utc

        today_local = datetime.now(local_tz).date()
        best: dict[str, Any] | None = None

        def _to_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        def _metric_from_pick(pick: dict[str, Any]) -> tuple[float, str]:
            avg_daily = _to_float(pick.get("avg_daily_move_pct"))
            if avg_daily > 0:
                return avg_daily, "avg_daily_move_pct"
            last_day = abs(_to_float(pick.get("last_day_move_pct")))
            if last_day > 0:
                return last_day, "last_day_move_pct"
            avg_move_alt = _to_float(pick.get("avg_move_pct"))
            if avg_move_alt > 0:
                return avg_move_alt, "avg_move_pct"
            return 0.0, "avg_daily_move_pct"

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT metadata_json, created_at
                FROM system_events
                WHERE event_type = 'trend_scan'
                ORDER BY id DESC
                LIMIT 500
                """
            ).fetchall()

        for row in rows:
            created_raw = str(row["created_at"] or "")
            try:
                created_dt = datetime.fromisoformat(created_raw)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if created_dt.astimezone(local_tz).date() != today_local:
                    continue
            except Exception:
                continue

            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}

            picks = metadata.get("top_picks", [])
            if not isinstance(picks, list):
                continue

            for pick in picks:
                if not isinstance(pick, dict):
                    continue
                symbol = str(pick.get("symbol", "")).upper()
                if not symbol:
                    continue
                kind = str(pick.get("kind", self._guess_kind(symbol))).lower()
                if not stocks_tradable_now and kind == "stock":
                    continue
                move_pct, metric = _metric_from_pick(pick)
                if best is None or move_pct > float(best.get("move_pct", 0.0) or 0.0):
                    best = {
                        "symbol": symbol,
                        "kind": kind,
                        "move_pct": round(move_pct, 2),
                        "as_of": created_raw,
                        "metric": metric,
                    }

        if best is None or float(best.get("move_pct", 0.0) or 0.0) <= 0.0:
            scan = latest_scan_report.get("scan", {}) if isinstance(latest_scan_report, dict) else {}
            cycle_picks = scan.get("top_picks", []) if isinstance(scan, dict) else []
            if isinstance(cycle_picks, list):
                for pick in cycle_picks:
                    if not isinstance(pick, dict):
                        continue
                    symbol = str(pick.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    kind = str(pick.get("kind", self._guess_kind(symbol))).lower()
                    if not stocks_tradable_now and kind == "stock":
                        continue
                    move_pct, metric = _metric_from_pick(pick)
                    if best is None or move_pct > float(best.get("move_pct", 0.0) or 0.0):
                        best = {
                            "symbol": symbol,
                            "kind": kind,
                            "move_pct": round(move_pct, 2),
                            "as_of": str(self.telemetry_cache.get("updated_at") or ""),
                            "metric": metric,
                        }

        return best or {
            "symbol": "n/a",
            "kind": "n/a",
            "move_pct": 0.0,
            "as_of": "",
            "metric": "avg_daily_move_pct",
        }

    def _trading_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                wait_seconds = self._seconds_until_next_candle(self._trade_timeframe)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                time.sleep(self._candle_delay_seconds)

                if runtime_state.kill_switch:
                    continue
                if not runtime_state.is_running:
                    continue

                self.engine.run_daily_cycle()
            except Exception as exc:
                logger.exception("Trading loop cycle failed: {}", exc)
                with self._lock:
                    errs = self.telemetry_cache.setdefault("errors", [])
                    errs.append({"time": self._now_iso(), "message": str(exc)})
                    self.telemetry_cache["errors"] = errs[-200:]
                time.sleep(2.0)

    def _scanner_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = time.time()
                if now - self._last_full_scan_at >= self._full_scan_seconds:
                    picks = scan_for_bullish_opportunities(
                        include_penny=True,
                        include_micro_crypto=True,
                        include_trending=bool(settings.dynamic_discovery_enabled),
                        min_score=settings.scan_min_score,
                        top_n=max(10, settings.scan_top_n),
                        min_avg_daily_move_stock_pct=settings.scan_min_avg_daily_move_stock_pct,
                        min_avg_daily_move_crypto_pct=settings.scan_min_avg_daily_move_crypto_pct,
                        min_last_day_move_pct=settings.scan_min_last_day_move_pct,
                        min_volume_vs_avg=settings.scan_min_volume_vs_avg,
                        volatility_weight=settings.scan_volatility_weight,
                        activity_weight=settings.scan_activity_weight,
                    )
                    self._last_full_scan_at = now
                    self._update_watchlist_from_scan(picks)

                if now - self._last_watchlist_update_at >= self._watchlist_refresh_seconds:
                    self._refresh_watchlist_from_cycle()
                    self._last_watchlist_update_at = now
            except Exception as exc:
                logger.warning("Scanner loop failed: {}", exc)
            time.sleep(3.0)

    def _shadow_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.shadow.maybe_retrain()
            except Exception as exc:
                logger.warning("Shadow loop failed: {}", exc)
            time.sleep(60.0)

    @staticmethod
    def _seconds_until_next_candle(timeframe: str) -> float:
        now = datetime.now(timezone.utc)
        tf = (timeframe or "1m").strip().lower()
        if tf.endswith("m"):
            mins = max(1, int(tf[:-1] or "1"))
            next_minute = (now.minute // mins + 1) * mins
            if next_minute >= 60:
                target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                target = now.replace(minute=next_minute, second=0, microsecond=0)
            return max(0.0, (target - now).total_seconds())
        if tf.endswith("h"):
            hours = max(1, int(tf[:-1] or "1"))
            next_hour = (now.hour // hours + 1) * hours
            if next_hour >= 24:
                target = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                target = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            return max(0.0, (target - now).total_seconds())
        target = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return max(0.0, (target - now).total_seconds())

    def _bot_state_value(self) -> str:
        if runtime_state.kill_switch:
            return "KILLED_BY_RISK"
        if not runtime_state.is_running:
            return "PAUSED"
        if runtime_state.consecutive_failures > 0:
            return "ERROR"
        return "RUNNING"

    def _paper_account_snapshot(self) -> dict[str, Any]:
        with get_connection() as conn:
            pnl_all = conn.execute("SELECT COALESCE(SUM(pnl), 0.0) AS pnl FROM trade_outcomes").fetchone()
            pnl_today = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) AS pnl FROM trade_outcomes WHERE closed_at >= date('now', 'start of day')"
            ).fetchone()
            exposure = conn.execute(
                """
                SELECT COALESCE(SUM(quantity * price), 0.0) AS notional
                FROM orders
                WHERE mode='paper' AND status IN ('paper_proposed', 'paper_filled')
                """
            ).fetchone()
        all_time = float(pnl_all["pnl"] if pnl_all else 0.0)
        today = float(pnl_today["pnl"] if pnl_today else 0.0)
        cap = float(settings.paper_working_capital_usd)
        wiggle = float(getattr(settings, "paper_extra_play_cash_usd", 0.0) or 0.0)
        base = cap + wiggle
        equity = max(0.0, base + all_time)
        open_notional = float(exposure["notional"] if exposure else 0.0)
        cash = max(0.0, equity - open_notional)
        return {
            "paper_equity": round(equity, 4),
            "paper_cash": round(cash, 4),
            "paper_pnl_today": round(today, 4),
            "paper_pnl_all_time": round(all_time, 4),
            "paper_max_capital": round(cap, 4),
            "paper_wiggle_capital": round(wiggle, 4),
        }

    def _live_account_snapshot(self) -> dict[str, Any]:
        account = self.robinhood.get_account_snapshot()
        totals = self.robinhood.get_portfolio_totals()
        equity = float(totals.get("equity", 0.0) or 0.0)
        prev = float(totals.get("portfolio_equity_previous_close", 0.0) or 0.0)
        pnl_live = equity - prev if prev > 0 else 0.0

        remaining_budget = max(0.0, float(self.engine.risk_policy.max_daily_realized_loss_pct) * max(1.0, equity) - max(0.0, -pnl_live))

        return {
            "session_authenticated": bool(account.get("session_authenticated", False)),
            "cash": float(account.get("cash", 0.0) or 0.0),
            "buying_power": float(account.get("buying_power", 0.0) or 0.0),
            "total_value": equity,
            "live_pnl": round(pnl_live, 4),
            "remaining_daily_loss_budget": round(remaining_budget, 4),
        }

    def _positions_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        paper_rows: list[dict[str, Any]] = []
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, quantity, price, created_at, rationale
                FROM orders
                WHERE mode='paper' AND status='paper_filled'
                ORDER BY created_at DESC
                """
            ).fetchall()

        now = datetime.now(timezone.utc)
        for row in rows:
            symbol = str(row["symbol"] or "")
            qty = float(row["quantity"] or 0.0)
            entry = float(row["price"] or 0.0)
            kind = self._guess_kind(symbol)
            quote = self.robinhood.get_quote(symbol, kind)
            current = float(quote.get("mark_price", entry) or entry)
            pnl = (current - entry) * qty

            age_minutes = 0
            try:
                created = datetime.fromisoformat(str(row["created_at"] or ""))
                age_minutes = int((now - created).total_seconds() // 60)
            except Exception:
                age_minutes = 0

            paper_rows.append(
                {
                    "order_id": int(row["id"]),
                    "symbol": symbol,
                    "kind": kind,
                    "qty": qty,
                    "avg_entry": round(entry, 8),
                    "current_price": round(current, 8),
                    "unrealized_pnl": round(pnl, 4),
                    "stop": round(entry * 0.96, 8),
                    "takeprofit": round(entry * 1.08, 8),
                    "age_minutes": age_minutes,
                    "strategy_mode": "council" if settings.use_council else "single_agent",
                }
            )

        live_rows = self._live_positions_broker()
        return {"paper": paper_rows, "live": live_rows}

    def _live_positions_broker(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            import robin_stocks.robinhood as rh

            stock_positions = rh.account.get_open_stock_positions() or []
            for p in stock_positions:
                if not isinstance(p, dict):
                    continue
                qty = float(p.get("quantity") or 0.0)
                if qty <= 0:
                    continue
                instrument = str(p.get("instrument") or "")
                symbol = instrument.split("/")[-2] if "/" in instrument else instrument
                rows.append(
                    {
                        "symbol": symbol,
                        "kind": "stock",
                        "qty": qty,
                        "avg_entry": float(p.get("average_buy_price") or 0.0),
                        "current_price": 0.0,
                        "unrealized_pnl": 0.0,
                        "source": "broker",
                    }
                )
        except Exception:
            pass
        return rows

    def _orders_snapshot(self, limit: int = 100) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, mode, side, quantity, price, status, rationale, created_at,
                       decision_confidence, decision_score, ai_provider
                FROM orders
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "symbol": str(r["symbol"]),
                "kind": self._guess_kind(str(r["symbol"])),
                "mode": str(r["mode"]),
                "side": str(r["side"]),
                "quantity": float(r["quantity"] or 0.0),
                "price": float(r["price"] or 0.0),
                "status": str(r["status"]),
                "rationale": str(r["rationale"] or ""),
                "created_at": str(r["created_at"] or ""),
                "decision_confidence": float(r["decision_confidence"] or 0.0),
                "decision_score": float(r["decision_score"] or 0.0),
                "ai_provider": str(r["ai_provider"] or ""),
            }
            for r in rows
        ]

    def _latest_errors(self) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, message, metadata_json, created_at
                FROM system_events
                WHERE event_type IN ('pipeline_error', 'kill_switch')
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        output: list[dict[str, Any]] = []
        for r in rows:
            try:
                metadata = json.loads(str(r["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            output.append(
                {
                    "id": int(r["id"]),
                    "message": str(r["message"] or ""),
                    "metadata": metadata,
                    "created_at": str(r["created_at"] or ""),
                }
            )
        return output

    def _latest_alerts(self) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, event_type, message, metadata_json, created_at
                FROM system_events
                WHERE event_type IN (
                    'drawdown_alert',
                    'quality_alert',
                    'order_blocked',
                    'order_blocked_goal',
                    'order_blocked_quality_guard',
                    'order_blocked_council_unanimous',
                    'order_blocked_daytrade_filter',
                    'order_blocked_market_closed',
                    'order_blocked_paper_cap',
                    'order_blocked_live_cap'
                )
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        alerts: list[dict[str, Any]] = []
        for r in rows:
            try:
                metadata = json.loads(str(r["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            alerts.append(
                {
                    "id": int(r["id"]),
                    "type": str(r["event_type"]),
                    "message": str(r["message"]),
                    "metadata": metadata,
                    "created_at": str(r["created_at"]),
                    "acked": False,
                }
            )
        return alerts

    def _latest_cycle_scan(self) -> dict[str, Any]:
        return self._read_json(CYCLE_REPORT_PATH, {})

    def _refresh_watchlist_from_cycle(self) -> None:
        report = self._latest_cycle_scan()
        scan = report.get("scan", {}) if isinstance(report, dict) else {}
        picks = scan.get("top_picks", []) if isinstance(scan, dict) else []
        cycle_targets = report.get("targets", []) if isinstance(report, dict) else []
        if not isinstance(picks, list):
            picks = []
        if not isinstance(cycle_targets, list):
            cycle_targets = []

        enriched: list[dict[str, Any]] = []
        with get_connection() as conn:
            for p in picks[:30]:
                if not isinstance(p, dict):
                    continue
                symbol = str(p.get("symbol", "")).upper()
                if not symbol:
                    continue
                decision_row = conn.execute(
                    """
                    SELECT decision_json, created_at
                    FROM ai_decisions
                    WHERE UPPER(symbol) = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (symbol,),
                ).fetchone()

                proposal = {}
                last_time = ""
                if decision_row:
                    try:
                        payload = json.loads(str(decision_row["decision_json"] or "{}"))
                    except Exception:
                        payload = {}
                    proposal = payload.get("decision", payload)
                    last_time = str(decision_row["created_at"] or "")

                reasons = []
                score = float(p.get("score", 0.0) or 0.0)
                if score >= 70:
                    reasons.append("High trend score")
                if float(p.get("volume_vs_avg", 0.0) or 0.0) >= 1.2:
                    reasons.append("Volume above average")
                if float(p.get("last_day_move_pct", 0.0) or 0.0) >= 2.0:
                    reasons.append("Strong recent move")
                if not reasons:
                    reasons.append("Scanner shortlisted this symbol")

                enriched.append(
                    {
                        "symbol": symbol,
                        "kind": str(p.get("kind", self._guess_kind(symbol))),
                        "regime": str(p.get("label", "NEUTRAL")),
                        "trend_score": score,
                        "vol_score": float(p.get("volume_vs_avg", 0.0) or 0.0),
                        "last_proposal": str(proposal.get("action", "")),
                        "last_decision_time": last_time,
                        "reasons": reasons,
                    }
                )

            existing = {str(e.get("symbol", "")).upper() for e in enriched}
            for t in cycle_targets[:60]:
                if not isinstance(t, dict):
                    continue
                symbol = str(t.get("symbol", "")).upper()
                if not symbol or symbol in existing:
                    continue
                kind = str(t.get("kind", self._guess_kind(symbol))).lower()
                enriched.append(
                    {
                        "symbol": symbol,
                        "kind": kind,
                        "regime": "TARGET",
                        "trend_score": float(t.get("trend_score", 0.0) or 0.0),
                        "vol_score": float(t.get("volume_vs_avg", 0.0) or 0.0),
                        "last_proposal": "",
                        "last_decision_time": "",
                        "reasons": [
                            f"Cycle target ({str(t.get('selection_reason', 'runtime_target'))})",
                        ],
                    }
                )
                existing.add(symbol)

        with self._lock:
            self.telemetry_cache["watchlist"] = enriched

    def _update_watchlist_from_scan(self, picks: list[Any]) -> None:
        rows = []
        for p in picks[:30]:
            rows.append(
                {
                    "symbol": str(getattr(p, "symbol", "")).upper(),
                    "kind": str(getattr(p, "kind", self._guess_kind(str(getattr(p, "symbol", ""))))),
                    "regime": str(getattr(p, "trend_label", "NEUTRAL")),
                    "trend_score": float(getattr(p, "trend_score", 0.0) or 0.0),
                    "vol_score": float(getattr(p, "volume_vs_avg", 0.0) or 0.0),
                    "last_proposal": "",
                    "last_decision_time": "",
                    "reasons": [
                        "Feature snapshot indicates bullish setup",
                        "Proposal confidence near shortlist threshold",
                    ],
                }
            )
        with self._lock:
            self.telemetry_cache["watchlist"] = rows

    def status(self) -> dict[str, Any]:
        with self._lock:
            base = dict(self.telemetry_cache)
        base["readiness"] = self.readiness()
        return base

    def positions(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.telemetry_cache.get("positions", {"paper": [], "live": []}))

    def orders(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.telemetry_cache.get("orders", []))

    def sells(self, limit: int = 200) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.id AS trade_id,
                    t.order_id,
                    t.pnl,
                    t.hold_minutes,
                    t.max_drawdown,
                    t.closed_at,
                    o.symbol,
                    o.mode,
                    o.quantity,
                    o.price,
                    o.created_at
                FROM trade_outcomes t
                JOIN orders o ON o.id = t.order_id
                ORDER BY t.id DESC
                LIMIT ?
                """,
                (max(1, min(1000, int(limit))),),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            qty = float(r["quantity"] or 0.0)
            entry_price = float(r["price"] or 0.0)
            pnl = float(r["pnl"] or 0.0)
            exit_price_est = None
            if qty > 0 and entry_price > 0:
                exit_price_est = entry_price + (pnl / qty)
            out.append(
                {
                    "trade_id": int(r["trade_id"]),
                    "order_id": int(r["order_id"] or 0),
                    "symbol": str(r["symbol"] or ""),
                    "kind": self._guess_kind(str(r["symbol"] or "")),
                    "mode": str(r["mode"] or ""),
                    "qty": qty,
                    "avg_entry": entry_price,
                    "exit_price_est": float(exit_price_est) if exit_price_est is not None else 0.0,
                    "realized_pnl": pnl,
                    "hold_minutes": int(r["hold_minutes"] or 0),
                    "max_drawdown": float(r["max_drawdown"] or 0.0),
                    "opened_at": str(r["created_at"] or ""),
                    "closed_at": str(r["closed_at"] or ""),
                }
            )
        return out

    def candidates(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.telemetry_cache.get("watchlist", []))

    def decisions_recent(self, symbol: str = "", action: str = "", limit: int = 200) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if symbol:
            where.append("UPPER(symbol) = ?")
            params.append(symbol.upper())

        query = """
            SELECT id, symbol, model_name, score, confidence, rationale, decision_json, created_at
            FROM ai_decisions
        """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(500, int(limit))))

        with get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            symbol_text = str(r["symbol"])
            kind = self._guess_kind(symbol_text)
            try:
                payload = json.loads(str(r["decision_json"] or "{}"))
            except Exception:
                payload = {}
            decision = payload.get("decision", payload)
            final_action = str(decision.get("action", "")).upper()
            if action and final_action != action.upper():
                continue

            votes = payload.get("raw", {}).get("votes", payload.get("raw", {}))
            out.append(
                {
                    "id": int(r["id"]),
                    "symbol": symbol_text,
                    "kind": kind,
                    "asset_kind": kind,
                    "model_name": str(r["model_name"]),
                    "score": float(r["score"] or 0.0),
                    "confidence": float(r["confidence"] or 0.0),
                    "rationale": str(r["rationale"] or ""),
                    "created_at": str(r["created_at"]),
                    "action": final_action,
                    "entry_path": str((decision.get("daytrade_profile", {}) or {}).get("eligible_path", "")),
                    "decision": decision,
                    "votes": votes,
                    "risk_manager": {
                        "blocked": final_action != "BUY",
                        "reason": str(r["rationale"] or ""),
                    },
                }
            )

        return out[: max(1, min(500, int(limit)))]

    def decision_by_id(self, decision_id: int) -> dict[str, Any]:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, symbol, model_name, score, confidence, rationale, decision_json, created_at
                FROM ai_decisions
                WHERE id = ?
                LIMIT 1
                """,
                (int(decision_id),),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Decision not found")

        try:
            payload = json.loads(str(row["decision_json"] or "{}"))
        except Exception:
            payload = {}
        decision = payload.get("decision", payload)
        raw = payload.get("raw", {})
        return {
            "id": int(row["id"]),
            "symbol": str(row["symbol"]),
            "model_name": str(row["model_name"]),
            "score": float(row["score"] or 0.0),
            "confidence": float(row["confidence"] or 0.0),
            "rationale": str(row["rationale"] or ""),
            "created_at": str(row["created_at"]),
            "final_action": str(decision.get("action", "")).upper(),
            "technician_vote": raw.get("technician_vote", raw.get("technician", {})),
            "sentinel_vote": raw.get("sentinel_vote", raw.get("sentinel", {})),
            "strategist_vote": raw.get("strategist_vote", raw.get("strategist", {})),
            "risk_manager": {
                "blocked": str(decision.get("action", "")).upper() != "BUY",
                "reason": str(row["rationale"] or ""),
            },
            "decision": decision,
            "raw": raw,
        }

    def pause(self) -> dict[str, Any]:
        runtime_state.is_running = False
        return {"ok": True, "bot_state": self._bot_state_value()}

    def resume(self) -> dict[str, Any]:
        runtime_state.is_running = True
        return {"ok": True, "bot_state": self._bot_state_value()}

    def flatten(self) -> dict[str, Any]:
        return self._flatten_impl(graceful=False)

    def safe_flatten(self) -> dict[str, Any]:
        try:
            return self._flatten_impl(graceful=True)
        except Exception:
            return self._flatten_impl(graceful=False)

    def _flatten_impl(self, graceful: bool) -> dict[str, Any]:
        now_iso = self._now_iso()
        closed_paper = 0
        live_attempts = 0
        live_errors: list[str] = []

        with get_connection() as conn:
            paper_rows = conn.execute(
                """
                SELECT id, symbol, quantity, price, created_at
                FROM orders
                WHERE mode='paper' AND status IN ('paper_proposed', 'paper_filled')
                ORDER BY id ASC
                """
            ).fetchall()

            for r in paper_rows:
                order_id = int(r["id"])
                symbol = str(r["symbol"])
                qty = float(r["quantity"] or 0.0)
                entry = float(r["price"] or 0.0)
                kind = "crypto" if "-" in symbol else "stock"
                quote = self.robinhood.get_quote(symbol, kind)
                mark = float(quote.get("mark_price", entry) or entry)
                pnl = (mark - entry) * qty

                conn.execute("UPDATE orders SET status='paper_closed' WHERE id=?", (order_id,))
                conn.execute(
                    """
                    INSERT INTO trade_outcomes (order_id, pnl, hold_minutes, max_drawdown, closed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (order_id, pnl, 0, 0.0, now_iso),
                )
                closed_paper += 1

            if not graceful:
                live_rows = conn.execute(
                    """
                    SELECT id, symbol, quantity, status
                    FROM orders
                    WHERE mode='live' AND status IN ('live_ready', 'live_submitted', 'live_filled')
                    ORDER BY id ASC
                    """
                ).fetchall()
            else:
                live_rows = conn.execute(
                    """
                    SELECT id, symbol, quantity, status
                    FROM orders
                    WHERE mode='live' AND status IN ('live_submitted', 'live_filled')
                    ORDER BY id ASC
                    """
                ).fetchall()

            for r in live_rows:
                symbol = str(r["symbol"])
                qty = float(r["quantity"] or 0.0)
                if qty <= 0:
                    continue
                kind = self._guess_kind(symbol)
                try:
                    self.robinhood.place_sell_order(symbol, kind, qty)
                    conn.execute("UPDATE orders SET status='live_closed' WHERE id=?", (int(r["id"]),))
                except Exception as exc:
                    live_errors.append(f"{symbol}: {exc}")
                    if not graceful:
                        conn.execute("UPDATE orders SET status='live_close_failed' WHERE id=?", (int(r["id"]),))
                live_attempts += 1

            conn.commit()

        with self._lock:
            self.telemetry_cache.setdefault("alerts", []).insert(
                0,
                {
                    "id": int(time.time()),
                    "type": "flatten",
                    "message": f"Flatten executed (paper_closed={closed_paper}, live_attempts={live_attempts})",
                    "metadata": {"graceful": graceful, "live_errors": live_errors},
                    "created_at": now_iso,
                    "acked": False,
                },
            )

        return {
            "ok": True,
            "graceful": graceful,
            "paper_closed": closed_paper,
            "live_attempts": live_attempts,
            "live_errors": live_errors,
        }

    def paper_size(self, direction: str, pct: float) -> dict[str, Any]:
        base = float(settings.paper_working_capital_usd)
        pct = float(pct)
        if direction == "increase":
            new_value = base * (1.0 + pct)
        else:
            new_value = base * (1.0 - pct)
        settings.paper_working_capital_usd = max(0.0, round(new_value, 2))
        persisted = _update_env_values(
            {
                "PAPER_WORKING_CAPITAL_USD": float(settings.paper_working_capital_usd),
                "PAPER_EXTRA_PLAY_CASH_USD": float(getattr(settings, "paper_extra_play_cash_usd", 0.0) or 0.0),
            }
        )
        with self._lock:
            self.telemetry_cache["paper"] = self._paper_account_snapshot()
        return {
            "ok": True,
            "paper_max_capital": float(settings.paper_working_capital_usd),
            "paper_wiggle_capital": float(getattr(settings, "paper_extra_play_cash_usd", 0.0) or 0.0),
            "persisted": persisted,
        }

    def set_paper_capital(self, max_capital: float, wiggle_capital: float) -> dict[str, Any]:
        settings.paper_working_capital_usd = max(0.0, round(float(max_capital), 2))
        settings.paper_extra_play_cash_usd = max(0.0, round(float(wiggle_capital), 2))
        persisted = _update_env_values(
            {
                "PAPER_WORKING_CAPITAL_USD": float(settings.paper_working_capital_usd),
                "PAPER_EXTRA_PLAY_CASH_USD": float(settings.paper_extra_play_cash_usd),
            }
        )
        with self._lock:
            self.telemetry_cache["paper"] = self._paper_account_snapshot()
        return {
            "ok": True,
            "paper_max_capital": float(settings.paper_working_capital_usd),
            "paper_wiggle_capital": float(settings.paper_extra_play_cash_usd),
            "persisted": persisted,
        }

    def clear_errors(self) -> dict[str, Any]:
        with self._lock:
            self.telemetry_cache["errors"] = []
            alerts = self.telemetry_cache.get("alerts", [])
            for a in alerts:
                a["acked"] = True
            self.telemetry_cache["alerts"] = alerts
        return {"ok": True}

    def adaptive_toggles(self) -> dict[str, Any]:
        return {
            "ok": True,
            **self._adaptive_snapshot(),
        }

    def set_adaptive_toggles(self, payload: AdaptiveTogglesPayload) -> dict[str, Any]:
        settings.adaptive_special_circumstances_enabled = bool(payload.adaptive_special_circumstances_enabled)
        settings.adaptive_special_circumstances_paper_enabled = bool(payload.adaptive_special_circumstances_paper_enabled)
        settings.adaptive_special_circumstances_live_enabled = bool(payload.adaptive_special_circumstances_live_enabled)
        settings.adaptive_technician_enabled = bool(payload.adaptive_technician_enabled)
        settings.adaptive_hourly_rule_generation_enabled = bool(payload.adaptive_hourly_rule_generation_enabled)
        scope = str(payload.adaptive_exception_scope or "asset_class").lower()
        if scope not in {"symbol", "asset_class", "global"}:
            scope = "asset_class"
        settings.adaptive_exception_scope = scope

        persisted = _update_env_values(
            {
                "ADAPTIVE_SPECIAL_CIRCUMSTANCES_ENABLED": bool(settings.adaptive_special_circumstances_enabled),
                "ADAPTIVE_SPECIAL_CIRCUMSTANCES_PAPER_ENABLED": bool(settings.adaptive_special_circumstances_paper_enabled),
                "ADAPTIVE_SPECIAL_CIRCUMSTANCES_LIVE_ENABLED": bool(settings.adaptive_special_circumstances_live_enabled),
                "ADAPTIVE_TECHNICIAN_ENABLED": bool(settings.adaptive_technician_enabled),
                "ADAPTIVE_HOURLY_RULE_GENERATION_ENABLED": bool(settings.adaptive_hourly_rule_generation_enabled),
                "ADAPTIVE_EXCEPTION_SCOPE": str(settings.adaptive_exception_scope),
            }
        )

        with self._lock:
            self.telemetry_cache["adaptive"] = self._adaptive_snapshot()

        return {
            "ok": True,
            "persisted": persisted,
            **self._adaptive_snapshot(),
        }

    def readiness(self) -> dict[str, Any]:
        status = self.status_minimal()
        paper = status.get("paper", {})

        hours_runtime = self._continuous_runtime_hours()
        flatten_ok = self._flatten_tested_recently()
        idempotent_ok = self._idempotency_check()
        rate_limit_ok = self._rate_limit_tested()

        op_score = 0
        op_score += min(40, int(hours_runtime * 4))
        op_score += 20 if flatten_ok else 0
        op_score += 20 if idempotent_ok else 0
        op_score += 20 if rate_limit_ok else 0

        perf = self._performance_readiness(paper)
        shadow = self._shadow_readiness(perf)

        return {
            "label": "Readiness meters are estimates only",
            "operational": {
                "score": max(0, min(100, op_score)),
                "is_estimate": True,
                "factors": {
                    "continuous_runtime_hours": round(hours_runtime, 2),
                    "flatten_tested": flatten_ok,
                    "idempotency_verified": idempotent_ok,
                    "rate_limiting_tested": rate_limit_ok,
                },
            },
            "performance": perf,
            "shadow": shadow,
        }

    def status_minimal(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paper": dict(self.telemetry_cache.get("paper", {})),
                "live": dict(self.telemetry_cache.get("live", {})),
            }

    @staticmethod
    def _continuous_runtime_hours() -> float:
        if runtime_state.last_run_started_at and runtime_state.last_run_finished_at:
            delta = runtime_state.last_run_finished_at - runtime_state.last_run_started_at
            return max(0.0, delta.total_seconds() / 3600.0)
        if runtime_state.last_run_started_at:
            return max(0.0, (datetime.now(timezone.utc) - runtime_state.last_run_started_at).total_seconds() / 3600.0)
        return 0.0

    @staticmethod
    def _flatten_tested_recently() -> bool:
        return (RUNTIME_DIR / "post_unanimous_check.txt").exists()

    @staticmethod
    def _idempotency_check() -> bool:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol, created_at, COUNT(1) AS cnt
                FROM orders
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY symbol, created_at
                HAVING cnt > 1
                LIMIT 1
                """
            ).fetchone()
        return rows is None

    @staticmethod
    def _rate_limit_tested() -> bool:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS n
                FROM system_events
                WHERE event_type='order_blocked_live_cap'
                  AND created_at >= datetime('now', '-30 days')
                """
            ).fetchone()
        return int(row["n"] if row else 0) > 0

    def _performance_readiness(self, paper_status: dict[str, Any]) -> dict[str, Any]:
        with get_connection() as conn:
            stats = conn.execute(
                """
                SELECT COUNT(1) AS trades,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                       COALESCE(SUM(pnl), 0.0) AS pnl
                FROM trade_outcomes
                """
            ).fetchone()
            dd_row = conn.execute(
                """
                SELECT COALESCE(MIN(max_drawdown), 0.0) AS min_dd
                FROM trade_outcomes
                """
            ).fetchone()

        trades = int(stats["trades"] if stats else 0)
        wins = int(stats["wins"] if stats else 0)
        pnl = float(stats["pnl"] if stats else 0.0)
        expectancy = (pnl / trades) if trades > 0 else 0.0
        max_dd = abs(float(dd_row["min_dd"] if dd_row else 0.0))

        score = 0
        score += min(40, int((trades / 50) * 40))
        score += 30 if max_dd <= 0.12 else 10
        score += 30 if expectancy > 0 else 0

        return {
            "score": max(0, min(100, score)),
            "is_estimate": True,
            "factors": {
                "trades": trades,
                "wins": wins,
                "expectancy_with_slippage": round(expectancy, 6),
                "max_drawdown": round(max_dd, 6),
                "paper_pnl_today": float(paper_status.get("paper_pnl_today", 0.0) or 0.0),
            },
        }

    def _shadow_readiness(self, perf_readiness: dict[str, Any]) -> dict[str, Any]:
        shadow_status = self.shadow.get_status()
        portfolio = self.shadow.get_portfolio()
        dataset = int(shadow_status.get("dataset_size", 0) or 0)
        walk = float(shadow_status.get("walk_forward_score", 0.0) or 0.0)
        rolling = float(shadow_status.get("resolved_win_rate", 0.0) or 0.0)

        outperf_margin = 0.0
        if portfolio:
            recent = portfolio[-1]
            outperf_margin = float(recent.get("equity", 10000.0) - 10000.0)

        drift_ok = bool(shadow_status.get("model_health", "cold") != "needs_attention")

        score = 0
        score += min(35, int((dataset / 300) * 35))
        score += min(35, int(max(0.0, walk) * 35))
        score += 20 if outperf_margin > 0 else 5
        score += 10 if drift_ok else 0

        return {
            "score": max(0, min(100, score)),
            "is_estimate": True,
            "factors": {
                "dataset_size": dataset,
                "walk_forward_stability": round(walk, 4),
                "shadow_vs_level1_margin": round(outperf_margin, 4),
                "rolling_score": round(rolling, 4),
                "drift_ok": drift_ok,
            },
        }

    def active_symbols(self) -> list[dict[str, Any]]:
        with self._lock:
            positions = dict(self.telemetry_cache.get("positions", {"paper": [], "live": []}))
            watchlist = list(self.telemetry_cache.get("watchlist", []))

        held: set[str] = set()
        for row in positions.get("paper", []):
            sym = str((row or {}).get("symbol", "")).upper().replace("-USD", "")
            if sym:
                held.add(sym)
        for row in positions.get("live", []):
            sym = str((row or {}).get("symbol", "")).upper().replace("-USD", "")
            if sym:
                held.add(sym)

        symbols: set[str] = set(held)
        for row in watchlist:
            sym = str((row or {}).get("symbol", "")).upper().replace("-USD", "")
            if sym:
                symbols.add(sym)

        if not symbols:
            symbols = set(sorted(self._kind_map.keys())[:50])

        ordered = sorted(symbols, key=lambda s: (0 if s in held else 1, s))
        return [
            {
                "symbol": sym,
                "kind": self._guess_kind(sym),
                "in_position": sym in held,
            }
            for sym in ordered
        ]

    @staticmethod
    def _rolling_sma(values: list[float], period: int) -> list[float | None]:
        out: list[float | None] = []
        if period <= 0:
            return [None for _ in values]
        for i in range(len(values)):
            if i + 1 < period:
                out.append(None)
                continue
            window = values[i - period + 1 : i + 1]
            out.append(sum(window) / period)
        return out

    @staticmethod
    def _rolling_ema(values: list[float], period: int) -> list[float | None]:
        if not values:
            return []
        alpha = 2.0 / (float(period) + 1.0)
        out: list[float | None] = []
        ema: float | None = None
        for i, value in enumerate(values):
            if i + 1 < period:
                out.append(None)
                continue
            if ema is None:
                seed = values[i - period + 1 : i + 1]
                ema = sum(seed) / float(period)
            else:
                ema = (value * alpha) + (ema * (1.0 - alpha))
            out.append(ema)
        return out

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _parse_candle(raw: dict[str, Any]) -> dict[str, Any] | None:
        ts = str(raw.get("begins_at") or raw.get("timestamp") or raw.get("date") or "")
        if not ts:
            return None
        close = DashboardController._safe_float(raw.get("close_price", raw.get("close")), 0.0)
        open_ = DashboardController._safe_float(raw.get("open_price", raw.get("open")), close)
        high = DashboardController._safe_float(raw.get("high_price", raw.get("high")), close)
        low = DashboardController._safe_float(raw.get("low_price", raw.get("low")), close)
        volume = DashboardController._safe_float(raw.get("volume"), 0.0)
        if close <= 0.0 and open_ <= 0.0 and high <= 0.0 and low <= 0.0:
            return None
        return {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def _load_cached_chart_rows(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        tf = (timeframe or "live").lower()
        cache_candidates: list[str]
        if tf == "hourly":
            cache_candidates = [f"{symbol}_60d_1h.json", f"{symbol}_2y_1d.json", f"{symbol}_1y_1d.json"]
        elif tf == "minute":
            cache_candidates = [f"{symbol}_60d_1h.json", f"{symbol}_1y_1d.json"]
        else:
            cache_candidates = [f"{symbol}_60d_1h.json", f"{symbol}_1y_1d.json"]

        base = Path("data/historical_cache")
        for name in cache_candidates:
            path = base / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            rows: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_candle(item)
                if parsed:
                    rows.append(parsed)
            if rows:
                return rows
        return []

    def _position_levels(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            positions = dict(self.telemetry_cache.get("positions", {"paper": [], "live": []}))

        needle = str(symbol or "").upper().replace("-USD", "")
        for row in positions.get("paper", []):
            sym = str((row or {}).get("symbol", "")).upper().replace("-USD", "")
            if sym != needle:
                continue
            entry = self._safe_float((row or {}).get("avg_entry"), 0.0)
            stop = self._safe_float((row or {}).get("stop"), 0.0)
            takeprofit = self._safe_float((row or {}).get("takeprofit"), 0.0)
            return {
                "entry": entry if entry > 0 else None,
                "stop": stop if stop > 0 else None,
                "takeprofit": takeprofit if takeprofit > 0 else None,
                "strategy_mode": str((row or {}).get("strategy_mode") or ""),
            }

        for row in positions.get("live", []):
            sym = str((row or {}).get("symbol", "")).upper().replace("-USD", "")
            if sym != needle:
                continue
            entry = self._safe_float((row or {}).get("avg_entry"), 0.0)
            return {
                "entry": entry if entry > 0 else None,
                "stop": None,
                "takeprofit": None,
                "strategy_mode": "live",
            }

        return {"entry": None, "stop": None, "takeprofit": None, "strategy_mode": ""}

    def chart_series(self, symbol: str, timeframe: str = "live") -> dict[str, Any]:
        sym = str(symbol or "").upper().replace("-USD", "")
        if not sym:
            raise HTTPException(status_code=400, detail="symbol is required")

        tf = (timeframe or "live").strip().lower()
        if tf not in {"live", "hourly", "minute"}:
            raise HTTPException(status_code=400, detail="timeframe must be one of: live, hourly, minute")

        kind = self._guess_kind(sym)
        mapping = {
            "live": {"interval": "5minute", "span": "day", "max_points": 200},
            "minute": {"interval": "5minute", "span": "day", "max_points": 280},
            "hourly": {"interval": "hour", "span": "month", "max_points": 320},
        }
        cfg = mapping[tf]

        source = "broker"
        rows: list[dict[str, Any]] = []
        try:
            raw = self.robinhood.get_historicals(sym, kind=kind, interval=str(cfg["interval"]), span=str(cfg["span"]))
            for item in raw:
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_candle(item)
                if parsed:
                    rows.append(parsed)
        except Exception:
            rows = []

        if not rows:
            rows = self._load_cached_chart_rows(sym, tf)
            source = "cache"

        rows = sorted(rows, key=lambda r: str(r.get("timestamp", "")))
        max_points = int(cfg["max_points"])
        if len(rows) > max_points:
            rows = rows[-max_points:]

        if tf == "live":
            try:
                quote = self.robinhood.get_quote(sym, kind=kind)
                mark = self._safe_float((quote or {}).get("mark_price"), 0.0)
                if mark > 0:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    if rows:
                        rows[-1]["close"] = mark
                        rows[-1]["high"] = max(float(rows[-1].get("high", mark) or mark), mark)
                        rows[-1]["low"] = min(float(rows[-1].get("low", mark) or mark), mark)
                    else:
                        rows.append(
                            {
                                "timestamp": now_iso,
                                "open": mark,
                                "high": mark,
                                "low": mark,
                                "close": mark,
                                "volume": 0.0,
                            }
                        )
            except Exception:
                pass

        closes = [self._safe_float(r.get("close"), 0.0) for r in rows]
        sma_fast = self._rolling_sma(closes, 7)
        sma_slow = self._rolling_sma(closes, 21)
        ema_9 = self._rolling_ema(closes, 9)
        levels = self._position_levels(sym)

        points: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            points.append(
                {
                    "t": row.get("timestamp"),
                    "price": float(closes[idx]),
                    "sma_fast": float(sma_fast[idx]) if sma_fast[idx] is not None else None,
                    "sma_slow": float(sma_slow[idx]) if sma_slow[idx] is not None else None,
                    "ema_9": float(ema_9[idx]) if ema_9[idx] is not None else None,
                    "entry": levels.get("entry"),
                    "stop": levels.get("stop"),
                    "takeprofit": levels.get("takeprofit"),
                }
            )

        return {
            "symbol": sym,
            "kind": kind,
            "timeframe": tf,
            "strategy_mode": levels.get("strategy_mode", ""),
            "points": points,
            "meta": {
                "source": source,
                "interval": cfg["interval"],
                "span": cfg["span"],
                "points": len(points),
                "in_position": bool(levels.get("entry") is not None),
            },
        }


controller = DashboardController()


app = FastAPI(title="TradingBot Dashboard API", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    controller.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    controller.stop()


static_path = Path("ui/static")
if static_path.exists():
    app.mount("/dashboard/static", StaticFiles(directory=str(static_path)), name="dashboard-static")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    html_path = Path("ui/static/dashboard.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard page missing")
    return FileResponse(html_path)


@app.get("/")
def root_page() -> RedirectResponse:
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/status")
def get_status() -> dict[str, Any]:
    return controller.status()


@app.get("/positions")
def get_positions() -> dict[str, Any]:
    return controller.positions()


@app.get("/orders")
def get_orders() -> list[dict[str, Any]]:
    return controller.orders()


@app.get("/sells")
def get_sells(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
    return controller.sells(limit=limit)


@app.get("/candidates")
def get_candidates() -> list[dict[str, Any]]:
    return controller.candidates()


@app.get("/symbols/active")
def get_active_symbols() -> list[dict[str, Any]]:
    return controller.active_symbols()


@app.get("/chart/series")
def get_chart_series(
    symbol: str = Query(..., min_length=1),
    timeframe: str = Query(default="live"),
) -> dict[str, Any]:
    return controller.chart_series(symbol=symbol, timeframe=timeframe)


@app.get("/decisions/recent")
def get_decisions_recent(
    symbol: str = Query(default=""),
    action: str = Query(default=""),
    limit: int = Query(default=200, ge=1, le=500),
) -> list[dict[str, Any]]:
    return controller.decisions_recent(symbol=symbol, action=action, limit=limit)


@app.get("/decision/{decision_id}")
def get_decision(decision_id: int) -> dict[str, Any]:
    return controller.decision_by_id(decision_id)


@app.post("/pause")
def post_pause() -> dict[str, Any]:
    return controller.pause()


@app.post("/resume")
def post_resume() -> dict[str, Any]:
    return controller.resume()


@app.post("/flatten")
def post_flatten() -> dict[str, Any]:
    return controller.flatten()


@app.post("/safe_flatten")
def post_safe_flatten() -> dict[str, Any]:
    return controller.safe_flatten()


@app.post("/paper/size")
def post_paper_size(payload: PaperSizePayload) -> dict[str, Any]:
    return controller.paper_size(direction=payload.direction, pct=payload.pct)


@app.post("/paper/capital")
def post_paper_capital(payload: PaperCapitalPayload) -> dict[str, Any]:
    return controller.set_paper_capital(max_capital=payload.max_capital, wiggle_capital=payload.wiggle_capital)


@app.post("/alerts/ack")
def post_alerts_ack() -> dict[str, Any]:
    return controller.clear_errors()


@app.get("/adaptive/toggles")
def get_adaptive_toggles() -> dict[str, Any]:
    return controller.adaptive_toggles()


@app.post("/adaptive/toggles")
def post_adaptive_toggles(payload: AdaptiveTogglesPayload) -> dict[str, Any]:
    return controller.set_adaptive_toggles(payload)


@app.get("/shadow/status")
def get_shadow_status() -> dict[str, Any]:
    return controller.shadow.get_status()


@app.get("/shadow/portfolio")
def get_shadow_portfolio() -> list[dict[str, Any]]:
    return controller.shadow.get_portfolio()


@app.get("/shadow/predictions")
def get_shadow_predictions(symbol: str = Query(default="")) -> list[dict[str, Any]]:
    return controller.shadow.get_predictions(symbol=symbol)


@app.get("/readiness")
def get_readiness() -> dict[str, Any]:
    return controller.readiness()


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "time": datetime.now(timezone.utc).isoformat()})
