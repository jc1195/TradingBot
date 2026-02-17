from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from .db import get_connection
from .watchlist import PENNY_WATCHLIST

try:
    sklearn_linear_model = importlib.import_module("sklearn.linear_model")
    sklearn_metrics = importlib.import_module("sklearn.metrics")
    sklearn_model_selection = importlib.import_module("sklearn.model_selection")
    LogisticRegression = sklearn_linear_model.LogisticRegression
    accuracy_score = sklearn_metrics.accuracy_score
    TimeSeriesSplit = sklearn_model_selection.TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None
    accuracy_score = None
    TimeSeriesSplit = None
    SKLEARN_AVAILABLE = False

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    optim = importlib.import_module("torch.optim")
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


RUNTIME_DIR = Path("runtime")
PREDICTIONS_PATH = RUNTIME_DIR / "shadow_predictions.json"
MODEL_STATUS_PATH = RUNTIME_DIR / "shadow_model_status.json"
SHADOW_PORTFOLIO_PATH = RUNTIME_DIR / "shadow_portfolio.json"
PENNY_SYMBOLS = {str(item.symbol).upper() for item in PENNY_WATCHLIST}
FEATURE_NAMES = [
    "trend_score",
    "confidence",
    "daytrade_score",
    "risk_flags_count",
    "action_num",
    "avg_bar_move_pct",
    "recent_bar_move_pct",
    "avg_bar_range_pct",
    "volume_ratio",
    "early_climb_eligible",
    "early_climb_net_climb_pct",
    "early_climb_acceleration_pct",
    "special_applied_count",
    "special_override_daytrade_min_score",
    "eligible_path_code",
]


@dataclass
class TrainingSummary:
    trained_at: str
    dataset_size: int
    rolling_score: float
    walk_forward_score: float
    model_health: str
    baseline_available: bool
    mlp_available: bool


class ShadowMLService:
    def __init__(self, horizon_minutes: int = 120, retrain_min_new_examples: int = 50) -> None:
        self.horizon_minutes = max(15, int(horizon_minutes))
        self.retrain_min_new_examples = max(20, int(retrain_min_new_examples))
        self._last_processed_decision_id = 0
        self._last_train_dataset_size = 0
        self._ensure_runtime_files()
        self._sync_last_processed_id_from_predictions()

    @staticmethod
    def _ensure_runtime_files() -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        if not PREDICTIONS_PATH.exists():
            PREDICTIONS_PATH.write_text("[]", encoding="utf-8")
        if not SHADOW_PORTFOLIO_PATH.exists():
            SHADOW_PORTFOLIO_PATH.write_text("[]", encoding="utf-8")
        if not MODEL_STATUS_PATH.exists():
            MODEL_STATUS_PATH.write_text(
                json.dumps(
                    {
                        "trained_at": "",
                        "dataset_size": 0,
                        "rolling_score": 0.0,
                        "walk_forward_score": 0.0,
                        "model_health": "cold",
                        "baseline_available": SKLEARN_AVAILABLE,
                        "mlp_available": TORCH_AVAILABLE,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _load_json(path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return fallback

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _sync_last_processed_id_from_predictions(self) -> None:
        rows = self._load_json(PREDICTIONS_PATH, [])
        if not isinstance(rows, list):
            return
        max_id = 0
        for row in rows:
            try:
                max_id = max(max_id, int(row.get("decision_id", 0) or 0))
            except Exception:
                continue
        self._last_processed_decision_id = max_id

    def _extract_features(self, decision_payload: dict[str, Any]) -> list[float]:
        decision = decision_payload.get("decision", decision_payload)
        trend_score = float(decision.get("trend_score", 0.0) or 0.0)
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        daytrade_score = float(decision.get("daytrade_score", 0.0) or 0.0)
        risk_count = float(len(decision.get("risk_flags", []) or []))
        action = str(decision.get("action", "HOLD")).upper()
        action_num = 1.0 if action == "BUY" else (0.5 if action == "HOLD" else 0.0)
        profile = decision.get("daytrade_profile", {}) if isinstance(decision, dict) else {}
        if not isinstance(profile, dict):
            profile = {}
        early = profile.get("early_climb", {}) if isinstance(profile.get("early_climb", {}), dict) else {}
        special = profile.get("special_circumstances", {}) if isinstance(profile.get("special_circumstances", {}), dict) else {}
        applied = special.get("applied", []) if isinstance(special.get("applied", []), list) else []

        eligible_path = str(profile.get("eligible_path", "") or "").strip().lower()
        eligible_path_code = 0.0
        if eligible_path == "strict":
            eligible_path_code = 1.0
        elif eligible_path == "early_climb":
            eligible_path_code = 0.8
        elif eligible_path == "special_circumstance":
            eligible_path_code = 0.9

        return [
            trend_score,
            confidence,
            daytrade_score,
            risk_count,
            action_num,
            float(profile.get("avg_bar_move_pct", 0.0) or 0.0),
            float(profile.get("recent_bar_move_pct", 0.0) or 0.0),
            float(profile.get("avg_bar_range_pct", 0.0) or 0.0),
            float(profile.get("volume_ratio", 0.0) or 0.0),
            1.0 if bool(early.get("eligible", False)) else 0.0,
            float(early.get("net_climb_pct", 0.0) or 0.0),
            float(early.get("acceleration_pct", 0.0) or 0.0),
            float(len(applied)),
            1.0 if bool(special.get("override_daytrade_min_score", False)) else 0.0,
            eligible_path_code,
        ]

    def _build_labeled_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        rows_x: list[list[float]] = []
        rows_y: list[int] = []

        with get_connection() as conn:
            decisions = conn.execute(
                """
                SELECT d.id, d.symbol, d.decision_json, d.created_at
                FROM ai_decisions d
                ORDER BY d.id ASC
                """
            ).fetchall()

            for row in decisions:
                try:
                    payload = json.loads(str(row["decision_json"] or "{}"))
                except Exception:
                    payload = {}
                features = self._extract_features(payload)
                symbol = str(row["symbol"] or "").upper()
                created_at = str(row["created_at"] or "")

                close = conn.execute(
                    """
                    SELECT t.pnl, t.hold_minutes
                    FROM trade_outcomes t
                    JOIN orders o ON o.id = t.order_id
                    WHERE UPPER(o.symbol) = ?
                      AND o.created_at >= ?
                    ORDER BY t.id ASC
                    LIMIT 1
                    """,
                    (symbol, created_at),
                ).fetchone()

                if not close:
                    continue

                pnl = float(close["pnl"] or 0.0)
                hold_minutes = int(close["hold_minutes"] or 0)
                if hold_minutes > self.horizon_minutes:
                    label = 0
                else:
                    label = 1 if pnl > 0 else 0

                rows_x.append(features)
                rows_y.append(label)

        if not rows_x:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float), np.empty((0,), dtype=int)
        return np.array(rows_x, dtype=float), np.array(rows_y, dtype=int)

    def _train_baseline(self, x: np.ndarray, y: np.ndarray) -> tuple[Any, float]:
        if not SKLEARN_AVAILABLE or len(x) < 30:
            return None, 0.0

        model = LogisticRegression(max_iter=500)
        model.fit(x, y)
        preds = model.predict(x)
        score = float(accuracy_score(y, preds))
        return model, score

    def _walk_forward_score(self, x: np.ndarray, y: np.ndarray) -> float:
        if not SKLEARN_AVAILABLE or len(x) < 40:
            return 0.0

        splitter = TimeSeriesSplit(n_splits=4)
        scores: list[float] = []
        for train_idx, test_idx in splitter.split(x):
            if len(train_idx) < 20 or len(test_idx) < 5:
                continue
            m = LogisticRegression(max_iter=500)
            m.fit(x[train_idx], y[train_idx])
            p = m.predict(x[test_idx])
            scores.append(float(accuracy_score(y[test_idx], p)))
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _train_mlp(self, x: np.ndarray, y: np.ndarray) -> tuple[Any, float]:
        if not TORCH_AVAILABLE or len(x) < 40:
            return None, 0.0
        input_dim = int(x.shape[1]) if len(x.shape) >= 2 else len(FEATURE_NAMES)
        input_dim = max(1, input_dim)

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 24),
                    nn.ReLU(),
                    nn.Linear(24, 12),
                    nn.ReLU(),
                    nn.Linear(12, 1),
                    nn.Sigmoid(),
                )

            def forward(self, input_x: Any) -> Any:
                return self.net(input_x)

        model = MLP()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        model.train()
        for _ in range(80):
            optimizer.zero_grad()
            out = model(x_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = model(x_t).numpy().reshape(-1)
            preds = (probs >= 0.5).astype(int)
            score = float((preds == y).mean())
        return model, score

    def maybe_retrain(self) -> TrainingSummary:
        self._ingest_recent_decisions()
        x, y = self._build_labeled_dataset()
        dataset_size = int(len(x))
        status = self._load_json(MODEL_STATUS_PATH, {})

        if dataset_size == 0:
            return TrainingSummary(
                trained_at=str(status.get("trained_at", "")),
                dataset_size=0,
                rolling_score=0.0,
                walk_forward_score=0.0,
                model_health="cold",
                baseline_available=SKLEARN_AVAILABLE,
                mlp_available=TORCH_AVAILABLE,
            )

        should_train = (
            dataset_size - self._last_train_dataset_size >= self.retrain_min_new_examples
            or not status.get("trained_at")
        )

        baseline_model = None
        mlp_model = None
        baseline_score = float(status.get("rolling_score", 0.0) or 0.0)
        mlp_score = float(status.get("mlp_score", 0.0) or 0.0)
        walk_score = float(status.get("walk_forward_score", 0.0) or 0.0)

        if should_train:
            baseline_model, baseline_score = self._train_baseline(x, y)
            mlp_model, mlp_score = self._train_mlp(x, y)
            walk_score = self._walk_forward_score(x, y)
            health = "good" if max(baseline_score, mlp_score, walk_score) >= 0.58 else "needs_attention"
            payload = {
                "trained_at": self._now_iso(),
                "dataset_size": dataset_size,
                "rolling_score": round(baseline_score, 4),
                "mlp_score": round(mlp_score, 4),
                "walk_forward_score": round(walk_score, 4),
                "model_health": health,
                "baseline_available": SKLEARN_AVAILABLE,
                "mlp_available": TORCH_AVAILABLE,
            }
            self._save_json(MODEL_STATUS_PATH, payload)
            self._last_train_dataset_size = dataset_size
            logger.info("Shadow ML retrained: size={} baseline={:.3f} mlp={:.3f}", dataset_size, baseline_score, mlp_score)

            if baseline_model is not None or mlp_model is not None:
                self._ingest_recent_decisions(baseline_model, mlp_model)

        latest = self._load_json(MODEL_STATUS_PATH, {})
        if not isinstance(latest, dict):
            latest = {}
        latest["baseline_available"] = bool(SKLEARN_AVAILABLE)
        latest["mlp_available"] = bool(TORCH_AVAILABLE)
        self._save_json(MODEL_STATUS_PATH, latest)
        return TrainingSummary(
            trained_at=str(latest.get("trained_at", "")),
            dataset_size=int(latest.get("dataset_size", 0) or 0),
            rolling_score=float(latest.get("rolling_score", 0.0) or 0.0),
            walk_forward_score=float(latest.get("walk_forward_score", 0.0) or 0.0),
            model_health=str(latest.get("model_health", "cold")),
            baseline_available=bool(latest.get("baseline_available", False) or SKLEARN_AVAILABLE),
            mlp_available=bool(latest.get("mlp_available", False) or TORCH_AVAILABLE),
        )

    def _ingest_recent_decisions(self, baseline_model: Any = None, mlp_model: Any = None) -> None:
        predictions = self._load_json(PREDICTIONS_PATH, [])
        if not isinstance(predictions, list):
            predictions = []

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, symbol, decision_json, created_at
                FROM ai_decisions
                WHERE id > ?
                ORDER BY id ASC
                LIMIT 400
                """,
                (self._last_processed_decision_id,),
            ).fetchall()

        for row in rows:
            decision_id = int(row["id"])
            symbol = str(row["symbol"] or "").upper()
            kind = self._asset_kind(symbol)
            is_penny = self._is_penny_symbol(symbol)
            tradable_now = True if kind == "crypto" else self._us_stock_market_open()
            created_at = str(row["created_at"] or self._now_iso())
            try:
                payload = json.loads(str(row["decision_json"] or "{}"))
            except Exception:
                payload = {}

            features = np.array(self._extract_features(payload), dtype=float).reshape(1, -1)
            decision_conf = float(payload.get("decision", payload).get("confidence", 0.5) or 0.5)
            trend_score = float(payload.get("decision", payload).get("trend_score", 50.0) or 50.0)
            daytrade_score = float(payload.get("decision", payload).get("daytrade_score", 0.0) or 0.0)
            decision_obj = payload.get("decision", payload)
            profile = decision_obj.get("daytrade_profile", {}) if isinstance(decision_obj, dict) else {}
            special = profile.get("special_circumstances", {}) if isinstance(profile, dict) and isinstance(profile.get("special_circumstances", {}), dict) else {}
            exception_boost = 0.06 if bool(special.get("applied", [])) else 0.0
            baseline_conf = min(0.95, max(0.05, decision_conf))
            mlp_conf = min(
                0.95,
                max(
                    0.05,
                    (0.35 * baseline_conf)
                    + (0.35 * (trend_score / 100.0))
                    + (0.2 * max(0.0, min(1.0, daytrade_score / 100.0)))
                    + exception_boost,
                ),
            )

            if baseline_model is not None and SKLEARN_AVAILABLE:
                baseline_conf = float(baseline_model.predict_proba(features)[0][1])

            if mlp_model is not None and TORCH_AVAILABLE:
                with torch.no_grad():
                    mlp_conf = float(mlp_model(torch.tensor(features, dtype=torch.float32)).numpy().reshape(-1)[0])

            baseline_action = "BUY" if baseline_conf >= 0.55 else "AVOID"
            mlp_action = "BUY" if mlp_conf >= 0.55 else "AVOID"
            blocked_market_closed = bool(kind == "stock" and not tradable_now)
            if blocked_market_closed:
                baseline_action = "AVOID"
                mlp_action = "AVOID"
            label_due = (self._parse_iso(created_at) + timedelta(minutes=self.horizon_minutes)).isoformat()

            predictions.append(
                {
                    "decision_id": decision_id,
                    "symbol": symbol,
                    "asset_kind": kind,
                    "asset_tag": "penny" if is_penny else kind,
                    "is_penny": bool(is_penny),
                    "tradable_now": bool(tradable_now),
                    "created_at": created_at,
                    "features": {name: float(features[0][idx]) for idx, name in enumerate(FEATURE_NAMES)},
                    "special_exception_applied": bool(exception_boost > 0),
                    "eligible_path": str((profile or {}).get("eligible_path", "")),
                    "baseline": {
                        "action": baseline_action,
                        "confidence": round(baseline_conf, 4),
                        "predicted_ev_proxy": round((baseline_conf - 0.5) * 2.0, 4),
                        "model_source": "logreg" if baseline_model is not None else "heuristic_cold_start",
                        "blocked_market_closed": blocked_market_closed,
                    },
                    "mlp": {
                        "action": mlp_action,
                        "confidence": round(mlp_conf, 4),
                        "predicted_ev_proxy": round((mlp_conf - 0.5) * 2.0, 4),
                        "model_source": "mlp" if mlp_model is not None else "heuristic_cold_start",
                        "blocked_market_closed": blocked_market_closed,
                    },
                    "realized_label": None,
                    "resolved_at": "",
                    "label_due_at": label_due,
                }
            )
            self._last_processed_decision_id = max(self._last_processed_decision_id, decision_id)

        predictions = predictions[-600:]
        self._save_json(PREDICTIONS_PATH, predictions)
        self._resolve_labels()
        self._refresh_shadow_portfolio()

        status = self._load_json(MODEL_STATUS_PATH, {})
        if not isinstance(status, dict):
            status = {}
        status["last_ingest_at"] = self._now_iso()
        status["predictions_total"] = len(predictions)
        status["pending_labels"] = sum(1 for p in predictions if p.get("realized_label") is None)
        status["last_prediction_at"] = str(predictions[-1].get("created_at", "")) if predictions else ""
        self._save_json(MODEL_STATUS_PATH, status)

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _asset_kind(symbol: str) -> str:
        sym = str(symbol or "").upper()
        return "crypto" if "-USD" in sym else "stock"

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        return str(symbol or "").upper().replace("-USD", "")

    @classmethod
    def _is_penny_symbol(cls, symbol: str) -> bool:
        return cls._clean_symbol(symbol) in PENNY_SYMBOLS

    @staticmethod
    def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        day_num = 1 + offset + (occurrence - 1) * 7
        return date(year, month, day_num)

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        cursor = next_month - timedelta(days=1)
        while cursor.weekday() != weekday:
            cursor -= timedelta(days=1)
        return cursor

    @staticmethod
    def _observed_us_holiday(day: date) -> date:
        if day.weekday() == 5:
            return day - timedelta(days=1)
        if day.weekday() == 6:
            return day + timedelta(days=1)
        return day

    @staticmethod
    def _easter_sunday(year: int) -> date:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day_num = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day_num)

    @classmethod
    def _us_equity_holidays(cls, year: int) -> set[date]:
        new_year = cls._observed_us_holiday(date(year, 1, 1))
        mlk = cls._nth_weekday_of_month(year, 1, 0, 3)
        presidents = cls._nth_weekday_of_month(year, 2, 0, 3)
        good_friday = cls._easter_sunday(year) - timedelta(days=2)
        memorial = cls._last_weekday_of_month(year, 5, 0)
        juneteenth = cls._observed_us_holiday(date(year, 6, 19))
        independence = cls._observed_us_holiday(date(year, 7, 4))
        labor = cls._nth_weekday_of_month(year, 9, 0, 1)
        thanksgiving = cls._nth_weekday_of_month(year, 11, 3, 4)
        christmas = cls._observed_us_holiday(date(year, 12, 25))

        holidays = {
            new_year,
            mlk,
            presidents,
            good_friday,
            memorial,
            juneteenth,
            independence,
            labor,
            thanksgiving,
            christmas,
        }

        next_year_new_year_observed = cls._observed_us_holiday(date(year + 1, 1, 1))
        if next_year_new_year_observed.year == year:
            holidays.add(next_year_new_year_observed)

        return holidays

    @classmethod
    def _us_stock_market_open(cls, now_utc: datetime | None = None) -> bool:
        now = now_utc or datetime.now(timezone.utc)
        et = now.astimezone(ZoneInfo("America/New_York"))
        day = et.date()
        if et.weekday() >= 5:
            return False
        if day in cls._us_equity_holidays(day.year):
            return False
        if et.time() < clock_time(9, 30):
            return False
        if et.time() >= clock_time(16, 0):
            return False
        return True

    def _resolve_labels(self) -> None:
        predictions = self._load_json(PREDICTIONS_PATH, [])
        if not isinstance(predictions, list) or not predictions:
            return

        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            for item in predictions:
                if item.get("realized_label") is not None:
                    continue
                due_at = self._parse_iso(str(item.get("label_due_at", "")))
                if due_at > now:
                    continue

                symbol = str(item.get("symbol", "")).upper()
                created_at = str(item.get("created_at", ""))
                close = conn.execute(
                    """
                    SELECT t.pnl
                    FROM trade_outcomes t
                    JOIN orders o ON o.id = t.order_id
                    WHERE UPPER(o.symbol) = ?
                      AND o.created_at >= ?
                    ORDER BY t.id ASC
                    LIMIT 1
                    """,
                    (symbol, created_at),
                ).fetchone()

                if close:
                    pnl = float(close["pnl"] or 0.0)
                    item["realized_label"] = 1 if pnl > 0 else 0
                    item["resolved_at"] = self._now_iso()
                else:
                    item["realized_label"] = 0
                    item["resolved_at"] = self._now_iso()

        self._save_json(PREDICTIONS_PATH, predictions)

    def _refresh_shadow_portfolio(self) -> None:
        predictions = self._load_json(PREDICTIONS_PATH, [])
        if not isinstance(predictions, list):
            predictions = []

        portfolio: list[dict[str, Any]] = []
        equity = 10000.0
        for item in predictions[-300:]:
            label = item.get("realized_label")
            if label is None:
                continue
            baseline = item.get("baseline", {})
            mlp = item.get("mlp", {})
            baseline_take = str(baseline.get("action", "")).upper() == "BUY"
            mlp_take = str(mlp.get("action", "")).upper() == "BUY"

            baseline_pnl = 0.0
            mlp_pnl = 0.0
            if baseline_take:
                baseline_pnl = 15.0 if int(label) == 1 else -12.0
            if mlp_take:
                mlp_pnl = 15.0 if int(label) == 1 else -12.0

            combined = baseline_pnl * 0.4 + mlp_pnl * 0.6
            equity += combined
            portfolio.append(
                {
                    "decision_id": item.get("decision_id"),
                    "symbol": item.get("symbol"),
                    "asset_kind": item.get("asset_kind", self._asset_kind(str(item.get("symbol", "")))),
                    "asset_tag": item.get("asset_tag", ""),
                    "is_penny": bool(item.get("is_penny", False)),
                    "tradable_now": bool(item.get("tradable_now", True)),
                    "resolved_at": item.get("resolved_at"),
                    "baseline_pnl": round(baseline_pnl, 2),
                    "mlp_pnl": round(mlp_pnl, 2),
                    "combined_pnl": round(combined, 2),
                    "equity": round(equity, 2),
                }
            )

        self._save_json(SHADOW_PORTFOLIO_PATH, portfolio[-500:])

    def get_predictions(self, symbol: str = "") -> list[dict[str, Any]]:
        rows = self._load_json(PREDICTIONS_PATH, [])
        if not isinstance(rows, list):
            return []
        if symbol:
            symbol_upper = symbol.upper()
            rows = [r for r in rows if str(r.get("symbol", "")).upper() == symbol_upper]
        out: list[dict[str, Any]] = []
        for row in rows[-200:]:
            item = dict(row)
            sym = str(item.get("symbol", "")).upper()
            kind = str(item.get("asset_kind", "") or self._asset_kind(sym)).lower()
            is_penny = bool(item.get("is_penny", self._is_penny_symbol(sym)))
            tradable_now = bool(item.get("tradable_now", True if kind == "crypto" else self._us_stock_market_open()))

            baseline = dict(item.get("baseline", {}))
            mlp = dict(item.get("mlp", {}))
            blocked_market_closed = bool(kind == "stock" and not tradable_now)
            if blocked_market_closed:
                if str(baseline.get("action", "")).upper() == "BUY":
                    baseline["action"] = "AVOID"
                if str(mlp.get("action", "")).upper() == "BUY":
                    mlp["action"] = "AVOID"
            baseline["blocked_market_closed"] = blocked_market_closed
            mlp["blocked_market_closed"] = blocked_market_closed

            item["asset_kind"] = kind
            item["asset_tag"] = "penny" if is_penny else kind
            item["is_penny"] = is_penny
            item["tradable_now"] = tradable_now
            item["baseline"] = baseline
            item["mlp"] = mlp
            out.append(item)
        return out

    def get_portfolio(self) -> list[dict[str, Any]]:
        rows = self._load_json(SHADOW_PORTFOLIO_PATH, [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows[-200:]:
            item = dict(row)
            sym = str(item.get("symbol", "")).upper()
            kind = str(item.get("asset_kind", "") or self._asset_kind(sym)).lower()
            is_penny = bool(item.get("is_penny", self._is_penny_symbol(sym)))
            tradable_now = bool(item.get("tradable_now", True if kind == "crypto" else self._us_stock_market_open()))
            item["asset_kind"] = kind
            item["asset_tag"] = "penny" if is_penny else kind
            item["is_penny"] = is_penny
            item["tradable_now"] = tradable_now
            out.append(item)
        return out

    def get_status(self) -> dict[str, Any]:
        status = self._load_json(MODEL_STATUS_PATH, {})
        preds = self.get_predictions()
        resolved = [p for p in preds if p.get("realized_label") is not None]
        rolling = 0.0
        if resolved:
            wins = sum(1 for p in resolved if int(p.get("realized_label", 0)) == 1)
            rolling = wins / max(1, len(resolved))

        return {
            "trained_at": str(status.get("trained_at", "")),
            "dataset_size": int(status.get("dataset_size", 0) or 0),
            "rolling_score": float(status.get("rolling_score", 0.0) or 0.0),
            "walk_forward_score": float(status.get("walk_forward_score", 0.0) or 0.0),
            "mlp_score": float(status.get("mlp_score", 0.0) or 0.0),
            "model_health": str(status.get("model_health", "cold")),
            "baseline_available": bool(status.get("baseline_available", False) or SKLEARN_AVAILABLE),
            "mlp_available": bool(status.get("mlp_available", False) or TORCH_AVAILABLE),
            "recent_predictions": len(preds),
            "resolved_predictions": len(resolved),
            "resolved_win_rate": round(rolling, 4),
            "last_ingest_at": str(status.get("last_ingest_at", "")),
            "pending_labels": int(status.get("pending_labels", 0) or 0),
            "predictions_total": int(status.get("predictions_total", len(preds)) or len(preds)),
            "last_prediction_at": str(status.get("last_prediction_at", "")),
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "exception_aware": True,
            "ml_execution_enabled": False,
            "ml_execution_allowed": False,
            "ml_execution_reason": "Read-only toggle unless enable_ml_execution=true",
        }
