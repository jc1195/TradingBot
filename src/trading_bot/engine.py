from __future__ import annotations

import json
import warnings
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

from .alerts import AlertRouter
from .council import TradingCouncil, CouncilDecision
from .db import get_connection
from .health_check import generate_health_report
from .indicators import compute_signals
from .news_client import NewsClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .risk import RiskPolicy
from .robinhood_client import RobinhoodClient
from .sentiment import SentimentAnalyzer, MarketSentiment
from .settings import settings
from .state import runtime_state
from .trend_detector import scan_for_bullish_opportunities, TrendAnalysis
from .watchlist import WatchlistItem, get_watchlist

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class TradingEngine:
    def __init__(self, risk_policy: RiskPolicy) -> None:
        self.risk_policy = risk_policy
        self.robinhood = RobinhoodClient()
        self.news = NewsClient()
        self.ollama = OllamaClient()
        self.openai = OpenAIClient()
        self.alerts = AlertRouter()
        self._council: TradingCouncil | None = None
        self._sentiment_analyzer = SentimentAnalyzer()
        self._market_mood: dict = {}
        self._sentiment_cache: dict[str, MarketSentiment | None] = {}
        self._use_council: bool = settings.use_council
        self._operator_memory_context: str = self._load_operator_memory_context()

    @staticmethod
    def _load_operator_memory_context(max_chars: int = 1200) -> str:
        memory_path = Path("memory.md")
        if not memory_path.exists():
            return ""
        try:
            text = memory_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[-max_chars:]

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

        # If New Year's Day observed lands in previous year (e.g., Jan 1 on Saturday),
        # include it when evaluating that previous year calendar.
        next_year_new_year_observed = cls._observed_us_holiday(date(year + 1, 1, 1))
        if next_year_new_year_observed.year == year:
            holidays.add(next_year_new_year_observed)

        return holidays

    @classmethod
    def _us_stock_market_status(cls, now_utc: datetime | None = None) -> dict[str, object]:
        now = now_utc or datetime.now(timezone.utc)
        et = now.astimezone(ZoneInfo("America/New_York"))
        et_day = et.date()
        if et.weekday() >= 5:
            return {"is_open": False, "reason": "weekend", "timestamp_et": et.isoformat()}

        if et_day in cls._us_equity_holidays(et_day.year):
            return {"is_open": False, "reason": "holiday", "timestamp_et": et.isoformat()}

        open_time = clock_time(9, 30)
        close_time = clock_time(16, 0)
        if et.time() < open_time:
            return {"is_open": False, "reason": "pre_market", "timestamp_et": et.isoformat()}
        if et.time() >= close_time:
            return {"is_open": False, "reason": "after_hours", "timestamp_et": et.isoformat()}
        return {"is_open": True, "reason": "regular_session", "timestamp_et": et.isoformat()}

    @staticmethod
    def _load_runtime_env_floats() -> dict[str, float]:
        env_path = Path(".env")
        defaults = {
            "PAPER_WORKING_CAPITAL_USD": float(settings.paper_working_capital_usd),
            "PAPER_EXTRA_PLAY_CASH_USD": float(getattr(settings, "paper_extra_play_cash_usd", 0.0) or 0.0),
        }
        if not env_path.exists():
            return defaults

        values = dict(defaults)
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in values:
                    continue
                try:
                    values[key] = float(value.strip())
                except Exception:
                    continue
        except Exception:
            return defaults

        return values

    def run_daily_cycle(self) -> None:
        if self._is_external_kill_switch_active():
            runtime_state.kill_switch = True
            self._persist_system_event(
                "kill_switch",
                "External kill switch file is active",
                datetime.now(timezone.utc).isoformat(),
                {"path": settings.external_kill_switch_path},
            )

        if runtime_state.kill_switch or self.risk_policy.kill_switch:
            logger.warning("Kill switch active; skipping cycle")
            return

        runtime_state.mark_start()
        started_at = datetime.now(timezone.utc).isoformat()
        dynamic_discovery_enabled = bool(getattr(settings, "dynamic_discovery_enabled", False))
        cycle_report: dict = {
            "started_at": started_at,
            "status": "running",
            "dynamic_discovery_enabled": dynamic_discovery_enabled,
            "scan": {
                "bullish_pick_count": 0,
                "top_picks": [],
            },
            "targets": [],
            "decisions": [],
            "skipped": [],
            "summary": {},
        }

        try:
            session_ok = self.robinhood.ensure_session()
            account = self.robinhood.get_account_snapshot()
            self._persist_system_event(
                "account_snapshot",
                "Fetched account snapshot",
                started_at,
                {
                    "session_authenticated": session_ok,
                    "buying_power": account.get("buying_power", 0.0),
                    "equity": account.get("equity", 0.0),
                    "mode": account.get("mode", settings.bot_mode),
                },
            )

            provider_name, ai_client = self._resolve_ai_backend()
            try:
                installed = ai_client.list_models()
                model_choice = ai_client.select_best_model(installed)
                ranked_models = ai_client.rank_models(installed)
            except Exception as backend_exc:
                if settings.ai_provider == "auto" and provider_name == "openai":
                    self._persist_system_event(
                        "ai_provider_fallback",
                        "OpenAI backend unavailable, falling back to Ollama",
                        started_at,
                        {"error": str(backend_exc)},
                    )
                    provider_name, ai_client = "ollama", self.ollama
                    installed = ai_client.list_models()
                    model_choice = ai_client.select_best_model(installed)
                    ranked_models = ai_client.rank_models(installed)
                else:
                    raise

            logger.info("Selected AI provider: {} model: {} ({})", provider_name, model_choice.name, model_choice.reason)
            self._persist_system_event(
                "ai_provider_selected",
                f"Using {provider_name} backend",
                started_at,
                {
                    "provider": provider_name,
                    "model": model_choice.name,
                },
            )

            # ── COUNCIL INIT ──
            if self._use_council and provider_name == "ollama":
                self._council = TradingCouncil(
                    ollama=self.ollama,
                    ranked_models=ranked_models,
                )
                try:
                    self._market_mood = self._sentiment_analyzer.get_market_mood()
                except Exception:
                    self._market_mood = {}
                logger.info("3-agent council ENABLED for this cycle")
            else:
                self._council = None

            broker_crypto_symbols = [
                str(s).upper().replace("-USD", "")
                for s in self.robinhood.get_crypto_symbols()
                if str(s).strip()
            ]
            broker_crypto_set = set(broker_crypto_symbols)
            watchlist_crypto_symbols = [
                str(w.symbol).upper().replace("-USD", "")
                for w in get_watchlist(kinds=["crypto"], include_penny=False, include_micro_crypto=True)
                if str(getattr(w, "symbol", "")).strip()
            ]
            if broker_crypto_set:
                watchlist_crypto_symbols = [s for s in watchlist_crypto_symbols if s in broker_crypto_set]
            symbols: list[str] = []
            seen_crypto: set[str] = set()
            for sym in watchlist_crypto_symbols + broker_crypto_symbols:
                clean = str(sym).upper().replace("-USD", "")
                if not clean or clean in seen_crypto:
                    continue
                seen_crypto.add(clean)
                symbols.append(clean)
            stock_market = self._us_stock_market_status()
            stocks_tradable_now = bool(stock_market.get("is_open", False))

            # ── SMART SCAN: Find bullish opportunities across all markets ──
            bullish_picks: list[TrendAnalysis] = []
            try:
                bullish_picks = scan_for_bullish_opportunities(
                    include_penny=True,
                    include_micro_crypto=True,
                    include_trending=dynamic_discovery_enabled,
                    min_score=settings.scan_min_score,
                    top_n=settings.scan_top_n,
                    min_avg_daily_move_stock_pct=settings.scan_min_avg_daily_move_stock_pct,
                    min_avg_daily_move_crypto_pct=settings.scan_min_avg_daily_move_crypto_pct,
                    min_last_day_move_pct=settings.scan_min_last_day_move_pct,
                    min_volume_vs_avg=settings.scan_min_volume_vs_avg,
                    volatility_weight=settings.scan_volatility_weight,
                    activity_weight=settings.scan_activity_weight,
                )
                self._persist_system_event(
                    "trend_scan_mode",
                    "Trend scan mode updated",
                    started_at,
                    {
                        "include_trending": dynamic_discovery_enabled,
                        "max_symbols_per_cycle": int(settings.max_symbols_per_cycle),
                    },
                )
                if bullish_picks:
                    pick_names = [p.symbol for p in bullish_picks[:5]]
                    logger.info("Trend scan found {} bullish assets: {}", len(bullish_picks), pick_names)
                    self._persist_system_event(
                        "trend_scan",
                        f"Found {len(bullish_picks)} bullish opportunities",
                        started_at,
                        {
                            "count": len(bullish_picks),
                            "top_picks": [
                                {
                                    "symbol": p.symbol,
                                    "kind": p.kind,
                                    "score": p.trend_score,
                                    "label": p.trend_label,
                                    "avg_daily_move_pct": float(p.avg_daily_move_pct),
                                    "last_day_move_pct": float(p.last_day_move_pct),
                                    "volume_vs_avg": float(p.volume_vs_avg),
                                }
                                for p in bullish_picks[:10]
                            ],
                        },
                    )
                else:
                    logger.info("No strong bullish signals found — will still check crypto watchlist")

                cycle_report["scan"] = {
                    "bullish_pick_count": len(bullish_picks),
                    "stocks_tradable_now": stocks_tradable_now,
                    "stock_market": stock_market,
                    "top_picks": [
                        {
                            "symbol": p.symbol,
                            "kind": p.kind,
                            "score": int(p.trend_score),
                            "label": p.trend_label,
                            "avg_daily_move_pct": float(p.avg_daily_move_pct),
                            "last_day_move_pct": float(p.last_day_move_pct),
                            "volume_vs_avg": float(p.volume_vs_avg),
                        }
                        for p in bullish_picks[:10]
                    ],
                }
            except Exception as scan_exc:
                logger.warning("Trend scan failed, falling back to crypto-only: {}", scan_exc)
                self._persist_system_event(
                    "trend_scan_failed",
                    f"Trend scan failed: {scan_exc}",
                    started_at,
                    {
                        "include_trending": dynamic_discovery_enabled,
                        "error": str(scan_exc),
                    },
                )
                cycle_report["scan"] = {
                    "bullish_pick_count": 0,
                    "stocks_tradable_now": stocks_tradable_now,
                    "stock_market": stock_market,
                    "top_picks": [],
                    "error": str(scan_exc),
                }

            # Build the trading universe: bullish picks + crypto watchlist
            # Priority: scan-discovered bullish assets first, then crypto defaults
            trade_targets: list[dict] = []

            # Add bullish picks from trend scanner (stocks & crypto)
            seen_symbols: set[str] = set()
            for pick in bullish_picks:
                if pick.kind == "stock" and not stocks_tradable_now:
                    cycle_report["skipped"].append(
                        {
                            "symbol": pick.symbol,
                            "kind": pick.kind,
                            "reason": "stock_market_closed",
                            "selection_reason": "scanner_bullish_pick",
                            "market": stock_market,
                        }
                    )
                    continue
                if pick.symbol not in seen_symbols:
                    trade_targets.append({
                        "symbol": pick.symbol,
                        "kind": pick.kind,
                        "trend_score": pick.trend_score,
                        "avg_daily_move_pct": float(pick.avg_daily_move_pct),
                        "last_day_move_pct": float(pick.last_day_move_pct),
                        "volume_vs_avg": float(pick.volume_vs_avg),
                        "activity_score": float(pick.activity_score),
                        "bullish": True,
                        "selection_reason": "scanner_bullish_pick",
                    })
                    seen_symbols.add(pick.symbol)

            # Add crypto from existing watchlist (if not already in targets)
            # Only add crypto defaults if scanner found < 3 bullish targets
            if len(trade_targets) < 3:
                for sym in symbols:
                    sym_upper = sym.upper().replace("-USD", "")
                    if sym_upper not in seen_symbols:
                        trade_targets.append({
                            "symbol": sym_upper,
                            "kind": "crypto",
                            "trend_score": 0,
                            "avg_daily_move_pct": 0.0,
                            "last_day_move_pct": 0.0,
                            "volume_vs_avg": 0.0,
                            "activity_score": 0.0,
                            "bullish": False,
                            "selection_reason": "watchlist_fallback",
                        })
                        seen_symbols.add(sym_upper)

            # Limit total targets per cycle
            trade_targets = trade_targets[: settings.max_symbols_per_cycle + 5]
            cycle_report["targets"] = [
                {
                    "symbol": str(t.get("symbol", "")),
                    "kind": str(t.get("kind", "")),
                    "trend_score": float(t.get("trend_score", 0.0) or 0.0),
                    "avg_daily_move_pct": float(t.get("avg_daily_move_pct", 0.0) or 0.0),
                    "last_day_move_pct": float(t.get("last_day_move_pct", 0.0) or 0.0),
                    "volume_vs_avg": float(t.get("volume_vs_avg", 0.0) or 0.0),
                    "activity_score": float(t.get("activity_score", 0.0) or 0.0),
                    "bullish": bool(t.get("bullish", False)),
                    "selection_reason": str(t.get("selection_reason", "unknown")),
                }
                for t in trade_targets
            ]
            goal_progress = self._get_profit_goal_progress(started_at)
            goal_pacing = self._compute_goal_pacing_multiplier(started_at)
            self._persist_system_event(
                "goal_progress",
                "Updated goal progress",
                started_at,
                {
                    **goal_progress,
                    **goal_pacing,
                },
            )

            # ── Sort targets: day-trade activity first, then trend score ──
            trade_targets.sort(
                key=lambda t: (
                    float(t.get("activity_score", 0.0) or 0.0),
                    float(t.get("trend_score", 0.0) or 0.0),
                ),
                reverse=True,
            )

            # Council budget: only spend GPU time on the top N most bullish assets.
            # If unanimous council buys are required, run council across all targets so
            # trade eligibility is evaluated consistently by the 3-agent vote.
            COUNCIL_BUDGET = len(trade_targets) if settings.council_require_unanimous_buy else 3

            for idx, target in enumerate(trade_targets):
                symbol = target["symbol"]
                kind = target["kind"]
                is_bullish = target["bullish"]
                trend_score = target["trend_score"]

                if kind == "stock" and not stocks_tradable_now:
                    self._persist_system_event(
                        "symbol_skipped_market_closed",
                        f"Skipped {symbol}: stock market closed",
                        started_at,
                        {
                            "symbol": symbol,
                            "kind": kind,
                            "selection_reason": str(target.get("selection_reason", "unknown")),
                            "market": stock_market,
                        },
                    )
                    cycle_report["skipped"].append(
                        {
                            "symbol": symbol,
                            "kind": kind,
                            "reason": "stock_market_closed",
                            "selection_reason": str(target.get("selection_reason", "unknown")),
                            "market": stock_market,
                        }
                    )
                    continue

                # Fetch quote using unified method (crypto or stock)
                quote = self.robinhood.get_quote(symbol, kind)
                if kind == "crypto":
                    candles = self.robinhood.get_crypto_historicals(symbol)
                else:
                    candles = self.robinhood.get_stock_historicals(symbol)

                if len(candles) < 6:
                    if kind == "crypto":
                        intraday_candles = self.robinhood.get_crypto_historicals(
                            symbol,
                            interval="5minute",
                            span="day",
                        )
                    else:
                        intraday_candles = self.robinhood.get_stock_historicals(
                            symbol,
                            interval="5minute",
                            span="day",
                        )
                    if len(intraday_candles) > len(candles):
                        candles = intraday_candles

                daytrade_profile = self._compute_daytrade_profile(candles, kind)
                daytrade_profile = self._apply_special_circumstances_daytrade(
                    symbol=symbol,
                    kind=kind,
                    daytrade_profile=daytrade_profile,
                    created_at=started_at,
                )
                if settings.daytrade_only_mode and not bool(daytrade_profile.get("eligible", False)):
                    self._persist_system_event(
                        "symbol_skipped_daytrade_filter",
                        f"Skipped {symbol}: insufficient movement/range for day-trading",
                        started_at,
                        {
                            "symbol": symbol,
                            "kind": kind,
                            "selection_reason": str(target.get("selection_reason", "unknown")),
                            "trend_score": float(trend_score or 0.0),
                            "daytrade_profile": daytrade_profile,
                        },
                    )
                    cycle_report["skipped"].append(
                        {
                            "symbol": symbol,
                            "kind": kind,
                            "reason": "daytrade_filter",
                            "selection_reason": str(target.get("selection_reason", "unknown")),
                            "daytrade_profile": daytrade_profile,
                        }
                    )
                    continue

                session_authenticated = bool(quote.get("session_authenticated", False))
                mark_price = float(quote.get("mark_price", 0.0))
                if (
                    settings.bot_mode == "paper"
                    and settings.paper_require_live_market_data
                    and (not session_authenticated or mark_price <= 0)
                ):
                    self._persist_system_event(
                        "symbol_skipped_no_live_data",
                        f"Skipped {symbol} in paper mode due to missing authenticated live quote",
                        started_at,
                        {
                            "symbol": symbol,
                            "session_authenticated": session_authenticated,
                            "mark_price": mark_price,
                            "paper_require_live_market_data": settings.paper_require_live_market_data,
                        },
                    )
                    continue

                self._persist_system_event(
                    "market_snapshot",
                    f"Fetched market data for {symbol}",
                    started_at,
                    {
                        "symbol": symbol,
                        "mark_price": quote.get("mark_price", 0.0),
                        "candles": len(candles),
                    },
                )

                news_items = self.news.fetch_news_for_symbol(symbol)
                self._persist_news_items(news_items, started_at)
                self._persist_system_event(
                    "news_snapshot",
                    f"Fetched news for {symbol}",
                    started_at,
                    {
                        "symbol": symbol,
                        "items": len(news_items),
                    },
                )

                # ── TIERED DECISION SYSTEM ──
                # Top picks (highest trend scores) → full 3-agent council
                # Lower-priority picks → fast indicator-only decision
                use_council_for_this = (
                    self._council
                    and self._use_council
                    and idx < COUNCIL_BUDGET
                    and (is_bullish or settings.council_require_unanimous_buy)
                )

                if use_council_for_this:
                    # ── COUNCIL MODE: 3 agents vote in parallel ──
                    decision, raw = self._council_decide(
                        symbol, kind, candles, mark_price, started_at, daytrade_profile,
                    )
                    provider_label = f"{provider_name}+council"

                elif is_bullish and candles:
                    # ── FAST MODE: indicator-only for lower-priority bullish ──
                    decision, raw = self._fast_indicator_decide(
                        symbol, kind, candles, mark_price, trend_score,
                    )
                    provider_label = f"{provider_name}+indicators"

                else:
                    # ── SINGLE AGENT MODE: for non-bullish / crypto defaults ──
                    prompt = self._build_prompt(
                        symbol,
                        news_items,
                        started_at,
                        kind=kind,
                        is_bullish=is_bullish,
                        trend_score=trend_score,
                        daytrade_profile=daytrade_profile,
                    )
                    inference = ai_client.generate_with_fallback(ranked_models, prompt)
                    raw = inference.response
                    provider_label = provider_name

                    if inference.fallback_used:
                        self._persist_system_event(
                            "inference_fallback",
                            f"Fallback used for {symbol}",
                            started_at,
                            {
                                "symbol": symbol,
                                "model_used": inference.model_used,
                                "profile_used": inference.profile_used,
                                "latency_seconds": round(inference.latency_seconds, 3),
                            },
                        )

                    decision = self._parse_decision(raw)

                decision["daytrade_score"] = float(daytrade_profile.get("score", 0.0) or 0.0)
                decision["daytrade_profile"] = daytrade_profile
                decision["used_council"] = bool(use_council_for_this)
                rationale_text = str(decision.get("rationale", "") or "").strip()
                cycle_report["decisions"].append(
                    {
                        "symbol": symbol,
                        "kind": kind,
                        "selected_by": str(target.get("selection_reason", "unknown")),
                        "trend_score": float(trend_score or 0.0),
                        "avg_daily_move_pct": float(target.get("avg_daily_move_pct", 0.0) or 0.0),
                        "last_day_move_pct": float(target.get("last_day_move_pct", 0.0) or 0.0),
                        "volume_vs_avg": float(target.get("volume_vs_avg", 0.0) or 0.0),
                        "daytrade_score": float(daytrade_profile.get("score", 0.0) or 0.0),
                        "daytrade_eligible": bool(daytrade_profile.get("eligible", False)),
                        "used_council": bool(use_council_for_this),
                        "provider": provider_label,
                        "action": str(decision.get("action", "")).upper(),
                        "confidence": float(decision.get("confidence", 0.0) or 0.0),
                        "rationale": rationale_text[:500],
                        "risk_flags": list(decision.get("risk_flags", [])),
                    }
                )
                self._persist_ai_decision(symbol, provider_label, decision, raw, started_at)
                self._propose_order_if_eligible(symbol, kind, decision, quote, account, started_at, provider_label)
                self._execute_paper_for_symbol(symbol, quote, started_at)

            self._refresh_daily_review(started_at)
            self._refresh_execution_quality_alerts(started_at)
            self._refresh_hourly_strategy_review(started_at)
            self._refresh_hourly_special_circumstances(started_at)

            runtime_state.consecutive_failures = 0
            buy_count = sum(1 for d in cycle_report["decisions"] if str(d.get("action", "")).upper() == "BUY")
            hold_count = sum(1 for d in cycle_report["decisions"] if str(d.get("action", "")).upper() == "HOLD")
            avoid_count = sum(1 for d in cycle_report["decisions"] if str(d.get("action", "")).upper() == "AVOID")
            cycle_report["summary"] = {
                "targets_total": len(trade_targets),
                "decisions_total": len(cycle_report["decisions"]),
                "skipped_total": len(cycle_report.get("skipped", [])),
                "buy": buy_count,
                "hold": hold_count,
                "avoid": avoid_count,
            }
            cycle_report["status"] = "ok"
            logger.info("Daily cycle completed for {} targets ({} bullish picks)", len(trade_targets), len(bullish_picks))

        except Exception as exc:
            runtime_state.consecutive_failures += 1
            logger.exception("Daily cycle failed: {}", exc)
            self._persist_system_event("pipeline_error", str(exc), started_at)
            cycle_report["status"] = "failed"
            cycle_report["error"] = str(exc)
            if runtime_state.consecutive_failures >= 5:
                runtime_state.kill_switch = True
                self._persist_system_event("kill_switch", "Activated due to repeated failures", started_at)
        finally:
            cycle_report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._write_cycle_report(cycle_report)
            runtime_state.mark_finish()

    def _resolve_ai_backend(self) -> tuple[str, object]:
        provider = settings.ai_provider

        if provider == "ollama":
            return "ollama", self.ollama

        if provider == "openai":
            if not self.openai.is_configured():
                raise RuntimeError("AI_PROVIDER=openai but OPENAI_API_KEY is not configured")
            return "openai", self.openai

        if settings.prefer_openai_for_reasoning and self.openai.is_configured():
            return "openai", self.openai

        return "ollama", self.ollama

    def _council_decide(
        self,
        symbol: str,
        kind: str,
        candles: list[dict],
        mark_price: float,
        created_at: str,
        daytrade_profile: dict | None = None,
    ) -> tuple[dict, dict]:
        """Run 3-agent council on a live trade opportunity.

        Returns (decision_dict, raw_response_dict) in the same format
        as the single-agent path so downstream code stays unchanged.
        """
        from .historical_data import Candle

        # Convert Robinhood candle dicts to Candle objects for compute_signals
        candle_objs: list[Candle] = []
        for c in candles:
            try:
                candle_objs.append(Candle(
                    timestamp=str(c.get("begins_at", "")),
                    open=float(c.get("open_price", c.get("open", 0))),
                    high=float(c.get("high_price", c.get("high", 0))),
                    low=float(c.get("low_price", c.get("low", 0))),
                    close=float(c.get("close_price", c.get("close", 0))),
                    volume=float(c.get("volume", 0)),
                ))
            except (ValueError, TypeError):
                continue

        # Compute technical signals
        signals = None
        if len(candle_objs) >= 2:
            signals = compute_signals(candle_objs[:-1], candle_objs[-1])

        # Maybe-fetch sentiment (cached per symbol)
        sentiment = self._sentiment_cache.get(symbol)
        if sentiment is None and symbol not in self._sentiment_cache:
            try:
                sentiment = self._sentiment_analyzer.get_sentiment(symbol, kind)
            except Exception:
                sentiment = None
            self._sentiment_cache[symbol] = sentiment

        # Last 6 candles as compact OHLCV for prompts
        candle_data = [
            {
                "o": round(c.open, 6), "h": round(c.high, 6),
                "l": round(c.low, 6), "c": round(c.close, 6),
                "v": round(c.volume, 2),
            }
            for c in candle_objs[-6:]
        ]

        # Portfolio state
        account = self.robinhood.get_account_snapshot()
        equity = float(account.get("equity", 500.0))
        if settings.bot_mode == "paper" and settings.paper_working_capital_usd > 0:
            try:
                paper_state = self._get_paper_capital_state(created_at)
                equity = float(paper_state.get("equity", equity))
            except Exception:
                pass
        open_positions = self._count_open_orders()
        exception_context = self._build_council_exception_context(symbol, kind, daytrade_profile or {})

        council_decision = self._council.decide(
            symbol=symbol,
            kind=kind,
            price=mark_price,
            signals=signals,
            sentiment=sentiment,
            market_mood=self._market_mood,
            candle_data=candle_data,
            equity=equity,
            goal=1000.0,
            open_positions=open_positions,
            max_positions=self.risk_policy.max_concurrent_positions,
            daytrade_profile=daytrade_profile or {},
            exception_context=exception_context,
        )

        # Convert CouncilDecision to the standard decision dict
        decision = {
            "action": council_decision.final_action,
            "confidence": council_decision.final_confidence,
            "trend_score": max(v.trend_score for v in council_decision.votes) if council_decision.votes else 0,
            "rationale": council_decision.rationale_summary,
            "risk_flags": [],
            "council_votes": f"BUY:{council_decision.buy_votes} HOLD:{council_decision.hold_votes} AVOID:{council_decision.avoid_votes}",
            "council_unanimous": council_decision.unanimous,
            "daytrade_score": float((daytrade_profile or {}).get("score", 0.0) or 0.0),
        }
        for v in council_decision.votes:
            decision["risk_flags"].extend(v.risk_flags)

        raw = {
            "council": True,
            "votes": [
                {
                    "agent": v.agent_name,
                    "action": v.action,
                    "confidence": v.confidence,
                    "rationale": v.rationale,
                    "risk_flags": v.risk_flags,
                }
                for v in council_decision.votes
            ],
            "latency": council_decision.total_latency,
        }

        logger.info(
            "Council {} | {} | BUY:{} HOLD:{} AVOID:{} | conf={:.0%} | {:.1f}s",
            symbol, council_decision.final_action,
            council_decision.buy_votes, council_decision.hold_votes,
            council_decision.avoid_votes, council_decision.final_confidence,
            council_decision.total_latency,
        )

        return decision, raw

    def _build_council_exception_context(self, symbol: str, kind: str, daytrade_profile: dict) -> dict:
        profile = daytrade_profile if isinstance(daytrade_profile, dict) else {}
        special_meta = profile.get("special_circumstances", {}) if isinstance(profile.get("special_circumstances", {}), dict) else {}
        applied = special_meta.get("applied", []) if isinstance(special_meta, dict) else []

        applied_rule_ids: list[str] = []
        source_roles: list[str] = []
        if isinstance(applied, list):
            for item in applied:
                if not isinstance(item, dict):
                    continue
                rule_id = str(item.get("rule_id", "") or "").strip()
                if rule_id:
                    applied_rule_ids.append(rule_id)
                role = str(item.get("source_role", "") or "").strip().lower()
                if role and role not in source_roles:
                    source_roles.append(role)

        context: dict[str, object] = {
            "has_applied_exception": bool(applied_rule_ids),
            "applied_rule_ids": applied_rule_ids,
            "source_roles": source_roles,
            "override_daytrade_min_score": bool(special_meta.get("override_daytrade_min_score", False)),
            "symbol": str(symbol or "").upper().replace("-USD", ""),
            "kind": str(kind or "").lower(),
            "summary": "No adaptive exception applied for this setup.",
            "pair_performance": [],
        }
        if not applied_rule_ids:
            return context

        payload = self._load_special_circumstances()
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        pair_stats = payload.get("pair_stats", {}) if isinstance(payload, dict) else {}
        if not isinstance(rules, list):
            rules = []
        if not isinstance(pair_stats, dict):
            pair_stats = {}

        rules_by_id = {
            str(rule.get("id", "") or ""): rule
            for rule in rules
            if isinstance(rule, dict) and str(rule.get("id", "") or "").strip()
        }

        pairs: list[dict[str, object]] = []
        seen_pair_ids: set[str] = set()
        for rule_id in applied_rule_ids:
            rule = rules_by_id.get(rule_id, {})
            pair_id = str(rule.get("pair_id", "") or "").strip()
            if not pair_id or pair_id in seen_pair_ids:
                continue
            seen_pair_ids.add(pair_id)
            stats = pair_stats.get(pair_id, {}) if isinstance(pair_stats.get(pair_id, {}), dict) else {}
            pairs.append(
                {
                    "pair_id": pair_id,
                    "trades": int(stats.get("trades", 0) or 0),
                    "wins": int(stats.get("wins", 0) or 0),
                    "losses": int(stats.get("losses", 0) or 0),
                    "net_pnl": round(float(stats.get("net_pnl", 0.0) or 0.0), 4),
                    "disabled": bool(stats.get("disabled", False)),
                }
            )

        pairs.sort(key=lambda x: (-int(x.get("trades", 0)), -float(x.get("net_pnl", 0.0))))
        context["pair_performance"] = pairs[:3]

        if pairs:
            first = pairs[0]
            context["summary"] = (
                f"Adaptive exception applied from {len(applied_rule_ids)} rule(s); "
                f"top pair {first.get('pair_id', '')} has {first.get('trades', 0)} trades and net PnL {first.get('net_pnl', 0.0)}."
            )
        else:
            context["summary"] = f"Adaptive exception applied from {len(applied_rule_ids)} rule(s); no pair performance stats yet."
        return context

    def _fast_indicator_decide(
        self,
        symbol: str,
        kind: str,
        candles: list[dict],
        mark_price: float,
        trend_score: int,
    ) -> tuple[dict, dict]:
        """Make a fast buy/avoid decision using only technical indicators.

        No GPU / AI calls — purely mathematical signal analysis.
        Used for lower-priority bullish picks to conserve GPU time for
        the council assets.  Returns the same (decision, raw) tuple.
        """
        from .historical_data import Candle

        candle_objs: list[Candle] = []
        for c in candles:
            try:
                candle_objs.append(Candle(
                    timestamp=str(c.get("begins_at", "")),
                    open=float(c.get("open_price", c.get("open", 0))),
                    high=float(c.get("high_price", c.get("high", 0))),
                    low=float(c.get("low_price", c.get("low", 0))),
                    close=float(c.get("close_price", c.get("close", 0))),
                    volume=float(c.get("volume", 0)),
                ))
            except (ValueError, TypeError):
                continue

        if len(candle_objs) < 5:
            decision = {"action": "AVOID", "confidence": 0.0, "rationale": "Insufficient candle data"}
            return decision, {"fast_mode": True}

        signals = compute_signals(candle_objs[:-1], candle_objs[-1])

        # Tally bullish vs bearish indicator votes
        bullish_signals = 0
        reasons = []

        if signals.rsi_14 and 30 <= signals.rsi_14 <= 55:
            bullish_signals += 1
            reasons.append(f"RSI={signals.rsi_14:.0f} buy zone")
        if signals.sma_fast and signals.sma_slow and signals.sma_fast > signals.sma_slow:
            bullish_signals += 1
            reasons.append("SMA bullish cross")
        if signals.momentum_5 and signals.momentum_5 > 0:
            bullish_signals += 1
            reasons.append(f"momentum +{signals.momentum_5:.1f}%")
        if signals.bollinger_position is not None and signals.bollinger_position < 0.3:
            bullish_signals += 1
            reasons.append("near Bollinger bottom")
        if signals.volume_ratio and signals.volume_ratio > 1.2:
            bullish_signals += 1
            reasons.append(f"volume {signals.volume_ratio:.1f}x avg")
        if trend_score >= 30:
            bullish_signals += 1
            reasons.append(f"trend score {trend_score}")

        # Need at least 3 bullish signals to buy
        if bullish_signals >= 3:
            confidence = min(0.50 + bullish_signals * 0.08, 0.85)
            action = "BUY"
        else:
            confidence = 0.30
            action = "HOLD"

        decision = {
            "action": action,
            "confidence": confidence,
            "trend_score": trend_score,
            "rationale": f"Fast indicators ({bullish_signals}/6 bullish): {'; '.join(reasons[:3])}",
            "risk_flags": [],
        }
        raw = {"fast_mode": True, "bullish_signals": bullish_signals, "reasons": reasons}

        logger.info(
            "Fast {} | {} | {}/{} signals | conf={:.0%} | {}",
            symbol, action, bullish_signals, 6, confidence,
            "; ".join(reasons[:3]),
        )
        return decision, raw

    def _build_prompt(self, symbol: str, news_items: list[dict], created_at: str,
                      kind: str = "crypto", is_bullish: bool = False, trend_score: int = 0,
                      daytrade_profile: dict | None = None) -> str:
        learning_context = self._get_learning_context(symbol, created_at)
        memory_context = self._operator_memory_context
        trend_context = ""
        if is_bullish:
            trend_context = (
                f" TREND SCAN says this asset is BULLISH (trend_score={trend_score}). "
                "Weight this heavily in your analysis — the scanner detected positive "
                "momentum / breakout / golden cross patterns. "
            )
        daytrade_context = (
            " Focus on day-trading suitability: this strategy only wants symbols with real movement. "
            f"Daytrade profile: {json.dumps(daytrade_profile or {})}. "
            "If movement/range is weak, choose HOLD or AVOID."
        )
        memory_hint = (
            f" Operator memory and lessons learned: {memory_context}. "
            "Avoid repeating known losing patterns from this memory."
            if memory_context
            else ""
        )
        return (
            f"You are a {'crypto' if kind == 'crypto' else 'stock'} trend analyst. Return strict JSON with keys: "
            "trend_score, confidence, action, rationale, risk_flags. "
            f"Symbol: {symbol} ({kind}). News context: {json.dumps(news_items)}. "
            f"Recent performance context: {json.dumps(learning_context)}.{trend_context}{daytrade_context}{memory_hint} "
            "Action must be one of: BUY, HOLD, AVOID."
        )

    def _compute_daytrade_profile(self, candles: list[dict], kind: str) -> dict:
        from .historical_data import Candle

        candle_objs: list[Candle] = []
        for c in candles[-24:]:
            try:
                candle_objs.append(Candle(
                    timestamp=str(c.get("begins_at", "")),
                    open=float(c.get("open_price", c.get("open", 0)) or 0),
                    high=float(c.get("high_price", c.get("high", 0)) or 0),
                    low=float(c.get("low_price", c.get("low", 0)) or 0),
                    close=float(c.get("close_price", c.get("close", 0)) or 0),
                    volume=float(c.get("volume", 0) or 0),
                ))
            except Exception:
                continue

        min_required_candles = 3
        if len(candle_objs) < min_required_candles:
            return {
                "eligible": False,
                "score": 0.0,
                "avg_bar_move_pct": 0.0,
                "recent_bar_move_pct": 0.0,
                "avg_bar_range_pct": 0.0,
                "volume_ratio": 0.0,
                "reason": f"insufficient_candles:{len(candle_objs)}/{min_required_candles}",
            }

        closes = [c.close for c in candle_objs]
        moves: list[float] = []
        ranges: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev > 0:
                moves.append(abs((curr - prev) / prev) * 100.0)

        for c in candle_objs:
            if c.close > 0:
                ranges.append(abs((c.high - c.low) / c.close) * 100.0)

        avg_move = (sum(moves) / len(moves)) if moves else 0.0
        recent_move = moves[-1] if moves else 0.0
        avg_range = (sum(ranges) / len(ranges)) if ranges else 0.0

        vols = [max(0.0, float(c.volume)) for c in candle_objs]
        volume_ratio = 1.0
        if len(vols) >= min_required_candles:
            baseline = (sum(vols[:-1]) / max(1, len(vols) - 1))
            if baseline > 0:
                volume_ratio = vols[-1] / baseline

        move_threshold = (
            settings.daytrade_min_avg_bar_move_crypto_pct
            if kind == "crypto"
            else settings.daytrade_min_avg_bar_move_stock_pct
        )
        range_threshold = (
            settings.daytrade_min_avg_bar_range_crypto_pct
            if kind == "crypto"
            else settings.daytrade_min_avg_bar_range_stock_pct
        )
        recent_threshold = settings.daytrade_min_recent_bar_move_pct
        vol_threshold = settings.daytrade_min_bar_volume_ratio

        move_norm = min(2.0, avg_move / max(move_threshold, 1e-6))
        range_norm = min(2.0, avg_range / max(range_threshold, 1e-6))
        recent_norm = min(2.0, recent_move / max(recent_threshold, 1e-6))
        vol_norm = min(2.0, volume_ratio / max(vol_threshold, 1e-6))
        score = min(100.0, (move_norm * 30.0) + (range_norm * 30.0) + (recent_norm * 25.0) + (vol_norm * 15.0))

        strict_eligible = (
            avg_move >= move_threshold
            and avg_range >= range_threshold
            and recent_move >= recent_threshold
            and volume_ratio >= vol_threshold
            and score >= settings.daytrade_min_score
        )

        recent_window = candle_objs[-6:] if len(candle_objs) >= 6 else candle_objs
        recent_closes = [float(c.close) for c in recent_window if float(c.close) > 0]
        recent_low = min(recent_closes) if recent_closes else 0.0
        recent_high = max(recent_closes) if recent_closes else 0.0
        latest_close = recent_closes[-1] if recent_closes else 0.0

        net_climb_pct = 0.0
        up_step_ratio = 0.0
        acceleration_pct = 0.0
        distance_from_low_pct = 0.0
        pullback_from_peak_pct = 0.0

        if len(recent_closes) >= 3:
            first_close = recent_closes[0]
            if first_close > 0:
                net_climb_pct = ((latest_close - first_close) / first_close) * 100.0

            total_steps = max(1, len(recent_closes) - 1)
            up_steps = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] > recent_closes[i - 1])
            up_step_ratio = up_steps / total_steps

            if len(recent_closes) >= 4:
                mid = len(recent_closes) // 2
                first_half = recent_closes[: max(2, mid)]
                second_half = recent_closes[-max(2, len(recent_closes) - mid):]

                slope_first = 0.0
                slope_second = 0.0
                if len(first_half) >= 2 and first_half[0] > 0:
                    slope_first = ((first_half[-1] - first_half[0]) / first_half[0]) * 100.0
                if len(second_half) >= 2 and second_half[0] > 0:
                    slope_second = ((second_half[-1] - second_half[0]) / second_half[0]) * 100.0
                acceleration_pct = slope_second - slope_first
            else:
                move_a = 0.0
                move_b = 0.0
                if recent_closes[0] > 0:
                    move_a = ((recent_closes[1] - recent_closes[0]) / recent_closes[0]) * 100.0
                if recent_closes[1] > 0:
                    move_b = ((recent_closes[2] - recent_closes[1]) / recent_closes[1]) * 100.0
                acceleration_pct = move_b - move_a

        if recent_low > 0 and latest_close > 0:
            distance_from_low_pct = ((latest_close - recent_low) / recent_low) * 100.0
        if recent_high > 0 and latest_close > 0:
            pullback_from_peak_pct = ((recent_high - latest_close) / recent_high) * 100.0

        early_climb_cfg = {
            "enabled": bool(settings.daytrade_early_climb_enabled),
            "min_net_climb_pct": float(settings.daytrade_early_climb_min_net_climb_pct),
            "min_up_step_ratio": float(settings.daytrade_early_climb_min_up_step_ratio),
            "min_acceleration_pct": float(settings.daytrade_early_climb_min_acceleration_pct),
            "max_distance_from_low_pct": float(settings.daytrade_early_climb_max_distance_from_low_pct),
            "max_pullback_pct": float(settings.daytrade_early_climb_max_pullback_pct),
            "min_volume_ratio": float(settings.daytrade_early_climb_min_volume_ratio),
            "min_score": float(settings.daytrade_early_climb_min_score),
        }

        early_climb_eligible = (
            early_climb_cfg["enabled"]
            and net_climb_pct >= early_climb_cfg["min_net_climb_pct"]
            and up_step_ratio >= early_climb_cfg["min_up_step_ratio"]
            and acceleration_pct >= early_climb_cfg["min_acceleration_pct"]
            and distance_from_low_pct <= early_climb_cfg["max_distance_from_low_pct"]
            and pullback_from_peak_pct <= early_climb_cfg["max_pullback_pct"]
            and volume_ratio >= early_climb_cfg["min_volume_ratio"]
            and score >= early_climb_cfg["min_score"]
        )

        eligible = strict_eligible or early_climb_eligible
        eligible_path = "strict" if strict_eligible else ("early_climb" if early_climb_eligible else "none")

        return {
            "eligible": bool(eligible),
            "eligible_path": eligible_path,
            "score": round(score, 2),
            "avg_bar_move_pct": round(avg_move, 3),
            "recent_bar_move_pct": round(recent_move, 3),
            "avg_bar_range_pct": round(avg_range, 3),
            "volume_ratio": round(volume_ratio, 3),
            "early_climb": {
                "enabled": bool(early_climb_cfg["enabled"]),
                "eligible": bool(early_climb_eligible),
                "net_climb_pct": round(net_climb_pct, 3),
                "up_step_ratio": round(up_step_ratio, 3),
                "acceleration_pct": round(acceleration_pct, 3),
                "distance_from_low_pct": round(distance_from_low_pct, 3),
                "pullback_from_peak_pct": round(pullback_from_peak_pct, 3),
            },
            "thresholds": {
                "min_avg_bar_move_pct": float(move_threshold),
                "min_avg_bar_range_pct": float(range_threshold),
                "min_recent_bar_move_pct": float(recent_threshold),
                "min_volume_ratio": float(vol_threshold),
                "min_score": float(settings.daytrade_min_score),
                "early_climb": early_climb_cfg,
            },
        }

    def _get_learning_context(self, symbol: str, created_at: str) -> dict:
        lookback_start = (self._parse_iso_datetime(created_at) - timedelta(days=settings.review_lookback_days)).isoformat()

        with get_connection() as conn:
            overall = conn.execute(
                """
                SELECT COUNT(1) AS trades,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                       COALESCE(SUM(pnl), 0.0) AS pnl
                FROM trade_outcomes
                WHERE closed_at >= ?
                """,
                (lookback_start,),
            ).fetchone()

            symbol_stats = conn.execute(
                """
                SELECT COUNT(1) AS trades,
                       COALESCE(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                       COALESCE(SUM(t.pnl), 0.0) AS pnl
                FROM trade_outcomes t
                JOIN orders o ON o.id = t.order_id
                WHERE t.closed_at >= ? AND o.symbol = ?
                """,
                (lookback_start, symbol),
            ).fetchone()

            latest_review = conn.execute(
                """
                SELECT review_date, summary, lessons
                FROM daily_reviews
                ORDER BY review_date DESC
                LIMIT 1
                """
            ).fetchone()

        def pack(row: dict | None) -> dict:
            trades = int(row["trades"] if row else 0)
            wins = int(row["wins"] if row else 0)
            pnl = float(row["pnl"] if row else 0.0)
            win_rate = (wins / trades) if trades > 0 else 0.0
            return {
                "trades": trades,
                "wins": wins,
                "win_rate": round(win_rate, 4),
                "realized_pnl": round(pnl, 6),
            }

        review_payload = {
            "review_date": str(latest_review["review_date"]) if latest_review else "",
            "summary": str(latest_review["summary"]) if latest_review else "",
            "lessons": str(latest_review["lessons"]) if latest_review else "",
        }

        return {
            "lookback_days": settings.review_lookback_days,
            "overall": pack(overall),
            "symbol": symbol,
            "symbol_stats": pack(symbol_stats),
            "latest_review": review_payload,
        }

    def _parse_decision(self, response: dict) -> dict:
        content = response.get("response", "{}")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {
                "trend_score": 0,
                "confidence": 0,
                "action": "AVOID",
                "rationale": "Model did not return valid JSON.",
                "risk_flags": ["invalid_json"],
            }

        payload.setdefault("trend_score", 0)
        payload.setdefault("confidence", 0)
        payload.setdefault("action", "AVOID")
        payload.setdefault("rationale", "No rationale provided")
        payload.setdefault("risk_flags", [])
        return payload

    def _persist_ai_decision(
        self,
        symbol: str,
        model_name: str,
        decision: dict,
        raw_response: dict,
        created_at: str,
    ) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ai_decisions (symbol, model_name, score, confidence, rationale, decision_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    model_name,
                    float(decision.get("trend_score", 0)),
                    float(decision.get("confidence", 0)),
                    str(decision.get("rationale", "")),
                    json.dumps({"decision": decision, "raw": raw_response}),
                    created_at,
                ),
            )
            conn.commit()

    def _propose_order_if_eligible(
        self,
        symbol: str,
        kind: str,
        decision: dict,
        quote: dict,
        account: dict,
        created_at: str,
        provider_name: str,
    ) -> None:
        action = str(decision.get("action", "AVOID")).upper()
        confidence = float(decision.get("confidence", 0.0))
        mark_price = float(quote.get("mark_price", 0.0))

        goal_progress = self._get_profit_goal_progress(created_at)
        goal_hit = bool(goal_progress.get("daily_goal_met", False) or goal_progress.get("weekly_goal_met", False))
        if settings.pause_new_orders_on_goal_hit and goal_hit:
            self._persist_system_event(
                "order_blocked_goal",
                f"Order blocked by earnings goal policy for {symbol}",
                created_at,
                {
                    "symbol": symbol,
                    "daily_pnl": goal_progress.get("daily_pnl", 0.0),
                    "weekly_pnl": goal_progress.get("weekly_pnl", 0.0),
                    "daily_goal": goal_progress.get("daily_goal", 0.0),
                    "weekly_goal": goal_progress.get("weekly_goal", 0.0),
                },
            )
            return

        quality_guard = self._evaluate_trade_quality_guard(created_at)
        if bool(quality_guard.get("block_new_orders", False)):
            self._persist_system_event(
                "order_blocked_quality_guard",
                f"Order blocked by quality guard for {symbol}",
                created_at,
                {
                    "symbol": symbol,
                    **quality_guard,
                },
            )
            return

        if action != "BUY":
            return

        if kind == "stock":
            stock_market = self._us_stock_market_status()
            if not bool(stock_market.get("is_open", False)):
                self._persist_system_event(
                    "order_blocked_market_closed",
                    f"Order blocked: stock market closed for {symbol}",
                    created_at,
                    {
                        "symbol": symbol,
                        "kind": kind,
                        "market": stock_market,
                    },
                )
                return

        if settings.daytrade_only_mode:
            daytrade_score = float(decision.get("daytrade_score", 0.0) or 0.0)
            daytrade_profile = decision.get("daytrade_profile", {}) if isinstance(decision, dict) else {}
            special_meta = daytrade_profile.get("special_circumstances", {}) if isinstance(daytrade_profile, dict) else {}
            allow_daytrade_min_score_override = bool(special_meta.get("override_daytrade_min_score", False))
            if daytrade_score < float(settings.daytrade_min_score) and not allow_daytrade_min_score_override:
                self._persist_system_event(
                    "order_blocked_daytrade_filter",
                    f"Order blocked by day-trade suitability filter for {symbol}",
                    created_at,
                    {
                        "symbol": symbol,
                        "daytrade_score": daytrade_score,
                        "daytrade_min_score": float(settings.daytrade_min_score),
                        "daytrade_profile": decision.get("daytrade_profile", {}),
                    },
                )
                return
            if daytrade_score < float(settings.daytrade_min_score) and allow_daytrade_min_score_override:
                self._persist_system_event(
                    "order_overridden_daytrade_min_score",
                    f"Order allowed via special circumstance override for {symbol}",
                    created_at,
                    {
                        "symbol": symbol,
                        "daytrade_score": daytrade_score,
                        "daytrade_min_score": float(settings.daytrade_min_score),
                        "special_circumstances": special_meta,
                    },
                )
        if confidence < self.risk_policy.min_confidence_to_trade:
            return
        if mark_price <= 0:
            return

        if settings.use_council and settings.council_require_unanimous_buy:
            council_unanimous = bool(decision.get("council_unanimous", False))
            if not council_unanimous:
                self._persist_system_event(
                    "order_blocked_council_unanimous",
                    f"Order blocked: unanimous council BUY required for {symbol}",
                    created_at,
                    {
                        "symbol": symbol,
                        "provider": provider_name,
                        "council_unanimous": council_unanimous,
                        "council_votes": str(decision.get("council_votes", "")),
                    },
                )
                return

        open_counts = self._count_open_orders_by_kind()
        open_positions = int(open_counts.get("total", 0))
        kind_open_positions = int(open_counts.get(kind, 0))

        MAX_POSITIONS_PER_KIND = 10
        if kind_open_positions >= MAX_POSITIONS_PER_KIND:
            self._persist_system_event(
                "order_blocked_asset_cap",
                f"Order blocked by {kind} position cap for {symbol}",
                created_at,
                {
                    "symbol": symbol,
                    "kind": kind,
                    "open_positions_kind": kind_open_positions,
                    "max_positions_kind": MAX_POSITIONS_PER_KIND,
                    "open_positions_total": open_positions,
                },
            )
            return

        if open_positions >= self.risk_policy.max_concurrent_positions:
            self._persist_system_event(
                "order_blocked",
                f"Order blocked by max positions for {symbol}",
                created_at,
                {"symbol": symbol, "open_positions": open_positions},
            )
            return

        pacing = self._compute_goal_pacing_multiplier(created_at)
        risk_multiplier = float(pacing.get("risk_multiplier", 1.0))

        buying_power = float(account.get("buying_power", 0.0))
        equity = float(account.get("equity", 0.0))
        paper_cap_meta: dict | None = None
        if settings.bot_mode == "paper" and settings.paper_working_capital_usd > 0:
            paper_cap_meta = self._get_paper_capital_state(created_at)
            buying_power = float(paper_cap_meta.get("buying_power", 0.0))
            equity = float(paper_cap_meta.get("equity", 0.0))

        # ── SLOT-BASED SIZING ──
        # Divide buying power evenly across remaining open slots so capital
        # is fully deployed instead of decaying with each order.
        remaining_slots = max(1, self.risk_policy.max_concurrent_positions - open_positions)
        slot_budget = buying_power / remaining_slots

        # Apply risk multiplier for goal pacing and per-position cap
        risk_budget = slot_budget * risk_multiplier
        position_cap = max(equity * self.risk_policy.max_position_pct, 0.0)
        notional = min(max(risk_budget, 0.0), position_cap if position_cap > 0 else max(risk_budget, 0.0))

        # Never exceed actual buying power
        notional = min(notional, buying_power)

        # Confidence-weighted sizing: high-confidence trades get full size,
        # lower-confidence trades get scaled down
        conf_scale = min(1.0, confidence / 0.80)  # 80%+ confidence = full size
        notional = notional * max(0.5, conf_scale)  # at least 50% of slot

        if notional <= 0:
            if settings.bot_mode == "paper" and settings.paper_working_capital_usd > 0:
                self._persist_system_event(
                    "order_blocked_paper_cap",
                    f"Order blocked by paper capital cap for {symbol}",
                    created_at,
                    {
                        "symbol": symbol,
                        "paper_capital": paper_cap_meta or self._get_paper_capital_state(created_at),
                    },
                )
            return

        quantity = round(notional / mark_price, 8)
        if quantity <= 0:
            return

        if settings.bot_mode == "live" and not self.risk_policy.allow_live_orders:
            status = "blocked_policy"
            rationale = "Live mode active but risk policy blocks live orders"
        elif settings.bot_mode == "live" and self.risk_policy.allow_live_orders:
            notional = quantity * mark_price
            allowed, reason, live_meta = self._check_live_order_caps(notional, created_at)
            if not allowed:
                self._persist_system_event(
                    "order_blocked_live_cap",
                    f"Live order blocked for {symbol}: {reason}",
                    created_at,
                    {"symbol": symbol, **live_meta},
                )
                return
            status = "live_ready"
            rationale = str(decision.get("rationale", ""))
        else:
            status = "paper_proposed"
            rationale = str(decision.get("rationale", ""))

        inserted_order_id = 0
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO orders (symbol, mode, side, quantity, price, status, rationale, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    settings.bot_mode,
                    "buy",
                    quantity,
                    mark_price,
                    status,
                    rationale,
                    created_at,
                ),
            )
            inserted_order_id = int(cursor.lastrowid)
            conn.execute(
                """
                UPDATE orders
                SET decision_confidence = ?, decision_score = ?, ai_provider = ?
                WHERE id = ?
                """,
                (
                    confidence,
                    float(decision.get("trend_score", 0.0)),
                    provider_name,
                    inserted_order_id,
                ),
            )
            conn.commit()

        live_submit_meta: dict | None = None
        if status == "live_ready" and inserted_order_id > 0:
            live_submit_meta = self._submit_live_buy_order(
                order_id=inserted_order_id,
                symbol=symbol,
                kind=kind,
                quantity=quantity,
                created_at=created_at,
            )

        self._persist_system_event(
            "order_proposed",
            f"Order proposal created for {symbol}",
            created_at,
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": mark_price,
                "status": status,
                "confidence": confidence,
                "risk_multiplier": risk_multiplier,
                "goal_pacing": pacing,
                "paper_capital": paper_cap_meta,
                "live_submission": live_submit_meta,
            },
        )

    def _evaluate_trade_quality_guard(self, created_at: str) -> dict:
        lookback_start = (self._parse_iso_datetime(created_at) - timedelta(days=settings.review_lookback_days)).isoformat()
        with get_connection() as conn:
            stats = conn.execute(
                """
                SELECT COUNT(1) AS n,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
                FROM trade_outcomes
                WHERE closed_at >= ?
                """,
                (lookback_start,),
            ).fetchone()

        trades = int(stats["n"] if stats else 0)
        wins = int(stats["wins"] if stats else 0)
        win_rate = (wins / trades) if trades > 0 else 0.0
        min_trades = int(settings.analytics_min_trades_for_quality_alert)
        min_win = float(settings.analytics_min_win_rate)

        block = trades >= min_trades and win_rate < min_win
        return {
            "block_new_orders": bool(block),
            "lookback_days": int(settings.review_lookback_days),
            "trades": trades,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "min_trades": min_trades,
            "min_win_rate": min_win,
        }

    def _submit_live_buy_order(
        self,
        order_id: int,
        symbol: str,
        kind: str,
        quantity: float,
        created_at: str,
    ) -> dict:
        try:
            response = self.robinhood.place_buy_order(symbol=symbol, kind=kind, quantity=quantity)
            with get_connection() as conn:
                conn.execute(
                    "UPDATE orders SET status = 'live_submitted' WHERE id = ?",
                    (order_id,),
                )
                conn.commit()
            self._persist_system_event(
                "live_order_submitted",
                f"Live buy submitted for {symbol}",
                created_at,
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "kind": kind,
                    "quantity": quantity,
                    "broker_response": response,
                },
            )
            return {"status": "submitted", "broker": response}
        except Exception as exc:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE orders SET status = 'live_failed' WHERE id = ?",
                    (order_id,),
                )
                conn.commit()
            self._persist_system_event(
                "live_order_failed",
                f"Live buy failed for {symbol}: {exc}",
                created_at,
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "kind": kind,
                    "quantity": quantity,
                },
            )
            return {"status": "failed", "error": str(exc)}

    def _get_paper_capital_state(self, created_at: str) -> dict:
        runtime_caps = self._load_runtime_env_floats()
        configured_cap = max(0.0, float(runtime_caps.get("PAPER_WORKING_CAPITAL_USD", settings.paper_working_capital_usd)))
        wiggle_cash = max(0.0, float(runtime_caps.get("PAPER_EXTRA_PLAY_CASH_USD", getattr(settings, "paper_extra_play_cash_usd", 0.0) or 0.0)))
        cap = configured_cap + wiggle_cash
        if cap <= 0:
            return {
                "enabled": False,
                "configured_capital": 0.0,
                "extra_play_cash": 0.0,
                "effective_capital": 0.0,
                "equity": 0.0,
                "buying_power": 0.0,
                "realized_pnl": 0.0,
                "open_notional": 0.0,
            }

        with get_connection() as conn:
            pnl_row = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0.0) AS pnl
                FROM trade_outcomes
                WHERE closed_at <= ?
                """,
                (created_at,),
            ).fetchone()
            exposure_row = conn.execute(
                """
                SELECT COALESCE(SUM(quantity * price), 0.0) AS notional
                FROM orders
                WHERE mode = 'paper' AND status IN ('paper_proposed', 'paper_filled')
                """
            ).fetchone()

        realized_pnl = float(pnl_row["pnl"] if pnl_row else 0.0)
        open_notional = float(exposure_row["notional"] if exposure_row else 0.0)
        equity = max(0.0, cap + realized_pnl)
        buying_power = max(0.0, equity - open_notional)

        return {
            "enabled": True,
            "configured_capital": round(configured_cap, 6),
            "extra_play_cash": round(wiggle_cash, 6),
            "effective_capital": round(cap, 6),
            "equity": round(equity, 6),
            "buying_power": round(buying_power, 6),
            "realized_pnl": round(realized_pnl, 6),
            "open_notional": round(open_notional, 6),
        }

    def _execute_paper_for_symbol(self, symbol: str, quote: dict, created_at: str) -> None:
        if settings.bot_mode != "paper" or not settings.paper_trading_enabled:
            return

        mark_price = float(quote.get("mark_price", 0.0))
        if mark_price <= 0:
            return

        self._fill_pending_paper_orders(symbol, mark_price, created_at)
        if settings.paper_auto_close_enabled:
            self._close_mature_paper_orders(symbol, mark_price, created_at)

    def _fill_pending_paper_orders(self, symbol: str, mark_price: float, created_at: str) -> None:
        with get_connection() as conn:
            pending = conn.execute(
                """
                SELECT id
                FROM orders
                WHERE symbol = ? AND status = 'paper_proposed'
                ORDER BY id ASC
                """,
                (symbol,),
            ).fetchall()

            if not pending:
                return

            fill_price = mark_price * (1 + settings.paper_slippage_bps / 10000)
            for row in pending:
                conn.execute(
                    "UPDATE orders SET status = 'paper_filled', price = ? WHERE id = ?",
                    (fill_price, row["id"]),
                )
            conn.commit()

        self._persist_system_event(
            "paper_fill",
            f"Filled {len(pending)} paper order(s) for {symbol}",
            created_at,
            {
                "symbol": symbol,
                "count": len(pending),
                "fill_price": round(fill_price, 8),
            },
        )

    def _close_mature_paper_orders(self, symbol: str, mark_price: float, created_at: str) -> None:
        """Close paper positions that hit profit target, stop loss, or max hold time.

        Exit rules (checked in order):
          1. PROFIT TARGET: +8% gain -> close immediately (take profit)
          2. STOP LOSS:     -4% loss -> close immediately (cut losses)
          3. TRAILING STOP: price dropped from recent highs -> close
          4. MAX HOLD:      held >= paper_min_hold_minutes -> close at market

        For entries opened via early-climb path, tighter exits are used by default
        (configurable via PAPER_EARLY_CLIMB_* settings).
        """
        PROFIT_TARGET_PCT = 0.08   # +8% take profit
        STOP_LOSS_PCT = -0.04      # -4% stop loss
        TRAILING_STOP_PCT = 0.03   # 3% from peak

        kind = "crypto" if "-" in str(symbol or "") else "stock"
        special_payload = self._load_special_circumstances()
        if kind == "crypto":
            candles = self.robinhood.get_crypto_historicals(symbol)
        else:
            candles = self.robinhood.get_stock_historicals(symbol)

        recent_high = float(mark_price)
        for c in (candles or [])[-24:]:
            try:
                high = float(c.get("high_price", c.get("high", 0.0)) or 0.0)
            except Exception:
                high = 0.0
            if high > recent_high:
                recent_high = high

        pullback_from_recent_high_pct = 0.0
        if recent_high > 0:
            pullback_from_recent_high_pct = max(0.0, (recent_high - mark_price) / recent_high)

        now_dt = self._parse_iso_datetime(created_at)
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, quantity, price, created_at
                FROM orders
                WHERE symbol = ? AND status = 'paper_filled'
                ORDER BY id ASC
                """,
                (symbol,),
            ).fetchall()

            close_count = 0
            early_climb_close_count = 0
            total_realized_pnl = 0.0
            close_details: list[dict] = []
            for row in rows:
                entry_dt = self._parse_iso_datetime(str(row["created_at"]))
                hold_minutes = int((now_dt - entry_dt).total_seconds() // 60)

                quantity = float(row["quantity"] or 0.0)
                entry_price = float(row["price"] or 0.0)
                if quantity <= 0 or entry_price <= 0:
                    continue

                entry_path = ""
                buy_rule_ids: list[str] = []
                if bool(settings.paper_early_climb_exit_enabled):
                    decision_row = conn.execute(
                        """
                        SELECT decision_json
                        FROM ai_decisions
                        WHERE UPPER(symbol) = UPPER(?)
                          AND created_at <= ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (symbol, str(row["created_at"] or "")),
                    ).fetchone()
                    if decision_row:
                        try:
                            payload = json.loads(str(decision_row["decision_json"] or "{}"))
                        except Exception:
                            payload = {}
                        decision_obj = payload.get("decision", payload) if isinstance(payload, dict) else {}
                        profile = decision_obj.get("daytrade_profile", {}) if isinstance(decision_obj, dict) else {}
                        entry_path = str(profile.get("eligible_path", "")).lower()
                        special_meta = profile.get("special_circumstances", {}) if isinstance(profile, dict) else {}
                        applied = special_meta.get("applied", []) if isinstance(special_meta, dict) else []
                        if isinstance(applied, list):
                            for item in applied:
                                if isinstance(item, dict):
                                    rule_id = str(item.get("rule_id", "") or "").strip()
                                    if rule_id:
                                        buy_rule_ids.append(rule_id)
                                elif isinstance(item, str):
                                    rule_id = str(item).strip()
                                    if rule_id:
                                        buy_rule_ids.append(rule_id)

                is_early_climb_entry = bool(settings.paper_early_climb_exit_enabled) and entry_path == "early_climb"
                if is_early_climb_entry:
                    early_climb_close_count += 1
                profit_target_pct = float(settings.paper_early_climb_take_profit_pct) if is_early_climb_entry else PROFIT_TARGET_PCT
                stop_loss_pct = -abs(float(settings.paper_early_climb_stop_loss_pct)) if is_early_climb_entry else STOP_LOSS_PCT
                trailing_stop_pct = float(settings.paper_early_climb_trailing_stop_pct) if is_early_climb_entry else TRAILING_STOP_PCT
                max_hold_minutes = min(
                    int(settings.paper_min_hold_minutes),
                    int(settings.paper_early_climb_max_hold_minutes),
                ) if is_early_climb_entry else int(settings.paper_min_hold_minutes)

                sell_override = self._resolve_special_sell_override(
                    symbol=symbol,
                    kind=kind,
                    buy_rule_ids=buy_rule_ids,
                    payload=special_payload,
                )
                if sell_override:
                    profit_target_pct = float(sell_override.get("take_profit_pct", profit_target_pct) or profit_target_pct)
                    stop_loss_pct = -abs(float(sell_override.get("stop_loss_pct", abs(stop_loss_pct)) or abs(stop_loss_pct)))
                    trailing_stop_pct = float(sell_override.get("trailing_stop_pct", trailing_stop_pct) or trailing_stop_pct)
                    max_hold_minutes = int(sell_override.get("max_hold_minutes", max_hold_minutes) or max_hold_minutes)

                price_change_pct = (mark_price - entry_price) / entry_price
                exit_reason = None

                # 1. Profit target hit
                if price_change_pct >= profit_target_pct:
                    exit_reason = f"TP hit ({price_change_pct:+.1%})"
                # 2. Stop loss hit
                elif price_change_pct <= stop_loss_pct:
                    exit_reason = f"SL hit ({price_change_pct:+.1%})"
                # 3. Trailing stop on pullback from recent highs
                elif pullback_from_recent_high_pct >= trailing_stop_pct and price_change_pct > 0:
                    exit_reason = f"TS hit (-{pullback_from_recent_high_pct:.1%} from local high)"
                # 4. Max hold time reached
                elif hold_minutes >= max_hold_minutes:
                    exit_reason = f"max hold ({hold_minutes}m)"
                else:
                    continue  # not ready to close yet

                exit_price = mark_price * (1 - settings.paper_slippage_bps / 10000)
                gross_pnl = (exit_price - entry_price) * quantity
                traded_notional = (entry_price + exit_price) * quantity
                fees = traded_notional * (settings.paper_fee_bps / 10000)
                net_pnl = gross_pnl - fees
                max_drawdown = min(0.0, price_change_pct)

                conn.execute(
                    "UPDATE orders SET status = 'paper_closed' WHERE id = ?",
                    (row["id"],),
                )
                conn.execute(
                    """
                    INSERT INTO trade_outcomes (order_id, pnl, hold_minutes, max_drawdown, closed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row["id"], net_pnl, hold_minutes, max_drawdown, created_at),
                )

                close_count += 1
                total_realized_pnl += net_pnl
                close_details.append(
                    {
                        "order_id": int(row["id"]),
                        "buy_rule_ids": buy_rule_ids,
                        "sell_override": sell_override,
                        "entry_path": entry_path or "strict",
                        "exit_reason": exit_reason,
                        "pnl": round(net_pnl, 6),
                        "hold_minutes": hold_minutes,
                    }
                )
                logger.info(
                    "Paper EXIT {} | {} | entry_path={} entry=${:.4f} exit=${:.4f} pnl=${:+.2f} held={}m",
                    symbol, exit_reason, (entry_path or "strict"), entry_price, exit_price, net_pnl, hold_minutes,
                )

            conn.commit()

        if close_count > 0:
            self._persist_system_event(
                "paper_close",
                f"Closed {close_count} paper order(s) for {symbol}",
                created_at,
                {
                    "symbol": symbol,
                    "count": close_count,
                    "early_climb_count": early_climb_close_count,
                    "realized_pnl": round(total_realized_pnl, 6),
                    "details": close_details[:25],
                },
            )

    def _refresh_daily_review(self, created_at: str) -> None:
        as_of = self._parse_iso_datetime(created_at)
        review_date = as_of.date().isoformat()
        lookback_start = (as_of - timedelta(days=settings.review_lookback_days)).isoformat()

        with get_connection() as conn:
            daily_stats = conn.execute(
                """
                SELECT COUNT(1) AS n, COALESCE(SUM(pnl), 0.0) AS pnl,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
                FROM trade_outcomes
                WHERE closed_at >= ?
                """,
                (lookback_start,),
            ).fetchone()

            trades = int(daily_stats["n"] if daily_stats else 0)
            pnl = float(daily_stats["pnl"] if daily_stats else 0.0)
            wins = int(daily_stats["wins"] if daily_stats else 0)
            win_rate = (wins / trades) if trades > 0 else 0.0

            summary = (
                f"{settings.review_lookback_days}d paper summary: trades={trades}, "
                f"realized_pnl={pnl:.4f}, win_rate={win_rate:.2%}"
            )
            lessons = (
                "Increase confidence threshold if win rate degrades; "
                "reduce max symbols per cycle during high volatility."
            )

            conn.execute("DELETE FROM daily_reviews WHERE review_date = ?", (review_date,))
            conn.execute(
                """
                INSERT INTO daily_reviews (review_date, summary, lessons, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (review_date, summary, lessons, created_at),
            )
            conn.commit()

        self._maybe_emit_drawdown_alert(created_at)

    def _maybe_emit_drawdown_alert(self, created_at: str) -> None:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT pnl
                FROM trade_outcomes
                WHERE closed_at >= ?
                ORDER BY closed_at ASC
                """,
                ((self._parse_iso_datetime(created_at) - timedelta(days=settings.review_lookback_days)).isoformat(),),
            ).fetchall()

            cumulative = 0.0
            peak = 0.0
            max_drawdown = 0.0
            for row in rows:
                cumulative += float(row["pnl"] or 0.0)
                peak = max(peak, cumulative)
                dd = cumulative - peak
                max_drawdown = min(max_drawdown, dd)

            drawdown_pct = abs(max_drawdown) / max(settings.paper_starting_equity, 1.0)
            if drawdown_pct < settings.analytics_drawdown_alert_pct:
                return

            recent = conn.execute(
                """
                SELECT COUNT(1) AS n
                FROM system_events
                WHERE event_type = 'drawdown_alert' AND created_at >= ?
                """,
                ((self._parse_iso_datetime(created_at) - timedelta(hours=12)).isoformat(),),
            ).fetchone()
            if int(recent["n"] if recent else 0) > 0:
                return

        self._persist_system_event(
            "drawdown_alert",
            "Paper drawdown threshold breached",
            created_at,
            {
                "drawdown_pct": round(drawdown_pct, 4),
                "threshold_pct": settings.analytics_drawdown_alert_pct,
            },
        )

    def _refresh_execution_quality_alerts(self, created_at: str) -> None:
        lookback_start = (self._parse_iso_datetime(created_at) - timedelta(days=settings.review_lookback_days)).isoformat()
        with get_connection() as conn:
            stats = conn.execute(
                """
                SELECT COUNT(1) AS n,
                       COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
                FROM trade_outcomes
                WHERE closed_at >= ?
                """,
                (lookback_start,),
            ).fetchone()

            trades = int(stats["n"] if stats else 0)
            wins = int(stats["wins"] if stats else 0)
            win_rate = (wins / trades) if trades > 0 else 0.0

            if trades < settings.analytics_min_trades_for_quality_alert:
                return
            if win_rate >= settings.analytics_min_win_rate:
                return

            recent = conn.execute(
                """
                SELECT COUNT(1) AS n
                FROM system_events
                WHERE event_type = 'quality_alert' AND created_at >= ?
                """,
                ((self._parse_iso_datetime(created_at) - timedelta(hours=12)).isoformat(),),
            ).fetchone()
            if int(recent["n"] if recent else 0) > 0:
                return

        self._persist_system_event(
            "quality_alert",
            "Execution quality degraded below threshold",
            created_at,
            {
                "win_rate": round(win_rate, 4),
                "min_win_rate": settings.analytics_min_win_rate,
                "trades": trades,
            },
        )

    def _refresh_hourly_strategy_review(self, created_at: str) -> None:
        with get_connection() as conn:
            last = conn.execute(
                """
                SELECT created_at
                FROM system_events
                WHERE event_type = 'hourly_strategy_review'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if last:
                now_dt = self._parse_iso_datetime(created_at)
                last_dt = self._parse_iso_datetime(str(last["created_at"]))
                if (now_dt - last_dt).total_seconds() < 3600:
                    return

            since = (self._parse_iso_datetime(created_at) - timedelta(hours=1)).isoformat()
            rows = conn.execute(
                """
                SELECT decision_json, model_name
                FROM ai_decisions
                WHERE created_at >= ?
                ORDER BY id DESC
                """,
                (since,),
            ).fetchall()

        total_recent = 0
        buy_recent = 0
        council_recent = 0
        unanimous_buy_recent = 0
        conf_values: list[float] = []

        for row in rows:
            try:
                payload = json.loads(str(row["decision_json"] or "{}"))
                decision = payload.get("decision", payload)
                action = str(decision.get("action", "")).upper()
                conf = float(decision.get("confidence", 0.0) or 0.0)
                total_recent += 1
                conf_values.append(conf)
                if action == "BUY":
                    buy_recent += 1

                model_name = str(row["model_name"] or "")
                if "council" in model_name:
                    council_recent += 1
                    if action == "BUY" and bool(decision.get("council_unanimous", False)):
                        unanimous_buy_recent += 1
            except Exception:
                continue

        buy_rate = (buy_recent / total_recent) if total_recent else 0.0
        avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0
        unanimous_buy_rate = (unanimous_buy_recent / buy_recent) if buy_recent else 0.0

        progress = self._get_profit_goal_progress(created_at)
        daily_ratio = float(progress.get("daily_pnl", 0.0)) / max(float(progress.get("daily_expected_by_now", 0.0)), 1e-9) if float(progress.get("daily_expected_by_now", 0.0)) > 0 else 1.0
        weekly_ratio = float(progress.get("weekly_pnl", 0.0)) / max(float(progress.get("weekly_expected_by_now", 0.0)), 1e-9) if float(progress.get("weekly_expected_by_now", 0.0)) > 0 else 1.0

        health = generate_health_report(settings.db_path)
        health_score = int(health.get("score", 0) or 0)
        health_issues = [str(x) for x in (health.get("issues", []) or [])][:5]

        notes: list[str] = []
        if (daily_ratio < 0.85 or weekly_ratio < 0.85) and buy_rate < 0.25:
            notes.append("Conservative pace while behind goals; consider slightly higher risk pacing.")
        if buy_rate > 0.85 and health_score < 70:
            notes.append("Aggressiveness may be too high for current edge; tighten entries/confidence.")
        if avg_conf < 0.60:
            notes.append("Average confidence is low; decisions may not be informed enough.")
        if council_recent > 0 and unanimous_buy_rate < 0.5:
            notes.append("Many council BUY calls are non-unanimous; conviction may be mixed.")
        if not notes:
            notes.append("Decision pace appears balanced for current goals and decision quality.")

        aggressiveness_score = max(0, min(100, int(round(buy_rate * 100))))
        informed_score = max(0, min(100, int(round((avg_conf * 100 + health_score) / 2))))
        strategy_score = max(0, min(100, int(round(aggressiveness_score * 0.35 + informed_score * 0.65))))

        metadata = {
            "window": "1h",
            "decisions_total": total_recent,
            "buy_rate": round(buy_rate, 4),
            "avg_confidence": round(avg_conf, 4),
            "council_decisions": council_recent,
            "unanimous_buy_rate": round(unanimous_buy_rate, 4),
            "daily_progress_ratio": round(daily_ratio, 4),
            "weekly_progress_ratio": round(weekly_ratio, 4),
            "health_score": health_score,
            "aggressiveness_score": aggressiveness_score,
            "informed_score": informed_score,
            "strategy_score": strategy_score,
            "agree_with_council": bool(strategy_score >= 65 and health_score >= 65),
            "notes": notes,
            "health_issues": health_issues,
        }

        message = f"Hourly strategy review score={strategy_score}/100, buy_rate={buy_rate:.0%}, avg_conf={avg_conf:.0%}"
        self._persist_system_event("hourly_strategy_review", message, created_at, metadata)
        self._write_hourly_strategy_review(
            {
                "generated_at": created_at,
                "message": message,
                "metadata": metadata,
            }
        )

    def _get_profit_goal_progress(self, created_at: str) -> dict:
        as_of = self._parse_iso_datetime(created_at)
        day_start = datetime(as_of.year, as_of.month, as_of.day, tzinfo=as_of.tzinfo)
        week_start = (day_start - timedelta(days=day_start.weekday()))

        day_elapsed_fraction = max(0.01, min(1.0, (as_of - day_start).total_seconds() / 86400.0))
        week_elapsed_seconds = (as_of - week_start).total_seconds()
        week_elapsed_fraction = max(0.01, min(1.0, week_elapsed_seconds / (7 * 86400.0)))

        with get_connection() as conn:
            day_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) AS pnl FROM trade_outcomes WHERE closed_at >= ?",
                (day_start.isoformat(),),
            ).fetchone()
            week_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) AS pnl FROM trade_outcomes WHERE closed_at >= ?",
                (week_start.isoformat(),),
            ).fetchone()

        daily_pnl = float(day_row["pnl"] if day_row else 0.0)
        weekly_pnl = float(week_row["pnl"] if week_row else 0.0)
        daily_goal = float(settings.daily_profit_goal_usd)
        weekly_goal = float(settings.weekly_profit_goal_usd)

        daily_met = daily_goal > 0 and daily_pnl >= daily_goal
        weekly_met = weekly_goal > 0 and weekly_pnl >= weekly_goal

        daily_expected_by_now = daily_goal * day_elapsed_fraction if daily_goal > 0 else 0.0
        weekly_expected_by_now = weekly_goal * week_elapsed_fraction if weekly_goal > 0 else 0.0

        return {
            "daily_pnl": round(daily_pnl, 6),
            "weekly_pnl": round(weekly_pnl, 6),
            "daily_goal": daily_goal,
            "weekly_goal": weekly_goal,
            "daily_goal_met": daily_met,
            "weekly_goal_met": weekly_met,
            "daily_expected_by_now": round(daily_expected_by_now, 6),
            "weekly_expected_by_now": round(weekly_expected_by_now, 6),
            "day_elapsed_fraction": round(day_elapsed_fraction, 6),
            "week_elapsed_fraction": round(week_elapsed_fraction, 6),
        }

    def _check_live_order_caps(self, order_notional: float, created_at: str) -> tuple[bool, str, dict]:
        as_of = self._parse_iso_datetime(created_at)
        day_start = datetime(as_of.year, as_of.month, as_of.day, tzinfo=as_of.tzinfo)

        with get_connection() as conn:
            stats = conn.execute(
                """
                SELECT
                  COUNT(1) AS order_count,
                  COALESCE(SUM(quantity * price), 0.0) AS notional_sum
                FROM orders
                WHERE mode = 'live'
                  AND side = 'buy'
                  AND status IN ('live_ready', 'live_submitted', 'live_filled')
                  AND created_at >= ?
                """,
                (day_start.isoformat(),),
            ).fetchone()

        existing_orders = int(stats["order_count"] if stats else 0)
        existing_notional = float(stats["notional_sum"] if stats else 0.0)

        if order_notional > settings.live_max_order_notional_usd:
            return False, "order_notional_limit", {
                "order_notional": round(order_notional, 6),
                "max_order_notional": settings.live_max_order_notional_usd,
                "existing_daily_notional": round(existing_notional, 6),
                "existing_orders": existing_orders,
            }

        if existing_notional + order_notional > settings.live_max_daily_notional_usd:
            return False, "daily_notional_limit", {
                "order_notional": round(order_notional, 6),
                "max_daily_notional": settings.live_max_daily_notional_usd,
                "existing_daily_notional": round(existing_notional, 6),
                "existing_orders": existing_orders,
            }

        if existing_orders + 1 > settings.live_max_orders_per_day:
            return False, "daily_order_count_limit", {
                "order_notional": round(order_notional, 6),
                "max_orders_per_day": settings.live_max_orders_per_day,
                "existing_daily_notional": round(existing_notional, 6),
                "existing_orders": existing_orders,
            }

        return True, "ok", {
            "order_notional": round(order_notional, 6),
            "existing_daily_notional": round(existing_notional, 6),
            "existing_orders": existing_orders,
        }

    def _compute_goal_pacing_multiplier(self, created_at: str) -> dict:
        progress = self._get_profit_goal_progress(created_at)
        if not settings.goal_pacing_enabled:
            return {
                "risk_multiplier": 1.0,
                "daily_ratio": 1.0,
                "weekly_ratio": 1.0,
                "enabled": False,
                "preset": settings.goal_pacing_preset,
            }

        preset = settings.goal_pacing_preset
        if preset == "conservative":
            min_mult = 0.8
            max_mult = 1.15
            weight_daily = 0.6
            weight_weekly = 0.4
            sensitivity = 0.22
        elif preset == "aggressive":
            min_mult = 0.5
            max_mult = 1.8
            weight_daily = 0.45
            weight_weekly = 0.55
            sensitivity = 0.55
        elif preset == "custom":
            min_mult = settings.goal_pacing_min_multiplier
            max_mult = settings.goal_pacing_max_multiplier
            weight_daily = settings.goal_pacing_weight_daily
            weight_weekly = settings.goal_pacing_weight_weekly
            sensitivity = settings.goal_pacing_sensitivity
        else:
            min_mult = 0.6
            max_mult = 1.4
            weight_daily = 0.5
            weight_weekly = 0.5
            sensitivity = 0.35

        def ratio(actual: float, expected: float) -> float:
            if expected <= 0:
                return 1.0
            return max(0.0, actual / expected)

        daily_ratio = ratio(float(progress.get("daily_pnl", 0.0)), float(progress.get("daily_expected_by_now", 0.0)))
        weekly_ratio = ratio(float(progress.get("weekly_pnl", 0.0)), float(progress.get("weekly_expected_by_now", 0.0)))

        total_weight = weight_daily + weight_weekly
        if total_weight <= 0:
            weighted_ratio = 1.0
        else:
            weighted_ratio = (
                daily_ratio * weight_daily
                + weekly_ratio * weight_weekly
            ) / total_weight

        delta = 1.0 - weighted_ratio
        raw_multiplier = 1.0 + (delta * sensitivity)
        bounded_multiplier = max(
            min_mult,
            min(max_mult, raw_multiplier),
        )

        return {
            "risk_multiplier": round(bounded_multiplier, 4),
            "daily_ratio": round(daily_ratio, 4),
            "weekly_ratio": round(weekly_ratio, 4),
            "weighted_ratio": round(weighted_ratio, 4),
            "enabled": True,
            "preset": preset,
            "bounds": {"min": round(min_mult, 4), "max": round(max_mult, 4)},
        }

    def _adaptive_special_circumstances_enabled_for_mode(self) -> bool:
        if not bool(getattr(settings, "adaptive_special_circumstances_enabled", False)):
            return False
        mode = str(getattr(settings, "bot_mode", "paper") or "paper").lower()
        if mode == "live":
            return bool(getattr(settings, "adaptive_special_circumstances_live_enabled", False))
        return bool(getattr(settings, "adaptive_special_circumstances_paper_enabled", False))

    def _load_special_circumstances(self) -> dict:
        path = Path(getattr(settings, "adaptive_special_circumstances_path", Path("runtime/special_circumstances.json")))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        default_payload = {
            "rules": [],
            "history": [],
            "pair_stats": {},
            "updated_at": "",
            "last_hourly_review_at": "",
            "last_close_event_scan_at": "",
        }
        if not path.exists():
            try:
                path.write_text(json.dumps(default_payload, indent=2), encoding="utf-8")
            except Exception:
                return default_payload
            return default_payload

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default_payload

        if isinstance(payload, list):
            return {"rules": payload, "history": [], "updated_at": ""}
        if not isinstance(payload, dict):
            return default_payload

        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            rules = []
        history = payload.get("history", [])
        if not isinstance(history, list):
            history = []
        pair_stats = payload.get("pair_stats", {})
        if not isinstance(pair_stats, dict):
            pair_stats = {}
        updated_at = str(payload.get("updated_at", "") or "")
        last_hourly_review_at = str(payload.get("last_hourly_review_at", "") or "")
        last_close_event_scan_at = str(payload.get("last_close_event_scan_at", "") or "")
        return {
            "rules": rules,
            "history": history,
            "pair_stats": pair_stats,
            "updated_at": updated_at,
            "last_hourly_review_at": last_hourly_review_at,
            "last_close_event_scan_at": last_close_event_scan_at,
        }

    def _save_special_circumstances(self, payload: dict) -> None:
        path = Path(getattr(settings, "adaptive_special_circumstances_path", Path("runtime/special_circumstances.json")))
        payload_copy = dict(payload or {})
        payload_copy["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload_copy, indent=2), encoding="utf-8")

    def _adaptive_hourly_rule_generation_enabled_for_mode(self) -> bool:
        if not bool(getattr(settings, "adaptive_hourly_rule_generation_enabled", False)):
            return False
        if not self._adaptive_special_circumstances_enabled_for_mode():
            return False
        mode = str(getattr(settings, "bot_mode", "paper") or "paper").lower()
        if mode == "live":
            return bool(getattr(settings, "adaptive_special_circumstances_live_enabled", False))
        return bool(getattr(settings, "adaptive_special_circumstances_paper_enabled", False))

    def _resolve_special_sell_override(
        self,
        symbol: str,
        kind: str,
        buy_rule_ids: list[str],
        payload: dict,
    ) -> dict:
        if not buy_rule_ids:
            return {}
        if not self._adaptive_special_circumstances_enabled_for_mode():
            return {}

        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        pair_stats = payload.get("pair_stats", {}) if isinstance(payload, dict) else {}
        if not isinstance(rules, list):
            return {}

        mode = str(getattr(settings, "bot_mode", "paper") or "paper").lower()
        sym = str(symbol or "").upper().replace("-USD", "")
        kd = str(kind or "").lower()
        buy_rule_id_set = {str(x).strip() for x in buy_rule_ids if str(x).strip()}

        for raw_rule in rules:
            if not isinstance(raw_rule, dict):
                continue
            if not bool(raw_rule.get("enabled", True)):
                continue
            if str(raw_rule.get("type", "") or "").lower() != "sell_exit_override":
                continue

            source_role = str(raw_rule.get("source_role", "") or "").strip().lower()
            if source_role == "technician" and not bool(getattr(settings, "adaptive_technician_enabled", True)):
                continue

            applies_to = str(raw_rule.get("applies_to", "both") or "both").lower()
            if applies_to not in {"both", mode}:
                continue

            pair_id = str(raw_rule.get("pair_id", "") or "")
            pair_stat = pair_stats.get(pair_id, {}) if isinstance(pair_stats, dict) else {}
            if isinstance(pair_stat, dict) and bool(pair_stat.get("disabled", False)):
                continue

            rule_symbols = [str(x).upper().replace("-USD", "") for x in (raw_rule.get("symbols", []) or []) if str(x).strip()]
            if rule_symbols and sym not in set(rule_symbols):
                continue
            rule_kinds = [str(x).lower() for x in (raw_rule.get("kinds", []) or []) if str(x).strip()]
            if rule_kinds and kd not in set(rule_kinds):
                continue

            entry_buy_rule_id = str(raw_rule.get("entry_buy_rule_id", "") or "").strip()
            if entry_buy_rule_id and entry_buy_rule_id not in buy_rule_id_set:
                continue

            return {
                "rule_id": str(raw_rule.get("id", "") or ""),
                "pair_id": pair_id,
                "take_profit_pct": float(raw_rule.get("take_profit_pct", settings.paper_early_climb_take_profit_pct) or settings.paper_early_climb_take_profit_pct),
                "stop_loss_pct": float(raw_rule.get("stop_loss_pct", settings.paper_early_climb_stop_loss_pct) or settings.paper_early_climb_stop_loss_pct),
                "trailing_stop_pct": float(raw_rule.get("trailing_stop_pct", settings.paper_early_climb_trailing_stop_pct) or settings.paper_early_climb_trailing_stop_pct),
                "max_hold_minutes": int(raw_rule.get("max_hold_minutes", settings.paper_early_climb_max_hold_minutes) or settings.paper_early_climb_max_hold_minutes),
                "source_role": source_role,
            }

        return {}

    def _refresh_hourly_special_circumstances(self, created_at: str) -> None:
        payload = self._load_special_circumstances()
        now_dt = self._parse_iso_datetime(created_at)
        last_review_at = str(payload.get("last_hourly_review_at", "") or "")
        if last_review_at:
            last_dt = self._parse_iso_datetime(last_review_at)
            if (now_dt - last_dt).total_seconds() < 3600:
                return

        stats = self._ingest_special_circumstance_outcomes(payload, created_at)
        created_pairs = []
        if self._adaptive_hourly_rule_generation_enabled_for_mode():
            created_pairs = self._generate_hourly_special_rule_pairs(payload, created_at)

        payload["last_hourly_review_at"] = created_at
        self._save_special_circumstances(payload)

        self._persist_system_event(
            "special_circumstances_hourly_review",
            f"Hourly special-circumstance review completed (new_pairs={len(created_pairs)})",
            created_at,
            {
                "new_rule_pairs": len(created_pairs),
                "created_pairs": created_pairs,
                "stats": stats,
                "hourly_generation_enabled": self._adaptive_hourly_rule_generation_enabled_for_mode(),
            },
        )

    def _ingest_special_circumstance_outcomes(self, payload: dict, created_at: str) -> dict:
        last_scan = str(payload.get("last_close_event_scan_at", "") or "")
        if not last_scan:
            last_scan = (self._parse_iso_datetime(created_at) - timedelta(hours=48)).isoformat()

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT created_at, metadata_json
                FROM system_events
                WHERE event_type = 'paper_close' AND created_at > ?
                ORDER BY id ASC
                """,
                (last_scan,),
            ).fetchall()

        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        pair_stats = payload.get("pair_stats", {}) if isinstance(payload, dict) else {}
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(pair_stats, dict):
            pair_stats = {}
        if not isinstance(history, list):
            history = []

        buy_rule_to_pair: dict[str, str] = {}
        pair_to_rule_ids: dict[str, set[str]] = {}
        for rule in rules if isinstance(rules, list) else []:
            if not isinstance(rule, dict):
                continue
            pair_id = str(rule.get("pair_id", "") or "")
            rule_id = str(rule.get("id", "") or "")
            if pair_id and rule_id:
                pair_to_rule_ids.setdefault(pair_id, set()).add(rule_id)
            if str(rule.get("type", "") or "").lower() == "daytrade_exception":
                if rule_id and pair_id:
                    buy_rule_to_pair[rule_id] = pair_id

        outcome_count = 0
        disabled_pairs: list[str] = []
        latest_scan_time = last_scan
        for row in rows:
            created = str(row["created_at"] or "")
            latest_scan_time = created or latest_scan_time
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            details = metadata.get("details", []) if isinstance(metadata, dict) else []
            if not isinstance(details, list):
                continue

            for detail in details:
                if not isinstance(detail, dict):
                    continue
                pnl = float(detail.get("pnl", 0.0) or 0.0)
                buy_rule_ids = detail.get("buy_rule_ids", [])
                if not isinstance(buy_rule_ids, list):
                    continue

                impacted_pairs = {
                    buy_rule_to_pair.get(str(rule_id).strip(), "")
                    for rule_id in buy_rule_ids
                    if str(rule_id).strip()
                }
                impacted_pairs.discard("")

                for pair_id in impacted_pairs:
                    stat = pair_stats.get(pair_id, {})
                    if not isinstance(stat, dict):
                        stat = {}
                    stat["trades"] = int(stat.get("trades", 0) or 0) + 1
                    stat["wins"] = int(stat.get("wins", 0) or 0) + (1 if pnl > 0 else 0)
                    stat["losses"] = int(stat.get("losses", 0) or 0) + (1 if pnl <= 0 else 0)
                    stat["net_pnl"] = round(float(stat.get("net_pnl", 0.0) or 0.0) + pnl, 6)
                    stat["last_closed_at"] = created
                    stat.setdefault("disabled", False)
                    pair_stats[pair_id] = stat
                    outcome_count += 1

                    max_losses = int(getattr(settings, "adaptive_rule_pair_max_losses", 2))
                    max_net_loss = float(getattr(settings, "adaptive_rule_pair_max_net_loss_usd", 20.0))
                    should_disable = (
                        int(stat.get("losses", 0) or 0) >= max_losses
                        or float(stat.get("net_pnl", 0.0) or 0.0) <= -abs(max_net_loss)
                    )
                    if should_disable and not bool(stat.get("disabled", False)):
                        stat["disabled"] = True
                        pair_stats[pair_id] = stat
                        disabled_pairs.append(pair_id)
                        for rule in rules if isinstance(rules, list) else []:
                            if isinstance(rule, dict) and str(rule.get("pair_id", "") or "") == pair_id:
                                rule["enabled"] = False
                        history.append(
                            {
                                "at": created,
                                "action": "disable_pair",
                                "pair_id": pair_id,
                                "reason": "underperforming",
                                "stats": stat,
                            }
                        )

        payload["pair_stats"] = pair_stats
        payload["history"] = history[-300:]
        if latest_scan_time:
            payload["last_close_event_scan_at"] = latest_scan_time

        return {
            "outcomes_processed": outcome_count,
            "disabled_pairs": disabled_pairs,
        }

    def _generate_hourly_special_rule_pairs(self, payload: dict, created_at: str) -> list[dict]:
        mode = str(getattr(settings, "bot_mode", "paper") or "paper").lower()
        pattern_scope = str(getattr(settings, "adaptive_exception_scope", "asset_class") or "asset_class").lower()
        if pattern_scope not in {"symbol", "asset_class", "global"}:
            pattern_scope = "asset_class"
        since = (self._parse_iso_datetime(created_at) - timedelta(hours=1)).isoformat()
        min_ratio = float(getattr(settings, "adaptive_min_candidate_daytrade_score_ratio", 0.75) or 0.75)
        min_score = float(settings.daytrade_min_score) * max(0.0, min(1.0, min_ratio))
        max_pairs = int(getattr(settings, "adaptive_max_new_rule_pairs_per_hour", 2) or 2)
        max_pairs = max(0, max_pairs)
        if max_pairs <= 0:
            return []

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT metadata_json, created_at
                FROM system_events
                WHERE event_type = 'symbol_skipped_daytrade_filter'
                  AND created_at >= ?
                ORDER BY id DESC
                """,
                (since,),
            ).fetchall()

        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        history = payload.get("history", []) if isinstance(payload, dict) else []
        pair_stats = payload.get("pair_stats", {}) if isinstance(payload, dict) else {}
        if not isinstance(rules, list):
            rules = []
        if not isinstance(history, list):
            history = []
        if not isinstance(pair_stats, dict):
            pair_stats = {}

        existing_buy_targets: set[tuple[str, str]] = set()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("type", "") or "").lower() != "daytrade_exception":
                continue
            if not bool(rule.get("enabled", True)):
                continue
            symbols = [str(x).upper().replace("-USD", "") for x in (rule.get("symbols", []) or []) if str(x).strip()]
            kinds = [str(x).lower() for x in (rule.get("kinds", []) or []) if str(x).strip()]
            for sym in symbols or [""]:
                for kd in kinds or [""]:
                    existing_buy_targets.add((sym, kd))

        created: list[dict] = []
        seen_targets: set[tuple[str, str]] = set()
        now_ts = int(self._parse_iso_datetime(created_at).timestamp())
        sequence = 0

        for row in rows:
            if len(created) >= max_pairs:
                break
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}

            symbol = str(metadata.get("symbol", "") or "").upper().replace("-USD", "")
            kind = str(metadata.get("kind", "") or "").lower()
            trend_score = float(metadata.get("trend_score", 0.0) or 0.0)
            profile = metadata.get("daytrade_profile", {}) if isinstance(metadata, dict) else {}
            score = float(profile.get("score", 0.0) or 0.0) if isinstance(profile, dict) else 0.0
            reason = str(profile.get("reason", "") or "").lower() if isinstance(profile, dict) else ""

            if not symbol or kind not in {"stock", "crypto"}:
                continue
            if score < min_score:
                continue
            if trend_score < 25.0:
                continue
            if reason and "insufficient_candles" not in reason and "weak" not in reason:
                continue

            if pattern_scope == "symbol":
                target_symbols = [symbol]
                target_kinds = [kind]
            elif pattern_scope == "global":
                target_symbols = []
                target_kinds = []
            else:
                target_symbols = []
                target_kinds = [kind]

            target_key = (target_symbols[0] if target_symbols else "", target_kinds[0] if target_kinds else "")
            if target_key in seen_targets:
                continue
            if target_key in existing_buy_targets:
                continue

            sequence += 1
            pair_id = f"pair_{symbol}_{kind}_{now_ts}_{sequence}"
            buy_rule_id = f"buy_{pair_id}"
            sell_rule_id = f"sell_{pair_id}"

            take_profit = max(0.01, float(settings.paper_early_climb_take_profit_pct) * 0.9)
            stop_loss = max(0.005, float(settings.paper_early_climb_stop_loss_pct) * 0.9)
            trailing = max(0.003, float(settings.paper_early_climb_trailing_stop_pct) * 0.9)
            max_hold = max(10, int(float(settings.paper_early_climb_max_hold_minutes) * 0.9))

            buy_rule = {
                "id": buy_rule_id,
                "pair_id": pair_id,
                "type": "daytrade_exception",
                "enabled": True,
                "generated": True,
                "generated_at": created_at,
                "source_role": "technician",
                "applies_to": mode,
                "symbols": target_symbols,
                "kinds": target_kinds,
                "reason_contains": ["insufficient_candles"],
                "allow_threshold_miss_match": True,
                "pattern_scope": pattern_scope,
                "min_daytrade_score": round(max(0.0, score - 8.0), 3),
                "force_eligible": True,
                "override_daytrade_min_score": True,
            }

            sell_rule = {
                "id": sell_rule_id,
                "pair_id": pair_id,
                "type": "sell_exit_override",
                "enabled": True,
                "generated": True,
                "generated_at": created_at,
                "source_role": "technician",
                "applies_to": mode,
                "symbols": target_symbols,
                "kinds": target_kinds,
                "entry_buy_rule_id": buy_rule_id,
                "take_profit_pct": round(take_profit, 6),
                "stop_loss_pct": round(stop_loss, 6),
                "trailing_stop_pct": round(trailing, 6),
                "max_hold_minutes": int(max_hold),
            }

            rules.append(buy_rule)
            rules.append(sell_rule)
            pair_stats[pair_id] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "net_pnl": 0.0,
                "disabled": False,
                "created_at": created_at,
            }

            creation_record = {
                "at": created_at,
                "action": "create_pair",
                "pair_id": pair_id,
                "symbol": symbol,
                "kind": kind,
                "pattern_scope": pattern_scope,
                "score": round(score, 3),
                "trend_score": round(trend_score, 3),
                "buy_rule_id": buy_rule_id,
                "sell_rule_id": sell_rule_id,
            }
            history.append(creation_record)
            created.append(creation_record)
            seen_targets.add(target_key)

        payload["rules"] = rules
        payload["history"] = history[-300:]
        payload["pair_stats"] = pair_stats
        return created

    def _apply_special_circumstances_daytrade(
        self,
        symbol: str,
        kind: str,
        daytrade_profile: dict,
        created_at: str,
    ) -> dict:
        profile = dict(daytrade_profile or {})
        profile.setdefault("special_circumstances", {})
        special_meta = {
            "enabled": False,
            "applied": [],
            "blocked": [],
            "override_daytrade_min_score": False,
        }

        if not self._adaptive_special_circumstances_enabled_for_mode():
            profile["special_circumstances"] = special_meta
            return profile

        payload = self._load_special_circumstances()
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        if not isinstance(rules, list) or not rules:
            special_meta["enabled"] = True
            profile["special_circumstances"] = special_meta
            return profile

        current_mode = str(getattr(settings, "bot_mode", "paper") or "paper").lower()
        sym = str(symbol or "").upper().replace("-USD", "")
        kd = str(kind or "").lower()
        reason_text = str(profile.get("reason", "") or "").lower()
        score = float(profile.get("score", 0.0) or 0.0)
        thresholds = profile.get("thresholds", {}) if isinstance(profile.get("thresholds", {}), dict) else {}
        min_score_threshold = float(thresholds.get("min_score", settings.daytrade_min_score) or settings.daytrade_min_score)
        threshold_miss_fallback = (
            not bool(profile.get("eligible", False))
            and score >= (min_score_threshold * 0.75)
            and (
                float(profile.get("avg_bar_move_pct", 0.0) or 0.0) >= (float(thresholds.get("min_avg_bar_move_pct", 0.0) or 0.0) * 0.55)
                or float(profile.get("avg_bar_range_pct", 0.0) or 0.0) >= (float(thresholds.get("min_avg_bar_range_pct", 0.0) or 0.0) * 0.50)
            )
        )

        for raw_rule in rules:
            if not isinstance(raw_rule, dict):
                continue
            if not bool(raw_rule.get("enabled", True)):
                continue

            source_role = str(raw_rule.get("source_role", "") or "").strip().lower()
            if source_role == "technician" and not bool(getattr(settings, "adaptive_technician_enabled", True)):
                special_meta["blocked"].append(
                    {
                        "rule_id": str(raw_rule.get("id", "")),
                        "reason": "technician_disabled",
                    }
                )
                continue

            rule_mode = str(raw_rule.get("applies_to", "both") or "both").lower()
            if rule_mode not in {"both", current_mode}:
                continue

            rule_type = str(raw_rule.get("type", "daytrade_exception") or "daytrade_exception").lower()
            if rule_type != "daytrade_exception":
                continue

            rule_symbols = [str(x).upper().replace("-USD", "") for x in (raw_rule.get("symbols", []) or []) if str(x).strip()]
            if rule_symbols and sym not in set(rule_symbols):
                continue

            rule_kinds = [str(x).lower() for x in (raw_rule.get("kinds", []) or []) if str(x).strip()]
            if rule_kinds and kd not in set(rule_kinds):
                continue

            reason_contains = [str(x).lower() for x in (raw_rule.get("reason_contains", []) or []) if str(x).strip()]
            reason_matched = any(token in reason_text for token in reason_contains) if reason_contains else False
            if reason_contains and not reason_matched:
                if not bool(raw_rule.get("allow_threshold_miss_match", True)):
                    continue
                if not threshold_miss_fallback:
                    continue

            min_score = float(raw_rule.get("min_daytrade_score", 0.0) or 0.0)
            if score < min_score:
                continue

            if bool(raw_rule.get("force_eligible", True)):
                profile["eligible"] = True
                profile["eligible_path"] = "special_circumstance"

            if bool(raw_rule.get("override_daytrade_min_score", False)):
                special_meta["override_daytrade_min_score"] = True

            special_meta["applied"].append(
                {
                    "rule_id": str(raw_rule.get("id", "")),
                    "source_role": source_role,
                }
            )

        special_meta["enabled"] = True
        profile["special_circumstances"] = special_meta
        if special_meta["applied"]:
            self._persist_system_event(
                "special_circumstance_applied",
                f"Applied {len(special_meta['applied'])} special circumstance rule(s) for {symbol}",
                created_at,
                {
                    "symbol": symbol,
                    "kind": kind,
                    "applied": special_meta["applied"],
                    "blocked": special_meta["blocked"],
                    "daytrade_score": score,
                    "threshold_miss_fallback": bool(threshold_miss_fallback),
                },
            )
        return profile

    @staticmethod
    def _parse_iso_datetime(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _is_external_kill_switch_active() -> bool:
        return Path(settings.external_kill_switch_path).exists()

    def _crypto_symbol_set(self) -> set[str]:
        symbols: set[str] = set()
        try:
            for item in get_watchlist(kinds=["crypto"], include_penny=False, include_micro_crypto=True):
                clean = str(getattr(item, "symbol", "")).upper().replace("-USD", "")
                if clean:
                    symbols.add(clean)
        except Exception:
            pass

        try:
            for symbol in self.robinhood.get_crypto_symbols():
                clean = str(symbol).upper().replace("-USD", "")
                if clean:
                    symbols.add(clean)
        except Exception:
            pass
        return symbols

    def _count_open_orders_by_kind(self) -> dict[str, int]:
        crypto_symbols = self._crypto_symbol_set()
        counts = {"stock": 0, "crypto": 0, "total": 0}
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol
                FROM orders
                WHERE side = 'buy'
                  AND status IN ('paper_proposed', 'paper_filled', 'live_ready', 'live_submitted', 'live_filled')
                """
            ).fetchall()

        for row in rows:
            sym = str(row["symbol"] or "").upper().replace("-USD", "")
            if not sym:
                continue
            kind = "crypto" if (sym in crypto_symbols or "-" in str(row["symbol"] or "")) else "stock"
            counts[kind] += 1
            counts["total"] += 1

        return counts

    def _count_open_orders(self) -> int:
        return int(self._count_open_orders_by_kind().get("total", 0))

    def _persist_news_items(self, news_items: list[dict], fetched_at: str) -> None:
        if not news_items:
            return

        rows = [
            (
                str(item.get("symbol", "")),
                str(item.get("source", "")),
                str(item.get("title", ""))[:500],
                str(item.get("url", "")),
                str(item.get("published_at", "")),
                float(item.get("sentiment", 0.0)),
                fetched_at,
            )
            for item in news_items
        ]

        with get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO news_items (symbol, source, title, url, published_at, sentiment, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def _persist_system_event(self, event_type: str, message: str, created_at: str, metadata: dict | None = None) -> None:
        event_metadata = metadata or {}
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO system_events (event_type, message, metadata_json, created_at) VALUES (?, ?, ?, ?)",
                (event_type, message, json.dumps(event_metadata), created_at),
            )
            conn.commit()

        try:
            self.alerts.send(event_type, message, event_metadata)
        except Exception as exc:
            logger.warning("Alert routing failed for {}: {}", event_type, exc)

    @staticmethod
    def _write_cycle_report(report: dict) -> None:
        runtime_dir = Path("runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        report_path = runtime_dir / "cycle_report_latest.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    @staticmethod
    def _write_hourly_strategy_review(review: dict) -> None:
        runtime_dir = Path("runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        report_path = runtime_dir / "hourly_strategy_review_latest.json"
        report_path.write_text(json.dumps(review, indent=2), encoding="utf-8")


def validate_live_mode_guardrails() -> None:
    if Path(settings.external_kill_switch_path).exists():
        raise RuntimeError(
            f"External kill switch is active at {settings.external_kill_switch_path}. Clear it before startup."
        )
    if settings.bot_mode == "live" and not settings.live_mode_unlock:
        raise RuntimeError("Live mode requested without unlock flag. Refusing to start.")
