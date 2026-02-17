"""Backtesting engine — replays historical candle data through the AI pipeline.

The backtester simulates what the bot *would* have done over a historical
period by presenting each candle window to the Ollama model, collecting
BUY / HOLD / AVOID decisions, managing a virtual $500 portfolio, and
recording every simulated trade with realistic slippage & fees.

Results are persisted to the ``backtest_runs`` and ``backtest_trades`` tables
so the dashboard (and you) can inspect them.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from .council import TradingCouncil, CouncilDecision
from .db import get_connection
from .historical_data import Candle, HistoricalDataResult, fetch_historical
from .indicators import TechnicalSignals, compute_signals
from .ollama_client import OllamaClient
from .sentiment import SentimentAnalyzer, MarketSentiment
from .settings import settings
from .watchlist import WatchlistItem, get_watchlist


# ── data classes ──────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Parameters that define a single backtest run."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    starting_capital: float = 500.0
    goal_equity: float = 1000.0      # target equity ($500 → $1000)
    max_risk_per_trade_pct: float = 0.05   # 5% risk budget per trade
    max_position_pct: float = 0.30         # up to 30% of equity per position
    max_concurrent_positions: int = 3
    min_confidence: float = 0.60
    slippage_bps: float = 8.0
    fee_bps: float = 5.0
    lookback_window: int = 24       # candles fed to AI prompt
    hold_candles_min: int = 3       # min candles before closing
    hold_candles_max: int = 15      # force-close after this many candles
    profit_target_pct: float = 5.0  # take profit at +5%
    stop_loss_pct: float = 3.0      # cut loss at -3% (asymmetric R:R)
    trailing_stop_pct: float = 2.5  # trail 2.5% from peak
    signal_score_gate: int = 10     # min technical score to even ask AI
    use_council: bool = True          # use 3-agent council (parallel)
    use_ai: bool = True               # False = indicator-only fast mode (no GPU)
    period: str = "6mo"             # yfinance period
    interval: str = "1d"            # candle size
    kinds: list[str] | None = None  # crypto / stock / both
    symbols: list[str] | None = None
    extra_items: list | None = None   # ad-hoc WatchlistItems for non-watchlist tickers


@dataclass
class VirtualPosition:
    symbol: str
    kind: str
    entry_price: float
    quantity: float
    entry_candle_idx: int
    entry_timestamp: str
    confidence: float
    rationale: str
    peak_price: float = 0.0  # for trailing stop


@dataclass
class ClosedTrade:
    symbol: str
    kind: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    hold_candles: int
    entry_timestamp: str
    exit_timestamp: str
    confidence: float
    rationale: str


@dataclass
class BacktestResult:
    run_id: str
    config: BacktestConfig
    trades: list[ClosedTrade] = field(default_factory=list)
    final_equity: float = 0.0
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_estimate: float = 0.0
    symbols_tested: list[str] = field(default_factory=list)
    candles_processed: int = 0
    ai_calls: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── main engine ───────────────────────────────────────────────────

class Backtester:
    """Replay historical data through the Ollama AI pipeline."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.ollama = OllamaClient()

        # portfolio state
        self._cash: float = self.config.starting_capital
        self._positions: list[VirtualPosition] = []
        self._closed: list[ClosedTrade] = []
        self._equity_curve: list[float] = []
        self._council: TradingCouncil | None = None
        self._sentiment_analyzer: SentimentAnalyzer | None = None
        self._sentiment_cache: dict[str, MarketSentiment | None] = {}
        self._market_mood: dict = {}

        # AI model resolution (done once)
        self._ranked_models: list[str] = []
        self._model_name: str = ""

    # ── public ────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the full backtest and return results."""
        t0 = time.perf_counter()
        logger.info("=== Backtest {} starting ===", self.config.run_id)
        logger.info(
            "Capital: ${:.2f}  Period: {}  Interval: {}  Kinds: {}  Symbols: {}",
            self.config.starting_capital,
            self.config.period,
            self.config.interval,
            self.config.kinds or "all",
            self.config.symbols or "all",
        )

        # resolve AI model
        if self.config.use_ai:
            try:
                installed = self.ollama.list_models()
                choice = self.ollama.select_best_model(installed)
                self._ranked_models = self.ollama.rank_models(installed)
                self._model_name = choice.name
                logger.info("Backtest using model: {} ({})", choice.name, choice.reason)
            except Exception as exc:
                return self._error_result(f"Failed to resolve Ollama model: {exc}", time.perf_counter() - t0)

            # Initialize council if enabled
            if self.config.use_council:
                self._council = TradingCouncil(
                    ollama=self.ollama,
                    ranked_models=self._ranked_models,
                )
                self._sentiment_analyzer = SentimentAnalyzer()
                try:
                    self._market_mood = self._sentiment_analyzer.get_market_mood()
                except Exception:
                    self._market_mood = {}
                logger.info("Council mode ENABLED — 3-agent parallel voting")
        else:
            logger.info("FAST MODE — indicator-only decisions (no AI calls)")

        # fetch historical data for all watchlist items
        items = get_watchlist(kinds=self.config.kinds, symbols=self.config.symbols)

        # Merge any extra ad-hoc items (for dynamically discovered tickers)
        if self.config.extra_items:
            existing_syms = {i.symbol for i in items}
            for extra in self.config.extra_items:
                if extra.symbol not in existing_syms:
                    items.append(extra)
                    existing_syms.add(extra.symbol)

        if not items:
            return self._error_result("No watchlist items matched filters", time.perf_counter() - t0)

        total_candles = 0
        total_ai_calls = 0
        symbols_tested: list[str] = []
        errors: list[str] = []

        try:
            for item in items:
                logger.info("── Fetching historical data for {} ({}) ──", item.symbol, item.yf_ticker)
                result = fetch_historical(item, period=self.config.period, interval=self.config.interval)
                if result.error:
                    errors.append(f"{item.symbol}: {result.error}")
                    logger.warning("Skipping {}: {}", item.symbol, result.error)
                    continue
                if len(result.candles) < self.config.lookback_window + 2:
                    errors.append(f"{item.symbol}: only {len(result.candles)} candles (need {self.config.lookback_window + 2})")
                    continue

                symbols_tested.append(item.symbol)
                candles_done, ai_done = self._replay_symbol(item, result.candles)
                total_candles += candles_done
                total_ai_calls += ai_done
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user — saving partial results")
            errors.append("Interrupted by user")
        except Exception as exc:
            logger.error("Backtest error — saving partial results: {}", exc)
            errors.append(f"Runtime error: {exc}")

        # force-close any remaining open positions at last price
        self._force_close_all()

        duration = time.perf_counter() - t0
        result = self._compile_result(symbols_tested, total_candles, total_ai_calls, duration, errors)

        # persist to DB (always, even on partial completion)
        self._persist_results(result)

        logger.info(
            "=== Backtest {} complete in {:.1f}s — PnL: ${:.2f}  Win rate: {:.1%}  Trades: {} ===",
            result.run_id, duration, result.total_pnl, result.win_rate,
            len(result.trades),
        )
        return result

    # ── per-symbol replay ─────────────────────────────────────────

    def _replay_symbol(self, item: WatchlistItem, candles: list[Candle]) -> tuple[int, int]:
        """Walk through candles for one symbol, calling the AI at each step.

        Returns (candles_processed, ai_calls_made).
        """
        window = self.config.lookback_window
        ai_calls = 0

        for i in range(window, len(candles)):
            current = candles[i]
            history_window = candles[i - window : i]

            # close positions that hit TP/SL/trailing-stop/max-hold
            self._try_close_positions(item, current, i)

            # skip if already have max positions or one open for this symbol
            open_for_symbol = [p for p in self._positions if p.symbol == item.symbol]
            if len(self._positions) >= self.config.max_concurrent_positions:
                continue
            if open_for_symbol:
                continue

            # compute technical indicators
            signals = compute_signals(history_window, current)

            # GATE 1: skip if technicals are weak (saves GPU time)
            if signals.signal_score < self.config.signal_score_gate:
                continue

            # GATE 2: TREND FILTER — only buy in uptrends
            # Require BOTH: SMA7 > SMA21 AND price above SMA21
            if signals.sma_fast > 0 and signals.sma_slow > 0:
                if signals.sma_fast < signals.sma_slow:
                    continue  # short-term bearish crossover — skip
                if current.close < signals.sma_slow:
                    continue  # price below slow MA — not a real uptrend

            # GATE 3: RSI sanity — don't buy if already overbought
            if signals.rsi_14 > 70:
                continue

            # Get sentiment (cached per symbol)
            sentiment = self._get_cached_sentiment(item.symbol, item.kind)

            # Compact candle data for prompts
            candle_data = [
                {
                    "o": round(c.open, 6), "h": round(c.high, 6),
                    "l": round(c.low, 6), "c": round(c.close, 6),
                    "v": round(c.volume, 2),
                }
                for c in history_window[-6:]
            ]

            # ── FAST MODE: indicator-only (no AI calls) ──
            if not self.config.use_ai:
                # Derive action/confidence purely from technical signals
                confidence = min(1.0, max(0.0, signals.signal_score / 100.0))
                # Boost confidence if multiple indicators align
                bullish_count = 0
                if signals.rsi_14 < 40:
                    bullish_count += 1  # oversold bounce
                if signals.momentum_5 > 0:
                    bullish_count += 1  # positive momentum
                if signals.bollinger_position < 0.2:
                    bullish_count += 1  # near lower Bollinger band
                if signals.momentum_10 is not None and signals.momentum_10 > 0:
                    bullish_count += 1  # longer-term momentum
                if signals.volume_ratio is not None and signals.volume_ratio > 1.5:
                    bullish_count += 1  # above-average volume

                # Need at least 2 bullish signals to buy
                if bullish_count >= 2:
                    confidence = min(1.0, confidence + bullish_count * 0.05)
                    action = "BUY"
                    rationale = f"Indicator-only: score={signals.signal_score}, bullish_signals={bullish_count}"
                else:
                    continue  # not enough bullish signals

                if confidence < self.config.min_confidence:
                    continue

                self._open_position(item, current, i, confidence, rationale, signals)
                continue

            # ── COUNCIL MODE: 3 agents vote in parallel ──
            if self.config.use_council and self._council:
                try:
                    equity = self._cash + self._open_notional()
                    win_count = sum(1 for t in self._closed if t.net_pnl > 0)
                    total_closed = len(self._closed)
                    win_rate = win_count / total_closed if total_closed > 0 else 0.5
                    recent_pnl = sum(t.net_pnl for t in self._closed[-10:])

                    council_decision = self._council.decide(
                        symbol=item.symbol,
                        kind=item.kind,
                        price=current.close,
                        signals=signals,
                        sentiment=sentiment,
                        market_mood=self._market_mood,
                        candle_data=candle_data,
                        equity=equity,
                        goal=self.config.goal_equity,
                        open_positions=len(self._positions),
                        max_positions=self.config.max_concurrent_positions,
                        win_rate=win_rate,
                        recent_pnl=recent_pnl,
                    )
                    ai_calls += 3  # 3 agents called
                    action = council_decision.final_action
                    confidence = council_decision.final_confidence
                    rationale = council_decision.rationale_summary
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.debug("Council failed for {} candle {}: {}", item.symbol, i, exc)
                    continue

            # ── SINGLE AGENT MODE (fallback) ──
            else:
                prompt = self._build_backtest_prompt(item, history_window, current, signals)
                try:
                    inference = self.ollama.generate_with_fallback(self._ranked_models, prompt)
                    ai_calls += 1
                    decision = self._parse_decision(inference.response)
                    action = str(decision.get("action", "AVOID")).upper()
                    confidence = float(decision.get("confidence", 0.0))
                    rationale = str(decision.get("rationale", ""))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.debug("AI inference failed for {} candle {}: {}", item.symbol, i, exc)
                    continue

            # act on decision
            if action == "BUY" and confidence >= self.config.min_confidence:
                self._open_position(item, current, i, confidence, rationale, signals)

        return len(candles) - window, ai_calls

    # ── AI prompt ─────────────────────────────────────────────────

    def _build_backtest_prompt(
        self,
        item: WatchlistItem,
        window: list[Candle],
        current: Candle,
        signals: TechnicalSignals | None = None,
    ) -> str:
        sig = signals.to_prompt_dict() if signals else {}

        # compact recent candles (last 6 only — indicators carry the rest)
        candle_data = [
            {
                "o": round(c.open, 6),
                "h": round(c.high, 6),
                "l": round(c.low, 6),
                "c": round(c.close, 6),
                "v": round(c.volume, 2),
            }
            for c in window[-6:]
        ]

        # goal context
        equity = self._cash + self._open_notional()
        goal_pct = ((self.config.goal_equity - equity) / equity * 100) if equity > 0 else 100

        return (
            "You are an expert short-term trader. Your goal is growing a portfolio "
            f"from ${self.config.starting_capital:.0f} to ${self.config.goal_equity:.0f}. "
            f"Current equity: ${equity:.2f} ({goal_pct:+.1f}% to goal). "
            "Return ONLY strict JSON: {trend_score: 0-100, confidence: 0.0-1.0, "
            "action: BUY|HOLD|AVOID, rationale: string, risk_flags: []}. "
            f"Asset: {item.symbol} ({item.kind}). "
            f"Price: {current.close:.6f}. "
            f"Technical signals: {json.dumps(sig)}. "
            f"Recent candles: {json.dumps(candle_data)}. "
            "RULES: Only BUY when technicals show strong momentum or oversold reversal. "
            "Set confidence > 0.7 only when multiple indicators align (RSI + trend + volume). "
            "AVOID when signal_score is low or trend is down without reversal signs. "
            "Risk:reward must be at least 1.5:1."
        )

    @staticmethod
    def _parse_decision(response: dict) -> dict:
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
        payload.setdefault("rationale", "")
        payload.setdefault("risk_flags", [])
        return payload

    # ── position management ───────────────────────────────────────

    def _open_position(
        self,
        item: WatchlistItem,
        candle: Candle,
        candle_idx: int,
        confidence: float,
        rationale: str,
        signals: TechnicalSignals | None = None,
    ) -> None:
        price = candle.close
        if price <= 0:
            return

        # SMART SIZING: scale position with confidence & signal strength
        # Base risk budget = 5% of cash
        base_risk = self._cash * self.config.max_risk_per_trade_pct
        equity = self._cash + self._open_notional()
        position_cap = equity * self.config.max_position_pct

        # Confidence multiplier: 0.6 conf → 0.5x, 0.8 conf → 1.5x, 1.0 → 2x
        conf_mult = max(0.5, min(2.0, (confidence - 0.5) * 4))
        # Signal score multiplier: score 10 → 0.5x, score 50 → 1.5x, score 80+ → 2x
        sig_score = signals.signal_score if signals else 20
        sig_mult = max(0.5, min(2.0, sig_score / 40))

        notional = min(base_risk * conf_mult * sig_mult, position_cap)
        if notional <= 0:
            return

        # apply slippage (buying at slightly worse price)
        fill_price = price * (1 + self.config.slippage_bps / 10_000)
        quantity = notional / fill_price
        cost = quantity * fill_price
        if cost > self._cash:
            quantity = self._cash / fill_price
            cost = quantity * fill_price

        if quantity <= 0 or cost <= 0:
            return

        self._cash -= cost
        self._positions.append(
            VirtualPosition(
                symbol=item.symbol,
                kind=item.kind,
                entry_price=fill_price,
                quantity=quantity,
                entry_candle_idx=candle_idx,
                entry_timestamp=candle.timestamp,
                confidence=confidence,
                rationale=rationale,
                peak_price=fill_price,
            )
        )
        logger.debug(
            "BT OPEN {} qty={:.6f} @ {:.6f} cost=${:.2f} cash=${:.2f} conf={:.0%} sig={}",
            item.symbol, quantity, fill_price, cost, self._cash, confidence, sig_score,
        )

    def _try_close_positions(self, item: WatchlistItem, candle: Candle, candle_idx: int) -> None:
        remaining: list[VirtualPosition] = []
        for pos in self._positions:
            if pos.symbol != item.symbol:
                remaining.append(pos)
                continue

            held = candle_idx - pos.entry_candle_idx
            current_price = candle.close
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100 if pos.entry_price > 0 else 0

            # Update trailing-stop peak
            if current_price > pos.peak_price:
                pos.peak_price = current_price
            trail_pct = ((pos.peak_price - current_price) / pos.peak_price) * 100 if pos.peak_price > 0 else 0

            close_reason = ""

            # PROFIT TARGET: take profit
            if pnl_pct >= self.config.profit_target_pct:
                close_reason = f"TP hit ({pnl_pct:+.1f}%)"
            # STOP LOSS: cut losses
            elif pnl_pct <= -self.config.stop_loss_pct:
                close_reason = f"SL hit ({pnl_pct:+.1f}%)"
            # TRAILING STOP: lock in gains
            elif held >= self.config.hold_candles_min and trail_pct >= self.config.trailing_stop_pct and pnl_pct > 0:
                close_reason = f"Trail stop ({trail_pct:.1f}% from peak)"
            # MAX HOLD: force close
            elif held >= self.config.hold_candles_max:
                close_reason = f"Max hold ({held} candles)"
            # HOLD: keep position
            elif held < self.config.hold_candles_min:
                remaining.append(pos)
                continue

            if close_reason:
                logger.debug("BT EXIT {}: {}", pos.symbol, close_reason)
                self._close_position(pos, candle, candle_idx)
            else:
                remaining.append(pos)

        self._positions = remaining

    def _close_position(self, pos: VirtualPosition, candle: Candle, candle_idx: int = 0) -> None:
        exit_price = candle.close * (1 - self.config.slippage_bps / 10_000)
        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        traded_notional = (pos.entry_price + exit_price) * pos.quantity
        fees = traded_notional * (self.config.fee_bps / 10_000)
        net_pnl = gross_pnl - fees
        proceeds = pos.quantity * exit_price
        self._cash += proceeds

        self._closed.append(
            ClosedTrade(
                symbol=pos.symbol,
                kind=pos.kind,
                side="buy",
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_pnl,
                hold_candles=max(0, candle_idx - pos.entry_candle_idx),
                entry_timestamp=pos.entry_timestamp,
                exit_timestamp=candle.timestamp,
                confidence=pos.confidence,
                rationale=pos.rationale,
            )
        )
        logger.debug(
            "BT CLOSE {} qty={:.6f} entry={:.6f} exit={:.6f} pnl=${:.4f}",
            pos.symbol, pos.quantity, pos.entry_price, exit_price, net_pnl,
        )

    def _force_close_all(self) -> None:
        """Close any remaining positions at their entry price (neutral exit)."""
        for pos in self._positions:
            # use entry price as exit → flat PnL minus fees
            dummy_candle = Candle(
                timestamp=datetime.now(timezone.utc).isoformat(),
                open=pos.entry_price,
                high=pos.entry_price,
                low=pos.entry_price,
                close=pos.entry_price,
                volume=0.0,
            )
            self._close_position(pos, dummy_candle)
        self._positions.clear()

    def _open_notional(self) -> float:
        return sum(p.entry_price * p.quantity for p in self._positions)

    def _get_cached_sentiment(self, symbol: str, kind: str) -> MarketSentiment | None:
        """Get sentiment for a symbol, caching results."""
        if symbol in self._sentiment_cache:
            return self._sentiment_cache[symbol]
        if not self._sentiment_analyzer:
            return None
        try:
            sentiment = self._sentiment_analyzer.get_sentiment(symbol, kind)
            self._sentiment_cache[symbol] = sentiment
            return sentiment
        except Exception:
            self._sentiment_cache[symbol] = None
            return None

    # ── result compilation ────────────────────────────────────────

    def _compile_result(
        self,
        symbols_tested: list[str],
        candles_processed: int,
        ai_calls: int,
        duration: float,
        errors: list[str],
    ) -> BacktestResult:
        trades = self._closed
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        total_pnl = sum(t.net_pnl for t in trades)
        final_equity = self.config.starting_capital + total_pnl

        # max drawdown on equity curve
        peak = self.config.starting_capital
        max_dd = 0.0
        running = self.config.starting_capital
        for t in trades:
            running += t.net_pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # simple sharpe estimate (pnl per trade / std of pnl)
        if len(trades) >= 2:
            pnls = [t.net_pnl for t in trades]
            mean_pnl = sum(pnls) / len(pnls)
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = variance ** 0.5
            sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            run_id=self.config.run_id,
            config=self.config,
            trades=trades,
            final_equity=round(final_equity, 2),
            total_pnl=round(total_pnl, 2),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            max_drawdown_pct=round(max_dd, 4),
            sharpe_estimate=round(sharpe, 4),
            symbols_tested=symbols_tested,
            candles_processed=candles_processed,
            ai_calls=ai_calls,
            duration_seconds=round(duration, 1),
            errors=errors,
        )

    def _error_result(self, msg: str, duration: float) -> BacktestResult:
        logger.error("Backtest aborted: {}", msg)
        return BacktestResult(
            run_id=self.config.run_id,
            config=self.config,
            errors=[msg],
            duration_seconds=round(duration, 1),
        )

    # ── persistence ───────────────────────────────────────────────

    def _persist_results(self, result: BacktestResult) -> None:
        """Save backtest results to SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO backtest_runs (
                        run_id, config_json, started_at,
                        starting_capital, final_equity, total_pnl,
                        win_count, loss_count, win_rate,
                        max_drawdown_pct, sharpe_estimate,
                        symbols_tested, candles_processed, ai_calls,
                        duration_seconds, errors_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.run_id,
                        json.dumps({
                            "starting_capital": result.config.starting_capital,
                            "max_risk_per_trade_pct": result.config.max_risk_per_trade_pct,
                            "max_position_pct": result.config.max_position_pct,
                            "max_concurrent_positions": result.config.max_concurrent_positions,
                            "min_confidence": result.config.min_confidence,
                            "slippage_bps": result.config.slippage_bps,
                            "fee_bps": result.config.fee_bps,
                            "lookback_window": result.config.lookback_window,
                            "hold_candles_min": result.config.hold_candles_min,
                            "hold_candles_max": result.config.hold_candles_max,
                            "profit_target_pct": result.config.profit_target_pct,
                            "stop_loss_pct": result.config.stop_loss_pct,
                            "trailing_stop_pct": result.config.trailing_stop_pct,
                            "signal_score_gate": result.config.signal_score_gate,
                            "goal_equity": result.config.goal_equity,
                            "use_ai": result.config.use_ai,
                            "period": result.config.period,
                            "interval": result.config.interval,
                            "kinds": result.config.kinds,
                            "symbols": result.config.symbols,
                        }),
                        now,
                        result.config.starting_capital,
                        result.final_equity,
                        result.total_pnl,
                        result.win_count,
                        result.loss_count,
                        result.win_rate,
                        result.max_drawdown_pct,
                        result.sharpe_estimate,
                        json.dumps(result.symbols_tested),
                        result.candles_processed,
                        result.ai_calls,
                        result.duration_seconds,
                        json.dumps(result.errors),
                    ),
                )

                for trade in result.trades:
                    conn.execute(
                        """
                        INSERT INTO backtest_trades (
                            run_id, symbol, kind, side,
                            entry_price, exit_price, quantity,
                            gross_pnl, fees, net_pnl,
                            hold_candles, entry_timestamp, exit_timestamp,
                            confidence, rationale
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result.run_id,
                            trade.symbol,
                            trade.kind,
                            trade.side,
                            trade.entry_price,
                            trade.exit_price,
                            trade.quantity,
                            trade.gross_pnl,
                            trade.fees,
                            trade.net_pnl,
                            trade.hold_candles,
                            trade.entry_timestamp,
                            trade.exit_timestamp,
                            trade.confidence,
                            trade.rationale,
                        ),
                    )

                conn.commit()
                logger.info("Backtest {} persisted: {} trades", result.run_id, len(result.trades))
        except Exception as exc:
            logger.error("Failed to persist backtest results: {}", exc)
