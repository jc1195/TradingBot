"""3-Agent AI Trading Council — majority-vote decision system.

Three specialized AI agents analyze each trade opportunity independently:

  Agent 1: TECHNICIAN  — Pure chart/indicator analysis. Reads RSI, SMA, 
           Bollinger Bands, ATR, momentum, volume. Ignores news entirely.
  
  Agent 2: SENTINEL    — News & sentiment specialist. Reads headlines,
           market mood, fear/greed index, social volume. Ignores charts.
  
  Agent 3: STRATEGIST  — Risk/reward evaluator. Sees BOTH technicals
           and sentiment, but focuses on position sizing, R:R ratio,
           portfolio heat, and whether the setup fits the current goal.

DECISION RULE: 2-out-of-3 must vote BUY for the trade to execute.
The final confidence is the MEDIAN of all three agents' confidences.

Architecture:
  - All 3 agents run in PARALLEL using ThreadPoolExecutor (maxes GPU)
  - Each gets a different system prompt and data subset
  - Votes are tallied and logged for transparency
  - Council can use different Ollama models per agent if desired
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .indicators import TechnicalSignals
from .ollama_client import OllamaClient, InferenceResult
from .sentiment import MarketSentiment
from .settings import settings


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class AgentVote:
    """Single agent's decision."""
    agent_name: str
    role: str                   # "technician" / "sentinel" / "strategist"
    action: str                 # "BUY" / "HOLD" / "AVOID"
    confidence: float           # 0.0 - 1.0
    trend_score: int            # 0 - 100
    rationale: str
    risk_flags: list[str]
    latency_seconds: float
    model_used: str
    raw_response: dict = field(default_factory=dict)


@dataclass
class CouncilDecision:
    """Aggregated decision from the 3-agent council."""
    symbol: str
    final_action: str           # majority vote result
    final_confidence: float     # median confidence
    buy_votes: int              # how many voted BUY
    hold_votes: int
    avoid_votes: int
    votes: list[AgentVote]
    unanimous: bool             # all 3 agreed
    council_agreement: float    # 0.33, 0.67, or 1.0
    total_latency: float        # wall-clock time for all 3
    rationale_summary: str      # combined reasoning


# ── Agent prompts ────────────────────────────────────────────────

def _technician_prompt(
    symbol: str,
    kind: str,
    price: float,
    signals: dict,
    candle_data: list[dict],
    daytrade_profile: dict,
    exception_context: dict,
    equity: float,
    goal: float,
) -> str:
    return (
        "You are AGENT TECHNICIAN — a pure technical analysis expert. "
        "You ONLY look at chart indicators, price action, and volume patterns. "
        "Ignore all news, sentiment, and fundamentals. Your job is to read "
        "the mathematical signals and determine if this is a high-probability setup.\n\n"
        f"Asset: {symbol} ({kind}) | Price: ${price:.6f}\n"
        f"Portfolio: ${equity:.2f} (goal: ${goal:.0f})\n\n"
        f"TECHNICAL INDICATORS:\n{json.dumps(signals, indent=2)}\n\n"
        f"DAY-TRADE PROFILE:\n{json.dumps(daytrade_profile, indent=2)}\n\n"
        f"ADAPTIVE EXCEPTION CONTEXT (ADVISORY ONLY):\n{json.dumps(exception_context, indent=2)}\n\n"
        f"RECENT CANDLES (OHLCV):\n{json.dumps(candle_data)}\n\n"
        "ANALYSIS CHECKLIST:\n"
        "1. Is RSI in buy zone (30-50) with upward momentum?\n"
        "2. Are SMAs in bullish alignment (fast > slow)?\n"
        "3. Is price near Bollinger bottom (bounce) or breaking above mid?\n"
        "4. Is volume confirming the move (above average)?\n"
        "5. Is momentum positive and accelerating?\n"
        "6. Are there consecutive green candles forming?\n\n"
        "7. Is this volatile enough for day trading (movement + range + participation)?\n"
        "8. Treat adaptive exception context as informative only, never as a forced BUY signal.\n"
        "Return ONLY strict JSON: "
        '{"action": "BUY"|"HOLD"|"AVOID", "confidence": 0.0-1.0, '
        '"trend_score": 0-100, "rationale": "1-2 sentences", '
        '"risk_flags": ["flag1", ...]}'
    )


def _sentinel_prompt(
    symbol: str,
    kind: str,
    price: float,
    sentiment: dict,
    market_mood: dict,
    daytrade_profile: dict,
    exception_context: dict,
    equity: float,
    goal: float,
) -> str:
    return (
        "You are AGENT SENTINEL — a news and sentiment specialist. "
        "You ONLY analyze market sentiment, news headlines, social trends, "
        "and fear/greed indicators. Ignore all technical chart data. "
        "Your job is to gauge whether market mood supports a trade.\n\n"
        f"Asset: {symbol} ({kind}) | Price: ${price:.6f}\n"
        f"Portfolio: ${equity:.2f} (goal: ${goal:.0f})\n\n"
        f"NEWS & SENTIMENT DATA:\n{json.dumps(sentiment, indent=2)}\n\n"
        f"MARKET MOOD:\n{json.dumps(market_mood, indent=2)}\n\n"
        f"DAY-TRADE PROFILE:\n{json.dumps(daytrade_profile, indent=2)}\n\n"
        f"ADAPTIVE EXCEPTION CONTEXT (ADVISORY ONLY):\n{json.dumps(exception_context, indent=2)}\n\n"
        "ANALYSIS CHECKLIST:\n"
        "1. Is overall sentiment positive (> 0.2) or turning positive?\n"
        "2. Are there more bullish headlines than bearish?\n"
        "3. Is fear/greed index in 'Greed' territory (supportive of buys)?\n"
        "4. Are there any red-flag news items (hacks, lawsuits, bans)?\n"
        "5. Is there social buzz or institutional interest signals?\n\n"
        "6. Does market attention align with high day-trade movement potential?\n\n"
        "7. Treat adaptive exception context as informative only, never as a forced BUY signal.\n\n"
        "Return ONLY strict JSON: "
        '{"action": "BUY"|"HOLD"|"AVOID", "confidence": 0.0-1.0, '
        '"trend_score": 0-100, "rationale": "1-2 sentences", '
        '"risk_flags": ["flag1", ...]}'
    )


def _strategist_prompt(
    symbol: str,
    kind: str,
    price: float,
    signals: dict,
    sentiment: dict,
    market_mood: dict,
    equity: float,
    goal: float,
    open_positions: int,
    max_positions: int,
    win_rate: float,
    recent_pnl: float,
    daytrade_profile: dict,
    exception_context: dict,
) -> str:
    return (
        "You are AGENT STRATEGIST — a risk/reward and portfolio management expert. "
        "You see BOTH technical signals AND sentiment data. Your job is to "
        "evaluate whether the RISK/REWARD makes sense for this specific portfolio "
        "situation and current goal progress.\n\n"
        f"Asset: {symbol} ({kind}) | Price: ${price:.6f}\n"
        f"Portfolio: ${equity:.2f} (goal: ${goal:.0f})\n"
        f"Open positions: {open_positions}/{max_positions}\n"
        f"Recent win rate: {win_rate:.0%} | Recent PnL: ${recent_pnl:+.2f}\n\n"
        f"TECHNICAL SIGNALS:\n{json.dumps(signals, indent=2)}\n\n"
        f"SENTIMENT DATA:\n{json.dumps(sentiment, indent=2)}\n\n"
        f"MARKET MOOD:\n{json.dumps(market_mood, indent=2)}\n\n"
        f"DAY-TRADE PROFILE:\n{json.dumps(daytrade_profile, indent=2)}\n\n"
        f"ADAPTIVE EXCEPTION CONTEXT (ADVISORY ONLY):\n{json.dumps(exception_context, indent=2)}\n\n"
        "ANALYSIS CHECKLIST:\n"
        "1. Does the reward potential justify the risk (min 2:1 R:R)?\n"
        "2. Are technicals AND sentiment aligned (both bullish)?\n"
        "3. Can the portfolio absorb a loss on this position?\n"
        "4. Is the timing right (not chasing, not jumping into overhead resistance)?\n"
        "5. Does this fit the aggressive growth strategy for the goal?\n"
        "6. With current win rate, is this a high-probability setup?\n\n"
        "7. If day-trade profile is weak, avoid forcing a low-volatility trade.\n\n"
        "8. Treat adaptive exception context as informative only, never as a forced BUY signal.\n\n"
        "Return ONLY strict JSON: "
        '{"action": "BUY"|"HOLD"|"AVOID", "confidence": 0.0-1.0, '
        '"trend_score": 0-100, "rationale": "1-2 sentences", '
        '"risk_flags": ["flag1", ...]}'
    )


# ── Council engine ────────────────────────────────────────────────

class TradingCouncil:
    """3-agent parallel AI council for trade decisions.
    
    Runs all 3 agents concurrently on the GPU, collects votes,
    and returns a majority-vote decision.
    """

    def __init__(
        self,
        ollama: OllamaClient | None = None,
        ranked_models: list[str] | None = None,
    ) -> None:
        self.ollama = ollama or OllamaClient()
        self._ranked_models = ranked_models or []
        self._call_count = 0
        self._total_latency = 0.0

    def set_models(self, ranked_models: list[str]) -> None:
        """Update the ranked model list (call after model discovery)."""
        self._ranked_models = ranked_models

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency": round(self._total_latency, 1),
            "avg_latency": round(self._total_latency / max(self._call_count, 1), 1),
        }

    def decide(
        self,
        symbol: str,
        kind: str,
        price: float,
        signals: TechnicalSignals | None = None,
        sentiment: MarketSentiment | None = None,
        market_mood: dict | None = None,
        candle_data: list[dict] | None = None,
        equity: float = 500.0,
        goal: float = 1000.0,
        open_positions: int = 0,
        max_positions: int = 3,
        win_rate: float = 0.5,
        recent_pnl: float = 0.0,
        daytrade_profile: dict | None = None,
        exception_context: dict | None = None,
    ) -> CouncilDecision:
        """Run all 3 agents in parallel and aggregate votes."""
        t0 = time.time()
        
        sig_dict = signals.to_prompt_dict() if signals else {}
        sent_dict = self._sentiment_to_dict(sentiment)
        mood_dict = market_mood or {}
        candles = candle_data or []
        daytrade = daytrade_profile or {}
        exception_meta = exception_context or {}

        # Build prompts for all 3 agents
        prompts = {
            "technician": _technician_prompt(
                symbol, kind, price, sig_dict, candles, daytrade, exception_meta, equity, goal,
            ),
            "sentinel": _sentinel_prompt(
                symbol, kind, price, sent_dict, mood_dict, daytrade, exception_meta, equity, goal,
            ),
            "strategist": _strategist_prompt(
                symbol, kind, price, sig_dict, sent_dict, mood_dict,
                equity, goal, open_positions, max_positions, win_rate, recent_pnl, daytrade, exception_meta,
            ),
        }

        agent_names = {
            "technician": "TECHNICIAN",
            "sentinel": "SENTINEL",
            "strategist": "STRATEGIST",
        }

        # Run all 3 in parallel threads (Ollama handles GPU scheduling)
        votes: list[AgentVote] = []
        
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._run_agent, role, prompts[role]): role
                for role in prompts
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    vote = future.result()
                    vote.agent_name = agent_names[role]
                    vote.role = role
                    votes.append(vote)
                except Exception as exc:
                    logger.warning("Council agent {} failed: {}", role, exc)
                    # Failed agent votes AVOID (conservative)
                    votes.append(AgentVote(
                        agent_name=agent_names[role],
                        role=role,
                        action="AVOID",
                        confidence=0.0,
                        trend_score=0,
                        rationale=f"Agent failed: {exc}",
                        risk_flags=["agent_failure"],
                        latency_seconds=0.0,
                        model_used="none",
                    ))

        elapsed = time.time() - t0
        self._call_count += len(votes)
        self._total_latency += elapsed

        return self._tally_votes(symbol, votes, elapsed)

    def _run_agent(self, role: str, prompt: str) -> AgentVote:
        """Run a single agent's inference (fast path, no retry overhead)."""
        t0 = time.time()
        model = self._ranked_models[0] if self._ranked_models else "llama3.2:latest"
        inference = self.ollama.generate_fast(model, prompt, max_tokens=200)
        latency = time.time() - t0

        decision = self._parse_decision(inference.response)
        action = str(decision.get("action", "AVOID")).upper()
        if action not in ("BUY", "HOLD", "AVOID"):
            action = "AVOID"

        return AgentVote(
            agent_name="",  # filled by caller
            role=role,
            action=action,
            confidence=float(decision.get("confidence", 0.0)),
            trend_score=int(decision.get("trend_score", 0)),
            rationale=str(decision.get("rationale", "")),
            risk_flags=list(decision.get("risk_flags", [])),
            latency_seconds=round(latency, 2),
            model_used=inference.model_used,
            raw_response=decision,
        )

    def _tally_votes(
        self, symbol: str, votes: list[AgentVote], elapsed: float,
    ) -> CouncilDecision:
        """Count votes and determine majority action."""
        buy_count = sum(1 for v in votes if v.action == "BUY")
        hold_count = sum(1 for v in votes if v.action == "HOLD")
        avoid_count = sum(1 for v in votes if v.action == "AVOID")

        # Majority wins
        if buy_count >= 2:
            final_action = "BUY"
        elif avoid_count >= 2:
            final_action = "AVOID"
        else:
            final_action = "HOLD"

        # Confidence = median of all agents
        confidences = [v.confidence for v in votes]
        final_confidence = statistics.median(confidences) if confidences else 0.0

        # Agreement level
        actions = [v.action for v in votes]
        unanimous = len(set(actions)) == 1
        agreement = max(buy_count, hold_count, avoid_count) / max(len(votes), 1)

        # Combined rationale
        rationale_parts = []
        for v in votes:
            tag = "YES" if v.action == "BUY" else "WAIT" if v.action == "HOLD" else "NO"
            rationale_parts.append(
                f"[{v.agent_name} {tag} {v.confidence:.0%}] {v.rationale}"
            )
        rationale_summary = " | ".join(rationale_parts)

        logger.info(
            "Council {} | {} | BUY:{} HOLD:{} AVOID:{} | conf={:.0%} | {:.1f}s",
            symbol, final_action, buy_count, hold_count, avoid_count,
            final_confidence, elapsed,
        )
        for v in votes:
            logger.debug(
                "  {} → {} (conf={:.0%}, trend={}, {:.1f}s) {}",
                v.agent_name, v.action, v.confidence, v.trend_score,
                v.latency_seconds, v.rationale[:80],
            )

        return CouncilDecision(
            symbol=symbol,
            final_action=final_action,
            final_confidence=final_confidence,
            buy_votes=buy_count,
            hold_votes=hold_count,
            avoid_votes=avoid_count,
            votes=votes,
            unanimous=unanimous,
            council_agreement=agreement,
            total_latency=round(elapsed, 2),
            rationale_summary=rationale_summary,
        )

    @staticmethod
    def _parse_decision(raw: dict) -> dict:
        """Extract structured decision from Ollama response."""
        response_text = raw.get("response", "")
        if isinstance(response_text, dict):
            return response_text
        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            # Try to find JSON in the text
            text = str(response_text)
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"action": "AVOID", "confidence": 0.0, "rationale": text[:100]}

    @staticmethod
    def _sentiment_to_dict(sentiment: MarketSentiment | None) -> dict:
        if not sentiment:
            return {"status": "no_data", "avg_sentiment": 0.0}
        return {
            "symbol": sentiment.symbol,
            "headline_count": sentiment.headline_count,
            "avg_sentiment": round(sentiment.avg_sentiment, 3),
            "positive_count": sentiment.positive_count,
            "negative_count": sentiment.negative_count,
            "neutral_count": sentiment.neutral_count,
            "fear_greed_index": sentiment.fear_greed_index,
            "fear_greed_label": sentiment.fear_greed_label,
            "top_headlines": sentiment.top_headlines[:3],
            "overall": sentiment.sentiment_label,
        }
