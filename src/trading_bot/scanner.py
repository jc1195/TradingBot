"""Asset scanner — finds the BEST assets to trade right now.

Screens all watchlist items (or any custom list) and ranks them by
a composite "tradability" score based on:
 1. Technical momentum (from indicators.py)
 2. News sentiment (from sentiment.py)
 3. Trend strength (SMA alignment, consecutive green/red candles)
 4. Volume surge (above-average volume = institutional interest)
 5. Volatility (ATR % — we WANT volatility for day-trading)

The scanner outputs a ranked list: best assets to trade first.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from .historical_data import Candle, fetch_historical
from .indicators import TechnicalSignals, compute_signals
from .sentiment import MarketSentiment, SentimentAnalyzer
from .trend_detector import TrendAnalysis, analyze_trend, scan_for_bullish_opportunities
from .watchlist import WatchlistItem, get_watchlist


@dataclass
class AssetScore:
    """Composite tradability score for one asset."""
    symbol: str
    kind: str
    price: float
    score: int              # -100 to +100 composite
    technical_score: int    # from indicators.py signal_score
    sentiment_score: int    # from news sentiment
    trend_score: int        # SMA alignment + momentum
    volume_score: int       # volume surge indicator
    volatility_pct: float  # ATR as % of price (higher = more opportunity)
    recommendation: str     # "STRONG_BUY" / "BUY" / "WATCH" / "AVOID"
    reasons: list[str]
    signals: TechnicalSignals | None = None
    sentiment: MarketSentiment | None = None
    trend_analysis: TrendAnalysis | None = None
    is_penny: bool = False
    golden_cross: bool = False
    breakout_detected: bool = False


@dataclass
class ScanResult:
    """Full scan output with ranked assets."""
    timestamp: str
    assets_scanned: int
    top_picks: list[AssetScore]
    watchlist_only: list[AssetScore]  # sorted worst-to-best
    market_mood: dict
    scan_duration_seconds: float


class AssetScanner:
    """Screens watchlist assets and ranks by tradability."""

    def __init__(self, period: str = "6mo", interval: str = "1d") -> None:
        self.period = period
        self.interval = interval
        self.sentiment_analyzer = SentimentAnalyzer()

    def scan(
        self,
        kinds: list[str] | None = None,
        symbols: list[str] | None = None,
        top_n: int = 5,
    ) -> ScanResult:
        """Screen all watchlist assets, return top N picks."""
        t0 = time.time()
        items = get_watchlist(kinds=kinds, symbols=symbols)
        logger.info("Scanner: screening {} assets...", len(items))

        scores: list[AssetScore] = []
        market_mood = self.sentiment_analyzer.get_market_mood()

        for item in items:
            try:
                score = self._score_asset(item)
                scores.append(score)
                logger.debug(
                    "Scanner: {} score={} (tech={} sent={} trend={} vol={})",
                    item.symbol, score.score, score.technical_score,
                    score.sentiment_score, score.trend_score, score.volume_score,
                )
            except Exception as e:
                logger.warning("Scanner: failed to score {}: {}", item.symbol, e)

        # Sort by composite score (highest first)
        scores.sort(key=lambda s: s.score, reverse=True)

        # Top picks
        top_picks = [s for s in scores if s.recommendation in ("STRONG_BUY", "BUY")][:top_n]
        if len(top_picks) < top_n:
            # Add WATCH items to fill
            watch_items = [s for s in scores if s.recommendation == "WATCH" and s not in top_picks]
            top_picks.extend(watch_items[:top_n - len(top_picks)])

        elapsed = time.time() - t0

        return ScanResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            assets_scanned=len(items),
            top_picks=top_picks,
            watchlist_only=scores,
            market_mood=market_mood,
            scan_duration_seconds=round(elapsed, 1),
        )

    def _score_asset(self, item: WatchlistItem) -> AssetScore:
        """Compute composite tradability score for one asset."""

        # 1) Fetch historical data
        hist = fetch_historical(
            item=item,
            period=self.period,
            interval=self.interval,
        )
        candles = hist.candles
        if len(candles) < 30:
            return AssetScore(
                symbol=item.symbol, kind=item.kind, price=0,
                score=-100, technical_score=-50, sentiment_score=0,
                trend_score=-50, volume_score=0, volatility_pct=0,
                recommendation="AVOID",
                reasons=["Insufficient data"],
            )

        current = candles[-1]
        window = candles[-min(30, len(candles)):]

        # 2) Compute technical signals
        signals = compute_signals(window[:-1], current)

        # 2b) Run full trend analysis (bullish pattern detection)
        trend_analysis = analyze_trend(item, candles)

        # 3) Get news sentiment
        sentiment = None
        try:
            sentiment = self.sentiment_analyzer.get_sentiment(item.symbol, item.kind)
        except Exception as e:
            logger.debug("Sentiment fetch failed for {}: {}", item.symbol, e)

        # 4) Compute sub-scores
        tech_score = signals.signal_score  # -100 to +100

        # Sentiment score: -30 to +30
        sent_score = 0
        if sentiment:
            sent_score = int(sentiment.avg_sentiment * 30)

        # Trend score: -30 to +30 (now enhanced with pattern detection)
        trend_score = 0
        if signals.sma_fast > signals.sma_slow and signals.sma_slow > 0:
            trend_score += 15  # bullish crossover
        elif signals.sma_fast < signals.sma_slow:
            trend_score -= 15

        if signals.trend_direction == "up":
            trend_score += 10
        elif signals.trend_direction == "down":
            trend_score -= 10

        if signals.momentum_5 > 3:
            trend_score += 5
        elif signals.momentum_5 < -3:
            trend_score -= 5

        # Boost from trend detector bullish signals
        if trend_analysis.golden_cross:
            trend_score += 10
        if trend_analysis.breakout_detected:
            trend_score += 8
        if trend_analysis.higher_highs >= 3:
            trend_score += 5

        trend_score = max(-30, min(30, trend_score))

        # Volume score: -10 to +20
        vol_score = 0
        if signals.volume_ratio > 2.0:
            vol_score = 20  # surging
        elif signals.volume_ratio > 1.3:
            vol_score = 10  # above average
        elif signals.volume_ratio < 0.5:
            vol_score = -10  # dry / no interest

        # Composite: weight technicals most, then trend, then sentiment
        composite = int(
            tech_score * 0.35 +
            trend_score * 0.30 +
            sent_score * 0.20 +
            vol_score * 0.15
        )
        # Bonus for strong bullish pattern confluence
        if len(trend_analysis.bullish_signals) >= 3:
            composite += 10
        if trend_analysis.golden_cross:
            composite += 5
        composite = max(-100, min(100, composite))

        # Recommendation
        if composite >= 40:
            rec = "STRONG_BUY"
        elif composite >= 15:
            rec = "BUY"
        elif composite >= -10:
            rec = "WATCH"
        else:
            rec = "AVOID"

        # Reasons (enhanced with trend detector patterns)
        reasons: list[str] = []
        if trend_analysis.golden_cross:
            reasons.append("GOLDEN CROSS detected (SMA50 > SMA200)")
        if trend_analysis.breakout_detected:
            reasons.append("Breakout above 20-day resistance")
        for bs in trend_analysis.bullish_signals[:2]:
            reasons.append(f"{bs.pattern}: {bs.description}")
        if signals.trend_direction == "up":
            reasons.append("Uptrend confirmed")
        if signals.sma_fast > signals.sma_slow:
            reasons.append("Bullish SMA crossover")
        if signals.rsi_14 < 30:
            reasons.append("RSI oversold (reversal signal)")
        elif signals.rsi_14 > 70:
            reasons.append("RSI overbought (caution)")
        if signals.volume_ratio > 1.5:
            reasons.append(f"Volume surge ({signals.volume_ratio:.1f}x avg)")
        if signals.momentum_5 > 5:
            reasons.append(f"Strong momentum (+{signals.momentum_5:.1f}%)")
        if sentiment and sentiment.avg_sentiment > 0.2:
            reasons.append(f"Positive news ({sentiment.headline_count} headlines)")
        elif sentiment and sentiment.avg_sentiment < -0.2:
            reasons.append(f"Negative news sentiment")
        if trend_analysis.is_penny:
            reasons.append(f"Penny stock (${current.close:.2f})")
        if not reasons:
            reasons.append("Mixed signals")

        return AssetScore(
            symbol=item.symbol,
            kind=item.kind,
            price=current.close,
            score=composite,
            technical_score=tech_score,
            sentiment_score=sent_score,
            trend_score=trend_score,
            volume_score=vol_score,
            volatility_pct=signals.atr_pct,
            recommendation=rec,
            reasons=reasons,
            signals=signals,
            sentiment=sentiment,
            trend_analysis=trend_analysis,
            is_penny=trend_analysis.is_penny,
            golden_cross=trend_analysis.golden_cross,
            breakout_detected=trend_analysis.breakout_detected,
        )


def print_scan_report(result: ScanResult) -> None:
    """Pretty-print a scan result to console."""
    print("=" * 60)
    print(f"ASSET SCANNER REPORT — {result.timestamp}")
    print(f"Scanned {result.assets_scanned} assets in {result.scan_duration_seconds}s")
    if result.market_mood:
        fg = result.market_mood.get("fear_greed_index", "N/A")
        label = result.market_mood.get("label", "")
        print(f"Crypto Fear & Greed: {fg} ({label})")
    print("=" * 60)

    print(f"\nTOP PICKS ({len(result.top_picks)}):")
    for i, s in enumerate(result.top_picks, 1):
        tags = ""
        if s.is_penny:
            tags += " [PENNY]"
        if s.golden_cross:
            tags += " [GOLDEN CROSS]"
        if s.breakout_detected:
            tags += " [BREAKOUT]"
        print(f"  {i}. {s.symbol:6s} [{s.recommendation:11s}] Score: {s.score:+4d}  "
              f"Price: ${s.price:.4f}  Vol: {s.volatility_pct:.1f}%{tags}")
        for r in s.reasons[:4]:
            print(f"       → {r}")

    print(f"\nFULL RANKING:")
    for s in result.watchlist_only:
        tag = "+" if s.recommendation in ("STRONG_BUY", "BUY") else "~" if s.recommendation == "WATCH" else "-"
        print(f"  {tag} {s.symbol:6s} {s.score:+4d}  tech={s.technical_score:+3d}  "
              f"sent={s.sentiment_score:+3d}  trend={s.trend_score:+3d}  "
              f"vol={s.volume_score:+3d}  rec={s.recommendation}")
