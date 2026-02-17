"""Bullish trend detector — multi-timeframe trend analysis engine.

Scans any list of assets for bullish setups using multiple confirmation layers:

  LAYER 1: Trend structure  — Higher highs + higher lows over 20+ candles
  LAYER 2: Moving averages  — EMA9 > SMA20 > SMA50 (triple MA alignment)
  LAYER 3: Momentum         — RSI 40-65 (strong but not overbought)
  LAYER 4: Volume           — Increasing volume on up-moves
  LAYER 5: Breakout         — Price above key resistance levels
  LAYER 6: Candle patterns  — Bullish engulfing, hammer, morning star

Designed to answer: "What should I be trading RIGHT NOW?"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yfinance as yf
from loguru import logger

from .historical_data import Candle
from .indicators import TechnicalSignals, compute_signals
from .watchlist import WatchlistItem


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class TrendSignal:
    """Bullish/bearish trend analysis for one asset."""
    symbol: str
    kind: str
    price: float
    trend_type: str           # "strong_bull" / "bull" / "emerging_bull" / "neutral" / "bear"
    trend_strength: int       # 0-100
    trend_duration: int       # how many candles in current trend

    # Layer scores
    structure_score: int      # higher-high / higher-low pattern (0-20)
    ma_alignment_score: int   # triple MA alignment (0-20)
    momentum_score: int       # RSI + rate of change (0-20)
    volume_score: int         # volume confirmation (0-20)
    breakout_score: int       # resistance breakout (0-10)
    pattern_score: int        # candle patterns (0-10)

    # Key levels
    support_level: float
    resistance_level: float
    sma_20: float
    sma_50: float
    ema_9: float

    # Signals
    golden_cross: bool        # SMA20 just crossed above SMA50
    death_cross: bool         # SMA20 just crossed below SMA50 (warning)
    triple_ma_bullish: bool   # EMA9 > SMA20 > SMA50
    volume_confirming: bool   # volume rises on up days
    rsi_sweet_spot: bool      # RSI 35-65 (room to run)

    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TrendScanResult:
    """Full trend scan output."""
    timestamp: str
    assets_scanned: int
    bullish_count: int
    strong_bulls: list[TrendSignal]
    emerging_bulls: list[TrendSignal]
    all_signals: list[TrendSignal]
    scan_duration_seconds: float


# ── Trend Scanner ─────────────────────────────────────────────────

class BullishTrendScanner:
    """Multi-layer bullish trend detection engine."""

    def __init__(self, lookback_days: int = 90) -> None:
        self.lookback_days = lookback_days

    def scan_symbols(
        self,
        symbols: list[str],
        kind: str = "stock",
    ) -> TrendScanResult:
        """Scan a list of symbols for bullish trends."""
        t0 = time.time()
        signals: list[TrendSignal] = []

        logger.info("Trend scanner: analyzing {} symbols...", len(symbols))

        for symbol in symbols:
            try:
                sig = self._analyze_trend(symbol, kind)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug("Trend scan skip {}: {}", symbol, e)

        # Sort by trend strength
        signals.sort(key=lambda s: s.trend_strength, reverse=True)

        strong_bulls = [s for s in signals if s.trend_type == "strong_bull"]
        emerging = [s for s in signals if s.trend_type in ("bull", "emerging_bull")]

        elapsed = time.time() - t0

        logger.info(
            "Trend scan: {} analyzed, {} strong bulls, {} emerging in {:.1f}s",
            len(signals), len(strong_bulls), len(emerging), elapsed,
        )

        return TrendScanResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            assets_scanned=len(symbols),
            bullish_count=len(strong_bulls) + len(emerging),
            strong_bulls=strong_bulls,
            emerging_bulls=emerging,
            all_signals=signals,
            scan_duration_seconds=round(elapsed, 1),
        )

    def scan_watchlist(
        self,
        items: list[WatchlistItem],
    ) -> TrendScanResult:
        """Scan watchlist items for bullish trends."""
        t0 = time.time()
        signals: list[TrendSignal] = []

        logger.info("Trend scanner: analyzing {} watchlist items...", len(items))

        for item in items:
            try:
                sig = self._analyze_trend(item.yf_ticker, item.kind, item.symbol)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug("Trend scan skip {}: {}", item.symbol, e)

        signals.sort(key=lambda s: s.trend_strength, reverse=True)

        strong_bulls = [s for s in signals if s.trend_type == "strong_bull"]
        emerging = [s for s in signals if s.trend_type in ("bull", "emerging_bull")]

        elapsed = time.time() - t0

        return TrendScanResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            assets_scanned=len(items),
            bullish_count=len(strong_bulls) + len(emerging),
            strong_bulls=strong_bulls,
            emerging_bulls=emerging,
            all_signals=signals,
            scan_duration_seconds=round(elapsed, 1),
        )

    def _analyze_trend(
        self,
        yf_ticker: str,
        kind: str = "stock",
        display_symbol: str | None = None,
    ) -> TrendSignal | None:
        """Full 6-layer trend analysis for one asset."""
        symbol = display_symbol or yf_ticker

        # Download history
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period="6mo", interval="1d")

        if hist.empty or len(hist) < 30:
            return None

        closes = list(hist["Close"].values)
        highs = list(hist["High"].values)
        lows = list(hist["Low"].values)
        volumes = list(hist["Volume"].values)
        opens = list(hist["Open"].values)

        price = float(closes[-1])
        n = len(closes)

        # === LAYER 1: Trend Structure (0-20 pts) ===
        structure_score, trend_duration = self._check_structure(closes, highs, lows)

        # === LAYER 2: Moving Average Alignment (0-20 pts) ===
        ema_9 = self._ema(closes, 9)
        sma_20 = sum(closes[-20:]) / 20 if n >= 20 else price
        sma_50 = sum(closes[-50:]) / 50 if n >= 50 else sma_20

        ma_score = 0
        triple_ma = False
        golden_cross = False
        death_cross = False

        if ema_9 > sma_20 > sma_50:
            ma_score = 20
            triple_ma = True
        elif ema_9 > sma_20:
            ma_score = 12
        elif price > sma_20:
            ma_score = 8
        elif price > sma_50:
            ma_score = 4

        # Check for golden/death cross
        if n >= 55:
            prev_sma20 = sum(closes[-25:-5]) / 20
            prev_sma50 = sum(closes[-55:-5]) / 50
            if prev_sma20 <= prev_sma50 and sma_20 > sma_50:
                golden_cross = True
                ma_score = min(20, ma_score + 5)
            elif prev_sma20 >= prev_sma50 and sma_20 < sma_50:
                death_cross = True
                ma_score = max(0, ma_score - 10)

        # === LAYER 3: Momentum (0-20 pts) ===
        rsi = self._calc_rsi(closes, 14)
        mom_5 = ((price - closes[-6]) / closes[-6] * 100) if n >= 6 else 0
        mom_10 = ((price - closes[-11]) / closes[-11] * 100) if n >= 11 else 0

        momentum_score = 0
        rsi_sweet = False

        if 35 <= rsi <= 65:
            momentum_score += 10
            rsi_sweet = True
        elif 25 <= rsi < 35:
            momentum_score += 8  # oversold, reversal potential
        elif rsi > 65:
            momentum_score += 4  # strong but risky

        if mom_5 > 3:
            momentum_score += 5
        elif mom_5 > 0:
            momentum_score += 2

        if mom_10 > 5:
            momentum_score += 5
        elif mom_10 > 0:
            momentum_score += 3

        momentum_score = min(20, momentum_score)

        # === LAYER 4: Volume Confirmation (0-20 pts) ===
        vol_score, vol_confirming = self._check_volume(closes, volumes)

        # === LAYER 5: Breakout Detection (0-10 pts) ===
        breakout_score = 0
        resistance = max(closes[-20:]) if n >= 20 else price
        support = min(closes[-20:]) if n >= 20 else price

        if price >= resistance * 0.98:
            breakout_score = 10  # at or above resistance
        elif price >= resistance * 0.95:
            breakout_score = 5   # approaching resistance

        # === LAYER 6: Candle Patterns (0-10 pts) ===
        pattern_score = self._check_patterns(opens, closes, highs, lows)

        # === COMPOSITE ===
        total = structure_score + ma_score + momentum_score + vol_score + breakout_score + pattern_score

        # Classify trend type
        if total >= 70:
            trend_type = "strong_bull"
        elif total >= 50:
            trend_type = "bull"
        elif total >= 35:
            trend_type = "emerging_bull"
        elif total >= 20:
            trend_type = "neutral"
        else:
            trend_type = "bear"

        # Build reasons
        reasons: list[str] = []
        warnings: list[str] = []

        if triple_ma:
            reasons.append("Triple MA bullish (EMA9 > SMA20 > SMA50)")
        if golden_cross:
            reasons.append("GOLDEN CROSS detected!")
        if rsi_sweet:
            reasons.append(f"RSI in sweet spot ({rsi:.0f})")
        if vol_confirming:
            reasons.append("Volume confirms uptrend")
        if breakout_score >= 8:
            reasons.append(f"Breakout above ${resistance:.2f}")
        if mom_5 > 3:
            reasons.append(f"Strong 5d momentum (+{mom_5:.1f}%)")
        if structure_score >= 15:
            reasons.append(f"Higher-highs pattern ({trend_duration}d)")

        if death_cross:
            warnings.append("DEATH CROSS — bearish signal")
        if rsi > 70:
            warnings.append(f"RSI overbought ({rsi:.0f})")
        if not vol_confirming:
            warnings.append("Volume not confirming")

        return TrendSignal(
            symbol=symbol,
            kind=kind,
            price=price,
            trend_type=trend_type,
            trend_strength=total,
            trend_duration=trend_duration,
            structure_score=structure_score,
            ma_alignment_score=ma_score,
            momentum_score=momentum_score,
            volume_score=vol_score,
            breakout_score=breakout_score,
            pattern_score=pattern_score,
            support_level=round(support, 4),
            resistance_level=round(resistance, 4),
            sma_20=round(sma_20, 4),
            sma_50=round(sma_50, 4),
            ema_9=round(ema_9, 4),
            golden_cross=golden_cross,
            death_cross=death_cross,
            triple_ma_bullish=triple_ma,
            volume_confirming=vol_confirming,
            rsi_sweet_spot=rsi_sweet,
            reasons=reasons,
            warnings=warnings,
        )

    def _check_structure(
        self, closes: list[float], highs: list[float], lows: list[float],
    ) -> tuple[int, int]:
        """Check for higher-highs & higher-lows structure. Returns (score, duration)."""
        n = len(closes)
        if n < 10:
            return 0, 0

        # Check last 20 candles in 5-candle windows
        check_len = min(n, 40)
        window = 5
        hh_count = 0
        hl_count = 0
        duration = 0

        for i in range(check_len - window, window - 1, -window):
            if i < window:
                break
            curr_high = max(highs[i:i + window])
            curr_low = min(lows[i:i + window])
            prev_high = max(highs[i - window:i])
            prev_low = min(lows[i - window:i])

            if curr_high > prev_high:
                hh_count += 1
            if curr_low > prev_low:
                hl_count += 1
            if curr_high > prev_high and curr_low > prev_low:
                duration += window

        score = 0
        if hh_count >= 3 and hl_count >= 3:
            score = 20  # strong uptrend structure
        elif hh_count >= 2 and hl_count >= 2:
            score = 15
        elif hh_count >= 2 or hl_count >= 2:
            score = 10
        elif hh_count >= 1 and hl_count >= 1:
            score = 5

        return score, duration

    def _check_volume(
        self, closes: list[float], volumes: list[float],
    ) -> tuple[int, bool]:
        """Check if volume confirms uptrend. Returns (score, confirming)."""
        n = len(closes)
        if n < 20:
            return 0, False

        # Compare up-day volume vs down-day volume (last 20 days)
        up_vol = []
        down_vol = []
        for i in range(-20, 0):
            if closes[i] > closes[i - 1]:
                up_vol.append(volumes[i])
            else:
                down_vol.append(volumes[i])

        avg_up = sum(up_vol) / len(up_vol) if up_vol else 0
        avg_down = sum(down_vol) / len(down_vol) if down_vol else 0

        score = 0
        confirming = False

        if avg_up > avg_down * 1.5:
            score = 20
            confirming = True
        elif avg_up > avg_down * 1.2:
            score = 15
            confirming = True
        elif avg_up > avg_down:
            score = 10
            confirming = True
        elif avg_up > avg_down * 0.8:
            score = 5

        # Recent volume surge bonus
        recent_avg = sum(volumes[-5:]) / 5
        overall_avg = sum(volumes[-20:]) / 20
        if recent_avg > overall_avg * 2:
            score = min(20, score + 5)

        return score, confirming

    def _check_patterns(
        self,
        opens: list[float],
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> int:
        """Check for bullish candle patterns. Returns score 0-10."""
        n = len(closes)
        if n < 3:
            return 0

        score = 0

        # Check last 3 candles for patterns
        for i in range(-3, 0):
            o, c, h, l = opens[i], closes[i], highs[i], lows[i]
            body = abs(c - o)
            full_range = h - l if h > l else 0.001

            # Bullish engulfing
            if i > -n and c > o:  # current is green
                prev_o, prev_c = opens[i - 1], closes[i - 1]
                if prev_c < prev_o and c > prev_o and o < prev_c:
                    score += 5  # bullish engulfing

            # Hammer (long lower wick, small body at top)
            if c > o:  # green candle
                lower_wick = o - l
                upper_wick = h - c
                if lower_wick > body * 2 and upper_wick < body * 0.5:
                    score += 3  # hammer

            # Strong green candle (conviction)
            if c > o and body / full_range > 0.7:
                score += 2

        return min(10, score)

    @staticmethod
    def _ema(data: list[float], period: int) -> float:
        """Exponential moving average."""
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        for val in data[period:]:
            ema = (val - ema) * multiplier + ema
        return ema

    @staticmethod
    def _calc_rsi(closes: list[float], period: int = 14) -> float:
        """Simple RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


def print_trend_report(result: TrendScanResult) -> None:
    """Pretty-print trend scan results."""
    print("=" * 70)
    print(f"BULLISH TREND SCANNER — {result.timestamp}")
    print(f"Scanned: {result.assets_scanned} | Bullish: {result.bullish_count} | {result.scan_duration_seconds}s")
    print("=" * 70)

    if result.strong_bulls:
        print(f"\n🟢 STRONG BULLS ({len(result.strong_bulls)}):")
        for s in result.strong_bulls:
            flags = []
            if s.golden_cross:
                flags.append("GoldenX")
            if s.triple_ma_bullish:
                flags.append("TripleMA")
            if s.volume_confirming:
                flags.append("VolConf")
            print(
                f"  {s.symbol:6s}  ${s.price:10.4f}  Strength:{s.trend_strength:3d}/100  "
                f"[{' '.join(flags)}]"
            )
            print(
                f"         Struct:{s.structure_score:2d} MA:{s.ma_alignment_score:2d} "
                f"Mom:{s.momentum_score:2d} Vol:{s.volume_score:2d} "
                f"Break:{s.breakout_score:2d} Pat:{s.pattern_score:2d}"
            )
            for r in s.reasons[:3]:
                print(f"         → {r}")

    if result.emerging_bulls:
        print(f"\n🟡 EMERGING BULLS ({len(result.emerging_bulls)}):")
        for s in result.emerging_bulls:
            print(
                f"  {s.symbol:6s}  ${s.price:10.4f}  Strength:{s.trend_strength:3d}/100  "
                f"Type: {s.trend_type}"
            )
            for r in s.reasons[:2]:
                print(f"         → {r}")

    # Neutral / bearish summary
    bears = [s for s in result.all_signals if s.trend_type in ("neutral", "bear")]
    if bears:
        print(f"\n⚪ NEUTRAL/BEAR ({len(bears)}):")
        for s in bears:
            warn = f"  ⚠️ {s.warnings[0]}" if s.warnings else ""
            print(f"  {s.symbol:6s}  ${s.price:10.4f}  Strength:{s.trend_strength:3d}  {s.trend_type}{warn}")
