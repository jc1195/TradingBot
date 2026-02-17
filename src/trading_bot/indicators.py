"""Technical analysis indicators computed from OHLCV candle data.

These give the AI *actual trading signals* instead of raw price data,
dramatically improving decision quality.  All functions work on lists
of Candle objects from historical_data.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .historical_data import Candle


@dataclass
class TechnicalSignals:
    """Computed indicators for a candle window — fed directly to AI."""
    # Trend
    sma_fast: float          # 7-period simple moving average
    sma_slow: float          # 21-period simple moving average
    sma_crossover: str       # "bullish" / "bearish" / "neutral"
    ema_9: float             # 9-period exponential moving average
    price_vs_sma_fast: float # % above/below fast SMA
    price_vs_sma_slow: float # % above/below slow SMA
    trend_direction: str     # "up" / "down" / "sideways"

    # Momentum
    rsi_14: float            # relative strength index (0-100)
    rsi_signal: str          # "oversold" / "overbought" / "neutral"
    momentum_5: float        # 5-period rate of change %
    momentum_10: float       # 10-period rate of change %

    # Volatility
    atr_14: float            # average true range (14-period)
    atr_pct: float           # ATR as % of price
    bollinger_upper: float
    bollinger_lower: float
    bollinger_position: float  # 0-1, where price sits in band

    # Volume
    volume_sma_10: float
    volume_ratio: float      # current vol / avg vol
    volume_trend: str        # "surging" / "above_avg" / "below_avg" / "dry"

    # Support / Resistance
    recent_high: float       # highest close in window
    recent_low: float        # lowest close in window
    support_distance_pct: float   # % above recent low
    resistance_distance_pct: float  # % below recent high

    # Pattern signals
    consecutive_green: int   # consecutive up-close candles
    consecutive_red: int     # consecutive down-close candles
    candle_body_ratio: float # avg body/range ratio (conviction)

    # Summary
    signal_score: int        # composite -100 to +100

    def to_prompt_dict(self) -> dict:
        """Compact dict for AI prompt injection."""
        return {
            "trend": self.trend_direction,
            "sma_cross": self.sma_crossover,
            "price_vs_sma7": f"{self.price_vs_sma_fast:+.2f}%",
            "price_vs_sma21": f"{self.price_vs_sma_slow:+.2f}%",
            "rsi": round(self.rsi_14, 1),
            "rsi_signal": self.rsi_signal,
            "momentum_5d": f"{self.momentum_5:+.2f}%",
            "momentum_10d": f"{self.momentum_10:+.2f}%",
            "atr_pct": f"{self.atr_pct:.2f}%",
            "bollinger_pos": f"{self.bollinger_position:.0%}",
            "vol_ratio": f"{self.volume_ratio:.1f}x",
            "vol_trend": self.volume_trend,
            "support_dist": f"{self.support_distance_pct:.1f}%",
            "resist_dist": f"{self.resistance_distance_pct:.1f}%",
            "consec_green": self.consecutive_green,
            "consec_red": self.consecutive_red,
            "signal_score": self.signal_score,
        }


# ── computation ───────────────────────────────────────────────────

def compute_signals(candles: Sequence[Candle], current: Candle) -> TechnicalSignals:
    """Compute technical signals from a candle window + current bar."""
    closes = [c.close for c in candles] + [current.close]
    highs = [c.high for c in candles] + [current.high]
    lows = [c.low for c in candles] + [current.low]
    volumes = [c.volume for c in candles] + [current.volume]
    opens = [c.open for c in candles] + [current.open]

    price = current.close

    # ── SMAs ──
    sma_fast = _sma(closes, 7)
    sma_slow = _sma(closes, 21)
    ema_9 = _ema(closes, 9)

    prev_sma_fast = _sma(closes[:-1], 7) if len(closes) > 7 else sma_fast
    prev_sma_slow = _sma(closes[:-1], 21) if len(closes) > 21 else sma_slow

    if sma_fast > sma_slow and prev_sma_fast <= prev_sma_slow:
        sma_crossover = "bullish"
    elif sma_fast < sma_slow and prev_sma_fast >= prev_sma_slow:
        sma_crossover = "bearish"
    else:
        sma_crossover = "bullish" if sma_fast > sma_slow else "bearish" if sma_fast < sma_slow else "neutral"

    price_vs_fast = ((price - sma_fast) / sma_fast * 100) if sma_fast > 0 else 0
    price_vs_slow = ((price - sma_slow) / sma_slow * 100) if sma_slow > 0 else 0

    # trend direction
    if price > sma_fast > sma_slow:
        trend = "up"
    elif price < sma_fast < sma_slow:
        trend = "down"
    else:
        trend = "sideways"

    # ── RSI ──
    rsi = _rsi(closes, 14)
    if rsi < 30:
        rsi_signal = "oversold"
    elif rsi > 70:
        rsi_signal = "overbought"
    else:
        rsi_signal = "neutral"

    # ── Momentum ──
    mom5 = _momentum_pct(closes, 5)
    mom10 = _momentum_pct(closes, 10)

    # ── ATR ──
    atr = _atr(highs, lows, closes, 14)
    atr_pct = (atr / price * 100) if price > 0 else 0

    # ── Bollinger Bands ──
    bb_sma = _sma(closes, 20)
    bb_std = _std(closes, 20)
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_position = ((price - bb_lower) / bb_range) if bb_range > 0 else 0.5

    # ── Volume ──
    vol_sma = _sma(volumes, 10)
    vol_current = volumes[-1] if volumes else 0
    vol_ratio = (vol_current / vol_sma) if vol_sma > 0 else 1.0

    if vol_ratio >= 2.5:
        vol_trend = "surging"
    elif vol_ratio >= 1.3:
        vol_trend = "above_avg"
    elif vol_ratio >= 0.7:
        vol_trend = "below_avg"
    else:
        vol_trend = "dry"

    # ── Support / Resistance ──
    recent_high = max(closes[-min(20, len(closes)):])
    recent_low = min(closes[-min(20, len(closes)):])
    support_dist = ((price - recent_low) / recent_low * 100) if recent_low > 0 else 0
    resist_dist = ((recent_high - price) / price * 100) if price > 0 else 0

    # ── Patterns ──
    consec_green = 0
    consec_red = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if consec_red > 0:
                break
            consec_green += 1
        elif closes[i] < closes[i - 1]:
            if consec_green > 0:
                break
            consec_red += 1
        else:
            break

    # candle body ratio (conviction)
    body_ratios = []
    for i in range(max(0, len(opens) - 10), len(opens)):
        rng = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        if rng > 0:
            body_ratios.append(body / rng)
    avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0.5

    # ── Composite signal score (-100 to +100) ──
    score = 0

    # RSI contribution
    if rsi < 25:
        score += 25  # deeply oversold → buy signal
    elif rsi < 35:
        score += 15
    elif rsi > 75:
        score -= 25  # deeply overbought → avoid
    elif rsi > 65:
        score -= 10

    # Trend contribution
    if trend == "up":
        score += 20
    elif trend == "down":
        score -= 15

    # SMA crossover
    if sma_crossover == "bullish":
        score += 15
    elif sma_crossover == "bearish":
        score -= 15

    # Momentum
    if mom5 > 5:
        score += 10
    elif mom5 < -5:
        score -= 10

    # Volume
    if vol_trend == "surging":
        score += 10
    elif vol_trend == "dry":
        score -= 5

    # Bollinger position
    if bb_position < 0.15:
        score += 15  # near lower band → potential bounce
    elif bb_position > 0.85:
        score -= 10  # near upper band → potential rejection

    # Consecutive moves
    if consec_red >= 4:
        score += 10  # potential reversal after selloff
    elif consec_green >= 5:
        score -= 5   # extended, might pull back

    score = max(-100, min(100, score))

    return TechnicalSignals(
        sma_fast=round(sma_fast, 6),
        sma_slow=round(sma_slow, 6),
        sma_crossover=sma_crossover,
        ema_9=round(ema_9, 6),
        price_vs_sma_fast=round(price_vs_fast, 2),
        price_vs_sma_slow=round(price_vs_slow, 2),
        trend_direction=trend,
        rsi_14=round(rsi, 2),
        rsi_signal=rsi_signal,
        momentum_5=round(mom5, 2),
        momentum_10=round(mom10, 2),
        atr_14=round(atr, 6),
        atr_pct=round(atr_pct, 2),
        bollinger_upper=round(bb_upper, 6),
        bollinger_lower=round(bb_lower, 6),
        bollinger_position=round(bb_position, 4),
        volume_sma_10=round(vol_sma, 2),
        volume_ratio=round(vol_ratio, 2),
        volume_trend=vol_trend,
        recent_high=round(recent_high, 6),
        recent_low=round(recent_low, 6),
        support_distance_pct=round(support_dist, 2),
        resistance_distance_pct=round(resist_dist, 2),
        consecutive_green=consec_green,
        consecutive_red=consec_red,
        candle_body_ratio=round(avg_body_ratio, 4),
        signal_score=score,
    )


# ── helper math ───────────────────────────────────────────────────

def _sma(values: list[float], period: int) -> float:
    if len(values) < period:
        return sum(values) / len(values) if values else 0
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0
    k = 2 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _std(values: list[float], period: int) -> float:
    window = values[-period:] if len(values) >= period else values
    if len(window) < 2:
        return 0
    mean = sum(window) / len(window)
    variance = sum((v - mean) ** 2 for v in window) / len(window)
    return variance ** 0.5


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50  # neutral when insufficient data
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(highs) < 2:
        return 0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0
    return sum(trs[-period:]) / period


def _momentum_pct(values: list[float], period: int) -> float:
    if len(values) <= period:
        return 0
    old = values[-period - 1]
    if old == 0:
        return 0
    return ((values[-1] - old) / old) * 100
