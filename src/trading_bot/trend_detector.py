"""Bullish trend detector & penny stock screener.

Finds assets showing bullish patterns by scanning for:
 • Golden Cross (SMA50 crossing above SMA200)
 • Momentum breakouts (price breaking above resistance)
 • Volume-confirmed rallies
 • Oversold reversals (RSI bouncing from < 30)
 • Bullish engulfing / hammer candle patterns
 • Consecutive higher-highs and higher-lows

Works with both the static watchlist AND dynamically-discovered
penny stocks / trending tickers pulled from free screeners.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from loguru import logger

try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]

try:
    import robin_stocks.robinhood as rh
except Exception:
    rh = None  # type: ignore[assignment]

from .historical_data import Candle, fetch_historical
from .settings import settings
from .watchlist import WatchlistItem, DEFAULT_WATCHLIST


# ── Penny stock universe ─────────────────────────────────────────
# Robinhood-available penny stocks (< $5) known for volatility.
# This list is expanded at runtime by the dynamic screener.

PENNY_STOCKS: list[WatchlistItem] = [
    # Biotech / pharma (big swings on trial news)
    WatchlistItem("SNDL", "stock", "SNDL", "Sundial Growers – cannabis penny stock"),
    WatchlistItem("CLOV", "stock", "CLOV", "Clover Health – fintech/health penny"),
    WatchlistItem("BNGO", "stock", "BNGO", "Bionano Genomics – biotech penny"),
    WatchlistItem("OPEN", "stock", "OPEN", "Opendoor – real estate tech penny"),
    WatchlistItem("TLRY", "stock", "TLRY", "Tilray Brands – cannabis penny"),
    WatchlistItem("DNA",  "stock", "DNA",  "Ginkgo Bioworks – synthetic bio penny"),
    WatchlistItem("BBIG", "stock", "BBIG", "Vinco Ventures – digital media penny"),
    WatchlistItem("HIMS", "stock", "HIMS", "Hims & Hers – telehealth penny"),

    # Crypto-related stocks (move with BTC)
    WatchlistItem("BITF", "stock", "BITF", "Bitfarms – Bitcoin mining penny"),
    WatchlistItem("HUT",  "stock", "HUT",  "Hut 8 Mining – crypto mining penny"),
    WatchlistItem("BTBT", "stock", "BTBT", "Bit Digital – Bitcoin mining penny"),

    # Energy / mining (commodity swings)

    WatchlistItem("UUUU", "stock", "UUUU", "Energy Fuels – uranium penny"),

    # Tech / AI penny stocks
    WatchlistItem("GSAT", "stock", "GSAT", "Globalstar – satellite penny"),
    WatchlistItem("BB",   "stock", "BB",   "BlackBerry – cybersecurity penny"),
    WatchlistItem("NOK",  "stock", "NOK",  "Nokia – 5G / IoT penny"),

    # Meme-adjacent
    WatchlistItem("WKHS", "stock", "WKHS", "Workhorse Group – EV penny"),
]

# Additional small-cap crypto tokens
MICRO_CRYPTO: list[WatchlistItem] = [
    WatchlistItem("HBAR", "crypto", "HBAR-USD", "Hedera – enterprise DLT"),
    WatchlistItem("VET",  "crypto", "VET-USD",  "VeChain – supply chain crypto"),
    WatchlistItem("ALGO", "crypto", "ALGO-USD", "Algorand – fast L1 chain"),
    WatchlistItem("CRO",  "crypto", "CRO-USD",  "Cronos – exchange token"),
    WatchlistItem("NEAR", "crypto", "NEAR-USD", "Near Protocol – sharded L1"),
    WatchlistItem("ATOM", "crypto", "ATOM-USD", "Cosmos – interop ecosystem"),
    WatchlistItem("FIL",  "crypto", "FIL-USD",  "Filecoin – storage crypto"),
    WatchlistItem("INJ",  "crypto", "INJ-USD",  "Injective – DeFi protocol"),
    WatchlistItem("RENDER","crypto","RENDER-USD","Render – GPU compute token"),
]


# ── Bullish pattern detectors ────────────────────────────────────

@dataclass
class BullishSignal:
    """A single bullish signal detected in price data."""
    pattern: str         # name of the pattern
    strength: int        # 1-10 how strong the signal is
    description: str     # human-readable explanation
    timeframe: str       # "short" / "medium" / "long"


@dataclass
class TrendAnalysis:
    """Complete trend analysis for one asset."""
    symbol: str
    kind: str
    price: float
    bullish_signals: list[BullishSignal]
    bearish_signals: list[BullishSignal]
    trend_score: int              # -100 to +100
    trend_label: str              # "strong_bull" / "bull" / "neutral" / "bear" / "strong_bear"
    is_penny: bool                # price < $5
    volatility_rank: float        # 0-100 (higher = more volatile)
    avg_daily_move_pct: float     # average daily % move
    last_day_move_pct: float      # latest 1-day % move (signed)
    volume_vs_avg: float          # current vol / 20-day avg
    activity_score: float         # composite movement + participation score
    sma_50: float | None
    sma_200: float | None
    golden_cross: bool            # SMA50 just crossed above SMA200
    death_cross: bool             # SMA50 just crossed below SMA200
    higher_highs: int             # count of recent higher-highs
    higher_lows: int              # count of recent higher-lows
    breakout_detected: bool       # price broke above recent resistance


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _detect_golden_death_cross(closes: list[float]) -> tuple[bool, bool]:
    """Check if SMA50 just crossed SMA200 (golden) or vice versa (death)."""
    if len(closes) < 202:
        return False, False
    sma50_now = sum(closes[-50:]) / 50
    sma200_now = sum(closes[-200:]) / 200
    sma50_prev = sum(closes[-51:-1]) / 50
    sma200_prev = sum(closes[-201:-1]) / 200

    golden = sma50_prev <= sma200_prev and sma50_now > sma200_now
    death = sma50_prev >= sma200_prev and sma50_now < sma200_now
    return golden, death


def _count_higher_highs(candles: list[Candle], window: int = 10) -> int:
    """Count consecutive higher-highs in last N candles."""
    if len(candles) < window:
        return 0
    recent = candles[-window:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i].high > recent[i - 1].high:
            count += 1
        else:
            break
    return count


def _count_higher_lows(candles: list[Candle], window: int = 10) -> int:
    """Count consecutive higher-lows in last N candles."""
    if len(candles) < window:
        return 0
    recent = candles[-window:]
    count = 0
    for i in range(1, len(recent)):
        if recent[i].low > recent[i - 1].low:
            count += 1
        else:
            break
    return count


def _detect_breakout(candles: list[Candle], lookback: int = 20) -> bool:
    """Check if current price broke above the highest close in the lookback window."""
    if len(candles) < lookback + 1:
        return False
    resistance = max(c.high for c in candles[-(lookback + 1):-1])
    return candles[-1].close > resistance


def _avg_daily_move(candles: list[Candle], window: int = 20) -> float:
    """Average absolute daily % move — measures volatility."""
    if len(candles) < window + 1:
        return 0.0
    recent = candles[-window:]
    moves = []
    for i in range(1, len(recent)):
        if recent[i - 1].close > 0:
            pct = abs((recent[i].close - recent[i - 1].close) / recent[i - 1].close) * 100
            moves.append(pct)
    return sum(moves) / len(moves) if moves else 0.0


def _rsi(closes: list[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _detect_bullish_patterns(candles: list[Candle], closes: list[float]) -> list[BullishSignal]:
    """Detect all bullish patterns in the data."""
    signals: list[BullishSignal] = []

    if len(candles) < 5:
        return signals

    current = candles[-1]
    prev = candles[-2]

    # 1. Golden Cross
    golden, _ = _detect_golden_death_cross(closes)
    if golden:
        signals.append(BullishSignal(
            "golden_cross", 9,
            "SMA50 crossed above SMA200 — strong long-term bullish signal",
            "long",
        ))

    # 2. RSI oversold bounce
    rsi_now = _rsi(closes)
    rsi_prev = _rsi(closes[:-1]) if len(closes) > 15 else 50
    if rsi_prev < 30 and rsi_now > 30:
        signals.append(BullishSignal(
            "rsi_bounce", 7,
            f"RSI bounced from oversold ({rsi_prev:.0f} -> {rsi_now:.0f})",
            "short",
        ))
    elif rsi_now < 30:
        signals.append(BullishSignal(
            "rsi_oversold", 5,
            f"RSI at {rsi_now:.0f} — oversold, reversal potential",
            "short",
        ))

    # 3. Bullish engulfing candle
    if (prev.close < prev.open and  # prev was red
            current.close > current.open and  # current is green
            current.close > prev.open and  # engulfs prev body
            current.open < prev.close):
        signals.append(BullishSignal(
            "bullish_engulfing", 6,
            "Bullish engulfing pattern — buyers overwhelmed sellers",
            "short",
        ))

    # 4. Hammer candle (long lower wick, small body at top)
    body = abs(current.close - current.open)
    candle_range = current.high - current.low
    lower_wick = min(current.open, current.close) - current.low
    if candle_range > 0 and lower_wick > body * 2 and body < candle_range * 0.3:
        signals.append(BullishSignal(
            "hammer", 5,
            "Hammer candle — rejection of lower prices",
            "short",
        ))

    # 5. Volume spike with green candle
    if len(candles) >= 20:
        avg_vol = sum(c.volume for c in candles[-20:]) / 20
        if avg_vol > 0 and current.volume > avg_vol * 2 and current.close > current.open:
            signals.append(BullishSignal(
                "volume_breakout", 7,
                f"Volume spike ({current.volume / avg_vol:.1f}x avg) on green candle",
                "short",
            ))

    # 6. Breakout above resistance
    if _detect_breakout(candles, lookback=20):
        signals.append(BullishSignal(
            "resistance_breakout", 8,
            "Price broke above 20-day resistance — momentum entry",
            "medium",
        ))

    # 7. Higher highs + higher lows (uptrend structure)
    hh = _count_higher_highs(candles)
    hl = _count_higher_lows(candles)
    if hh >= 3 and hl >= 3:
        signals.append(BullishSignal(
            "uptrend_structure", 7,
            f"{hh} higher-highs and {hl} higher-lows — classic uptrend",
            "medium",
        ))
    elif hh >= 2 and hl >= 2:
        signals.append(BullishSignal(
            "developing_uptrend", 4,
            f"{hh} higher-highs, {hl} higher-lows — trend forming",
            "medium",
        ))

    # 8. SMA alignment (fast > slow — bullish stacking)
    sma7 = _sma(closes, 7)
    sma21 = _sma(closes, 21)
    sma50 = _sma(closes, 50)
    if sma7 and sma21 and sma50:
        if sma7 > sma21 > sma50 and closes[-1] > sma7:
            signals.append(BullishSignal(
                "sma_bull_stack", 8,
                "Price > SMA7 > SMA21 > SMA50 — perfect bullish alignment",
                "medium",
            ))
        elif sma7 > sma21 and closes[-1] > sma7:
            signals.append(BullishSignal(
                "sma_bullish", 5,
                "Price above fast & slow SMA — short-term bullish",
                "short",
            ))

    # 9. Consecutive green candles
    green_count = 0
    for c in reversed(candles[-10:]):
        if c.close > c.open:
            green_count += 1
        else:
            break
    if green_count >= 4:
        signals.append(BullishSignal(
            "green_streak", 6,
            f"{green_count} consecutive green candles — strong buying pressure",
            "short",
        ))

    # 10. Bollinger squeeze breakout (volatility expanding upward)
    if len(closes) >= 20:
        sma20 = sum(closes[-20:]) / 20
        std20 = (sum((c - sma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        upper_band = sma20 + 2 * std20
        if closes[-1] > upper_band and closes[-2] <= upper_band:
            signals.append(BullishSignal(
                "bollinger_breakout", 6,
                "Price broke above upper Bollinger Band — volatility expansion",
                "short",
            ))

    return signals


def _detect_bearish_patterns(candles: list[Candle], closes: list[float]) -> list[BullishSignal]:
    """Detect bearish warning signals."""
    signals: list[BullishSignal] = []

    if len(candles) < 5:
        return signals

    # 1. Death Cross
    _, death = _detect_golden_death_cross(closes)
    if death:
        signals.append(BullishSignal(
            "death_cross", 9,
            "SMA50 crossed below SMA200 — strong bearish signal",
            "long",
        ))

    # 2. RSI overbought
    rsi_now = _rsi(closes)
    if rsi_now > 80:
        signals.append(BullishSignal(
            "rsi_overbought", 6,
            f"RSI at {rsi_now:.0f} — heavily overbought, pullback risk",
            "short",
        ))

    # 3. SMA bearish alignment
    sma7 = _sma(closes, 7)
    sma21 = _sma(closes, 21)
    if sma7 and sma21 and sma7 < sma21 and closes[-1] < sma7:
        signals.append(BullishSignal(
            "sma_bearish", 6,
            "Price below all short-term SMAs — bearish",
            "short",
        ))

    # 4. Lower lows
    if len(candles) >= 5:
        recent = candles[-5:]
        lower_lows = sum(1 for i in range(1, len(recent)) if recent[i].low < recent[i - 1].low)
        if lower_lows >= 3:
            signals.append(BullishSignal(
                "lower_lows", 5,
                f"{lower_lows} consecutive lower-lows — downtrend structure",
                "short",
            ))

    return signals


# ── Full analysis function ───────────────────────────────────────

def analyze_trend(item: WatchlistItem, candles: list[Candle]) -> TrendAnalysis:
    """Full trend analysis for one asset."""
    if len(candles) < 10:
        return TrendAnalysis(
            symbol=item.symbol, kind=item.kind, price=0,
            bullish_signals=[], bearish_signals=[],
            trend_score=-50, trend_label="insufficient_data",
            is_penny=False, volatility_rank=0,
            avg_daily_move_pct=0, last_day_move_pct=0, volume_vs_avg=0, activity_score=0,
            sma_50=None, sma_200=None,
            golden_cross=False, death_cross=False,
            higher_highs=0, higher_lows=0,
            breakout_detected=False,
        )

    closes = [c.close for c in candles]
    current_price = closes[-1]

    # Detect all patterns
    bull_signals = _detect_bullish_patterns(candles, closes)
    bear_signals = _detect_bearish_patterns(candles, closes)

    # Compute trend score
    bull_score = sum(s.strength for s in bull_signals)
    bear_score = sum(s.strength for s in bear_signals)
    raw_score = bull_score - bear_score
    # Normalize to -100..+100
    trend_score = max(-100, min(100, raw_score * 5))

    # Label
    if trend_score >= 50:
        trend_label = "strong_bull"
    elif trend_score >= 20:
        trend_label = "bull"
    elif trend_score >= -20:
        trend_label = "neutral"
    elif trend_score >= -50:
        trend_label = "bear"
    else:
        trend_label = "strong_bear"

    # Supplementary data
    golden, death = _detect_golden_death_cross(closes)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    hh = _count_higher_highs(candles)
    hl = _count_higher_lows(candles)
    breakout = _detect_breakout(candles)
    avg_move = _avg_daily_move(candles)
    last_day_move = 0.0
    if len(closes) >= 2 and closes[-2] > 0:
        last_day_move = ((closes[-1] - closes[-2]) / closes[-2]) * 100.0

    vol_vs_avg = 1.0
    if len(candles) >= 20:
        avg_vol = sum(c.volume for c in candles[-20:]) / 20
        if avg_vol > 0:
            vol_vs_avg = candles[-1].volume / avg_vol

    # Volatility rank (0-100 based on avg daily move)
    # < 1% = 10, 1-3% = 30, 3-5% = 50, 5-10% = 70, > 10% = 90
    if avg_move > 10:
        vol_rank = 95.0
    elif avg_move > 5:
        vol_rank = 70 + (avg_move - 5) * 5
    elif avg_move > 3:
        vol_rank = 50 + (avg_move - 3) * 10
    elif avg_move > 1:
        vol_rank = 20 + (avg_move - 1) * 15
    else:
        vol_rank = avg_move * 20

    activity_score = (
        (avg_move * 7.0)
        + (abs(last_day_move) * 8.0)
        + (max(0.0, vol_vs_avg - 1.0) * 10.0)
    )

    return TrendAnalysis(
        symbol=item.symbol,
        kind=item.kind,
        price=current_price,
        bullish_signals=bull_signals,
        bearish_signals=bear_signals,
        trend_score=trend_score,
        trend_label=trend_label,
        is_penny=(current_price < 5.0),
        volatility_rank=vol_rank,
        avg_daily_move_pct=round(avg_move, 2),
        last_day_move_pct=round(last_day_move, 2),
        volume_vs_avg=round(vol_vs_avg, 2),
        activity_score=round(activity_score, 2),
        sma_50=round(sma50, 4) if sma50 else None,
        sma_200=round(sma200, 4) if sma200 else None,
        golden_cross=golden,
        death_cross=death,
        higher_highs=hh,
        higher_lows=hl,
        breakout_detected=breakout,
    )


# ── Penny stock dynamic discovery ───────────────────────────────

def discover_trending_tickers() -> list[WatchlistItem]:
    """Pull currently trending tickers from free sources.

    Uses:
     • Yahoo Finance trending tickers API
     • Finviz screener for penny stocks with unusual volume
    Returns WatchlistItem objects ready for analysis.
    """
    discovered: list[WatchlistItem] = []
    seen: set[str] = set()

    def _validate_stock_symbol(sym: str) -> bool:
        if not yf:
            return True
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            last_price = float(getattr(info, "last_price", 0) or 0)
            return last_price > 0
        except Exception:
            return False

    def _validate_crypto_symbol(sym: str) -> bool:
        if not yf:
            return True
        try:
            ticker = yf.Ticker(f"{sym}-USD")
            info = ticker.fast_info
            last_price = float(getattr(info, "last_price", 0) or 0)
            return last_price > 0
        except Exception:
            return False

    def _add_discovered(sym: str, kind: str, reason: str) -> None:
        clean = str(sym or "").strip().upper()
        if not clean or clean in seen:
            return
        if not clean.isalpha() or len(clean) > 6:
            return
        if kind == "crypto":
            if not _validate_crypto_symbol(clean):
                return
            seen.add(clean)
            discovered.append(WatchlistItem(clean, "crypto", f"{clean}-USD", reason))
            return
        if not _validate_stock_symbol(clean):
            return
        seen.add(clean)
        discovered.append(WatchlistItem(clean, "stock", clean, reason))

    watchlist_symbols = {str(w.symbol).upper().replace("-USD", "") for w in DEFAULT_WATCHLIST}
    watchlist_crypto_symbols: list[str] = []
    seen_watchlist_crypto: set[str] = set()
    for w in DEFAULT_WATCHLIST:
        clean = str(getattr(w, "symbol", "")).upper().replace("-USD", "")
        kind = str(getattr(w, "kind", "")).lower()
        if kind != "crypto" or not clean or clean in seen_watchlist_crypto:
            continue
        seen_watchlist_crypto.add(clean)
        watchlist_crypto_symbols.append(clean)

    crypto_seed_limit = max(0, min(len(watchlist_crypto_symbols), int(settings.scan_dynamic_discovery_limit // 2)))
    for sym in watchlist_crypto_symbols[:crypto_seed_limit]:
        _add_discovered(sym, "crypto", "Watchlist crypto seed")

    def _scan_reddit_subreddit(subreddit: str, reason_prefix: str) -> None:
        post_limit = max(10, int(settings.discovery_reddit_post_limit))
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/hot.json",
            params={"limit": post_limit},
            headers={"User-Agent": "TradingBot/1.0 (by /u/trading-bot)"},
            timeout=12,
        )
        if not resp.ok:
            return

        data = resp.json()
        children = data.get("data", {}).get("children", [])
        explicit_counter: Counter[str] = Counter()
        loose_counter: Counter[str] = Counter()
        blocked_words = {
            "THE", "AND", "FOR", "ARE", "BUT", "ALL", "NOW", "NEW", "YOU", "YOUR",
            "WITH", "THIS", "THAT", "FROM", "WILL", "HOLD", "SELL", "BUY", "MOON",
            "WSB", "YOLO", "GAIN", "LOSS", "POST", "EDIT", "OPEN", "CLOSE", "HIGH",
            "LOW", "CHAT", "USA", "USD", "DD", "IMO", "FOMO", "AI", "ETF", "CALL",
            "CALLS", "PUT", "PUTS", "BULL", "BEAR", "LONG", "SHORT", "GREEN", "RED",
            "MARKET", "STOCK", "STOCKS", "OPTION", "OPTIONS", "CRYPTO", "COIN", "COINS",
            "TODAY", "TOMORROW", "WEEK", "MONTH", "YEAR", "NEWS", "THREAD", "PRICE",
        }

        for child in children:
            post = child.get("data", {}) if isinstance(child, dict) else {}
            title = str(post.get("title", "") or "")

            for match in re.findall(r"\$([A-Z]{1,6})|\(([A-Z]{1,6})\)", title.upper()):
                ticker = (match[0] or match[1]).strip()
                if 3 <= len(ticker) <= 6 and ticker not in blocked_words:
                    explicit_counter[ticker] += 1

            for token in re.findall(r"\b[A-Z]{3,6}\b", title.upper()):
                if token in blocked_words:
                    continue
                if token.isalpha():
                    loose_counter[token] += 1

        for ticker, count in explicit_counter.most_common(30):
            if count >= 2 and (ticker in watchlist_symbols or ticker in seen):
                kind = "crypto" if ticker in seen_watchlist_crypto else "stock"
                _add_discovered(ticker, kind, f"{reason_prefix} explicit mentions ({count})")

        for ticker, count in loose_counter.most_common(40):
            if count >= 5 and (ticker in watchlist_symbols or ticker in seen):
                kind = "crypto" if ticker in seen_watchlist_crypto else "stock"
                _add_discovered(ticker, kind, f"{reason_prefix} momentum mentions ({count})")

    # 1. Yahoo Finance trending
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/trending/US",
            params={"count": 30},
            headers={"User-Agent": "TradingBot/1.0"},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            for q in quotes:
                sym = q.get("symbol", "")
                if sym:
                    _add_discovered(sym, "stock", "Trending on Yahoo Finance")
            logger.info("Trending discovery: found {} Yahoo trending tickers", len(discovered))
    except Exception as e:
        logger.debug("Yahoo trending fetch failed: {}", e)

    if settings.discovery_enable_yahoo_most_active:
        try:
            resp = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                params={"scrIds": "most_actives", "count": max(10, settings.scan_dynamic_discovery_limit)},
                headers={"User-Agent": "TradingBot/1.0"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                quotes = (
                    data.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
                )
                for q in quotes:
                    sym = q.get("symbol", "")
                    if sym:
                        _add_discovered(sym, "stock", "Yahoo most active")
                logger.info("Trending discovery: total after Yahoo most-actives = {}", len(discovered))
        except Exception as e:
            logger.debug("Yahoo most-active fetch failed: {}", e)

    if settings.discovery_enable_coingecko_trending:
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                headers={"User-Agent": "TradingBot/1.0"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                for coin in data.get("coins", []):
                    item = coin.get("item", {}) if isinstance(coin, dict) else {}
                    sym = str(item.get("symbol", "")).upper().strip()
                    if sym:
                        _add_discovered(sym, "crypto", "CoinGecko trending")
                logger.info("Trending discovery: total after CoinGecko = {}", len(discovered))
        except Exception as e:
            logger.debug("CoinGecko trending fetch failed: {}", e)

    if settings.discovery_enable_robinhood_top100 and rh is not None:
        try:
            top100 = rh.get_top_100()
            if isinstance(top100, list):
                for row in top100:
                    if isinstance(row, dict):
                        sym = str(row.get("symbol", "")).upper().strip()
                        if sym:
                            _add_discovered(sym, "stock", "Robinhood top 100 popularity")
                logger.info("Trending discovery: total after Robinhood top100 = {}", len(discovered))
        except Exception as e:
            logger.debug("Robinhood top100 discovery failed: {}", e)

    if settings.discovery_enable_robinhood_top_movers and rh is not None:
        try:
            movers = rh.get_top_movers()
            if isinstance(movers, list):
                for row in movers:
                    if isinstance(row, dict):
                        sym = str(row.get("symbol", "")).upper().strip()
                        if sym:
                            _add_discovered(sym, "stock", "Robinhood top movers")
                logger.info("Trending discovery: total after Robinhood movers = {}", len(discovered))
        except Exception as e:
            logger.debug("Robinhood top movers discovery failed: {}", e)

    if settings.discovery_enable_stocktwits_trending:
        try:
            resp = requests.get(
                "https://api.stocktwits.com/api/2/trending/symbols.json",
                headers={"User-Agent": "TradingBot/1.0"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                for entry in data.get("symbols", []):
                    sym = ""
                    if isinstance(entry, dict):
                        sym = str(entry.get("symbol", "")).upper().strip()
                    if sym:
                        _add_discovered(sym, "stock", "Stocktwits trending")
                logger.info("Trending discovery: total after Stocktwits = {}", len(discovered))
        except Exception as e:
            logger.debug("Stocktwits trending fetch failed: {}", e)

    if settings.discovery_enable_reddit_wsb:
        try:
            _scan_reddit_subreddit("wallstreetbets", "Reddit WSB")
            logger.info("Trending discovery: total after Reddit WSB = {}", len(discovered))
        except Exception as e:
            logger.debug("Reddit WSB discovery failed: {}", e)

    if settings.discovery_enable_reddit_extra:
        try:
            raw = str(settings.discovery_reddit_extra_subreddits_csv or "")
            extra_subreddits = [s.strip() for s in raw.split(",") if s.strip()]
            for subreddit in extra_subreddits:
                _scan_reddit_subreddit(subreddit, f"Reddit r/{subreddit}")
            logger.info("Trending discovery: total after Reddit extras = {}", len(discovered))
        except Exception as e:
            logger.debug("Reddit extra discovery failed: {}", e)

    # 2. Google News RSS for "penny stock breakout" stories
    # NOTE: only accept explicitly-marked ticker tokens ($TSLA, (TSLA)) to
    # avoid pulling normal words like THESE/CLOSE as fake symbols.
    try:
        resp = requests.get(
            "https://news.google.com/rss/search?q=penny+stock+breakout+when:3d&hl=en-US&gl=US&ceid=US:en",
            headers={"User-Agent": "TradingBot/1.0"},
            timeout=10,
        )
        if resp.ok:
            root = ET.fromstring(resp.text)
            for item_el in root.findall(".//item")[:20]:
                title = item_el.findtext("title", "")
                candidates: set[str] = set()
                for match in re.findall(r"\$([A-Z]{1,5})|\(([A-Z]{1,5})\)", title.upper()):
                    candidate = (match[0] or match[1]).strip()
                    if 2 <= len(candidate) <= 5 and candidate.isalpha():
                        candidates.add(candidate)

                for clean in candidates:
                    _add_discovered(clean, "stock", f"Mentioned in penny stock news: {title[:60]}")
    except Exception as e:
        logger.debug("News ticker extraction failed: {}", e)

    limit = max(0, int(settings.scan_dynamic_discovery_limit))
    if limit == 0:
        return []
    return discovered[:limit]


# ── Full scan orchestrator ───────────────────────────────────────

def scan_for_bullish_opportunities(
    include_penny: bool = True,
    include_micro_crypto: bool = True,
    include_trending: bool = True,
    min_score: int = 15,
    top_n: int = 10,
    min_avg_daily_move_stock_pct: float = 2.0,
    min_avg_daily_move_crypto_pct: float = 3.0,
    min_last_day_move_pct: float = 1.0,
    min_volume_vs_avg: float = 0.7,
    volatility_weight: float = 0.35,
    activity_weight: float = 0.25,
) -> list[TrendAnalysis]:
    """Master scanner: screens everything and returns top bullish picks.

    Scans:
     • Static watchlist (20 items)
     • Penny stocks (20 items)
     • Micro-cap crypto (10 items)
     • Dynamically discovered trending tickers

    Returns the top N assets with bullish trend scores.
    """
    from .watchlist import DEFAULT_WATCHLIST

    # Build the full universe, deduplicating by symbol
    seen: set[str] = set()
    universe: list[WatchlistItem] = []

    def _add(items: list[WatchlistItem]) -> None:
        for item in items:
            if item.symbol not in seen:
                universe.append(item)
                seen.add(item.symbol)

    _add(DEFAULT_WATCHLIST)

    if include_penny:
        _add(PENNY_STOCKS)

    if include_micro_crypto:
        _add(MICRO_CRYPTO)

    if include_trending:
        try:
            trending = discover_trending_tickers()
            _add(trending)
            logger.info("Added trending tickers to scan universe (total: {})", len(universe))
        except Exception as e:
            logger.warning("Trending discovery failed: {}", e)

    logger.info("Scanning {} assets for bullish trends...", len(universe))

    results: list[TrendAnalysis] = []
    for item in universe:
        try:
            hist = fetch_historical(item=item, period="1y", interval="1d")
            if len(hist.candles) < 30:
                continue
            analysis = analyze_trend(item, hist.candles)
            results.append(analysis)
            if analysis.trend_score >= 20:
                logger.info(
                    "BULLISH: {} score={:+d} label={} price=${:.4f} {} bullish signals",
                    item.symbol, analysis.trend_score, analysis.trend_label,
                    analysis.price, len(analysis.bullish_signals),
                )
        except Exception as e:
            logger.debug("Failed to scan {}: {}", item.symbol, e)

    # Filter to bullish only if requested
    bullish = [r for r in results if r.trend_score >= min_score]

    def _min_avg_move_for_kind(kind: str) -> float:
        return min_avg_daily_move_crypto_pct if kind == "crypto" else min_avg_daily_move_stock_pct

    activity_filtered = [
        r for r in bullish
        if (
            r.avg_daily_move_pct >= _min_avg_move_for_kind(r.kind)
            and abs(r.last_day_move_pct) >= min_last_day_move_pct
            and r.volume_vs_avg >= min_volume_vs_avg
        )
    ]

    if not activity_filtered and bullish:
        if settings.daytrade_only_mode:
            logger.warning(
                "No bullish assets met strict movement/volume filters; day-trade mode keeps strict filter"
            )
            activity_filtered = []
        else:
            relaxed = [
                r for r in bullish
                if (
                    r.avg_daily_move_pct >= (_min_avg_move_for_kind(r.kind) * 0.75)
                    and abs(r.last_day_move_pct) >= (min_last_day_move_pct * 0.75)
                )
            ]
            if relaxed:
                logger.warning(
                    "No bullish assets met strict movement filters; using relaxed thresholds for this cycle"
                )
                activity_filtered = relaxed

    if not activity_filtered and not settings.daytrade_only_mode:
        activity_filtered = bullish

    def _rank_score(r: TrendAnalysis) -> float:
        return (
            float(r.trend_score)
            + (float(r.volatility_rank) * volatility_weight)
            + (float(r.activity_score) * activity_weight)
        )

    activity_filtered.sort(key=_rank_score, reverse=True)

    top_preview = [
        {
            "symbol": r.symbol,
            "kind": r.kind,
            "trend": r.trend_score,
            "avg_move_pct": r.avg_daily_move_pct,
            "last_day_move_pct": r.last_day_move_pct,
            "volume_vs_avg": r.volume_vs_avg,
        }
        for r in activity_filtered[:10]
    ]
    logger.info("Activity-ranked picks (top 10): {}", json.dumps(top_preview))

    logger.info(
        "Scan complete: {}/{} bullish; {}/{} passed movement filters",
        len(bullish), len(results), len(activity_filtered), len(bullish),
    )

    return activity_filtered[:top_n] if top_n else activity_filtered


def print_trend_report(results: list[TrendAnalysis]) -> None:
    """Pretty-print the trend scan results."""
    print()
    print("=" * 70)
    print("BULLISH TREND SCANNER REPORT")
    print("=" * 70)
    print(f"Found {len(results)} bullish opportunities\n")

    for i, r in enumerate(results, 1):
        penny_tag = " [PENNY]" if r.is_penny else ""
        gc_tag = " [GOLDEN CROSS]" if r.golden_cross else ""
        bo_tag = " [BREAKOUT]" if r.breakout_detected else ""
        print(
            f"{i:2d}. {r.symbol:6s} ({r.kind:6s}) "
            f"Score: {r.trend_score:+4d}  Price: ${r.price:>10.4f}  "
            f"Volatility: {r.avg_daily_move_pct:.1f}%/day  1d: {r.last_day_move_pct:+.1f}%"
            f"{penny_tag}{gc_tag}{bo_tag}"
        )
        print(f"    Trend: {r.trend_label:12s}  HH:{r.higher_highs} HL:{r.higher_lows}  "
              f"Vol: {r.volume_vs_avg:.1f}x avg")

        if r.bullish_signals:
            for s in r.bullish_signals[:4]:
                print(f"    + [{s.strength}/10] {s.description}")

        if r.bearish_signals:
            for s in r.bearish_signals[:2]:
                print(f"    - [{s.strength}/10] {s.description}")

        print()
