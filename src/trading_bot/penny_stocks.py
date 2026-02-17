"""Penny stock & small-cap screener — finds cheap stocks with bullish setups.

Uses yfinance (FREE) to scan a curated universe of Robinhood-available
penny stocks and small-caps (stocks under $10) for bullish patterns:
  • Golden cross (SMA50 crossing above SMA200)
  • High relative volume (3x+ average)
  • Oversold RSI bouncing up
  • Price breaking above resistance
  • Recent news catalysts

Discovered stocks are added to the dynamic watchlist for trading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import yfinance as yf
from loguru import logger

from .watchlist import WatchlistItem


# ── Penny stock universe ─────────────────────────────────────────
# Robinhood-available stocks under $10 with decent volume.
# This list is periodically refreshable, but here's a solid starter set.

PENNY_STOCK_UNIVERSE: list[str] = [
    # Biotech / Pharma (huge swing potential)
    "SNDL", "CLOV", "TLRY", "ACB", "MNMD", "BNGO",
    # EV / Clean Energy
    "FCEL", "PLUG", "LCID", "NKLA", "GOEV", "WKHS",
    # Tech / Fintech
    "BB", "WISH", "OPEN", "BARK", "SKLZ",
    # Mining / Materials
    "BTG", "SWN", "VALE", "CLF",
    # Meme / Retail favorites
    "BBBY", "EXPR", "CENN", "MULN",
    # Cannabis
    "CGC", "HEXO", "GRWG", "VFF",
    # China ADRs (volatile)
    "XPEV", "LI", "BABA",
    # Crypto-adjacent
    "HUT", "BITF", "CIFR", "CLSK",
    # Misc volatile small-caps
    "DNA", "PSFE", "UWMC", "LMND", "BYND",
]

# Small-cap stocks with higher prices but huge day-trade swings
SMALL_CAP_VOLATILE: list[str] = [
    "UPST", "AFRM", "HOOD", "COIN", "RBLX",
    "SNAP", "PINS", "U", "DKNG", "CHPT",
    "LAZR", "STEM", "QS", "JOBY", "IONQ",
]


@dataclass
class PennyStockCandidate:
    """A screened penny stock with bullish signals."""
    symbol: str
    price: float
    market_cap: float
    avg_volume: float
    relative_volume: float      # today's vol / 20d avg
    change_pct: float           # recent % change
    rsi_14: float
    sma_20: float
    sma_50: float
    golden_cross: bool          # SMA20 crossed above SMA50
    breakout: bool              # price > recent resistance
    oversold_bounce: bool       # RSI < 30 turning up
    volume_surge: bool          # rel_vol > 2.0
    score: int                  # composite bullish score 0-100
    category: str               # "penny" / "small_cap"
    reasons: list[str] = field(default_factory=list)


@dataclass
class PennyScreenResult:
    """Results from penny stock screening."""
    timestamp: str
    stocks_scanned: int
    candidates_found: int
    top_picks: list[PennyStockCandidate]
    scan_duration_seconds: float


class PennyStockScreener:
    """Scans penny stock universe for bullish setups using yfinance."""

    def __init__(self, max_price: float = 10.0, min_volume: float = 500_000) -> None:
        self.max_price = max_price
        self.min_volume = min_volume

    def scan(self, top_n: int = 10, include_small_caps: bool = True) -> PennyScreenResult:
        """Screen all penny stocks, return top N bullish candidates."""
        from datetime import datetime, timezone

        t0 = time.time()
        universe = list(PENNY_STOCK_UNIVERSE)
        if include_small_caps:
            universe.extend(SMALL_CAP_VOLATILE)

        # Deduplicate
        universe = list(dict.fromkeys(universe))

        candidates: list[PennyStockCandidate] = []
        scanned = 0

        # Batch download with yfinance for speed
        logger.info("Penny screener: scanning {} stocks...", len(universe))

        # Process in batches of 10 for efficiency
        batch_size = 10
        for batch_start in range(0, len(universe), batch_size):
            batch = universe[batch_start:batch_start + batch_size]
            batch_str = " ".join(batch)

            try:
                tickers = yf.Tickers(batch_str)
                for sym in batch:
                    try:
                        candidate = self._analyze_stock(tickers.tickers.get(sym), sym)
                        scanned += 1
                        if candidate and candidate.score > 0:
                            candidates.append(candidate)
                    except Exception as e:
                        logger.debug("Penny screener: skip {} — {}", sym, e)
                        scanned += 1
            except Exception as e:
                logger.warning("Penny screener batch failed: {}", e)

        # Sort by composite score
        candidates.sort(key=lambda c: c.score, reverse=True)
        top_picks = candidates[:top_n]

        elapsed = time.time() - t0
        logger.info(
            "Penny screener: {} scanned, {} candidates, top {} selected in {:.1f}s",
            scanned, len(candidates), len(top_picks), elapsed,
        )

        return PennyScreenResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stocks_scanned=scanned,
            candidates_found=len(candidates),
            top_picks=top_picks,
            scan_duration_seconds=round(elapsed, 1),
        )

    def _analyze_stock(self, ticker: yf.Ticker | None, symbol: str) -> PennyStockCandidate | None:
        """Analyze a single stock for bullish signals."""
        if ticker is None:
            return None

        # Get 3 months of daily history
        try:
            hist = ticker.history(period="3mo", interval="1d")
        except Exception:
            return None

        if hist.empty or len(hist) < 20:
            return None

        closes = hist["Close"].values
        volumes = hist["Volume"].values
        current_price = float(closes[-1])

        # Price filter (relax for small-caps)
        is_small_cap = symbol in SMALL_CAP_VOLATILE
        price_limit = 50.0 if is_small_cap else self.max_price
        if current_price > price_limit or current_price < 0.01:
            return None

        # Volume filter
        avg_vol_20 = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
        if avg_vol_20 < self.min_volume:
            return None

        # Calculate indicators
        sma_20 = float(closes[-20:].mean()) if len(closes) >= 20 else current_price
        sma_50 = float(closes[-50:].mean()) if len(closes) >= 50 else sma_20

        # RSI (14-period)
        rsi = self._calc_rsi(closes, 14)

        # Relative volume
        current_vol = float(volumes[-1])
        rel_vol = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # Recent change
        if len(closes) >= 5:
            change_5d = ((current_price - float(closes[-5])) / float(closes[-5])) * 100
        else:
            change_5d = 0.0

        # Market cap estimate
        try:
            info = ticker.info
            market_cap = float(info.get("marketCap", 0) or 0)
        except Exception:
            market_cap = 0

        # Bullish pattern detection
        golden_cross = sma_20 > sma_50 and len(closes) >= 50

        # Check if SMA20 JUST crossed above SMA50 (within last 5 days)
        recent_golden = False
        if len(closes) >= 55:
            prev_sma20 = float(closes[-25:-5].mean())
            prev_sma50 = float(closes[-55:-5].mean())
            if prev_sma20 <= prev_sma50 and sma_20 > sma_50:
                recent_golden = True

        # Resistance breakout: price above 20-day high
        recent_high = float(closes[-20:].max()) if len(closes) >= 20 else current_price
        breakout = current_price >= recent_high * 0.98  # within 2% of high

        # Oversold bounce: RSI was < 30, now rising
        oversold_bounce = rsi < 35 and change_5d > 0

        # Volume surge
        volume_surge = rel_vol > 2.0

        # Composite bullish score (0-100)
        score = 0
        reasons: list[str] = []

        if golden_cross:
            score += 20
            reasons.append("Golden cross (SMA20 > SMA50)")
        if recent_golden:
            score += 15
            reasons.append("FRESH golden cross!")

        if breakout:
            score += 20
            reasons.append(f"Breakout (near ${recent_high:.2f} high)")

        if oversold_bounce:
            score += 15
            reasons.append(f"Oversold bounce (RSI={rsi:.0f})")
        elif rsi < 40 and rsi > 25:
            score += 10
            reasons.append(f"RSI approaching oversold ({rsi:.0f})")

        if volume_surge:
            score += 15
            reasons.append(f"Volume surge ({rel_vol:.1f}x average)")
        elif rel_vol > 1.3:
            score += 5
            reasons.append(f"Above-avg volume ({rel_vol:.1f}x)")

        if change_5d > 5:
            score += 10
            reasons.append(f"Strong 5d momentum (+{change_5d:.1f}%)")
        elif change_5d > 2:
            score += 5
            reasons.append(f"Positive momentum (+{change_5d:.1f}%)")

        if current_price > sma_20:
            score += 5
            reasons.append("Price above SMA20")

        if sma_20 > sma_50:
            score += 5

        # Cap at 100
        score = min(100, score)

        category = "small_cap" if is_small_cap else "penny"

        return PennyStockCandidate(
            symbol=symbol,
            price=current_price,
            market_cap=market_cap,
            avg_volume=avg_vol_20,
            relative_volume=round(rel_vol, 2),
            change_pct=round(change_5d, 2),
            rsi_14=round(rsi, 1),
            sma_20=round(sma_20, 4),
            sma_50=round(sma_50, 4),
            golden_cross=golden_cross,
            breakout=breakout,
            oversold_bounce=oversold_bounce,
            volume_surge=volume_surge,
            score=score,
            category=category,
            reasons=reasons,
        )

    @staticmethod
    def _calc_rsi(closes, period: int = 14) -> float:
        """Simple RSI calculation."""
        if len(closes) < period + 1:
            return 50.0  # neutral default
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


def candidates_to_watchlist(candidates: list[PennyStockCandidate]) -> list[WatchlistItem]:
    """Convert penny stock candidates to WatchlistItems for trading."""
    items: list[WatchlistItem] = []
    for c in candidates:
        items.append(WatchlistItem(
            symbol=c.symbol,
            kind="stock",
            yf_ticker=c.symbol,
            description=f"Screened {c.category} — {', '.join(c.reasons[:2])}",
        ))
    return items


def print_penny_report(result: PennyScreenResult) -> None:
    """Pretty-print penny stock screening results."""
    print("=" * 70)
    print(f"PENNY STOCK SCREENER — {result.timestamp}")
    print(f"Scanned {result.stocks_scanned} stocks in {result.scan_duration_seconds}s")
    print(f"Found {result.candidates_found} candidates")
    print("=" * 70)

    if not result.top_picks:
        print("\nNo bullish candidates found right now.")
        return

    print(f"\nTOP {len(result.top_picks)} BULLISH PICKS:")
    for i, c in enumerate(result.top_picks, 1):
        flags = []
        if c.golden_cross:
            flags.append("🟢GoldenX")
        if c.breakout:
            flags.append("🚀Breakout")
        if c.oversold_bounce:
            flags.append("📈Bounce")
        if c.volume_surge:
            flags.append("📊VolSurge")

        flag_str = " ".join(flags) if flags else ""
        print(
            f"  {i:2d}. {c.symbol:6s}  ${c.price:8.2f}  Score:{c.score:3d}  "
            f"RSI:{c.rsi_14:5.1f}  Vol:{c.relative_volume:4.1f}x  "
            f"Chg:{c.change_pct:+6.1f}%  {flag_str}"
        )
        for r in c.reasons[:3]:
            print(f"       → {r}")
