"""Historical market-data fetcher for backtesting.

Uses **yfinance** for free, unauthenticated access to years of OHLCV candle data
for both crypto (``DOGE-USD``, ``BTC-USD``, …) and stocks (``AMC``, ``TSLA``, …).
This avoids the short-lookback limitation of Robinhood's historicals API.

All data is returned as a list of plain dicts so the backtester can iterate
candle-by-candle without pandas being required downstream.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass, field

from loguru import logger

# Suppress noisy yfinance FutureWarning about pandas 3.0 ChainedAssignment
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

try:
    import yfinance as yf  # type: ignore[import-untyped]
except ImportError:
    yf = None  # type: ignore[assignment]

from .watchlist import WatchlistItem, get_watchlist


@dataclass
class Candle:
    """Single OHLCV bar."""
    timestamp: str      # ISO-8601 datetime string
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class HistoricalDataResult:
    """Container returned by the fetcher for one ticker."""
    symbol: str
    kind: str           # "crypto" or "stock"
    yf_ticker: str
    candles: list[Candle] = field(default_factory=list)
    error: str | None = None


# ── public API ────────────────────────────────────────────────────

def fetch_historical(
    item: WatchlistItem,
    period: str = "6mo",
    interval: str = "1h",
) -> HistoricalDataResult:
    """Download historical candles for a single watchlist item.

    Parameters
    ----------
    item : WatchlistItem
        Ticker metadata (symbol, kind, yf_ticker).
    period : str
        yfinance period string.  Common values:
        ``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``, ``"max"``.
    interval : str
        Candle interval.  Common values:
        ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``, ``"1wk"``.
        Note: intraday intervals (< 1d) are limited to 60 days of history on
        Yahoo Finance.  Use ``"1d"`` for longer lookbacks.
    """
    if yf is None:
        return HistoricalDataResult(
            symbol=item.symbol,
            kind=item.kind,
            yf_ticker=item.yf_ticker,
            error="yfinance is not installed – run `pip install yfinance`",
        )

    try:
        ticker = yf.Ticker(item.yf_ticker)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return HistoricalDataResult(
                symbol=item.symbol,
                kind=item.kind,
                yf_ticker=item.yf_ticker,
                error=f"No data returned for {item.yf_ticker} period={period} interval={interval}",
            )

        candles: list[Candle] = []
        for ts, row in df.iterrows():
            candles.append(
                Candle(
                    timestamp=str(ts.isoformat()) if hasattr(ts, "isoformat") else str(ts),
                    open=float(row.get("Open", 0.0)),
                    high=float(row.get("High", 0.0)),
                    low=float(row.get("Low", 0.0)),
                    close=float(row.get("Close", 0.0)),
                    volume=float(row.get("Volume", 0.0)),
                )
            )

        logger.info(
            "Fetched {} candles for {} ({}) period={} interval={}",
            len(candles), item.symbol, item.yf_ticker, period, interval,
        )
        return HistoricalDataResult(
            symbol=item.symbol,
            kind=item.kind,
            yf_ticker=item.yf_ticker,
            candles=candles,
        )

    except Exception as exc:
        logger.warning("Failed to fetch historical data for {}: {}", item.yf_ticker, exc)
        return HistoricalDataResult(
            symbol=item.symbol,
            kind=item.kind,
            yf_ticker=item.yf_ticker,
            error=str(exc),
        )


def fetch_all_historicals(
    kinds: list[str] | None = None,
    symbols: list[str] | None = None,
    period: str = "6mo",
    interval: str = "1d",
) -> list[HistoricalDataResult]:
    """Fetch historical data for every ticker on the watchlist.

    Parameters
    ----------
    kinds : list[str] | None
        ``["crypto"]``, ``["stock"]``, or both.  ``None`` = all.
    symbols : list[str] | None
        Restrict to specific symbols.  ``None`` = all on watchlist.
    period / interval :
        Passed through to yfinance.  Defaults are conservative (daily bars,
        6 months) to stay within free-tier limits.
    """
    items = get_watchlist(kinds=kinds, symbols=symbols)
    results: list[HistoricalDataResult] = []

    for item in items:
        result = fetch_historical(item, period=period, interval=interval)
        results.append(result)

    ok = sum(1 for r in results if r.error is None)
    fail = sum(1 for r in results if r.error is not None)
    logger.info("Historical fetch complete: {} succeeded, {} failed out of {} tickers", ok, fail, len(results))
    return results
