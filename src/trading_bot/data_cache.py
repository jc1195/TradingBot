"""Historical data downloader & local cache.

Downloads OHLCV candle data from yfinance and saves it locally as JSON
so we don't re-download on every backtest run.  Supports:
 • Bulk download of all watchlist assets
 • Multiple timeframes (1d, 1h, etc.)
 • Auto-refresh if data is stale (> 1 day old for daily, > 1 hour for hourly)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .historical_data import Candle, fetch_historical
from .watchlist import WatchlistItem, get_watchlist


CACHE_DIR = Path("data/historical_cache")


def _cache_path(symbol: str, period: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{period}_{interval}.json"


def _is_fresh(path: Path, interval: str) -> bool:
    """Check if cached data is still fresh enough."""
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    if interval in ("1h", "2h", "4h"):
        return age_seconds < 3600  # 1 hour
    return age_seconds < 86400  # 1 day for daily data


def save_candles(candles: list[Candle], path: Path) -> None:
    """Write candles to JSON cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_candles(path: Path) -> list[Candle]:
    """Read candles from JSON cache file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Candle(
            timestamp=d["timestamp"],
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            volume=d["volume"],
        )
        for d in data
    ]


def fetch_cached(
    item: WatchlistItem,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> list[Candle]:
    """Fetch candle data — from cache if fresh, otherwise from yfinance."""
    path = _cache_path(item.symbol, period, interval)

    if not force_refresh and _is_fresh(path, interval):
        candles = load_candles(path)
        logger.debug("Cache hit: {} {} — {} candles", item.symbol, period, len(candles))
        return candles

    # Download fresh
    logger.info("Downloading {} {} {} from yfinance...", item.symbol, period, interval)
    hist = fetch_historical(
        item=item,
        period=period,
        interval=interval,
    )
    candles = hist.candles
    if candles:
        save_candles(candles, path)
        logger.info("Cached {} candles for {} at {}", len(candles), item.symbol, path)
    return candles


def bulk_download(
    kinds: list[str] | None = None,
    symbols: list[str] | None = None,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict[str, int]:
    """Download and cache data for all watchlist items.

    Returns dict of symbol -> candle_count.
    """
    items = get_watchlist(kinds=kinds, symbols=symbols)
    results: dict[str, int] = {}

    logger.info("Bulk download: {} assets, period={}, interval={}", len(items), period, interval)

    for item in items:
        try:
            candles = fetch_cached(item, period, interval, force_refresh)
            results[item.symbol] = len(candles)
        except Exception as e:
            logger.error("Download failed for {}: {}", item.symbol, e)
            results[item.symbol] = 0

    total = sum(results.values())
    logger.info("Bulk download complete: {} candles across {} assets", total, len(results))
    return results


def get_cache_stats() -> dict:
    """Return statistics about cached data."""
    if not CACHE_DIR.exists():
        return {"files": 0, "total_candles": 0, "total_size_mb": 0}

    files = list(CACHE_DIR.glob("*.json"))
    total_candles = 0
    total_bytes = 0

    for f in files:
        total_bytes += f.stat().st_size
        try:
            candles = load_candles(f)
            total_candles += len(candles)
        except Exception:
            pass

    return {
        "files": len(files),
        "total_candles": total_candles,
        "total_size_mb": round(total_bytes / 1_048_576, 2),
    }
