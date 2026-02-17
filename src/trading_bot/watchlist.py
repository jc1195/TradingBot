"""Volatile-asset watchlist for backtesting and live trading.

Curated list of volatile crypto and stock tickers that fluctuate frequently,
suitable for day-trading strategies.  Every ticker has a ``kind`` (crypto / stock)
so the data layer knows which API / yfinance suffix to use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchlistItem:
    symbol: str          # canonical symbol, e.g. "DOGE", "AMC"
    kind: str            # "crypto" or "stock"
    yf_ticker: str       # yahoo-finance ticker, e.g. "DOGE-USD", "AMC"
    description: str


# ── Volatile crypto assets ────────────────────────────────────────
CRYPTO_WATCHLIST: list[WatchlistItem] = [
    WatchlistItem("DOGE", "crypto", "DOGE-USD",  "Dogecoin – high retail volatility"),
    WatchlistItem("SHIB", "crypto", "SHIB-USD",  "Shiba Inu – meme-coin swings"),
    WatchlistItem("SOL",  "crypto", "SOL-USD",   "Solana – large intraday range"),
    WatchlistItem("AVAX", "crypto", "AVAX-USD",  "Avalanche – momentum bursts"),

    WatchlistItem("XRP",  "crypto", "XRP-USD",   "Ripple – news-driven spikes"),
    WatchlistItem("ADA",  "crypto", "ADA-USD",   "Cardano – moderate swings"),
    WatchlistItem("BTC",  "crypto", "BTC-USD",   "Bitcoin – baseline benchmark"),
    WatchlistItem("ETH",  "crypto", "ETH-USD",   "Ethereum – DeFi beta"),
    WatchlistItem("MATIC","crypto", "MATIC-USD",  "Polygon – layer-2 momentum"),
]

# ── Volatile stock tickers ────────────────────────────────────────
STOCK_WATCHLIST: list[WatchlistItem] = [
    WatchlistItem("AMC",  "stock", "AMC",   "AMC Entertainment – meme-stock"),
    WatchlistItem("GME",  "stock", "GME",   "GameStop – high short-interest"),
    WatchlistItem("TSLA", "stock", "TSLA",  "Tesla – momentum & sentiment"),
    WatchlistItem("NVDA", "stock", "NVDA",  "Nvidia – AI/GPU sector volatility"),
    WatchlistItem("PLTR", "stock", "PLTR",  "Palantir – retail sentiment plays"),
    WatchlistItem("RIVN", "stock", "RIVN",  "Rivian – EV sector swings"),
    WatchlistItem("SOFI", "stock", "SOFI",  "SoFi – fintech small-cap moves"),
    WatchlistItem("NIO",  "stock", "NIO",   "NIO – China EV volatility"),
    WatchlistItem("MARA", "stock", "MARA",  "Marathon Digital – crypto proxy"),
    WatchlistItem("RIOT", "stock", "RIOT",  "Riot Platforms – crypto proxy"),
]

# ── Penny stocks (< $5, Robinhood-available, high volatility) ────
PENNY_WATCHLIST: list[WatchlistItem] = [
    WatchlistItem("SNDL", "stock", "SNDL", "Sundial Growers – cannabis penny stock"),
    WatchlistItem("CLOV", "stock", "CLOV", "Clover Health – fintech/health penny"),
    WatchlistItem("BNGO", "stock", "BNGO", "Bionano Genomics – biotech penny"),
    WatchlistItem("OPEN", "stock", "OPEN", "Opendoor – real estate tech penny"),
    WatchlistItem("TLRY", "stock", "TLRY", "Tilray Brands – cannabis penny"),
    WatchlistItem("HIMS", "stock", "HIMS", "Hims & Hers – telehealth"),
    WatchlistItem("BITF", "stock", "BITF", "Bitfarms – Bitcoin mining penny"),
    WatchlistItem("BB",   "stock", "BB",   "BlackBerry – cybersecurity penny"),
    WatchlistItem("GSAT", "stock", "GSAT", "Globalstar – satellite penny"),
    WatchlistItem("NOK",  "stock", "NOK",  "Nokia – 5G / IoT"),
    WatchlistItem("WKHS", "stock", "WKHS", "Workhorse Group – EV penny"),

    WatchlistItem("UUUU", "stock", "UUUU", "Energy Fuels – uranium penny"),
]

# ── Small-cap crypto tokens ──────────────────────────────────────
MICRO_CRYPTO_WATCHLIST: list[WatchlistItem] = [
    WatchlistItem("HBAR",  "crypto", "HBAR-USD",  "Hedera – enterprise DLT"),
    WatchlistItem("VET",   "crypto", "VET-USD",   "VeChain – supply chain"),
    WatchlistItem("ALGO",  "crypto", "ALGO-USD",  "Algorand – fast L1"),
    WatchlistItem("CRO",   "crypto", "CRO-USD",   "Cronos – exchange token"),
    WatchlistItem("NEAR",  "crypto", "NEAR-USD",  "Near Protocol – sharded L1"),
    WatchlistItem("ATOM",  "crypto", "ATOM-USD",  "Cosmos – interop ecosystem"),
    WatchlistItem("FIL",   "crypto", "FIL-USD",   "Filecoin – storage crypto"),
    WatchlistItem("RENDER","crypto", "RENDER-USD", "Render – GPU compute token"),
    WatchlistItem("INJ",   "crypto", "INJ-USD",   "Injective – DeFi protocol"),
]

# ── Combined default watchlist ────────────────────────────────────
DEFAULT_WATCHLIST: list[WatchlistItem] = (
    CRYPTO_WATCHLIST + STOCK_WATCHLIST
    + PENNY_WATCHLIST + MICRO_CRYPTO_WATCHLIST
)


def get_watchlist(
    kinds: list[str] | None = None,
    symbols: list[str] | None = None,
    include_penny: bool = True,
    include_micro_crypto: bool = True,
) -> list[WatchlistItem]:
    """Return filtered watchlist items.

    Parameters
    ----------
    kinds : list[str] | None
        Filter to ``["crypto"]``, ``["stock"]``, or both.
    symbols : list[str] | None
        Limit to these symbols (case-insensitive).  ``None`` = all.
    include_penny : bool
        Include penny stocks in results.
    include_micro_crypto : bool
        Include micro-cap crypto in results.
    """
    # Build base list depending on flags
    items: list[WatchlistItem] = CRYPTO_WATCHLIST + STOCK_WATCHLIST
    if include_penny:
        items = items + PENNY_WATCHLIST
    if include_micro_crypto:
        items = items + MICRO_CRYPTO_WATCHLIST

    if kinds:
        lower_kinds = {k.lower() for k in kinds}
        items = [i for i in items if i.kind in lower_kinds]
    if symbols:
        upper_syms = {s.upper() for s in symbols}
        items = [i for i in items if i.symbol in upper_syms]
    return items
