"""Enhanced news & sentiment aggregator — FREE sources only.

Pulls headlines from multiple free RSS/API sources:
 • Google News RSS (any stock/crypto)
 • CoinDesk RSS (crypto)
 • CoinTelegraph RSS (crypto)
 • Yahoo Finance RSS (stocks)
 • Fear & Greed Index API (crypto market mood)
 • Finviz RSS (stocks)

Computes a simple sentiment score (-1.0 to +1.0) from keyword matching.
This is injected into AI prompts so the model can weigh market mood.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import requests
from loguru import logger


# ── Sentiment keyword lists ──────────────────────────────────────

POSITIVE_WORDS = [
    "surge", "rally", "gain", "bull", "breakout", "adoption", "approval",
    "rocket", "soar", "jump", "spike", "upgrade", "outperform", "boost",
    "record", "high", "profit", "growth", "momentum", "buy", "bullish",
    "optimism", "recover", "beat", "exceed", "strong", "upbeat", "moon",
    "partnership", "launch", "milestone", "innovation", "expansion",
]

NEGATIVE_WORDS = [
    "drop", "selloff", "loss", "bear", "hack", "exploit", "ban", "lawsuit",
    "crash", "plunge", "dump", "downgrade", "underperform", "fraud",
    "scam", "risk", "concern", "warning", "decline", "sell", "bearish",
    "pessimism", "fear", "miss", "weak", "problem", "delay", "bankrupt",
    "investigation", "regulation", "fine", "penalty", "layoff", "cut",
]


# ── Aliases for matching headlines to symbols ─────────────────────

SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "SOL": ["sol", "solana"],
    "DOGE": ["doge", "dogecoin"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["ada", "cardano"],
    "SHIB": ["shib", "shiba"],
    "AVAX": ["avax", "avalanche"],
    "PEPE": ["pepe"],
    "MATIC": ["matic", "polygon"],
    "HBAR": ["hbar", "hedera"],
    "VET": ["vet", "vechain"],
    "ALGO": ["algo", "algorand"],
    "CRO": ["cro", "cronos"],
    "NEAR": ["near protocol", "near"],
    "ATOM": ["atom", "cosmos"],
    "FIL": ["fil", "filecoin"],
    "RENDER": ["render", "rndr"],
    "INJ": ["inj", "injective"],
    "TSLA": ["tsla", "tesla"],
    "NVDA": ["nvda", "nvidia"],
    "PLTR": ["pltr", "palantir"],
    "AMC": ["amc"],
    "GME": ["gme", "gamestop"],
    "RIVN": ["rivn", "rivian"],
    "SOFI": ["sofi"],
    "NIO": ["nio"],
    "MARA": ["mara", "marathon digital"],
    "RIOT": ["riot"],
    "SNDL": ["sndl", "sundial"],
    "CLOV": ["clov", "clover health"],
    "BNGO": ["bngo", "bionano"],
    "NKLA": ["nkla", "nikola"],
    "OPEN": ["opendoor"],
    "TLRY": ["tlry", "tilray"],
    "HIMS": ["hims"],
    "BITF": ["bitf", "bitfarms"],
    "BB": ["blackberry"],
    "GSAT": ["gsat", "globalstar"],
    "NOK": ["nok", "nokia"],
    "WKHS": ["wkhs", "workhorse"],
    "TELL": ["tell", "tellurian"],
    "UUUU": ["uuuu", "energy fuels", "uranium"],
}


@dataclass
class NewsHeadline:
    symbol: str
    title: str
    source: str
    url: str
    published: str
    sentiment: float  # -1.0 to 1.0


@dataclass
class MarketSentiment:
    """Aggregated sentiment for a symbol."""
    symbol: str
    headline_count: int
    avg_sentiment: float        # -1.0 to 1.0
    positive_count: int
    negative_count: int
    neutral_count: int
    fear_greed_index: int | None  # 0-100 (crypto market only)
    fear_greed_label: str | None  # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    top_headlines: list[str]      # top 3 most relevant headlines
    sentiment_label: str          # "very_bullish" / "bullish" / "neutral" / "bearish" / "very_bearish"


# ── Feed URLs ─────────────────────────────────────────────────────

CRYPTO_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://cryptopanic.com/news/rss/",
]

STOCK_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
]

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+when:7d&hl=en-US&gl=US&ceid=US:en"

FEAR_GREED_API = "https://api.alternative.me/fng/?limit=1"


class SentimentAnalyzer:
    """Multi-source news aggregator with sentiment scoring."""

    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "TradingBot/1.0 (news aggregator)",
        })
        self._fear_greed_cache: dict | None = None
        self._fear_greed_ts: float = 0

    # ── Public API ────────────────────────────────────────────────

    def get_sentiment(self, symbol: str, kind: str = "crypto") -> MarketSentiment:
        """Get aggregated sentiment for a symbol from all available sources."""
        headlines = self._fetch_all_headlines(symbol, kind)

        sentiments = [h.sentiment for h in headlines]
        avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
        pos = sum(1 for s in sentiments if s > 0.1)
        neg = sum(1 for s in sentiments if s < -0.1)
        neu = len(sentiments) - pos - neg

        fg = self._get_fear_greed() if kind == "crypto" else None

        # Label
        if avg > 0.3:
            label = "very_bullish"
        elif avg > 0.1:
            label = "bullish"
        elif avg < -0.3:
            label = "very_bearish"
        elif avg < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        top = [h.title for h in sorted(headlines, key=lambda h: abs(h.sentiment), reverse=True)[:3]]

        return MarketSentiment(
            symbol=symbol,
            headline_count=len(headlines),
            avg_sentiment=round(avg, 3),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            fear_greed_index=fg.get("value") if fg else None,
            fear_greed_label=fg.get("value_classification") if fg else None,
            top_headlines=top,
            sentiment_label=label,
        )

    def get_market_mood(self) -> dict:
        """Get overall crypto market mood via Fear & Greed Index."""
        fg = self._get_fear_greed()
        if fg:
            return {
                "fear_greed_index": fg.get("value", 50),
                "label": fg.get("value_classification", "Neutral"),
                "timestamp": fg.get("timestamp", ""),
            }
        return {"fear_greed_index": 50, "label": "Neutral", "timestamp": ""}

    def to_prompt_dict(self, sentiment: MarketSentiment) -> dict:
        """Convert sentiment to a compact dict for AI prompt injection."""
        d = {
            "news_sentiment": sentiment.sentiment_label,
            "avg_score": sentiment.avg_sentiment,
            "headlines": sentiment.headline_count,
            "positive": sentiment.positive_count,
            "negative": sentiment.negative_count,
            "top_news": sentiment.top_headlines[:2],
        }
        if sentiment.fear_greed_index is not None:
            d["crypto_fear_greed"] = f"{sentiment.fear_greed_index} ({sentiment.fear_greed_label})"
        return d

    # ── Internal fetchers ─────────────────────────────────────────

    def _fetch_all_headlines(self, symbol: str, kind: str) -> list[NewsHeadline]:
        headlines: list[NewsHeadline] = []
        aliases = SYMBOL_ALIASES.get(symbol.upper(), [symbol.lower()])

        # 1) Google News RSS (works for everything)
        try:
            query = "+OR+".join(aliases[:2])
            url = GOOGLE_NEWS_RSS.format(query=query)
            items = self._parse_rss(url)
            for item in items[:8]:
                headlines.append(NewsHeadline(
                    symbol=symbol,
                    title=item.get("title", ""),
                    source="google_news",
                    url=item.get("url", ""),
                    published=item.get("published", ""),
                    sentiment=self._score_text(item.get("title", "")),
                ))
        except Exception as e:
            logger.debug("Google News fetch failed for {}: {}", symbol, e)

        # 2) Crypto-specific feeds
        if kind == "crypto":
            for feed_url in CRYPTO_FEEDS:
                try:
                    items = self._parse_rss(feed_url)
                    for item in items[:15]:
                        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
                        if not any(a in text for a in aliases):
                            continue
                        headlines.append(NewsHeadline(
                            symbol=symbol,
                            title=item.get("title", ""),
                            source=feed_url.split("/")[2],
                            url=item.get("url", ""),
                            published=item.get("published", ""),
                            sentiment=self._score_text(text),
                        ))
                except Exception as e:
                    logger.debug("Crypto feed failed {}: {}", feed_url, e)

        # 3) Stock-specific feeds (Yahoo Finance RSS)
        if kind == "stock":
            for feed_tpl in STOCK_FEEDS:
                try:
                    url = feed_tpl.format(symbol=symbol)
                    items = self._parse_rss(url)
                    for item in items[:8]:
                        headlines.append(NewsHeadline(
                            symbol=symbol,
                            title=item.get("title", ""),
                            source="yahoo_finance",
                            url=item.get("url", ""),
                            published=item.get("published", ""),
                            sentiment=self._score_text(item.get("title", "")),
                        ))
                except Exception as e:
                    logger.debug("Yahoo feed failed for {}: {}", symbol, e)

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[NewsHeadline] = []
        for h in headlines:
            key = h.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(h)

        return unique

    def _parse_rss(self, url: str) -> list[dict]:
        """Parse RSS/Atom feed and return list of items."""
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        results: list[dict] = []

        # RSS 2.0
        for item in root.findall(".//item"):
            results.append({
                "title": self._xt(item.find("title")),
                "summary": self._xt(item.find("description")),
                "url": self._xt(item.find("link")),
                "published": self._xt(item.find("pubDate")),
            })

        # Atom
        if not results:
            ns = "{http://www.w3.org/2005/Atom}"
            for entry in root.findall(f".//{ns}entry"):
                link_el = entry.find(f"{ns}link")
                results.append({
                    "title": self._xt(entry.find(f"{ns}title")),
                    "summary": self._xt(entry.find(f"{ns}summary")),
                    "url": link_el.attrib.get("href", "") if link_el is not None else "",
                    "published": self._xt(entry.find(f"{ns}updated")),
                })

        return results

    def _get_fear_greed(self) -> dict:
        """Fetch crypto Fear & Greed Index (cached 30 min)."""
        now = time.time()
        if self._fear_greed_cache and (now - self._fear_greed_ts) < 1800:
            return self._fear_greed_cache

        try:
            resp = self._session.get(FEAR_GREED_API, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("data"):
                self._fear_greed_cache = data["data"][0]
                self._fear_greed_ts = now
                return self._fear_greed_cache
        except Exception as e:
            logger.debug("Fear & Greed API failed: {}", e)

        return {}

    @staticmethod
    def _score_text(text: str) -> float:
        """Simple keyword-based sentiment score."""
        text_lower = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / total))

    @staticmethod
    def _xt(node: ET.Element | None) -> str:
        if node is None or node.text is None:
            return ""
        return node.text.strip()
