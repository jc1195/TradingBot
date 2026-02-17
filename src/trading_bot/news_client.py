from __future__ import annotations

from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from loguru import logger
import requests

from .settings import settings


SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "SOL": ["sol", "solana"],
    "DOGE": ["doge", "dogecoin"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["ada", "cardano"],
}


class NewsClient:
    def __init__(self) -> None:
        self.feed_urls = [url.strip() for url in settings.news_feed_urls_csv.split(",") if url.strip()]

    def fetch_news_for_symbol(self, symbol: str) -> list[dict]:
        aliases = SYMBOL_ALIASES.get(symbol.upper(), [symbol.lower()])
        collected: list[dict] = []
        seen_keys: set[str] = set()

        for feed_url in self.feed_urls:
            try:
                items = self._fetch_feed_items(feed_url)
                for item in items:
                    text_blob = f"{item.get('title', '')} {item.get('summary', '')}".lower()
                    if not any(alias in text_blob for alias in aliases):
                        continue

                    dedupe_key = f"{item.get('url', '')}::{item.get('title', '')}"
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)

                    collected.append(
                        {
                            "symbol": symbol,
                            "source": item.get("source", feed_url),
                            "title": item.get("title", f"News for {symbol}"),
                            "url": item.get("url", ""),
                            "published_at": item.get("published_at") or datetime.now(timezone.utc).isoformat(),
                            "sentiment": self._estimate_sentiment(text_blob),
                        }
                    )

                    if len(collected) >= settings.news_max_items_per_symbol:
                        return collected
            except Exception as exc:
                logger.warning("News feed read failed for {}: {}", feed_url, exc)

        if collected:
            return collected

        return [
            {
                "symbol": symbol,
                "source": "fallback",
                "title": f"No matching live feed items found for {symbol}",
                "url": "",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "sentiment": 0.0,
            }
        ]

    def _fetch_feed_items(self, feed_url: str) -> list[dict]:
        response = requests.get(feed_url, timeout=settings.news_request_timeout_seconds)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        results: list[dict] = []

        rss_items = root.findall(".//item")
        atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        if rss_items:
            for item in rss_items[: settings.news_max_items_per_feed]:
                title = self._xml_text(item.find("title"))
                summary = self._xml_text(item.find("description"))
                link = self._xml_text(item.find("link"))
                published = self._xml_text(item.find("pubDate"))
                results.append(
                    {
                        "source": feed_url,
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "published_at": published,
                    }
                )
            return results

        for entry in atom_entries[: settings.news_max_items_per_feed]:
            title = self._xml_text(entry.find("{http://www.w3.org/2005/Atom}title"))
            summary = self._xml_text(entry.find("{http://www.w3.org/2005/Atom}summary"))
            link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
            link = ""
            if link_elem is not None:
                link = link_elem.attrib.get("href", "")
            published = self._xml_text(entry.find("{http://www.w3.org/2005/Atom}updated"))
            results.append(
                {
                    "source": feed_url,
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "published_at": published,
                }
            )

        return results

    @staticmethod
    def _xml_text(node: ET.Element | None) -> str:
        if node is None or node.text is None:
            return ""
        return node.text.strip()

    @staticmethod
    def _estimate_sentiment(text: str) -> float:
        positive = ["surge", "rally", "gain", "bull", "breakout", "adoption", "approval"]
        negative = ["drop", "selloff", "loss", "bear", "hack", "exploit", "ban", "lawsuit"]

        pos_hits = sum(1 for word in positive if word in text)
        neg_hits = sum(1 for word in negative if word in text)
        if pos_hits == 0 and neg_hits == 0:
            return 0.0

        score = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
        return max(-1.0, min(1.0, float(score)))
