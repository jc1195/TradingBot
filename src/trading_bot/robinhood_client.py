from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger
import robin_stocks.robinhood as rh

from .settings import settings


class RobinhoodApiError(RuntimeError):
    pass


@dataclass
class AccountSnapshot:
    buying_power: float
    equity: float
    mode: str
    session_authenticated: bool
    cash: float = 0.0
    portfolio_cash: float = 0.0
    cash_available_for_withdrawal: float = 0.0
    cash_held_for_orders: float = 0.0
    crypto_buying_power: float = 0.0


class RobinhoodClient:
    def __init__(self) -> None:
        self._authenticated = False
        self._crypto_symbols_cache: set[str] = set()
        self._crypto_symbols_cache_at: float = 0.0

    def ensure_session(self) -> bool:
        if self._authenticated:
            return True

        if not settings.robinhood_username or not settings.robinhood_password:
            if settings.bot_mode == "paper":
                logger.warning("Robinhood credentials not configured; continuing in paper-safe mode")
                return False
            raise RobinhoodApiError("Robinhood credentials are required in live mode")

        login_kwargs: dict[str, Any] = {
            "username": settings.robinhood_username,
            "password": settings.robinhood_password,
            "store_session": True,
        }
        if settings.robinhood_mfa_code:
            login_kwargs["mfa_code"] = settings.robinhood_mfa_code

        try:
            response = self._call_with_retry(rh.login, **login_kwargs)
            if isinstance(response, dict) and response.get("access_token"):
                self._authenticated = True
                logger.info("Robinhood session established")
                return True

            if settings.bot_mode == "paper":
                logger.warning("Robinhood login unavailable; using paper-safe fallback data")
                return False

            raise RobinhoodApiError(f"Robinhood login failed: {response}")
        except Exception as exc:
            if settings.bot_mode == "paper":
                logger.warning("Robinhood login failed in paper mode: {}", exc)
                return False
            raise

    def get_account_snapshot(self) -> dict:
        if not self.ensure_session():
            return AccountSnapshot(
                buying_power=0.0,
                equity=0.0,
                mode=settings.bot_mode,
                session_authenticated=False,
            ).__dict__

        profile = self._call_with_retry(rh.profiles.load_account_profile)
        buying_power = float(profile.get("buying_power") or 0.0)
        equity = float(profile.get("portfolio_cash") or profile.get("cash") or 0.0)
        cash = float(profile.get("cash") or 0.0)
        portfolio_cash = float(profile.get("portfolio_cash") or 0.0)
        cash_available_for_withdrawal = float(profile.get("cash_available_for_withdrawal") or 0.0)
        cash_held_for_orders = float(profile.get("cash_held_for_orders") or 0.0)
        crypto_buying_power = float(profile.get("crypto_buying_power") or 0.0)
        return AccountSnapshot(
            buying_power=buying_power,
            equity=equity,
            mode=settings.bot_mode,
            session_authenticated=True,
            cash=cash,
            portfolio_cash=portfolio_cash,
            cash_available_for_withdrawal=cash_available_for_withdrawal,
            cash_held_for_orders=cash_held_for_orders,
            crypto_buying_power=crypto_buying_power,
        ).__dict__

    def get_unified_transfers(self, limit: int = 50) -> list[dict]:
        if not self.ensure_session():
            return []
        try:
            rows = self._call_with_retry(rh.account.get_unified_transfers)
            if not isinstance(rows, list):
                return []
            cleaned = [row for row in rows if isinstance(row, dict)]
            return cleaned[: max(1, limit)]
        except Exception as exc:
            logger.warning("Failed to fetch unified transfers: {}", exc)
            return []

    def get_portfolio_totals(self) -> dict:
        """Return aggregate investing account value including current holdings."""
        if not self.ensure_session():
            return {
                "equity": 0.0,
                "market_value": 0.0,
                "withdrawable_amount": 0.0,
                "portfolio_equity_previous_close": 0.0,
                "session_authenticated": False,
            }

        try:
            profile = self._call_with_retry(rh.profiles.load_portfolio_profile)
            if not isinstance(profile, dict):
                return {
                    "equity": 0.0,
                    "market_value": 0.0,
                    "withdrawable_amount": 0.0,
                    "portfolio_equity_previous_close": 0.0,
                    "session_authenticated": True,
                }
            return {
                "equity": float(profile.get("equity") or 0.0),
                "market_value": float(profile.get("market_value") or 0.0),
                "withdrawable_amount": float(profile.get("withdrawable_amount") or 0.0),
                "portfolio_equity_previous_close": float(profile.get("portfolio_equity_previous_close") or 0.0),
                "session_authenticated": True,
            }
        except Exception as exc:
            logger.warning("Failed to fetch portfolio totals: {}", exc)
            return {
                "equity": 0.0,
                "market_value": 0.0,
                "withdrawable_amount": 0.0,
                "portfolio_equity_previous_close": 0.0,
                "session_authenticated": True,
            }

    def get_crypto_symbols(self) -> list[str]:
        if not self.ensure_session():
            return ["BTC", "ETH", "SOL", "DOGE"]

        try:
            pairs = self._call_with_retry(rh.crypto.get_crypto_currency_pairs)
            symbols = {
                str(pair.get("symbol") or "").upper().replace("-USD", "")
                for pair in pairs
                if isinstance(pair, dict)
                and pair.get("symbol")
                and pair.get("tradability") in {"tradable", "untradable"}
            }
            cleaned = sorted(symbol for symbol in symbols if symbol)
            self._crypto_symbols_cache = {str(s).upper().replace("-USD", "") for s in cleaned}
            self._crypto_symbols_cache_at = time.time()
            return cleaned or ["BTC", "ETH", "SOL", "DOGE"]
        except Exception as exc:
            logger.warning("Failed fetching Robinhood symbols, using fallback list: {}", exc)
            return ["BTC", "ETH", "SOL", "DOGE"]

    def _get_crypto_symbols_set(self) -> set[str]:
        stale = (time.time() - self._crypto_symbols_cache_at) > 3600
        if not self._crypto_symbols_cache or stale:
            try:
                self.get_crypto_symbols()
            except Exception:
                pass
        return set(self._crypto_symbols_cache)

    def get_crypto_quote(self, symbol: str) -> dict:
        if not self.ensure_session():
            return {
                "symbol": symbol,
                "mark_price": 0.0,
                "bid_price": 0.0,
                "ask_price": 0.0,
                "session_authenticated": False,
            }

        symbol_upper = str(symbol or "").upper().replace("-USD", "")
        supported = self._get_crypto_symbols_set()
        if supported and symbol_upper not in supported:
            return {
                "symbol": symbol,
                "mark_price": 0.0,
                "bid_price": 0.0,
                "ask_price": 0.0,
                "session_authenticated": True,
            }

        try:
            quote = self._call_with_retry(rh.crypto.get_crypto_quote, symbol)
        except Exception as exc:
            logger.warning("Failed to fetch crypto quote for {}: {}", symbol, exc)
            quote = None

        if not isinstance(quote, dict):
            return {
                "symbol": symbol,
                "mark_price": 0.0,
                "bid_price": 0.0,
                "ask_price": 0.0,
                "session_authenticated": True,
            }

        return {
            "symbol": symbol,
            "mark_price": float(quote.get("mark_price") or 0.0),
            "bid_price": float(quote.get("bid_price") or 0.0),
            "ask_price": float(quote.get("ask_price") or 0.0),
            "session_authenticated": True,
        }

    def get_crypto_historicals(
        self,
        symbol: str,
        interval: str = "hour",
        span: str = "day",
        bounds: str = "24_7",
    ) -> list[dict]:
        if not self.ensure_session():
            return []

        try:
            candles = self._call_with_retry(
                rh.crypto.get_crypto_historicals,
                symbol,
                interval=interval,
                span=span,
                bounds=bounds,
            )
        except Exception as exc:
            logger.warning("Failed to fetch crypto historicals for {}: {}", symbol, exc)
            return []
        return candles if isinstance(candles, list) else []

    # ── Stock support ─────────────────────────────────────────────

    def get_stock_quote(self, symbol: str) -> dict:
        """Fetch a real-time stock quote from Robinhood."""
        if not self.ensure_session():
            return {
                "symbol": symbol,
                "mark_price": 0.0,
                "bid_price": 0.0,
                "ask_price": 0.0,
                "session_authenticated": False,
            }

        try:
            quote = self._call_with_retry(rh.stocks.get_latest_price, symbol)
        except Exception as exc:
            logger.warning("Failed to fetch stock quote for {}: {}", symbol, exc)
            quote = None

        price = 0.0
        if isinstance(quote, list) and quote:
            price = float(quote[0] or 0.0)
        elif isinstance(quote, str):
            price = float(quote or 0.0)

        return {
            "symbol": symbol,
            "mark_price": price,
            "bid_price": price,
            "ask_price": price,
            "session_authenticated": True,
        }

    def get_stock_historicals(
        self,
        symbol: str,
        interval: str = "hour",
        span: str = "day",
        bounds: str = "regular",
    ) -> list[dict]:
        """Fetch stock historical candles from Robinhood."""
        if not self.ensure_session():
            return []

        try:
            candles = self._call_with_retry(
                rh.stocks.get_stock_historicals,
                symbol,
                interval=interval,
                span=span,
                bounds=bounds,
            )
        except Exception as exc:
            logger.warning("Failed to fetch stock historicals for {}: {}", symbol, exc)
            return []
        return candles if isinstance(candles, list) else []

    def get_quote(self, symbol: str, kind: str = "crypto") -> dict:
        """Unified quote fetcher.  *kind* is ``"crypto"`` or ``"stock"``."""
        if kind == "stock":
            return self.get_stock_quote(symbol)
        return self.get_crypto_quote(symbol)

    def get_historicals(
        self,
        symbol: str,
        kind: str = "crypto",
        interval: str = "hour",
        span: str = "day",
    ) -> list[dict]:
        """Unified historicals fetcher."""
        if kind == "stock":
            return self.get_stock_historicals(symbol, interval=interval, span=span)
        return self.get_crypto_historicals(symbol, interval=interval, span=span)

    def place_buy_order(self, symbol: str, kind: str, quantity: float) -> dict:
        """Submit a live market buy order.

        Returns raw broker response dict when available.
        """
        if quantity <= 0:
            raise RobinhoodApiError("Quantity must be > 0 for buy order")
        if not self.ensure_session():
            raise RobinhoodApiError("Robinhood session not available")

        try:
            if kind == "crypto":
                response = self._call_with_retry(
                    rh.orders.order_buy_crypto_by_quantity,
                    symbol,
                    quantity,
                )
            else:
                response = self._call_with_retry(
                    rh.orders.order_buy_fractional_by_quantity,
                    symbol,
                    quantity,
                )
            return response if isinstance(response, dict) else {"raw": str(response)}
        except Exception as exc:
            raise RobinhoodApiError(f"Failed live buy order for {symbol}: {exc}") from exc

    def place_sell_order(self, symbol: str, kind: str, quantity: float) -> dict:
        """Submit a live market sell order.

        Returns raw broker response dict when available.
        """
        if quantity <= 0:
            raise RobinhoodApiError("Quantity must be > 0 for sell order")
        if not self.ensure_session():
            raise RobinhoodApiError("Robinhood session not available")

        try:
            if kind == "crypto":
                response = self._call_with_retry(
                    rh.orders.order_sell_crypto_by_quantity,
                    symbol,
                    quantity,
                )
            else:
                response = self._call_with_retry(
                    rh.orders.order_sell_fractional_by_quantity,
                    symbol,
                    quantity,
                )
            return response if isinstance(response, dict) else {"raw": str(response)}
        except Exception as exc:
            raise RobinhoodApiError(f"Failed live sell order for {symbol}: {exc}") from exc

    def _call_with_retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None

        for attempt in range(1, settings.robinhood_max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                category = self._classify_error(exc)
                logger.warning(
                    "Robinhood call failed [{}] attempt {}/{}: {}",
                    category,
                    attempt,
                    settings.robinhood_max_retries,
                    exc,
                )

                if attempt >= settings.robinhood_max_retries:
                    break

                if self._is_not_found_error(exc):
                    break

                delay = settings.robinhood_retry_base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)

        raise RobinhoodApiError(f"Robinhood call failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        text = str(exc).lower()
        if "timeout" in text:
            return "timeout"
        if "401" in text or "403" in text or "unauthorized" in text:
            return "auth"
        if "429" in text or "rate" in text:
            return "rate_limit"
        return "api"

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "404" in text or "not found" in text
