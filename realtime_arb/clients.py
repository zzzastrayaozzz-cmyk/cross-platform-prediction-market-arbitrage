from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from requests import Session

from .models import Event, FetchBundle, MarketPrices
from .utils import chunked, ensure_dir, now_utc_iso, parse_iso_datetime, safe_float, write_json

class BaseHttpClient:
    def __init__(self) -> None:
        self.session = Session()
        self.session.headers.update({"User-Agent": "Codex-Arbitrage-Realtime/3.0", "Accept": "application/json"})

    def request_json(self, method: str, url: str, *, timeout: float = 30.0, max_retries: int = 6, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, timeout=timeout, **kwargs)
                if response.status_code == 429:
                    retry_after = safe_float(response.headers.get("Retry-After")) or float(min(2 ** attempt, 10))
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, requests.SSLError) as error:
                last_error = error
                if attempt == max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 10))
        if last_error is not None:
            raise last_error
        raise RuntimeError("HTTP request failed without an explicit error.")

class PolymarketClient(BaseHttpClient):
    MARKETS_API = "https://gamma-api.polymarket.com/markets"
    ORDERBOOK_API = "https://clob.polymarket.com/book"
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    @staticmethod
    def _event_from_market(market: Dict[str, Any]) -> Optional[Event]:
        if market.get("closed") or not market.get("active", True) or market.get("archived"):
            return None
        outcome_prices = None
        price_blob = market.get("outcomePrices")
        if isinstance(price_blob, str):
            try:
                parsed = json.loads(price_blob)
                if len(parsed) >= 2:
                    outcome_prices = (float(parsed[0]), float(parsed[1]))
            except Exception:
                outcome_prices = None
        elif isinstance(price_blob, list) and len(price_blob) >= 2:
            try:
                outcome_prices = (float(price_blob[0]), float(price_blob[1]))
            except Exception:
                outcome_prices = None
        token_ids: List[str] = []
        token_blob = market.get("clobTokenIds")
        if isinstance(token_blob, str):
            try:
                token_ids = [str(item) for item in json.loads(token_blob)]
            except Exception:
                token_ids = []
        elif isinstance(token_blob, list):
            token_ids = [str(item) for item in token_blob]
        title = market.get("question") or market.get("title") or ""
        return Event(
            platform="polymarket",
            event_id=str(market.get("id", "")),
            title=title,
            description=market.get("description", "") or "",
            resolution_date=parse_iso_datetime(market.get("endDateIso") or market.get("endDate")),
            category=market.get("category"),
            slug=market.get("slug"),
            token_ids=token_ids,
            outcome_prices=outcome_prices,
            best_ask=safe_float(market.get("bestAsk")),
            best_bid=safe_float(market.get("bestBid")),
            last_trade_price=safe_float(market.get("lastTradePrice")),
            liquidity=safe_float(market.get("liquidityNum") or market.get("liquidity")),
            volume=safe_float(market.get("volumeNum") or market.get("volume")),
            volume_24h=safe_float(market.get("volume24hr")),
            open_interest=safe_float(market.get("openInterest")),
            updated_at=now_utc_iso(),
        )

    def fetch_active_markets(self, raw_pages_dir: Path, page_size: int = 500, max_pages: int = 0) -> FetchBundle:
        ensure_dir(raw_pages_dir)
        raw_items: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        page_count = 0
        offset = 0
        current_page_size = max(50, min(page_size, 500))
        while True:
            params = {"active": "true", "closed": "false", "archived": "false", "limit": str(current_page_size), "offset": str(offset)}
            try:
                payload = self.request_json("GET", self.MARKETS_API, params=params, timeout=45)
            except Exception:
                if current_page_size > 100:
                    current_page_size = max(100, current_page_size // 2)
                    continue
                raise
            batch = payload if isinstance(payload, list) else []
            page_count += 1
            write_json(raw_pages_dir / f"page_{page_count:04d}_offset_{offset:06d}.json", {
                "platform": "polymarket",
                "page": page_count,
                "offset": offset,
                "limit": current_page_size,
                "received": len(batch),
                "items": batch,
            })
            for market in batch:
                market_id = str(market.get("id", ""))
                if market_id and market_id not in seen_ids:
                    seen_ids.add(market_id)
                    raw_items.append(market)
            if len(batch) < current_page_size or (max_pages and page_count >= max_pages):
                break
            offset += len(batch)
        events = [event for event in (self._event_from_market(market) for market in raw_items) if event is not None]
        return FetchBundle("polymarket", now_utc_iso(), page_count, current_page_size, raw_items, events, {"pagination": "offset", "final_offset": offset})

    def prices_from_event(self, event: Event) -> Optional[MarketPrices]:
        if event.outcome_prices:
            yes_mid, no_mid = event.outcome_prices
        elif event.best_ask is not None and event.best_bid is not None:
            yes_mid = (event.best_ask + event.best_bid) / 2.0
            no_mid = 1.0 - yes_mid
        elif event.last_trade_price is not None:
            yes_mid = event.last_trade_price
            no_mid = 1.0 - yes_mid
        else:
            return None
        yes_ask = event.best_ask if event.best_ask is not None else yes_mid
        yes_bid = event.best_bid if event.best_bid is not None else yes_mid
        prices = MarketPrices(yes_mid, no_mid, yes_ask, yes_bid, 1.0 - yes_bid, 1.0 - yes_ask, event.liquidity or 0.0, event.last_trade_price)
        return prices if prices.validate() else None

    def get_order_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.request_json("GET", self.ORDERBOOK_API, params={"token_id": token_id}, timeout=10, max_retries=3)
        except Exception:
            return None

class KalshiClient(BaseHttpClient):
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    @staticmethod
    def _event_from_market(market: Dict[str, Any]) -> Optional[Event]:
        status = str(market.get("status", "")).lower()
        if status and status not in {"open", "active"}:
            return None
        if str(market.get("result", "")).strip():
            return None
        title = market.get("title", "") or ""
        yes_sub_title = market.get("yes_sub_title") or ""
        if yes_sub_title:
            title = f"{title} - {yes_sub_title}"
        return Event(
            platform="kalshi",
            event_id=str(market.get("ticker", "")),
            title=title,
            description=market.get("subtitle", "") or "",
            resolution_date=parse_iso_datetime(market.get("expiration_time")),
            slug=market.get("ticker"),
            best_ask=safe_float(market.get("yes_ask_dollars")),
            best_bid=safe_float(market.get("yes_bid_dollars")),
            last_trade_price=safe_float(market.get("last_price_dollars")),
            liquidity=safe_float(market.get("liquidity")),
            volume=safe_float(market.get("volume_fp") or market.get("volume")),
            volume_24h=safe_float(market.get("volume_24h_fp") or market.get("volume_24h")),
            open_interest=safe_float(market.get("open_interest")),
            updated_at=now_utc_iso(),
        )

    def fetch_active_markets(self, raw_pages_dir: Path, page_size: int = 1000, max_pages: int = 0) -> FetchBundle:
        ensure_dir(raw_pages_dir)
        raw_items: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        page_count = 0
        cursor: Optional[str] = None
        current_page_size = max(100, min(page_size, 1000))
        while True:
            params: Dict[str, Any] = {"status": "open", "limit": str(current_page_size), "mve_filter": "exclude"}
            if cursor:
                params["cursor"] = cursor
            try:
                payload = self.request_json("GET", f"{self.BASE_URL}/markets", params=params, timeout=45)
            except Exception:
                if current_page_size > 200:
                    current_page_size = max(200, current_page_size // 2)
                    continue
                raise
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            next_cursor = payload.get("cursor") if isinstance(payload, dict) else None
            page_count += 1
            write_json(raw_pages_dir / f"page_{page_count:04d}.json", {
                "platform": "kalshi",
                "page": page_count,
                "limit": current_page_size,
                "cursor_in": cursor,
                "cursor_out": next_cursor,
                "received": len(markets),
                "items": markets,
            })
            for market in markets:
                ticker = str(market.get("ticker", ""))
                if ticker and ticker not in seen_ids:
                    seen_ids.add(ticker)
                    raw_items.append(market)
            cursor = next_cursor
            if not cursor or (max_pages and page_count >= max_pages):
                break
        events = [event for event in (self._event_from_market(market) for market in raw_items) if event is not None]
        return FetchBundle("kalshi", now_utc_iso(), page_count, current_page_size, raw_items, events, {"pagination": "cursor", "final_cursor": cursor, "mve_filter": "exclude"})

    def fetch_market_snapshots(self, tickers: Sequence[str]) -> List[Event]:
        events: List[Event] = []
        for batch in chunked(list(tickers), 100):
            params = {"tickers": ",".join(batch), "mve_filter": "exclude", "limit": str(len(batch))}
            payload = self.request_json("GET", f"{self.BASE_URL}/markets", params=params, timeout=20, max_retries=3)
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                event = self._event_from_market(market)
                if event is not None:
                    events.append(event)
        return events

    def prices_from_event(self, event: Event) -> Optional[MarketPrices]:
        if event.best_ask is None and event.best_bid is None and event.last_trade_price is None:
            return None
        if event.best_ask is not None and event.best_bid is not None:
            yes_mid = (event.best_ask + event.best_bid) / 2.0
        elif event.last_trade_price is not None:
            yes_mid = event.last_trade_price
        else:
            yes_mid = event.best_ask if event.best_ask is not None else event.best_bid
        yes_ask = event.best_ask if event.best_ask is not None else yes_mid
        yes_bid = event.best_bid if event.best_bid is not None else yes_mid
        prices = MarketPrices(yes_mid, 1.0 - yes_mid, yes_ask, yes_bid, 1.0 - yes_bid, 1.0 - yes_ask, event.liquidity or 0.0, event.last_trade_price)
        return prices if prices.validate() else None

    def get_order_book(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return self.request_json("GET", f"{self.BASE_URL}/markets/{ticker}/orderbook", timeout=10, max_retries=3)
        except Exception:
            return None

    def websocket_headers(self) -> Optional[Dict[str, str]]:
        key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
        private_key_pem = os.environ.get("KALSHI_PRIVATE_KEY_PEM", "").strip()
        private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
        if not key_id:
            return None
        if private_key_pem:
            pem_bytes = private_key_pem.encode("utf-8")
        elif private_key_path:
            pem_bytes = Path(private_key_path).read_bytes()
        else:
            return None
        private_key = serialization.load_pem_private_key(pem_bytes, password=None)
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}GET/trade-api/ws/v2".encode("utf-8")
        signature = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

