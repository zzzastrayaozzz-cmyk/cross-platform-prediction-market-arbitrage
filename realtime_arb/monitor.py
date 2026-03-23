from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .clients import KalshiClient, PolymarketClient
from .constants import DEFAULT_CATEGORIES
from .execution import (
    PaperTrader,
    build_adjusted_prices,
    close_value_for_side,
    detect_arbitrage,
    execute_split_order,
    parse_kalshi_orderbook,
    parse_polymarket_orderbook,
)
from .matching import CategoryClassifier, FastStreamingMatcher, benchmark_vector_pipeline
from .models import ArbitrageOpportunity, Event, ExecutionReport, FetchBundle, MatchedPair, OpportunityRecord
from .persistence import load_bundle_from_raw, persist_fetch_bundle, write_combined_structured_data, write_matches
from .utils import append_jsonl, build_run_id, chunked, ensure_dir, now_utc, now_utc_iso, safe_float, write_csv, write_json

class StreamingArbitrageMonitor:
    def __init__(self, args: argparse.Namespace, root_dir: Path) -> None:
        self.args = args
        self.root_dir = root_dir
        self.run_id = build_run_id()
        self.raw_dir = ensure_dir(root_dir / "raw_data")
        self.structured_dir = ensure_dir(root_dir / "structured_data")
        self.analysis_dir = ensure_dir(root_dir / "analysis")
        self.report_dir = ensure_dir(root_dir / "report")
        self.stream_dir = ensure_dir(self.raw_dir / "stream")

        self.pm_client = PolymarketClient()
        self.kalshi_client = KalshiClient()
        self.classifier = CategoryClassifier(DEFAULT_CATEGORIES)
        self.matcher = FastStreamingMatcher(args.similarity_threshold, self.classifier, args.similarity_chunk_size, args.hash_features)
        self.trader = PaperTrader(args.take_profit, args.flatten_spread, args.reopen_cooldown)

        self.http_session: Optional[aiohttp.ClientSession] = None
        self.stop_event = asyncio.Event()
        self.bootstrap_lock = asyncio.Lock()
        self.stream_tasks: List[asyncio.Task[Any]] = []
        self.analysis_task: Optional[asyncio.Task[Any]] = None
        self.refresh_task: Optional[asyncio.Task[Any]] = None

        self.pm_bundle: Optional[FetchBundle] = None
        self.kalshi_bundle: Optional[FetchBundle] = None
        self.pm_events: Dict[str, Event] = {}
        self.kalshi_events: Dict[str, Event] = {}
        self.watch_pairs: List[MatchedPair] = []
        self.pm_watch_asset_ids: List[str] = []
        self.kalshi_watch_tickers: List[str] = []
        self.pm_orderbooks: Dict[str, Dict[str, Any]] = {}
        self.pm_token_to_event: Dict[str, Event] = {}
        self.kalshi_ticker_to_event: Dict[str, Event] = {}

        self.bootstrap_summary: Dict[str, Any] = {}
        self.vector_benchmark: Dict[str, Any] = {}
        self.last_cycle_summary: Dict[str, Any] = {}
        self.cycle_index = 0
        self.pm_stream_messages = 0
        self.kalshi_stream_messages = 0
        self.pm_stream_connected = False
        self.kalshi_stream_connected = False
        self.pm_stream_mode = "polling"
        self.kalshi_stream_mode = "polling"
        self.last_pm_stream_at: Optional[str] = None
        self.last_kalshi_stream_at: Optional[str] = None
        self.pm_poll_cursor = 0

    def append_decision_log(self, payload: Dict[str, Any]) -> None:
        append_jsonl(self.analysis_dir / "decision_log.jsonl", {
            "run_id": self.run_id,
            "recorded_at": now_utc_iso(),
            **payload,
        })

    async def bootstrap(self, reason: str) -> None:
        async with self.bootstrap_lock:
            pm_pages_dir = ensure_dir(self.raw_dir / "polymarket" / "pages")
            kalshi_pages_dir = ensure_dir(self.raw_dir / "kalshi" / "pages")
            if self.args.reuse_existing_raw and (self.raw_dir / "polymarket" / "all_active_markets_raw.json").exists() and (self.raw_dir / "kalshi" / "all_active_markets_raw.json").exists():
                pm_bundle = load_bundle_from_raw(self.raw_dir, "polymarket", self.pm_client._event_from_market)
                kalshi_bundle = load_bundle_from_raw(self.raw_dir, "kalshi", self.kalshi_client._event_from_market)
            elif self.args.bootstrap_from and (Path(self.args.bootstrap_from) / "polymarket" / "all_active_markets_raw.json").exists() and (Path(self.args.bootstrap_from) / "kalshi" / "all_active_markets_raw.json").exists():
                source_root = Path(self.args.bootstrap_from)
                pm_bundle = load_bundle_from_raw(source_root, "polymarket", self.pm_client._event_from_market)
                kalshi_bundle = load_bundle_from_raw(source_root, "kalshi", self.kalshi_client._event_from_market)
            else:
                pm_bundle = await asyncio.to_thread(self.pm_client.fetch_active_markets, pm_pages_dir, self.args.polymarket_page_size, self.args.polymarket_max_pages)
                kalshi_bundle = await asyncio.to_thread(self.kalshi_client.fetch_active_markets, kalshi_pages_dir, self.args.kalshi_page_size, self.args.kalshi_max_pages)

            persist_fetch_bundle(self.root_dir, pm_bundle)
            persist_fetch_bundle(self.root_dir, kalshi_bundle)
            write_combined_structured_data(self.root_dir, pm_bundle.events, kalshi_bundle.events)

            self.matcher.fit(pm_bundle.events, kalshi_bundle.events)
            matched_pairs = self.matcher.match_bidirectional(pm_bundle.events, kalshi_bundle.events)
            write_matches(self.root_dir, matched_pairs)

            benchmark_titles = [event.title for event in pm_bundle.events] + [event.title for event in kalshi_bundle.events]
            if len(benchmark_titles) > self.args.benchmark_docs:
                benchmark_titles = benchmark_titles[:self.args.benchmark_docs]
            self.vector_benchmark = benchmark_vector_pipeline(benchmark_titles, self.args.hash_features)
            write_json(self.analysis_dir / "vector_benchmark.json", self.vector_benchmark)

            self.pm_bundle = pm_bundle
            self.kalshi_bundle = kalshi_bundle
            self.pm_events = {event.event_id: event for event in pm_bundle.events}
            self.kalshi_events = {event.event_id: event for event in kalshi_bundle.events}
            self.watch_pairs = [pair for pair in matched_pairs if pair.pm_event.token_ids][:self.args.watchlist_limit]
            self.pm_watch_asset_ids = sorted({pair.pm_event.token_ids[0] for pair in self.watch_pairs if pair.pm_event.token_ids})
            self.kalshi_watch_tickers = sorted({pair.kalshi_event.event_id for pair in self.watch_pairs})
            self.pm_token_to_event = {pair.pm_event.token_ids[0]: pair.pm_event for pair in self.watch_pairs if pair.pm_event.token_ids}
            self.kalshi_ticker_to_event = {pair.kalshi_event.event_id: pair.kalshi_event for pair in self.watch_pairs}

            write_json(self.structured_dir / "watchlist.json", {
                "generated_at": now_utc_iso(),
                "reason": reason,
                "watch_pairs": len(self.watch_pairs),
                "polymarket_asset_ids": len(self.pm_watch_asset_ids),
                "kalshi_tickers": len(self.kalshi_watch_tickers),
                "pairs": [{
                    "label": pair.label,
                    "similarity": round(pair.similarity, 6),
                    "pm_event_id": pair.pm_event.event_id,
                    "pm_title": pair.pm_event.title,
                    "pm_asset_id": pair.pm_event.token_ids[0] if pair.pm_event.token_ids else None,
                    "kalshi_event_id": pair.kalshi_event.event_id,
                    "kalshi_title": pair.kalshi_event.title,
                } for pair in self.watch_pairs],
            })

            self.bootstrap_summary = {
                "bootstrap_reason": reason,
                "bootstrapped_at": now_utc_iso(),
                "polymarket_raw_market_count": len(pm_bundle.raw_items),
                "kalshi_raw_market_count": len(kalshi_bundle.raw_items),
                "polymarket_structured_market_count": len(pm_bundle.events),
                "kalshi_structured_market_count": len(kalshi_bundle.events),
                "polymarket_page_count": pm_bundle.page_count,
                "kalshi_page_count": kalshi_bundle.page_count,
                "matched_pairs": len(matched_pairs),
                "watch_pairs": len(self.watch_pairs),
                "hash_feature_dimension": self.matcher.feature_dimension,
                "analysis_interval_seconds": self.args.analysis_interval,
                "full_refresh_seconds": self.args.full_refresh_seconds,
                "polymarket_poll_seconds": self.args.polymarket_poll_seconds,
                "kalshi_poll_seconds": self.args.kalshi_poll_seconds,
            }
            write_json(self.analysis_dir / "bootstrap_summary.json", self.bootstrap_summary)

    async def start_stream_tasks(self) -> None:
        self.stream_tasks = [
            asyncio.create_task(self.polymarket_ws_loop(), name="polymarket-ws"),
            asyncio.create_task(self.polymarket_poll_loop(), name="polymarket-poll"),
            asyncio.create_task(self.kalshi_ws_loop(), name="kalshi-ws"),
            asyncio.create_task(self.kalshi_poll_loop(), name="kalshi-poll"),
        ]

    async def stop_stream_tasks(self) -> None:
        if not self.stream_tasks:
            return
        for task in self.stream_tasks:
            task.cancel()
        await asyncio.gather(*self.stream_tasks, return_exceptions=True)
        self.stream_tasks = []

    async def restart_stream_tasks(self) -> None:
        await self.stop_stream_tasks()
        await self.start_stream_tasks()

    async def polymarket_ws_loop(self) -> None:
        if not self.pm_watch_asset_ids:
            return
        assert self.http_session is not None
        while not self.stop_event.is_set():
            heartbeat_task: Optional[asyncio.Task[Any]] = None
            try:
                async with self.http_session.ws_connect(PolymarketClient.WS_URL, heartbeat=25, autoping=True) as ws:
                    self.pm_stream_connected = True
                    self.pm_stream_mode = "websocket"
                    heartbeat_task = asyncio.create_task(self.polymarket_heartbeat(ws))
                    first_chunk = True
                    for batch in chunked(self.pm_watch_asset_ids, self.args.stream_subscribe_batch_size):
                        if first_chunk:
                            payload = {"type": "market", "assets_ids": list(batch), "custom_feature_enabled": True}
                            first_chunk = False
                        else:
                            payload = {"operation": "subscribe", "assets_ids": list(batch)}
                        await ws.send_json(payload)
                    async for message in ws:
                        if self.stop_event.is_set():
                            break
                        if message.type == aiohttp.WSMsgType.TEXT:
                            if message.data == "PONG":
                                continue
                            try:
                                payload = json.loads(message.data)
                            except json.JSONDecodeError:
                                continue
                            await self.handle_polymarket_message(payload)
                        elif message.type in {aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED}:
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                self.pm_stream_mode = "polling"
                await asyncio.sleep(self.args.stream_reconnect_seconds)
            finally:
                self.pm_stream_connected = False
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def polymarket_poll_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.polymarket_poll_seconds)
                return
            except asyncio.TimeoutError:
                pass
            if self.pm_stream_connected or not self.pm_watch_asset_ids:
                continue
            self.pm_stream_mode = "polling"
            batch = self.pm_watch_asset_ids[self.pm_poll_cursor:self.pm_poll_cursor + self.args.polymarket_poll_batch_size]
            if not batch:
                self.pm_poll_cursor = 0
                batch = self.pm_watch_asset_ids[:self.args.polymarket_poll_batch_size]
            self.pm_poll_cursor = (self.pm_poll_cursor + len(batch)) % max(len(self.pm_watch_asset_ids), 1)
            semaphore = asyncio.Semaphore(self.args.max_concurrent_verifications)

            async def fetch_and_update(token_id: str) -> None:
                async with semaphore:
                    book = await asyncio.to_thread(self.pm_client.get_order_book, token_id)
                if not book or token_id not in self.pm_token_to_event:
                    return
                self.pm_orderbooks[token_id] = book
                event = self.pm_token_to_event[token_id]
                yes_asks = parse_polymarket_orderbook(book, "YES")
                yes_bids: List[Tuple[float, float]] = []
                for raw_bid in book.get("bids", []):
                    if isinstance(raw_bid, dict):
                        price = safe_float(raw_bid.get("price")) or 0.0
                        size = safe_float(raw_bid.get("size")) or 0.0
                    elif isinstance(raw_bid, (list, tuple)) and len(raw_bid) >= 2:
                        price = safe_float(raw_bid[0]) or 0.0
                        size = safe_float(raw_bid[1]) or 0.0
                    else:
                        continue
                    if 0.0 < price < 1.0 and size > 0:
                        yes_bids.append((price, size))
                event.best_ask = yes_asks[0][0] if yes_asks else event.best_ask
                event.best_bid = max((price for price, _ in yes_bids), default=event.best_bid)
                event.updated_at = now_utc_iso()
                self.last_pm_stream_at = event.updated_at

            await asyncio.gather(*(fetch_and_update(token_id) for token_id in batch), return_exceptions=True)

    async def polymarket_heartbeat(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(10)
            try:
                await ws.send_str("PING")
            except Exception:
                return

    async def handle_polymarket_message(self, payload: Any) -> None:
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            event_type = str(item.get("event_type") or item.get("type") or "").lower()
            if not event_type:
                continue
            token_id = str(item.get("asset_id") or item.get("market") or item.get("token_id") or "")
            if token_id not in self.pm_token_to_event:
                continue
            event = self.pm_token_to_event[token_id]
            self.pm_stream_messages += 1
            self.last_pm_stream_at = now_utc_iso()
            if self.args.persist_stream_events:
                append_jsonl(self.stream_dir / "polymarket_ws.jsonl", item)
            if event_type == "book":
                bids = item.get("bids") or item.get("buys") or []
                asks = item.get("asks") or item.get("sells") or []
                book = {"bids": bids, "asks": asks, "asset_id": token_id, "updated_at": self.last_pm_stream_at}
                self.pm_orderbooks[token_id] = book
                yes_asks = parse_polymarket_orderbook(book, "YES")
                yes_bids = []
                for raw_bid in bids:
                    if isinstance(raw_bid, dict):
                        price = safe_float(raw_bid.get("price")) or 0.0
                        size = safe_float(raw_bid.get("size")) or 0.0
                    elif isinstance(raw_bid, (list, tuple)) and len(raw_bid) >= 2:
                        price = safe_float(raw_bid[0]) or 0.0
                        size = safe_float(raw_bid[1]) or 0.0
                    else:
                        continue
                    if 0.0 < price < 1.0 and size > 0:
                        yes_bids.append((price, size))
                event.best_ask = yes_asks[0][0] if yes_asks else event.best_ask
                event.best_bid = max((price for price, _ in yes_bids), default=event.best_bid)
                event.updated_at = self.last_pm_stream_at
            elif event_type == "best_bid_ask":
                event.best_bid = safe_float(item.get("best_bid")) or safe_float(item.get("bid")) or event.best_bid
                event.best_ask = safe_float(item.get("best_ask")) or safe_float(item.get("ask")) or event.best_ask
                event.updated_at = self.last_pm_stream_at
            elif event_type in {"last_trade_price", "price_change"}:
                event.last_trade_price = safe_float(item.get("price")) or safe_float(item.get("last_trade_price")) or event.last_trade_price
                event.updated_at = self.last_pm_stream_at
            elif event_type == "market_resolved":
                event.active = False
                event.updated_at = self.last_pm_stream_at

    async def kalshi_ws_loop(self) -> None:
        if not self.kalshi_watch_tickers:
            return
        assert self.http_session is not None
        headers = self.kalshi_client.websocket_headers() or {}
        ws_url = os.environ.get("KALSHI_WS_URL", KalshiClient.WS_URL)
        while not self.stop_event.is_set():
            try:
                async with self.http_session.ws_connect(ws_url, heartbeat=25, autoping=True, headers=headers or None) as ws:
                    self.kalshi_stream_connected = True
                    self.kalshi_stream_mode = "websocket"
                    for batch in chunked(self.kalshi_watch_tickers, self.args.stream_subscribe_batch_size):
                        payload = {
                            "id": int(time.time() * 1000) % 1000000,
                            "cmd": "subscribe",
                            "params": {"channels": ["ticker"], "market_tickers": list(batch)},
                        }
                        await ws.send_json(payload)
                    async for message in ws:
                        if self.stop_event.is_set():
                            break
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(message.data)
                            except json.JSONDecodeError:
                                continue
                            await self.handle_kalshi_message(payload)
                        elif message.type in {aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED}:
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                self.kalshi_stream_mode = "polling"
                await asyncio.sleep(self.args.stream_reconnect_seconds)
            finally:
                self.kalshi_stream_connected = False

    async def handle_kalshi_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        message_type = str(payload.get("type") or payload.get("channel") or "").lower()
        if message_type in {"subscribed", "error", "subscription_updated", "welcome"}:
            return
        body = payload.get("msg") if isinstance(payload.get("msg"), dict) else payload
        ticker = str(body.get("market_ticker") or body.get("ticker") or "")
        if ticker not in self.kalshi_ticker_to_event:
            return
        event = self.kalshi_ticker_to_event[ticker]
        self.kalshi_stream_messages += 1
        self.last_kalshi_stream_at = now_utc_iso()
        if self.args.persist_stream_events:
            append_jsonl(self.stream_dir / "kalshi_ws.jsonl", payload)
        event.best_bid = safe_float(body.get("yes_bid_dollars")) or safe_float(body.get("yes_bid")) or event.best_bid
        event.best_ask = safe_float(body.get("yes_ask_dollars")) or safe_float(body.get("yes_ask")) or event.best_ask
        event.last_trade_price = safe_float(body.get("last_price_dollars")) or safe_float(body.get("price_dollars")) or safe_float(body.get("last_price")) or event.last_trade_price
        event.volume = safe_float(body.get("volume_fp")) or event.volume
        event.open_interest = safe_float(body.get("open_interest")) or event.open_interest
        event.updated_at = self.last_kalshi_stream_at

    async def kalshi_poll_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.kalshi_poll_seconds)
                return
            except asyncio.TimeoutError:
                pass
            if self.kalshi_stream_connected or not self.kalshi_watch_tickers:
                continue
            try:
                snapshots = await asyncio.to_thread(self.kalshi_client.fetch_market_snapshots, self.kalshi_watch_tickers)
            except Exception:
                continue
            self.kalshi_stream_mode = "polling"
            now = now_utc_iso()
            for snapshot in snapshots:
                if snapshot.event_id not in self.kalshi_events:
                    continue
                current = self.kalshi_events[snapshot.event_id]
                current.best_ask = snapshot.best_ask
                current.best_bid = snapshot.best_bid
                current.last_trade_price = snapshot.last_trade_price
                current.liquidity = snapshot.liquidity
                current.volume = snapshot.volume
                current.volume_24h = snapshot.volume_24h
                current.open_interest = snapshot.open_interest
                current.updated_at = now
            if snapshots:
                self.last_kalshi_stream_at = now

    async def analysis_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.bootstrap_lock.locked():
                await asyncio.sleep(1.0)
                continue
            await self.run_analysis_cycle("scheduled")
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.analysis_interval)
            except asyncio.TimeoutError:
                continue

    async def periodic_refresh_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.args.full_refresh_seconds <= 0:
                return
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.args.full_refresh_seconds)
                return
            except asyncio.TimeoutError:
                pass
            await self.bootstrap("periodic_refresh")
            await self.restart_stream_tasks()
            await self.run_analysis_cycle("post_refresh")

    async def verify_candidate(self, cycle_index: int, cycle_started_at: str, source: str, pair: MatchedPair,
                               ideal: ArbitrageOpportunity) -> OpportunityRecord:
        token_id = pair.pm_event.token_ids[0]
        pm_book = self.pm_orderbooks.get(token_id)
        if pm_book is None:
            pm_book = await asyncio.to_thread(self.pm_client.get_order_book, token_id)
        kalshi_book = await asyncio.to_thread(self.kalshi_client.get_order_book, pair.kalshi_event.event_id)

        pm_levels = parse_polymarket_orderbook(pm_book, ideal.pm_side) if pm_book else []
        kalshi_levels = parse_kalshi_orderbook(kalshi_book, ideal.kalshi_side) if kalshi_book else []

        pm_execution = execute_split_order(pm_levels, self.args.trade_amount, self.args.split_chunk, "polymarket", ideal.pm_side) if pm_levels else ExecutionReport("polymarket", ideal.pm_side, self.args.trade_amount, self.args.trade_amount, ideal.pm_price, 0.0, True, [], 0)
        kalshi_execution = execute_split_order(kalshi_levels, self.args.trade_amount, self.args.split_chunk, "kalshi", ideal.kalshi_side) if kalshi_levels else ExecutionReport("kalshi", ideal.kalshi_side, self.args.trade_amount, self.args.trade_amount, ideal.kalshi_price, 0.0, True, [], 0)

        pm_prices = self.pm_client.prices_from_event(pair.pm_event)
        kalshi_prices = self.kalshi_client.prices_from_event(pair.kalshi_event)
        verified = None
        opened_position = False
        if pm_prices and kalshi_prices and pm_execution.filled and kalshi_execution.filled:
            adjusted_pm = build_adjusted_prices(pm_prices, ideal.pm_side, pm_execution.avg_price or ideal.pm_price)
            adjusted_kalshi = build_adjusted_prices(kalshi_prices, ideal.kalshi_side, kalshi_execution.avg_price or ideal.kalshi_price)
            verified = detect_arbitrage(adjusted_pm, adjusted_kalshi, self.args.min_profit_threshold, self.args.pm_fee, self.args.kalshi_fee)
            if verified and self.args.auto_paper_trade and self.trader.can_open(pair.label, self.args.max_open_positions):
                position = self.trader.open_position(pair.label, pair.pm_event.event_id, pair.kalshi_event.event_id, ideal.pm_side, ideal.kalshi_side, pm_execution, kalshi_execution)
                self.append_decision_log({
                    "decision_type": "OPEN",
                    "cycle_index": cycle_index,
                    "cycle_started_at": cycle_started_at,
                    "source": source,
                    "label": pair.label,
                    "position_id": position.position_id,
                    "strategy": ideal.strategy,
                    "pm_event_id": pair.pm_event.event_id,
                    "kalshi_event_id": pair.kalshi_event.event_id,
                    "pm_side": ideal.pm_side,
                    "kalshi_side": ideal.kalshi_side,
                    "entry_cost": round(position.entry_cost, 6),
                    "pm_entry_price": round(position.pm_entry_price, 6),
                    "kalshi_entry_price": round(position.kalshi_entry_price, 6),
                    "ideal_total_cost": round(ideal.total_cost, 6),
                    "verified_total_cost": round(verified.total_cost, 6),
                    "verified_net_profit": round(verified.net_profit, 6),
                    "verified_roi_percent": round(verified.roi_percent, 6),
                })
                opened_position = True

        return OpportunityRecord(
            cycle_index=cycle_index,
            cycle_started_at=cycle_started_at,
            source=source,
            label=pair.label,
            similarity=round(pair.similarity, 6),
            pm_event_id=pair.pm_event.event_id,
            pm_title=pair.pm_event.title,
            kalshi_event_id=pair.kalshi_event.event_id,
            kalshi_title=pair.kalshi_event.title,
            strategy=ideal.strategy,
            pm_side=ideal.pm_side,
            kalshi_side=ideal.kalshi_side,
            ideal_total_cost=round(ideal.total_cost, 6),
            ideal_net_profit=round(ideal.net_profit, 6),
            ideal_roi_percent=round(ideal.roi_percent, 6),
            verified=verified is not None,
            verified_total_cost=round(verified.total_cost, 6) if verified else None,
            verified_net_profit=round(verified.net_profit, 6) if verified else None,
            verified_roi_percent=round(verified.roi_percent, 6) if verified else None,
            pm_execution_avg=round(pm_execution.avg_price, 6) if pm_execution.avg_price else None,
            kalshi_execution_avg=round(kalshi_execution.avg_price, 6) if kalshi_execution.avg_price else None,
            pm_slippage_percent=round(pm_execution.slippage_percent, 6) if pm_execution.avg_price else None,
            kalshi_slippage_percent=round(kalshi_execution.slippage_percent, 6) if kalshi_execution.avg_price else None,
            opened_position=opened_position,
        )

    async def run_analysis_cycle(self, source: str) -> None:
        if not self.watch_pairs:
            return
        cycle_started = now_utc()
        cycle_started_at = cycle_started.isoformat()
        self.cycle_index += 1

        ideal_candidates: List[Tuple[MatchedPair, ArbitrageOpportunity]] = []
        for pair in self.watch_pairs:
            if not pair.pm_event.active or not pair.kalshi_event.active:
                continue
            pm_prices = self.pm_client.prices_from_event(pair.pm_event)
            kalshi_prices = self.kalshi_client.prices_from_event(pair.kalshi_event)
            if pm_prices is None or kalshi_prices is None:
                continue
            ideal = detect_arbitrage(pm_prices, kalshi_prices, self.args.min_profit_threshold, self.args.pm_fee, self.args.kalshi_fee)
            if ideal is not None:
                ideal_candidates.append((pair, ideal))

        ideal_candidates.sort(key=lambda item: (item[1].roi_percent, item[0].similarity), reverse=True)
        verification_targets = ideal_candidates[:self.args.max_orderbook_checks_per_cycle]
        semaphore = asyncio.Semaphore(self.args.max_concurrent_verifications)

        async def limited_verify(pair: MatchedPair, ideal: ArbitrageOpportunity) -> OpportunityRecord:
            async with semaphore:
                return await self.verify_candidate(self.cycle_index, cycle_started_at, source, pair, ideal)

        records = await asyncio.gather(*(limited_verify(pair, ideal) for pair, ideal in verification_targets), return_exceptions=False)

        closed_now = 0
        for position in [position for position in self.trader.positions if position.status == "OPEN"]:
            pm_event = self.pm_events.get(position.pm_event_id)
            kalshi_event = self.kalshi_events.get(position.kalshi_event_id)
            if pm_event is None or kalshi_event is None:
                continue
            pm_prices = self.pm_client.prices_from_event(pm_event)
            kalshi_prices = self.kalshi_client.prices_from_event(kalshi_event)
            if pm_prices and kalshi_prices and self.trader.maybe_flatten(position, pm_prices, kalshi_prices):
                closed_now += 1
                self.append_decision_log({
                    "decision_type": "CLOSE",
                    "cycle_index": self.cycle_index,
                    "cycle_started_at": cycle_started_at,
                    "source": source,
                    "label": position.label,
                    "position_id": position.position_id,
                    "pm_event_id": position.pm_event_id,
                    "kalshi_event_id": position.kalshi_event_id,
                    "pm_side": position.pm_side,
                    "kalshi_side": position.kalshi_side,
                    "exit_value": round(position.exit_value or 0.0, 6),
                    "pnl": round(position.pnl or 0.0, 6),
                    "close_reason": position.close_reason,
                    "pm_close_value": round(close_value_for_side(pm_prices, position.pm_side), 6),
                    "kalshi_close_value": round(close_value_for_side(kalshi_prices, position.kalshi_side), 6),
                })

        records = sorted(records, key=lambda item: ((1 if item.verified else 0), (item.verified_roi_percent or item.ideal_roi_percent)), reverse=True)
        live_summary = {
            "mode": "continuous_realtime",
            "run_id": self.run_id,
            "cycle_index": self.cycle_index,
            "source": source,
            "cycle_started_at": cycle_started_at,
            "cycle_finished_at": now_utc_iso(),
            "polymarket_structured_market_count": len(self.pm_events),
            "kalshi_structured_market_count": len(self.kalshi_events),
            "matched_pairs": len(self.watch_pairs),
            "ideal_candidates": len(ideal_candidates),
            "verified_candidates": len([record for record in records if record.verified]),
            "open_positions": len([position for position in self.trader.positions if position.status == "OPEN"]),
            "closed_positions_total": len([position for position in self.trader.positions if position.status == "CLOSED"]),
            "closed_positions_this_cycle": closed_now,
            "polymarket_stream_connected": self.pm_stream_connected,
            "polymarket_stream_mode": self.pm_stream_mode,
            "kalshi_stream_connected": self.kalshi_stream_connected,
            "kalshi_stream_mode": self.kalshi_stream_mode,
            "pm_stream_messages": self.pm_stream_messages,
            "kalshi_stream_messages": self.kalshi_stream_messages,
            "last_pm_stream_at": self.last_pm_stream_at,
            "last_kalshi_stream_at": self.last_kalshi_stream_at,
            "trade_amount": self.args.trade_amount,
            "split_chunk": self.args.split_chunk,
            "analysis_interval_seconds": self.args.analysis_interval,
            "full_refresh_seconds": self.args.full_refresh_seconds,
            "hash_feature_dimension": self.matcher.feature_dimension,
            "vector_benchmark": self.vector_benchmark,
            "bootstrap_summary": self.bootstrap_summary,
            "top_live_candidates": [asdict(record) for record in records[:10]],
        }
        self.last_cycle_summary = live_summary
        write_json(self.analysis_dir / "live_summary.json", live_summary)
        write_json(self.analysis_dir / "current_opportunities.json", {"run_id": self.run_id, "generated_at": now_utc_iso(), "records": records})
        write_csv(self.analysis_dir / "live_opportunities.csv", (asdict(record) for record in records), list(OpportunityRecord.__annotations__.keys()))
        positions_summary = self.trader.summary("realtime_monitor")
        write_json(self.analysis_dir / "positions.json", {"run_id": self.run_id, **asdict(positions_summary)})
        append_jsonl(self.analysis_dir / "cycle_history.jsonl", {
            "run_id": self.run_id,
            "cycle_index": self.cycle_index,
            "cycle_started_at": cycle_started_at,
            "source": source,
            "ideal_candidates": len(ideal_candidates),
            "verified_candidates": len([record for record in records if record.verified]),
            "open_positions": len([position for position in self.trader.positions if position.status == "OPEN"]),
            "closed_positions_total": len([position for position in self.trader.positions if position.status == "CLOSED"]),
            "pm_stream_messages": self.pm_stream_messages,
            "kalshi_stream_messages": self.kalshi_stream_messages,
        })

    def build_report_tex(self) -> str:
        summary = self.last_cycle_summary or self.bootstrap_summary
        benchmark = self.vector_benchmark or {}
        paper_summary = self.trader.summary("realtime_monitor")
        return rf"""\documentclass[12pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\setmainfont{{Times New Roman}}
\title{{Version3 Realtime Cross-Platform Arbitrage Monitor}}
\author{{Codex}}
\date{{{now_utc_iso()}}}
\begin{{document}}
\maketitle
\section*{{Task}}
This version merges the guide modules into a single Python program, keeps the full-universe bootstrap, and then switches to continuous monitoring. Polymarket uses public websocket updates. Kalshi prefers websocket when credentials are available and otherwise falls back to slower ticker refresh.
\section*{{Architecture}}
\begin{{itemize}}
\item Bootstrap all active markets from Polymarket and Kalshi and persist raw snapshots.
\item Build hashed TF-IDF features and cross-platform matched pairs once per universe refresh cycle.
\item Subscribe to live Polymarket market updates and maintain a rolling state snapshot.
\item Recompute arbitrage candidates every {self.args.analysis_interval} seconds by default.
\item Verify the top candidates with slippage, split-order execution, and spread-flatten close logic.
\end{{itemize}}
\section*{{Current Runtime Summary}}
\begin{{longtable}}{{lp{{0.48\linewidth}}}}
\toprule
Metric & Value \\
\midrule
Polymarket structured markets & {summary.get("polymarket_structured_market_count", self.bootstrap_summary.get("polymarket_structured_market_count", 0))} \\
Kalshi structured markets & {summary.get("kalshi_structured_market_count", self.bootstrap_summary.get("kalshi_structured_market_count", 0))} \\
Matched watch pairs & {summary.get("matched_pairs", self.bootstrap_summary.get("watch_pairs", 0))} \\
Ideal candidates in latest cycle & {summary.get("ideal_candidates", 0)} \\
Verified candidates in latest cycle & {summary.get("verified_candidates", 0)} \\
Open positions & {summary.get("open_positions", 0)} \\
Closed positions total & {summary.get("closed_positions_total", 0)} \\
Polymarket stream connected & {summary.get("polymarket_stream_connected", False)} \\
Kalshi stream connected & {summary.get("kalshi_stream_connected", False)} \\
Polymarket stream mode & {summary.get("polymarket_stream_mode", "polling")} \\
Kalshi stream mode & {summary.get("kalshi_stream_mode", "polling")} \\
Analysis interval (seconds) & {self.args.analysis_interval} \\
Universe refresh interval (seconds) & {self.args.full_refresh_seconds} \\
Hash feature dimension & {self.matcher.feature_dimension} \\
\bottomrule
\end{{longtable}}
\section*{{Vector Benchmark}}
Documents benchmarked: {benchmark.get("documents", 0)}. Baseline elapsed: {benchmark.get("baseline_elapsed_seconds", "n/a")} seconds. Version2-style TF-IDF elapsed: {benchmark.get("version2_elapsed_seconds", "n/a")} seconds. Version3 hashed TF-IDF elapsed: {benchmark.get("version3_elapsed_seconds", "n/a")} seconds. Version3 speedup vs baseline: {benchmark.get("version3_speedup_vs_baseline_x", "n/a")}x. Version3 speedup vs version2: {benchmark.get("version3_speedup_vs_version2_x", "n/a")}x.
\section*{{Paper Trading State}}
Opened positions: {paper_summary.opened_positions}. Closed positions: {paper_summary.closed_positions}. Total PnL: {round(paper_summary.total_pnl, 6)}.
\section*{{Artifacts}}
\texttt{{raw\_data/}} stores bootstrap market snapshots and optional stream logs. \texttt{{structured\_data/}} stores cleaned markets, matched pairs, and the realtime watchlist. \texttt{{analysis/}} stores cycle summaries, opportunity tables, runtime positions, and the vector benchmark.
\end{{document}}
"""

    async def finalize(self) -> None:
        await self.run_analysis_cycle("shutdown")
        report_path = self.report_dir / "report.tex"
        report_path.write_text(self.build_report_tex(), encoding="utf-8")
        write_json(self.analysis_dir / "run_overview.json", {
            "generated_at": now_utc_iso(),
            "run_id": self.run_id,
            "main_script": "cross_platform_arbitrage_realtime.py",
            "raw_data_dir": "raw_data/",
            "structured_data_dir": "structured_data/",
            "analysis_dir": "analysis/",
            "report_tex": "report/report.tex",
            "live_summary_file": "analysis/live_summary.json",
            "live_opportunities_file": "analysis/live_opportunities.csv",
            "positions_file": "analysis/positions.json",
            "decision_log_file": "analysis/decision_log.jsonl",
            "vector_benchmark_file": "analysis/vector_benchmark.json",
            "bootstrap_summary_file": "analysis/bootstrap_summary.json",
            "watch_pairs": len(self.watch_pairs),
            "open_positions": len([position for position in self.trader.positions if position.status == "OPEN"]),
            "closed_positions": len([position for position in self.trader.positions if position.status == "CLOSED"]),
            "polymarket_stream_mode": self.pm_stream_mode,
            "kalshi_stream_mode": self.kalshi_stream_mode,
        })

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.http_session = session
            await self.bootstrap("initial")
            await self.run_analysis_cycle("bootstrap")
            if self.args.bootstrap_only:
                await self.finalize()
                return
            await self.start_stream_tasks()
            self.analysis_task = asyncio.create_task(self.analysis_loop(), name="analysis-loop")
            if self.args.full_refresh_seconds > 0:
                self.refresh_task = asyncio.create_task(self.periodic_refresh_loop(), name="refresh-loop")
            try:
                if self.args.runtime_seconds > 0:
                    await asyncio.sleep(self.args.runtime_seconds)
                    self.stop_event.set()
                else:
                    await self.stop_event.wait()
            finally:
                self.stop_event.set()
                await self.stop_stream_tasks()
                tasks = [task for task in [self.analysis_task, self.refresh_task] if task is not None]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await self.finalize()

