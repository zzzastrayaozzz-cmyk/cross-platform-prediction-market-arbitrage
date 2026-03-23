from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import ArbitrageOpportunity, ExecutionReport, FillLine, MarketPrices, PaperPosition, PaperTradeSummary, SlippageInfo
from .utils import now_utc_iso, safe_float

def parse_polymarket_orderbook(data: Dict[str, Any], side: str) -> List[Tuple[float, float]]:
    levels: List[Tuple[float, float]] = []
    if side == "YES":
        asks = data.get("asks", []) or data.get("sells", [])
        for ask in asks:
            if isinstance(ask, dict):
                price = safe_float(ask.get("price")) or 0.0
                size = safe_float(ask.get("size")) or 0.0
            elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                price = safe_float(ask[0]) or 0.0
                size = safe_float(ask[1]) or 0.0
            else:
                continue
            if 0.0 < price < 1.0 and size > 0:
                levels.append((price, size))
    else:
        bids = data.get("bids", []) or data.get("buys", [])
        for bid in bids:
            if isinstance(bid, dict):
                yes_bid = safe_float(bid.get("price")) or 0.0
                size = safe_float(bid.get("size")) or 0.0
            elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                yes_bid = safe_float(bid[0]) or 0.0
                size = safe_float(bid[1]) or 0.0
            else:
                continue
            no_ask = 1.0 - yes_bid
            if 0.0 < no_ask < 1.0 and size > 0:
                levels.append((no_ask, size))
    levels.sort(key=lambda item: item[0])
    return levels

def parse_kalshi_orderbook(data: Dict[str, Any], side: str) -> List[Tuple[float, float]]:
    orderbook = data.get("orderbook_fp", {})
    levels: List[Tuple[float, float]] = []
    if side == "YES":
        for price, size in orderbook.get("no_dollars", []):
            ask_price = 1.0 - float(price)
            if 0.0 < ask_price < 1.0 and float(size) > 0:
                levels.append((ask_price, float(size)))
    else:
        for price, size in orderbook.get("yes_dollars", []):
            ask_price = 1.0 - float(price)
            if 0.0 < ask_price < 1.0 and float(size) > 0:
                levels.append((ask_price, float(size)))
    levels.sort(key=lambda item: item[0])
    return levels

def calculate_slippage(levels: Sequence[Tuple[float, float]], notional: float) -> SlippageInfo:
    if not levels or notional <= 0:
        return SlippageInfo(0.0, 0.0, False, 0.0, 0.0)
    best_price = levels[0][0]
    remaining_notional = notional
    total_cost = 0.0
    total_contracts = 0.0
    for price, size in levels:
        level_notional = price * size
        if remaining_notional >= level_notional:
            total_cost += level_notional
            total_contracts += size
            remaining_notional -= level_notional
        else:
            buy_size = remaining_notional / price
            total_cost += remaining_notional
            total_contracts += buy_size
            remaining_notional = 0.0
            break
    avg_price = total_cost / total_contracts if total_contracts > 1e-12 else 0.0
    slippage_percent = ((avg_price - best_price) / best_price) * 100.0 if best_price > 0 else 0.0
    return SlippageInfo(avg_price, slippage_percent, remaining_notional <= 1e-9, notional - remaining_notional, total_contracts)

def execute_split_order(levels: Sequence[Tuple[float, float]], total_notional: float, chunk_notional: float,
                        venue: str, side: str) -> ExecutionReport:
    if not levels:
        return ExecutionReport(venue, side, total_notional, 0.0, 0.0, 0.0, False, [], 0)
    remaining_levels = [[float(price), float(size)] for price, size in levels]
    remaining_notional = total_notional
    all_fills: List[FillLine] = []
    total_contracts = 0.0
    total_cost = 0.0
    chunks = 0
    best_price = levels[0][0]
    while remaining_notional > 1e-9 and remaining_levels:
        chunks += 1
        current_chunk = min(chunk_notional, remaining_notional)
        chunk_remaining = current_chunk
        for level in remaining_levels:
            price, size = level
            if chunk_remaining <= 1e-9:
                break
            level_notional = price * size
            if chunk_remaining >= level_notional:
                all_fills.append(FillLine(price, size, level_notional))
                total_contracts += size
                total_cost += level_notional
                chunk_remaining -= level_notional
                level[1] = 0.0
            else:
                buy_size = chunk_remaining / price
                all_fills.append(FillLine(price, buy_size, chunk_remaining))
                total_contracts += buy_size
                total_cost += chunk_remaining
                level[1] -= buy_size
                chunk_remaining = 0.0
                break
        filled_now = current_chunk - chunk_remaining
        remaining_notional -= filled_now
        remaining_levels = [level for level in remaining_levels if level[1] > 1e-9]
        if filled_now <= 1e-9:
            break
    avg_price = total_cost / total_contracts if total_contracts > 1e-12 else 0.0
    slip = ((avg_price - best_price) / best_price * 100.0) if best_price > 0 and avg_price > 0 else 0.0
    return ExecutionReport(venue, side, total_notional, total_notional - remaining_notional, avg_price, slip, remaining_notional <= 1e-9, all_fills, chunks)

def detect_arbitrage(pm_prices: MarketPrices, kalshi_prices: MarketPrices, min_profit_threshold: float,
                     pm_fee: float = 0.01, kalshi_fee: float = 0.01) -> Optional[ArbitrageOpportunity]:
    if not pm_prices.validate() or not kalshi_prices.validate():
        return None
    total_fees = pm_fee + kalshi_fee
    candidates = [
        ("Buy Yes on Kalshi + Buy No on Polymarket", "NO", "YES", pm_prices.no_ask, kalshi_prices.yes_ask),
        ("Buy No on Kalshi + Buy Yes on Polymarket", "YES", "NO", pm_prices.yes_ask, kalshi_prices.no_ask),
    ]
    best: Optional[ArbitrageOpportunity] = None
    for strategy_name, pm_side, kalshi_side, pm_price, kalshi_price in candidates:
        total_cost = pm_price + kalshi_price
        gross_profit = 1.0 - total_cost
        if gross_profit <= total_fees + min_profit_threshold:
            continue
        net_profit = gross_profit - total_fees
        roi_percent = (net_profit / total_cost) * 100.0 if total_cost > 0 else 0.0
        candidate = ArbitrageOpportunity(strategy_name, pm_side, kalshi_side, pm_price, kalshi_price, total_cost, gross_profit, total_fees, net_profit, roi_percent)
        if best is None or candidate.roi_percent > best.roi_percent:
            best = candidate
    return best

def build_adjusted_prices(base_prices: MarketPrices, side: str, avg_price: float) -> MarketPrices:
    if side == "YES":
        return MarketPrices(avg_price, 1.0 - avg_price, avg_price, base_prices.yes_bid, base_prices.no_ask, base_prices.no_bid, base_prices.liquidity, base_prices.last_price)
    return MarketPrices(1.0 - avg_price, avg_price, base_prices.yes_ask, base_prices.yes_bid, avg_price, base_prices.no_bid, base_prices.liquidity, base_prices.last_price)

def close_value_for_side(prices: MarketPrices, side: str) -> float:
    return prices.yes_bid if side == "YES" else prices.no_bid

class PaperTrader:
    def __init__(self, take_profit: float = 0.02, flatten_spread: float = 0.03, reopen_cooldown: float = 300.0) -> None:
        self.take_profit = take_profit
        self.flatten_spread = flatten_spread
        self.reopen_cooldown = reopen_cooldown
        self.positions: List[PaperPosition] = []
        self.last_closed_at: Dict[str, float] = {}

    def find_open(self, label: str) -> Optional[PaperPosition]:
        for position in reversed(self.positions):
            if position.label == label and position.status == "OPEN":
                return position
        return None

    def can_open(self, label: str, max_open_positions: int) -> bool:
        if self.find_open(label) is not None:
            return False
        if len([position for position in self.positions if position.status == "OPEN"]) >= max_open_positions:
            return False
        if label in self.last_closed_at and time.time() - self.last_closed_at[label] < self.reopen_cooldown:
            return False
        return True

    def open_position(self, label: str, pm_event_id: str, kalshi_event_id: str, pm_side: str, kalshi_side: str,
                      pm_report: ExecutionReport, kalshi_report: ExecutionReport) -> PaperPosition:
        position = PaperPosition(
            position_id=f"paper-{len(self.positions) + 1}",
            label=label,
            pm_event_id=pm_event_id,
            kalshi_event_id=kalshi_event_id,
            pm_side=pm_side,
            kalshi_side=kalshi_side,
            entry_cost=pm_report.avg_price + kalshi_report.avg_price,
            pm_entry_price=pm_report.avg_price,
            kalshi_entry_price=kalshi_report.avg_price,
            opened_at=now_utc_iso(),
        )
        self.positions.append(position)
        return position

    def maybe_flatten(self, position: PaperPosition, pm_prices: MarketPrices, kalshi_prices: MarketPrices) -> bool:
        if position.status != "OPEN":
            return False
        exit_value = close_value_for_side(pm_prices, position.pm_side) + close_value_for_side(kalshi_prices, position.kalshi_side)
        pnl = exit_value - position.entry_cost
        spread_gap = abs(1.0 - exit_value)
        if pnl >= self.take_profit or spread_gap <= self.flatten_spread:
            position.status = "CLOSED"
            position.exit_value = exit_value
            position.pnl = pnl
            position.closed_at = now_utc_iso()
            position.close_reason = "spread_flatten_close" if spread_gap <= self.flatten_spread else "take_profit_close"
            self.last_closed_at[position.label] = time.time()
            return True
        return False

    def summary(self, scenario: str) -> PaperTradeSummary:
        closed = [position for position in self.positions if position.status == "CLOSED"]
        return PaperTradeSummary(scenario, len(self.positions), len(closed), sum(position.pnl or 0.0 for position in closed), self.positions.copy())

