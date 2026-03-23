from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Event:
    platform: str
    event_id: str
    title: str
    description: str = ""
    resolution_date: Optional[datetime] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    slug: Optional[str] = None
    token_ids: List[str] = field(default_factory=list)
    outcome_prices: Optional[Tuple[float, float]] = None
    best_ask: Optional[float] = None
    best_bid: Optional[float] = None
    last_trade_price: Optional[float] = None
    liquidity: Optional[float] = None
    volume: Optional[float] = None
    volume_24h: Optional[float] = None
    open_interest: Optional[float] = None
    categories: List[str] = field(default_factory=list)
    active: bool = True
    updated_at: Optional[str] = None

@dataclass
class FetchBundle:
    platform: str
    fetched_at: str
    page_count: int
    page_size: int
    raw_items: List[Dict[str, Any]]
    events: List[Event]
    request_summary: Dict[str, Any]

@dataclass
class MarketPrices:
    yes_mid: float
    no_mid: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    liquidity: float = 0.0
    last_price: Optional[float] = None

    def validate(self) -> bool:
        return 0.0 <= self.yes_mid <= 1.0 and 0.0 <= self.no_mid <= 1.0 and abs(self.yes_mid + self.no_mid - 1.0) < 0.05

@dataclass
class MatchedPair:
    pm_event: Event
    kalshi_event: Event
    similarity: float

    @property
    def label(self) -> str:
        return f"{self.pm_event.event_id}__{self.kalshi_event.event_id}"

@dataclass
class ArbitrageOpportunity:
    strategy: str
    pm_side: str
    kalshi_side: str
    pm_price: float
    kalshi_price: float
    total_cost: float
    gross_profit: float
    fees: float
    net_profit: float
    roi_percent: float

@dataclass
class SlippageInfo:
    avg_price: float
    slippage_percent: float
    filled: bool
    filled_notional: float
    filled_contracts: float

@dataclass
class FillLine:
    price: float
    contracts: float
    notional: float

@dataclass
class ExecutionReport:
    venue: str
    side: str
    target_notional: float
    filled_notional: float
    avg_price: float
    slippage_percent: float
    filled: bool
    fills: List[FillLine] = field(default_factory=list)
    chunks: int = 1

@dataclass
class OpportunityRecord:
    cycle_index: int
    cycle_started_at: str
    source: str
    label: str
    similarity: float
    pm_event_id: str
    pm_title: str
    kalshi_event_id: str
    kalshi_title: str
    strategy: str
    pm_side: str
    kalshi_side: str
    ideal_total_cost: float
    ideal_net_profit: float
    ideal_roi_percent: float
    verified: bool
    verified_total_cost: Optional[float]
    verified_net_profit: Optional[float]
    verified_roi_percent: Optional[float]
    pm_execution_avg: Optional[float]
    kalshi_execution_avg: Optional[float]
    pm_slippage_percent: Optional[float]
    kalshi_slippage_percent: Optional[float]
    opened_position: bool

@dataclass
class PaperPosition:
    position_id: str
    label: str
    pm_event_id: str
    kalshi_event_id: str
    pm_side: str
    kalshi_side: str
    entry_cost: float
    pm_entry_price: float
    kalshi_entry_price: float
    opened_at: str
    status: str = "OPEN"
    exit_value: Optional[float] = None
    pnl: Optional[float] = None
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None

@dataclass
class PaperTradeSummary:
    scenario: str
    opened_positions: int
    closed_positions: int
    total_pnl: float
    positions: List[PaperPosition] = field(default_factory=list)

