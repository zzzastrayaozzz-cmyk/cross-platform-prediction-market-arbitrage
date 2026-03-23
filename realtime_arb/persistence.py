from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .models import Event, FetchBundle, MatchedPair
from .utils import ensure_dir, now_utc_iso, write_csv, write_json

def load_bundle_from_raw(raw_root: Path, platform: str, converter: Any) -> FetchBundle:
    payload = json.loads((raw_root / platform / "all_active_markets_raw.json").read_text(encoding="utf-8"))
    raw_items = payload.get("items", [])
    events = [event for event in (converter(item) for item in raw_items) if event is not None]
    return FetchBundle(
        platform=platform,
        fetched_at=payload.get("fetched_at", now_utc_iso()),
        page_count=int(payload.get("page_count", 0)),
        page_size=int(payload.get("page_size", 0)),
        raw_items=raw_items,
        events=events,
        request_summary=payload.get("request_summary", {}),
    )

def persist_fetch_bundle(root_dir: Path, bundle: FetchBundle) -> None:
    platform_raw_dir = ensure_dir(root_dir / "raw_data" / bundle.platform)
    structured_dir = ensure_dir(root_dir / "structured_data")
    write_json(platform_raw_dir / "all_active_markets_raw.json", {
        "platform": bundle.platform,
        "fetched_at": bundle.fetched_at,
        "page_count": bundle.page_count,
        "page_size": bundle.page_size,
        "request_summary": bundle.request_summary,
        "raw_market_count": len(bundle.raw_items),
        "items": bundle.raw_items,
    })
    write_json(platform_raw_dir / "fetch_metadata.json", {
        "platform": bundle.platform,
        "fetched_at": bundle.fetched_at,
        "page_count": bundle.page_count,
        "page_size": bundle.page_size,
        "raw_market_count": len(bundle.raw_items),
        "structured_market_count": len(bundle.events),
        "request_summary": bundle.request_summary,
    })
    sorted_events = sorted(bundle.events, key=lambda event: ((event.volume_24h or 0.0), (event.volume or 0.0), event.title.lower()), reverse=True)
    write_json(structured_dir / f"{bundle.platform}_active_markets.json", {
        "platform": bundle.platform,
        "fetched_at": bundle.fetched_at,
        "market_count": len(sorted_events),
        "markets": sorted_events,
    })
    write_csv(structured_dir / f"{bundle.platform}_active_markets.csv", ({
        "platform": event.platform,
        "event_id": event.event_id,
        "title": event.title,
        "resolution_date": event.resolution_date.isoformat() if event.resolution_date else "",
        "best_ask": event.best_ask,
        "best_bid": event.best_bid,
        "last_trade_price": event.last_trade_price,
        "volume": event.volume,
        "volume_24h": event.volume_24h,
        "liquidity": event.liquidity,
        "open_interest": event.open_interest,
    } for event in sorted_events), ["platform", "event_id", "title", "resolution_date", "best_ask", "best_bid", "last_trade_price", "volume", "volume_24h", "liquidity", "open_interest"])

def write_combined_structured_data(root_dir: Path, pm_events: Sequence[Event], kalshi_events: Sequence[Event]) -> None:
    combined = list(pm_events) + list(kalshi_events)
    combined.sort(key=lambda event: (event.platform, -float(event.volume_24h or 0.0), -float(event.volume or 0.0), event.title.lower()))
    write_json(root_dir / "structured_data" / "combined_active_markets.json", {
        "generated_at": now_utc_iso(),
        "market_count": len(combined),
        "markets": combined,
    })

def write_matches(root_dir: Path, matched_pairs: Sequence[MatchedPair]) -> None:
    write_json(root_dir / "structured_data" / "matched_pairs.json", {
        "generated_at": now_utc_iso(),
        "matched_pair_count": len(matched_pairs),
        "pairs": [{
            "label": pair.label,
            "similarity": round(pair.similarity, 6),
            "pm_event_id": pair.pm_event.event_id,
            "pm_title": pair.pm_event.title,
            "kalshi_event_id": pair.kalshi_event.event_id,
            "kalshi_title": pair.kalshi_event.title,
            "pm_resolution_date": pair.pm_event.resolution_date,
            "kalshi_resolution_date": pair.kalshi_event.resolution_date,
        } for pair in matched_pairs],
    })

