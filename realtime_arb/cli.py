from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .monitor import StreamingArbitrageMonitor
from .utils import ensure_dir


def build_readme_text() -> str:
    return """# Version4 Realtime Monitor

`version4` keeps the realtime monitoring behavior from `version3`, but splits the old
single-file implementation into a clearer package layout that is easier to maintain
and publish to GitHub.

## Layout

- `cross_platform_arbitrage_realtime.py`
  - Thin entry script. Run this file the same way as `version3`.
- `realtime_arb/constants.py`
  - Category rules, stop words, generic content token filters.
- `realtime_arb/models.py`
  - Dataclasses for events, prices, arbitrage records, executions, and paper positions.
- `realtime_arb/utils.py`
  - Time helpers, path helpers, JSONL/CSV writers, and common parsing utilities.
- `realtime_arb/matching.py`
  - Category classifier, baseline vectorizer, hashing TF-IDF matcher, and benchmark logic.
- `realtime_arb/clients.py`
  - Polymarket and Kalshi HTTP clients plus market-to-price conversion logic.
- `realtime_arb/execution.py`
  - Orderbook parsing, slippage, split execution, arbitrage detection, and paper trading logic.
- `realtime_arb/persistence.py`
  - Raw-data loading and structured output persistence helpers.
- `realtime_arb/monitor.py`
  - Continuous monitoring controller, websocket/polling loops, cycle analysis, report output.
- `realtime_arb/cli.py`
  - CLI args, README generation, and async startup.

## Run

```bash
python cross_platform_arbitrage_realtime.py
```

Useful examples:

```bash
python cross_platform_arbitrage_realtime.py --runtime-seconds 120
python cross_platform_arbitrage_realtime.py --analysis-interval 30 --full-refresh-seconds 900
python cross_platform_arbitrage_realtime.py --reuse-existing-raw
python cross_platform_arbitrage_realtime.py --bootstrap-from D:/path/to/version2/raw_data
```

## Default cadence

- `analysis_interval = 30`
- `full_refresh_seconds = 900`
- `polymarket_poll_seconds = 45`
- `kalshi_poll_seconds = 60`

## Runtime artifacts

These directories are created during runtime:

- `raw_data/`
- `structured_data/`
- `analysis/`
- `report/`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Version 4 modular realtime Kalshi / Polymarket arbitrage monitor")
    parser.add_argument("--polymarket-page-size", type=int, default=500)
    parser.add_argument("--kalshi-page-size", type=int, default=1000)
    parser.add_argument("--polymarket-max-pages", type=int, default=0)
    parser.add_argument("--kalshi-max-pages", type=int, default=0)
    parser.add_argument("--reuse-existing-raw", action="store_true")
    parser.add_argument("--bootstrap-from", type=str, default="")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--runtime-seconds", type=float, default=0.0)
    parser.add_argument("--analysis-interval", type=float, default=30.0)
    parser.add_argument("--full-refresh-seconds", type=float, default=900.0)
    parser.add_argument("--polymarket-poll-seconds", type=float, default=45.0)
    parser.add_argument("--polymarket-poll-batch-size", type=int, default=50)
    parser.add_argument("--kalshi-poll-seconds", type=float, default=60.0)
    parser.add_argument("--stream-reconnect-seconds", type=float, default=5.0)
    parser.add_argument("--stream-subscribe-batch-size", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--similarity-chunk-size", type=int, default=384)
    parser.add_argument("--hash-features", type=int, default=65536)
    parser.add_argument("--benchmark-docs", type=int, default=12000)
    parser.add_argument("--watchlist-limit", type=int, default=1000)
    parser.add_argument("--min-profit-threshold", type=float, default=0.02)
    parser.add_argument("--pm-fee", type=float, default=0.01)
    parser.add_argument("--kalshi-fee", type=float, default=0.01)
    parser.add_argument("--trade-amount", type=float, default=100.0)
    parser.add_argument("--split-chunk", type=float, default=25.0)
    parser.add_argument("--max-orderbook-checks-per-cycle", type=int, default=12)
    parser.add_argument("--max-concurrent-verifications", type=int, default=4)
    parser.add_argument("--no-auto-paper-trade", dest="auto_paper_trade", action="store_false")
    parser.set_defaults(auto_paper_trade=True)
    parser.add_argument("--max-open-positions", type=int, default=3)
    parser.add_argument("--take-profit", type=float, default=0.02)
    parser.add_argument("--flatten-spread", type=float, default=0.03)
    parser.add_argument("--reopen-cooldown", type=float, default=300.0)
    parser.add_argument("--persist-stream-events", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    root_dir = ensure_dir(Path(__file__).resolve().parent.parent)
    ensure_dir(root_dir / "raw_data")
    ensure_dir(root_dir / "structured_data")
    ensure_dir(root_dir / "analysis")
    ensure_dir(root_dir / "report")
    (root_dir / "README.md").write_text(build_readme_text(), encoding="utf-8")
    monitor = StreamingArbitrageMonitor(args, root_dir)
    await monitor.run()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
