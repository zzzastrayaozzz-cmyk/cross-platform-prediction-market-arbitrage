## Layout

- `cross_platform_arbitrage_realtime.py`
  Thin entry script that preserves the original run command.
- `realtime_arb/constants.py`
  Category rules, stop words, and generic content token filters.
- `realtime_arb/models.py`
  Dataclasses for events, prices, executions, opportunities, and paper positions.
- `realtime_arb/utils.py`
  Time helpers, path helpers, JSON/JSONL/CSV writers, and parsing helpers.
- `realtime_arb/matching.py`
  Category classifier, vectorization logic, matcher, and benchmark pipeline.
- `realtime_arb/clients.py`
  Polymarket and Kalshi REST/WebSocket support helpers.
- `realtime_arb/execution.py`
  Orderbook parsing, slippage, split execution, arbitrage scoring, and paper trading.
- `realtime_arb/persistence.py`
  Raw-data loading and structured output persistence helpers.
- `realtime_arb/monitor.py`
  Continuous monitoring controller and cycle orchestration.
- `realtime_arb/cli.py`
  CLI arguments and startup entry logic.

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

The program writes runtime data to:

- `raw_data/`
- `structured_data/`
- `analysis/`
- `report/`
