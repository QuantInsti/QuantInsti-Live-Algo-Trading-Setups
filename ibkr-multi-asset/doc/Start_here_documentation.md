## Start Here: IBKR Multi-Asset Trading Setup

Licensed under the QuantInsti Open License (QOL) v1.1.

## Purpose

This setup is for live portfolio trading with Interactive Brokers across multiple asset classes from one runtime:

- FX
- futures
- spot metals
- crypto
- optionally stocks

The engine and the strategy layer are separated. You change runtime and broker settings in `user_config/main.py` and keep trading logic in the selected strategy module.

## Default Runtime

The bundled configuration in [main.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py) is a validation profile:

- universe:
  - `EURUSD`
  - `MES`
  - `XAUUSD`
  - `ETH`
- `strategy_file = "strategies/strategy.py"`
- `optimization_frequency = "daily"`
- `strategy_frequency = "5min"`
- `strategy_optimization_lookback = 3000`
- `fixed_max_leverage = 1.0`
- `portfolio_parallel_order_submission = False`
- `trading_day_origin = "18:00"`
- timezone: `America/Lima`

## Default Strategy

The bundled strategy in [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py) is:

- moving-average trend following per asset
- daily optimization of:
  - fast MA window
  - slow MA window
  - ATR window
- hierarchical risk parity across the full universe
- one portfolio-level Kelly-style leverage multiplier capped by `fixed_max_leverage`
- ATR-based stop-loss and take-profit thresholds
- long-only crypto handling controlled by `long_only_symbols`

The strategy is replaceable. Future users can point `strategy_file` to another module that keeps the same interface.

## What Happens In A Trading Cycle

1. The engine computes the current trading-day bucket using `trading_day_origin`.
2. It validates or reruns the strategy optimization for that bucket.
3. It refreshes historical data and base frames for the active universe.
4. It collects shared account, position, order, and execution snapshots.
5. It computes portfolio targets once for the full universe.
6. It sends orders:
   - sequentially on one shared app when `portfolio_parallel_order_submission = False`
   - or in parallel with isolated apps when `True`
7. It saves live state and generates the PDF report.

## Files You Will Edit Most Often

- [main.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py)
  - account settings
  - market universe
  - runtime cadence
  - broker metadata
  - strategy selection
- [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py)
  - signal logic
  - portfolio allocation logic
  - optimization logic
  - stop-loss / take-profit logic

## Persistence Layout

Live workbook:
- `user_config/data/database.xlsx`

Sheets used:
- `open_orders`
- `orders_status`
- `executions`
- `commissions`
- `positions`
- `portfolio_snapshots`
- `cash_balance`
- `app_time_spent`
- `periods_traded`
- `account_updates`
- `contract_details`

Additional live state:
- `user_config/data/strategy_state.json`

Log directory:
- `user_config/data/log/`

## First Run Notes

- the first live run may need to build historical caches under `user_config/data/historical/`
- if optimization has no usable history yet, the strategy falls back to default parameters until history exists
- the first few paper runs should be treated as validation, not deployment

## Broker Constraint Discovery Before Backtesting

There is a standalone broker-constraint inspection utility:

- [broker_constraint_report.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/broker_constraint_report.py)

Use it to generate:
- `broker_constraints_report.json`
- `broker_constraints_report.pdf`

This is useful before writing a backtest because some execution limitations are broker-specific:

- crypto concentration caps
- minimum increments
- quantity-step rules
- futures roll behavior
- venue-specific stop-order support

The script is intentionally independent of `main.py`.

## Running The Setup

From the project root:

```bash
python -m build
python -m pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
python user_config/main.py
```

## IBKR Requirements

- TWS or IB Gateway with API access enabled
- correct market-data subscriptions
- correct exchange permissions for the selected universe
- correct crypto permissions if using PAXOS crypto symbols
- enough account equity and margin for the configured products

## Operational Helpers

- `user_config/reset_paper_trading_state.py`
  - flattens the paper account and removes generated local state
- `user_config/close_all_positions.py`
  - flattens the configured paper account without deleting local files
- `user_config/broker_constraint_report.py`
  - generates a standalone broker-constraint JSON/PDF report

## Recommendation

Use the current four-asset validation setup until:

- target weights look sensible
- broker constraints are understood
- paper fills and stop behavior are stable
- the backtest constraint model is defined
