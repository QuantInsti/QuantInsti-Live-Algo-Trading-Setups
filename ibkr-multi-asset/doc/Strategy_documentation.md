# Multi-Asset Trend / HRP Strategy Documentation

Licensed under the QuantInsti Open License (QOL) v1.1.

## Overview

The bundled default strategy module is [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py).

It implements:

- moving-average trend following at the asset level
- daily parameter optimization
- hierarchical risk parity at the portfolio level
- one portfolio-level Kelly-style leverage multiplier capped by `fixed_max_leverage`
- ATR-based risk-management thresholds

This setup uses a trend-following / HRP design.

## Strategy Structure

### 1. Asset-Level Signal Model

Each asset is converted into an OHLC frame and resampled to the configured strategy frequency.

Core derived features:
- `ret`
- `fast_ma`
- `slow_ma`
- `trend_spread`
- `atr`
- `realized_vol`

Signal rule:
- `trend_spread > 0` => long
- `trend_spread < 0` => short
- if the asset is configured as long-only, negative trend is flattened to zero instead of short

### 2. Daily Optimization

The engine validates or reruns the optimization once per trading-day bucket.

Bucket definition:
- derived from `trading_day_origin` in `main.py`
- in the bundled configuration, `18:00` in `America/Lima`

Data rule:
- optimization only uses data strictly earlier than the active bucket start
- this keeps the optimization at `t-1` relative to the active trading day

Current optimization search grids:
- `fast_window_grid = [10, 20, 30, 40, 60]`
- `slow_window_grid = [80, 120, 160, 200, 260]`
- `atr_window_grid = [10, 14, 20, 30]`

Objective:
- validation-window Sharpe ratio

### 3. Portfolio Construction

Once each asset has optimized parameters, the strategy:

1. builds validation returns for each asset model
2. estimates cross-asset dependency from those returns
3. allocates capital with hierarchical risk parity

If SciPy clustering is unavailable, the strategy falls back to inverse-variance weights.

### 4. Portfolio-Level Leverage

The strategy then computes one portfolio-level Kelly-style leverage multiplier from the portfolio validation returns.

Rules:
- it is not per asset
- it is capped by `fixed_max_leverage`
- in the current `main.py`, `fixed_max_leverage = 1.0`

### 5. Risk Management

The strategy returns stop-loss and take-profit thresholds from ATR:

- stop-loss multiple: `2.0 x ATR`
- take-profit multiple: `3.0 x ATR`

These thresholds are direction-aware:
- long => stop below price, take-profit above price
- short => stop above price, take-profit below price

## Config Inputs Read From main.py

The strategy reads:

- `strategy_frequency`
- `strategy_optimization_lookback`
- `fixed_max_leverage`
- `long_only_symbols`
- `optimization_frequency`

Bundled defaults in `main.py`:
- frequency: `5min`
- lookback: `3000`
- max leverage: `1.0`
- long-only symbols: `["ETH"]`

## Long-Only Handling

The strategy treats crypto symbols in the built-in `CRYPTO_LONG_ONLY` set as long-only by default, and it also respects `long_only_symbols` from `main.py`.

In the bundled configuration:
- `ETH` is long-only

## Files Written By The Strategy

Optimization manifest:
- `user_config/data/models/strategy_optimization_manifest.json`

Feature inventory:
- `user_config/data/models/optimal_features_df.xlsx`

The engine validates the manifest against:
- universe
- optimization frequency
- optimization bucket
- strategy config hash

## Strategy Interface

The engine expects the active strategy module to expose these hooks:

- `get_asset_runtime_policy(symbol, asset_class=None)`
- `get_asset_frequency(symbol)`
- `get_asset_train_span(symbol)`
- `prepare_base_df(historical_data, data_frequency, ticker, train_span=...)`
- `strategy_parameter_optimization(...)`
- `validate_strategy_optimization(...)`
- `get_signal(app, ...)`
- `set_stop_loss_price(app)`
- `set_take_profit_price(app)`

This is why the setup remains strategy-agnostic even though the bundled default is a trend/HRP implementation.

## Operational Notes

- the strategy can fall back to default params if no history is available yet
- if the optimization manifest is missing or stale, the engine rebuilds it automatically
- portfolio leverage from the strategy is propagated into live sizing by the engine

## Constraint Modeling Note

This strategy does not yet encode every broker-side execution limit directly in the optimization layer.

Examples:
- crypto concentration caps
- minimum practical FX order sizes
- whole-unit XAUUSD behavior
- futures minimum contract size

These should be discovered first through [broker_constraint_report.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/broker_constraint_report.py) and then incorporated into backtest assumptions and, later, live sizing rules.
