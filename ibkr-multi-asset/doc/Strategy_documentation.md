# Multi-Asset Mean-Reversion Markowitz Strategy Documentation

#### This document details the default `strategy.py` file currently used in this setup.

###### QuantInsti Webpage: [https://www.quantinsti.com/](https://www.quantinsti.com/)

**Version 1.2.0**
**Last Updated**: 2026-04-07

-----

# Disclaimer

#### This file is documentation only and you should not use it for live trading without appropriate backtesting and parameter adjustments.

## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this document except in compliance with the License.
- You may obtain a copy of the License in `LICENSE.md` at the repository root or at [https://www.quantinsti.com/](https://www.quantinsti.com/)
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.

-----

## Table of Contents
1. [Introduction](#introduction)
2. [Portfolio Architecture](#portfolio-architecture)
3. [Asset Sleeve Logic](#asset-logic)
    - 3.1 [MES (Futures)](#mes-logic)
    - 3.2 [XAUUSD (Gold)](#xau-logic)
    - 3.3 [Cryptocurrency](#crypto-logic)
    - 3.4 [Forex (FX)](#fx-logic)
4. [Risk Management & Scaling](#risk-management)
5. [User Interface (strategy.py)](#user-interface)

<a id='introduction'></a>
## 1. Introduction
The current `strategy.py` module implements a **mean-reversion-based multi-asset strategy** with **Markowitz portfolio allocation**. Each asset gets its own optimized mean-reversion parameters, and the portfolio layer allocates capital using the validation-window strategy returns produced by those asset models.

<a id='portfolio-architecture'></a>
## 2. Portfolio Architecture & Construction
The core engine follows a **bottom-up asset modeling plus top-down allocation** approach:
1.  **Per-Asset Mean-Reversion Models**: Each symbol is optimized independently over a parameter grid that includes lookback, entry Z-score, exit Z-score, and target volatility.
2.  **Validation-Based Portfolio Inputs**: The strategy uses the validation-window returns from each optimized asset model to estimate the expected return vector and covariance matrix.
3.  **Markowitz Allocation**: Portfolio weights are derived from the resulting mean and covariance estimates with non-negative weights and clipping controls.
4.  **Cash Buffer**: A configurable cash buffer is preserved through the portfolio layer.

<a id='asset-logic'></a>
## 3. Asset Sleeve Logic

<a id='mes-logic'></a>
### 3.1 MES (Futures)
- **Timeframe (Default)**: Determined by `get_asset_frequency("MES")`.
- **Core Signal**: Mean reversion using rolling Z-scores of the price relative to its moving average.
- **Parameter Optimization**: The optimizer searches over lookback, entry threshold, exit threshold, and target volatility.

<a id='xau-logic'></a>
### 3.2 XAUUSD (Gold)
- **Timeframe (Default)**: Determined by `get_asset_frequency("XAUUSD")`.
- **Core Signal**: Mean reversion using Z-score entries and exits.
- **Portfolio Role**: Gold is treated like any other asset in the Markowitz input set rather than through a dedicated defensive sleeve.

<a id='crypto-logic'></a>
### 3.3 Cryptocurrency (BTC, ETH, etc.)
- **Timeframe (Default)**: Determined by `get_asset_frequency(symbol)`.
- **Signal**: Mean reversion with long-only constraints for the bundled crypto symbols.
- **Execution Constraint**: Because the execution layer prevents unsupported short crypto exposure, the bundled strategy keeps crypto assets long-only by default.

<a id='fx-logic'></a>
### 3.4 Forex (FX)
- **Timeframe (Default)**: Determined by `get_asset_frequency(symbol)`.
- **Signal**: Mean reversion using rolling Z-scores with configurable long and short entries.

<a id='risk-management'></a>
## 4. Risk Management & Scaling
- **Dynamic ATR Stops**: Every asset calculates a rolling ATR. Stop-loss and take-profit targets are set from the latest ATR-based distance.
- **Volatility Scaling**: Each asset target is scaled by its optimized target volatility and realized strategy volatility.
- **Markowitz Weights**: Final leverage is the product of the asset signal and the optimized portfolio weight.
- **Broker-Side RM Direction Rule**: The stop-loss and take-profit orders are always sent on the opposite side of the protected live position.
  - Long position => `SELL` RM orders.
  - Short position => `BUY` RM orders.
- **Carry-Protection Rebuild**: If the app starts after broker reopen but before the configured `trading_day_origin`, the engine can rebuild missing broker-side stop-loss and take-profit orders from persisted state or from fresh market-derived thresholds.
- **Fresh Threshold Fallback**: If the previous stop-loss / take-profit levels are missing or invalid against the current market, the engine recomputes both thresholds from the current live market instead of skipping protection.
- **Partial/Duplicate Cleanup**: If the broker has an incomplete or duplicated RM set for an asset, the live engine cancels the stale set and rebuilds one clean pair.
- **Futures Carry Exception**: Futures are excluded from the reopen-to-origin carry refresh because orders accepted in that window may not rest at the exchange immediately.

<a id='user-interface'></a>
## 5. User Interface (strategy.py)
The current `strategy.py` file exposes the live interface expected by the multi-asset engine and contains the mean-reversion plus Markowitz implementation. If you want to add another model, create a new module under `user_config/` that implements the same hooks and point `strategy_file` to it from `main.py`.

Key live-execution hooks exposed by `strategy.py`:
- `get_asset_frequency(symbol)`
- `get_asset_train_span(symbol)`
- `get_asset_runtime_policy(symbol, asset_class=None)`
- `get_signal(app, ...)`
- `strategy_parameter_optimization(symbol_specs=None)`
- `validate_strategy_optimization(symbol_specs=None, optimization_result=None)`

Related runtime controls in `main.py`:
- `strategy_file`: selects which strategy module under `user_config/` is active.
- `optimization_frequency`: selects whether the live engine reruns the optimization schedule daily or weekly.
