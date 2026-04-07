## A setup to trade multiple assets algorithmically

#### This is your “Start here” document to set up your system for trading multiple asset classes (Forex, Futures, Metals, and Crypto) algorithmically.
###### QuantInsti Webpage: https://www.quantinsti.com/

**Version 1.2.0**
**Last Updated**: 2026-04-07

-----

## Disclaimer

#### This file is documentation only and you should not use it for live trading without appropriate backtesting and tweaking of the strategy parameters.

## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this document except in compliance with the License.
- You may obtain a copy of the License in `LICENSE.md` at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.

## Table of contents
1. [Introduction](#introduction)
2. [Unique Multi-Asset Traits](#multi_asset_traits)
3. [Setup Notes](#setup_notes)
4. [Interactive Brokers setup requirements](#ib_requirements)
5. [Setup of variables](#variables_setup)
6. [Operational Helper Scripts](#helper_scripts)

<a id='introduction'></a>
## Introduction
This document describes a Python-based setup for algorithmic multi-asset trading using the Interactive Brokers API. The setup allows you to execute transactions across **Forex, Futures (MES), Spot Metals (XAUUSD), and Cryptocurrency** under one portfolio process.

The setup is organized so that engine code and user strategy code remain separate, which makes it easier to switch strategies or modify trading logic without rewriting the full live workflow.

- **Selectable Strategy Files**: The active strategy module is selected from `user_config/main.py` with `strategy_file`.
- **Default Strategy Module**: `strategy.py` contains the mean-reversion portfolio strategy with Markowitz allocation and exposes the interface expected by the engine.
- **Dual-Mode Execution**: The system automatically detects your universe. 
    - **Single-Asset Mode**: If you trade only one asset class (e.g., only Crypto), the system bypasses complex portfolio math and allocates 100% of your chosen leverage to that asset.
    - **Multi-Asset Mode**: If you trade two or more classes (e.g., Forex + Futures), the selected strategy module determines how portfolio capital is allocated across the universe.
- **Unified Portfolio Engine**: Unlike single-asset setups, this engine processes your entire universe (Forex, MES, XAU, Crypto) in one loop, which allows portfolio-level allocation and coordinated rebalancing across asset classes.
- **Exchange and Location Flexibility**: The setup is not restricted to one country or one exchange. You can run it from any location and configure it for assets available through Interactive Brokers, as long as the account has the required permissions, data subscriptions, and exchange access.
- **Selectable Strategy Interface**: The core engine is decoupled from the strategy logic. You can change strategies by updating `strategy_file` in `main.py` without editing the engine.
- **Asset-Specific Frequencies**: The engine is frequency-agnostic. It dynamically requests the required bar size for each asset (e.g., '1m', '15m', '4h', '1D') as defined by the selected strategy file. This allows high-frequency and low-frequency assets to coexist in the same loop.
- **PDF Reporting**: Automatically generates a multi-page performance report (`portfolio_report.pdf`) every trading period so you can review portfolio analytics from the live run.
- **Fallback Parameter Handling**: Includes a strategy-side defaults layer. If a parameter is missing or misconfigured in the selected strategy file, the system can still fall back to default values instead of stopping immediately.
- **Dynamic Market Hours**: Automatically detects the asset class and applies the correct session rules (e.g., 24/7 for Crypto, 23-hour for Futures, 24/5 for Forex).
- **Trading-Day Origin Control**: The live engine can start the trading day at a user-defined local time such as `18:00`, which is useful when your operational day does not match the broker's midnight boundary.
- **Carry-Protection Bridge**: Between the broker reopen and the configured `trading_day_origin`, the engine can restore broker-side stop-loss and take-profit protection for eligible live positions before the normal trading loop starts.

<a id='setup_notes'></a>
## Setup notes
1. **File Locations**: All your interactions happen in the `user_config` folder. Use `main.py` for connection/meta-settings, strategy selection, and optimization cadence. Use the selected strategy file for trading rules.
2. **First Run**: The first execution will download historical 1-minute data for all assets. Due to the multi-asset nature, this may take several hours. Subsequent runs are instantaneous updates.
3. **Data Storage**: Historical data is saved per-symbol in `data/historical/historical_{symbol}.csv`. Live trading results, broker state, and audit data are consolidated in `user_config/data/database.xlsx`.
4. **Portfolio Report**: Check `data/portfolio_report.pdf` every few hours for an updated view of your CAGR, Sharpe, Drawdown, and Equity curve.
5. **Weekend Positions**: While Forex closes on weekends, the Crypto sleeve of this setup will continue to operate 24/7 if enabled.
6. **Carry Window Behavior**: When the app starts after the broker reopen but before `trading_day_origin`, it runs a protection-only window. In that window, it restores or rebuilds risk-management orders for eligible live positions, then sleeps until the normal trading day begins.
7. **Risk-Management Order Side**: The stop-loss and take-profit orders must always be the opposite side of the protected live position. Long positions are protected with `SELL` RM orders, and short positions are protected with `BUY` RM orders.
8. **Futures Exception in the Carry Window**: Futures carry refresh is intentionally skipped during the reopen-to-origin bridge because the broker may accept the order but hold it until the next exchange session instead of resting it immediately.

<a id='helper_scripts'></a>
## Operational Helper Scripts

### reset_paper_trading_state.py
- Use `python reset_paper_trading_state.py` from the `user_config/` folder when you want to reset the paper-trading environment before a fresh paper run.
- The script runs in this order:
  1. It cancels open orders and closes open positions in the configured paper account.
  2. It deletes generated local outputs such as the live workbook, reports, logs, downloaded historical caches, base frames, and optimization artifacts.
  3. It removes local marker files used by the runtime loop.
- This script is intended for paper-trading cleanup. Review `main.py` carefully before using it so you confirm the account and connection settings.

### close_all_positions.py
- Use `python close_all_positions.py` from the `user_config/` folder when you only want to flatten the configured paper account without deleting local files.
- This helper now reuses the position-closing flow defined inside `reset_paper_trading_state.py`.

<a id='ib_requirements'></a>
## Interactive Brokers setup requirements
*(Standard IB setup - see Forex/Stock docs for full details)*
1. Install **TWS (Offline Stable)** and **IB API**.
2. Enable **ActiveX/Socket Clients** and disable **Read-Only API** in TWS Settings.
3. Ensure **Paper Trading** is configured with real-time market data sharing.
4. **Margin Requirements**: Trading MES (Futures) and Forex requires a **Margin Account**. Ensure your account permissions are updated.
5. **Crypto Permissions**: To trade Crypto, you must have an active **Paxos** integration enabled within your IBKR account settings.

<a id='variables_setup'></a>
## Setup of variables

### main.py (Meta-Parameters)
- **account**: Your IB account ID (e.g., 'DU1234567').
- **fx_pairs / futures_symbols / metals_symbols / crypto_symbols**: Define your tradable universe here.
- **timezone**: Your local timezone (e.g., 'America/New_York').
- **port**: 7497 for Paper, 7496 for Live.
- **trading_day_origin**: Local start time of your trading day (for example, `"18:00"` in `America/Lima`).
- **local_restart_hour**: Daily application restart hour in local time.
- **strict_targets_validation**: When `True`, the engine validates portfolio targets before sending live orders.
- **strategy_file**: The user strategy module to load, for example `"strategy.py"` or another module you add under `user_config/`.
- **optimization_frequency**: Controls when `strategy_parameter_optimization()` reruns inside the live loop. Supported values are `"daily"` and `"weekly"`.

### strategy.py (Strategy Parameters)
- **strategy.py**: The default mean-reversion plus Markowitz strategy.
- **Credentials**: Prefer environment variables for secrets such as `IBKR_ACCOUNT`, `SMTP_USERNAME`, `TO_EMAIL`, and `SMTP_APP_PASSWORD` instead of committing personal values.
- **strategy_parameter_optimization()**: The multi-asset engine now supports scheduled optimization using the cadence configured in `main.py`.
- **get_asset_frequency / get_asset_train_span**: Define the per-symbol data frequency and warm-up history.
- **ASSET_CLASS_RUNTIME_POLICY**: Controls session rules per asset class and is used by the engine to decide when an asset is tradable during the live loop.
