## Developer's Guidelines for the Multi-Asset Trading Setup

#### This document provides instructions for modifying the source code of the multi-asset trading application, rebuilding the package, and running your modified setup.
###### QuantInsti Webpage: https://www.quantinsti.com/

**Version 1.2.0**
**Last Updated**: 2026-04-07

-----

## Disclaimer

#### This document provides instructions for you if you want to modify the source code of the multi-asset trading application, rebuild the package, and run your modified setup.

## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this document except in compliance with the License.
- You may obtain a copy of the License in `LICENSE.md` at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.

## Table of contents
1.  [Introduction](#introduction)
2.  [Step-by-Step Guide to Modifying and Running](#guide)
    1. [Step 1: Navigate to the Project Root](#navigate)
    2. [Step 2: Make Your Code Modifications](#modify)
    3. [Step 3: Build the Package](#build)
    4. [Step 4: Reinstall the Modified Package](#reinstall)
    5. [Step 5: Run the Trading Setup](#run) 
3.  [Developer Notes](#traits)
4.  [Helper Scripts](#helper_scripts)

<a id='introduction'><a>
## 1. Introduction

The multi-asset setup is a Python-based trading application for live trading across multiple asset classes with Interactive Brokers. This guide enables you to:
1. Build the setup wheel once you tweak any part of the source code.
2. Force the reinstallation of the setup package in the Python environment.
3. Run your setup with the new changes made.
4. Understand the carry-protection bridge that runs between broker reopen and the configured `trading_day_origin`.

<a id='guide'><a>
## 2. Step-by-Step Guide to Modifying and Running

<a id='navigate'><a>
### Step 1: Navigate to the Project Root
You should navigate to the root directory of the multi-asset project. This is the directory that contains the `src/` folder. You can check any of the Python files located in the `src/ibkr_multi_asset/` folder to modify core application logic.

<a id='modify'><a>
### Step 2: Make Your Code Modifications

1.  **Identify the files to modify**:
    * **Strategy logic** (signals, feature engineering, rebalancing): Edit the strategy file selected in `user_config/main.py` through `strategy_file`. The default implementation lives in `user_config/strategy.py`.
    * **Core application logic** (engine, data handling, IB interactions): You should edit files within `src/ibkr_multi_asset/`.
    * **Reporting logic**: To modify the PDF reports, edit `src/ibkr_multi_asset/report_generator.py`.
    * **Run parameters** (connection settings, strategy selection, optimization cadence): You can modify `user_config/main.py`.
    * **Carry-protection logic** (reopen-to-origin RM restoration): The main entrypoints are `src/ibkr_multi_asset/engine.py` and `src/ibkr_multi_asset/setup_functions.py`.

2.  **Edit the Python files**: Use your preferred IDE. For example, you might change the mean-reversion plus Markowitz logic in `user_config/strategy.py`, add a second strategy module under `user_config/`, or adjust order execution in `src/ibkr_multi_asset/setup_functions.py`.

<a id='build'><a>
### Step 3: Build the Package

Once you have made your changes to the `src/` folder, you need to rebuild the Python package.

1. **Open a terminal and type the following:**
```bash
python -m build
```

If you are using the provided helper script, you can also rebuild and run from `user_config/` with:
```bash
./rebuild_and_run.sh
```

<a id='reinstall'><a>
### Step 4: Reinstall the Modified Package

To ensure your Python environment uses your newly modified code, you must reinstall the package from the wheel file.

```bash
pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
```

<a id='run'><a>
### Step 5: Run the Trading Setup

Navigate to the `user_config` directory and run:
```bash
python main.py
```

If `rebuild_and_run.sh` is present, it is the preferred workflow after source changes because it rebuilds the wheel, reinstalls it, and starts the live setup in one step.

<a id='traits'><a>
## 3. Developer Notes

### Strategy-Agnostic Engine
The engine is designed to call a specific interface from the strategy module selected by `strategy_file` in `main.py`. As a developer, you can implement entirely different trading paradigms (HFT, Arbitrage, etc.) as long as you maintain the standard function signatures:
- `get_asset_frequency(symbol)`
- `prepare_base_df(symbol, ...)`
- `get_signal(app, ...)`
- `strategy_parameter_optimization(...)`
- `validate_strategy_optimization(...)`

- **Selectable Strategy Modules**: The setup supports multiple strategy files under `user_config/`, with runtime selection handled by `strategy_file` in `main.py`.
- **Scheduled Optimization**: The engine can rerun `strategy_parameter_optimization()` automatically on a `daily` or `weekly` cadence according to `optimization_frequency` in `main.py`.

### Multi-Asset Data Handling
The engine dynamically handles different market types:
- **Crypto**: 24/7 sessions, fractional share support, `TRADES` data type.
- **Futures**: Multiplier-based quantity calculation, tick-size rounding, `TRADES` data type.
- **Forex/Metals**: 24/5 sessions, unit-based quantity, `BID/ASK` data type.
- **Stocks**: Exchange-aware contract resolution, primary-exchange support, regular-trading-hours history requests, adjustable tick-size rounding, and optional fractional-share handling.

### Broker-Side Stock Restrictions
- US stock orders can still be rejected even when the engine constructs them correctly.
- A common case is Interactive Brokers error 201 triggered by the Pattern Day Trader rule when the securities segment equity is below USD 25,000.
- When you debug stock-order behavior, distinguish broker/account restrictions from actual engine bugs.

### Carry-Protection Bridge
- The engine supports a local `trading_day_origin` in `user_config/main.py`.
- If the app starts after broker reopen but before that origin, the engine enters a protection-only window instead of running the normal portfolio decision loop.
- During that window, the engine restores or rebuilds broker-side stop-loss and take-profit orders for eligible live positions, then sleeps until the configured origin.
- The carry refresh currently runs in parallel across the configured universe.
- Futures are intentionally skipped in this bridge because broker acceptance in that interval may not mean immediate exchange placement.

### Risk-Management Direction Rule
- Risk-management orders must always be on the opposite side of the protected live position.
- Long position => `SELL` stop-loss and `SELL` take-profit.
- Short position => `BUY` stop-loss and `BUY` take-profit.
- For forex symbols, the protected position is resolved using the base symbol and quote currency together. This avoids mixing rows such as `USD.JPY` and `USD.CHF` when both share `Symbol = USD`.

### Live Persistence Notes
- Live state is persisted to `user_config/data/database.xlsx`.
- Important sheets include `positions`, `open_orders`, `orders_status`, `account_summary`, `account_update_times`, `runtime_audit`, `ib_errors`, and `strategy_state`.
- When you change persistence logic, test repeated saves in the same live session. The setup keeps historical and current broker data in the same workbook, so duplicate-index and duplicate-column normalization matters.

### Modular PDF Reporting
The reporting logic is encapsulated in `report_generator.py` and runs as a background thread managed by the engine. This allows developers to add new analytics pages (e.g., VaR analysis, Correlation heatmaps) without impacting the main trading heartbeat.

<a id='helper_scripts'><a>
## 4. Helper Scripts

### Rebuild and Run Helpers
The `user_config/` folder contains three rebuild helpers with the same workflow for different operating systems:

- `rebuild_and_run.sh`
  - Use on Linux, WSL, or any environment where you run the setup from a Bash shell.
  - It rebuilds the wheel, reinstalls it with `--no-deps`, returns to `user_config/`, and starts `main.py`.
- `rebuild_and_run_mac.command`
  - Use on macOS when you want a Finder-double-clickable or Terminal-friendly script.
  - It rebuilds the wheel, reinstalls it, returns to `user_config/`, and starts `main.py`.
- `rebuild_and_run_windows.bat`
  - Use on Windows Command Prompt or by double-clicking the batch file.
  - It rebuilds the wheel, reinstalls it, returns to `user_config/`, and starts `main.py`.

Use one of these helpers when you changed source code under `src/ibkr_multi_asset/`, changed packaging metadata, or changed strategy/runtime files and want to make sure the installed wheel matches the current project files.

If you only changed configuration values in `main.py` and your environment already points to the latest installed wheel, running `python main.py` directly is usually enough.

### Paper-Reset Helpers
- `reset_paper_trading_state.py`
  - Use this when you want to reset the paper account workflow and the local generated state together.
  - The script first cancels orders and closes open paper-account positions, then deletes generated local outputs such as the workbook, logs, reports, caches, and optimization artifacts.
- `close_all_positions.py`
  - Use this when you only want to flatten the configured paper account and keep the local files.
  - This script now calls the position-closing flow implemented inside `reset_paper_trading_state.py`.

### IB API order ID reset
- After running `reset_paper_trading_state.py`, open your TWS or Gateway application.
- Navigate to `File > Global Configuration > API > Settings`.
- Click **Reset API order ID sequence**, then **Apply**, then **OK** to avoid order ID conflicts when you next start the paper trading loop.
