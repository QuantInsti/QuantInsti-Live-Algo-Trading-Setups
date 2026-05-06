## Developer Documentation

Licensed under the QuantInsti Open License (QOL) v1.1.

## Scope

This document is for modifying the multi-asset source code, rebuilding the wheel, and running the installed package again.

## Architecture

### Runtime Layer

Main runtime files:
- [engine.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/engine.py)
- [setup.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/setup.py)
- [setup_functions.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/setup_functions.py)
- [trading_functions.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/trading_functions.py)
- [report_generator.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/report_generator.py)

### Strategy Layer

Bundled default strategy:
- [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py)

Strategy selection is runtime-configurable through:
- [main.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py)

The engine loads the active strategy through:
- `ibkr_multi_asset.strategy_runtime`

Do not hardcode `user_config/strategies/strategy.py` imports in new source changes. Use the runtime strategy proxy.

## Execution Modes

The source code supports two execution modes:

1. sequential order submission on one shared IB app
2. parallel order submission with isolated IB apps per asset

The bundled default config uses:
- `portfolio_parallel_order_submission = False`

That is intentional because it is easier to audit during validation.

If you re-enable parallel isolated apps, the code will open one client session per tradable asset. Do not try to run parallel order submission through one mutable shared app instance.

## Optimization Flow

The engine:

1. computes the trading-day bucket from `trading_day_origin`
2. validates the saved optimization manifest
3. reruns `strategy_parameter_optimization()` when needed
4. resolves portfolio leverage from the validated manifest

This means:
- optimization is not just a strategy helper
- it is an explicit engine-controlled dependency of the live run

## Persistence Layout

Live workbook:
- `user_config/data/database.xlsx`

Workbook sheets:
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

Strategy state store:
- `user_config/data/strategy_state.json`

Live log directory:
- `user_config/data/log/`

Important:
- workbook-sheet references such as `account_summary`, `account_update_times`, `ib_errors`, and workbook `strategy_state` are not part of the active persistence model

## How To Modify And Rebuild

From the project root:

```bash
python -m build
python -m pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
python user_config/main.py
```

Helper scripts:
- `user_config/rebuild_and_run.sh`
- `user_config/rebuild_and_run_mac.command`
- `user_config/rebuild_and_run_windows.bat`

## What To Edit For Common Tasks

### Change Runtime Settings

Edit:
- [main.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py)

Examples:
- connection settings
- trading-day origin
- universe
- strategy module selection
- optimization frequency
- leverage cap

### Change Strategy Logic

Edit:
- the active module under `user_config/strategies/`

The bundled default strategy is trend-following plus HRP, but the interface is intentionally generic.

### Change Broker Execution Logic

Edit:
- [setup_functions.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/setup_functions.py)

Typical changes:
- quantity rounding
- order construction
- broker refresh behavior
- risk-management order handling
- synthetic stop logic

### Change Reporting

Edit:
- [report_generator.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/report_generator.py)

### Change Pre-Backtest Constraint Discovery

Edit:
- [broker_constraint_report.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/broker_constraint_report.py)

This script is intentionally standalone and should remain reusable outside the live setup.

## Developer Caveats

### 1. Relative Paths

Recent source changes normalized several config lookups to explicit `user_config/main.py` paths. Keep doing that. Do not assume the process is launched from a specific cwd unless you verify it.

### 2. Broker Constraints Are Real

Examples already observed:
- crypto concentration rejections
- whole-unit behavior for XAUUSD in this setup
- futures minimum tradable size
- odd-lot FX warnings

These should inform both live sizing and backtesting assumptions.

### 3. Strategy-Agnostic Means Interface Stability

If you modify engine-to-strategy interaction, preserve the strategy interface or update documentation and all bundled strategy modules together.

### 4. Shared-State Changes Need Care

The code distinguishes between:
- global broker/account snapshots
- symbol-specific order logic

Avoid reintroducing per-symbol broker refreshes in the portfolio loop unless the behavior is explicitly required.

## Paper-Trading Helpers

- `reset_paper_trading_state.py`
  - flattens the paper account and deletes generated local outputs
- `close_all_positions.py`
  - flattens positions without deleting local outputs

Use these carefully and only against the intended paper account configuration.
