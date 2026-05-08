# IBKR Multi-Asset Trading Setup

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: QOL v1.1](https://img.shields.io/badge/license-QOL%20v1.1-blue.svg)](LICENSE.md)

Python setup for live portfolio trading with Interactive Brokers across multiple asset classes from one strategy/runtime framework.

## What It Does

- trades a mixed universe of FX, futures, spot metals, crypto, and optionally stocks
- keeps strategy logic in `user_config/strategies/`
- runs a portfolio-level optimization and allocation process
- persists broker state and trading state to Excel/JSON
- generates a live PDF performance report
- includes a standalone broker-constraint discovery utility for pre-backtest work

## Default Configuration

The bundled `user_config/main.py` is configured as a conservative validation profile:

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
- `trading_day_origin = "18:00"` in `America/Lima`

These are defaults, not hard requirements. Change them in [`main.py`](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py).

## Default Strategy

The default strategy in [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py) is:

- moving-average trend following per asset
- daily parameter optimization using only data before the active trading-day bucket
- hierarchical risk parity across the full tradable universe
- one portfolio-level Kelly-style leverage multiplier capped by `fixed_max_leverage`
- ATR-based stop-loss and take-profit thresholds
- crypto long-only behavior controlled by `long_only_symbols`

The strategy interface is generic. Future users can replace the strategy module as long as the expected hooks remain available.

## Create Your Own Strategy With An LLM

You don't need to write your strategy file from scratch. The included [`llm-guide.md`](llm-guide.md) is a complete prompt :  copy its entire contents and paste it into an LLM (Claude, GPT-4, DeepSeek, or any other) along with a description of your strategy. The LLM will generate a fully functional `my_strategy.py` file.

**Two ways to use it:**

| Path | What you provide | What the LLM generates |
|---|---|---|
| **From a backtest** | Your backtesting script + `llm-guide.md` | `my_strategy.py` with your exact signal logic, portfolio construction, and parameters adapted to the live engine |
| **From scratch** | `llm-guide.md` + plain-language description | `my_strategy.py` with your described strategy (e.g., "Bollinger breakout on 5-min bars, inverse-vol weights, Kelly leverage") |

The LLM guide covers every function the engine requires, full boilerplate implementations of data normalization and optimization helpers, a mandatory question checklist to catch gaps in your strategy, and troubleshooting advice. No source code modifications needed :  just drop the generated file into `user_config/strategies/` and point `main.py` to it.

```python
# In main.py:
strategy_file = "strategies/my_strategy.py"
```

The guide also covers:
- converting backtest logic to live engine signals
- IBKR broker constraints (whole-units, crypto long-only, synthetic stops)
- risk management (stops, leverage, drawdown limits, concentration caps)
- daily vs weekly parameter optimization scheduling
- strategy state persistence across bars and restarts

## Repository Layout

```text
ibkr-multi-asset/
├── src/ibkr_multi_asset/
│   ├── engine.py
│   ├── setup.py
│   ├── setup_functions.py
│   ├── trading_functions.py
│   ├── ib_functions.py
│   ├── report_generator.py
│   └── create_database.py
├── llm-guide.md
├── user_config/
│   ├── main.py
│   ├── broker_constraint_report.py
│   ├── close_all_positions.py
│   ├── reset_paper_trading_state.py
│   └── strategies/
│       └── strategy.py
└── doc/
```

## Installation

### Prerequisites

- Python 3.12+
- TWS or IB Gateway
- IB API-enabled account

### Build And Install

```bash
cd /home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset
python -m build
python -m pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
```

## Running The Live Setup

```bash
cd /home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset
python user_config/main.py
```

## Standalone Broker Constraint Report

Use the standalone utility before writing a backtest or when you want to inspect broker-side execution constraints for a chosen asset list.

Script:
- [broker_constraint_report.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/broker_constraint_report.py)

Outputs:
- `user_config/data/reports/broker_constraints_report.json`
- `user_config/data/reports/broker_constraints_report.pdf`

Run:

```bash
cd /home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset
python user_config/broker_constraint_report.py
```

This script is intentionally independent of `main.py` so it can be shared and reused outside this setup.

## Live Data And Persistence

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

JSON state file:
- `user_config/data/strategy_state.json`

Live logs:
- `user_config/data/log/`

## Execution Model

The source code supports both:

- one shared app with sequential order submission
- isolated per-asset apps for parallel order submission

The bundled default config uses the first mode:

- `portfolio_parallel_order_submission = False`

That keeps the validation workflow easier to audit.

## Documentation

- [Start Here](doc/Start_here_documentation.md)
- [Strategy Documentation](doc/Strategy_documentation.md)
- [Developer Documentation](doc/Developer_documentation.md)
- [Technical References](doc/The_trading_setup_references.md)
- [Contributing](CONTRIBUTING.md)

## Risk Notes

- broker-side limitations such as crypto concentration caps, venue-specific stop support, quantity-step rules, and minimum increments are real execution constraints
- those constraints should be modeled before backtesting or live scaling
- the included broker-constraint report is meant to help build that constraint layer before backtest implementation

## Disclaimer

This project is for educational and research use. Test in paper trading before using any live capital.
