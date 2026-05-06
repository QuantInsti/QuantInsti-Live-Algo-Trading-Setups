# Contributing

## Scope

This package centers on:

- the multi-asset IBKR runtime under `src/ibkr_multi_asset/`
- live configuration under `user_config/main.py`
- strategy modules under `user_config/strategies/`
- the standalone broker-constraint utility under `user_config/broker_constraint_report.py`

## Before You Change Code

1. decide whether the change belongs in:
   - runtime engine code
   - strategy code
   - reporting code
   - standalone utilities
2. keep strategy-specific logic out of the engine where possible
3. update the markdown docs when behavior changes

## Development Workflow

```bash
cd /home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset
python -m build
python -m pip install dist/ibkr_multi_asset-1.0.0-py3-none-any.whl --force-reinstall
python user_config/main.py
```

## Current Interface Expectations

If you contribute a new strategy module, keep the live interface stable:

- `get_asset_runtime_policy(symbol, asset_class=None)`
- `get_asset_frequency(symbol)`
- `get_asset_train_span(symbol)`
- `prepare_base_df(...)`
- `strategy_parameter_optimization(...)`
- `validate_strategy_optimization(...)`
- `get_signal(app, ...)`
- `set_stop_loss_price(app)`
- `set_take_profit_price(app)`

## Current Persistence Model

The active workbook sheets are:

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

Additional state lives in:
- `user_config/data/strategy_state.json`

Current logs live in:
- `user_config/data/log/`

## Testing Expectations

At minimum:

- run `py_compile` on touched Python modules
- run a paper-trading smoke test for behavior that touches execution
- if you change broker constraints or sizing logic, validate against paper-account behavior

## Documentation Expectations

If you change:

- default runtime behavior
- strategy behavior
- persistence layout
- helper scripts
- broker-constraint workflow

then update:

- `README.md`
- relevant files in `doc/`

## Security

- do not commit real account credentials
- prefer environment variables for account IDs, email usernames, and app passwords
- do not publish paper/live credentials in examples
