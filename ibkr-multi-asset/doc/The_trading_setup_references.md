## Technical References

Licensed under the QuantInsti Open License (QOL) v1.1.

## Interactive Brokers

- IBKR TWS / Gateway API:
  - https://ibkrcampus.com/ibkr-api-page/trader-workstation-api/
- IBKR cryptocurrencies:
  - https://www.interactivebrokers.com/en/trading/products-cryptocurrencies.php
- CME Micro E-mini S&P 500 futures:
  - https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html

## Portfolio Construction

- Marcos López de Prado, *Advances in Financial Machine Learning*
  - hierarchical risk parity
- risk parity overview:
  - https://blog.quantinsti.com/risk-parity/

## Strategy Concepts

The bundled strategy uses:

- moving-average trend following
- ATR-based stop and take-profit thresholds
- hierarchical risk parity
- capped Kelly-style portfolio leverage

Useful references:
- trend following and moving averages
- volatility targeting
- Kelly criterion under estimation error

## Broker-Constraint Modeling

The codebase includes a standalone constraint-discovery utility:

- [broker_constraint_report.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/broker_constraint_report.py)

This should be used before writing a realistic backtest because some execution constraints are broker- and account-specific:

- crypto concentration limits
- quantity-step restrictions
- minimum tradable sizes
- price-tick rounding
- native vs synthetic stop support

## Runtime Files

Core runtime:
- [engine.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/engine.py)
- [setup.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/setup.py)
- [setup_functions.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/setup_functions.py)
- [trading_functions.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/src/ibkr_multi_asset/trading_functions.py)

Bundled default strategy:
- [strategy.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/strategies/strategy.py)

Bundled runtime configuration:
- [main.py](/home/josgt/Downloads/alpha_quant/QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset/user_config/main.py)
