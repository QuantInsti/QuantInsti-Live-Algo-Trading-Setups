# IB Multi-Asset Trading Setup

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: QOL v1.1](https://img.shields.io/badge/license-QOL%20v1.1-blue.svg)](../LICENSE.md)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)
[![Security](https://img.shields.io/badge/security-policy-brightgreen.svg)](SECURITY.md)
[![Code of Conduct](https://img.shields.io/badge/code%20of-conduct-ff69b4.svg)](CODE_OF_CONDUCT.md)

> A Python framework for automated multi-asset trading using the Interactive Brokers API.

## Overview

This framework provides a working multi-asset trading setup for a unified universe (**Forex + Futures + Spot Metals + Crypto + Stocks**) using the Interactive Brokers API. It separates engine logic from user strategy logic, supports portfolio rebalancing across asset classes, and generates PDF reports during live operation.

## Key Features

### Unified Multi-Asset Engine
- **Cross-Asset Support**: Simultaneously trade Forex, Futures, Spot Metals, Crypto, and Stocks in a single capital pool.
- **IB-Supported Market Coverage**: The setup is not tied to one country or exchange. It can be configured from any operating location to trade assets available through Interactive Brokers, subject to your account permissions, market data subscriptions, and exchange access.
- **Dynamic Portfolio Rebalancing**: Supports strategy-defined portfolio allocation and margin scaling across all active assets.
- **Frequency Agnostic**: The engine supports any valid IB bar size (e.g., 1m, 5m, 1h, 1D). Different assets can run on different timeframes within the same setup.

### Strategy-Agnostic Framework
- **Selectable Strategy Files**: You can now choose the live strategy module from `user_config/main.py` with `strategy_file`.
- **Configurable Strategy Layer**: `user_config/strategy.py` exposes the live strategy hooks used by the engine and can be replaced with another module that follows the same interface.
- **Stable Strategy Interface**: Strategy logic stays isolated under `user_config/`. You can replace the active model without changing the engine code.
- **Default Parameter Fallbacks**: The setup can continue running with fallback values if some strategy parameters are missing from the selected strategy file.

### Reporting & Monitoring
- **Automated PDF Reports**: Generates a multi-page performance report (`portfolio_report.pdf`) every trading cycle.
- **Live Equity Tracking**: Real-time aggregation of CAGR, Sharpe, Drawdown, and PnL across the entire portfolio.

## Architecture

```
ibkr-multi-asset/
├── src/ibkr_multi_asset/
│   ├── engine.py              # Main multi-asset trading engine
│   ├── setup.py               # Core setup class
│   ├── report_generator.py    # Automated PDF Portfolio Reporting
│   ├── ib_functions.py        # Generic Multi-Asset IB API integration
│   ├── trading_functions.py   # Advanced Trading Logic
│   ├── setup_for_download_data.py # Bulk Historical Data Bootstrap
│   ├── setup_functions.py     # Execution & Order Management Utility
│   └── create_database.py     # Excel-based Live Data Storage
├── user_config/
│   ├── main.py                # Meta-Parameters (Connection, Universe, strategy selection)
│   └── strategy.py            # Mean-Reversion + Markowitz strategy
└── doc/                       # Comprehensive Documentation
```

## Installation

### Prerequisites
- Python 3.12 or higher
- Interactive Brokers account
- TWS (Trader Workstation) or IB Gateway

### Setup

```bash
# Create virtual environment
conda create --name forex_trading python=3.12
conda activate forex_trading

# Install the trading setup
pip install dist/ibkr-multi-asset-1.0.0-py3-none-any.whl

# Install Interactive Brokers API
# You should download it from IB and install in your environment
```

## Usage

To get started with the trading bot, follow these steps:

1.  **Install the package**: Make sure you have installed the package from the `.whl` file as described in the Installation section.
2.  **Configure Runtime Parameters**: Open `user_config/main.py` to set account, runtime values, `strategy_file`, and `optimization_frequency`.
3.  **Choose or Customize a Strategy File**:
    - `user_config/strategy.py`: default mean-reversion plus Markowitz portfolio strategy.
4.  **Edit Only The Selected Strategy File**: Strategy changes should be made in the file referenced by `strategy_file`.
5.  **Keep secrets local**: Do not commit real account IDs, email addresses, or app passwords. The default config reads these values from environment variables when available.
6.  **Run the Bot**: Once configured, you can start the trading bot by running the following command from the `ibkr-multi-asset` directory:
    ```bash
    python user_config/main.py
    ```

You can further customize the setup by adjusting risk management settings and configuring email notifications within the configuration files.

## Documentation

| Documentation | Description |
|---------------|-------------|
| [Quick Start Guide](doc/Start_here_documentation.md) | How to get up and running in minutes |
| [Strategy Development](doc/Strategy_documentation.md) | How to build and customize trading strategies |
| [Technical Reference](doc/The_trading_setup_references.md) | API reference and implementation notes |
| [Developer Guide](doc/Developer_documentation.md) | Customization and development workflow |

## Contributing

We welcome contributions. You can see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- You can report bugs and issues.
- You can suggest new features.
- You can improve documentation.
- You can submit code improvements.
- You can add tests and examples.

## Support

For questions and support:
- **Documentation**: You can check the documentation in the `doc/` folder.
- **Email**: You can contact your support manager (if you're a present EPAT student) or the alumni team (if you're a past EPAT student).

## Author and Maintainer

*   **Author:** [José Carlos Gonzáles Tanaka](https://www.linkedin.com/in/jose-carlos-gonzales-tanaka/)
*   **Maintainer:** EPAT Content Team

## Disclaimers

### Risk Warning
**Trading leveraged financial instruments involves substantial risk and may not be suitable for all investors. Forex, futures, stocks, metals, and crypto can all result in significant losses.**

### Account Restrictions
If you include US stock symbols in the multi-asset universe, Interactive Brokers may reject otherwise valid stock orders because of the Pattern Day Trader rule. In practice, this can happen when the securities segment equity is below USD 25,000 and the account attempts frequent intraday equity trading. These rejections are broker-side account restrictions and are separate from the order-construction logic in the setup.

### Educational Purpose
This trading setup is provided for **educational purposes only**. You should not consider it as investment advice. You should always:
- Test thoroughly in paper trading first.
- Understand the risks involved.
- Never risk more than you can afford to lose.
- Consult with financial professionals before live trading.
