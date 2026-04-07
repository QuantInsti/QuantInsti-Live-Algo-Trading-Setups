# IBKR Multi-Asset Trading Setup

This repository now centers on the Interactive Brokers multi-asset live trading setup. The active framework supports Forex, futures, spot metals, and crypto inside one portfolio process, with a configurable strategy layer, scheduled optimization, and automated reporting.

## Contributors
- Author: [José Carlos Gonzáles Tanaka](https://www.linkedin.com/in/jose-carlos-gonzales-tanaka/)
- QuantInsti's EPAT Content Team is responsible for maintaining and contributing to this repository.

## Repository Focus

The main setup is:

-   **Multi-Asset Trading (`ibkr-multi-asset/`)**: A strategy-agnostic framework for trading FX, MES futures, XAUUSD spot metals, and crypto with a unified portfolio engine.

Legacy IBKR examples are still present in `ibkr-forex/` and `ibkr-stock/`, but the current maintained workflow and documentation target `ibkr-multi-asset/`.

## Features

-   **Unified Portfolio Engine**: The multi-asset setup processes all configured asset classes simultaneously, enabling portfolio-level allocation and rebalancing.
-   **Strategy-Agnostic Framework**: The core engine is decoupled from specific trading logic. Users can implement or swap entire strategies by modifying `user_config/strategy.py` without altering source code.
-   **Asset-Specific Frequencies**: Supports different timeframes (e.g., 4h, 1D) for different assets within the same setup.
-   **Automated PDF Reporting**: Generates comprehensive performance reports automatically every trading cycle.
-   **Self-Healing Configuration**: Includes robust default fallbacks for parameters, ensuring safe operation even if user configurations are incomplete.
-   **Modular Structure**: The runnable setup is self-contained under `ibkr-multi-asset/`.
-   **User Configuration**: Provides straightforward ways to configure strategy, credentials, and runtime parameters without editing core engine files.
-   **Data Management**: Includes utilities for historical data download and management.
-   **Advanced Risk Control**: Implements dynamic ATR stops, margin scaling, and regime-based filters.

## Architecture

```text
QuantInsti-Live-Algo-Trading-Setups/
├── ibkr-multi-asset/
│   ├── src/ibkr_multi_asset/   # Core engine, broker integration, reporting
│   ├── user_config/            # Account settings, universe, strategy selection
│   ├── doc/                    # Setup and developer documentation
│   └── res/                    # Images used by the docs
├── ibkr-forex/                 # Legacy example
├── ibkr-stock/                 # Legacy example
├── pyproject.toml              # Repository metadata
├── LICENSE.md
└── README.md
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QuantInsti/QuantInsti-Live-Algo-Trading-Setups
    ```
2.  **Move into the maintained setup directory**:
    ```bash
    cd QuantInsti-Live-Algo-Trading-Setups/ibkr-multi-asset
    ```
3.  **Review the setup guide** in `doc/Start_here_documentation.md`.
4.  **Set your local credentials and account values** in `user_config/main.py` or through environment variables before running the bot.
5.  **Start the application**:
    ```bash
    python user_config/main.py
    ```

## Disclaimer

Trading involves substantial risk, and this project is for educational purposes only. The authors or contributors are not responsible for any financial losses. You should always test your strategies thoroughly in a paper trading account before deploying with real capital.
