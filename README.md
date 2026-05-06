# QuantInsti Live Algo Trading Setups

This repository provides production-ready trading frameworks for Interactive Brokers (IBKR). Each setup is designed for live execution with automated data management, configurable strategies, and robust error handling.

## Contributors
- Author: [José Carlos Gonzáles Tanaka](https://www.linkedin.com/in/jose-carlos-gonzales-tanaka/)
- QuantInsti's EPAT Content Team is responsible for maintaining and contributing to this repository.

## Repository Focus

The repository includes three specialized trading setups:

-   **Multi-Asset Trading (`ibkr-multi-asset/`)**: A strategy-agnostic framework for trading FX, MES futures, XAUUSD spot metals, and crypto with a unified portfolio engine.
-   **Forex Trading (`ibkr-forex/`)**: A dedicated framework optimized for currency pairs and FX-specific execution logic.
-   **Stock Trading (`ibkr-stock/`)**: A modular setup for systematic equity trading with integrated contract details and data acquisition utilities.

## Key Features

-   **Unified Execution Engines**: Production-grade frameworks supporting concurrent execution across multiple asset classes or focused single-asset strategies.
-   **Strategy-Agnostic Architecture**: Decoupled core engines from strategy logic, allowing users to implement or swap strategies via `user_config/` without modifying the underlying framework.
-   **Automated Data Infrastructure**: Integrated utilities for bulk historical data acquisition and local data management across all setups.
-   **Performance Analytics & Reporting**: Automated generation of performance metrics and reports for portfolio monitoring.
-   **Robust Error Recovery**: Self-healing configuration with default parameter fallbacks and connection monitoring to ensure operational continuity.

## Architecture

The project is organized to separate the core trading engine from user-specific configurations and strategies.

### Multi-Asset Setup (`ibkr-multi-asset/`)
-   **`src/ibkr_multi_asset/`**: Core engine, portfolio rebalancing, and PDF reporting logic.
-   **`user_config/`**: Primary entry point (`main.py`) and strategy implementation.

### Forex Setup (`ibkr-forex/`)
-   **`src/ibkr_forex/`**: Core engine optimized for FX-specific order types and data handling.
-   **`user_config/`**: Connection settings and Forex strategy logic.

### Stock Setup (`ibkr-stock/`)
-   **`src/ibkr_stock/`**: Core engine featuring contract detail utilities and equity data management.
-   **`user_config/`**: Portfolio settings and equity strategy logic.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QuantInsti/QuantInsti-Live-Algo-Trading-Setups
    ```
2.  **Select a setup directory**:
    Navigate to `ibkr-multi-asset/`, `ibkr-forex/`, or `ibkr-stock/` depending on your trading requirements.
3.  **Review the documentation**:
    Each setup includes a `doc/` folder with detailed setup and strategy development guides.
4.  **Configure environment**: Set your credentials in `user_config/main.py` or via environment variables.
5.  **Execution**: Launch the trading system:
    ```bash
    python user_config/main.py
    ```

## Disclaimer

Trading involves substantial risk, and this project is for educational purposes only. The authors or contributors are not responsible for any financial losses. You should always test your strategies thoroughly in a paper trading account before deploying with real capital.
