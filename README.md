# Live Algorithmic Trading Setups

This repository contains a collection of ready-to-use algorithmic trading setups for different brokers and asset classes. Each setup provides a foundational structure for you to develop, test, and deploy live trading strategies.

## Setups Included

This monorepo is designed to be extended with setups for any broker or asset class. Currently, it includes examples for Interactive Brokers:

-   **Forex Trading (`ibkr-forex/`)**: A complete setup for you to trade forex pairs via Interactive Brokers.
-   **Stock Trading (`ibkr-stock/`)**: A complete setup for you to trade stocks via Interactive Brokers.

## Features

-   **Modular Structure**: Each setup is self-contained within its own directory, with a clear and organized structure.
-   **Broker Agnostic**: Designed to accommodate setups for various broker APIs.
-   **User Configuration**: Each setup provides an easy way for you to configure your strategy, credentials, and other parameters.
-   **Data Management**: Includes scripts and a structure for you to handle historical data, logs, and trading reports.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to a setup directory:**
    ```bash
    cd ibkr-crypto
    ```
3.  **Follow the instructions** in the `README.md` and `doc/` folder within the specific setup directory to configure and run your trading bot.

## Disclaimer

Trading involves substantial risk, and this project is for educational purposes only. The authors or contributors are not responsible for any financial losses. You should always test your strategies thoroughly in a paper trading account before deploying with real capital.
