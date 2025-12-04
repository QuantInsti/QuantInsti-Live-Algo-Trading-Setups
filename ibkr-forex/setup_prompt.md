**Prompt:**

"My objective is to integrate the core logic from a standalone trading strategy, contained in `user_config/backtester.py`, into the existing forex trading framework. The framework requires a specific file structure, so the logic must be carefully adapted.

Here are the required steps:

1.  **Scaffold the New Strategy File:** Create a new file, `user_config/new_strategy.py`. This file must be structured to be compatible with the trading engine, meaning it must define a specific set of functions: `prepare_base_df`, `strategy_parameter_optimization`, `get_signal`, `set_stop_loss_price`, and `set_take_profit_price`.

2.  **Port and Adapt the Strategy Logic:** Populate the functions in `new_strategy.py` by extracting and adapting the corresponding logic from `user_config/backtester.py`.
    *   **`prepare_base_df`:** Re-implement this function to perform the necessary feature engineering. It should be adapted to calculate all the specific technical indicators, returns, or other data transformations that the strategy in `backtester.py` requires for its decisions.
    *   **`strategy_parameter_optimization`:** This function must be rewritten to execute the training, backtesting, and parameter selection routine found in `backtester.py`. The goal is to identify the optimal set of parameters for the strategy and save any resulting artifacts (like trained model files or configuration settings) to the `data/models/` directory for later use in live trading.
    *   **`get_signal`:** This function will contain the core signal generation logic. It should be adapted to load any necessary models or parameters from the optimization step and apply the strategy's rules to the most recent market data to produce a live trading signal (e.g., long, short, or hold).
    *   **`set_stop_loss_price` & `set_take_profit_price`:** Implement the specific risk management logic from `backtester.py` here. These functions should calculate the precise stop-loss and take-profit prices based on the strategy's rules.

3.  **Integrate with the Framework's Data Source:** A critical requirement is to modify the strategy's data handling. Instead of using its original method for acquiring data (e.g., downloading from a web API), it must be adapted to use the historical data file provided by and already present within this trading framework. The implementation must locate and utilize this existing data source.

4.  **Activate the New Strategy:** Once the `user_config/new_strategy.py` file is complete and fully functional, perform the final integration step. Rename the original `user_config/strategy.py` to `user_config/old_strategy.py` to preserve it as a backup. Then, rename `user_config/new_strategy.py` to `user_config/strategy.py` to make it the active strategy file used by the trading engine."