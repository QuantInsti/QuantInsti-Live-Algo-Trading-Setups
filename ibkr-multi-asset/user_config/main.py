"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
USER_CONFIG_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(USER_CONFIG_ROOT) not in sys.path:
    sys.path.insert(0, str(USER_CONFIG_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
User guide
- Edit this file to choose your account settings, trading universe, and broker metadata.
- Keep custom strategy logic in the selected strategy file under user_config/.
- You normally should not need to edit source files under src/.
"""

# Connection and account settings
account = os.getenv("IBKR_ACCOUNT", "DU1234567")
timezone = "America/New_York"
host = "127.0.0.1"
port = 7497
client_id = 1
portfolio_client_id_offset = 100
account_currency = "USD"

# Engine runtime settings
local_restart_hour = 23
trading_day_origin = "18:00"
trail = False
strict_targets_validation = True
optimization_frequency = "daily"  # supported: "daily", "weekly"
portfolio_leverage = 1.0  # Fallback only; the active strategy can override this with optimized portfolio leverage.
portfolio_parallel_order_submission = False  # Use one shared IB app while validating the new strategy so execution is easier to audit.

# Strategy controls
strategy_file = "strategies/strategy.py"
strategy_frequency = "5min"
strategy_optimization_lookback = 3000
fixed_max_leverage = 1.0
long_only_symbols = ["ETH"]

# Universe
fx_pairs = ["EURUSD"]
futures_symbols = ["MES"]
metals_symbols = ["XAUUSD"]
crypto_symbols = ["ETH"]
stock_symbols = []

# Venue metadata
forex_exchange = "IDEALPRO"
forex_currency = "USD"
futures_exchange = "CME"
futures_currency = "USD"
futures_roll_policy = "AUTO_FRONT_MONTH"
# Leave as None to auto-resolve the nearest non-expired front month for symbols like "MES".
# Set an explicit value such as "202606" to force a specific futures month.
futures_contract_month = None
metals_exchange = "SMART"
metals_currency = "USD"
metals_sec_type = "CMDTY"
metals_quantity_step = 1.0  # IBKR spot metals such as XAUUSD reject fractional quantities in this setup.
crypto_exchange = "PAXOS"
crypto_currency = "USD"

# Notifications
smtp_username = os.getenv("SMTP_USERNAME", "your_email@example.com")
to_email = os.getenv("TO_EMAIL", "recipient@example.com")
# Use a provider-specific app password or token and keep it out of version control.
password = os.getenv("SMTP_APP_PASSWORD", "YOUR_SMTP_APP_PASSWORD")


from ibkr_multi_asset import engine


if __name__ == "__main__":
    engine.main()
