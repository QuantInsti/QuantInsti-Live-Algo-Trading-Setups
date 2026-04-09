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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

"""
User guide
- Edit this file to choose your account settings, trading universe, and broker metadata.
- Keep custom strategy logic in the selected strategy file under user_config/.
- You normally should not need to edit source files under src/.
"""

# Connection and account settings
account = os.getenv("IBKR_ACCOUNT", "DU1234567")
timezone = "America/Lima"
host = "127.0.0.1"
port = 7497
client_id = 1
account_currency = "USD"

# Engine runtime settings
local_restart_hour = 23
trading_day_origin = "18:00"
trail = False
strict_targets_validation = True
strategy_file = "strategy.py"
optimization_frequency = "weekly"  # supported: "daily", "weekly"

# Universe
fx_pairs = ["USDJPY", "AUDUSD", "USDCHF"]
futures_symbols = ["MES"]
metals_symbols = ["XAUUSD"]
crypto_symbols = ["BTC", "ETH", "SOL", "LTC", "BCH"]

# Venue metadata
forex_exchange = "IDEALPRO"
forex_currency = "USD"
futures_exchange = "CME"
futures_currency = "USD"
futures_roll_policy = "AUTO_FRONT_MONTH"
metals_exchange = "SMART"
metals_currency = "USD"
metals_sec_type = "CMDTY"
crypto_exchange = "PAXOS"
crypto_currency = "USD"

# Notifications
smtp_username = os.getenv("SMTP_USERNAME", "your_email@gmail.com")
to_email = os.getenv("TO_EMAIL", "recipient_email@gmail.com")
# Use a provider-specific app password or token and keep it out of version control.
password = os.getenv("SMTP_APP_PASSWORD", "your-app-password")


from ibkr_multi_asset import engine


if __name__ == "__main__":
    engine.main()
