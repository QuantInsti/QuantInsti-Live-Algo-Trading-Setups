"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import importlib.util
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force CWD to this file's directory so all relative paths resolve correctly.
# Resolve __file__ first — it may be relative when invoked as `python user_config/main.py`.
_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)

load_dotenv(_HERE / ".env")

PROJECT_ROOT = _HERE.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
USER_CONFIG_ROOT = _HERE
STRATEGIES_ROOT = USER_CONFIG_ROOT / "strategies"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(USER_CONFIG_ROOT) not in sys.path:
    sys.path.insert(0, str(USER_CONFIG_ROOT))
if str(STRATEGIES_ROOT) not in sys.path:
    sys.path.insert(0, str(STRATEGIES_ROOT))

# --- Dynamic strategy loading ---
# The strategy_file variable (set below) points to the user's chosen strategy.
# We load it now and inject it as the 'strategy' module so that both engine.py
# and setup_functions.py can keep their hardcoded "import strategy as stra".
# Set all the variables you need for the trading app
###############################################################################
###############################################################################
""" Set the variables as per your trading specifications"""
# Set account name in case you do paper trading
account = os.getenv("IBKR_ACCOUNT") or "DU1234567"
# Set the time zone
timezone = 'America/Lima'
# The app port
port = 7497
# The account base currency symbol
account_currency = 'USD'
# The asset symbol
symbol = 'EURUSD'
# The data frequency for trading
data_frequency = '5min'
# The auto-restart hour
local_restart_hour = 23
# The historical data file address
historical_data_address = 'historical_data.csv'
# The base_df file address
base_df_address = 'app_base_df.csv'
# The train span 
train_span = 3500
# Set the number of days to set the test data number of rows   
test_span_days = 1
# The app host
host='127.0.0.1'
# The client id
client_id=1
# Set the seed for the machine-learning model
seed = 2024  
# Set the trailing stop loss boolean
trail = True
# Set the seed you would like to use for your trading setup computations
seed = 2025
# The email that will be used to send the emails
smtp_username = os.getenv("SMTP_USERNAME")
to_email = os.getenv("TO_EMAIL")
# The app password that was obtained in Google. You need to allow app password in Google: 
# https://support.google.com/mail/answer/185833?hl=en
password = os.getenv("SMTP_APP_PASSWORD")
###############################################################################
""" Set the additional variables you would like to add for the strategy functions"""
# Example (uncomment the below code line to incorporate this variable in the trading setup)
# num_technical_indicators = 50
# The strategy module location under user_config/
strategy_file = 'strategies/strategy.py'
# Set the leverage of your position
leverage = None
# ...
###############################################################################

# --- Load the chosen strategy module ---
_strategy_path = USER_CONFIG_ROOT / strategy_file
if not _strategy_path.exists():
    raise FileNotFoundError(f"Strategy file not found: {_strategy_path}")
_spec = importlib.util.spec_from_file_location("strategy", str(_strategy_path))
_strategy_module = importlib.util.module_from_spec(_spec)
sys.modules['strategy'] = _strategy_module
_spec.loader.exec_module(_strategy_module)

# Now engine.py and setup_functions.py can "import strategy as stra"
from ibkr_forex import engine
