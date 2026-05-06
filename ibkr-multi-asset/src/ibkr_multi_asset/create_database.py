"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import os
import pandas as pd
from ibkr_multi_asset import trading_functions as tf


WORKBOOK_SCHEMAS = {
    'open_orders': ['datetime', 'PermId', 'ClientId', 'OrderId', 'Account', 'Symbol', 'Currency', 'SecType', 'Exchange', 'Action', 'OrderType', 'TotalQty', 'CashQty', 'LmtPrice', 'AuxPrice', 'Status', 'market_open_time', 'market_close_time'],
    'orders_status': ['datetime', 'OrderId', 'Status', 'Filled', 'PermId', 'ClientId', 'Remaining', 'AvgFillPrice', 'LastFillPrice', 'market_open_time', 'market_close_time'],
    'executions': ['datetime', 'OrderId', 'PermId', 'ExecutionId', 'Symbol', 'Side', 'Price', 'AvPrice', 'cumQty', 'Currency', 'SecType', 'Position', 'Execution Time', 'Last Liquidity', 'OrderRef', 'market_open_time', 'market_close_time'],
    'commissions': ['datetime', 'ExecutionId', 'Commission', 'Currency', 'Realized PnL', 'market_open_time', 'market_close_time'],
    'positions': ['datetime', 'Account', 'Symbol', 'SecType', 'Currency', 'Position', 'Avg cost', 'market_open_time', 'market_close_time'],
    'portfolio_snapshots': ['datetime', 'Account', 'Symbol', 'LocalSymbol', 'SecType', 'Exchange', 'Currency', 'ConId', 'Position', 'MarketPrice', 'MarketValue', 'AverageCost', 'UnrealizedPnL', 'RealizedPnL', 'market_open_time', 'market_close_time'],
    'cash_balance': ['datetime', 'value', 'market_open_time', 'market_close_time'],
    'app_time_spent': ['datetime', 'seconds', 'market_open_time', 'market_close_time'],
    'periods_traded': ['datetime', 'trade_time', 'trade_done', 'market_open_time', 'market_close_time'],
    'account_updates': ['key', 'Account', 'Value', 'Currency', 'datetime'],
    'contract_details': ['Symbol', 'SecType', 'Currency', 'Exchange', 'PrimaryExchange', 'LocalSymbol', 'LastTradeDateOrContractMonth', 'ValidExchanges', 'TradingClass', 'MarketName', 'LongName', 'ConId', 'MinTick', 'MinSize', 'SizeIncrement', 'SuggestedSizeIncrement', 'Multiplier', 'TimeZoneId', 'OrderTypes', 'LiquidHours', 'TradingHours', 'datetime'],
}


def _empty_workbook_frames():
    return {sheet: pd.DataFrame(columns=columns) for sheet, columns in WORKBOOK_SCHEMAS.items()}


def _normalize_sheet(df, columns):
    df = df.copy() if df is not None else pd.DataFrame()
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    extra = [col for col in df.columns if col not in columns]
    ordered = list(columns) + extra
    return df.loc[:, ordered]


def ensure_trading_info_workbook(smtp_username, to_email, password, database_path='data/database.xlsx', email_info_path='data/email_info.xlsx'):
    os.makedirs(os.path.dirname(database_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(email_info_path) or '.', exist_ok=True)

    dictfiles = _empty_workbook_frames()
    if os.path.exists(database_path):
        try:
            database = pd.ExcelFile(database_path)
            for sheet_name, columns in WORKBOOK_SCHEMAS.items():
                if sheet_name in database.sheet_names:
                    dictfiles[sheet_name] = _normalize_sheet(database.parse(sheet_name), columns)
        except Exception:
            dictfiles = _empty_workbook_frames()

    tf.save_xlsx(dict_df=dictfiles, path=database_path)

    email_password = pd.DataFrame(columns=['smtp_username', 'to_email', 'password'], index=[0])
    email_password.loc[0, 'smtp_username'] = smtp_username
    email_password.loc[0, 'to_email'] = to_email
    email_password.loc[0, 'password'] = password
    email_password.to_excel(email_info_path)


def create_trading_info_workbook(smtp_username, to_email, password, database_path='data/database.xlsx', email_info_path='data/email_info.xlsx'):
    ensure_trading_info_workbook(
        smtp_username=smtp_username,
        to_email=to_email,
        password=password,
        database_path=database_path,
        email_info_path=email_info_path,
    )
