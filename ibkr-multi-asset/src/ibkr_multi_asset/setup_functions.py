"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

# Import the necessary libraries
import os
import math
import time
import smtplib
import inspect
import json
from copy import deepcopy
from decimal import Decimal
from email.message import EmailMessage
import numpy as np
import pandas as pd
import datetime as dt
import pytz
import yfinance as yf
from ibkr_multi_asset import trading_functions as tf
from ibkr_multi_asset import ib_functions as ibf
from ibkr_multi_asset.create_database import WORKBOOK_SCHEMAS
from ibkr_multi_asset.report_generator import generate_live_portfolio_report
from ibkr_multi_asset.strategy_runtime import stra, get_strategy_file
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import threading
from ibapi.order_cancel import OrderCancel


ACCOUNT_SUMMARY_TAGS = ",".join([
    "AccountType",
    "NetLiquidation",
    "TotalCashValue",
    "AvailableFunds",
    "ExcessLiquidity",
    "BuyingPower",
    "GrossPositionValue",
    "FullInitMarginReq",
    "FullMaintMarginReq",
    "InitMarginReq",
    "MaintMarginReq",
    "LookAheadAvailableFunds",
    "LookAheadExcessLiquidity",
    "LookAheadInitMarginReq",
    "LookAheadMaintMarginReq",
    "Leverage",
    "Cushion",
    "DayTradesRemaining",
])

CRYPTO_FILL_WAIT_SECONDS = 20
CRYPTO_STOP_POLL_SECONDS = 1.0


def _main_config_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "user_config", "main.py")


def _flush_temp_sheet(app, temp_attr, target_attr, sheet_name, dedupe_subset=None):
    temp_df = getattr(app, temp_attr, pd.DataFrame())
    if temp_df.empty:
        return

    frame = temp_df.copy()
    if 'datetime' in frame.columns:
        frame['datetime'] = pd.to_datetime(frame['datetime'], errors='coerce')
        frame = frame.dropna(subset=['datetime'])
        frame.set_index('datetime', inplace=True)
        frame.index.name = ''

    target_df = _frame_with_datetime_column(getattr(app, target_attr, pd.DataFrame()).copy())
    if 'datetime' in target_df.columns:
        target_df['datetime'] = pd.to_datetime(target_df['datetime'], errors='coerce')
        target_df = target_df.dropna(subset=['datetime'])
        target_df.set_index('datetime', inplace=True)
        target_df.index.name = ''
    target_df = pd.concat([target_df, frame])
    target_df = target_df[~target_df.index.isna()]

    if dedupe_subset:
        target_df.reset_index(inplace=True)
        target_df = _frame_with_datetime_column(target_df)
        if 'datetime' not in target_df.columns:
            target_df['datetime'] = pd.to_datetime(target_df.index, errors='coerce')
        target_df.sort_values('datetime', inplace=True)
        target_df.drop_duplicates(subset=[col for col in dedupe_subset if col in target_df.columns], keep='last', inplace=True)
        target_df.set_index('datetime', inplace=True)
        target_df.index.name = ''
    else:
        target_df.drop_duplicates(inplace=True)
        target_df.sort_index(ascending=True, inplace=True)

    setattr(app, target_attr, target_df)
    setattr(app, temp_attr, pd.DataFrame(columns=WORKBOOK_SCHEMAS[sheet_name]))


def _flush_live_trading_buffers(app):
    if getattr(app, 'temp_open_orders', pd.DataFrame()).empty is False:
        temp_df = app.temp_open_orders.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df.index.name = ''
        app.open_orders = pd.concat([app.open_orders, temp_df])
        app.open_orders.drop_duplicates(inplace=True)
        app.open_orders.sort_index(ascending=True, inplace=True)
        app.temp_open_orders = pd.DataFrame()

    if getattr(app, 'temp_orders_status', pd.DataFrame()).empty is False:
        temp_df = app.temp_orders_status.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df.index.name = ''
        app.orders_status = pd.concat([app.orders_status, temp_df])
        app.orders_status.drop_duplicates(inplace=True)
        app.orders_status.sort_index(ascending=True, inplace=True)
        app.temp_orders_status = pd.DataFrame()

    if getattr(app, 'temp_exec_df', pd.DataFrame()).empty is False:
        temp_df = app.temp_exec_df.copy()
        if 'Execution Time' in temp_df.columns:
            temp_df['Execution Time'] = pd.to_datetime(temp_df['Execution Time'].replace(rf"{app.zone}", "", regex=True).values)
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df.index.name = ''
        app.exec_df = pd.concat([app.exec_df, temp_df])
        app.exec_df.drop_duplicates(inplace=True)
        app.exec_df.sort_index(ascending=True, inplace=True)
        app.temp_exec_df = pd.DataFrame()

    if getattr(app, 'temp_comm_df', pd.DataFrame()).empty is False:
        temp_df = app.temp_comm_df.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df.index.name = ''
        if 'Realized PnL' in temp_df.columns:
            mask = pd.to_numeric(temp_df['Realized PnL'], errors='coerce') == 1.7976931348623157e+308
            temp_df.loc[mask, 'Realized PnL'] = np.nan
        app.comm_df = pd.concat([app.comm_df, temp_df])
        app.comm_df.drop_duplicates(inplace=True)
        app.comm_df.sort_index(ascending=True, inplace=True)
        app.temp_comm_df = pd.DataFrame()

    if getattr(app, 'temp_pos_df', pd.DataFrame()).empty is False:
        temp_df = app.temp_pos_df.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
        temp_df = temp_df.dropna(subset=['datetime'])
        temp_df.set_index('datetime', inplace=True)
        temp_df.index.name = ''
        app.pos_df = _merge_position_snapshots(app.pos_df, temp_df)
        app.temp_pos_df = pd.DataFrame()


def _flush_contract_details_buffer(app):
    temp_df = getattr(app, 'temp_contract_details', pd.DataFrame())
    if temp_df.empty:
        return

    existing = getattr(app, 'contract_details_df', pd.DataFrame())
    existing = existing.copy() if isinstance(existing, pd.DataFrame) else pd.DataFrame(columns=WORKBOOK_SCHEMAS['contract_details'])
    incoming = temp_df.copy()

    if 'datetime' in existing.columns:
        existing['datetime'] = pd.to_datetime(existing['datetime'], errors='coerce')
    if 'datetime' in incoming.columns:
        incoming['datetime'] = pd.to_datetime(incoming['datetime'], errors='coerce')

    merged = pd.concat([existing, incoming], ignore_index=True)
    dedupe_subset = [col for col in ['Symbol', 'ConId', 'LocalSymbol', 'Exchange'] if col in merged.columns]
    if dedupe_subset:
        merged.drop_duplicates(subset=dedupe_subset, keep='last', inplace=True)
    else:
        merged.drop_duplicates(inplace=True)
    if 'datetime' in merged.columns:
        merged = merged.sort_values('datetime', kind='stable', na_position='last')
    merged.reset_index(drop=True, inplace=True)
    app.contract_details_df = merged
    app.temp_contract_details = pd.DataFrame(columns=WORKBOOK_SCHEMAS['contract_details'])


def _safe_datetime_series(series):
    source = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(source):
        return pd.to_datetime(source, errors='coerce')
    if pd.api.types.is_numeric_dtype(source):
        return pd.Series(pd.NaT, index=source.index)
    return pd.to_datetime(source, errors='coerce')


def _merge_position_snapshots(existing_df, incoming_df):
    existing = _frame_with_datetime_column(existing_df.copy()) if isinstance(existing_df, pd.DataFrame) else pd.DataFrame()
    incoming = _frame_with_datetime_column(incoming_df.copy()) if isinstance(incoming_df, pd.DataFrame) else pd.DataFrame()
    if existing.empty and incoming.empty:
        return pd.DataFrame()
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if 'datetime' in combined.columns:
        combined['datetime'] = pd.to_datetime(combined['datetime'], errors='coerce')
        combined = combined.dropna(subset=['datetime'])
    dedupe_subset = [col for col in ['datetime', 'Account', 'Symbol', 'SecType', 'Currency', 'Position', 'Avg cost'] if col in combined.columns]
    if dedupe_subset:
        combined = combined.drop_duplicates(subset=dedupe_subset, keep='last')
    else:
        combined = combined.drop_duplicates(keep='last')
    combined = combined.sort_values('datetime', kind='stable')
    combined = combined.set_index('datetime')
    combined.index.name = ''
    return combined


def _contract_row_is_resolved(row, asset_class):
    asset_class = str(asset_class or '').lower()
    if row is None:
        return False
    if asset_class in {'futures', 'future', 'fut'}:
        local_symbol = str(row.get('LocalSymbol', '') or '').strip()
        con_id = pd.to_numeric(pd.Series([row.get('ConId', np.nan)]), errors='coerce').iloc[0]
        return bool(local_symbol) or pd.notna(con_id)
    return True


def _frame_with_datetime_column(df):
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    blank_named_columns = [col for col in out.columns if str(col).strip() == '']
    if blank_named_columns:
        out = out.drop(columns=blank_named_columns, errors='ignore')
    for candidate in ['Unnamed: 0', 'index', 'level_0']:
        if 'datetime' not in out.columns and candidate in out.columns:
            source = out[candidate]
            looks_textual = source.astype(str).str.contains(r'[-/: T]', regex=True, na=False).any()
            parsed = pd.to_datetime(source, errors='coerce')
            if looks_textual and parsed.notna().any():
                out = out.rename(columns={candidate: 'datetime'})
                break
    unnamed_cols = [col for col in out.columns if str(col).startswith('Unnamed:')]
    if unnamed_cols:
        out = out.drop(columns=unnamed_cols)
    if out.empty:
        return out
    if isinstance(out.index, pd.DatetimeIndex):
        index_datetimes = pd.Series(pd.to_datetime(out.index, errors='coerce'), index=out.index)
        if 'datetime' not in out.columns:
            out = out.copy()
            if out.index.name in (None, ''):
                out.index = out.index.rename('datetime')
            elif out.index.name in out.columns:
                out = out.drop(columns=[out.index.name], errors='ignore')
            out = out.reset_index()
            first_col = out.columns[0]
            if first_col != 'datetime':
                out = out.rename(columns={first_col: 'datetime'})
        else:
            existing_dt = pd.to_datetime(out['datetime'], errors='coerce')
            if existing_dt.notna().sum() == 0:
                out = out.copy()
                out['datetime'] = index_datetimes.values
    if 'datetime' in out.columns:
        out['datetime'] = _safe_datetime_series(out['datetime'])
    return out


def append_runtime_audit(app, event, detail=''):
    entry = {
        'Symbol': app.ticker,
        'event': event,
        'detail': str(detail),
        'datetime': dt.datetime.now().replace(microsecond=0),
    }
    app.temp_runtime_audit = pd.concat([app.temp_runtime_audit, pd.DataFrame(entry, index=[0])], ignore_index=True)


def _normalize_sheet_to_datetime_index(frame, invalid_before_year=None):
    local = _frame_with_datetime_column(frame)
    if local.empty or 'datetime' not in local.columns:
        return local
    local['datetime'] = _safe_datetime_series(local['datetime'])
    local = local.dropna(subset=['datetime'])
    if invalid_before_year is not None:
        local = local[local['datetime'].dt.year >= int(invalid_before_year)]
    local = local.reset_index(drop=True)
    local = local.set_index('datetime')
    local.index.name = ''
    return local


def _portfolio_snapshots_from_positions(positions_frame):
    local = _frame_with_datetime_column(positions_frame)
    if local.empty:
        return pd.DataFrame(columns=WORKBOOK_SCHEMAS['portfolio_snapshots'])
    for column in ['Account', 'Symbol', 'SecType', 'Currency', 'Position', 'Avg cost', 'datetime']:
        if column not in local.columns:
            local[column] = pd.NA
    out = pd.DataFrame({
        'datetime': _safe_datetime_series(local['datetime']),
        'Account': local['Account'],
        'Symbol': local['Symbol'],
        'LocalSymbol': local.get('Symbol', pd.Series(dtype='object')),
        'SecType': local['SecType'],
        'Exchange': pd.NA,
        'Currency': local['Currency'],
        'ConId': pd.NA,
        'Position': pd.to_numeric(local['Position'], errors='coerce'),
        'MarketPrice': np.nan,
        'MarketValue': np.nan,
        'AverageCost': pd.to_numeric(local['Avg cost'], errors='coerce'),
        'UnrealizedPnL': np.nan,
        'RealizedPnL': np.nan,
        'market_open_time': local['market_open_time'] if 'market_open_time' in local.columns else pd.NaT,
        'market_close_time': local['market_close_time'] if 'market_close_time' in local.columns else pd.NaT,
    })
    out = out.dropna(subset=['datetime', 'Symbol']).drop_duplicates(subset=['datetime', 'Account', 'Symbol'], keep='last')
    if out.empty:
        return pd.DataFrame(columns=WORKBOOK_SCHEMAS['portfolio_snapshots'])
    out = out.set_index('datetime')
    out.index.name = ''
    return out


def _repair_account_update_times_frame(frame, current_period=None):
    local = _frame_with_datetime_column(frame)
    if local.empty:
        return pd.DataFrame(columns=['datetime', 'Account', 'UpdateTime'])
    if 'datetime' in local.columns:
        local['datetime'] = _safe_datetime_series(local['datetime'])
    else:
        local['datetime'] = pd.NaT
    if 'UpdateTime' in local.columns and current_period is not None:
        bad_mask = local['datetime'].isna() | (local['datetime'].dt.year < 2000)
        if bad_mask.any():
            time_text = local.loc[bad_mask, 'UpdateTime'].astype(str)
            rebuilt = pd.to_datetime(
                current_period.strftime('%Y-%m-%d') + ' ' + time_text,
                errors='coerce'
            )
            local.loc[bad_mask, 'datetime'] = rebuilt.values
    local = local.dropna(subset=['datetime'])
    local = local.reset_index(drop=True).set_index('datetime')
    local.index.name = ''
    return local


def _flush_runtime_audit_buffer(app):
    temp_df = getattr(app, 'temp_runtime_audit', pd.DataFrame())
    base_df = _normalize_sheet_to_datetime_index(getattr(app, 'runtime_audit', pd.DataFrame()))
    if temp_df.empty:
        app.runtime_audit = base_df
        return
    audit_df = temp_df.copy()
    audit_df['datetime'] = _safe_datetime_series(audit_df['datetime'])
    audit_df = audit_df.dropna(subset=['datetime'])
    audit_df = audit_df.set_index('datetime')
    audit_df.index.name = ''
    app.runtime_audit = pd.concat([base_df, audit_df])
    app.runtime_audit.drop_duplicates(inplace=True)
    app.runtime_audit.sort_index(ascending=True, inplace=True)
    app.temp_runtime_audit = pd.DataFrame(columns=['Symbol', 'event', 'detail', 'datetime'])


def _normalize_strategy_state_frame(frame):
    local = _frame_with_datetime_column(frame)
    if local.empty:
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue'])
    extra_reset_columns = [col for col in local.columns if str(col).startswith('level_')]
    if extra_reset_columns:
        local = local.drop(columns=extra_reset_columns, errors='ignore')
    local['datetime'] = _safe_datetime_series(local['datetime'])
    local = local.dropna(subset=['datetime'])
    if 'Symbol' in local.columns:
        local['Symbol'] = local['Symbol'].fillna('PORTFOLIO').astype(str).replace({'': 'PORTFOLIO'}).str.upper()
    local.index = local.index.set_names([None] * getattr(local.index, 'nlevels', 1))
    try:
        local = local.reset_index(drop=True)
    except ValueError:
        conflict_names = [name for name in local.index.names if isinstance(name, str)]
        conflict_columns = [col for col in local.columns if str(col) in conflict_names]
        if conflict_columns:
            local = local.drop(columns=conflict_columns, errors='ignore')
        local.index = local.index.set_names([None] * getattr(local.index, 'nlevels', 1))
        local = local.reset_index(drop=True)
    local.sort_values('datetime', inplace=True)
    dedupe_keys = [col for col in ['Symbol', 'Scope', 'StateKey'] if col in local.columns]
    if dedupe_keys:
        local.drop_duplicates(subset=dedupe_keys, keep='last', inplace=True)
    local.set_index('datetime', inplace=True)
    local.index.name = ''
    return local


def _strategy_state_with_datetime_column(frame):
    normalized = _normalize_strategy_state_frame(frame)
    if normalized.empty:
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])

    local = normalized.reset_index()
    if 'datetime' not in local.columns:
        first_column = local.columns[0]
        local = local.rename(columns={first_column: 'datetime'})

    columns = ['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime']
    for column in columns:
        if column not in local.columns:
            local[column] = pd.NA

    local['datetime'] = _safe_datetime_series(local['datetime'])
    local = local.dropna(subset=['datetime'])
    local = local.loc[:, columns]
    local.sort_values('datetime', inplace=True)
    local.reset_index(drop=True, inplace=True)
    return local


def _strategy_state_store_path(database_path):
    base_dir = os.path.dirname(database_path) or '.'
    return os.path.join(base_dir, 'strategy_state.json')


def _load_database_sheets(database_path):
    sheets = {
        sheet_name: pd.DataFrame(columns=columns)
        for sheet_name, columns in WORKBOOK_SCHEMAS.items()
    }

    if not database_path or not os.path.exists(database_path):
        return sheets

    try:
        database = pd.ExcelFile(database_path)
    except Exception:
        return sheets

    for sheet_name, columns in WORKBOOK_SCHEMAS.items():
        if sheet_name not in database.sheet_names:
            continue
        try:
            frame = database.parse(sheet_name)
        except Exception:
            continue
        unnamed = [column for column in frame.columns if str(column).startswith('Unnamed:')]
        if unnamed:
            frame = frame.drop(columns=unnamed, errors='ignore')
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        sheets[sheet_name] = frame.loc[:, [column for column in columns if column in frame.columns]]
    return sheets


def _load_strategy_state(database_path):
    path = _strategy_state_store_path(database_path)
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except Exception:
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    rows = payload.get('rows', []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
    return _strategy_state_with_datetime_column(pd.DataFrame(rows))


def _save_strategy_state(database_path, state_df):
    normalized = _strategy_state_with_datetime_column(state_df)
    rows = []
    if not normalized.empty:
        export_df = normalized.copy()
        export_df['datetime'] = pd.to_datetime(export_df['datetime'], errors='coerce')
        export_df = export_df.dropna(subset=['datetime'])
        export_df['datetime'] = export_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        rows = export_df.to_dict(orient='records')
    path = _strategy_state_store_path(database_path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'rows': rows}, handle, indent=2)


def _flush_strategy_state_buffer(app):
    base_df = _normalize_strategy_state_frame(getattr(app, 'strategy_state_df', pd.DataFrame()))
    temp_df = getattr(app, 'temp_strategy_state', pd.DataFrame())
    if temp_df.empty:
        app.strategy_state_df = base_df
        return
    state_df = temp_df.copy()
    if 'Symbol' in state_df.columns:
        state_df['Symbol'] = state_df['Symbol'].fillna('PORTFOLIO').astype(str).replace({'': 'PORTFOLIO'}).str.upper()
    state_df['datetime'] = _safe_datetime_series(state_df['datetime'])
    state_df = state_df.dropna(subset=['datetime'])
    state_df.set_index('datetime', inplace=True)
    state_df.index.name = ''
    app.strategy_state_df = pd.concat([base_df, state_df])
    app.strategy_state_df = _normalize_strategy_state_frame(app.strategy_state_df)
    app.temp_strategy_state = pd.DataFrame(columns=app.strategy_state_df.columns)


def connection_monitor(app):
    ''' Check continuously if there's a need to disconnect the app '''

    while True:
        # If the app is disconnected
        if not app.isConnected():
            print("Not connected. Breaking loop...")
            app.logging.info("Not connected. Breaking loop.")
            stop(app)
            break
        # If the app is disconnected based on the errors' dictionary
        if (502 in list(app.errors_dict.keys())):
            print("Not connected. Breaking loop...")
            app.logging.info("Not connected. Breaking loop.")
            stop(app)
            break
        if app.last_value_count >= 50:
            print("count got to 50. Let's disconnect...")
            print("Not connected. Breaking loop.")
            app.logging.info("Not connected. Breaking loop.")
            stop(app)
            break
        # If the strategy was finished correctly
        if app.strategy_end == True:
            print("Strategy is done. Let's disconnect...")
            app.logging.info("Strategy is done. Let's disconnect...")
            stop(app)
            break      
        if (1100 in list(app.errors_dict.keys())):
            print("Not connected. Breaking loop...")
            app.logging.info("Not connected. Breaking loop.")
            stop(app)
            break
        
def request_orders(app, verbose=True):
    ''' Function to request the open orders and orders status'''
    if not _shared_cycle_symbol_broker_pull_allowed(app):
        return
    with app.broker_sync_lock:
        if verbose:
            print(f'[{app.ticker}] Requesting open positions and orders status...')
            app.logging.info(f'[{app.ticker}] Requesting open positions and orders status...')
        
        if app.isConnected():
            app.orders_request_event.clear()
            previous_silent = bool(getattr(app, 'silent_broker_sync', False))
            current_depth = int(getattr(app, 'silent_broker_sync_depth', 0))
            app.silent_broker_sync = not verbose
            if not verbose:
                app.silent_broker_sync_depth = current_depth + 1
            try:
                app.reqOpenOrders()
                app.orders_request_event.wait()
            finally:
                app.silent_broker_sync = previous_silent
                app.silent_broker_sync_depth = current_depth
        else:
            return
        
        app.current_open_orders_snapshot = app.temp_open_orders.copy()

        if (app.temp_open_orders.empty == False):
            app.temp_open_orders.set_index('datetime', inplace=True)
            app.temp_open_orders.index.name = ''
            app.open_orders = pd.concat([app.open_orders, app.temp_open_orders])
            app.open_orders.drop_duplicates(inplace=True)
            app.open_orders.sort_index(ascending=True, inplace=True)
            app.temp_open_orders = pd.DataFrame()

        if (app.temp_orders_status.empty == False):
            app.temp_orders_status.set_index('datetime', inplace=True)
            app.temp_orders_status.index.name = ''
            app.orders_status = pd.concat([app.orders_status, app.temp_orders_status])
            app.orders_status.drop_duplicates(inplace=True)
            app.orders_status.sort_index(ascending=True, inplace=True)
            app.temp_orders_status = pd.DataFrame()

        if verbose:
            print(f'[{app.ticker}] Open positions and orders status successfully requested...')
            app.logging.info(f'[{app.ticker}] Open positions and orders status successfully requested...')


def _current_open_orders_snapshot(app):
    shared_state = getattr(app, 'shared_broker_state', {}) or {}
    snapshot = shared_state.get('current_open_orders_snapshot', pd.DataFrame()) if _shared_cycle_quiet_mode(app) else getattr(app, 'current_open_orders_snapshot', pd.DataFrame())
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        return pd.DataFrame()
    return _filter_orders_for_contract(snapshot.copy(), app)


def _active_open_orders_snapshot(app):
    local = _current_open_orders_snapshot(app)
    if local.empty or 'Status' not in local.columns:
        return local
    active_statuses = {'SUBMITTED', 'PRESUBMITTED', 'PENDINGSUBMIT', 'APIPENDING'}
    status_values = local['Status'].astype(str).str.upper().str.strip()
    return local[status_values.isin(active_statuses)].copy()


def _required_risk_management_action(position_quantity):
    if float(position_quantity) > 0:
        return 'SELL'
    if float(position_quantity) < 0:
        return 'BUY'
    return ''


def _filter_orders_for_contract(orders, app):
    if not isinstance(orders, pd.DataFrame) or orders.empty:
        return pd.DataFrame()
    local = orders.copy()
    symbol = str(getattr(app.contract, 'symbol', app.ticker)).upper()
    currency = str(getattr(app.contract, 'currency', '') or '').upper()
    if 'Symbol' in local.columns:
        local = local[local['Symbol'].astype(str).str.upper() == symbol]
    if currency and 'Currency' in local.columns:
        raw_currency_values = local['Currency']
        valid_currency_mask = raw_currency_values.notna()
        if valid_currency_mask.any():
            currency_values = raw_currency_values.astype(str).str.upper()
            local = local[currency_values == currency]
    return local


def _latest_strategy_state_risk_management_prices(app):
    state = getattr(app, 'strategy_state', {}) or {}
    rm_state = state.get('risk_management', {}) if isinstance(state, dict) else {}
    if not isinstance(rm_state, dict):
        return {}
    prices = {}
    for source_key, target_key in [('sl_price', 'sl'), ('tp_price', 'tp')]:
        value = pd.to_numeric(pd.Series([rm_state.get(source_key, np.nan)]), errors='coerce').iloc[0]
        if np.isfinite(value):
            prices[target_key] = float(value)
    return prices


def _latest_position_average_cost(app):
    positions = getattr(app, 'pos_df', pd.DataFrame())
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return np.nan
    local = positions.copy()
    symbol = str(getattr(app.contract, 'symbol', app.ticker)).upper()
    currency = str(getattr(app.contract, 'currency', '') or '').upper()
    if 'Symbol' in local.columns:
        local = local[local['Symbol'].astype(str).str.upper() == symbol]
    if currency and 'Currency' in local.columns:
        local = local[local['Currency'].astype(str).str.upper() == currency]
    if local.empty or 'Avg cost' not in local.columns:
        return np.nan
    if 'Position' in local.columns:
        nonzero = pd.to_numeric(local['Position'], errors='coerce').fillna(0.0).abs() > 1e-8
        if nonzero.any():
            local = local.loc[nonzero].copy()
    avg_costs = pd.to_numeric(local['Avg cost'], errors='coerce').dropna()
    if avg_costs.empty:
        return np.nan
    return float(avg_costs.iloc[-1])


def _select_price_near_reference(price_series, reference_price):
    values = pd.to_numeric(price_series, errors='coerce').dropna()
    if values.empty:
        return np.nan
    if not np.isfinite(reference_price):
        return float(values.iloc[-1])
    deltas = (values - float(reference_price)).abs()
    return float(values.iloc[deltas.argmin()])


def _latest_historical_risk_management_prices(app):
    local = _frame_with_datetime_column(getattr(app, 'open_orders', pd.DataFrame()))
    if not isinstance(local, pd.DataFrame) or local.empty:
        return {}
    orders = local.copy().reset_index(drop=True)
    if 'datetime' in orders.columns:
        orders['datetime'] = pd.to_datetime(orders['datetime'], errors='coerce')
        if orders['datetime'].notna().any():
            orders = orders.sort_values('datetime')
    elif 'OrderId' in orders.columns:
        orders = orders.sort_values('OrderId')
    orders = _filter_orders_for_contract(orders, app)
    if orders.empty:
        return {}

    prices = {}
    reference_price = _latest_position_average_cost(app)
    stop_types = {'TRAIL'} if bool(getattr(app, 'trail', False)) else {'STP', 'TRAIL'}
    if 'OrderType' in orders.columns:
        stop_rows = orders[orders['OrderType'].astype(str).str.upper().isin(stop_types)].copy()
        tp_rows = orders[orders['OrderType'].astype(str).str.upper() == 'LMT'].copy()
    else:
        stop_rows = pd.DataFrame()
        tp_rows = pd.DataFrame()

    if not stop_rows.empty and 'AuxPrice' in stop_rows.columns:
        stop_price = _select_price_near_reference(stop_rows['AuxPrice'], reference_price)
        if np.isfinite(stop_price):
            prices['sl'] = float(stop_price)
    if not tp_rows.empty and 'LmtPrice' in tp_rows.columns:
        tp_price = _select_price_near_reference(tp_rows['LmtPrice'], reference_price)
        if np.isfinite(tp_price):
            prices['tp'] = float(tp_price)
    return prices


def _carry_prices_valid_for_market(live_quantity, last_value, carry_prices):
    if not np.isfinite(last_value):
        return False
    sl_price = pd.to_numeric(pd.Series([carry_prices.get('sl', np.nan)]), errors='coerce').iloc[0]
    tp_price = pd.to_numeric(pd.Series([carry_prices.get('tp', np.nan)]), errors='coerce').iloc[0]
    if not (np.isfinite(sl_price) and np.isfinite(tp_price)):
        return False
    if live_quantity > 0:
        return float(sl_price) < float(last_value) < float(tp_price)
    if live_quantity < 0:
        return float(tp_price) < float(last_value) < float(sl_price)
    return False

def request_positions(app, verbose=True):
    ''' Function to request the trading positions'''
    if not _shared_cycle_symbol_broker_pull_allowed(app):
        return
    with app.broker_sync_lock:
        if verbose:
            print(f'[{app.ticker}] Requesting positions...')
            app.logging.info(f'[{app.ticker}] Requesting positions...')
        
        if app.isConnected():
            app.positions_request_event.clear()
            app.temp_pos_df = pd.DataFrame()
            previous_silent = bool(getattr(app, 'silent_broker_sync', False))
            current_depth = int(getattr(app, 'silent_broker_sync_depth', 0))
            app.silent_broker_sync = not verbose
            if not verbose:
                app.silent_broker_sync_depth = current_depth + 1
            try:
                app.reqPositions()
                app.positions_request_event.wait()
            finally:
                app.silent_broker_sync = previous_silent
                app.silent_broker_sync_depth = current_depth
        else:
            return

        batch_df = app.temp_pos_df.copy() if isinstance(getattr(app, 'temp_pos_df', pd.DataFrame()), pd.DataFrame) else pd.DataFrame()
        if not batch_df.empty:
            symbol_value = str(getattr(app.contract, 'symbol', app.ticker)).upper()
            sec_type_value = str(getattr(app.contract, 'secType', '')).upper()
            currency_value = str(getattr(app.contract, 'currency', '') or '').upper()
            symbol_mask = batch_df['Symbol'].astype(str).str.upper() == symbol_value if 'Symbol' in batch_df.columns else pd.Series(False, index=batch_df.index)
            sec_type_mask = batch_df['SecType'].astype(str).str.upper() == sec_type_value if 'SecType' in batch_df.columns else pd.Series(True, index=batch_df.index)
            current_mask = symbol_mask & sec_type_mask
            if currency_value and 'Currency' in batch_df.columns:
                current_mask = current_mask & (batch_df['Currency'].astype(str).str.upper() == currency_value)
            has_current_symbol = bool(current_mask.any())
        else:
            has_current_symbol = False

        if not has_current_symbol:
            zero_row = pd.DataFrame([{
                'Account': getattr(app, 'account', pd.NA),
                'Symbol': str(getattr(app.contract, 'symbol', app.ticker)),
                'SecType': str(getattr(app.contract, 'secType', pd.NA)),
                'Currency': str(getattr(app.contract, 'currency', pd.NA)),
                'Position': 0.0,
                'Avg cost': np.nan,
                'datetime': dt.datetime.now().replace(microsecond=0),
            }])
            app.temp_pos_df = pd.concat([app.temp_pos_df, zero_row], ignore_index=True)
        
        if (app.temp_pos_df.empty == False):
            pd.set_option('display.max_columns', None)
            app.temp_pos_df.set_index('datetime', inplace=True)
            app.temp_pos_df.index.name = ''
            app.pos_df = _merge_position_snapshots(app.pos_df, app.temp_pos_df)
            app.temp_pos_df = pd.DataFrame()
        if verbose:
            print('Open positions successfully requested...')
            app.logging.info('Open positions successfully requested...')


def request_contract_details(app, verbose=True):
    ''' Function to request the instrument contract details '''
    if verbose:
        print('Requesting contract details...')
        app.logging.info('Requesting contract details...')

    def _hydrate_contract_from_row(row):
        contract = ibf.build_contract_from_spec(app.asset_spec)
        for attr, column in (
            ('symbol', 'Symbol'),
            ('secType', 'SecType'),
            ('currency', 'Currency'),
            ('exchange', 'Exchange'),
            ('primaryExchange', 'PrimaryExchange'),
            ('localSymbol', 'LocalSymbol'),
            ('lastTradeDateOrContractMonth', 'LastTradeDateOrContractMonth'),
            ('tradingClass', 'TradingClass'),
            ('multiplier', 'Multiplier'),
        ):
            if column in row and pd.notna(row[column]) and str(row[column]) != '':
                normalized = _normalized_contract_field(attr, row[column])
                if normalized not in (None, ''):
                    setattr(contract, attr, normalized)
        if 'ConId' in row and pd.notna(row['ConId']):
            normalized_conid = _normalized_contract_field('conId', row['ConId'])
            if normalized_conid is not None:
                contract.conId = normalized_conid
        return contract

    existing = app.contract_details_df.copy()
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        if 'datetime' not in existing.columns and isinstance(existing.index, pd.DatetimeIndex):
            existing = existing.reset_index().rename(columns={'index': 'datetime'})
        elif not isinstance(existing.index, pd.RangeIndex):
            existing = existing.reset_index(drop=True)
        if 'datetime' in existing.columns:
            existing['datetime'] = pd.to_datetime(existing['datetime'], errors='coerce')
    if not existing.empty and 'Symbol' in existing.columns:
        same_symbol = existing[existing['Symbol'].astype(str).str.upper() == str(app.ticker).upper()]
        if not same_symbol.empty:
            candidate_row = _select_contract_details_row(app, same_symbol)
            asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
            if candidate_row is not None and _contract_row_is_resolved(candidate_row, asset_class):
                hydrated = _hydrate_contract_from_row(candidate_row)
                app.contract = hydrated
                app.resolved_contract = deepcopy(hydrated)
                append_runtime_audit(app, 'contract_details_cached', f'{app.ticker} contract details already available')
                app.contract_details_df = existing
                return

    if app.isConnected():
        app.contract_details_event.clear()
        if verbose:
            print(f"[{app.ticker}] Requesting contract details for: Symbol={app.contract.symbol}, SecType={app.contract.secType}, Exchange={app.contract.exchange}, Multiplier={getattr(app.contract, 'multiplier', 'None')}")
        app.reqContractDetails(9000, app.contract)
        app.contract_details_event.wait(timeout=30)
    else:
        return

    # Debug: show what was received
    rows_received = len(app.temp_contract_details)
    if rows_received == 0 and verbose:
        print(f"[{app.ticker}] No contract details received from IBKR.")
    elif verbose:
        unique_symbols = app.temp_contract_details['Symbol'].unique().tolist()
        print(f"[{app.ticker}] Received {rows_received} contract detail rows from IBKR. Symbols found: {unique_symbols}")

    if not app.temp_contract_details.empty:
        refreshed = app.temp_contract_details.copy()
        if 'datetime' in refreshed.columns:
            refreshed['datetime'] = pd.to_datetime(refreshed['datetime'], errors='coerce')
        merged = pd.concat([existing, refreshed], ignore_index=True)
        merged.drop_duplicates(subset=['Symbol', 'ConId'], keep='last', inplace=True)
        if 'datetime' in merged.columns:
            merged = merged.sort_values('datetime', kind='stable', na_position='last')
        merged.reset_index(drop=True, inplace=True)
        app.contract_details_df = merged
        same_symbol = merged[merged['Symbol'].astype(str).str.upper() == str(app.ticker).upper()]
        if not same_symbol.empty:
            candidate_row = _select_contract_details_row(app, same_symbol)
            if candidate_row is not None:
                hydrated = _hydrate_contract_from_row(candidate_row)
                app.contract = hydrated
                app.resolved_contract = deepcopy(hydrated)
        elif getattr(app, 'resolved_contract', None) is not None:
            app.contract = deepcopy(app.resolved_contract)
        append_runtime_audit(app, 'contract_details_refreshed', f'rows={len(app.temp_contract_details)}')
        app.temp_contract_details = pd.DataFrame(columns=app.contract_details_df.columns)
    if verbose:
        print('Contract details successfully requested...')
        app.logging.info('Contract details successfully requested...')

def _contract_for_order(app):
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    if getattr(app, 'resolved_contract', None) is not None:
        return deepcopy(app.resolved_contract)

    if asset_class in {'futures', 'future', 'fut', 'metals', 'metal', 'stock', 'stocks', 'equity', 'equities', 'stk'}:
        request_contract_details(app, verbose=False)
        if getattr(app, 'resolved_contract', None) is not None:
            return deepcopy(app.resolved_contract)
        return app.contract

    details = getattr(app, 'contract_details_df', None)
    if isinstance(details, pd.DataFrame) and not details.empty and 'Symbol' in details.columns:
        same_symbol = details[details['Symbol'].astype(str).str.upper() == str(app.ticker).upper()]
        if not same_symbol.empty:
            resolved_rows = same_symbol[same_symbol.apply(lambda row: _contract_row_is_resolved(row, asset_class), axis=1)]
            candidate_rows = resolved_rows if not resolved_rows.empty else same_symbol
            row = _select_contract_details_row(app, candidate_rows)
            if row is None:
                row = candidate_rows.iloc[-1]
            contract = ibf.build_contract_from_spec(app.asset_spec)
            for attr, column in (
                ('symbol', 'Symbol'),
                ('secType', 'SecType'),
                ('currency', 'Currency'),
                ('exchange', 'Exchange'),
                ('primaryExchange', 'PrimaryExchange'),
                ('localSymbol', 'LocalSymbol'),
                ('lastTradeDateOrContractMonth', 'LastTradeDateOrContractMonth'),
                ('tradingClass', 'TradingClass'),
                ('multiplier', 'Multiplier'),
            ):
                if column in row and pd.notna(row[column]) and str(row[column]) != '':
                    normalized = _normalized_contract_field(attr, row[column])
                    if normalized not in (None, ''):
                        setattr(contract, attr, normalized)
            if 'ConId' in row and pd.notna(row['ConId']):
                normalized_conid = _normalized_contract_field('conId', row['ConId'])
                if normalized_conid is not None:
                    contract.conId = normalized_conid
            return contract

    if getattr(app, 'resolved_contract', None) is not None:
        return deepcopy(app.resolved_contract)
    return app.contract


def _contract_for_history(app):
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    if asset_class in {'futures', 'future', 'fut', 'crypto', 'metals', 'metal', 'stock', 'stocks', 'equity', 'equities', 'stk'}:
        if getattr(app, 'resolved_contract', None) is not None:
            return deepcopy(app.resolved_contract)
        details = getattr(app, 'contract_details_df', None)
        if isinstance(details, pd.DataFrame) and not details.empty and 'Symbol' in details.columns:
            same_symbol = details[details['Symbol'].astype(str).str.upper() == str(app.ticker).upper()]
            if not same_symbol.empty:
                resolved_rows = same_symbol[same_symbol.apply(lambda row: _contract_row_is_resolved(row, asset_class), axis=1)]
                candidate_rows = resolved_rows if not resolved_rows.empty else same_symbol
                row = candidate_rows.iloc[-1]
                contract = ibf.build_contract_from_spec(app.asset_spec)
                for attr, column in (
                    ('symbol', 'Symbol'),
                    ('secType', 'SecType'),
                    ('currency', 'Currency'),
                    ('exchange', 'Exchange'),
                    ('primaryExchange', 'PrimaryExchange'),
                    ('localSymbol', 'LocalSymbol'),
                    ('lastTradeDateOrContractMonth', 'LastTradeDateOrContractMonth'),
                    ('tradingClass', 'TradingClass'),
                    ('multiplier', 'Multiplier'),
                ):
                    if column in row and pd.notna(row[column]) and str(row[column]) != '':
                        normalized = _normalized_contract_field(attr, row[column])
                        if normalized not in (None, ''):
                            setattr(contract, attr, normalized)
                if 'ConId' in row and pd.notna(row['ConId']):
                    normalized_conid = _normalized_contract_field('conId', row['ConId'])
                    if normalized_conid is not None:
                        contract.conId = normalized_conid
                if _contract_row_is_resolved(row, asset_class):
                    return contract
    return app.contract

def download_hist_data(app, params):
    """Function to download the historical data"""
    # Set the function inputs as per the params list
    hist_id, duration, candle_size, whatToShow = params[0], params[1], params[2], params[3]
    end_datetime = params[5] if len(params) > 5 else ''
    
    # If the app is connected
    if app.isConnected():
        # Clear the threading event
        app.hist_data_events[f'{hist_id}'].clear()
        app.hist_request_errors.pop(str(hist_id), None)
        
        # Downlod the data
        app.reqHistoricalData(reqId=hist_id, 
                               contract=_contract_for_history(app),
                               endDateTime=end_datetime,
                               durationStr=duration,
                               barSizeSetting=candle_size,
                               whatToShow=whatToShow,
                               useRTH=1 if _use_regular_trading_hours_for_history(app) else 0,
                               formatDate=1,
                               keepUpToDate=False,
                               # EClient function to request contract details
                               chartOptions=[])	
        
        # Make the event wait until the download is finished
        completed = app.hist_data_events[f'{hist_id}'].wait(timeout=45)
        if not completed:
            app.hist_request_errors[str(hist_id)] = {'code': 'TIMEOUT', 'msg': f'Historical data request timed out for {whatToShow}.'}
            print(f'Historical data request timed out for reqId {hist_id} ({whatToShow}).')
            app.logging.info(f'Historical data request timed out for reqId {hist_id} ({whatToShow}).')
    else:
        # Return the function in case we couldn't download the data
        return

def prepare_downloaded_data(app, params):
    data_label = params[4] if len(params) > 4 else params[-1]
    print(f'[{app.ticker}] preparing the {data_label} data...')
    app.logging.info(f'[{app.ticker}] preparing the {data_label} data...')
    
    # Rename the downloaded historical data columns
    app.new_df[f'{params[0]}'].rename(columns={'open':f'{str(data_label).lower()}_open','high':f'{str(data_label).lower()}_high',\
                                              'low':f'{str(data_label).lower()}_low','close':f'{str(data_label).lower()}_close'},inplace=True)
    
    # Set the index to datetime type. IB can return either full timestamps or date-only strings depending on bar type.
    raw_index = app.new_df[f'{params[0]}'].index.astype(str)
    parsed_index = pd.Series(pd.to_datetime(raw_index, format='%Y%m%d %H:%M:%S %Z', errors='coerce'), index=app.new_df[f'{params[0]}'].index)
    if parsed_index.isna().any():
        fallback_mask = parsed_index.isna()
        parsed_index.loc[fallback_mask] = pd.to_datetime(raw_index[fallback_mask], format='%Y%m%d', errors='coerce')
    if parsed_index.isna().any():
        parsed_index = pd.Series(pd.to_datetime(raw_index, format='mixed', errors='coerce'), index=app.new_df[f'{params[0]}'].index)
    app.new_df[f'{params[0]}'].index = pd.DatetimeIndex(parsed_index)
    app.new_df[f'{params[0]}'] = app.new_df[f'{params[0]}'][~app.new_df[f'{params[0]}'].index.isna()]
    # Get rid of the timezone tag when present
    try:
        app.new_df[f'{params[0]}'].index = app.new_df[f'{params[0]}'].index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    
    print(f'[{app.ticker}] {data_label} data is prepared...')
    app.logging.info(f'[{app.ticker}] {data_label} data is prepared...')
       

def _frequency_to_tws_bar_size(data_frequency):
    frequency_number, frequency_string = tf.get_data_frequency_values(data_frequency)
    if frequency_string == "min":
        unit = "min" if frequency_number == 1 else "mins"
        return f"{frequency_number} {unit}"
    if frequency_string == "h":
        unit = "hour" if frequency_number == 1 else "hours"
        return f"{frequency_number} {unit}"
    if frequency_string == "D":
        unit = "day" if frequency_number == 1 else "days"
        return f"{frequency_number} {unit}"
    raise ValueError(f"Unsupported TWS bar size frequency: {data_frequency!r}")


def _history_request_params(app, duration):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    candle_size = _frequency_to_tws_bar_size(app.data_frequency)

    if asset_class == "forex":
        return [[0, duration, candle_size, "MIDPOINT", "MIDPOINT", ""]]
    if asset_class == "metals":
        return [[0, duration, candle_size, "MIDPOINT", "MIDPOINT", ""]]
    if asset_class == "stock":
        return [[0, duration, candle_size, "ADJUSTED_LAST", "ADJUSTED_LAST", ""]]
    if asset_class == "crypto":
        total_days = min(int(duration.split()[0]), _max_bootstrap_days_per_cycle(app))
        chunk_days = _crypto_chunk_days(app)
        params = []
        offset_days = 0
        while offset_days < total_days:
            this_chunk_days = min(chunk_days, total_days - offset_days)
            chunk_end = pd.Timestamp(app.current_period) - dt.timedelta(days=offset_days)
            params.append([
                len(params),
                f"{this_chunk_days} D",
                candle_size,
                "AGGTRADES",
                "AGGTRADES",
                _format_ib_end_datetime(chunk_end, app.zone),
            ])
            offset_days += this_chunk_days
        return params
    return [[0, duration, candle_size, "TRADES", "TRADES", ""]]


def _use_regular_trading_hours_for_history(app):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    return asset_class == "stock"


def _minimum_history_days_for_asset(app):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    if asset_class != "crypto":
        return 0
    # Build enough daily bars in a single bootstrap pass for the live crypto
    # sleeve to clear its immediate warm-up gate (~30 daily rows) with a small
    # buffer, without forcing a deep backfill every cycle.
    return 35


def _max_history_request_days_for_asset(app):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    if asset_class == "crypto":
        return 35
    return None


def _max_bootstrap_days_per_cycle(app):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    if asset_class == "crypto":
        return _max_history_request_days_for_asset(app)
    return None


def _crypto_chunk_days(app):
    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    if asset_class == "crypto":
        return 3
    return None


def _format_ib_end_datetime(timestamp, timezone_name):
    ts = pd.Timestamp(timestamp).to_pydatetime().replace(microsecond=0)
    return f"{ts.strftime('%Y%m%d %H:%M:%S')} {timezone_name}"


def _resample_anchor_for_asset(app):
    hour_string = str(app.market_open_time.hour) if (app.market_open_time.hour)>=10 else '0'+str(app.market_open_time.hour)
    minute_string = str(app.market_open_time.minute) if (app.market_open_time.minute)>=10 else '0'+str(app.market_open_time.minute)
    return f'{hour_string}h{minute_string}min'

def update_hist_data(app):
    ''' Request the historical data '''
    
    print(f"[{app.ticker}] Requesting the historical data...")
    app.logging.info(f"[{app.ticker}] Requesting the historical data...")

    asset_class = str(app.asset_spec.get("asset_class", "forex")).lower()
    if asset_class in {"futures", "future", "fut"}:
        contract = getattr(app, 'resolved_contract', None) or getattr(app, 'contract', None)
        local_symbol = getattr(contract, 'localSymbol', '') if contract is not None else ''
        expiry = getattr(contract, 'lastTradeDateOrContractMonth', '') if contract is not None else ''
        if not local_symbol and not expiry:
            request_contract_details(app, verbose=False)
            contract = getattr(app, 'resolved_contract', None) or getattr(app, 'contract', None)
            local_symbol = getattr(contract, 'localSymbol', '') if contract is not None else ''
            expiry = getattr(contract, 'lastTradeDateOrContractMonth', '') if contract is not None else ''
            if not local_symbol and not expiry:
                message = f'{app.ticker} futures contract is unresolved; missing localSymbol/expiry for historical request. Continuing with cached history.'
                print(f'Historical data request failed for {app.ticker}; continuing with cached history...')
                app.logging.warning(message)
                append_runtime_audit(app, 'history_refresh_skipped', message)
                return
    
    # Distinguish bootstrap backfills from incremental refreshes so later cycles stay fast.
    last_history_dt = pd.Timestamp(app.historical_data.index[-1])
    gap_delta = pd.Timestamp(app.current_period) - last_history_dt
    days_passed_number = max(0, int(math.ceil(gap_delta.total_seconds() / 86400.0)))
    min_history_days = _minimum_history_days_for_asset(app)
    existing_span_days = 0
    try:
        existing_span_days = max(0, (app.historical_data.index.max() - app.historical_data.index.min()).days)
    except Exception:
        existing_span_days = 0
    bootstrap_days = max(0, min_history_days - existing_span_days)
    if bootstrap_days > 0:
        request_days = max(bootstrap_days, days_passed_number if days_passed_number > 0 else 1)
    else:
        request_days = max(1, days_passed_number)
    max_request_days = _max_history_request_days_for_asset(app)
    if max_request_days is not None:
        request_days = min(int(request_days), int(max_request_days))
    days_passed = f'{request_days} D'
    
    # Determine the historical request payload for the asset class
    params_list = _history_request_params(app, days_passed)
    
    # If the app is connected
    if app.isConnected():
        for req_id in {str(params[0]) for params in params_list}:
            if req_id not in app.new_df:
                app.new_df[req_id] = pd.DataFrame()
            if req_id not in app.hist_data_events:
                app.hist_data_events[req_id] = Event()
            app.new_df[req_id] = pd.DataFrame()
            app.hist_request_errors.pop(req_id, None)
        # Download the historical data
        if str(app.asset_spec.get("asset_class", "forex")).lower() == "crypto":
            for params in params_list:
                download_hist_data(app, params)
        else:
            with ThreadPoolExecutor(len(params_list)) as executor:
                list(executor.map(download_hist_data, [app]*len(params_list), params_list))
    else:
        return

    failed_histories = [params for params in params_list if str(params[0]) in getattr(app, 'hist_request_errors', {})]
    if failed_histories:
        if getattr(app, 'historical_data', pd.DataFrame()).empty is False:
            details = '; '.join(
                f"reqId {params[0]} {params[3]}: {app.hist_request_errors[str(params[0])]['code']} - {app.hist_request_errors[str(params[0])]['msg']}"
                for params in failed_histories
            )
            app.logging.warning(f'Historical data request failed for {app.ticker}; continuing with cached history. {details}')
            print(f'Historical data request failed for {app.ticker}; continuing with cached history...')
            return
        details = '; '.join(
            f"reqId {params[0]} {params[3]}: {app.hist_request_errors[str(params[0])]['code']} - {app.hist_request_errors[str(params[0])]['msg']}"
            for params in failed_histories
        )
        raise RuntimeError(f'Historical data request failed for {app.ticker}. {details}')

    # If the app is connected
    if app.isConnected():
        # Prepare the data
        unique_prepare_params = []
        seen_req_ids = set()
        for params in params_list:
            req_id = str(params[0])
            if req_id in seen_req_ids:
                continue
            seen_req_ids.add(req_id)
            unique_prepare_params.append(params)
        with ThreadPoolExecutor(len(unique_prepare_params)) as executor:
            list(executor.map(prepare_downloaded_data, [app]*len(unique_prepare_params), unique_prepare_params))
    else:
        return    

    if any(app.new_df[str(params[0])].empty for params in unique_prepare_params):
        raise RuntimeError(f'Historical data for {app.ticker} is empty after download.')

    if len(params_list) == 2 and asset_class != "crypto":
        # Concatenate the BID and ASK data
        df = pd.concat([app.new_df['0'], app.new_df['1']], axis=1)
        # Get the mid prices based on the BID and ASK prices
        df = tf.get_mid_series(df)
    elif asset_class == "crypto":
        chunk_frames = [app.new_df[str(params[0])].copy() for params in unique_prepare_params if str(params[0]) in app.new_df and not app.new_df[str(params[0])].empty]
        if len(chunk_frames) == 0:
            raise RuntimeError(f'Historical data for {app.ticker} is empty after download.')
        df = pd.concat(chunk_frames, axis=0)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        df.columns = ["Close", "Open", "High", "Low"]
    else:
        # Use TRADES data directly
        df = app.new_df['0'].copy()
        df.columns = ["close", "open", "high", "low"] # Standardize names if needed
        # Ensure column names match mid-price series structure for resampling
        df.columns = ["Close", "Open", "High", "Low"]
    
    anchor = _resample_anchor_for_asset(app)

    # Resample the data as per the data frequency
    df = tf.resample_df(df, frequency=app.data_frequency, start=anchor)

    # Concatenate the current historical dataframe with the whole one
    app.historical_data = pd.concat([app.historical_data, df])
    # Sor the historical data by index
    app.historical_data.sort_index(inplace=True)
    # Drop duplicates
    app.historical_data = app.historical_data[~app.historical_data.index.duplicated(keep='last')]
    
    print(f"[{app.ticker}] Historical data was successfully prepared...")
    app.logging.info(f"[{app.ticker}] Historical data was successfully prepared...")

def _fallback_last_price(app):
    for frame_name in ('base_df', 'historical_data'):
        frame = getattr(app, frame_name, None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for column in ('Close', 'close'):
                if column in frame.columns:
                    value = pd.to_numeric(frame[column], errors='coerce').dropna()
                    if not value.empty:
                        return float(value.iloc[-1])
    return 0.0


def _is_integer_only_asset(app):
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    if asset_class in ('futures', 'future', 'fut'):
        return True
    if asset_class == 'stock':
        return float(_quantity_step_for_asset(app)) >= 1.0
    if asset_class in ('metals', 'metal', 'crypto'):
        return float(_quantity_step_for_asset(app)) >= 1.0
    return False


def _contract_quantity_step(app):
    details = getattr(app, 'contract_details_df', None)
    if not isinstance(details, pd.DataFrame) or details.empty or 'Symbol' not in details.columns:
        return np.nan

    same_symbol = details[details['Symbol'].astype(str).str.upper() == str(getattr(app, 'ticker', '')).upper()]
    if same_symbol.empty:
        return np.nan

    row = same_symbol.iloc[-1]
    for column in ('SizeIncrement', 'SuggestedSizeIncrement', 'MinSize'):
        value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors='coerce').iloc[0]
        if np.isfinite(value) and float(value) > 0:
            return float(value)
    return np.nan


def _quantity_step_decimals(step):
    try:
        exponent = Decimal(str(float(step))).normalize().as_tuple().exponent
        return max(0, -int(exponent))
    except Exception:
        return 0


def _quantity_step_for_asset(app):
    target = _strategy_target_for_app(app)
    explicit_step = None
    if isinstance(target, dict):
        explicit_step = target.get('quantity_step')
    if explicit_step in (None, ''):
        explicit_step = getattr(app, 'asset_spec', {}).get('quantity_step')

    try:
        if explicit_step not in (None, ''):
            step = float(explicit_step)
            if np.isfinite(step) and step > 0:
                return step
    except Exception:
        pass

    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    if asset_class in ('metals', 'metal'):
        return 1.0

    broker_step = _contract_quantity_step(app)
    if np.isfinite(broker_step) and broker_step > 0:
        return float(broker_step)

    if asset_class == 'crypto':
        return 1e-8
    if asset_class in ('futures', 'future', 'fut'):
        return 1.0
    if asset_class == 'stock':
        if bool(getattr(app, 'asset_spec', {}).get('fractional_shares', False)):
            return 0.0001
        return 1.0
    return 1.0


def _get_execution_residual(app):
    execution_state = getattr(app, 'strategy_state', {}).get('execution', {})
    try:
        return float(execution_state.get('qty_residual', 0.0))
    except Exception:
        return 0.0


def _queue_execution_residual(app, residual):
    residual = float(residual)
    app.strategy_state.setdefault('execution', {})['qty_residual'] = residual
    app.queue_strategy_state({'execution': {'qty_residual': residual}})


def _normalize_order_quantity(app, quantity):
    qty = abs(float(quantity))
    step = float(_quantity_step_for_asset(app))
    if not np.isfinite(qty) or not np.isfinite(step) or step <= 0:
        return 0.0
    if qty + 1e-12 < step:
        return 0.0
    units = math.floor((qty / step) + 1e-12)
    normalized = units * step
    if step >= 1.0:
        return int(round(normalized))
    decimals = _quantity_step_decimals(step)
    return round(float(normalized), decimals)


def _is_effectively_flat_quantity(app, quantity):
    qty = abs(float(quantity))
    step = float(_quantity_step_for_asset(app))
    if not np.isfinite(qty):
        return True
    if not np.isfinite(step) or step <= 0:
        return np.isclose(qty, 0.0, atol=1e-8)
    if step >= 1.0:
        return qty + 1e-12 < step
    return qty <= step + 1e-12


def _effective_position_sign(app, quantity):
    if _is_effectively_flat_quantity(app, quantity):
        return 0.0
    return float(np.sign(float(quantity)))


def _strategy_target_for_app(app):
    strategy_targets = getattr(app, 'strategy_targets', {}) or {}
    if not isinstance(strategy_targets, dict):
        return {}
    return strategy_targets.get(str(getattr(app, 'ticker', '')).upper(), {}) or {}


def _strategy_allows_outside_hours(app):
    target = _strategy_target_for_app(app)
    return bool(target.get('allow_outside_hours', False)) if isinstance(target, dict) else False


def _parse_ib_session_ranges(hours_text, tz_name):
    if not isinstance(hours_text, str) or not hours_text.strip():
        return []
    ranges = []
    for part in hours_text.split(';'):
        part = part.strip()
        if not part or 'CLOSED' in part.upper() or '-' not in part:
            continue
        start_text, end_text = part.split('-', 1)
        try:
            start_dt = dt.datetime.strptime(start_text, '%Y%m%d:%H%M')
            end_dt = dt.datetime.strptime(end_text, '%Y%m%d:%H%M')
            start_dt = pd.Timestamp(start_dt)
            end_dt = pd.Timestamp(end_dt)
            if end_dt <= start_dt:
                end_dt = end_dt + dt.timedelta(days=1)
            if tz_name:
                start_dt = start_dt.tz_localize(tz_name)
                end_dt = end_dt.tz_localize(tz_name)
            ranges.append((start_dt, end_dt))
        except Exception:
            continue
    return ranges


def _parse_contract_expiry_value(raw_value):
    text = str(raw_value or '').strip()
    if not text:
        return (9999, 12, 31)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return (int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    if len(digits) >= 6:
        return (int(digits[:4]), int(digits[4:6]), 1)
    return (9999, 12, 31)


def _normalize_contract_month_text(raw_value):
    text = str(raw_value or '').strip()
    if not text:
        return ''
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric > 0:
            if numeric.is_integer():
                return str(int(numeric))
    except Exception:
        pass
    return ''.join(ch for ch in text if ch.isdigit())


def _normalized_contract_field(attr, value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if attr == 'conId':
        try:
            return int(float(value))
        except Exception:
            return None
    if attr == 'lastTradeDateOrContractMonth':
        normalized = _normalize_contract_month_text(value)
        return normalized or None
    if attr == 'multiplier':
        try:
            numeric = float(value)
        except Exception:
            text = str(value).strip()
            return text or None
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
    text = str(value).strip()
    return text or None


def _select_contract_details_row(app, same_symbol):
    if not isinstance(same_symbol, pd.DataFrame) or same_symbol.empty:
        return None

    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    resolved_rows = same_symbol[same_symbol.apply(lambda row: _contract_row_is_resolved(row, asset_class), axis=1)]
    candidate_rows = resolved_rows if not resolved_rows.empty else same_symbol
    if candidate_rows.empty:
        print(f"[{app.ticker}] No candidate contract detail rows found.")
        return None

    if asset_class in {'futures', 'future', 'fut'}:
        expiries = [
            _normalized_contract_field('lastTradeDateOrContractMonth', value) or value
            for value in candidate_rows['LastTradeDateOrContractMonth'].unique().tolist()
        ]
        print(f"[{app.ticker}] Found expiries: {expiries}")
        
        explicit_contract_month = _normalize_contract_month_text(
            app.asset_spec.get('contract_month') or app.asset_spec.get('expiry')
        )
        if explicit_contract_month:
            print(f"[{app.ticker}] Filtering for explicit month: {explicit_contract_month}")
            explicit_matches = candidate_rows[
                candidate_rows['LastTradeDateOrContractMonth'].astype(str).apply(_normalize_contract_month_text) == explicit_contract_month
            ]
            if not explicit_matches.empty:
                candidate_rows = explicit_matches
            else:
                print(f"[{app.ticker}] No matches for explicit contract month: {explicit_contract_month}")
        else:
            roll_policy = str(app.asset_spec.get('roll_policy', 'AUTO_FRONT_MONTH') or 'AUTO_FRONT_MONTH').strip().upper()
            if roll_policy == 'AUTO_FRONT_MONTH':
                now_utc = dt.datetime.now(dt.timezone.utc)
                today_key = (now_utc.year, now_utc.month, now_utc.day)
                non_expired = candidate_rows[
                    candidate_rows['LastTradeDateOrContractMonth'].astype(str).apply(_parse_contract_expiry_value) >= today_key
                ]
                if not non_expired.empty:
                    candidate_rows = non_expired
                    remaining_expiries = [
                        _normalized_contract_field('lastTradeDateOrContractMonth', value) or value
                        for value in candidate_rows['LastTradeDateOrContractMonth'].unique().tolist()
                    ]
                    print(f"[{app.ticker}] Filtered out expired contracts. Remaining expiries: {remaining_expiries}")
                else:
                    print(f"[{app.ticker}] All candidate contracts have expired (Today: {today_key}).")

        candidate_rows = candidate_rows.copy()
        candidate_rows['_expiry_key'] = candidate_rows['LastTradeDateOrContractMonth'].astype(str).apply(_parse_contract_expiry_value)
        candidate_rows['_local_symbol_key'] = candidate_rows['LocalSymbol'].astype(str)
        candidate_rows = candidate_rows.sort_values(['_expiry_key', '_local_symbol_key'], kind='stable')
        chosen = candidate_rows.iloc[0]
        chosen_expiry = _normalized_contract_field('lastTradeDateOrContractMonth', chosen.get('LastTradeDateOrContractMonth'))
        chosen_conid = _normalized_contract_field('conId', chosen.get('ConId'))
        print(f"[{app.ticker}] Selected contract: LocalSymbol={chosen.get('LocalSymbol')}, Expiry={chosen_expiry}, ConId={chosen_conid}")
        return chosen

    return candidate_rows.iloc[-1]


def _is_within_asset_trading_hours(app):
    try:
        if hasattr(stra, "get_asset_runtime_policy"):
            policy = stra.get_asset_runtime_policy(
                getattr(app, "ticker", ""),
                asset_class=getattr(app, "asset_spec", {}).get("asset_class"),
            )
            session = str(policy.get("session", "weekdays")).strip().lower()
            current_ts = pd.Timestamp(getattr(app, 'current_period', dt.datetime.now()))
            local_tz = pytz.timezone(getattr(app, 'zone', 'UTC'))
            if current_ts.tzinfo is None:
                current_ts = local_tz.localize(current_ts)
            else:
                current_ts = current_ts.astimezone(local_tz)
            current_utc = current_ts.astimezone(pytz.UTC)

            if session in {"24_7", "24x7", "always", "always_on"}:
                weekly_open = True
            elif session == "weekdays":
                weekly_open = current_utc.weekday() != 5
            elif session == "auto":
                weekly_open = None
            else:
                weekly_open = None

            if weekly_open is not None:
                if not weekly_open:
                    return False
                maintenance_start = str(policy.get("daily_maintenance_utc_start", "00:00")).strip()
                maintenance_minutes = int(policy.get("daily_maintenance_minutes", 0) or 0)
                if maintenance_minutes > 0:
                    start_hour, start_minute = [int(x) for x in maintenance_start.split(":", 1)]
                    maintenance_begin = current_utc.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
                    maintenance_end = maintenance_begin + dt.timedelta(minutes=maintenance_minutes)
                    if maintenance_begin <= current_utc < maintenance_end:
                        return False
                return True
    except Exception:
        pass

    if _strategy_allows_outside_hours(app):
        return True

    details = getattr(app, 'contract_details_df', None)
    if isinstance(details, pd.DataFrame) and not details.empty and 'Symbol' in details.columns:
        same = details[details['Symbol'].astype(str).str.upper() == str(getattr(app, 'ticker', '')).upper()]
        if not same.empty:
            row = _select_contract_details_row(app, same)
            if row is None:
                row = same.iloc[-1]
            tz_name = str(row.get('TimeZoneId') or '').strip() or None
            asset_class = str(getattr(app, 'asset_spec', {}).get('asset_class', 'forex')).lower()
            if asset_class in {'futures', 'future', 'fut'}:
                hours_text = row.get('TradingHours') or row.get('LiquidHours')
            else:
                hours_text = row.get('LiquidHours') or row.get('TradingHours')
            if isinstance(hours_text, str) and hours_text.strip():
                current_ts = pd.Timestamp(getattr(app, 'current_period', dt.datetime.now()))
                if current_ts.tzinfo is None:
                    current_ts = current_ts.tz_localize(getattr(app, 'zone', 'UTC'))
                if tz_name:
                    current_ts = current_ts.tz_convert(tz_name)
                # Check for an explicit match in IB sessions.
                found_match = False
                for start_dt, end_dt in _parse_ib_session_ranges(hours_text, tz_name):
                    if start_dt <= current_ts <= end_dt:
                        found_match = True
                        break
                if found_match:
                    return True
                # If we have session data and no match, it's CLOSED.
                return False

    # Fallback to market open/close times if provided in the app (usually from main.py)
    try:
        current_ts = pd.Timestamp(getattr(app, 'current_period', dt.datetime.now()))
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize(getattr(app, 'zone', 'UTC'))
        
        market_open = getattr(app, 'market_open_time', None)
        market_close = getattr(app, 'market_close_time', None)
        
        if market_open is not None and market_close is not None:
            market_open = pd.Timestamp(market_open)
            market_close = pd.Timestamp(market_close)
            if market_open.tzinfo is None:
                market_open = market_open.tz_localize(getattr(app, 'zone', 'UTC'))
            if market_close.tzinfo is None:
                market_close = market_close.tz_localize(getattr(app, 'zone', 'UTC'))
            return bool(market_open <= current_ts <= market_close)
    except Exception:
        pass

    # If policy is 'auto' and we have no data, default to False to be safe.
    # Otherwise, for legacy reasons, if it's not 'auto', default to True.
    try:
        if hasattr(stra, "get_asset_runtime_policy"):
            policy = stra.get_asset_runtime_policy(getattr(app, "ticker", ""))
            if policy.get("session") == "auto":
                return False
    except Exception:
        pass

    return True


def _uses_direct_contract_target(app):
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    target = _strategy_target_for_app(app)
    return asset_class == 'futures' and str(target.get('quantity_mode') or '').lower() == 'contracts' and pd.notna(target.get('target_quantity', np.nan))


def _uses_explicit_target_quantity(app):
    target = _strategy_target_for_app(app)
    quantity_mode = str(target.get('quantity_mode') or '').lower()
    return quantity_mode in {'contracts', 'units', 'quantity'} and pd.notna(target.get('target_quantity', np.nan))


def _contract_min_tick(app):
    explicit_tick = pd.to_numeric(pd.Series([getattr(app, 'asset_spec', {}).get('tick_size', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(explicit_tick) and explicit_tick > 0:
        return float(explicit_tick)
    details = getattr(app, 'contract_details_df', None)
    if isinstance(details, pd.DataFrame) and not details.empty and 'Symbol' in details.columns and 'MinTick' in details.columns:
        same = details[details['Symbol'].astype(str).str.upper() == str(getattr(app, 'ticker', '')).upper()]
        if not same.empty:
            val = pd.to_numeric(same['MinTick'], errors='coerce').dropna()
            if not val.empty and float(val.iloc[-1]) > 0:
                min_tick = float(val.iloc[-1])
                asset_class = str(getattr(app, 'asset_spec', {}).get('asset_class', '')).lower()
                if asset_class == 'crypto':
                    return max(min_tick, 0.01)
                return min_tick
    ticker = str(getattr(app, 'ticker', '')).upper()
    fallback_ticks = {
        'MES': 0.25,
        'XAUUSD': 0.01,
    }
    if ticker in fallback_ticks:
        return float(fallback_ticks[ticker])
    asset_class = str(getattr(app, 'asset_spec', {}).get('asset_class', '')).lower()
    if asset_class == 'crypto':
        return 0.01
    if asset_class == 'stock':
        return 0.01
    if asset_class == 'forex':
        return 0.005 if ticker.endswith('JPY') else 0.00005
    return float(np.nan)


def _round_price_to_contract_tick(app, price, side='nearest'):
    tick = _contract_min_tick(app)
    value = float(price)
    if not np.isfinite(value) or not np.isfinite(tick) or tick <= 0:
        return value
    units = value / tick
    if side == 'up':
        rounded_units = math.ceil(units - 1e-12)
    elif side == 'down':
        rounded_units = math.floor(units + 1e-12)
    else:
        rounded_units = round(units)
    rounded = rounded_units * tick
    tick_str = format(float(tick), 'f').rstrip('0').rstrip('.')
    decimals = max(0, len(tick_str.split('.')[-1]) if '.' in tick_str else 0)
    return round(rounded, decimals)


def _apply_residual_quantity_policy(app):
    if not _is_integer_only_asset(app):
        app.execution_effective_signed_quantity = float(np.sign(getattr(app, 'signal', 0.0))) * abs(float(getattr(app, 'current_quantity', 0.0)))
        app.execution_residual_quantity = 0.0
        _queue_execution_residual(app, 0.0)
        return

    if float(getattr(app, 'leverage', 0.0)) == 0.0 or float(getattr(app, 'signal', 0.0)) == 0.0:
        _queue_execution_residual(app, 0.0)
        app.current_quantity = abs(float(getattr(app, 'current_quantity', 0.0)))
        return

    desired_signed = float(np.sign(app.signal)) * abs(float(getattr(app, 'current_quantity', 0.0)))
    residual = _get_execution_residual(app)
    if residual != 0.0 and np.sign(residual) != np.sign(desired_signed):
        residual = 0.0

    effective_signed = desired_signed + residual
    tradable_signed = float(np.sign(effective_signed)) * math.floor(abs(effective_signed))
    new_residual = effective_signed - tradable_signed

    app.current_quantity = abs(tradable_signed)
    app.execution_effective_signed_quantity = effective_signed
    app.execution_residual_quantity = new_residual
    _queue_execution_residual(app, new_residual)


def update_asset_last_value(app, verbose=True):
    """Set the execution reference price from the latest downloaded data."""
    if verbose:
        print("Updating the last value of the asset...")
        app.logging.info("Updating the last value of the asset...")
    app.last_value = _fallback_last_price(app)
    if app.last_value > 0:
        if verbose:
            print('Using fallback last price from downloaded data...')
            app.logging.info('Using fallback last price from downloaded data...')
    else:
        if verbose:
            print('No downloaded last price was available...')
            app.logging.info('No downloaded last price was available...')

def get_capital_as_per_forex_base_currency(app, capital_datetime):

    # For non-forex assets, the capital calculation is different.
    # Forex uses exchange rates to convert to the base currency of the pair.
    # Futures and Crypto are simpler as we just need the account value in base currency.
    asset_class = app.asset_spec.get("asset_class", "forex")
    
    if asset_class not in ("forex", "metals"):
        # For Futures and Crypto, we use the account value directly
        app.capital = app.cash_balance.loc[capital_datetime, 'value']
        return app.capital

    # Set the yfinance data
    usd_symbol_forex = np.nan
    usd_acc_symbol_forex = np.nan
    capital = np.nan # Initialize capital

    # If the contract symbol is the same as the account base currency
    if app.contract.symbol==app.account_currency:
        # Get the account capital value
        capital = app.cash_balance.loc[capital_datetime, 'value']
    else:
        # The exchange rate where the divisor is the account base currency and the dividend is the forex pair base currency
        exchange_rate_list = app.acc_update[(app.acc_update['key']=='ExchangeRate') & \
                                        (app.acc_update['Currency'] == app.contract.symbol)]['Value'].values.tolist()

        # If there is an exchange rate from IB
        if len(exchange_rate_list)!=0:
            try:
                exchange_rate_val = float(exchange_rate_list[0])
                if exchange_rate_val == 0: # Avoid division by zero
                     app.logging.warning("Exchange rate from IB is zero. Falling back to Yahoo Finance.")
                     exchange_rate_list = [] # Force fallback to trigger Yahoo Finance part
                else:
                    capital = app.cash_balance.loc[capital_datetime, 'value'] / exchange_rate_val
            except ValueError:
                app.logging.error(f"Could not convert exchange rate '{exchange_rate_list[0]}' to float. Falling back to Yahoo Finance.")
                exchange_rate_list = [] # Force fallback
            except ZeroDivisionError: # Should be caught by the if exchange_rate_val == 0, but as a safeguard
                app.logging.error("Division by zero error with IB exchange rate. Falling back to Yahoo Finance.")
                exchange_rate_list = [] # Force fallback


        # If no valid exchange rate from IB, or capital calculation failed, try Yahoo Finance
        if len(exchange_rate_list)==0 or pd.isna(capital):
            app.logging.info("Attempting to fetch exchange rates from Yahoo Finance.")
            # Set the end date to download forex data from yahoo finance
            end = app.current_period + dt.timedelta(days=1)
            # Set the start date to download forex data from yahoo finance
            start = end - dt.timedelta(days=2)

            usd_symbol_ticker = f'USD{app.contract.symbol}=X'
            usd_acc_symbol_ticker = f'USD{app.account_currency}=X'
            calculated_exchange_rate_yf = np.nan # Renamed to avoid conflict

            try:
                usd_symbol_data_full = yf.download(usd_symbol_ticker, start=start, end=end, interval='1m', group_by='ticker', progress=False, show_errors=False)
                if not usd_symbol_data_full.empty and usd_symbol_ticker in usd_symbol_data_full.columns.levels[0]:
                    usd_symbol_data = usd_symbol_data_full[usd_symbol_ticker]
                    usd_symbol_data.index = pd.to_datetime(usd_symbol_data.index).tz_convert(app.zone)
                    if not usd_symbol_data.empty and not usd_symbol_data['Close'].isnull().all():
                        usd_symbol_forex = usd_symbol_data['Close'].ffill().bfill().iloc[-1] # ffill and bfill for robustness
                        index_for_acc_data = usd_symbol_data.index[-1]

                        usd_acc_symbol_data_full = yf.download(usd_acc_symbol_ticker, start=start, end=end, interval='1m', group_by='ticker', progress=False, show_errors=False)
                        if not usd_acc_symbol_data_full.empty and usd_acc_symbol_ticker in usd_acc_symbol_data_full.columns.levels[0]:
                            usd_acc_symbol_data = usd_acc_symbol_data_full[usd_acc_symbol_ticker]
                            usd_acc_symbol_data.index = pd.to_datetime(usd_acc_symbol_data.index).tz_convert(app.zone)

                            if not usd_acc_symbol_data.empty and not usd_acc_symbol_data['Close'].isnull().all():
                                # Try to get value at the same index, otherwise fallback to last
                                if index_for_acc_data in usd_acc_symbol_data.index:
                                    usd_acc_symbol_forex = usd_acc_symbol_data.loc[index_for_acc_data,'Close']
                                else: # Fallback to last known if exact timestamp match fails
                                    usd_acc_symbol_forex = usd_acc_symbol_data['Close'].ffill().bfill().iloc[-1]

                                if not (pd.isna(usd_symbol_forex) or pd.isna(usd_acc_symbol_forex) or usd_acc_symbol_forex == 0):
                                    calculated_exchange_rate_yf = usd_symbol_forex / usd_acc_symbol_forex
                                else:
                                    app.logging.warning(f"Could not calculate valid YF exchange rate: USD_Symbol_Forex={usd_symbol_forex}, USD_Acc_Symbol_Forex={usd_acc_symbol_forex}")
                            else:
                                 app.logging.warning(f"No 'Close' data for {usd_acc_symbol_ticker} from Yahoo Finance.")
                        else:
                            app.logging.warning(f"Could not download or find ticker {usd_acc_symbol_ticker} from Yahoo Finance.")
                    else:
                         app.logging.warning(f"No 'Close' data for {usd_symbol_ticker} from Yahoo Finance.")
                else:
                    app.logging.warning(f"Could not download or find ticker {usd_symbol_ticker} from Yahoo Finance.")

                if not pd.isna(calculated_exchange_rate_yf):
                    # Use the 90% of the portfolio value just in case the forex pair has changed dramatically (Yahoo Finance data is not up to date)
                    capital = app.cash_balance.loc[capital_datetime, 'value'] * calculated_exchange_rate_yf * 0.9
                    app.logging.info(f"Capital calculated using Yahoo Finance exchange rate: {calculated_exchange_rate_yf}")
                else:
                    app.logging.error("Failed to get a valid exchange rate from Yahoo Finance. Capital calculation will use unconverted base currency value.")
                    capital = app.cash_balance.loc[capital_datetime, 'value'] # Fallback to unconverted

            except Exception as e:
                app.logging.error(f"Error during Yahoo Finance download or processing: {e}")
                capital = app.cash_balance.loc[capital_datetime, 'value'] # Fallback to unconverted

    # If after all attempts, capital is still NaN, fallback to unconverted base currency value
    if pd.isna(capital):
        app.logging.warning("All attempts to get/calculate exchange rate failed. Using unconverted base currency value for capital.")
        capital = app.cash_balance.loc[capital_datetime, 'value']

    app.capital = capital # Assign to app.capital at the end
    return capital      # Return the calculated capital

def update_capital(app):
    ''' Function to update the capital value'''
    print('Update the cash balance datetime and value...')
    app.logging.info('Update the cash balance datetime and value...')

    shared_acc_update = getattr(app, 'shared_acc_update', None)
    shared_cash_balance = getattr(app, 'shared_cash_balance', None)
    shared_portfolio_snapshots = getattr(app, 'shared_portfolio_snapshots_df', None)
    if isinstance(shared_acc_update, pd.DataFrame) and not shared_acc_update.empty:
        app.acc_update = shared_acc_update.copy(deep=True)
        if isinstance(shared_cash_balance, pd.DataFrame) and not shared_cash_balance.empty:
            app.cash_balance = shared_cash_balance.copy(deep=True)
        if isinstance(shared_portfolio_snapshots, pd.DataFrame) and not shared_portfolio_snapshots.empty:
            app.portfolio_snapshots_df = shared_portfolio_snapshots.copy(deep=True)
        print('Account values successfully updated from shared collector ......')
        app.logging.info('Account values successfully updated from shared collector...')
    elif app.isConnected():
        app.last_account_update_time_printed = None
        app.account_update_event.clear()
        app.reqAccountUpdates(True,app.account)
        app.account_update_event.wait()
        app.reqAccountUpdates(False,app.account)
        _flush_temp_sheet(app, 'temp_portfolio_snapshots', 'portfolio_snapshots_df', 'portfolio_snapshots', dedupe_subset=['datetime', 'Account', 'Symbol', 'ConId'])
        print('Account values successfully updated ......')
        app.logging.info('Account values successfully requested...')
    else:
        return

    if app.acc_update.empty:
        append_runtime_audit(app, 'account_update_missing', 'No account update rows received from IB')
        app.logging.warning('No account update rows received from IB; keeping previous capital value.')
        return

    priority_filters = [
        (app.acc_update['key'] == 'NetLiquidationByCurrency') & (app.acc_update['Currency'].astype(str).str.upper() == 'BASE'),
        (app.acc_update['key'] == 'NetLiquidation') & (app.acc_update['Currency'].astype(str).str.upper() == 'BASE'),
        (app.acc_update['key'] == 'NetLiquidation'),
        (app.acc_update['key'] == 'TotalCashBalance') & (app.acc_update['Currency'].astype(str).str.upper() == 'BASE'),
    ]
    capital_rows = pd.DataFrame()
    for mask in priority_filters:
        capital_rows = app.acc_update.loc[mask].copy()
        if not capital_rows.empty:
            break
    if capital_rows.empty:
        append_runtime_audit(app, 'account_update_missing', 'No NetLiquidation or TotalCashBalance account value received from IB')
        app.logging.warning('No NetLiquidation or TotalCashBalance account value received from IB; keeping previous capital value.')
        return

    capital_rows['datetime'] = pd.to_datetime(capital_rows['datetime'], errors='coerce')
    capital_rows['Value'] = pd.to_numeric(capital_rows['Value'], errors='coerce')
    capital_rows = capital_rows.dropna(subset=['datetime', 'Value']).sort_values('datetime')
    if capital_rows.empty:
        app.logging.warning('Account value rows were invalid after numeric/datetime coercion; keeping previous capital value.')
        return

    capital_datetime = capital_rows['datetime'].iloc[-1]
    account_value = float(capital_rows['Value'].iloc[-1])
    app.cash_balance.loc[capital_datetime, 'value'] = account_value
    app.net_liquidation = account_value
    app.account_value = account_value
    if getattr(app, 'portfolio_snapshots_df', pd.DataFrame()).empty and not getattr(app, 'pos_df', pd.DataFrame()).empty:
        app.portfolio_snapshots_df = _portfolio_snapshots_from_positions(app.pos_df)
       
    app.capital = get_capital_as_per_forex_base_currency(app, capital_datetime)
    
    # Forward fill the cash balance dataframe
    app.cash_balance.ffill(inplace=True)
        
    print('Capital value successfully updated ...')
    app.logging.info('Capital value successfully updated ...')


def _shared_pretrade_snapshot_active(app):
    return bool(getattr(app, 'use_shared_pretrade_snapshot', False)) and bool(getattr(app, 'shared_broker_snapshot_ready', False))
    
def update_risk_management_orders(app, verbose=True):
    ''' Function to update the risk management orders IDs and their status'''

    if verbose:
        print('Updating the risk management orders IDs and their status...')
        app.logging.info('Updating the risk management orders IDs and their status...')
    updated_any_rm_order = False
    
    # If the open orders dataframe is not empty
    if not app.open_orders.empty:
        symbol_orders = _filter_orders_for_contract(app.open_orders, app)
        sl_order_type = 'TRAIL' if app.trail else 'STP'
        sl_orders = symbol_orders[symbol_orders["OrderType"] == sl_order_type]["OrderId"].sort_values(ascending=True)
        tp_orders = symbol_orders[symbol_orders["OrderType"] == 'LMT']["OrderId"].sort_values(ascending=True)

        app.sl_order_id = int(sl_orders.values[-1]) if not sl_orders.empty else np.nan
        app.tp_order_id = int(tp_orders.values[-1]) if not tp_orders.empty else np.nan
        updated_any_rm_order = pd.notna(app.sl_order_id) or pd.notna(app.tp_order_id)

        if pd.notna(app.sl_order_id):
            sl_status = app.open_orders[app.open_orders['OrderId'] == app.sl_order_id]['Status'].astype(str)
            app.sl_filled_or_canceled_bool = sl_status.str.contains('canceled', case=False).any() or \
                                             sl_status.str.contains('Filled', case=False).any()
        else:
            app.sl_filled_or_canceled_bool = False

        if pd.notna(app.tp_order_id):
            tp_status = app.open_orders[app.open_orders['OrderId'] == app.tp_order_id]['Status'].astype(str)
            app.tp_filled_or_canceled_bool = tp_status.str.contains('canceled', case=False).any() or \
                                             tp_status.str.contains('Filled', case=False).any()
        else:
            app.tp_filled_or_canceled_bool = False

    else:
        # Set the last stop loss order to NaN
        app.sl_order_id = np.nan
        # Set the last take profit order to NaN
        app.tp_order_id = np.nan            

        # Set a boolean to False if the previous stop loss is not filled or canceled
        app.sl_filled_or_canceled_bool = False
        # Set a boolean to False if the previous take profit is not filled or canceled
        app.tp_filled_or_canceled_bool = False

    if verbose:
        if updated_any_rm_order:
            print('The risk management orders IDs and their status were successfully updated...')
            app.logging.info('The risk management orders IDs and their status were successfully updated...')
        else:
            print('There were no live risk management orders to update...')
            app.logging.info('There were no live risk management orders to update...')
    
def update_remaining_position_based_on_risk_management(app, risk_management_threshold):
    ''' Function to update the remaining position and cash balance based on the selected risk management threshold'''

    def _write_remaining_position_row(order_id):
        order_rows = app.orders_status[app.orders_status['OrderId'] == order_id]
        exec_rows = app.exec_df[app.exec_df['OrderId'] == order_id]
        source_rows = app.pos_df[(app.pos_df['Symbol'] == app.contract.symbol) & (app.pos_df['Currency'] == app.contract.currency)]
        if order_rows.empty or exec_rows.empty or source_rows.empty:
            return False

        remaining = float(order_rows['Remaining'].iloc[-1])
        remaining_datetime = pd.to_datetime(order_rows.index[-1], errors='coerce')
        if pd.isna(remaining_datetime):
            return False
        average_price = pd.to_numeric(pd.Series([exec_rows['AvPrice'].iloc[-1]]), errors='coerce').iloc[0]

        source_row = source_rows.iloc[-1].copy()
        for column in app.pos_df.columns:
            value = source_row[column] if column in source_row.index else pd.NA
            if pd.api.types.is_datetime64_any_dtype(app.pos_df[column]):
                value = pd.to_datetime(value, errors='coerce')
                if pd.isna(value):
                    value = pd.NaT
            source_row[column] = value

        app.pos_df.loc[remaining_datetime, :] = source_row.reindex(app.pos_df.columns).values
        app.pos_df.loc[remaining_datetime, 'Position'] = remaining
        app.pos_df.loc[remaining_datetime, 'Avg cost'] = average_price
        return True
    
    # If the risk management selected is the stop-loss order
    if risk_management_threshold == 'sl':
        # If the previous stop loss order is filled or canceled
        if app.sl_filled_or_canceled_bool == True:
            if _write_remaining_position_row(app.sl_order_id):
                app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'leverage'] = app.leverage
                app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'signal'] = 0
                app.cash_balance.ffill(inplace=True)
        
    # If the risk management selected is the take-profit order
    elif risk_management_threshold == 'tp':
        # If the previous take profit order is filled or canceled
        if app.tp_filled_or_canceled_bool == True:
            if _write_remaining_position_row(app.tp_order_id):
                app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'leverage'] = app.leverage
                app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'signal'] = 0
                app.cash_balance.ffill(inplace=True)
  
def update_submitted_orders(app, verbose=True):
    ''' Function to update the submitted orders'''
    if not _shared_cycle_symbol_broker_pull_allowed(app):
        return
    with app.broker_sync_lock:
        if verbose:
            print(f'[{app.ticker}] Updating the submitted orders ...')
            app.logging.info(f'[{app.ticker}] Updating the submitted orders ...')
        
        symbol_periods = app.periods_traded.copy()
        last_trade_dt = None
        try:
            if 'trade_time' in symbol_periods.columns:
                symbol_periods['trade_time'] = pd.to_datetime(symbol_periods['trade_time'], errors='coerce')
            if 'trade_done' in symbol_periods.columns:
                symbol_periods['trade_done'] = pd.to_numeric(symbol_periods['trade_done'], errors='coerce').fillna(0)
            completed = symbol_periods[symbol_periods.get('trade_done', 0) == 1] if 'trade_done' in symbol_periods.columns else pd.DataFrame()
            if not completed.empty and 'trade_time' in completed.columns:
                completed = completed.dropna(subset=['trade_time']).sort_values('trade_time')
                if not completed.empty:
                    last_trade_dt = completed['trade_time'].iloc[-1]
        except Exception:
            last_trade_dt = None

        if last_trade_dt is None:
            last_trade_dt = getattr(app, 'previous_period', dt.datetime.now())
            append_runtime_audit(app, 'last_trade_time_fallback', f'Using previous_period={last_trade_dt} for execution sync')

        last_trade_time = pd.Timestamp(last_trade_dt).strftime(f'%Y%m%d %H:%M:%S {app.zone}')
        
        if app.isConnected():
            if verbose:
                print(f'[{app.ticker}] Requesting executions...')
                app.logging.info(f'[{app.ticker}] Requesting executions...')
            app.executions_request_event.clear()
            previous_silent = bool(getattr(app, 'silent_broker_sync', False))
            current_depth = int(getattr(app, 'silent_broker_sync_depth', 0))
            app.silent_broker_sync = not verbose
            if not verbose:
                app.silent_broker_sync_depth = current_depth + 1
            try:
                app.reqExecutions(0, ibf.executionFilter(last_trade_time))
                app.executions_request_event.wait()
            finally:
                app.silent_broker_sync = previous_silent
                app.silent_broker_sync_depth = current_depth
            
            if verbose:
                print(f'[{app.ticker}] Successfully requested execution and commissions details...')
                app.logging.info(f'[{app.ticker}] Successfully requested execution and commissions details...')
        else:
            return
        
        if app.isConnected():
            update_risk_management_orders(app, verbose=verbose)  
        
            if (app.temp_exec_df.empty == False):  
                app.temp_exec_df['Execution Time'] = pd.to_datetime(
                    app.temp_exec_df['Execution Time'].replace(rf"{ app.zone}", "", regex=True).values
                )
                app.temp_exec_df.set_index('datetime', inplace=True)
                app.temp_exec_df.index.name = ''
                app.exec_df = pd.concat([app.exec_df, app.temp_exec_df])
                app.exec_df.drop_duplicates(inplace=True)
                app.exec_df.sort_index(ascending=True, inplace=True)
                app.temp_exec_df = pd.DataFrame()
                
                if (app.orders_status.empty == False) and (app.pos_df.empty == False):
                    if (app.sl_filled_or_canceled_bool == True) and (app.tp_filled_or_canceled_bool == True):
                        sl_order_val = app.exec_df[app.exec_df['OrderId'] == app.sl_order_id]['Execution Time'].values[-1]
                        if hasattr(sl_order_val, 'strftime'):
                            sl_order_val = sl_order_val.strftime('%Y-%m-%d %H:%M:%S')
                        sl_order_execution_time = dt.datetime.strptime(str(sl_order_val), '%Y-%m-%d %H:%M:%S')
                        tp_order_val = app.exec_df[app.exec_df['OrderId'] == app.sl_order_id]['Execution Time'].values[-1]
                        if hasattr(tp_order_val, 'strftime'):
                            tp_order_val = tp_order_val.strftime('%Y-%m-%d %H:%M:%S')
                        tp_order_execution_time = dt.datetime.strptime(str(tp_order_val), '%Y-%m-%d %H:%M:%S')
                        
                        if sl_order_execution_time > tp_order_execution_time:
                            update_remaining_position_based_on_risk_management(app, 'sl')
                        else:
                            update_remaining_position_based_on_risk_management(app, 'tp')
                    elif app.sl_filled_or_canceled_bool == True:                    
                        update_remaining_position_based_on_risk_management(app, 'sl')
                    elif app.tp_filled_or_canceled_bool == True:
                        update_remaining_position_based_on_risk_management(app, 'tp')
            if (app.temp_comm_df.empty == False):
                app.temp_comm_df.set_index('datetime', inplace=True)
                app.temp_comm_df.index.name = ''
                mask = pd.to_numeric(app.temp_comm_df['Realized PnL'], errors='coerce') == 1.7976931348623157e+308
                app.temp_comm_df.loc[mask, 'Realized PnL'] = np.nan
                app.comm_df = pd.concat([app.comm_df, app.temp_comm_df])
                app.comm_df.drop_duplicates(inplace=True)
                app.comm_df.sort_index(ascending=True, inplace=True)
                app.temp_comm_df = pd.DataFrame()
                        
        if verbose:
            print('The submitted orders were successfully updated...')
            app.logging.info('The submitted orders were successfully updated...')
    
def portfolio_allocation(app): 
    ''' Function to update the portfolio allocation'''

    print('Make the portfolio allocation ...')
    app.logging.info('Make the portfolio allocation ...')
    
    # If the app is connected
    if app.isConnected():
        if _shared_pretrade_snapshot_active(app) and np.isfinite(float(getattr(app, 'shared_unlevered_capital', np.nan))):
            app.capital = float(app.shared_unlevered_capital)
        else:
            # Update the capital value
            update_capital(app)            
        # Leveraged Equity
        app.capital *= app.leverage
    else:
        return

    print('Successfully Portfolio Allocation...')
    app.logging.info('Successfully Portfolio Allocation...')
                                                
def cancel_previous_stop_loss_order(app):
    ''' Function to cancel the previous stop-loss order'''

    # If there is a previous stop-loss order
    if isinstance(app.sl_order_id, int):
        # If the previous stop-loss order is not filled or canceled
        if (app.sl_filled_or_canceled_bool == False):
            # If the app is connected
            if app.isConnected():
                # Cancel the previous stop loss order
                app.cancelOrder(app.sl_order_id, OrderCancel())
                time.sleep(1)
                print('Canceled old stop-loss order to create a new one...')
                app.logging.info('Canceled old stop-loss order to create a new one...')
            else:
                return

def cancel_previous_take_profit_order(app):
    ''' Function to cancel the previous take profit order'''

    # If there is a previous take-profit order
    if isinstance(app.tp_order_id, int):
        # If the previous take-profit order is not filled or canceled
        if (app.tp_filled_or_canceled_bool == False):
            # If the app is connected
            if app.isConnected():
                # Cancel the previous take-profit order
                app.cancelOrder(app.tp_order_id, OrderCancel())
                time.sleep(1)
                print('Canceled old take-profit order to create a new one...')
                app.logging.info('Canceled old take-profit order to create a new one...')
            else:
                return

def cancel_risk_management_previous_orders(app):
    ''' Function to cancel the previous risk management orders'''
    
    print('Canceling the previous risk management orders if needed...')
    app.logging.info('Canceling the previous risk management orders if needed...')
    _stop_synthetic_crypto_monitor(app, wait=True)
    _clear_pending_synthetic_crypto_monitor(app)
               
    # Drop the code errors related to canceling orders                                         
    app.errors_dict.pop(202, None)  
    app.errors_dict.pop(10147, None)  
    app.errors_dict.pop(10148, None)  
 
    # Create a list of executors
    executors_list = []
    # Create the executors as per each function
    with ThreadPoolExecutor(2) as executor:
        executors_list.append(executor.submit(cancel_previous_stop_loss_order, app)) 
        executors_list.append(executor.submit(cancel_previous_take_profit_order, app)) 

    # Run the executors
    for x in executors_list:
        x.result()
        
    # Drop the code errors related to canceling orders                                         
    app.errors_dict.pop(202, None)  
    app.errors_dict.pop(10147, None)  
    app.errors_dict.pop(10148, None)  
 
    print('The previous risk management orders were canceled if needed...')
    app.logging.info('The previous risk management orders were canceled if needed...')
               
def send_stop_loss_order(app, order_id, quantity): 
    ''' Function to send a stop loss order
        - The function has a while loop to incorporate the fact that sometimes
          the order is not sent due to decimal errors'''
    
    override_prices = getattr(app, 'risk_management_price_overrides', {}) or {}

    # If the previous position sign is different from the current signal
    force_new_rm = bool(getattr(app, 'force_new_risk_management_prices', False))
    if np.isfinite(pd.to_numeric(pd.Series([override_prices.get('sl', np.nan)]), errors='coerce').iloc[0]):
        order_price = float(override_prices['sl'])
        quantity = abs(quantity)
    elif (not force_new_rm) and (app.previous_quantity!=0) and (np.sign(app.previous_quantity)==app.signal) and (app.open_orders.empty==False):
        # Trailing stops may continue from the prior trail, but fixed stop-loss
        # orders must be recalculated each period before the old order is replaced.
        order_price = stra.set_stop_loss_price(app)
        quantity = abs(app.previous_quantity)
    # If they're equal
    else:
        # Set a new stop-loss target price
        order_price = stra.set_stop_loss_price(app)
        quantity = abs(quantity)
   
    quantity = _normalize_order_quantity(app, quantity)
    if quantity <= 0:
        print('Stop-loss quantity rounded to zero. Skipping stop-loss order...')
        app.logging.info('Stop-loss quantity rounded to zero. Skipping stop-loss order...')
        return

    position_sign = _risk_management_position_sign(app)
    if position_sign == 0:
        print('Risk-management position sign is zero. Skipping stop-loss order...')
        app.logging.info('Risk-management position sign is zero. Skipping stop-loss order...')
        return

    price_side = 'up' if position_sign < 0 else 'down'
    order_price = _round_price_to_contract_tick(app, order_price, side=price_side)

    if position_sign > 0:
        direction = 'SELL'
    elif position_sign < 0:
        direction = 'BUY'

    # Set the add decimal to zero
    tick_size = _contract_min_tick(app)
    if not np.isfinite(tick_size) or tick_size <= 0:
        tick_size = 0.00001
    price_step = float(tick_size)
    attempts = 0
    
    current_order_id = int(order_id)
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    if asset_class == 'crypto':
        desired_type = 'TRAIL' if app.trail else 'STP'
        if desired_type not in _contract_order_types(app):
            print(f'[{app.ticker}] {desired_type} stop orders are not supported on this crypto venue. Using synthetic stop monitoring instead...')
            app.logging.info(f'[{app.ticker}] {desired_type} stop orders are not supported on this crypto venue. Using synthetic stop monitoring instead...')
            app.queue_strategy_state({'risk_management': {'sl_price': float(order_price)}})
            return None
    while attempts < 10:
        # Send the stop-loss order
        _clear_order_submission_failures(app)
        app.placeOrder(current_order_id, _contract_for_order(app), ibf.stopOrder(direction, quantity, order_price, app.trail, override=(asset_class == 'crypto')))
        time.sleep(3)
        # Save the output errors in data as a boolean that corresponds to any error while sending the stop-loss order
        failures = _order_submission_failures(app)
        data = bool(failures) or \
                (110 in list(app.errors_dict.keys())) or \
                (463 in list(app.errors_dict.keys()))
        # If data is true
        if data == True:
            append_runtime_audit(app, 'stop_loss_rejected', json.dumps(failures, default=str))
            attempts += 1
            order_price = _round_price_to_contract_tick(app, order_price + price_step, side=price_side)
            current_order_id = _next_order_id(app)
            
            print("Couldn't transmit the stop-loss order, the app will try again...")
            app.logging.info("Couldn't transmit the-stop loss order, the app will try again...")
            
            # Clean the errors dictionary
            app.errors_dict = {}
        else:
            print(f'Stop loss sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.logging.info(f'Stop loss sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.queue_strategy_state({'risk_management': {'sl_price': float(order_price)}})
            return current_order_id
        # If the app is disconnected
        if 504 in list(app.errors_dict.keys()):
            return current_order_id
    return current_order_id
        
def send_take_profit_order(app, order_id, quantity): 
    ''' Function to send a take profit order
        - The function has a while loop to incorporate the fact that sometimes
          the order is not sent due to decimal errors'''
    
    override_prices = getattr(app, 'risk_management_price_overrides', {}) or {}

    # If the previous position sign is different from the current signal
    force_new_rm = bool(getattr(app, 'force_new_risk_management_prices', False))
    if np.isfinite(pd.to_numeric(pd.Series([override_prices.get('tp', np.nan)]), errors='coerce').iloc[0]):
        order_price = float(override_prices['tp'])
        quantity = abs(quantity)
    elif (not force_new_rm) and (app.previous_quantity!=0) and (np.sign(app.previous_quantity)==app.signal) and (app.open_orders.empty==False):
        previous_tp = app.open_orders[app.open_orders["OrderId"] == app.tp_order_id]["LmtPrice"] if pd.notna(getattr(app, 'tp_order_id', np.nan)) else pd.Series(dtype=float)
        previous_tp = pd.to_numeric(previous_tp, errors='coerce').dropna()
        if not previous_tp.empty:
            # Set the previous take-profit target price
            order_price = float(previous_tp.values[-1])
            quantity = abs(app.previous_quantity)
        else:
            order_price = stra.set_take_profit_price(app)
            quantity = abs(quantity)
            
    # If they're equal
    else:
        # Set the take-profit target price
        order_price = stra.set_take_profit_price(app)
        quantity = abs(quantity)

    quantity = _normalize_order_quantity(app, quantity)
    if quantity <= 0:
        print('Take-profit quantity rounded to zero. Skipping take-profit order...')
        app.logging.info('Take-profit quantity rounded to zero. Skipping take-profit order...')
        return

    position_sign = _risk_management_position_sign(app)
    if position_sign == 0:
        print('Risk-management position sign is zero. Skipping take-profit order...')
        app.logging.info('Risk-management position sign is zero. Skipping take-profit order...')
        return

    price_side = 'down' if position_sign < 0 else 'up'
    order_price = _round_price_to_contract_tick(app, order_price, side=price_side)

    if position_sign > 0:
        direction = 'SELL'
    elif position_sign < 0:
        direction = 'BUY'
        
    # Set the add decimal to zero
    tick_size = _contract_min_tick(app)
    if not np.isfinite(tick_size) or tick_size <= 0:
        tick_size = 0.00001
    price_step = float(tick_size)
    attempts = 0

    current_order_id = int(order_id)
    asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
    while attempts < 10:
        if asset_class == 'crypto' and position_sign > 0:
            live_position = float(_latest_position_for_symbol(app, refresh=(attempts > 0), verbose=False))
            confirmed_quantity = _normalize_order_quantity(app, abs(live_position))
            if confirmed_quantity <= 0:
                attempts += 1
                time.sleep(2)
                continue
            quantity = min(quantity, confirmed_quantity)
            if quantity <= 0:
                print(f'[{app.ticker}] Crypto long position is not confirmed yet. Deferring take-profit order...')
                app.logging.info(f'[{app.ticker}] Crypto long position is not confirmed yet. Deferring take-profit order...')
                return None

        # Send the take-profit order
        _clear_order_submission_failures(app)
        app.placeOrder(current_order_id, _contract_for_order(app), ibf.tpOrder(direction, quantity, order_price, override=(asset_class == 'crypto')))
        time.sleep(3)
        # Save the output errors in data as a boolean that corresponds to any error while sending the take-profit order
        failures = _order_submission_failures(app) or {}
        crypto_short_reject = asset_class == 'crypto' and any(
            'Short sales are not allowed for this asset' in str(msg)
            for msg in failures.values()
        )
        data = bool(failures) or \
                (110 in list(app.errors_dict.keys())) or \
                (463 in list(app.errors_dict.keys()))
        # If data is true
        if data == True:
            append_runtime_audit(app, 'take_profit_rejected', json.dumps(failures, default=str))
            if crypto_short_reject:
                refreshed_live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
                confirmed_quantity = _normalize_order_quantity(app, abs(refreshed_live_quantity))
                if confirmed_quantity > 0 and confirmed_quantity < quantity:
                    quantity = confirmed_quantity
                    attempts += 1
                    current_order_id = _next_order_id(app)
                    print(f'[{app.ticker}] Crypto take-profit was oversized versus the confirmed live position. Retrying with quantity {quantity}...')
                    app.logging.info(f'[{app.ticker}] Crypto take-profit was oversized versus the confirmed live position. Retrying with quantity {quantity}...')
                    app.errors_dict = {}
                    continue
                print(f'[{app.ticker}] Crypto take-profit rejected before the position was fully confirmed. Deferring retry to the next sync...')
                app.logging.info(f'[{app.ticker}] Crypto take-profit rejected before the position was fully confirmed. Deferring retry to the next sync...')
                app.errors_dict = {}
                return None
            attempts += 1
            order_price = _round_price_to_contract_tick(app, order_price - price_step, side=price_side)
            current_order_id = _next_order_id(app)
            
            print("Couldn't transmit the take-profit order, the app will try again...")
            app.logging.info("Couldn't transmit the take-profit order, the app will try again...")
            
            # Clean the errors dictionary
            app.errors_dict = {}
        else:
            print(f'Take profit sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.logging.info(f'Take profit sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.queue_strategy_state({'risk_management': {'tp_price': float(order_price)}})
            return current_order_id
        # If the app is disconnected
        if 504 in list(app.errors_dict.keys()):
            return current_order_id
    return current_order_id
        
def send_market_order(app, order_id, quantity):
    ''' Function to send a market order '''
    
    print(f'[{app.ticker}] Sending the market order...')
    app.logging.info(f'[{app.ticker}] Sending the market order...')
    
    # If the current period is not the last of the day
    if app.current_period != app.trading_day_end_datetime:
        # If the signal tells you to go long
        if app.signal > 0:
            # Direction will be to go long
            direction = 'BUY'
        # If the signal tells you to short-sell the asset
        elif app.signal < 0:
            # Direction will be to short-sell the asset
            direction = 'SELL'
        # If the app is connected
        if app.isConnected():
            # Place the market order
            order_qty = _normalize_order_quantity(app, quantity)
            if order_qty <= 0:
                print('Order quantity rounded to zero. Skipping market order...')
                app.logging.info('Order quantity rounded to zero. Skipping market order...')
                return False
            asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
            if asset_class == 'crypto':
                live_position = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
                if direction == 'SELL' and live_position > 0:
                    live_qty = _normalize_order_quantity(app, live_position)
                    if live_qty <= 0:
                        print(f'[{app.ticker}] No confirmed live crypto long position is available to close...')
                        app.logging.info(f'[{app.ticker}] No confirmed live crypto long position is available to close...')
                        return False
                    order_qty = live_qty
                elif direction == 'BUY' and live_position < 0:
                    live_qty = _normalize_order_quantity(app, abs(live_position))
                    if live_qty <= 0:
                        print(f'[{app.ticker}] No confirmed live crypto short position is available to close...')
                        app.logging.info(f'[{app.ticker}] No confirmed live crypto short position is available to close...')
                        return False
                    order_qty = live_qty
            _clear_order_submission_failures(app)
            order_contract = _contract_for_order(app)
            if asset_class in {'futures', 'future', 'fut'}:
                local_symbol = str(getattr(order_contract, 'localSymbol', '') or '').strip()
                expiry = str(getattr(order_contract, 'lastTradeDateOrContractMonth', '') or '').strip()
                if not local_symbol and not expiry:
                    message = f'[{app.ticker}] Futures contract is unresolved. Skipping market order until a front-month contract is resolved.'
                    print(message)
                    app.logging.warning(message)
                    append_runtime_audit(app, 'market_order_skipped', message)
                    return False
                contract_trace = {
                    'symbol': getattr(order_contract, 'symbol', ''),
                    'secType': getattr(order_contract, 'secType', ''),
                    'exchange': getattr(order_contract, 'exchange', ''),
                    'currency': getattr(order_contract, 'currency', ''),
                    'localSymbol': getattr(order_contract, 'localSymbol', ''),
                    'expiry': getattr(order_contract, 'lastTradeDateOrContractMonth', ''),
                    'tradingClass': getattr(order_contract, 'tradingClass', ''),
                    'multiplier': getattr(order_contract, 'multiplier', ''),
                    'conId': getattr(order_contract, 'conId', ''),
                }
                print(f"[{app.ticker}] Order contract trace: {contract_trace}")
                app.logging.info(f"[{app.ticker}] Order contract trace: {contract_trace}")
            if asset_class == 'crypto':
                lmt_price = _crypto_marketable_limit_price(app, direction)
                app.placeOrder(order_id, order_contract, ibf.cryptoLimitOrder(direction, round(float(order_qty), 8), lmt_price, override=True))
            else:
                app.placeOrder(order_id, order_contract, ibf.marketOrder(direction, order_qty))
            time.sleep(3)
            failures = _market_order_submission_failures(app)
            if failures:
                print(f'[{app.ticker}] Market order was rejected. Skipping downstream RM orders...')
                app.logging.info(f'[{app.ticker}] Market order was rejected. Skipping downstream RM orders...')
                return False
            signed_qty = order_qty if direction == 'BUY' else -order_qty
            app.ordered_quantity = float(getattr(app, 'ordered_quantity', 0.0)) + float(signed_qty)
            print(f"[{app.ticker}] Market order sent...")
            app.logging.info(f"[{app.ticker}] Market order sent...")
            return True
        else:
            return False
    # If the current period is the last of the day
    else:
        # If the previous quantity belongs to a long position (Close the position)
        if quantity > 0:
            # Set the direction to sell the long position
            direction = 'SELL'
        # If the previous quantity belongs to a short position
        elif quantity < 0:
            # Set the direction to buy the short position (Close the position)
            direction = 'BUY'   
        # If the app is connected
        if app.isConnected():
            # Send the market order to close the position
            order_qty = _normalize_order_quantity(app, quantity)
            if order_qty <= 0:
                print('Order quantity rounded to zero. Skipping market order...')
                app.logging.info('Order quantity rounded to zero. Skipping market order...')
                return False
            asset_class = str(app.asset_spec.get('asset_class', 'forex')).lower()
            _clear_order_submission_failures(app)
            order_contract = _contract_for_order(app)
            if asset_class in {'futures', 'future', 'fut'}:
                contract_trace = {
                    'symbol': getattr(order_contract, 'symbol', ''),
                    'secType': getattr(order_contract, 'secType', ''),
                    'exchange': getattr(order_contract, 'exchange', ''),
                    'currency': getattr(order_contract, 'currency', ''),
                    'localSymbol': getattr(order_contract, 'localSymbol', ''),
                    'expiry': getattr(order_contract, 'lastTradeDateOrContractMonth', ''),
                    'tradingClass': getattr(order_contract, 'tradingClass', ''),
                    'multiplier': getattr(order_contract, 'multiplier', ''),
                    'conId': getattr(order_contract, 'conId', ''),
                }
                print(f"[{app.ticker}] Order contract trace: {contract_trace}")
                app.logging.info(f"[{app.ticker}] Order contract trace: {contract_trace}")
            if asset_class == 'crypto':
                lmt_price = _crypto_marketable_limit_price(app, direction)
                app.placeOrder(order_id, order_contract, ibf.cryptoLimitOrder(direction, round(float(order_qty), 8), lmt_price, override=True))
            else:
                app.placeOrder(order_id, order_contract, ibf.marketOrder(direction, order_qty))
            time.sleep(3)
            failures = _market_order_submission_failures(app)
            if failures:
                print(f'[{app.ticker}] Market order was rejected. Skipping downstream RM orders...')
                app.logging.info(f'[{app.ticker}] Market order was rejected. Skipping downstream RM orders...')
                return False
            signed_qty = order_qty if direction == 'BUY' else -order_qty
            app.ordered_quantity = float(getattr(app, 'ordered_quantity', 0.0)) + float(signed_qty)
            print(f"[{app.ticker}] Market order sent...")
            app.logging.info(f"[{app.ticker}] Market order sent...")
            return True
        else:
            return False
                    
def get_previous_quantity(app):
    ''' Function to get the previous position quantity'''

    if app.isConnected() and not _shared_pretrade_snapshot_active(app):
        try:
            app.previous_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
            return
        except Exception:
            pass

    if app.pos_df.empty:
        app.previous_quantity = 0
        return

    symbol = str(getattr(app.contract, 'symbol', ''))
    currency = str(getattr(app.contract, 'currency', ''))
    pos_df = app.pos_df.copy()

    if 'Symbol' in pos_df.columns:
        pos_df['Symbol'] = pos_df['Symbol'].astype(str)
    if 'Currency' in pos_df.columns:
        pos_df['Currency'] = pos_df['Currency'].astype(str)

    matches = pos_df[
        (pos_df['Symbol'] == symbol) &
        (pos_df['Currency'] == currency)
    ]

    if matches.empty and 'Symbol' in pos_df.columns:
        matches = pos_df[pos_df['Symbol'] == symbol]

    if matches.empty:
        app.previous_quantity = 0
        append_runtime_audit(app, 'previous_quantity_missing', f'No position row matched {symbol}/{currency}; defaulted to 0')
        return

    app.previous_quantity = float(pd.to_numeric(matches['Position'], errors='coerce').fillna(0.0).iloc[-1])
        
def get_current_quantity(app):
    ''' Function to get the current position quantity'''
    asset_class = app.asset_spec.get("asset_class", "forex")
    multiplier = float(app.asset_spec.get("multiplier") or 1.0)
    strategy_targets = getattr(app, 'strategy_targets', {}) or {}
    symbol_target = strategy_targets.get(str(getattr(app, 'ticker', '')).upper(), {}) if isinstance(strategy_targets, dict) else {}
    quantity_mode = str(symbol_target.get('quantity_mode') or '').lower()
    target_quantity = symbol_target.get('target_quantity', np.nan)
    target_weight = pd.to_numeric(pd.Series([symbol_target.get('target_weight', np.nan)]), errors='coerce').iloc[0]
    target_notional_value = pd.to_numeric(pd.Series([symbol_target.get('target_notional', np.nan)]), errors='coerce').iloc[0]

    if quantity_mode in {'contracts', 'units', 'quantity'} and pd.notna(target_quantity):
        app.current_quantity = abs(float(_normalize_order_quantity(app, float(target_quantity))))
        app.execution_effective_signed_quantity = float(np.sign(getattr(app, 'signal', 0.0))) * app.current_quantity
        app.execution_residual_quantity = 0.0
        _queue_execution_residual(app, 0.0)
        return
    
    # app.capital has already been scaled by app.leverage in portfolio_allocation(app)
    if np.isfinite(target_weight):
        target_notional = abs(float(app.capital)) * max(0.0, float(target_weight))
    elif np.isfinite(target_notional_value):
        target_notional = max(0.0, float(target_notional_value))
    else:
        target_notional = app.capital
    
    if float(getattr(app, 'signal', 0.0)) == 0.0 or target_notional <= 0:
        app.current_quantity = 0.0
    elif app.last_value <= 0 and asset_class in ("futures", "crypto", "metals", "stock"):
        app.current_quantity = 0.0
    elif asset_class == "futures":
        # For futures: contracts = Target Notional / (Price * Multiplier)
        app.current_quantity = (target_notional) / (app.last_value * multiplier)
    elif asset_class in ("crypto", "metals", "stock"):
        # For crypto and spot metals (XAUUSD): units = Target Notional / Price
        app.current_quantity = (target_notional) / app.last_value
    else:
        # Forex: Quantity is in base units = Target Notional
        app.current_quantity = target_notional

    _apply_residual_quantity_policy(app)

def get_previous_and_current_quantities(app):
    ''' Function to get the previous and current positions quantities'''
    verbose = not _shared_cycle_quiet_mode(app)

    if verbose:
        print('Update the previous and current positions quantities...')
        app.logging.info('Update the previous and current positions quantities...')
    
    # If the app is connected
    if app.isConnected():
        # Only recompute portfolio capital locally when the strategy uses
        # weight/notional-based sizing. Explicit target quantities are already
        # computed globally in the portfolio target pass.
        if not _uses_explicit_target_quantity(app):
            portfolio_allocation(app)
        # Update the last value of the asset
        update_asset_last_value(app, verbose=verbose)
    else:
        return

    # If the app is connected
    if app.isConnected():
        # Set the executors list
        executors_list = []
        # Append the functions to be multithreaded
        with ThreadPoolExecutor(2) as executor:
            executors_list.append(executor.submit(get_previous_quantity, app)) 
            executors_list.append(executor.submit(get_current_quantity, app)) 

        # Run the executors in parallel
        for x in executors_list:
            x.result()
    else:
        return
            
    if verbose:
        print('The previous and current positions quantities were successfully updated...')
        app.logging.info('The previous and current positions quantities were successfully updated...')
    
def update_trading_info(app, verbose=True):
    ''' Function to get the previous trading information'''

    if verbose:
        print('Update the previous trading information...')
        app.logging.info('Update the previous trading information...')
    append_runtime_audit(app, 'update_trading_info', f'period={app.current_period}')

    if _defer_posttrade_sync(app) and (not verbose) and (not _shared_pretrade_snapshot_active(app)):
        return
    
    if _shared_pretrade_snapshot_active(app):
        if verbose:
            print(f'[{app.ticker}] Using the shared pre-trade broker snapshot for positions, orders, and executions...')
            app.logging.info(f'[{app.ticker}] Using the shared pre-trade broker snapshot for positions, orders, and executions...')
        if getattr(app, 'resolved_contract', None) is None:
            request_contract_details(app, verbose=verbose)
        update_risk_management_orders(app, verbose=verbose)
        return
    request_contract_details(app, verbose=verbose)
    request_positions(app, verbose=verbose)
    request_orders(app, verbose=verbose)
    update_submitted_orders(app, verbose=verbose)
        
    
def update_cash_balance_values_for_signals(app):
    ''' Function to update the cash balance signal and leverage'''

    print('Update the cash balance signal and leverage...')
    app.logging.info('Update the cash balance signal and leverage...')
    
    # Update the leverage value in the cash balance dataframe
    app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'leverage'] = app.leverage
    # Update the signal value in the cash balance dataframe
    app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'signal'] = app.signal     
    # Forward fill the cash balance dataframe
    app.cash_balance.ffill(inplace=True)
            
    print('The cash balance signal and leverage were successfully updated...')
    app.logging.info('The cash balance signal and leverage were successfully updated...')


def _next_order_id(app):
    allocator = getattr(app, 'shared_order_id_allocator', None)
    if allocator is not None:
        return int(allocator.reserve())

    if app.isConnected():
        app.reqIds(-1)
        time.sleep(2)
        return int(app.nextValidOrderId)

    raise RuntimeError(f'[{app.ticker}] Could not reserve an order id because the app is disconnected.')


def _next_bracket_order_id(app, previous_order_id=None):
    next_id = _next_order_id(app)
    if previous_order_id is None:
        return next_id
    try:
        previous_int = int(previous_order_id)
    except (TypeError, ValueError):
        return next_id
    return max(next_id, previous_int + 1)


def _risk_management_position_sign(app):
    explicit_sign = pd.to_numeric(pd.Series([getattr(app, 'risk_management_position_sign', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(explicit_sign) and explicit_sign != 0:
        return float(np.sign(explicit_sign))
    live_quantity = pd.to_numeric(pd.Series([getattr(app, 'previous_quantity', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(live_quantity):
        live_sign = _effective_position_sign(app, live_quantity)
        if live_sign != 0:
            return live_sign
    signal_value = pd.to_numeric(pd.Series([getattr(app, 'signal', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(signal_value) and signal_value != 0:
        return float(np.sign(signal_value))
    return 0.0


def _defer_posttrade_sync(app):
    return bool(getattr(app, 'defer_posttrade_sync', False))


def _defer_synthetic_monitors(app):
    return bool(getattr(app, 'defer_synthetic_monitors', False))


def _allow_symbol_level_broker_refresh(app):
    if bool(getattr(app, 'parallel_isolated_order_worker', False)):
        return True
    return (not _defer_posttrade_sync(app)) and (not _shared_pretrade_snapshot_active(app))


def _shared_cycle_quiet_mode(app):
    return bool(getattr(app, 'defer_posttrade_sync', False))


def _shared_cycle_symbol_broker_pull_allowed(app):
    return bool(getattr(app, 'force_shared_broker_pull', False)) or _allow_symbol_level_broker_refresh(app)


def _is_crypto_asset(app):
    return str(getattr(app, 'asset_spec', {}).get('asset_class', 'forex')).lower() == 'crypto'


def _crypto_native_stop_supported(app):
    if not _is_crypto_asset(app):
        return True
    desired_type = 'TRAIL' if getattr(app, 'trail', False) else 'STP'
    return desired_type in _contract_order_types(app)


def _best_live_market_price(app):
    for candidate in [
        getattr(app, 'last_trade_price', np.nan),
        getattr(app, 'last_value', np.nan),
        getattr(app, 'bid_price', np.nan),
        getattr(app, 'ask_price', np.nan),
    ]:
        value = pd.to_numeric(pd.Series([candidate]), errors='coerce').iloc[0]
        if np.isfinite(value) and float(value) > 0:
            return float(value)
    return np.nan


def _request_live_market_price_snapshot(app, timeout_seconds=8):
    req_id = None
    try:
        req_id = _start_market_data_stream(app)
        deadline = time.time() + max(float(timeout_seconds), 0.0)
        while time.time() < deadline:
            if app.market_data_event.wait(timeout=0.5):
                app.market_data_event.clear()
                price = _best_live_market_price(app)
                if np.isfinite(price) and float(price) > 0:
                    return float(price)
        return _best_live_market_price(app)
    finally:
        if req_id is not None:
            _stop_market_data_stream(app, req_id=req_id)


def _stop_is_breached(position_quantity, live_price, stop_price):
    if not np.isfinite(live_price) or not np.isfinite(stop_price):
        return False
    if float(position_quantity) > 0:
        return float(live_price) <= float(stop_price)
    if float(position_quantity) < 0:
        return float(live_price) >= float(stop_price)
    return False


def _start_market_data_stream(app, req_id=None):
    if not app.isConnected():
        return None
    market_data_req_id = int(req_id) if req_id is not None else _next_order_id(app)
    app.market_data_event.clear()
    app.active_market_data_req_id = market_data_req_id
    app.reqMktData(market_data_req_id, _contract_for_order(app), "", False, False, [])
    return market_data_req_id


def _stop_market_data_stream(app, req_id=None):
    market_data_req_id = req_id if req_id is not None else getattr(app, 'active_market_data_req_id', None)
    if market_data_req_id is None:
        return
    try:
        app.cancelMktData(int(market_data_req_id))
    except Exception:
        pass
    if getattr(app, 'active_market_data_req_id', None) == market_data_req_id:
        app.active_market_data_req_id = None


def _wait_for_live_position(app, expected_sign=None, timeout_seconds=CRYPTO_FILL_WAIT_SECONDS):
    if _defer_posttrade_sync(app):
        time.sleep(min(max(float(timeout_seconds), 0.0), 2.0))
        live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        if expected_sign is None:
            return live_quantity
        if _is_effectively_flat_quantity(app, live_quantity):
            return live_quantity
        if np.sign(live_quantity) == np.sign(expected_sign):
            return live_quantity
        return live_quantity
    deadline = time.time() + max(float(timeout_seconds), 0.0)
    last_live_quantity = float(_latest_position_for_symbol(app))
    while time.time() < deadline:
        live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        last_live_quantity = live_quantity
        if not _is_effectively_flat_quantity(app, live_quantity):
            if expected_sign is None or np.sign(live_quantity) == np.sign(expected_sign):
                return live_quantity
        time.sleep(1)
    return last_live_quantity


def _stop_synthetic_crypto_monitor(app, wait=False):
    cancel_event = getattr(app, 'synthetic_stop_cancel_event', None)
    if isinstance(cancel_event, Event):
        cancel_event.set()
    worker = getattr(app, 'synthetic_stop_thread', None)
    if wait and worker is not None and worker.is_alive():
        worker.join(timeout=2)
    app.synthetic_stop_thread = None
    app.synthetic_stop_metadata = {}


def _clear_pending_synthetic_crypto_monitor(app):
    symbol = str(getattr(app, 'ticker', '')).upper()
    pending = getattr(app, 'pending_synthetic_monitors', []) or []
    app.pending_synthetic_monitors = [
        item for item in pending
        if str(item.get('symbol', '')).upper() != symbol
    ]


def _queue_synthetic_crypto_monitor(app, take_profit_order_id=None, stop_price=None):
    if not _is_crypto_asset(app) or _crypto_native_stop_supported(app):
        return
    if stop_price is None:
        stop_price = _latest_synthetic_stop_price(app)
    stop_price = pd.to_numeric(pd.Series([stop_price]), errors='coerce').iloc[0]
    if not np.isfinite(stop_price):
        return
    _clear_pending_synthetic_crypto_monitor(app)
    pending = getattr(app, 'pending_synthetic_monitors', []) or []
    pending.append({
        'symbol': str(getattr(app, 'ticker', '')).upper(),
        'asset_spec': deepcopy(getattr(app, 'asset_spec', {}) or {}),
        'resolved_contract': deepcopy(getattr(app, 'resolved_contract', None)),
        'take_profit_order_id': int(take_profit_order_id) if isinstance(take_profit_order_id, (int, np.integer)) else None,
        'stop_price': float(stop_price),
    })
    app.pending_synthetic_monitors = pending


def _latest_synthetic_stop_price(app):
    override_prices = getattr(app, 'risk_management_price_overrides', {}) or {}
    override_sl = pd.to_numeric(pd.Series([override_prices.get('sl', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(override_sl):
        return float(override_sl)
    state_prices = _latest_strategy_state_risk_management_prices(app)
    state_sl = pd.to_numeric(pd.Series([state_prices.get('sl', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(state_sl):
        return float(state_sl)
    history_prices = _latest_historical_risk_management_prices(app)
    history_sl = pd.to_numeric(pd.Series([history_prices.get('sl', np.nan)]), errors='coerce').iloc[0]
    if np.isfinite(history_sl):
        return float(history_sl)
    return np.nan


def _sync_synthetic_crypto_state(app):
    if not app.isConnected():
        return
    request_orders(app, verbose=False)
    update_submitted_orders(app, verbose=False)
    request_positions(app, verbose=False)
    save_data(app, verbose=False)


def _refresh_live_state_for_persistence(app, verbose=False):
    if not app.isConnected():
        return
    if _defer_posttrade_sync(app):
        return
    request_orders(app, verbose=verbose)
    update_submitted_orders(app, verbose=verbose)
    request_positions(app, verbose=verbose)


def _finalize_apps_for_persistence(apps):
    for app in apps:
        if not app.isConnected():
            continue
        _refresh_live_state_for_persistence(app, verbose=False)
        _flush_live_trading_buffers(app)
        _flush_contract_details_buffer(app)
        _flush_temp_sheet(app, 'temp_portfolio_snapshots', 'portfolio_snapshots_df', 'portfolio_snapshots', dedupe_subset=['datetime', 'Account', 'Symbol', 'ConId'])
        _flush_runtime_audit_buffer(app)
        _flush_strategy_state_buffer(app)


def _synthetic_crypto_stop_monitor(app, cancel_event, stop_price, deadline):
    req_id = None
    try:
        if not app.isConnected():
            return
        print(f'[{app.ticker}] Starting synthetic crypto stop monitor until {deadline}...')
        app.logging.info(f'[{app.ticker}] Starting synthetic crypto stop monitor until {deadline}...')
        req_id = _start_market_data_stream(app)
        live_quantity = float(_latest_position_for_symbol(app))
        if _is_effectively_flat_quantity(app, live_quantity):
            return
        while dt.datetime.now() < deadline and not cancel_event.is_set():
            app.market_data_event.wait(timeout=CRYPTO_STOP_POLL_SECONDS)
            app.market_data_event.clear()
            live_price = _best_live_market_price(app)
            if not np.isfinite(live_price):
                continue
            if not _stop_is_breached(live_quantity, live_price, stop_price):
                continue

            print(
                f'[{app.ticker}] Synthetic stop breached at {live_price:.8f} '
                f'against stop {stop_price:.8f}. Canceling take-profit and sending market exit...'
            )
            app.logging.info(
                f'[{app.ticker}] Synthetic stop breached at {live_price:.8f} '
                f'against stop {stop_price:.8f}. Canceling take-profit and sending market exit...'
            )
            request_orders(app, verbose=False)
            update_submitted_orders(app, verbose=False)
            cancel_previous_take_profit_order(app)
            time.sleep(1)
            live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
            if _is_effectively_flat_quantity(app, live_quantity):
                _sync_synthetic_crypto_state(app)
                return
            original_signal = float(getattr(app, 'signal', 0.0))
            app.signal = -1.0 if live_quantity > 0 else 1.0
            try:
                exit_order_id = _next_order_id(app)
                send_market_order(app, exit_order_id, abs(live_quantity))
            finally:
                app.signal = original_signal
            print(
                f'[{app.ticker}] Synthetic stop exit order sent: '
                f'order_id={exit_order_id}, side={"SELL" if live_quantity > 0 else "BUY"}, '
                f'quantity={abs(live_quantity):.8f}'
            )
            app.logging.info(
                f'[{app.ticker}] Synthetic stop exit order sent: '
                f'order_id={exit_order_id}, side={"SELL" if live_quantity > 0 else "BUY"}, '
                f'quantity={abs(live_quantity):.8f}'
            )
            _sync_synthetic_crypto_state(app)
            return
        if not cancel_event.is_set():
            _sync_synthetic_crypto_state(app)
    finally:
        _stop_market_data_stream(app, req_id)


def _start_synthetic_crypto_monitor(app):
    if not _is_crypto_asset(app) or _crypto_native_stop_supported(app):
        return
    stop_price = _latest_synthetic_stop_price(app)
    if not np.isfinite(stop_price):
        return
    deadline = getattr(app, 'next_period', None)
    if deadline is None or dt.datetime.now() >= deadline:
        return
    _stop_synthetic_crypto_monitor(app, wait=True)
    cancel_event = Event()
    app.synthetic_stop_cancel_event = cancel_event
    app.synthetic_stop_metadata = {'stop_price': float(stop_price), 'deadline': deadline}
    worker = threading.Thread(
        target=_synthetic_crypto_stop_monitor,
        args=(app, cancel_event, float(stop_price), deadline),
        daemon=True,
    )
    app.synthetic_stop_thread = worker
    worker.start()


ORDER_SUBMISSION_FAILURE_CODES = {
    110,
    10052,
    10287,
    10288,
    10289,
    10293,
    10286,
    10330,
    2163,
    321,
    320,
    387,
    201,
}

CRYPTO_RECONCILE_MAX_ATTEMPTS = 8


def _clear_order_submission_failures(app):
    for code in ORDER_SUBMISSION_FAILURE_CODES:
        app.errors_dict.pop(code, None)


def _market_order_submission_failures(app):
    failures = {
        code: app.errors_dict[code]
        for code in ORDER_SUBMISSION_FAILURE_CODES
        if code in app.errors_dict
    }
    if failures:
        append_runtime_audit(app, 'market_order_rejected', json.dumps(failures, default=str))
    return failures


def _order_submission_failures(app):
    return {
        code: app.errors_dict[code]
        for code in ORDER_SUBMISSION_FAILURE_CODES
        if code in app.errors_dict
    }


def _crypto_marketable_limit_price(app, direction):
    reference_price = _request_live_market_price_snapshot(app)
    if not np.isfinite(reference_price) or float(reference_price) <= 0:
        reference_price = pd.to_numeric(pd.Series([getattr(app, 'last_value', np.nan)]), errors='coerce').iloc[0]
    if not np.isfinite(reference_price) or float(reference_price) <= 0:
        reference_price = 1.0

    if str(direction).upper() == 'BUY':
        raw_price = float(reference_price) * 2.0
        return _round_price_to_contract_tick(app, raw_price, side='up')

    raw_price = float(reference_price) * 0.5
    return _round_price_to_contract_tick(app, raw_price, side='down')


def _contract_order_types(app):
    details = getattr(app, 'contract_details_df', None)
    if not isinstance(details, pd.DataFrame) or details.empty or 'Symbol' not in details.columns:
        return set()
    same_symbol = details[details['Symbol'].astype(str).str.upper() == str(app.ticker).upper()]
    if same_symbol.empty or 'OrderTypes' not in same_symbol.columns:
        return set()
    raw = str(same_symbol.iloc[-1].get('OrderTypes', '') or '')
    return {part.strip().upper() for part in raw.split(',') if part.strip()}


def _latest_position_for_symbol(app, refresh=True, verbose=True):
    if refresh and app.isConnected() and _allow_symbol_level_broker_refresh(app):
        request_positions(app, verbose=verbose)
    shared_state = getattr(app, 'shared_broker_state', {}) or {}
    positions = shared_state.get('positions') if _shared_cycle_quiet_mode(app) else getattr(app, 'pos_df', None)
    if not isinstance(positions, pd.DataFrame) or positions.empty:
        return 0.0
    local = positions.reset_index(drop=False) if not isinstance(positions.index, pd.RangeIndex) else positions.copy()
    symbol_value = str(getattr(app.contract, 'symbol', app.ticker)).upper()
    sec_type_value = str(getattr(app.contract, 'secType', '')).upper()
    currency_value = str(getattr(app.contract, 'currency', '') or '').upper()
    symbol_mask = local['Symbol'].astype(str).str.upper() == symbol_value
    sec_type_mask = local['SecType'].astype(str).str.upper() == sec_type_value
    same_contract = local[symbol_mask & sec_type_mask].copy()
    if currency_value and 'Currency' in same_contract.columns:
        currency_mask = same_contract['Currency'].astype(str).str.upper() == currency_value
        if currency_mask.any():
            same_contract = same_contract[currency_mask].copy()
    if same_contract.empty:
        return 0.0
    return float(pd.to_numeric(same_contract['Position'], errors='coerce').fillna(0.0).iloc[-1])


def _reconcile_crypto_live_position(app, target_signed_quantity):
    previous_distance = None

    for attempt in range(1, CRYPTO_RECONCILE_MAX_ATTEMPTS + 1):
        live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        distance = abs(float(target_signed_quantity) - live_quantity)
        if np.isclose(distance, 0.0, atol=1e-8):
            return live_quantity, True

        executable_delta_quantity = _normalize_order_quantity(app, abs(float(target_signed_quantity) - live_quantity))
        if executable_delta_quantity <= 0:
            return live_quantity, True

        original_signal = float(app.signal)
        app.signal = float(np.sign(float(target_signed_quantity) - live_quantity))
        try:
            market_sent = send_market_order(app, _next_order_id(app), executable_delta_quantity)
        finally:
            app.signal = original_signal

        if not market_sent:
            refreshed_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
            return refreshed_quantity, False

        time.sleep(2)
        refreshed_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        refreshed_distance = abs(float(target_signed_quantity) - refreshed_quantity)

        append_runtime_audit(
            app,
            'crypto_reconcile',
            f'attempt={attempt}, target={target_signed_quantity}, before={live_quantity}, after={refreshed_quantity}',
        )

        if np.isclose(refreshed_distance, 0.0, atol=1e-8):
            return refreshed_quantity, True

        if previous_distance is not None and refreshed_distance >= previous_distance - 1e-8 and refreshed_distance >= distance - 1e-8:
            return refreshed_quantity, False

        previous_distance = refreshed_distance

    final_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
    return final_quantity, np.isclose(abs(float(target_signed_quantity) - final_quantity), 0.0, atol=1e-8)
    
def send_orders_as_bracket(app, order_id, quantity, mkt_order, sl_order, tp_order, rm_quantity=None):
    ''' Function to send the orders as a bracket'''

    # Send a market and risk management orders
    if (mkt_order==True) and (sl_order==True) and (tp_order==True):
        market_sent = send_market_order(app, _next_order_id(app), quantity)
        if not market_sent:
            return
        risk_qty = quantity if rm_quantity is None else rm_quantity
        rm_sign = float(np.sign(getattr(app, 'signal', 0.0)))
        if _is_crypto_asset(app):
            confirmed_live_quantity = _wait_for_live_position(app, expected_sign=rm_sign, timeout_seconds=CRYPTO_FILL_WAIT_SECONDS)
            if _is_effectively_flat_quantity(app, confirmed_live_quantity):
                print(f'[{app.ticker}] Crypto entry position is not confirmed yet. Skipping downstream risk-management orders...')
                app.logging.info(f'[{app.ticker}] Crypto entry position is not confirmed yet. Skipping downstream risk-management orders...')
                return
            risk_qty = abs(confirmed_live_quantity)
            rm_sign = float(np.sign(confirmed_live_quantity))
        app.risk_management_position_sign = rm_sign
        try:
            sl_id = send_stop_loss_order(app, _next_order_id(app), risk_qty)
            tp_id = send_take_profit_order(app, _next_bracket_order_id(app, sl_id), risk_qty)
            if _is_crypto_asset(app) and not _crypto_native_stop_supported(app):
                if _defer_synthetic_monitors(app):
                    _queue_synthetic_crypto_monitor(app, tp_id)
                else:
                    _start_synthetic_crypto_monitor(app)
            _refresh_live_state_for_persistence(app, verbose=False)
        finally:
            app.risk_management_position_sign = np.nan
            
    # Send only the risk management orders
    elif (mkt_order==False) and (sl_order==True) and (tp_order==True):
        risk_qty = quantity
        rm_sign = float(np.sign(getattr(app, 'previous_quantity', getattr(app, 'signal', 0.0))))
        if _is_crypto_asset(app):
            live_quantity = float(_latest_position_for_symbol(app))
            if _is_effectively_flat_quantity(app, live_quantity):
                print(f'[{app.ticker}] No confirmed crypto position is live. Skipping risk-management refresh...')
                app.logging.info(f'[{app.ticker}] No confirmed crypto position is live. Skipping risk-management refresh...')
                return
            risk_qty = abs(live_quantity)
            rm_sign = float(np.sign(live_quantity))
        app.risk_management_position_sign = rm_sign
        try:
            sl_id = send_stop_loss_order(app, _next_order_id(app), risk_qty)
            tp_id = send_take_profit_order(app, _next_bracket_order_id(app, sl_id), risk_qty)
            if _is_crypto_asset(app) and not _crypto_native_stop_supported(app):
                if _defer_synthetic_monitors(app):
                    _queue_synthetic_crypto_monitor(app, tp_id)
                else:
                    _start_synthetic_crypto_monitor(app)
            _refresh_live_state_for_persistence(app, verbose=False)
        finally:
            app.risk_management_position_sign = np.nan
    # Send only the market order
    elif (mkt_order==True) and (sl_order==False) and (tp_order==False):
        send_market_order(app, _next_order_id(app), quantity)
    else:
        pass


def _cancel_live_risk_management_orders(app, current_orders):
    if not isinstance(current_orders, pd.DataFrame) or current_orders.empty or not app.isConnected():
        return
    if 'OrderType' not in current_orders.columns or 'OrderId' not in current_orders.columns:
        return
    rm_mask = current_orders['OrderType'].astype(str).str.upper().isin(['STP', 'TRAIL', 'LMT'])
    rm_orders = current_orders.loc[rm_mask, 'OrderId']
    rm_order_ids = pd.to_numeric(rm_orders, errors='coerce').dropna().astype(int).drop_duplicates().tolist()
    if not rm_order_ids:
        return
    for order_id in rm_order_ids:
        app.cancelOrder(int(order_id), OrderCancel())
        time.sleep(1)
    print(f'[{app.ticker}] Canceled existing live RM orders to rebuild a clean carry-protection set...')
    app.logging.info(f'[{app.ticker}] Canceled existing live RM orders to rebuild a clean carry-protection set...')


def _fresh_carry_prices_from_market(app):
    return {
        'sl': float(stra.set_stop_loss_price(app)),
        'tp': float(stra.set_take_profit_price(app)),
    }


def restore_carry_risk_management(app):
    """Recreate broker-dropped protection between market reopen and trading-day origin."""

    print(f'[{app.ticker}] Restoring carry risk management orders if needed...')
    app.logging.info(f'[{app.ticker}] Restoring carry risk management orders if needed...')

    update_trading_info(app)

    asset_class = str(getattr(app, 'asset_spec', {}).get('asset_class', 'forex')).lower()
    if asset_class in {'futures', 'future', 'fut'}:
        print(f'[{app.ticker}] Carry protection refresh is disabled for futures during the reopen bridge. Skipping...')
        app.logging.info(f'[{app.ticker}] Carry protection refresh is disabled for futures during the reopen bridge. Skipping...')
        return

    if not _is_within_asset_trading_hours(app):
        print(f'[{app.ticker}] Outside this asset trading window. Skipping carry protection refresh...')
        app.logging.info(f'[{app.ticker}] Outside this asset trading window. Skipping carry protection refresh...')
        return

    live_quantity = float(_latest_position_for_symbol(app))
    if _is_effectively_flat_quantity(app, live_quantity):
        print(f'[{app.ticker}] No live position. Carry protection refresh not needed...')
        app.logging.info(f'[{app.ticker}] No live position. Carry protection refresh not needed...')
        return

    update_asset_last_value(app)

    required_action = _required_risk_management_action(live_quantity)
    crypto_without_native_stop = _is_crypto_asset(app) and not _crypto_native_stop_supported(app)
    current_orders = _active_open_orders_snapshot(app)
    if not current_orders.empty and 'OrderType' in current_orders.columns:
        order_types = current_orders['OrderType'].astype(str).str.upper()
        if required_action and 'Action' in current_orders.columns:
            action_mask = current_orders['Action'].astype(str).str.upper().str.strip() == required_action
            current_orders = current_orders[action_mask].copy()
            order_types = current_orders['OrderType'].astype(str).str.upper() if not current_orders.empty else pd.Series(dtype=str)
        live_sl_rows = current_orders[order_types.isin(['TRAIL', 'STP'])].copy()
        live_tp_rows = current_orders[order_types == 'LMT'].copy()
        live_sl_count = live_sl_rows['OrderId'].nunique() if 'OrderId' in live_sl_rows.columns else len(live_sl_rows)
        live_tp_count = live_tp_rows['OrderId'].nunique() if 'OrderId' in live_tp_rows.columns else len(live_tp_rows)
        has_complete_crypto_protection = crypto_without_native_stop and live_tp_count == 1
        has_complete_native_protection = live_sl_count == 1 and live_tp_count == 1
        if has_complete_crypto_protection or has_complete_native_protection:
            if crypto_without_native_stop:
                _start_synthetic_crypto_monitor(app)
            print(f'[{app.ticker}] Broker already has live carry protection orders. Skipping refresh...')
            app.logging.info(f'[{app.ticker}] Broker already has live carry protection orders. Skipping refresh...')
            return
        all_live_rm_orders = _active_open_orders_snapshot(app)
        if not all_live_rm_orders.empty and 'OrderType' in all_live_rm_orders.columns:
            all_rm_types = all_live_rm_orders['OrderType'].astype(str).str.upper()
            any_live_rm = bool(all_rm_types.isin(['TRAIL', 'STP', 'LMT']).any())
        else:
            any_live_rm = False
        if any_live_rm:
            print(f'[{app.ticker}] Broker has an incomplete or duplicate carry protection set. Recreating a clean SL/TP pair...')
            app.logging.info(f'[{app.ticker}] Broker has an incomplete or duplicate carry protection set. Recreating a clean SL/TP pair...')
            _cancel_live_risk_management_orders(app, all_live_rm_orders)
            time.sleep(2)
            request_orders(app)
            update_submitted_orders(app)
            live_quantity = float(_latest_position_for_symbol(app))
            if _is_effectively_flat_quantity(app, live_quantity):
                print(f'[{app.ticker}] Position closed while rebuilding carry protection. Skipping refresh...')
                app.logging.info(f'[{app.ticker}] Position closed while rebuilding carry protection. Skipping refresh...')
                return

    live_quantity = float(_latest_position_for_symbol(app))
    if _is_effectively_flat_quantity(app, live_quantity):
        print(f'[{app.ticker}] No live position after broker refresh. Carry protection refresh not needed...')
        app.logging.info(f'[{app.ticker}] No live position after broker refresh. Carry protection refresh not needed...')
        return

    carry_prices = _latest_strategy_state_risk_management_prices(app)
    if 'sl' not in carry_prices or 'tp' not in carry_prices:
        carry_prices = _latest_historical_risk_management_prices(app)
    if 'sl' not in carry_prices or 'tp' not in carry_prices:
        print(f'[{app.ticker}] Previous carry protection prices were not found. Computing fresh SL/TP from current market...')
        app.logging.info(f'[{app.ticker}] Previous carry protection prices were not found. Computing fresh SL/TP from current market...')
        carry_prices = _fresh_carry_prices_from_market(app)

    app.previous_quantity = live_quantity
    app.signal = float(np.sign(live_quantity))
    if not _carry_prices_valid_for_market(live_quantity, float(getattr(app, 'last_value', np.nan)), carry_prices):
        print(f'[{app.ticker}] Previous carry prices are invalid against current market. Recomputing fresh SL/TP...')
        app.logging.info(f'[{app.ticker}] Previous carry prices are invalid against current market. Recomputing fresh SL/TP...')
        carry_prices = _fresh_carry_prices_from_market(app)

    live_quantity = float(_latest_position_for_symbol(app))
    if _is_effectively_flat_quantity(app, live_quantity):
        print(f'[{app.ticker}] Position closed before RM transmission. Skipping carry protection refresh...')
        app.logging.info(f'[{app.ticker}] Position closed before RM transmission. Skipping carry protection refresh...')
        return
    app.risk_management_price_overrides = carry_prices
    app.risk_management_position_sign = float(np.sign(live_quantity))
    app.previous_quantity = live_quantity
    app.signal = float(np.sign(live_quantity))
    try:
        sl_id = send_stop_loss_order(app, _next_order_id(app), abs(live_quantity))
        send_take_profit_order(app, _next_bracket_order_id(app, sl_id), abs(live_quantity))
        if crypto_without_native_stop:
            _start_synthetic_crypto_monitor(app)
        _refresh_live_state_for_persistence(app, verbose=False)
    finally:
        app.risk_management_price_overrides = {}
        app.risk_management_position_sign = np.nan

    time.sleep(2)
    refreshed_quantity = float(_latest_position_for_symbol(app))
    if not _is_effectively_flat_quantity(app, refreshed_quantity):
        size_changed = not np.isclose(abs(refreshed_quantity), abs(live_quantity), atol=1e-8)
        sign_changed = np.sign(refreshed_quantity) != np.sign(live_quantity)
        if size_changed or sign_changed:
            print(f'[{app.ticker}] Position changed during carry refresh. Rebinding RM orders to the updated live position...')
            app.logging.info(f'[{app.ticker}] Position changed during carry refresh. Rebinding RM orders to the updated live position...')
            request_orders(app)
            _cancel_live_risk_management_orders(app, _current_open_orders_snapshot(app))
            time.sleep(2)
            request_orders(app)
            update_submitted_orders(app)
            update_asset_last_value(app)
            app.previous_quantity = refreshed_quantity
            app.signal = float(np.sign(refreshed_quantity))
            app.risk_management_price_overrides = _fresh_carry_prices_from_market(app)
            app.risk_management_position_sign = float(np.sign(refreshed_quantity))
            try:
                sl_id = send_stop_loss_order(app, _next_order_id(app), abs(refreshed_quantity))
                send_take_profit_order(app, _next_bracket_order_id(app, sl_id), abs(refreshed_quantity))
                if crypto_without_native_stop:
                    _start_synthetic_crypto_monitor(app)
                _refresh_live_state_for_persistence(app, verbose=False)
            finally:
                app.risk_management_price_overrides = {}
                app.risk_management_position_sign = np.nan
    return True
    
def _reconcile_direct_target_position(app, order_id):
    target_signed_quantity = float(app.signal) * float(app.current_quantity)
    previous_quantity = float(app.previous_quantity)
    if _is_crypto_asset(app):
        previous_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        app.previous_quantity = previous_quantity
    delta_quantity = target_signed_quantity - previous_quantity
    executable_delta_quantity = _normalize_order_quantity(app, abs(delta_quantity))
    effective_signed_quantity = target_signed_quantity

    if executable_delta_quantity > 0 and app.risk_management_bool:
        cancel_risk_management_previous_orders(app)

    if executable_delta_quantity > 0 and not np.isclose(delta_quantity, 0.0, atol=1e-8):
        if _is_crypto_asset(app):
            reconciled_quantity, market_sent = _reconcile_crypto_live_position(app, target_signed_quantity)
            effective_signed_quantity = reconciled_quantity
            if not market_sent:
                return
        else:
            original_signal = float(app.signal)
            app.signal = float(np.sign(delta_quantity))
            market_sent = send_market_order(app, _next_order_id(app), executable_delta_quantity)
            app.signal = original_signal
            if not market_sent:
                return
    elif executable_delta_quantity <= 0 and app.risk_management_bool:
        live_quantity = float(_latest_position_for_symbol(app))
        if not _is_effectively_flat_quantity(app, live_quantity) and np.sign(live_quantity) == np.sign(target_signed_quantity):
            effective_signed_quantity = live_quantity

    if app.risk_management_bool and not _is_effectively_flat_quantity(app, effective_signed_quantity):
        if executable_delta_quantity <= 0:
            cancel_risk_management_previous_orders(app)
        rm_quantity = abs(effective_signed_quantity)
        rm_sign = float(np.sign(effective_signed_quantity))
        if _is_crypto_asset(app):
            confirmed_live_quantity = _wait_for_live_position(app, expected_sign=rm_sign, timeout_seconds=CRYPTO_FILL_WAIT_SECONDS)
            if _is_effectively_flat_quantity(app, confirmed_live_quantity):
                print(f'[{app.ticker}] Crypto target position is not confirmed yet. Skipping downstream risk-management orders...')
                app.logging.info(f'[{app.ticker}] Crypto target position is not confirmed yet. Skipping downstream risk-management orders...')
                return
            rm_quantity = abs(confirmed_live_quantity)
            rm_sign = float(np.sign(confirmed_live_quantity))
        app.risk_management_position_sign = rm_sign
        app.force_new_risk_management_prices = True
        try:
            sl_id = send_stop_loss_order(app, _next_order_id(app), rm_quantity)
            tp_id = send_take_profit_order(app, _next_bracket_order_id(app, sl_id), rm_quantity)
            if _is_crypto_asset(app) and not _crypto_native_stop_supported(app):
                if _defer_synthetic_monitors(app):
                    _queue_synthetic_crypto_monitor(app, tp_id)
                else:
                    _start_synthetic_crypto_monitor(app)
            _refresh_live_state_for_persistence(app, verbose=False)
        finally:
            app.force_new_risk_management_prices = False
            app.risk_management_position_sign = np.nan


def _reconcile_signed_target_position(app, order_id):
    target_signed_quantity = float(app.signal) * float(app.current_quantity)
    previous_quantity = float(app.previous_quantity)
    if _is_crypto_asset(app):
        previous_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
        app.previous_quantity = previous_quantity
    delta_quantity = target_signed_quantity - previous_quantity
    executable_delta_quantity = _normalize_order_quantity(app, abs(delta_quantity))
    effective_signed_quantity = target_signed_quantity

    if executable_delta_quantity > 0 and app.risk_management_bool:
        cancel_risk_management_previous_orders(app)

    if executable_delta_quantity > 0 and not np.isclose(delta_quantity, 0.0, atol=1e-8):
        if _is_crypto_asset(app):
            reconciled_quantity, market_sent = _reconcile_crypto_live_position(app, target_signed_quantity)
            effective_signed_quantity = reconciled_quantity
            if not market_sent:
                return
        else:
            original_signal = float(app.signal)
            app.signal = float(np.sign(delta_quantity))
            market_sent = send_market_order(app, _next_order_id(app), executable_delta_quantity)
            app.signal = original_signal
            if not market_sent:
                return
    elif executable_delta_quantity <= 0 and app.risk_management_bool:
        live_quantity = float(_latest_position_for_symbol(app))
        if not _is_effectively_flat_quantity(app, live_quantity) and np.sign(live_quantity) == np.sign(target_signed_quantity):
            effective_signed_quantity = live_quantity

    if app.risk_management_bool and not _is_effectively_flat_quantity(app, effective_signed_quantity):
        if executable_delta_quantity <= 0:
            cancel_risk_management_previous_orders(app)
        rm_quantity = abs(effective_signed_quantity)
        rm_sign = float(np.sign(effective_signed_quantity))
        if _is_crypto_asset(app):
            confirmed_live_quantity = _wait_for_live_position(app, expected_sign=rm_sign, timeout_seconds=CRYPTO_FILL_WAIT_SECONDS)
            if _is_effectively_flat_quantity(app, confirmed_live_quantity):
                print(f'[{app.ticker}] Crypto target position is not confirmed yet. Skipping downstream risk-management orders...')
                app.logging.info(f'[{app.ticker}] Crypto target position is not confirmed yet. Skipping downstream risk-management orders...')
                return
            rm_quantity = abs(confirmed_live_quantity)
            rm_sign = float(np.sign(confirmed_live_quantity))
        app.risk_management_position_sign = rm_sign
        app.force_new_risk_management_prices = True
        try:
            sl_id = send_stop_loss_order(app, _next_order_id(app), rm_quantity)
            tp_id = send_take_profit_order(app, _next_bracket_order_id(app, sl_id), rm_quantity)
            if _is_crypto_asset(app) and not _crypto_native_stop_supported(app):
                if _defer_synthetic_monitors(app):
                    _queue_synthetic_crypto_monitor(app, tp_id)
                else:
                    _start_synthetic_crypto_monitor(app)
            _refresh_live_state_for_persistence(app, verbose=False)
        finally:
            app.force_new_risk_management_prices = False
            app.risk_management_position_sign = np.nan


def set_new_and_rm_orders_quantities(app):
    
    # Set the signal
    signal = app.signal
    is_crypto = app.asset_spec.get("asset_class") == "crypto"
    
    # Set the new quantity for the current market order
    if app.previous_leverage != 0:
        new_quantity = (app.leverage - app.previous_leverage)*app.previous_quantity/app.previous_leverage
    else:
        new_quantity = app.current_quantity

    if new_quantity < 0:
        new_quantity = abs(new_quantity) if is_crypto else math.floor(abs(new_quantity))
        if app.previous_leverage != 0:
            rm_qty_val = app.previous_quantity - new_quantity
            rm_quantity = rm_qty_val if is_crypto else int(rm_qty_val)
        else:
            rm_quantity = None
        signal = -1.0
    elif new_quantity > 0:
        new_quantity = abs(new_quantity) if is_crypto else math.floor(abs(new_quantity))
        if app.previous_leverage != 0:
            rm_qty_val = app.previous_quantity + new_quantity
            rm_quantity = rm_qty_val if is_crypto else int(rm_qty_val)
        else:
            rm_quantity = None
    else:
        new_quantity = app.previous_quantity
        rm_quantity = None
    
    return signal, new_quantity, rm_quantity 
                
def send_orders(app):
    ''' Function to send the orders if needed'''

    print(f'[{app.ticker}] Sending the corresponding orders if needed...')
    app.logging.info(f'[{app.ticker}] Sending the corresponding orders if needed...')
    app.ordered_quantity = 0.0
    
    if len(app.cash_balance.loc[:, 'leverage'].index) != 0:
        prev_lev = pd.to_numeric(pd.Series([app.cash_balance['leverage'].iloc[-1]]), errors='coerce').iloc[0]
        prev_sig = pd.to_numeric(pd.Series([app.cash_balance['signal'].iloc[-1]]), errors='coerce').iloc[0]
        app.previous_leverage = float(prev_lev) if pd.notna(prev_lev) else 0.0
        app.previous_signal = float(prev_sig) if pd.notna(prev_sig) else 0.0
    else:
        app.previous_leverage = 0.0
        app.previous_signal = 0.0
        
    # In the portfolio engine, shared broker/account state is collected once
    # before entering the symbol loop. Only fall back to the legacy full refresh
    # when that shared context is not available.
    if _shared_pretrade_snapshot_active(app):
        update_risk_management_orders(app, verbose=True)
    else:
        update_trading_info(app)  

    if not _is_within_asset_trading_hours(app):
        print(f"[{app.ticker}] Outside this asset's trading hours. Skipping order submission for this cycle...")
        app.logging.info(f"[{app.ticker}] Outside this asset's trading hours. Skipping order submission for this cycle...")
        return

    # Update the previous and current positions quantities
    get_previous_and_current_quantities(app)
    app.use_shared_pretrade_snapshot = False

    if not np.isfinite(float(app.previous_signal)):
        app.previous_signal = _effective_position_sign(app, app.previous_quantity)
    if not np.isfinite(float(app.previous_leverage)):
        app.previous_leverage = 0.0

    if not app.isConnected():
        return

    order_id = _next_order_id(app)
    
    print('='*50)
    print('='*50)
    print(f'[{app.ticker}] previous quantity is {app.previous_quantity}')
    print(f'[{app.ticker}] previous signal is {app.previous_signal}')
    print(f'[{app.ticker}] signal is {app.signal}')
    print(f'[{app.ticker}] previous leverage is {app.previous_leverage}')
    print(f'[{app.ticker}] leverage is {app.leverage}')
    print(f'[{app.ticker}] current quantity is {app.signal*app.current_quantity}')
    if _is_integer_only_asset(app):
        print(f'[{app.ticker}] execution residual quantity is {getattr(app, "execution_residual_quantity", _get_execution_residual(app))}')
    print('='*50)
    print('='*50)

    if _is_integer_only_asset(app) and app.previous_quantity == 0 and app.current_quantity == 0 and app.signal != 0 and app.leverage != 0.0:
        print('Target quantity is below the tradable minimum. Residual quantity was carried forward for the next cycle...')
        app.logging.info('Target quantity is below the tradable minimum. Residual quantity was carried forward for the next cycle...')
        return

    target_signed_quantity = float(app.signal) * float(app.current_quantity)
    if _uses_direct_contract_target(app) and not np.isclose(float(app.previous_quantity), target_signed_quantity, atol=1e-8):
        _reconcile_direct_target_position(app, order_id)
        update_cash_balance_values_for_signals(app)
        update_trading_info(app, verbose=False)
        return

    # Reconcile all non-direct assets to the final signed target quantity.
    # This avoids the legacy leverage-delta branches sending market or
    # risk-management orders against the wrong size during reductions.
    if not _uses_direct_contract_target(app):
        if np.isclose(float(app.previous_quantity), target_signed_quantity, atol=1e-8):
            if app.signal != 0 and app.previous_quantity != 0 and app.risk_management_bool:
                cancel_risk_management_previous_orders(app)
                rm_quantity = abs(target_signed_quantity)
                rm_sign = float(np.sign(target_signed_quantity))
                if _is_crypto_asset(app):
                    live_quantity = float(_latest_position_for_symbol(app, refresh=True, verbose=False))
                    if np.isclose(live_quantity, 0.0, atol=1e-8):
                        print(f'[{app.ticker}] No confirmed crypto position is live. Skipping risk-management refresh...')
                        app.logging.info(f'[{app.ticker}] No confirmed crypto position is live. Skipping risk-management refresh...')
                        update_cash_balance_values_for_signals(app)
                        update_trading_info(app, verbose=False)
                        return
                    rm_quantity = abs(live_quantity)
                    rm_sign = float(np.sign(live_quantity))
                app.risk_management_position_sign = rm_sign
                app.force_new_risk_management_prices = True
                try:
                    sl_id = send_stop_loss_order(app, _next_order_id(app), rm_quantity)
                    tp_id = send_take_profit_order(app, _next_bracket_order_id(app, sl_id), rm_quantity)
                    if _is_crypto_asset(app) and not _crypto_native_stop_supported(app):
                        if _defer_synthetic_monitors(app):
                            _queue_synthetic_crypto_monitor(app, tp_id)
                        else:
                            _start_synthetic_crypto_monitor(app)
                    _refresh_live_state_for_persistence(app, verbose=False)
                finally:
                    app.force_new_risk_management_prices = False
                    app.risk_management_position_sign = np.nan
                print(f'[{app.ticker}] Current position already matches the target quantity. Only risk management orders were refreshed...')
                app.logging.info(f'[{app.ticker}] Current position already matches the target quantity. Only risk management orders were refreshed...')
            else:
                print(f'[{app.ticker}] Current position already matches the target quantity. No market order was needed...')
                app.logging.info(f'[{app.ticker}] Current position already matches the target quantity. No market order was needed...')

            update_cash_balance_values_for_signals(app)
            update_trading_info(app, verbose=False)
            return

        _reconcile_signed_target_position(app, order_id)
        update_cash_balance_values_for_signals(app)
        update_trading_info(app, verbose=False)
        return

    if np.isclose(float(app.previous_quantity), target_signed_quantity, atol=1e-8):
        if app.signal != 0 and app.previous_quantity != 0 and app.risk_management_bool:
            with ThreadPoolExecutor(2) as executor:
                futures = []
                futures.append(executor.submit(cancel_risk_management_previous_orders, app))
                futures.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, False, True, True))
                for future in futures:
                    future.result()
            print(f'[{app.ticker}] Current position already matches the target quantity. Only risk management orders were refreshed...')
            app.logging.info(f'[{app.ticker}] Current position already matches the target quantity. Only risk management orders were refreshed...')
        else:
            print(f'[{app.ticker}] Current position already matches the target quantity. No market order was needed...')
            app.logging.info(f'[{app.ticker}] Current position already matches the target quantity. No market order was needed...')

        update_cash_balance_values_for_signals(app)
        update_trading_info(app, verbose=False)
        return
        
    if (app.leverage == 0.0):
        if app.previous_quantity > 0:
            app.signal = -1.0

            if app.risk_management_bool:
                # Set the executors list
                executors_list = []
                # Append the functions to be used in parallel
                with ThreadPoolExecutor(2) as executor:
                    # Cancel the previous risk management orders
                    executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                    # Short-sell the asset and send the risk management orders
                    executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, True, False, False))
        
                # Run the functions in parallel
                for x in executors_list:
                    x.result()
                    
                print('The previous long position is closed and the risk management thresholds were closed if needed...')
                app.logging.info('We proceed to close the position...')
            else:
                send_orders_as_bracket(app, app, order_id, app.previous_quantity, True, False, False)
                print("Closed the long position...")
                app.logging.info("Closed the long position...")
            
            app.signal = 0.0
            
        elif app.previous_quantity < 0:
            app.signal = 1.0

            if app.risk_management_bool:
                # Set the executors list
                executors_list = []
                # Append the functions to be used in parallel
                with ThreadPoolExecutor(2) as executor:
                    # Cancel the previous risk management orders
                    executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                    # Short-sell the asset and send the risk management orders
                    executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, True, False, False))
        
                # Run the functions in parallel
                for x in executors_list:
                    x.result()
                    
                print('The previous long position is closed and the risk management thresholds were closed if needed...')
                app.logging.info('We proceed to close the position...')
            else:
                send_orders_as_bracket(app, app, order_id, app.previous_quantity, True, False, False)
                print("Closed the long position...")
                app.logging.info("Closed the long position...")
            
            app.signal = 0.0
            
        else:
            print('Leverage is 0.0. There will be no orders to send...')
            app.logging.info('Leverage is 0.0. There will be no orders to send...')

    elif app.previous_leverage == app.leverage:
        # If the previous position is short and the current signal is to go long
        if app.previous_quantity > 0 and app.signal > 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Send the new risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, False, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('Only the new risk management orders were sent...')
            app.logging.info('Only the new risk management orders were sent...')
            
        elif app.previous_quantity > 0 and app.signal < 0:
                
            new_qty_val = abs(app.previous_quantity) + app.current_quantity
            new_quantity = new_qty_val if app.asset_spec.get("asset_class") == "crypto" else int(new_qty_val)

            print(f'new quantity is {new_quantity}')
    
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Short-sell the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, new_quantity, True, True, True, app.current_quantity))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
                
            print('The market and the new risk management orders were sent...')
            app.logging.info('The market and the new risk management orders were sent...')
            
        elif app.previous_quantity < 0 and app.signal < 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Send the new risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, False, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('Only the new risk management orders were sent...')
            app.logging.info('Only the new risk management orders were sent...')
            
        elif app.previous_quantity < 0 and app.signal > 0:
                        
            new_qty_val = abs(app.previous_quantity) + app.current_quantity
            new_quantity = new_qty_val if app.asset_spec.get("asset_class") == "crypto" else int(new_qty_val)
    
            print(f'new quantity is {new_quantity}')
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Buy the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, new_quantity, True, True, True, app.current_quantity))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
                
            print('The market and the new risk management orders were sent...')
            app.logging.info('The market and the new risk management orders were sent...')
            
        elif app.previous_quantity != 0 and app.signal == 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Close the previous position
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, True, False, False))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A market order was sent to close the previous position...')
            app.logging.info('A market order was sent to close the previous position...')
            
        elif app.previous_quantity == 0 and app.signal != 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                qty = app.current_quantity if app.asset_spec.get("asset_class") == "crypto" else int(app.current_quantity)
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, qty, True, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A new position was just opened together with new risk management orders...')
            app.logging.info('A new position was just opened together with new risk management orders...')
        
        # Update the signal and leverage values in the cash balance dataframe
        update_cash_balance_values_for_signals(app)
            
        # Update the trading information
        update_trading_info(app, verbose=False)  
                    
    else:
        if app.previous_quantity > 0 and app.signal > 0:
            
            app.signal, new_quantity, rm_quantity = set_new_and_rm_orders_quantities(app) 
                   
            # Send the new risk management orders
            send_orders_as_bracket(app, order_id, int(new_quantity), True, True, True, rm_quantity)
            
            # Set the signal as per the net signal
            if app.signal < 0:
                app.signal = 1.0
                        
            print('The long position has been increased as per the increased leverage...')
            app.logging.info('The long position has been increased as per the increased leverage...')
            
        elif app.previous_quantity > 0 and app.signal < 0:
                
            new_qty_val = abs(app.previous_quantity) + app.current_quantity
            new_quantity = new_qty_val if app.asset_spec.get("asset_class") == "crypto" else int(new_qty_val)
    
            print(f'new quantity is {new_quantity}')
    
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Short-sell the asset and send the risk management orders
                rm_qty = app.current_quantity if app.asset_spec.get("asset_class") == "crypto" else int(app.current_quantity)
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, new_quantity, True, True, True, rm_qty))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
                
            print('The market and the new risk management orders were sent...')
            app.logging.info('The market and the new risk management orders were sent...')
            
        elif app.previous_quantity < 0 and app.signal < 0:
            
            app.signal, new_quantity, rm_quantity = set_new_and_rm_orders_quantities(app) 
                   
            # Send the new risk management orders
            send_orders_as_bracket(app, order_id, int(new_quantity), True, True, True, rm_quantity)
            
            # Set the signal as per the net signal
            if app.signal > 0:
                app.signal = -1.0
                            
            print('The long position has been increased as per the increased leverage...')
            app.logging.info('The long position has been increased as per the increased leverage...')
            
            
        elif app.previous_quantity < 0 and app.signal > 0:
                        
            new_qty_val = abs(app.previous_quantity) + app.current_quantity
            new_quantity = new_qty_val if app.asset_spec.get("asset_class") == "crypto" else int(new_qty_val)
    
            print(f'new quantity is {new_quantity}')
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Buy the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, new_quantity, True, True, True, int(app.current_quantity)))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
                
            print('The market and the new risk management orders were sent...')
            app.logging.info('The market and the new risk management orders were sent...')
            
        elif app.previous_quantity != 0 and app.signal == 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Close the previous position
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.previous_quantity, True, False, False))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A market order was sent to close the previous position...')
            app.logging.info('A market order was sent to close the previous position...')
            
        elif app.previous_quantity == 0 and app.signal != 0:
            
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                qty = app.current_quantity if app.asset_spec.get("asset_class") == "crypto" else int(app.current_quantity)
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, qty, True, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A new position was just opened together with new risk management orders...')
            app.logging.info('A new position was just opened together with new risk management orders...')

        # Update the signal and leverage values in the cash balance dataframe
        update_cash_balance_values_for_signals(app)
            
        # Update the trading information
        update_trading_info(app, verbose=False)  
                        
def strategy(app):
    ''' Function to get the strategy run'''

    print('Running the strategy for the period...')
    app.logging.info('Running the strategy for the period...')

    # Set a default dataframe
    base_df = pd.DataFrame()

    # Get the variables set in the main file (user_config/main.py)
    try:
        variables = tf.extract_variables(_main_config_path())
    except FileNotFoundError:
        app.logging.error("main.py (from user_config) not found. Cannot extract strategy variables.")
        print("Error: user_config/main.py not found. Ensure it's in the correct location.")
        # Potentially stop execution or use defaults if main.py is critical
        return # Or raise an error

    # The historical minute-frequency data address is constructed in engine.py and passed via app
    # historical_minute_data_address = f'data/app_{app.ticker}_df.csv' # This line is redundant here

    # Pass app attributes that might be needed by strategy.py functions
    # These will be merged/overridden by variables from main.py if names conflict,
    # or used if main.py doesn't define them but strategy functions need them.
    effective_vars = vars(app).copy()
    effective_vars.update(variables) # main.py variables override app attributes if names clash

    # Get the inputs of the prepare_base_df function from strategy.py
    try:
        signature_prepare_base = inspect.signature(stra.prepare_base_df)
        return_variables_prepare = tf.get_return_variable_names(get_strategy_file() or "strategy.py", "prepare_base_df")
    except FileNotFoundError:
        app.logging.error("Selected strategy file not found. Cannot inspect prepare_base_df.")
        print("Error: selected strategy file not found.")
        return
    except AttributeError: # If prepare_base_df is not in strategy.py
        app.logging.error("Function prepare_base_df not found in the selected strategy file.")
        print("Error: Function prepare_base_df not found in the selected strategy file.")
        return


    # Set a list for the function input parameters for prepare_base_df
    prepare_base_func_params = []
    for name, param in signature_prepare_base.parameters.items():
        if name in effective_vars:
            prepare_base_func_params.append(effective_vars[name])
        elif param.default is not inspect.Parameter.empty:
            prepare_base_func_params.append(param.default)
        else:
            err_msg = f"Parameter '{name}' for strategy.prepare_base_df not found in app attributes or main.py, and no default value."
            app.logging.error(err_msg)
            print(f"Error: {err_msg}")
            return

    # Determine the correct path for base_df_address (should be data/filename.csv)
    # app.base_df_address is set in trading_app.__init__ based on main.py
    # It should be like 'data/app_base_df.csv'

    current_base_df_path = app.base_df_address # This should be data/app_base_df.csv or similar

    if not os.path.exists(os.path.dirname(current_base_df_path)) and os.path.dirname(current_base_df_path) != '':
        os.makedirs(os.path.dirname(current_base_df_path))


    # If the base_df file exists
    if os.path.exists(current_base_df_path):
        try:
            base_df = pd.read_csv(current_base_df_path, index_col=0)
            base_df.index = pd.to_datetime(base_df.index)
        except Exception as e:
            app.logging.error(f"Error reading existing base_df from {current_base_df_path}: {e}")
            # Fallback to creating a new one if reading fails
            base_df = pd.DataFrame() # Ensure base_df is empty for the next block

        if base_df.empty or base_df.index[-1] < app.current_period: # If empty or outdated
            update_hist_data(app)
            if app.isConnected():
                # Logic for train_span for update (simplified for robustness)
                # Re-prepare using current params, function prepare_base_df should handle train_span internally if needed
                results_prepare = stra.prepare_base_df(*prepare_base_func_params)

                if 'base_df' not in return_variables_prepare:
                    err_msg = "'base_df' not found in return values of strategy.prepare_base_df. Check the selected strategy file."
                    app.logging.error(err_msg)
                    print(f"Error: {err_msg}")
                    return
                base_df_to_concat = results_prepare[return_variables_prepare.index('base_df')]
                if not isinstance(base_df_to_concat, pd.DataFrame):
                    app.logging.error("strategy.prepare_base_df did not return a DataFrame for 'base_df'.")
                    return

                if base_df.empty: # If it was empty due to read error or initial state
                    base_df = base_df_to_concat
                else: # Concatenate/update existing
                    base_df = pd.concat([base_df,base_df_to_concat])
                    base_df = base_df[~base_df.index.duplicated(keep='last')].sort_index()

                base_df.to_csv(current_base_df_path)
            else:
                app.logging.warning("Not connected to IB. Cannot update base_df.")
                return # Cannot proceed without connection for update
    else: # File does not exist, create it
        update_hist_data(app)
        if app.isConnected():
            results_prepare = stra.prepare_base_df(*prepare_base_func_params)
            if 'base_df' not in return_variables_prepare:
                err_msg = "'base_df' not found in return values of strategy.prepare_base_df. Check the selected strategy file."
                app.logging.error(err_msg)
                print(f"Error: {err_msg}")
                return

            base_df = results_prepare[return_variables_prepare.index('base_df')]
            if not isinstance(base_df, pd.DataFrame):
                app.logging.error("strategy.prepare_base_df did not return a DataFrame for 'base_df' when creating new.")
                return

            base_df.index = pd.to_datetime(base_df.index)
            base_df = base_df[~base_df.index.duplicated(keep='last')].sort_index()
            base_df.to_csv(current_base_df_path)
        else:
            app.logging.warning("Not connected to IB. Cannot create initial base_df.")
            return # Cannot proceed

    # Get the signal value for the current period
    if app.isConnected() and not base_df.empty:
        print('Getting the current signal...')
        app.logging.info('Getting the current signal...')
        app.base_df = base_df.copy() # Ensure app has the latest base_df

        try:
            signature_get_signal = inspect.signature(stra.get_signal)
            return_variables_signal = tf.get_return_variable_names(get_strategy_file() or "strategy.py", "get_signal")
        except FileNotFoundError:
             app.logging.error("Selected strategy file not found. Cannot inspect get_signal.")
             return
        except AttributeError:
             app.logging.error("Function get_signal not found in the selected strategy file.")
             return


        get_signal_func_params = []
        for name, param in signature_get_signal.parameters.items():
            if name == 'app': # Special case for 'app' object itself
                 get_signal_func_params.append(app)
            elif name in effective_vars:
                 get_signal_func_params.append(effective_vars[name])
            elif param.default is not inspect.Parameter.empty:
                 get_signal_func_params.append(param.default)
            else:
                err_msg = f"Parameter '{name}' for strategy.get_signal not found or no default."
                app.logging.error(err_msg)
                print(f"Error: {err_msg}")
                return

        results_signal = stra.get_signal(*get_signal_func_params)

        decisions = {}
        state_updates = {}
        targets = None
        if isinstance(results_signal, dict):
            decisions = results_signal
            targets = decisions.get('targets')
            state_updates = decisions.get('state_updates', {})
        elif 'targets' in return_variables_signal:
            targets = results_signal[return_variables_signal.index('targets')]
            if 'state_updates' in return_variables_signal:
                state_updates = results_signal[return_variables_signal.index('state_updates')]
        if targets is not None:
            if not isinstance(targets, dict):
                err_msg = "strategy.get_signal returned 'targets' but it is not a dictionary."
                app.logging.error(err_msg)
                print(f"Error: {err_msg}")
                return

            target_symbols = [str(k).upper() for k in targets.keys()]
            allowed = set([str(s).upper() for s in getattr(app, "allowed_symbols", [])])
            if getattr(app, "strict_targets_validation", True):
                unknown = sorted(set(target_symbols) - allowed)
                if len(unknown) > 0:
                    err_msg = f"strategy.get_signal returned unknown symbols not declared in main.py universe: {unknown}"
                    app.logging.error(err_msg)
                    print(f"Error: {err_msg}")
                    return

            symbol_key = str(app.ticker).upper()
            target = targets.get(symbol_key, {"signal": 0})
            app.signal = int(np.sign(float(target.get("signal", 0.0))))
        else:
            if isinstance(results_signal, dict):
                app.signal = int(np.sign(float(results_signal.get('signal', 0.0))))
            else:
                if 'signal' not in return_variables_signal:
                     err_msg = "'signal' or 'targets' not found in return values of strategy.get_signal. Check the selected strategy file."
                     app.logging.error(err_msg)
                     print(f"Error: {err_msg}")
                     return
                app.signal = results_signal[return_variables_signal.index('signal')]

                if app.leverage is None and 'leverage' in effective_vars and effective_vars['leverage'] is not None:
                    app.leverage = effective_vars['leverage']
                elif app.leverage is None:
                    app.leverage = 1.0

        if isinstance(state_updates, dict) and len(state_updates) > 0:
            app.queue_strategy_state(state_updates)
        elif isinstance(getattr(app, 'strategy_state_updates', {}), dict) and len(getattr(app, 'strategy_state_updates', {})) > 0:
            app.queue_strategy_state(app.strategy_state_updates)
            app.strategy_state_updates = {}

        print('The current signal was successfully created...')
        app.logging.info('The current signal was successfully created...')
    elif base_df.empty:
        app.logging.error("base_df is empty. Cannot get signal.")
        return
    else: # Not connected
        app.logging.warning("Not connected to IB. Cannot get signal.")
        return

    print('The strategy for the period was successfully run...')
    app.logging.info('The strategy for the period was successfully run...')
    
def _strategy_context(app):
    variables = tf.extract_variables(_main_config_path())
    effective_vars = vars(app).copy()
    effective_vars.update(variables)
    return variables, effective_vars


def _call_strategy_callable(func, app, effective_vars):
    params = []
    signature = inspect.signature(func)
    for name, param in signature.parameters.items():
        if name == 'app':
            params.append(app)
        elif name in effective_vars:
            params.append(effective_vars[name])
        elif param.default is not inspect.Parameter.empty:
            params.append(param.default)
        else:
            raise ValueError(f"Parameter '{name}' for strategy function '{func.__name__}' is missing.")
    return func(*params)


def _coerce_base_df_result(result):
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    raise ValueError('strategy.prepare_base_df did not return a pandas DataFrame.')


def refresh_symbol_market_data(app):
    print(f'[{app.ticker}] Refreshing market data...')
    app.logging.info(f'[{app.ticker}] Refreshing market data...')
    update_hist_data(app)
    if not app.isConnected():
        raise RuntimeError(f'{app.ticker} disconnected while refreshing historical data.')

    _, effective_vars = _strategy_context(app)
    base_df = _coerce_base_df_result(_call_strategy_callable(stra.prepare_base_df, app, effective_vars))
    base_df.index = pd.to_datetime(base_df.index)
    base_df = base_df[~base_df.index.duplicated(keep='last')].sort_index()
    app.base_df = base_df.copy()
    os.makedirs(os.path.dirname(app.base_df_address) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(app.historical_data_address) or '.', exist_ok=True)
    base_df.to_csv(app.base_df_address)
    app.historical_data.to_csv(app.historical_data_address)
    return app



def collect_shared_account_snapshot(apps):
    if len(apps) == 0:
        return
    lead_app = apps[0]
    update_capital(lead_app)
    lead_app.shared_unlevered_capital = float(getattr(lead_app, 'capital', np.nan))
    shared_acc_update = lead_app.acc_update.copy(deep=True)
    shared_cash_balance = lead_app.cash_balance.copy(deep=True)
    shared_portfolio_snapshots = lead_app.portfolio_snapshots_df.copy(deep=True)
    for app in apps:
        app.shared_unlevered_capital = float(getattr(lead_app, 'shared_unlevered_capital', np.nan))
        app.shared_acc_update = shared_acc_update.copy(deep=True)
        app.shared_cash_balance = shared_cash_balance.copy(deep=True)
        app.shared_portfolio_snapshots_df = shared_portfolio_snapshots.copy(deep=True)


def collect_shared_broker_snapshot(app):
    if app is None or not app.isConnected():
        return False

    previous_use_shared = bool(getattr(app, 'use_shared_pretrade_snapshot', False))
    previous_force_pull = bool(getattr(app, 'force_shared_broker_pull', False))
    app.use_shared_pretrade_snapshot = False
    app.force_shared_broker_pull = True
    try:
        request_positions(app, verbose=False)
        request_orders(app, verbose=False)
        update_submitted_orders(app, verbose=False)
    finally:
        app.use_shared_pretrade_snapshot = previous_use_shared
        app.force_shared_broker_pull = previous_force_pull

    app.shared_broker_snapshot_ready = True
    app.shared_broker_snapshot_period = getattr(app, 'current_period', None)
    app.shared_broker_state = {
        'period': getattr(app, 'current_period', None),
        'positions': getattr(app, 'pos_df', pd.DataFrame()).copy(deep=True),
        'open_orders': getattr(app, 'open_orders', pd.DataFrame()).copy(deep=True),
        'orders_status': getattr(app, 'orders_status', pd.DataFrame()).copy(deep=True),
        'executions': getattr(app, 'exec_df', pd.DataFrame()).copy(deep=True),
        'commissions': getattr(app, 'comm_df', pd.DataFrame()).copy(deep=True),
        'current_open_orders_snapshot': getattr(app, 'current_open_orders_snapshot', pd.DataFrame()).copy(deep=True),
    }
    append_runtime_audit(app, 'shared_broker_snapshot', f'period={getattr(app, "current_period", None)}')
    return True


def collect_shared_contract_details(app, symbols):
    if app is None or not app.isConnected():
        return False
    original_asset_spec = deepcopy(getattr(app, 'asset_spec', {}) or {})
    original_ticker = getattr(app, 'ticker', '')
    original_contract = deepcopy(getattr(app, 'contract', None))
    original_resolved_contract = deepcopy(getattr(app, 'resolved_contract', None))
    for symbol_spec in symbols:
        app.asset_spec = symbol_spec
        app.ticker = str(symbol_spec.get('symbol', '')).upper()
        app.contract = ibf.build_contract_from_spec(symbol_spec)
        request_contract_details(app, verbose=False)
    app.asset_spec = original_asset_spec
    app.ticker = original_ticker
    app.contract = original_contract
    app.resolved_contract = original_resolved_contract
    return True


def compute_portfolio_targets_once(apps):
    if len(apps) == 0:
        return {}

    lead_app = apps[0]
    _, effective_vars = _strategy_context(lead_app)
    results_signal = _call_strategy_callable(stra.get_signal, lead_app, effective_vars)
    if not isinstance(results_signal, dict):
        raise ValueError('strategy.get_signal must return a dictionary in the multi-asset setup.')

    targets = results_signal.get('targets')
    if not isinstance(targets, dict):
        raise ValueError("strategy.get_signal must return a 'targets' dictionary in the multi-asset setup.")

    allowed = {str(s).upper() for s in getattr(lead_app, 'allowed_symbols', [])}
    unknown = sorted(set(str(k).upper() for k in targets.keys()) - allowed)
    if getattr(lead_app, 'strict_targets_validation', True) and len(unknown) > 0:
        raise ValueError(f'strategy.get_signal returned unknown symbols not declared in main.py universe: {unknown}')

    shared_targets = getattr(lead_app, 'strategy_targets', {}) or {}
    shared_state_updates = results_signal.get('state_updates', {})
    if isinstance(getattr(lead_app, 'strategy_state_updates', {}), dict) and len(getattr(lead_app, 'strategy_state_updates', {})) > 0:
        shared_state_updates = lead_app.strategy_state_updates

    for app in apps:
        symbol_key = str(app.ticker).upper()
        target = targets.get(symbol_key, {'signal': 0})
        app.signal = int(np.sign(float(target.get('signal', 0.0))))
        app.strategy_targets = shared_targets
        if hasattr(lead_app, 'target_weights'):
            app.target_weights = getattr(lead_app, 'target_weights', {})
            app.applied_target_weights = getattr(lead_app, 'applied_target_weights', {})
            app.margin_scale = getattr(lead_app, 'margin_scale', 1.0)
            app.required_capital_frac = getattr(lead_app, 'required_capital_frac', 0.0)
            app.used_capital_frac = getattr(lead_app, 'used_capital_frac', 0.0)
            app.cash_weight = getattr(lead_app, 'cash_weight', 0.0)
        if app is lead_app and isinstance(shared_state_updates, dict) and len(shared_state_updates) > 0:
            app.queue_strategy_state(shared_state_updates, symbol='PORTFOLIO')

    return targets


def finalize_portfolio_cycle_app(app):
    app.strategy_end = True


def _build_portfolio_app_time_spent_row(apps):
    if len(apps) == 0:
        return pd.DataFrame(columns=WORKBOOK_SCHEMAS['app_time_spent'])
    started_at = min(getattr(app, 'app_start_time', dt.datetime.now()) for app in apps)
    completed_at = dt.datetime.now().replace(microsecond=0)
    sample_app = apps[0]
    return pd.DataFrame([{
        'datetime': completed_at,
        'seconds': float((completed_at - started_at).total_seconds()),
        'market_open_time': getattr(sample_app, 'market_open_time', pd.NaT),
        'market_close_time': getattr(sample_app, 'market_close_time', pd.NaT),
    }])


def _build_portfolio_period_row(apps):
    if len(apps) == 0:
        return pd.DataFrame(columns=WORKBOOK_SCHEMAS['periods_traded'])
    sample_app = apps[0]
    return pd.DataFrame([{
        'datetime': dt.datetime.now().replace(microsecond=0),
        'trade_time': getattr(sample_app, 'current_period', pd.NaT),
        'trade_done': 1,
        'market_open_time': getattr(sample_app, 'market_open_time', pd.NaT),
        'market_close_time': getattr(sample_app, 'market_close_time', pd.NaT),
    }])



def _merge_unique_frames(frames, sort_index=False, sort_columns=None, dedupe_subset=None):
    valid = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        local = _frame_with_datetime_column(frame)
        if local.empty:
            continue
        valid.append(local)
    if not valid:
        return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True)
    requested_dedupe_subset = [col for col in (dedupe_subset or []) if col in merged.columns]
    if dedupe_subset is None:
        merged = merged.drop_duplicates()
    elif requested_dedupe_subset:
        merged = merged.drop_duplicates(subset=requested_dedupe_subset, keep='last')
    else:
        merged = merged.drop_duplicates()
    if sort_columns is not None and all(col in merged.columns for col in sort_columns):
        merged = merged.sort_values(sort_columns)
    elif sort_index:
        if 'datetime' in merged.columns:
            merged = merged.sort_values('datetime')
        else:
            merged = merged.sort_index()
    if 'datetime' in merged.columns:
        merged['datetime'] = pd.to_datetime(merged['datetime'], errors='coerce')
    return merged



def save_portfolio_cycle_data(apps, send_email_summary=True):
    if len(apps) == 0:
        return
    _finalize_apps_for_persistence(apps)
    lead_app = apps[0]

    for app in apps:
        _flush_live_trading_buffers(app)
        _flush_temp_sheet(app, 'temp_portfolio_snapshots', 'portfolio_snapshots_df', 'portfolio_snapshots', dedupe_subset=['datetime', 'Account', 'Symbol', 'ConId'])

    lead_app.open_orders = _merge_unique_frames([app.open_orders for app in apps], sort_index=True)
    lead_app.orders_status = _merge_unique_frames([app.orders_status for app in apps], sort_index=True)
    lead_app.exec_df = _merge_unique_frames([app.exec_df for app in apps], sort_index=True)
    lead_app.comm_df = _merge_unique_frames([app.comm_df for app in apps], sort_index=True)
    lead_app.pos_df = _merge_unique_frames([app.pos_df for app in apps], sort_index=True)
    lead_app.portfolio_snapshots_df = _merge_unique_frames([app.portfolio_snapshots_df for app in apps], sort_index=True, dedupe_subset=['datetime', 'Account', 'Symbol', 'ConId'] if all(hasattr(app, 'portfolio_snapshots_df') for app in apps) else None)
    lead_app.cash_balance = _merge_unique_frames([app.cash_balance for app in apps], sort_index=True)
    lead_app.account_updates_df = _merge_unique_frames([app.account_updates_df for app in apps], sort_index=True)
    lead_app.contract_details_df = _merge_unique_frames(
        [app.contract_details_df for app in apps],
        sort_columns=['Symbol', 'datetime'] if all(col in app.contract_details_df.columns for app in apps for col in ['Symbol', 'datetime']) else (['Symbol'] if all('Symbol' in app.contract_details_df.columns for app in apps) else None),
        dedupe_subset=['Symbol', 'ConId', 'LocalSymbol', 'Exchange'] if all(col in lead_app.contract_details_df.columns for col in ['Symbol', 'ConId', 'LocalSymbol', 'Exchange']) else (['Symbol', 'ConId'] if all(col in lead_app.contract_details_df.columns for col in ['Symbol', 'ConId']) else None),
    )
    lead_app.strategy_state_df = _merge_unique_frames([app.strategy_state_df for app in apps], sort_index=True, dedupe_subset=['Symbol', 'Scope', 'StateKey'] if all(col in lead_app.strategy_state_df.columns for col in ['Symbol', 'Scope', 'StateKey']) else None)
    existing_time_spent = _frame_with_datetime_column(getattr(lead_app, 'app_time_spent_all', pd.DataFrame()))
    per_app_time_spent = [_frame_with_datetime_column(getattr(app, 'app_time_spent_all', getattr(app, 'app_time_spent', pd.DataFrame()))) for app in apps]
    current_time_spent = _build_portfolio_app_time_spent_row(apps)
    lead_app.app_time_spent_all = _merge_unique_frames(
        [existing_time_spent, *per_app_time_spent, current_time_spent],
        sort_columns=['datetime'] if 'datetime' in current_time_spent.columns else None,
        dedupe_subset=['datetime'] if 'datetime' in current_time_spent.columns else None,
    )

    existing_periods = _frame_with_datetime_column(getattr(lead_app, 'periods_traded_all', pd.DataFrame()))
    per_app_periods = [_frame_with_datetime_column(getattr(app, 'periods_traded_all', getattr(app, 'periods_traded', pd.DataFrame()))) for app in apps]
    current_period_row = _build_portfolio_period_row(apps)
    lead_app.periods_traded_all = _merge_unique_frames(
        [existing_periods, *per_app_periods, current_period_row],
        sort_columns=['trade_time'] if 'trade_time' in current_period_row.columns else None,
        dedupe_subset=['trade_time'] if 'trade_time' in current_period_row.columns else None,
    )
    lead_app.app_time_spent = lead_app.app_time_spent_all.copy()
    lead_app.periods_traded = lead_app.periods_traded_all.copy()

    save_data(lead_app)
    for app in apps:
        app.historical_data.to_csv(app.historical_data_address)
        if hasattr(app, 'base_df') and isinstance(app.base_df, pd.DataFrame):
            app.base_df.to_csv(app.base_df_address)
    if send_email_summary:
        send_email(lead_app)



def print_portfolio_order_summary(apps):
    headers = ['Asset', 'Signal', 'Leverage', 'OrderedQty']
    rows = []
    for app in apps:
        rows.append([
            str(app.ticker),
            str(int(getattr(app, 'signal', 0))),
            f"{float(getattr(app, 'leverage', 0.0)):.6f}",
            f"{float(getattr(app, 'ordered_quantity', 0.0)):.6f}",
        ])
    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))]
    line = ' | '.join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = '-+-'.join('-' * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(' | '.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def run_portfolio_cycle_for_the_period(apps):
    active_apps = [app for app in apps if app.isConnected()]
    if len(active_apps) == 0:
        return

    print('Refreshing all symbols before portfolio decisioning...')
    crypto_apps = [app for app in active_apps if str(app.asset_spec.get("asset_class", "forex")).lower() == "crypto"]
    non_crypto_apps = [app for app in active_apps if app not in crypto_apps]

    if non_crypto_apps:
        with ThreadPoolExecutor(len(non_crypto_apps)) as executor:
            futures = [executor.submit(refresh_symbol_market_data, app) for app in non_crypto_apps]
            for future in futures:
                future.result()

    if crypto_apps:
        with ThreadPoolExecutor(len(crypto_apps)) as executor:
            futures = [executor.submit(refresh_symbol_market_data, app) for app in crypto_apps]
            for future in futures:
                future.result()

    print('Collecting shared account updates once for the full universe...')
    collect_shared_account_snapshot(active_apps)

    print('Computing portfolio targets once for the full universe...')
    compute_portfolio_targets_once(active_apps)

    print('Preparing and sending orders across all symbols...')
    cycle_errors = []
    with ThreadPoolExecutor(len(active_apps)) as executor:
        futures = {executor.submit(send_orders, app): app for app in active_apps}
        for future, app in futures.items():
            try:
                future.result()
            except Exception as exc:
                msg = f'[{app.ticker}] Order preparation failed: {exc}'
                print(msg)
                app.logging.exception(msg)
                append_runtime_audit(app, 'send_orders_failed', str(exc))
                cycle_errors.append(msg)

    print('Portfolio order summary...')
    print_portfolio_order_summary(active_apps)

    if len(cycle_errors) > 0:
        print('Portfolio cycle warnings:')
        for msg in cycle_errors:
            print(msg)

    for app in active_apps:
        finalize_portfolio_cycle_app(app)

    print('Saving the data and sending the email...')
    save_portfolio_cycle_data(active_apps, send_email_summary=True)

    for app in active_apps:
        stop(app)

def save_week_open_and_close_datetimes(app, verbose=True):
    """ Function to fill all the dataframes with the week's open and close datetimes"""
    
    if verbose:
        print("Saving the corresponding week's open and close datetimes in the corresponding dataframes...")
        app.logging.info("Saving the corresponding week's open and close datetimes in the corresponding dataframes...")
    
    # A for loop to iterate through each of the corresponding dataframes
    for dataframe in [app.open_orders, app.orders_status, app.exec_df, app.comm_df, \
                      app.pos_df, app.portfolio_snapshots_df, app.cash_balance]:
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            continue
        if dataframe.index is None or len(dataframe.index) == 0:
            continue
        index_datetimes = pd.to_datetime(dataframe.index, errors='coerce')
        if not getattr(index_datetimes, 'notna', lambda: pd.Series(dtype=bool))().any():
            continue
        # Get the rows which correspond to the week's datetimes
        mask = (index_datetimes >= app.market_open_time) & (index_datetimes <= app.market_close_time)
        if not np.any(mask):
            continue
        # Set the corresponding market open time in each dataframe
        dataframe.loc[mask,'market_open_time'] = app.market_open_time
        # Set the corresponding market close time in each dataframe
        dataframe.loc[mask,'market_close_time'] = app.market_close_time
        # Drop duplicates in place without reindex warnings
        dataframe.drop_duplicates(inplace=True)
        
    # Get the rows which correspond to the week's datetimes in the periods_traded dataframe
    if isinstance(app.periods_traded, pd.DataFrame) and not app.periods_traded.empty and 'trade_time' in app.periods_traded.columns:
        mask = (app.periods_traded['trade_time']>=app.market_open_time) & (app.periods_traded['trade_time']<=app.market_close_time)
        if np.any(mask):
            # Set the corresponding market open time in the dataframe
            app.periods_traded.loc[mask,'market_open_time'] = app.market_open_time
            # Set the corresponding market close time in the dataframe
            app.periods_traded.loc[mask,'market_close_time'] = app.market_close_time

    if verbose:
        print("The corresponding week's open and close datetimes were successfully added on the dataframes...")
        app.logging.info("The corresponding week's open and close datetimes were successfully added on the dataframes...")


def _build_commissions_export(app):
    commissions = _frame_with_datetime_column(getattr(app, 'comm_df', pd.DataFrame()))
    executions = _frame_with_datetime_column(getattr(app, 'exec_df', pd.DataFrame()))

    if commissions.empty and executions.empty:
        return commissions

    shell = pd.DataFrame(columns=WORKBOOK_SCHEMAS['commissions'])
    if not executions.empty and 'ExecutionId' in executions.columns:
        shell = pd.DataFrame({
            'datetime': _safe_datetime_series(executions.get('datetime', pd.Series(dtype='object'))),
            'ExecutionId': executions.get('ExecutionId', pd.Series(dtype='object')),
            'Commission': np.nan,
            'Currency': executions.get('Currency', pd.Series(dtype='object')),
            'Realized PnL': np.nan,
        })

    if not commissions.empty:
        commissions = commissions.copy()
        if 'datetime' in commissions.columns:
            commissions['datetime'] = _safe_datetime_series(commissions['datetime'])

    combined = pd.concat([shell, commissions], ignore_index=True) if not commissions.empty else shell
    if combined.empty:
        return combined
    if 'datetime' in combined.columns:
        combined['datetime'] = _safe_datetime_series(combined['datetime'])
    if 'ExecutionId' in combined.columns:
        combined = combined.sort_values('datetime', kind='stable', na_position='last')
        combined = combined.drop_duplicates(subset=['ExecutionId'], keep='last')
    return combined
    
def save_data(app, verbose=True):
    """ Function to save the data"""

    if verbose:
        print("Saving all the data...")
        app.logging.info("Saving all the data...")

    _flush_live_trading_buffers(app)
    _flush_contract_details_buffer(app)
    app.open_orders = _normalize_sheet_to_datetime_index(getattr(app, 'open_orders', pd.DataFrame()))
    app.orders_status = _normalize_sheet_to_datetime_index(getattr(app, 'orders_status', pd.DataFrame()))
    app.exec_df = _normalize_sheet_to_datetime_index(getattr(app, 'exec_df', pd.DataFrame()))
    app.comm_df = _normalize_sheet_to_datetime_index(getattr(app, 'comm_df', pd.DataFrame()))
    app.pos_df = _normalize_sheet_to_datetime_index(getattr(app, 'pos_df', pd.DataFrame()))
    app.portfolio_snapshots_df = _normalize_sheet_to_datetime_index(getattr(app, 'portfolio_snapshots_df', pd.DataFrame()))
    if app.portfolio_snapshots_df.empty and not app.pos_df.empty:
        app.portfolio_snapshots_df = _portfolio_snapshots_from_positions(app.pos_df)
    app.cash_balance = _normalize_sheet_to_datetime_index(getattr(app, 'cash_balance', pd.DataFrame()))
    _flush_runtime_audit_buffer(app)
    _flush_strategy_state_buffer(app)
    app.cash_balance.ffill(inplace=True)
    app.app_time_spent_all = _frame_with_datetime_column(getattr(app, 'app_time_spent_all', getattr(app, 'app_time_spent', pd.DataFrame())))
    if not app.app_time_spent_all.empty:
        app.app_time_spent_all = app.app_time_spent_all.reset_index(drop=True)
        if 'datetime' not in app.app_time_spent_all.columns:
            app.app_time_spent_all['datetime'] = pd.NaT
        app.app_time_spent_all.sort_values(['datetime'], inplace=True, ignore_index=True)
        app.app_time_spent_all.drop_duplicates(subset=['datetime'], keep='last', inplace=True)

    app.periods_traded_all = _frame_with_datetime_column(getattr(app, 'periods_traded_all', getattr(app, 'periods_traded', pd.DataFrame())))
    if not app.periods_traded_all.empty:
        app.periods_traded_all = app.periods_traded_all.reset_index(drop=True)
        if 'datetime' not in app.periods_traded_all.columns:
            app.periods_traded_all['datetime'] = pd.NaT
        app.periods_traded_all['trade_time'] = pd.to_datetime(app.periods_traded_all['trade_time'], errors='coerce')
        app.periods_traded_all.sort_values(['trade_time'], inplace=True, ignore_index=True)
        app.periods_traded_all.drop_duplicates(subset=['trade_time'], keep='last', inplace=True)

    app.account_updates_df = _frame_with_datetime_column(getattr(app, 'account_updates_df', pd.DataFrame()))
    if not app.account_updates_df.empty and 'datetime' in app.account_updates_df.columns:
        app.account_updates_df['datetime'] = pd.to_datetime(app.account_updates_df['datetime'], errors='coerce')
        app.account_updates_df = app.account_updates_df.dropna(subset=['datetime'])
        app.account_updates_df = app.account_updates_df.reset_index(drop=True).set_index('datetime')
        app.account_updates_df.index.name = ''

    if not app.acc_update.empty:
        acc_updates = app.acc_update.copy()
        acc_updates['datetime'] = pd.to_datetime(acc_updates['datetime'], errors='coerce')
        acc_updates = acc_updates.dropna(subset=['datetime'])
        acc_updates.set_index('datetime', inplace=True)
        acc_updates.index.name = ''
        app.account_updates_df = pd.concat([app.account_updates_df, acc_updates])
        app.account_updates_df = app.account_updates_df[~app.account_updates_df.index.isna()]
        app.account_updates_df.drop_duplicates(inplace=True)
        app.account_updates_df.sort_index(ascending=True, inplace=True)

    _flush_temp_sheet(app, 'temp_portfolio_snapshots', 'portfolio_snapshots_df', 'portfolio_snapshots', dedupe_subset=['datetime', 'Account', 'Symbol', 'ConId'])
    app.portfolio_snapshots_df = _normalize_sheet_to_datetime_index(getattr(app, 'portfolio_snapshots_df', pd.DataFrame()), invalid_before_year=2000)

    _flush_runtime_audit_buffer(app)
    _flush_strategy_state_buffer(app)
    save_week_open_and_close_datetimes(app, verbose=verbose)
    _save_strategy_state(app.database_path, getattr(app, 'strategy_state_df', pd.DataFrame()))

    cash_balance_export = _frame_with_datetime_column(app.cash_balance)
    if not cash_balance_export.empty:
        keep_cols = [col for col in ['datetime', 'value', 'market_open_time', 'market_close_time'] if col in cash_balance_export.columns]
        cash_balance_export = cash_balance_export.loc[:, keep_cols]

    app_time_spent_export = _frame_with_datetime_column(app.app_time_spent_all)
    if not app_time_spent_export.empty:
        keep_cols = [col for col in WORKBOOK_SCHEMAS['app_time_spent'] if col in app_time_spent_export.columns]
        app_time_spent_export = app_time_spent_export.loc[:, keep_cols]

    periods_traded_export = _frame_with_datetime_column(app.periods_traded_all)
    if not periods_traded_export.empty:
        keep_cols = [col for col in WORKBOOK_SCHEMAS['periods_traded'] if col in periods_traded_export.columns]
        periods_traded_export = periods_traded_export.loc[:, keep_cols]

    account_updates_export = _frame_with_datetime_column(app.account_updates_df)
    if not account_updates_export.empty:
        keep_cols = [col for col in WORKBOOK_SCHEMAS['account_updates'] if col in account_updates_export.columns]
        account_updates_export = account_updates_export.loc[:, keep_cols]

    commissions_export = _build_commissions_export(app)
    if not commissions_export.empty and 'datetime' in commissions_export.columns:
        commissions_export = commissions_export.reset_index(drop=True)
        commissions_export = commissions_export.set_index('datetime')
        commissions_export.index.name = ''
        commissions_export.loc[:, 'market_open_time'] = getattr(app, 'market_open_time', pd.NaT)
        commissions_export.loc[:, 'market_close_time'] = getattr(app, 'market_close_time', pd.NaT)

    dictfiles = {'open_orders': app.open_orders,
                 'orders_status': app.orders_status,
                 'executions': app.exec_df,
                 'commissions': commissions_export,
                 'positions': app.pos_df,
                 'portfolio_snapshots': app.portfolio_snapshots_df,
                 'cash_balance': cash_balance_export,
                 'app_time_spent': app_time_spent_export,
                 'periods_traded': periods_traded_export,
                 'account_updates': account_updates_export,
                 'contract_details': app.contract_details_df}

    tf.save_xlsx(dict_df=dictfiles, path=app.database_path)
    app.historical_data.to_csv(app.historical_data_address)

    app.acc_update = pd.DataFrame(columns=app.acc_update.columns)
    if verbose:
        print("All data saved...")
        app.logging.info("All data saved...")

def save_data_and_send_email(app):
    """ Function to save the data and send email"""
    
    print("Saving the data and sending the email...")
    app.logging.info("Saving the data and sending the email...")
    
    # If the app is connected
    if app.isConnected():
        save_data(app)
        try:
            generate_live_portfolio_report(app)
        except Exception as exc:
            app.logging.error("Failed to generate portfolio PDF before email: %s", exc)
        send_email(app)

        print("The data was saved successfully...")
        app.logging.info("The data was saved successfully...")
    
def run_strategy(app):
    """ Function to run the whole strategy, including the signal"""

    print("Running the strategy, the signal and sending the orders if necessary...")
    app.logging.info("Running the strategy, the signal and sending the orders if necessary...")
    append_runtime_audit(app, 'run_strategy_start', f'period={app.current_period}')
    
    # Run the strategy
    strategy(app)
    
    # If the app is connected
    if app.isConnected():
        # Send the orders
        send_orders(app)
        print("The strategy, the signal and sending the orders were successfully run...")
        app.logging.info("The strategy, the signal and sending the orders were successfully run...")
    
        
    # Save the total seconds spent while trading in each period
    app.app_time_spent.loc[app.app_time_spent.index[-1], 'seconds'] = (dt.datetime.now() - app.app_start_time).total_seconds() + 3
    
    # Set the current period as traded
    app.periods_traded.loc[app.periods_traded.index[-1], 'trade_done'] = 1
    
    save_data_and_send_email(app)

    append_runtime_audit(app, 'run_strategy_end', f'signal={getattr(app, 'signal', 'na')}')

    # Tell the app the strategy is done so it can be disconnected       
    app.strategy_end = True
    
def run_strategy_for_the_period(app):
    """ Function to run the whole strategy together with the connection monitor function"""

    # Run the strategy        
    run_strategy(app)
    # app.connection_monitor()
        
    # Disconnect the app
    stop(app)
    
    print("Let's wait for the next period to trade...")
    app.logging.info("Let's wait for the next period to trade...")

def wait_for_next_period(app): 
    """ Function to wait for the next period"""
    
    print("Let's wait for the next period to trade...")
    app.logging.info("Let's wait for the next period to trade...")
    
    # Disconnect the app
    stop(app)
                
    # Wait until we arrive at the next trading period
    time.sleep(0 if (app.next_period-dt.datetime.now()).total_seconds()<0 else (app.next_period-dt.datetime.now()).total_seconds())

def update_and_close_positions(app):
    """ Function to update and close the current position before the day closes"""

    app.logging.info('[%s] Update the trading info and close the position...', app.ticker)
    
    # Update the trading info        
    update_trading_info(app)  
    
    # Cancel the previous risk management orders
    cancel_risk_management_previous_orders(app)                        

    # Signal and leverage are zero at the end of the day
    app.signal = app.leverage = 0
    
    # Get the previous and current quantities
    get_previous_and_current_quantities(app)
    had_position = float(getattr(app, 'previous_quantity', 0.0)) != 0.0
    
    # If the app is connected
    if app.isConnected():
        order_id = _next_order_id(app)
    
    # If the app is connected
    if app.isConnected():
        # If a position exists
        if app.previous_quantity != 0.0:
            # Send a market order
            send_market_order(app, order_id, app.previous_quantity) 
    
    # Update the signal and leverage values in the cash balance dataframe
    update_cash_balance_values_for_signals(app)
    
    # If the app is connected
    if app.isConnected():
        # Update the trading info
        update_trading_info(app)  
    
    # Update the current equity value
    update_capital(app)
    
    # Update the current period trading status
    app.periods_traded.loc[app.periods_traded.index[-1], 'trade_done'] = 1

    # Save the data and send the email
    save_data_and_send_email(app)
    
    if had_position:
        print(f'[{app.ticker}] End-of-day closeout completed successfully...')
    app.logging.info('[%s] The trading info was updated and the position was closed successfully...', app.ticker)
    
    if (app.next_period != app.market_close_time):
        app.logging.info('[%s] Waiting for the next trading day to start...', app.ticker)
    else:
        app.logging.info('[%s] Waiting for the market to close...', app.ticker)
        
    # Disconnect the app
    stop(app)
    
    # Wait until we arrive at the next trading period
    time.sleep(0 if (app.next_period-dt.datetime.now()).total_seconds()<0 else (app.next_period-dt.datetime.now()).total_seconds())

def send_email(app): 
    """ Function to send an email with relevant information of the trading current period"""
    try:
        email_password = pd.read_excel(app.email_info_path, index_col=0)
        smtp_username = email_password["smtp_username"].iloc[0]
        smtp_password = email_password["password"].iloc[0]
        from_email = email_password["smtp_username"].iloc[0]
        to_email = email_password["to_email"].iloc[0]

        cash_value = np.nan
        if getattr(app, "cash_balance", pd.DataFrame()).empty is False and "value" in app.cash_balance.columns:
            cash_value = pd.to_numeric(app.cash_balance["value"], errors="coerce").dropna()
            cash_value = float(cash_value.iloc[-1]) if len(cash_value) else np.nan

        position_qty = getattr(app, "current_quantity", np.nan)
        symbol_name = getattr(getattr(app, "contract", None), "symbol", getattr(app, "ticker", "UNKNOWN"))

        market_order_price = np.nan
        sl_order_price = np.nan
        tp_order_price = np.nan

        if getattr(app, "open_orders", pd.DataFrame()).empty is False and getattr(app, "orders_status", pd.DataFrame()).empty is False:
            symbol_orders = app.open_orders[app.open_orders["Symbol"] == symbol_name].copy()
            if not symbol_orders.empty:
                try:
                    mkt_ids = symbol_orders[symbol_orders["OrderType"] == "MKT"]["OrderId"].sort_values()
                    if len(mkt_ids):
                        mkt_order_id = int(mkt_ids.iloc[-1])
                        filled = app.orders_status[
                            (app.orders_status["OrderId"] == mkt_order_id) &
                            (app.orders_status["Status"] == "Filled")
                        ]["AvgFillPrice"].sort_values()
                        if len(filled):
                            market_order_price = float(filled.iloc[-1])
                except Exception:
                    pass

                try:
                    stop_type = "TRAIL" if getattr(app, "trail", False) else "STP"
                    stop_ids = symbol_orders[symbol_orders["OrderType"] == stop_type]["OrderId"].sort_values()
                    if len(stop_ids):
                        sl_order_id = int(stop_ids.iloc[-1])
                        sl_prices = pd.to_numeric(
                            symbol_orders[symbol_orders["OrderId"] == sl_order_id]["AuxPrice"], errors="coerce"
                        ).dropna()
                        if len(sl_prices):
                            sl_order_price = float(sl_prices.iloc[-1])
                except Exception:
                    pass

                try:
                    tp_ids = symbol_orders[symbol_orders["OrderType"] == "LMT"]["OrderId"].sort_values()
                    if len(tp_ids):
                        tp_order_id = int(tp_ids.iloc[-1])
                        tp_prices = pd.to_numeric(
                            symbol_orders[symbol_orders["OrderId"] == tp_order_id]["LmtPrice"], errors="coerce"
                        ).dropna()
                        if len(tp_prices):
                            tp_order_price = float(tp_prices.iloc[-1])
                except Exception:
                    pass

        lines = [
            f"- The period {app.current_period} was successfully traded",
            f"- The symbol is {app.ticker}",
            f"- The signal is {getattr(app, 'signal', 'n/a')}",
            f"- The leverage is {getattr(app, 'leverage', 'n/a')}",
            f"- The cash balance value is {round(cash_value, 2) if np.isfinite(cash_value) else 'n/a'} {app.account_currency}",
            f"- The current position quantity is {position_qty} {symbol_name}",
            f"- The stop-loss price is {round(sl_order_price, 6) if np.isfinite(sl_order_price) else 'n/a'}",
            f"- The market price is {round(market_order_price, 6) if np.isfinite(market_order_price) else 'n/a'}",
            f"- The take-profit price is {round(tp_order_price, 6) if np.isfinite(tp_order_price) else 'n/a'}",
            "- The live portfolio PDF report is attached when available.",
        ]

        report_path = Path("data") / "portfolio_report.pdf"
        subject = f"EPAT Trading App Status | {app.current_period}"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content("\n".join(lines))

        if report_path.exists():
            with open(report_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="application",
                    subtype="pdf",
                    filename=report_path.name,
                )
        else:
            app.logging.warning("Portfolio PDF not found at email time: %s", report_path)

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(msg)

        print("The email was sent successfully...")
        app.logging.info("The email was sent successfully to %s", to_email)
    except Exception as exc:
        app.logging.error("Failed to send email with portfolio report: %s", exc)
     
# Disconnect the app
def stop(app):
    app.logging.info('[%s] Disconnecting...', getattr(app, 'ticker', 'UNKNOWN'))
    _stop_synthetic_crypto_monitor(app, wait=False)
    app.disconnect()
