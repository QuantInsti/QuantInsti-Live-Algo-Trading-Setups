"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import datetime as dt
from copy import deepcopy
from threading import Event, RLock

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

from ibkr_multi_asset import ib_functions as ibf
from ibkr_multi_asset import trading_functions as tf
from ibkr_multi_asset.create_database import WORKBOOK_SCHEMAS


class trading_app(EClient, EWrapper):
    def __init__(self, logging, account, account_currency, symbol, timezone, data_frequency, historical_data_address, base_df_address,
                 market_open_time, market_close_time,
                 previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, current_period, previous_period, next_period, train_span, test_span, trail, leverage,
                 asset_spec=None, strict_targets_validation=True, allowed_symbols=None, database_path='data/database.xlsx', email_info_path='data/email_info.xlsx'):
        EClient.__init__(self, self)
        self._serverVersion = None

        self.app_start_time = dt.datetime.now()
        self.database_path = database_path
        self.email_info_path = email_info_path
        self.account_currency = account_currency

        if self.port == 4796 or self.port == 4001:
            self.account = 'account'
        else:
            self.account = account

        self.data_frequency = data_frequency
        self.historical_data_address = historical_data_address
        self.base_df_address = base_df_address
        self.zone = timezone
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.previous_day_start_datetime = previous_day_start_datetime
        self.trading_day_end_datetime = trading_day_end_datetime
        self.day_end_datetime = day_end_datetime
        self.current_period = current_period
        self.previous_period = previous_period
        self.next_period = next_period
        self.train_span = train_span
        self.test_span = test_span
        self.frequency_number, self.frequency_string = tf.get_data_frequency_values(data_frequency)

        database = pd.ExcelFile(self.database_path)
        self.open_orders = self._load_sheet(database, 'open_orders', index_col=0, parse_index_datetime=True)
        self.orders_status = self._load_sheet(database, 'orders_status', index_col=0, parse_index_datetime=True)
        self.exec_df = self._load_sheet(database, 'executions', index_col=0, parse_index_datetime=True)
        self.comm_df = self._load_sheet(database, 'commissions', index_col=0, parse_index_datetime=True)
        self.pos_df = self._load_sheet(database, 'positions', index_col=0, parse_index_datetime=True)
        self.portfolio_snapshots_df = self._load_sheet(database, 'portfolio_snapshots', index_col=0, parse_index_datetime=True)
        self.cash_balance = self._load_sheet(database, 'cash_balance', index_col=0, parse_index_datetime=True)
        if 'leverage' not in self.cash_balance.columns:
            self.cash_balance['leverage'] = np.nan
        if 'signal' not in self.cash_balance.columns:
            self.cash_balance['signal'] = np.nan
        self.account_updates_df = self._load_sheet(database, 'account_updates', index_col=0)
        self.contract_details_df = self._load_sheet(database, 'contract_details', index_col=None)
        self.ib_errors = pd.DataFrame()
        self.runtime_audit = pd.DataFrame()
        self.strategy_state_path = os.path.join(os.path.dirname(self.database_path) or '.', 'strategy_state.json')
        self.strategy_state_df = self._load_strategy_state_store(self.strategy_state_path)

        self.app_time_spent_all = self._load_sheet(database, 'app_time_spent', index_col=0, parse_index_datetime=True)
        if 'datetime' not in self.app_time_spent_all.columns and isinstance(self.app_time_spent_all.index, pd.DatetimeIndex):
            self.app_time_spent_all = self.app_time_spent_all.reset_index().rename(columns={'index': 'datetime'})
        self.app_time_spent = self.app_time_spent_all.copy()
        if self.app_time_spent.empty:
            self.app_time_spent = pd.DataFrame(columns=WORKBOOK_SCHEMAS['app_time_spent'])
        self.previous_time_spent = float(pd.to_numeric(self.app_time_spent.get('seconds', pd.Series(dtype=float)), errors='coerce').fillna(0.0).iloc[-1]) if not self.app_time_spent.empty else 0.0

        self.periods_traded_all = self._load_sheet(database, 'periods_traded', index_col=0, parse_index_datetime=True)
        if 'trade_time' in self.periods_traded_all.columns:
            self.periods_traded_all['trade_time'] = pd.to_datetime(self.periods_traded_all['trade_time'], errors='coerce')
        if 'datetime' not in self.periods_traded_all.columns and isinstance(self.periods_traded_all.index, pd.DatetimeIndex):
            self.periods_traded_all = self.periods_traded_all.reset_index().rename(columns={'index': 'datetime'})
        self.periods_traded = self.periods_traded_all.copy()
        period_exists = (self.periods_traded['trade_time'] == current_period).any() if not self.periods_traded.empty else False
        if not period_exists:
            self.periods_traded = pd.concat([
                self.periods_traded,
                pd.DataFrame([{
                    'datetime': dt.datetime.now().replace(microsecond=0),
                    'trade_time': current_period,
                    'trade_done': 0,
                    'market_open_time': market_open_time,
                    'market_close_time': market_close_time,
                }])
            ], ignore_index=True)

        self.new_df = {'0': pd.DataFrame(), '1': pd.DataFrame()}
        self.ticker = symbol
        self.asset_spec = asset_spec or {'symbol': symbol, 'asset_class': 'forex', 'exchange': 'IDEALPRO', 'currency': 'USD'}
        self.strict_targets_validation = strict_targets_validation
        self.allowed_symbols = [str(s).upper() for s in (allowed_symbols or [symbol])]

        self.historical_data = pd.read_csv(historical_data_address, index_col=0)
        self.historical_data.index = pd.to_datetime(self.historical_data.index)
        keep_rows = int(self.train_span)
        if str(self.asset_spec.get('asset_class', 'forex')).lower() == 'crypto':
            keep_rows = max(keep_rows, int(tf.get_periods_per_day(data_frequency) * 40))
        self.historical_data = self.historical_data.tail(keep_rows)
        self.contract = ibf.build_contract_from_spec(self.asset_spec)
        self.resolved_contract = None
        self.errors_dict = {}
        self.hist_request_errors = {}

        if os.path.exists('data/models/optimal_features_df.xlsx'):
            features_df = pd.read_excel('data/models/optimal_features_df.xlsx')
            if 'final_features' in features_df.columns:
                self.final_input_features = features_df['final_features'].dropna().tolist()
            elif len(features_df.columns) >= 1:
                self.final_input_features = pd.Series(features_df.iloc[:, -1]).dropna().tolist()

        self.sl_order_id = np.nan
        self.tp_order_id = np.nan
        self.count = 0
        self.last_value_count = 0
        self.last_value = np.nan

        self.hist_data_events = {'0': Event(), '1': Event()}
        self.orders_request_event = Event()
        self.positions_request_event = Event()
        self.account_update_event = Event()
        self.account_summary_event = Event()
        self.executions_request_event = Event()
        self.contract_details_event = Event()
        self.market_data_event = Event()
        self.broker_sync_lock = RLock()
        self.nextValidOrderId = None
        self.shared_order_id_allocator = None
        self._seen_info_messages = set()
        self.silent_broker_sync = False
        self.silent_broker_sync_depth = 0
        self.bid_price = np.nan
        self.ask_price = np.nan
        self.last_trade_price = np.nan
        self.active_market_data_req_id = None
        self.synthetic_stop_cancel_event = Event()
        self.synthetic_stop_thread = None
        self.synthetic_stop_metadata = {}
        self.last_account_update_time_printed = None

        self.acc_update = pd.DataFrame(columns=WORKBOOK_SCHEMAS['account_updates'])
        self.temp_open_orders = pd.DataFrame()
        self.current_open_orders_snapshot = pd.DataFrame()
        self.temp_orders_status = pd.DataFrame()
        self.temp_exec_df = pd.DataFrame()
        self.temp_comm_df = pd.DataFrame()
        self.temp_pos_df = pd.DataFrame()
        self.temp_portfolio_snapshots = pd.DataFrame(columns=WORKBOOK_SCHEMAS['portfolio_snapshots'])
        self.temp_account_summary = pd.DataFrame()
        self.temp_account_update_times = pd.DataFrame()
        self.temp_contract_details = pd.DataFrame(columns=WORKBOOK_SCHEMAS['contract_details'])
        self.temp_ib_errors = pd.DataFrame()
        self.temp_runtime_audit = pd.DataFrame()
        self.temp_strategy_state = pd.DataFrame(columns=['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime'])
        self.strategy_state = self._deserialize_strategy_state(self.strategy_state_df, symbol)
        self.strategy_state_updates = {}
        self.risk_management_price_overrides = {}

        self.trail = trail
        self.leverage = leverage
        self.signal = 0.0
        self.risk_management_bool = True
        self.strategy_end = False
        self.logging = logging

    @staticmethod
    def _load_sheet(database, sheet_name, index_col=0, parse_index_datetime=False):
        columns = WORKBOOK_SCHEMAS.get(sheet_name, [])
        if sheet_name in database.sheet_names:
            df = database.parse(sheet_name, index_col=index_col)
        else:
            df = pd.DataFrame(columns=columns)
        unnamed_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        if 'datetime' not in df.columns and parse_index_datetime:
            try:
                index_as_series = pd.Series(df.index, dtype='object')
                looks_textual = index_as_series.astype(str).str.contains(r'[-/: T]', regex=True, na=False).any()
                if looks_textual:
                    datetime_index = pd.to_datetime(index_as_series, errors='coerce', format='mixed')
                    if getattr(datetime_index, 'notna', lambda: [])().any():
                        df = df.copy()
                        df['datetime'] = datetime_index.values
            except Exception:
                pass
        if parse_index_datetime:
            try:
                index_as_series = pd.Series(df.index, dtype='object')
                looks_textual = index_as_series.astype(str).str.contains(r'[-/: T]', regex=True, na=False).any()
                if looks_textual:
                    df.index = pd.to_datetime(index_as_series, errors='coerce', format='mixed')
                    df = df[~df.index.isna()]
            except Exception:
                pass
        return df

    @staticmethod
    def _load_strategy_state_store(path):
        columns = ['Symbol', 'Scope', 'StateKey', 'StateValue', 'datetime']
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=columns)
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return pd.DataFrame(columns=columns)
        rows = payload.get('rows', []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame.loc[:, columns]

    @staticmethod
    def _deserialize_strategy_state(df, symbol):
        if df is None or df.empty:
            return {}
        local = df.copy()
        if 'datetime' in local.columns:
            local['datetime'] = pd.to_datetime(local['datetime'], errors='coerce')
        if 'Symbol' not in local.columns:
            local['Symbol'] = symbol
        scoped = local[local['Symbol'].astype(str).str.upper().isin([str(symbol).upper(), '__GLOBAL__'])].copy()
        if scoped.empty:
            return {}
        if 'datetime' in scoped.columns:
            scoped = scoped.sort_values('datetime')
        scoped = scoped.drop_duplicates(subset=['Symbol', 'Scope', 'StateKey'], keep='last')
        out = {}
        for row in scoped.itertuples(index=False):
            scope = str(getattr(row, 'Scope', 'symbol'))
            key = str(getattr(row, 'StateKey', 'state'))
            raw = getattr(row, 'StateValue', '{}')
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                value = raw
            out.setdefault(scope, {})[key] = value
        return out

    def queue_strategy_state(self, updates, symbol=None):
        if not isinstance(updates, dict) or len(updates) == 0:
            return
        now = dt.datetime.now().replace(microsecond=0)
        target_symbol = str(symbol or self.ticker).upper()
        rows = []
        for scope, values in updates.items():
            if not isinstance(values, dict):
                values = {'value': values}
            self.strategy_state.setdefault(str(scope), {}).update(values)
            for key, value in values.items():
                try:
                    encoded = json.dumps(value, default=str)
                except Exception:
                    encoded = json.dumps(str(value))
                rows.append({'Symbol': target_symbol, 'Scope': str(scope), 'StateKey': str(key), 'StateValue': encoded, 'datetime': now})
        if rows:
            self.temp_strategy_state = pd.concat([self.temp_strategy_state, pd.DataFrame(rows)], ignore_index=True)

    def error(self, reqId, code_or_time, msg_or_code, *args, **kwargs):
        actual_code = code_or_time
        actual_msg = msg_or_code

        # ibapi versions differ here:
        # old: error(reqId, errorCode, errorString)
        # new: error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson='')
        if isinstance(code_or_time, int) and isinstance(msg_or_code, int):
            actual_code = msg_or_code
            actual_msg = args[0] if len(args) > 0 else ''
        elif actual_code == 0 and isinstance(msg_or_code, int):
            actual_code = msg_or_code
            actual_msg = args[0] if len(args) > 0 else ''

        self.errors_dict[actual_code] = actual_msg
        if str(reqId) in getattr(self, 'hist_data_events', {}):
            self.hist_request_errors[str(reqId)] = {'code': actual_code, 'msg': actual_msg}
            if actual_code in (162, 165, 166, 200, 354, 366, 10299):
                self.hist_data_events[str(reqId)].set()

        info_codes = {2100, 2103, 2104, 2106, 2158}
        message_key = (actual_code, str(actual_msg))
        is_repeated_info = actual_code in info_codes
        if is_repeated_info:
            if message_key not in self._seen_info_messages:
                self._seen_info_messages.add(message_key)
                self.logging.info('[%s] IB info: %s - %s', self.ticker, actual_code, actual_msg)
        else:
            print('Error: {} - {} - {}'.format(reqId, actual_code, actual_msg))
            self.logging.info('Error: {} - {} - {}'.format(reqId, actual_code, actual_msg))

    def serverVersion(self):
        """Return the negotiated IB API server version after connect()."""
        value = getattr(self, 'serverVersion_', None)
        if value is not None:
            return value
        return 0

    def run(self):
        super().run()

    def _is_verbose_broker_callback_enabled(self):
        return not (
            bool(getattr(self, 'silent_broker_sync', False))
            or int(getattr(self, 'silent_broker_sync_depth', 0)) > 0
            or bool(getattr(self, 'defer_posttrade_sync', False))
        )

    def _account_callback_label(self):
        return f'[ACCOUNT {self.account}]'

    def nextValidId(self, orderId):
        if self.isConnected():
            super().nextValidId(orderId)
            self.nextValidOrderId = orderId
            self.logging.info('[%s] NextValidId: %s', self.ticker, orderId)
            time.sleep(1)
        else:
            return

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        dictionary = {'PermId': order.permId,
                      'ClientId': order.clientId,
                      'OrderId': orderId,
                      'Account': order.account,
                      'Symbol': contract.symbol,
                      'Currency': getattr(contract, 'currency', pd.NA),
                      'SecType': contract.secType,
                      'Exchange': contract.exchange,
                      'Action': order.action,
                      'OrderType': order.orderType,
                      'TotalQty': float(order.totalQuantity),
                      'CashQty': order.cashQty,
                      'LmtPrice': order.lmtPrice,
                      'AuxPrice': order.auxPrice,
                      'Status': orderState.status,
                      'datetime': dt.datetime.now().replace(microsecond=0)}
        self.temp_open_orders = pd.concat([self.temp_open_orders, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld,
                    mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice,
                            clientId, whyHeld, mktCapPrice)
        dictionary = {'OrderId': orderId,
                      'Status': status,
                      'Filled': filled,
                      'PermId': permId,
                      'ClientId': clientId,
                      'Remaining': float(remaining),
                      'AvgFillPrice': avgFillPrice,
                      'LastFillPrice': lastFillPrice,
                      'datetime': dt.datetime.now().replace(microsecond=0)}
        self.temp_orders_status = pd.concat([self.temp_orders_status, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def openOrderEnd(self):
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Open orders request was successfully completed')
        self.orders_request_event.set()

    def execDetails(self, reqId: int, contract, execution):
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Requesting the trading executions...')
            self.logging.info(f'[{self.ticker}] Requesting the trading executions...')
        super().execDetails(reqId, contract, execution)
        dictionary = {'OrderId': execution.orderId,
                      'PermId': execution.permId,
                      'ExecutionId': execution.execId,
                      'Symbol': contract.symbol,
                      'Side': execution.side,
                      'Price': execution.price,
                      'AvPrice': execution.avgPrice,
                      'cumQty': execution.cumQty,
                      'Currency': contract.currency,
                      'SecType': contract.secType,
                      'Position': float(execution.shares),
                      'Execution Time': execution.time,
                      'Last Liquidity': execution.lastLiquidity,
                      'OrderRef': execution.orderRef,
                      'datetime': dt.datetime.now().replace(microsecond=0)}
        self.temp_exec_df = pd.concat([self.temp_exec_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def commissionReport(self, commissionReport):
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Requesting the trading commissions...')
            self.logging.info(f'[{self.ticker}] Requesting the trading commissions...')
        super().commissionReport(commissionReport)
        dictionary = {'ExecutionId': commissionReport.execId,
                      'Commission': commissionReport.commission,
                      'Currency': commissionReport.currency,
                      'Realized PnL': float(commissionReport.realizedPNL),
                      'datetime': dt.datetime.now().replace(microsecond=0)}
        self.temp_comm_df = pd.concat([self.temp_comm_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def execDetailsEnd(self, reqId: int):
        super().execDetailsEnd(reqId)
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Trading executions request was successfully finished. ReqId: {reqId}')
        self.executions_request_event.set()

    def position(self, account, contract, position, avgCost):
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Requesting the trading positions...')
            self.logging.info(f'[{self.ticker}] Requesting the trading positions...')
        super().position(account, contract, position, avgCost)
        dictionary = {'Account': account, 'Symbol': contract.symbol,
                      'SecType': contract.secType,
                      'Currency': contract.currency, 'Position': float(position),
                      'Avg cost': avgCost, 'datetime': dt.datetime.now().replace(microsecond=0)}
        self.temp_pos_df = pd.concat([self.temp_pos_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def positionEnd(self):
        if self._is_verbose_broker_callback_enabled():
            print(f'[{self.ticker}] Positions Retrieved.')
        self.positions_request_event.set()

    def updatePortfolio(self, contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName):
        super().updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName)
        dictionary = {
            'Account': accountName,
            'Symbol': getattr(contract, 'symbol', ''),
            'LocalSymbol': getattr(contract, 'localSymbol', ''),
            'SecType': getattr(contract, 'secType', ''),
            'Exchange': getattr(contract, 'exchange', ''),
            'Currency': getattr(contract, 'currency', ''),
            'ConId': getattr(contract, 'conId', np.nan),
            'Position': float(position),
            'MarketPrice': marketPrice,
            'MarketValue': marketValue,
            'AverageCost': averageCost,
            'UnrealizedPnL': unrealizedPNL,
            'RealizedPnL': realizedPNL,
            'datetime': dt.datetime.now().replace(microsecond=0),
        }
        self.temp_portfolio_snapshots = pd.concat([self.temp_portfolio_snapshots, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def historicalData(self, reqId, bar):
        self.new_df[f'{reqId}'].loc[bar.date, 'close'] = bar.close
        self.new_df[f'{reqId}'].loc[bar.date, 'open'] = bar.open
        self.new_df[f'{reqId}'].loc[bar.date, 'high'] = bar.high
        self.new_df[f'{reqId}'].loc[bar.date, 'low'] = bar.low

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print(f'[{self.ticker}] Historical Data Download finished...')
        self.logging.info(f'[{self.ticker}] Historical Data Download finished...')
        self.hist_data_events[f'{reqId}'].set()

    def tickByTickMidPoint(self, reqId, tick_time, midpoint):
        self.last_value = midpoint

    def tickPrice(self, reqId, tickType, price, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        try:
            value = float(price)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        if tickType in {1, 66}:
            self.bid_price = value
        elif tickType in {2, 67}:
            self.ask_price = value
        elif tickType in {4, 68, 9, 75}:
            self.last_trade_price = value
            self.last_value = value
        self.market_data_event.set()

    def updateAccountValue(self, key, value, currency, accountName):
        super().updateAccountValue(key, value, currency, accountName)
        dictionary = {'key': key,
                      'Account': accountName,
                      'Value': value,
                      'Currency': currency,
                      'datetime': dt.datetime.now().replace(microsecond=0)}
        self.acc_update = pd.concat([self.acc_update, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def updateAccountTime(self, timeStamp: str):
        if str(timeStamp) != str(getattr(self, 'last_account_update_time_printed', None)):
            print(f'{self._account_callback_label()} Account update time is: {timeStamp}')
            self.last_account_update_time_printed = str(timeStamp)
        dictionary = {
            'Account': self.account,
            'UpdateTime': timeStamp,
            'datetime': dt.datetime.now().replace(microsecond=0),
        }
        if isinstance(self.temp_account_update_times, pd.DataFrame):
            self.temp_account_update_times = pd.concat([self.temp_account_update_times, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def accountDownloadEnd(self, accountName: str):
        print(f'{self._account_callback_label()} Account download was done for account: {accountName}')
        self.logging.info(f'Account download was done for account: {accountName}')
        self.account_update_event.set()

    def accountSummary(self, reqId, account, tag, value, currency):
        super().accountSummary(reqId, account, tag, value, currency)
        dictionary = {
            'ReqId': reqId,
            'Account': account,
            'Tag': tag,
            'Value': value,
            'Currency': currency,
            'datetime': dt.datetime.now().replace(microsecond=0),
        }
        if isinstance(self.temp_account_summary, pd.DataFrame):
            self.temp_account_summary = pd.concat([self.temp_account_summary, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def accountSummaryEnd(self, reqId):
        super().accountSummaryEnd(reqId)
        self.account_summary_event.set()

    def contractDetails(self, reqId, contractDetails):
        super().contractDetails(reqId, contractDetails)
        summary = getattr(contractDetails, 'contract', None)
        if summary is None:
            summary = getattr(contractDetails, 'summary', contractDetails)
        dictionary = {
            'Symbol': getattr(summary, 'symbol', ''),
            'SecType': getattr(summary, 'secType', ''),
            'Currency': getattr(summary, 'currency', ''),
            'Exchange': getattr(summary, 'exchange', ''),
            'PrimaryExchange': getattr(summary, 'primaryExchange', ''),
            'LocalSymbol': getattr(summary, 'localSymbol', ''),
            'LastTradeDateOrContractMonth': getattr(summary, 'lastTradeDateOrContractMonth', ''),
            'ValidExchanges': getattr(contractDetails, 'validExchanges', ''),
            'TradingClass': getattr(summary, 'tradingClass', ''),
            'MarketName': getattr(contractDetails, 'marketName', ''),
            'LongName': getattr(contractDetails, 'longName', ''),
            'ConId': getattr(summary, 'conId', np.nan),
            'MinTick': getattr(contractDetails, 'minTick', np.nan),
            'MinSize': getattr(contractDetails, 'minSize', np.nan),
            'SizeIncrement': getattr(contractDetails, 'sizeIncrement', np.nan),
            'SuggestedSizeIncrement': getattr(contractDetails, 'suggestedSizeIncrement', np.nan),
            'Multiplier': getattr(summary, 'multiplier', ''),
            'TimeZoneId': getattr(contractDetails, 'timeZoneId', ''),
            'OrderTypes': getattr(contractDetails, 'orderTypes', ''),
            'LiquidHours': getattr(contractDetails, 'liquidHours', ''),
            'TradingHours': getattr(contractDetails, 'tradingHours', ''),
            'datetime': dt.datetime.now().replace(microsecond=0),
        }
        resolved_contract = ibf.build_contract_from_spec(self.asset_spec)
        for attr in ('symbol', 'secType', 'currency', 'exchange', 'primaryExchange', 'localSymbol', 'tradingClass', 'multiplier', 'lastTradeDateOrContractMonth'):
            value = getattr(summary, attr, '')
            if value not in (None, ''):
                setattr(resolved_contract, attr, value)
        con_id = getattr(summary, 'conId', None)
        if con_id not in (None, ''):
            resolved_contract.conId = con_id
        self.resolved_contract = resolved_contract
        self.temp_contract_details = pd.concat([self.temp_contract_details, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        self.contract_details_event.set()
