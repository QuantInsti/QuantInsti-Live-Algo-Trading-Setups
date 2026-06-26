"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

# Import the necessary libraries
import os
import time
import numpy as np
import pandas as pd
from threading import Event
from threading import Thread
from datetime import datetime
from concurrent import futures
from ibkr_multi_asset import trading_functions as tf
from ibapi.client import EClient
from ibkr_multi_asset import ib_functions as ibf
from ibapi.wrapper import EWrapper

import warnings
warnings.filterwarnings('ignore')

##########################################################################
    
class app_for_download_data(EWrapper, EClient):
    ''' Serves as the client and the wrapper '''

    def __init__(self, addr, client_id, file_name, update = False, \
                 contract = {}, now = '', download_span = '1 D', timezone= 'US/Eastern', \
                 saturdays = list(), what_to_show = None, silent = False, bar_size = '1 min'):
        
        EClient.__init__(self, self)
        
        # Silent mode for parallel bulk downloads
        self.silent = silent
        
        # Bar size for historical data requests (matches asset frequency)
        self.bar_size = bar_size
        
        # Number of threads
        self.workers = 4
        
        # Set the time zone of the trader
        self.zone = timezone
                
        # Get the Forex contract
        self.contract = contract

        # Dataframe dictionary to save the historical dataframes
        self.dfs = {}
        
        # List of lists of params to be used while downloading the historical data
        self.params_list = list()
        
        # DataFrame to save the completed dataframe to be saved
        self.end_df = pd.DataFrame()
        
        # Dictionary to save the threading events to be used while downloading the historical data
        self.events = {}
                
        # File name to be used to save the historical data
        self.file_name = file_name
        
        # If it's the first time you download data or you haven't completed to download from previous years
        if update == 'false':
            # Check if you already have a historical data file
            try: 
                # Import the historical data
                self.end_df = pd.read_csv(self.file_name, index_col=0)
                # Set the now datetime to the first index value of the dataframe
                self.now = self.end_df.index[0]
                # Set the index to datetime type
                self.end_df.index = pd.to_datetime(self.end_df.index)
                # Get the Saturdays list from January 2005 to the now datetime
                self.saturdays = [date0 for date0 in saturdays if date0<=self.now]
            except:
                # Set the now datetime
                self.now = now
                # Create the dataframe to be used to save the historical data
                self.end_df = pd.DataFrame()
                # Save the Saturdays list inside the class app
                self.saturdays = saturdays
        # If we already have a historical complete historical data and we just need to update it to the most recent datetime
        elif update == 'true':
            # Set the now datetime
            self.now = now
            # Import the historical dataframe
            self.end_df = pd.read_csv(self.file_name, index_col=0)
            # Set the index to datetime type
            self.end_df.index = pd.to_datetime(self.end_df.index)
            # From this first date to now we're going to download data
            first_date = self.end_df.index[-1].strftime('%Y%m%d-%H:%M:%S')  
            # Subset the Saturdays list from the first date onwards
            self.saturdays = [date0 for date0 in saturdays if date0>=first_date]
            
            if (len(self.saturdays)==1) and (self.saturdays[0] >= first_date):
                return
        # In case you want to just fill
        elif update == 'fill':
            self.end_df = pd.read_csv(self.file_name, index_col=0)
            self.end_df.index = pd.to_datetime(self.end_df.index)
            self.now = now
            self.saturdays = saturdays
        
        # Normalise what_to_show to a list (default BID+ASK for bid/ask mid-point pairs)
        if what_to_show is None:
            self.what_to_show = ['BID', 'ASK']
        elif isinstance(what_to_show, str):
            self.what_to_show = [what_to_show]
        else:
            self.what_to_show = list(what_to_show)
        # Create the params list — one entry per whatToShow value per Saturday
        j = 0
        for date0 in self.saturdays:
            for wts in self.what_to_show:
                self.params_list.append([j, self.contract, date0, wts])
                self.dfs[f'{j}'] = pd.DataFrame()
                j += 1
            
        # Set the order-IDs list as per the Saturday’s list
        self.orderIDs = list(range(0,(len(self.params_list))))
        
        # Set the span for each download request
        self.span = download_span
        # A list of all the IB ports
        self.ports = [7497,7496,4001,4002]
        # The addr value
        self.addr = addr
        # Set the client id number
        self.client_id = client_id

        # Error Dictionary
        self.errors_code_dict = {}

        # Launch app message boolean
        self.launch_message_bool = False
        
        # Launch the app
        self.LaunchApp()
        
        if self.launch_message_bool == True: # Successful App Launching
            # Launch the client thread
            thread = Thread(target=self.run)
            thread.start()
            # Once the thread starts, Initiate the app algorithm
            result = self.InitiateAlgorithm(1)
            # If the initialization failed
            if isinstance(result,ValueError):
                print(result)
                # In case there is an error, disconnect the app
                time.sleep(3)
                self.disconnect()
            # If the initialization was successful
            elif isinstance(result, pd.DataFrame):
                if self.comment == True:
                    print(result)
                # In case the whole algorithm ends, disconnect the app
                time.sleep(3)
                self.disconnect()
            
    def error(self, reqId, code, msg, *args, **kwargs):
        ''' Called if an error occurs '''
        self.errors_code_dict[code] = msg

        # Suppress informational farm-connection messages (IB sends these
        # with code=0 and the info-code as the text)
        info_codes = {'2103', '2104', '2105', '2106', '2107', '2157', '2158'}
        if str(code) in info_codes or str(msg).strip() in info_codes:
            return

        # Identify the symbol for context
        ct = getattr(self, 'contract', None) or {}
        sym = str(getattr(ct, 'symbol', '') or '')
        cur = str(getattr(ct, 'currency', '') or '')
        symbol = sym + cur if (sym and cur and len(sym) <= 3) else (sym or '?')
        print(f'[{symbol}] IB Error {code} (reqId={reqId}): {msg}')

        if msg == 'Not connected' or \
            msg == \
            "Connectivity between IB and Trader Workstation has been lost." \
            or msg == 'HMDS data farm connection is broken:cashhmds':
            time.sleep(3)
            self.disconnect()
            
    def LaunchApp(self):
        ''' Function to launch the app '''
        for port in self.ports:
            # Create the client and connect to TWS with live trading
            self.connect(self.addr, port, self.client_id)
            time.sleep(4)
            # If the connection wasn't established or the port wasn't the correct one
            if (504 in self.errors_code_dict) == False and \
               (502 in self.errors_code_dict) == False:
                if not self.silent:
                    if port == 7496:  
                        print("="*80)
                        print('Setup successfully launched with IB TWS for downloading historical data...')
                        print("="*80)
                    elif port == 7497:
                        print("="*80)
                        print('Setup successfully launched with IB TWS for downloading historical data...')
                        print("="*80)
                    elif port == 4001:
                        print("="*80)
                        print('Setup successfully launched with IB Gateway for downloading historical data...')
                        print("="*80)
                    elif port == 4002:
                        print("="*80)
                        print('Setup successfully launched with IB Gateway for downloading historical data...')
                        print("="*80)
                # Let's download the historical data
                self.launch_message_bool = True
                break
            
            # Clean the errors dictionary
            self.errors_code_dict = {}
        # If any port worked
        if 504 in self.errors_code_dict:
            print('Please open the IB Gateway or TWS...')
        # Print the 502 error in case you haven't followed the appropriate instructions
        elif 502 in self.errors_code_dict:
            print(self.errors_code_dict[502])
            
    def contractDetails(self, reqId, contractDetails):
        '''Called in response to reqContractDetails — resolves futures front month.'''
        super().contractDetails(reqId, contractDetails)
        summary = getattr(contractDetails, 'summary', contractDetails)
        local_symbol = str(getattr(summary, 'localSymbol', '') or '')
        expiry = str(getattr(summary, 'lastTradeDateOrContractMonth', '') or '')
        con_id = getattr(summary, 'conId', None)
        multiplier = str(getattr(summary, 'multiplier', '') or '')
        if local_symbol and expiry:
            if not hasattr(self, '_resolved_contract'):
                self._resolved_contract = None
            # Keep the earliest (front-month) contract
            if self._resolved_contract is None or expiry < str(getattr(self._resolved_contract, 'lastTradeDateOrContractMonth', '99999999')):
                from ibapi.client import Contract as IBContract
                resolved = IBContract()
                resolved.symbol = getattr(self.contract, 'symbol', '')
                resolved.secType = 'FUT'
                resolved.exchange = getattr(self.contract, 'exchange', 'CME')
                resolved.currency = getattr(self.contract, 'currency', 'USD')
                resolved.localSymbol = local_symbol
                resolved.lastTradeDateOrContractMonth = expiry
                if multiplier:
                    resolved.multiplier = multiplier
                if con_id is not None:
                    resolved.conId = int(con_id)
                self._resolved_contract = resolved

    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        self._contract_details_event.set()

    def _resolve_contract_if_needed(self):
        '''If this is a futures contract without expiry, resolve via reqContractDetails.'''
        if getattr(self.contract, 'secType', '') != 'FUT':
            return
        if getattr(self.contract, 'lastTradeDateOrContractMonth', ''):
            return  # already has expiry
        self._contract_details_event = Event()
        self._contract_details_event.clear()
        self._resolved_contract = None
        self.reqContractDetails(9999, self.contract)
        self._contract_details_event.wait(timeout=15)
        if self._resolved_contract is not None:
            print(f"[{self.contract.symbol}] Resolved contract: {self._resolved_contract.localSymbol} expiry={self._resolved_contract.lastTradeDateOrContractMonth}")
            self.contract = self._resolved_contract

    def InitiateAlgorithm(self, activate = False):
        ''' Function to initiate the historical data download algorithm '''
        
        # If the app was successfully launched
        if activate == True:
            time.sleep(3)
            # Resolve futures contract if needed
            self._resolve_contract_if_needed()
            # Rebuild params with resolved contract
            self.params_list = []
            self.dfs = {}
            j = 0
            for date0 in self.saturdays:
                for wts in self.what_to_show:
                    self.params_list.append([j, self.contract, date0, wts])
                    self.dfs[f'{j}'] = pd.DataFrame()
                    j += 1
            self.orderIDs = list(range(0, len(self.params_list)))
            print(f'Download starts at {datetime.now()}')
            # Download the data and save the result
            result = self.multithreading_loop()
            # Disconnect the app
            self.disconnect()
            return result
        
    #@iswrapper
    def historicalData(self, reqId, bar):
        ''' Called in response to reqHistoricalData '''
        
        self.dfs[f'{reqId}'].loc[bar.date,'close'] = bar.close
        self.dfs[f'{reqId}'].loc[bar.date,'open'] = bar.open
        self.dfs[f'{reqId}'].loc[bar.date,'high'] = bar.high
        self.dfs[f'{reqId}'].loc[bar.date,'low'] = bar.low

    def historicalDataEnd(self, reqId, start, end):
        ''' Called when the historical data for reqId is finished '''
        super().historicalDataEnd(reqId, start, end)
        print(f"Historical Data for ID {reqId} Download End")
        self.events[f'{reqId}'].set()
            
    def request_data(self, params):
        ''' Called when the historical data for reqId is finished '''
        
        print(f"Historical Data for ID {params[0]} Download Starts...")
        
        # Set the threading event for the download request
        self.events[f'{params[0]}'] = Event()
        # Clear the threading event
        self.events[f'{params[0]}'].clear()
        # Download the data
        self.reqHistoricalData(params[0], params[1], params[2], self.span, self.bar_size, \
                               params[3], False, 1, False, [])
        # Make the event to wait until the download is completed
        self.events[f'{params[0]}'].wait()
        
    def update_df(self, params_list):
        ''' Function to update the whole historical dataframe '''
        
        # Set the last param list number
        last_params_num = params_list[-1][0]
        # Set the first param list number
        j = params_list[0][0]
        stride = len(self.what_to_show)  # 2 for BID+ASK, 1 for TRADES
        # Iterate through each params list
        while j <= last_params_num:
            if stride == 2:
                # ── Smart date parsing: daily bars have "YYYYMMDD", intraday have "YYYYMMDD HH:MM:SS TZ" ──
                def _parse_bar_index(idx):
                    sample = str(idx[0]) if len(idx) > 0 else ''
                    if sample and len(sample) <= 10 and ' ' not in sample:
                        return pd.to_datetime(idx, format='%Y%m%d')
                    return pd.to_datetime(idx, format='%Y%m%d %H:%M:%S %Z')
                # ── Bid/Ask pair mode ──
                # Change the BID dataframe columns' names
                bid_columns = ['bid_'+column for column in self.dfs[f'{j}'].columns.tolist()]
                self.dfs[f'{j}'].columns = bid_columns
                self.dfs[f'{j}'].index = _parse_bar_index(self.dfs[f'{j}'].index)
                if len(self.dfs[f'{j}'].index) > 0 and ' ' in str(self.dfs[f'{j}'].index[0]):
                    self.dfs[f'{j}'].index = self.dfs[f'{j}'].index.tz_localize(None)
                
                # Change the ASK dataframe columns' names
                ask_columns = ['ask_'+column for column in self.dfs[f'{j+1}'].columns.tolist()]
                self.dfs[f'{j+1}'].columns = ask_columns
                self.dfs[f'{j+1}'].index = _parse_bar_index(self.dfs[f'{j+1}'].index)
                if len(self.dfs[f'{j+1}'].index) > 0 and ' ' in str(self.dfs[f'{j+1}'].index[0]):
                    self.dfs[f'{j+1}'].index = self.dfs[f'{j+1}'].index.tz_localize(None)
                
                # Concatenate the BID and ASK dataframes into a single one
                temp_df = pd.concat([self.dfs[f'{j}'],self.dfs[f'{j+1}']], axis=1)
            else:
                # ── Single whatToShow (e.g. TRADES) — OHLC columns, no rename ──
                def _parse_bar_index_single(idx):
                    sample = str(idx[0]) if len(idx) > 0 else ''
                    if sample and len(sample) <= 10 and ' ' not in sample:
                        return pd.to_datetime(idx, format='%Y%m%d')
                    return pd.to_datetime(idx, format='%Y%m%d %H:%M:%S %Z')
                self.dfs[f'{j}'].index = _parse_bar_index_single(self.dfs[f'{j}'].index)
                if len(self.dfs[f'{j}'].index) > 0 and ' ' in str(self.dfs[f'{j}'].index[0]):
                    self.dfs[f'{j}'].index = self.dfs[f'{j}'].index.tz_localize(None)
                temp_df = self.dfs[f'{j}']
            # Update the whole historical dataframe
            self.end_df = pd.concat([self.end_df,temp_df])
            # Sort the dataframe by index
            self.end_df = self.end_df.sort_index()
            # Drop duplicates
            self.end_df = self.end_df[~self.end_df.index.duplicated(keep='first')]
            # Save the historical dataframe into a CSV file
            self.end_df.to_csv(self.file_name, encoding='utf-8', index=True)
            # Update the params list number
            j += stride

    def multithreading_loop(self):
        ''' Function to update the whole historical dataframe '''
        
        # Get the number of groups of 50 params lists (50 is the maximum number of requests allowed by IB)
        num_lists = int(np.floor(float(len(self.params_list))/float(50)))
        # Get the params lists that will be in each group
        params_sublists = [self.params_list[(i*50):((i+1)*50)] for i in range(num_lists)]
        # Get the group of params lists left after params sublists
        params_sublists_left = self.params_list[int(num_lists*50):]
        
        # Iterate through each params sublist
        for params_sublist in params_sublists:
            # Get the number of groups of "workers" params list 
            num_sublist = int(np.floor(float(len(params_sublist))/float(self.workers)))
            # Get the params lists that will be in each group
            params_sublist_for_loop = [params_sublist[(i*self.workers):((i+1)*self.workers)] for i in range(num_sublist)]
            # Get the group of params lists left after the params sublist
            params_sublist_left = params_sublist[int(num_sublist*self.workers):]
            
            # Iterate through each list in the params sublist
            for i in range(0,num_sublist):
                with futures.ThreadPoolExecutor(self.workers) as executor:
                    # Make "workers" requests in parallel
                    executor.map(self.request_data, params_sublist_for_loop[i])
                # Update the historical dataframe
                self.update_df(params_sublist_for_loop[i])                        
                
            # Set the length of the second sublist
            sublist_left_len = len(params_sublist_left)
            
            if sublist_left_len != 0:
                # Request the data belonging to the second sub-params list
                with futures.ThreadPoolExecutor(sublist_left_len) as executor:
                    executor.map(self.request_data, params_sublist_left)
                # Update the historical dataframe
                self.update_df(params_sublist_left)  
 
            # Sleep loop for 10 minutes to refresh the IB limitations
            time.sleep(10*60) 
                    
        # The last sublist: params_sublists_left
        num_sublist = int(np.floor(float(len(params_sublists_left))/float(self.workers)))
        # Get the params list in groups of "workers"
        params_sublist_for_loop = [params_sublists_left[(i*self.workers):((i+1)*self.workers)] for i in range(num_sublist)]
        # Get the remaining params lists after the previous sublist
        params_sublist_left = params_sublists_left[int(num_sublist*self.workers):]
        
        # Iterate through each sublist
        for i in range(0,num_sublist):
            with futures.ThreadPoolExecutor(self.workers) as executor:
                # Make "workers" requests in parallel
                executor.map(self.request_data, params_sublist_for_loop[i])
            # Update the historical dataframe
            self.update_df(params_sublist_for_loop[i])                        
            
        # Set the length of the remaining group
        sublist_left_len = len(params_sublist_left)
        
        if sublist_left_len != 0:
            # Request the data belonging to the remaining sub-params list
            with futures.ThreadPoolExecutor(sublist_left_len) as executor:
                executor.map(self.request_data, params_sublist_left)
                
            # Update the historical dataframe
            self.update_df(params_sublist_left) 
            
        # Save the historical dataframe into a CSV file
        self.end_df.to_csv(self.file_name, encoding='utf-8', index=True)
        
        print('Download of historical minute data is completed')
                                         
# -------------------------x-----------------------x--------------------------#
# -------------------------x-----------------------x--------------------------#
# -------------------------x-----------------------x--------------------------#
def _resolve_ohlc_for_resample(raw_df):
    """If the raw minute data has bid/ask columns, compute mid-prices.
    Otherwise (TRADES OHLC) build a frame in the exact column order
    that tf.get_mid_series produces, so tf.resample_df doesn't
    create duplicate lowercase columns."""
    cols_lower = {str(c).lower() for c in raw_df.columns}
    if 'bid_close' in cols_lower:
        return tf.get_mid_series(raw_df)
    # OHLC data — map columns case-insensitively, output in get_mid_series order
    col_map = {str(c).lower(): c for c in raw_df.columns}
    ordered = {}
    for target in ('Open', 'Close', 'High', 'Low'):
        src = col_map.get(target.lower())
        if src is not None:
            ordered[target] = pd.to_numeric(raw_df[src], errors='coerce')
    if len(ordered) == 4:
        return pd.DataFrame(ordered, index=raw_df.index)
    return raw_df

def update_historical_resampled_data(historical_minute_data, historical_data_address, train_span, data_frequency, market_open_time):
    
    # Set the hour string to resample the data
    hour_string = str(market_open_time.hour) if (market_open_time.hour)>=10 else '0'+str(market_open_time.hour)
    minute_string = str(market_open_time.minute) if (market_open_time.minute)>=10 else '0'+str(market_open_time.minute)

    # If the historical minute data variable is a dataframe
    if isinstance(historical_minute_data, pd.DataFrame):
        try:
            resampled_path = historical_data_address if os.path.isabs(historical_data_address) or historical_data_address.startswith("data") else 'data/'+historical_data_address
            historical_resampled_data = pd.read_csv(resampled_path, index_col=0)
            historical_resampled_data.index = pd.to_datetime(historical_resampled_data.index)
            if (historical_minute_data.index[-1].day==historical_resampled_data.index[-1].day) or \
                (historical_minute_data.index[-1].day+1==historical_resampled_data.index[-1].day):
                print('Resampling was already done...')
            else:
                print('Resample of historical minute data as per the data frequency is in process...')
                # Resample the data as per the trading frequency
                ohlc = _resolve_ohlc_for_resample(historical_minute_data)
                historical_data = tf.resample_df(ohlc, data_frequency, start=f'{hour_string}h{minute_string}min')

                # Subset the resample historical data to "train_span observations
                historical_data.tail(train_span).to_csv(historical_data_address)
    
        except:

            # Resample the data as per the trading frequency
            ohlc = _resolve_ohlc_for_resample(historical_minute_data)
            historical_data = tf.resample_df(ohlc, data_frequency, start=f'{hour_string}h{minute_string}min')

            # Subset the resample historical data to "train_span observations
            historical_data.tail(train_span).to_csv(historical_data_address)
    
    # If it's a string address
    else:
        # Import the historical minute-frequency data
        historical_minute_data = pd.read_csv(historical_minute_data, index_col=0)
        # Convert the index to datetime type
        historical_minute_data.index = pd.to_datetime(historical_minute_data.index)

        try:
            resampled_path = historical_data_address if os.path.isabs(historical_data_address) or historical_data_address.startswith("data") else 'data/'+historical_data_address
            historical_resampled_data = pd.read_csv(resampled_path, index_col=0)
            historical_resampled_data.index = pd.to_datetime(historical_resampled_data.index)
            if (historical_minute_data.index[-1].day==historical_resampled_data.index[-1].day) or \
                (historical_minute_data.index[-1].day+1==historical_resampled_data.index[-1].day):
                print('Resampling was already done...')
            else:
                print('Resample of historical minute data as per the data frequency is in process...')
                # Resample the data as per the trading frequency
                ohlc = _resolve_ohlc_for_resample(historical_minute_data)
                historical_data = tf.resample_df(ohlc, data_frequency, start=f'{hour_string}h{minute_string}min')

                # Subset the resample historical data to "train_span observations
                historical_data.tail(train_span).to_csv(historical_data_address)
        
        except:
            print('Resample of historical minute data as per the data frequency is in process...')
            # Resample the data as per the trading frequency
            ohlc = _resolve_ohlc_for_resample(historical_minute_data)
            historical_data = tf.resample_df(ohlc, data_frequency, start=f'{hour_string}h{minute_string}min')
  
            # Subset the resample historical data to "train_span observations
            historical_data.tail(train_span).to_csv(historical_data_address)
    
    print('Resample of historical minute data as per the data frequency is completed...')
    
def _ibkr_bar_size(data_frequency):
    """Map a strategy frequency string to the IBKR barSizeSetting.
    Returns the IBKR bar size string (e.g. '5 mins', '1 day')."""
    freq = str(data_frequency).strip().upper()
    if freq in ('1D', '1 D', '1DAY', 'DAILY', 'DAY'):
        return '1 day'
    try:
        n, unit = tf.get_data_frequency_values(data_frequency)
    except Exception:
        return '1 min'  # safe fallback
    if unit == 'min':
        return f'{n} min' if n == 1 else f'{n} mins'
    if unit == 'h':
        return f'{n} hour' if n == 1 else f'{n} hours'
    if unit == 'D':
        return '1 day'
    return '1 min'  # safe fallback


def run_hist_data_download_app(historical_minute_data, historical_data_address, symbol, timezone, data_frequency, update, download_span, train_span, market_open_time, date0=None, client_id=0, silent=False):
    ''' Function to download the historical data and create the resampled historical data '''
    # If there is no date
    if date0 is None:
        # Set the date to the now datetime
        date0 = datetime.now()
        
    # Set the year month and day 
    yearEnd, monthEnd, dayEnd, = date0.year, date0.month, date0.day
    
    # Extract variables from main.py to find the asset spec
    variables = tf.extract_variables('main.py')
    
    # Identify asset class and build spec
    symbol_upper = str(symbol).upper()
    asset_spec = None
    
    if symbol_upper in [s.upper() for s in variables.get('fx_pairs', [])]:
        asset_spec = {
            "symbol": symbol_upper, "asset_class": "forex",
            "exchange": variables.get("forex_exchange", "IDEALPRO"),
            "currency": variables.get("forex_currency", "USD"),
            "sec_type": "CASH"
        }
    elif symbol_upper in [s.upper() for s in variables.get('futures_symbols', [])]:
        asset_spec = {
            "symbol": symbol_upper, "asset_class": "futures",
            "exchange": variables.get("futures_exchange", "CME"),
            "currency": variables.get("futures_currency", "USD"),
            "sec_type": "FUT",
            "multiplier": variables.get("futures_multiplier", None),
            "contract_month": variables.get("futures_contract_month", None)
        }
    elif symbol_upper in [s.upper() for s in variables.get('metals_symbols', [])]:
        asset_spec = {
            "symbol": symbol_upper, "asset_class": "metals",
            "exchange": variables.get("metals_exchange", "IDEALPRO"),
            "currency": variables.get("metals_currency", "USD"),
            "sec_type": variables.get("metals_sec_type", "CASH")
        }
    elif symbol_upper in [s.upper() for s in variables.get('crypto_symbols', [])]:
        asset_spec = {
            "symbol": symbol_upper, "asset_class": "crypto",
            "exchange": variables.get("crypto_exchange", "PAXOS"),
            "currency": variables.get("crypto_currency", "USD"),
            "sec_type": "CRYPTO"
        }
    elif symbol_upper in [s.upper() for s in variables.get('stock_symbols', [])]:
        stock_primary_exchanges = variables.get("stock_primary_exchanges", {}) or {}
        asset_spec = {
            "symbol": symbol_upper, "asset_class": "stock",
            "exchange": variables.get("stock_exchange", "SMART"),
            "currency": variables.get("stock_currency", "USD"),
            "sec_type": "STK",
            "primary_exchange": stock_primary_exchanges.get(symbol_upper, variables.get("stock_primary_exchange", "NASDAQ")),
            "fractional_shares": bool(variables.get("stock_fractional_shares", False)),
            "quantity_step": float(variables.get("stock_default_quantity_step", 0.0001 if bool(variables.get("stock_fractional_shares", False)) else 1.0)),
            "tick_size": float(variables.get("stock_tick_size", 0.01)),
        }
    
    if asset_spec is None:
        print(f"Warning: Symbol {symbol} not found in main.py universe. Defaulting to Forex.")
        contract = ibf.ForexContract(symbol)
    else:
        # Create a contract object using the generic builder
        contract = ibf.build_contract_from_spec(asset_spec)
        # For futures without a contract month, compute the front month
        asset_class = str(asset_spec.get('asset_class', '')).lower() if asset_spec else ''
        if asset_class == 'futures' and not getattr(contract, 'lastTradeDateOrContractMonth', ''):
            quarterly = [3, 6, 9, 12]
            cm = next((m for m in quarterly if m >= date0.month), quarterly[0])
            yr = date0.year if cm >= date0.month else date0.year + 1
            contract.lastTradeDateOrContractMonth = f"{yr}{cm:02d}"
            if not getattr(contract, 'multiplier', ''):
                contract.multiplier = '5'
            print(f"[{symbol}] Auto-resolved front month: {contract.lastTradeDateOrContractMonth}")
        
    # Set the now datetime with the 23:59:00 time
    now = datetime(yearEnd,monthEnd,dayEnd,23,59,00)
    # Get the Saturdays list (from 2005 to now — typically ~1000 entries)
    saturdays = tf.saturdays_list(now.date())
    # Truncate to only cover the download_span period so the download completes
    # in minutes rather than hours.  Parse span days (e.g. '10 D' → 10).
    span_days = int(''.join(c for c in str(download_span) if c.isdigit()) or '10')
    # saturdays is sorted most-recent-first; keep enough to cover span_days + 1 week buffer
    needed = max(2, span_days // 7 + 2)
    saturdays = saturdays[:needed]
    
    # Run the download app — futures use TRADES (IB doesn't support BID/ASK for futures)
    asset_class = str(asset_spec.get('asset_class', '')).lower() if asset_spec else ''
    wts = ['TRADES'] if asset_class == 'futures' else None  # None → default BID+ASK
    
    # Determine bar size from asset frequency — download at the strategy's declared
    # frequency so we stream 5-20× fewer bars than the old 1-min-for-everything approach.
    is_daily = str(data_frequency).strip().upper() in ('1D', '1 D', '1DAY', 'DAILY', 'DAY')
    bar_size = _ibkr_bar_size(data_frequency)
    
    # For daily bars, use a longer span (2 years) and skip Saturday logic
    if is_daily:
        app_span = '2 Y'
        app_saturdays = saturdays[:1]  # single Saturday, just to initialise params
    else:
        app_span = download_span
        app_saturdays = saturdays
    
    app_for_download_data('127.0.0.1', client_id, historical_minute_data, update, contract, now, app_span, timezone, app_saturdays, what_to_show=wts, silent=silent, bar_size=bar_size)
    
    # Skip resample if requesting at the target frequency (daily→daily needs no resample)
    if is_daily:
        import shutil
        raw_bars = len(pd.read_csv(historical_minute_data, index_col=0))
        if raw_bars > 0:
            shutil.copy(historical_minute_data, historical_data_address)
            print(f'[{symbol}] Daily bars saved directly — {raw_bars} bars.')
        else:
            print(f'[{symbol}] Daily download returned 0 bars — keeping existing historical data (if any).')
    else:
        update_historical_resampled_data(historical_minute_data, historical_data_address, train_span, data_frequency, market_open_time)
    
