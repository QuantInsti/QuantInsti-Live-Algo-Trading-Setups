"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

# Import the necessary libraries
import time
import numpy as np
import pandas as pd
from threading import Event
from threading import Thread
from datetime import datetime
from concurrent import futures
from ibkr_forex import trading_functions as tf
from ibapi.client import EClient
from ibkr_forex import ib_functions as ibf
from ibapi.wrapper import EWrapper

import warnings
warnings.filterwarnings('ignore')

##########################################################################
    
class app_for_download_data(EWrapper, EClient):
    ''' Serves as the client and the wrapper '''
    def __init__(self, addr, client_id, file_name, update = False, \
                 contract = {}, now = '', download_span = '1 D', timezone= 'US/Eastern', \
                 saturdays = list()):
        
        EClient.__init__(self, self)
        
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
        
        # Create the params list with the BID and ASK quotes
        j = 0
        # A loop to iterate through each Saturday
        for date0 in self.saturdays:
            # Append a list with the BID quote
            self.params_list.append([j, self.contract, date0, 'BID'])
            # Append a list with the ASK quote
            self.params_list.append([j+1, self.contract, date0, 'ASK'])
            # Create an empty dataframe corresponding to the BID params list
            self.dfs[f'{j}'] = pd.DataFrame()
            # Create an empty dataframe corresponding to the ASK params list
            self.dfs[f'{j+1}'] = pd.DataFrame()
            # Update the params list iterator
            j+=2
            
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
            
    def error(self, reqId, code, msg, advancedOrderRejectJson=''):
        ''' Called if an error occurs '''
        
        # Save tje ,essage
        self.errors_code_dict[code] = msg
        # Print only the following messages
        if (code != 502 or code != 504 or \
            code != 2103 or code != 2104  or code != 2105 or \
            code != 2106 or code != 2107 or code != 2157 or code != 2158):
            print('Error {}: {}'.format(code, msg))
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
            
    def InitiateAlgorithm(self, activate = False):
        ''' Function to initiate the historical data download algorithm '''
        
        # If the app was successfully launched
        if activate == True:
            time.sleep(3)
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
        self.reqHistoricalData(params[0], params[1], params[2], self.span, '1 min', \
                               params[3], False, 1, False, [])
        # Make the event to wait until the download is completed
        self.events[f'{params[0]}'].wait()
        
    def update_df(self, params_list):
        ''' Function to update the whole historical dataframe '''
        
        # Set the last param list number
        last_params_num = params_list[-1][0]
        # Set the first param list number
        j = params_list[0][0]
        # Iterate through each params list
        while j <= last_params_num:
            # Change the BID dataframe columns' names
            bid_columns = ['bid_'+column for column in self.dfs[f'{j}'].columns.tolist()]
            self.dfs[f'{j}'].columns = bid_columns
                
            # try:
            #     # Set the index to datetime type            
            #     self.dfs[f'{j}'].index = pd.to_datetime(self.dfs[f'{j}'].index, format='%Y%m%d %H:%M:%S %Z')
            #     # Get rid of the timezone tag
            #     self.dfs[f'{j}'].index = self.dfs[f'{j}'].index.tz_localize(None)
            # except:
            #     # Set the index to datetime type
            #     self.dfs[f'{j}'].index = pd.to_datetime(self.dfs[f'{j}'].index, format='%Y%m%d  %H:%M:%S')
            # Set the index to datetime type            
            self.dfs[f'{j}'].index = pd.to_datetime(self.dfs[f'{j}'].index, format='%Y%m%d %H:%M:%S %Z')
            # Get rid of the timezone tag
            self.dfs[f'{j}'].index = self.dfs[f'{j}'].index.tz_localize(None)
            
            # Change the ASK dataframe columns' names
            ask_columns = ['ask_'+column for column in self.dfs[f'{j+1}'].columns.tolist()]
            self.dfs[f'{j+1}'].columns = ask_columns

            # try:
            #     # Set the index to datetime type            
            #     self.dfs[f'{j+1}'].index = pd.to_datetime(self.dfs[f'{j+1}'].index, format='%Y%m%d %H:%M:%S %Z')
            #     # Get rid of the timezone tag
            #     self.dfs[f'{j+1}'].index = self.dfs[f'{j+1}'].index.tz_localize(None)
            # except:
            #     # Set the index to datetime type            
            #     self.dfs[f'{j+1}'].index = pd.to_datetime(self.dfs[f'{j+1}'].index, format='%Y%m%d  %H:%M:%S')
            # Set the index to datetime type            
            self.dfs[f'{j+1}'].index = pd.to_datetime(self.dfs[f'{j+1}'].index, format='%Y%m%d %H:%M:%S %Z')
            # Get rid of the timezone tag
            self.dfs[f'{j+1}'].index = self.dfs[f'{j+1}'].index.tz_localize(None)
            
            # Concatenate the BID and ASK dataframes into a single one
            temp_df = pd.concat([self.dfs[f'{j}'],self.dfs[f'{j+1}']], axis=1)
            # Update the whole historical dataframe
            self.end_df = pd.concat([self.end_df,temp_df])
            # Sort the dataframe by index
            self.end_df = self.end_df.sort_index()
            # Drop duplicates
            self.end_df = self.end_df[~self.end_df.index.duplicated(keep='first')]
            # Save the historical dataframe into a CSV file
            self.end_df.to_csv(self.file_name, encoding='utf-8', index=True)
            # Update the params list number
            j+=2

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
def update_historical_resampled_data(historical_minute_data, historical_data_address, train_span, data_frequency, market_open_time):
    
    # Set the hour string to resample the data
    hour_string = str(market_open_time.hour) if (market_open_time.hour)>=10 else '0'+str(market_open_time.hour)
    minute_string = str(market_open_time.minute) if (market_open_time.minute)>=10 else '0'+str(market_open_time.minute)

    # If the historical minute data variable is a dataframe
    if isinstance(historical_minute_data, pd.DataFrame):
        try:
            historical_resampled_data = pd.read_csv('data/'+historical_data_address, index_col=0)
            historical_resampled_data.index = pd.to_datetime(historical_resampled_data.index)
            if (historical_minute_data.index[-1].day==historical_resampled_data.index[-1].day) or \
                (historical_minute_data.index[-1].day+1==historical_resampled_data.index[-1].day):
                print('Resampling was already done...')
            else:
                print('Resample of historical minute data as per the data frequency is in process...')
                # Resample the data as per the trading frequency
                historical_data = tf.resample_df(tf.get_mid_series(historical_minute_data), data_frequency, start=f'{hour_string}h{minute_string}min')

                # Subset the resample historical data to "train_span observations
                historical_data.tail(train_span).to_csv(historical_data_address)
    
        except:

            # Resample the data as per the trading frequency
            historical_data = tf.resample_df(tf.get_mid_series(historical_minute_data), data_frequency, start=f'{hour_string}h{minute_string}min')

            # Subset the resample historical data to "train_span observations
            historical_data.tail(train_span).to_csv(historical_data_address)
    
    # If it's a string address
    else:
        # Import the historical minute-frequency data
        historical_minute_data = pd.read_csv(historical_minute_data, index_col=0)
        # Convert the index to datetime type
        historical_minute_data.index = pd.to_datetime(historical_minute_data.index)

        try:
            historical_resampled_data = pd.read_csv('data/'+historical_data_address, index_col=0)
            historical_resampled_data.index = pd.to_datetime(historical_resampled_data.index)
            if (historical_minute_data.index[-1].day==historical_resampled_data.index[-1].day) or \
                (historical_minute_data.index[-1].day+1==historical_resampled_data.index[-1].day):
                print('Resampling was already done...')
            else:
                print('Resample of historical minute data as per the data frequency is in process...')
                # Resample the data as per the trading frequency
                historical_data = tf.resample_df(tf.get_mid_series(historical_minute_data), data_frequency, start=f'{hour_string}h{minute_string}min')

                # Subset the resample historical data to "train_span observations
                historical_data.tail(train_span).to_csv(historical_data_address)
        
        except:
            print('Resample of historical minute data as per the data frequency is in process...')
            # Resample the data as per the trading frequency
            historical_data = tf.resample_df(tf.get_mid_series(historical_minute_data), data_frequency, start=f'{hour_string}h{minute_string}min')
  
            # Subset the resample historical data to "train_span observations
            historical_data.tail(train_span).to_csv(historical_data_address)
    
    print('Resample of historical minute data as per the data frequency is completed...')
    
def run_hist_data_download_app(historical_minute_data, historical_data_address, symbol, timezone, data_frequency, update, download_span, train_span, market_open_time, date0=None):
    ''' Function to download the historical data and create the resampled historical data '''
    # If there is no date
    if date0 is None:
        # Set the date to the now datetime
        date0 = datetime.now()
        
    # Set the year month and day 
    yearEnd, monthEnd, dayEnd, = date0.year, date0.month, date0.day
    # Create a forex contract object
    contract = ibf.ForexContract(symbol)
    # Set the now datetime with the 23:59:00 time
    now = datetime(yearEnd,monthEnd,dayEnd,23,59,00)
    # Get the Saturdays list
    saturdays = tf.saturdays_list(now.date())
    
    # Run the download app
    app_for_download_data('127.0.0.1', 0, historical_minute_data, update, contract, now, download_span, timezone, saturdays)
    
    # Resample the data to the trading frequency
    update_historical_resampled_data(historical_minute_data, historical_data_address, train_span, data_frequency, market_open_time)
    
