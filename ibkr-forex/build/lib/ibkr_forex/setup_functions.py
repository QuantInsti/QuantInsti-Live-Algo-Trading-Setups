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
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import strategy as stra
from ibkr_forex import trading_functions as tf
from ibkr_forex import ib_functions as ibf
from concurrent.futures import ThreadPoolExecutor
from ibapi.order_cancel import OrderCancel

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
        
def request_orders(app):
    ''' Function to request the open orders and orders status'''
    print('Requesting open positions and orders status...')
    app.logging.info('Requesting open positions and orders status...')
    
    # If the app is connected
    if app.isConnected():
        # Clear the threading event
        app.orders_request_event.clear()
        # Request the open orders and orders status
        app.reqOpenOrders()
        # Make the event end when the request is done
        app.orders_request_event.wait()
    else:
        return
    
    # If the temporary open orders dataframe is not empty
    if (app.temp_open_orders.empty == False):
        # Set the datetime column as index
        app.temp_open_orders.set_index('datetime',inplace=True)
        # Erase the index name
        app.temp_open_orders.index.name = ''
        # Concatenate the temporary dataframe with the whole corresponding dataframe
        app.open_orders = pd.concat([app.open_orders,app.temp_open_orders])
        # Drop row duplicates
        app.open_orders.drop_duplicates(inplace=True)
        # Sort the whole dataframe by index
        app.open_orders.sort_index(ascending=True, inplace=True)
        # Clear the temporary dataframe
        app.temp_open_orders = pd.DataFrame()

    # If the temporary orders status dataframe is not empty
    if (app.temp_orders_status.empty == False):
        # Set the datetime column as index
        app.temp_orders_status.set_index('datetime',inplace=True)
        # Erase the index name
        app.temp_orders_status.index.name = ''
        # Concatenate the temporary dataframe with the whole corresponding dataframe
        app.orders_status = pd.concat([app.orders_status,app.temp_orders_status])
        # Drop row duplicates            
        app.orders_status.drop_duplicates(inplace=True)
        # Sort the whole dataframe by index
        app.orders_status.sort_index(ascending=True, inplace=True)
        # Clear the temporary dataframe
        app.temp_orders_status = pd.DataFrame()

    print('Open positions and orders status successfully requested...')
    app.logging.info('Open positions and orders status successfully requested...')

def request_positions(app):
    ''' Function to request the trading positions'''
    print('Requesting positions...')
    app.logging.info('Requesting positions...')
    
    # If the app is connected
    if app.isConnected():
        # Clear the threading event
        app.positions_request_event.clear()
        # Request the trading positions
        app.reqPositions()
        # Set the event to wait until the request is finished
        app.positions_request_event.wait()
    else:
        return
    
    # If the temporary positions dataframe is not empty
    if (app.temp_pos_df.empty == False):
        pd.set_option('display.max_columns', None)
        # Set the datetime column as index
        app.temp_pos_df.set_index('datetime',inplace=True)
        # Erase the index name
        app.temp_pos_df.index.name = ''
        # Concatenate the temporary dataframe with the main dataframe
        app.pos_df = pd.concat([app.pos_df, app.temp_pos_df])
        # Drop duplicates
        app.pos_df.drop_duplicates(inplace=True)
        # Sort the positions dataframe by index
        app.pos_df.sort_index(ascending=True, inplace=True)
        # Clear the temporary dataframe
        app.temp_pos_df = pd.DataFrame()
    print('Open positions successfully requested...')
    app.logging.info('Open positions successfully requested...')

def download_hist_data(app, params):
    """Function to download the historical data"""
    # Set the function inputs as per the params list
    hist_id, duration, candle_size, whatToShow = params[0], params[1], params[2], params[3]
    
    # If the app is connected
    if app.isConnected():
        # Clear the threading event
        app.hist_data_events[f'{hist_id}'].clear()
        
        # Downlod the data
        app.reqHistoricalData(reqId=hist_id, 
                               contract=app.contract,
                               endDateTime='',
                               durationStr=duration,
                               barSizeSetting=candle_size,
                               whatToShow=whatToShow,
                               useRTH=False,
                               formatDate=1,
                               keepUpToDate=False,
                               # EClient function to request contract details
                               chartOptions=[])	
        
        # Make the event wait until the download is finished
        app.hist_data_events[f'{hist_id}'].wait()
    else:
        # Return the function in case we couldn't download the data
        return

def prepare_downloaded_data(app, params):
    print(f'preparing the {params[-1]} data...')
    app.logging.info(f'preparing {params[-1]} the data...')
    
    # Rename the downloaded historical data columns
    app.new_df[f'{params[0]}'].rename(columns={'open':f'{params[-1].lower()}_open','high':f'{params[-1].lower()}_high',\
                                              'low':f'{params[-1].lower()}_low','close':f'{params[-1].lower()}_close'},inplace=True)
    
    # Set the index to datetime type            
    app.new_df[f'{params[0]}'].index = pd.to_datetime(app.new_df[f'{params[0]}'].index, format='%Y%m%d %H:%M:%S %Z')
    # Get rid of the timezone tag
    app.new_df[f'{params[0]}'].index = app.new_df[f'{params[0]}'].index.tz_localize(None)
    
    print(f'{params[-1]} data is prepared...')
    app.logging.info(f'{params[-1]} data is prepared...')
       
def update_hist_data(app):
    ''' Request the historical data '''
    
    print("Requesting the historical data...")
    app.logging.info("Requesting the historical data...")
    
    # Set the number of days that have passed from the last historical data datetime up to the current period
    days_passed_number = ((app.current_period - app.historical_data.index[-1]) + dt.timedelta(days=1)).days
    # Set the days to be used to download the historical data
    days_passed = f'{days_passed_number if days_passed_number>1 else (days_passed_number+2)} D'
    
    # Set the params list to download the data
    params_list = [[0, days_passed, '1 min', 'BID'], [1, days_passed, '1 min', 'ASK']]
    
    # If the app is connected
    if app.isConnected():
        # Download the historical BID and ASK data
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(download_hist_data, [app]*len(params_list), params_list)) 
        
    else:
        return

    # If the app is connected
    if app.isConnected():
        # Prepare the data
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(prepare_downloaded_data, [app]*len(params_list), params_list)) 

    else:
        return    
        
    # Concatenate the BID and ASK data
    df = pd.concat([app.new_df['0'],app.new_df['1']], axis=1)
    
    # Get the mid prices based on the BID and ASK prices
    df = tf.get_mid_series(df)
    
    # Set the hour string to resample the data
    hour_string = str(app.market_open_time.hour) if (app.market_open_time.hour)>=10 else '0'+str(app.market_open_time.hour)
    minute_string = str(app.market_open_time.minute) if (app.market_open_time.minute)>=10 else '0'+str(app.market_open_time.minute)

    # Resample the data as per the data frequency
    df = tf.resample_df(df, frequency=app.data_frequency, start=f'{hour_string}h{minute_string}min')

    # Concatenate the current historical dataframe with the whole one
    app.historical_data = pd.concat([app.historical_data, df])
    # Sor the historical data by index
    app.historical_data.sort_index(inplace=True)
    # Drop duplicates
    app.historical_data = app.historical_data[~app.historical_data.index.duplicated(keep='last')]
    
    print("Historical data was successfully prepared...")
    app.logging.info("Historical data was successfully prepared...")

def update_asset_last_value(app):
    ''' Request the update of the last value of the asset'''
    print("Updating the last value of the asset...")
    app.logging.info("Updating the last value of the asset...")
    # Set the last value to zero
    app.last_value = 0
    # Use the while loop in case the app has issues while requesting the last value
    while True:
        # Reques the last value of the asset price
        app.reqTickByTickData(0, app.contract, \
                                'MidPoint', 0, True)
        time.sleep(2)
        # Cancel the request
        app.cancelTickByTickData(0)
        time.sleep(1)
        # Check if the app tried more than 50 times
        if app.last_value_count >= 50:
            print("The app couldn't get the midpoint data, it will restart...")
            app.logging.info("The app couldn't get the midpoint data, it will restart...")
            break
        # Check if the last value is different from zero
        if app.last_value != 0:
            print('Midpoint data obtained...')
            app.logging.info('Midpoint data obtained...')
            break
        # Check if the app is disconnected
        if not app.isConnected(): return
                
        print("Couldn't get Tick midpoint data, it will try again...")
        app.logging.info("Couldn't get Tick midpoint data, it will try again...")
        
        # Update the last value count
        app.last_value_count += 1

def get_capital_as_per_forex_base_currency(app, capital_datetime):

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
    
    # If the app is connected
    if app.isConnected():
        # Clear the threading event
        app.account_update_event.clear()
        # Request the account update of app.account
        app.reqAccountUpdates(True,app.account)
        # Wait until the update is finished
        app.account_update_event.wait()
        # Cancel the request
        app.reqAccountUpdates(False,app.account)
        time.sleep(1)
        print('Account values successfully updated ......')
        app.logging.info('Account values successfully requested...')
    else:
        return
    
    # Set the cash balance datetime
    capital_datetime = \
        app.acc_update[(app.acc_update['key']=='TotalCashBalance') & \
                        (app.acc_update['Currency']=='BASE') ]['datetime'].tail(1).values[0]
            
    # Save the cash balance value
    app.cash_balance.loc[capital_datetime, 'value'] = \
        float(app.acc_update[(app.acc_update['key']=='TotalCashBalance') & \
                        (app.acc_update['Currency']=='BASE') ]['Value'].tail(1).values[0])
       
    app.capital = get_capital_as_per_forex_base_currency(app, capital_datetime)
    
    # Forward fill the cash balance dataframe
    app.cash_balance.ffill(inplace=True)
        
    print('Capital value successfully updated ...')
    app.logging.info('Capital value successfully updated ...')
    
def update_risk_management_orders(app):
    ''' Function to update the risk management orders IDs and their status'''

    print('Updating the risk management orders IDs and their status...')
    app.logging.info('Updating the risk management orders IDs and their status...')
    
    # If the open orders dataframe is not empty
    if not app.open_orders.empty:
        if app.trail:
            # Set the last stop loss order
            app.sl_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='TRAIL')]["OrderId"].sort_values(ascending=True).values[-1])
        else:
            # Set the last stop loss order
            app.sl_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='STP')]["OrderId"].sort_values(ascending=True).values[-1])
        # Set the last take profit order
        app.tp_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='LMT')]["OrderId"].sort_values(ascending=True).values[-1])
        
        # Set a boolean to True if the previous stop loss is filled or canceled
        app.sl_filled_or_canceled_bool = (app.open_orders[app.open_orders['OrderId'] == app.sl_order_id]['Status'].str.contains('canceled').sum()==1) or \
                                           (app.open_orders[app.open_orders['OrderId'] == app.sl_order_id]['Status'].str.contains('Filled').sum()==1) 
            
        # Set a boolean to True if the previous take profit is filled or canceled
        app.tp_filled_or_canceled_bool = (app.open_orders[app.open_orders['OrderId'] == app.tp_order_id]['Status'].str.contains('canceled').sum()==1) or \
                                           (app.open_orders[app.open_orders['OrderId'] == app.tp_order_id]['Status'].str.contains('Filled').sum()==1) 

    else:
        # Set the last stop loss order to NaN
        app.sl_order_id = np.nan
        # Set the last take profit order to NaN
        app.tp_order_id = np.nan            

        # Set a boolean to False if the previous stop loss is not filled or canceled
        app.sl_filled_or_canceled_bool = False
        # Set a boolean to False if the previous take profit is not filled or canceled
        app.tp_filled_or_canceled_bool = False
    
    print('The risk management orders IDs and their status were successfully updated...')
    app.logging.info('The risk management orders IDs and their status were successfully updated...')
    
def update_remaining_position_based_on_risk_management(app, risk_management_threshold):
    ''' Function to update the remaining position and cash balance based on the selected risk management threshold'''
    
    # If the risk management selected is the stop-loss order
    if risk_management_threshold == 'sl':
        # If the previous stop loss order is filled or canceled
        if app.sl_filled_or_canceled_bool == True:
            
            # Set the remaining position value from the orders status dataframe
            remaining = float(app.orders_status[app.orders_status['OrderId'] == app.sl_order_id]['Remaining'].values[-1])
            # Set the position remaining datetime
            remaining_datetime = app.orders_status[app.orders_status['OrderId'] == app.sl_order_id].index.values[-1]
            
            # Set the average traded price from the position
            average_price = pd.to_numeric(app.exec_df[app.exec_df['OrderId'] == app.sl_order_id]['AvPrice'].values[-1])
            
            # Create a new row for the positions dataframe with the remaining datetime
            app.pos_df.loc[remaining_datetime,:] = app.pos_df[(app.pos_df['Symbol']==app.contract.symbol) & 
                                                                (app.pos_df['Currency']==app.contract.currency)].iloc[-1,:]
            # Save the last position value in the positions dataframe
            app.pos_df.loc[remaining_datetime,'Position'] = remaining
            # Save the last average cost in the positions dataframe
            app.pos_df.loc[remaining_datetime,'Avg cost'] = average_price
            
            # Update the leverage value in the cash balance dataframe
            app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'leverage'] = app.leverage
            # Update the signal value in the cash balance dataframe
            app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'signal'] = 0     
            # Forward fill the cash balance dataframe
            app.cash_balance.ffill(inplace=True)
        
    # If the risk management selected is the take-profit order
    elif risk_management_threshold == 'tp':
        # If the previous take profit order is filled or canceled
        if app.tp_filled_or_canceled_bool == True:
            
            # Set the remaining position value from the orders status dataframe
            remaining = float(app.orders_status[app.orders_status['OrderId'] == app.tp_order_id]['Remaining'].values[-1])
            # Set the position remaining datetime
            remaining_datetime = app.orders_status[app.orders_status['OrderId'] == app.tp_order_id].index.values[-1]
            
            # Set the average traded price from the position
            average_price = pd.to_numeric(app.exec_df[app.exec_df['OrderId'] == app.tp_order_id]['AvPrice'].values[-1])
            
            # Create a new row for the positions dataframe with the remaining datetime
            app.pos_df.loc[remaining_datetime,:] = app.pos_df[(app.pos_df['Symbol']==app.contract.symbol) & 
                                                                (app.pos_df['Currency']==app.contract.currency)].iloc[-1,:]
            # Save the last position value in the positions dataframe
            app.pos_df.loc[remaining_datetime,'Position'] = remaining
            # Save the last average cost in the positions dataframe
            app.pos_df.loc[remaining_datetime,'Avg cost'] = average_price
        
            # Update the leverage value in the cash balance dataframe
            app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'leverage'] = app.leverage
            # Update the signal value in the cash balance dataframe
            app.cash_balance.loc[dt.datetime.now().replace(microsecond=0), 'signal'] = 0     
            # Forward fill the cash balance dataframe
            app.cash_balance.ffill(inplace=True)
  
def update_submitted_orders(app):
    ''' Function to update the submitted orders'''
    
    print('Updating the submitted orders ...')
    app.logging.info('Updating the submitted orders ...')
    
    # If it is our first trade period of the week
    if len(app.periods_traded[app.periods_traded['trade_time']>app.market_open_time].index)==1:
        # Set the last trade period 
        last_trade_time = app.previous_period.strftime(f'%Y%m%d %H:%M:%S {app.zone}')
    # If we have already traded more than one time
    else:
        # Set the last trade period as the previous period traded
        last_trade_time = app.periods_traded[app.periods_traded.trade_done == 1]['trade_time'].iloc[-1].strftime(f'%Y%m%d %H:%M:%S {app.zone}')
    
    # If the app is connected
    if app.isConnected():
        print('Requesting executions...')
        app.logging.info('Requesting executions...')
        # Clear the threading event
        app.executions_request_event.clear()
        # Request the previous trading executions and commissions
        app.reqExecutions(0, ibf.executionFilter(last_trade_time))
        # Make the event wait until the request is done
        app.executions_request_event.wait()
        
        print('Successfully requested execution and commissions details...')
        app.logging.info('Successfully requested execution and commissions details...')
    else:
        return
    
    # If the app is connected
    if app.isConnected():
        # Update the risk management orders IDs and their status
        update_risk_management_orders(app)  
    
        # If the temporal executions dataframe is not empty
        if (app.temp_exec_df.empty == False):  
            # Get rid of the time zone in the execution time column values
            app.temp_exec_df['Execution Time'] = \
                pd.to_datetime(app.temp_exec_df['Execution Time'].replace(rf"{ app.zone}", "", regex=True).values)
            # Set the datetime as index
            app.temp_exec_df.set_index('datetime',inplace=True)
            # Erase the index name
            app.temp_exec_df.index.name = ''
            # Concatenate the temporary dataframe with the main dataframe
            app.exec_df = pd.concat([app.exec_df,app.temp_exec_df])
            # Drop duplicates
            app.exec_df.drop_duplicates(inplace=True)
            # Sort the dataframe by index
            app.exec_df.sort_index(ascending=True, inplace=True)
            # Clear the temporary dataframe
            app.temp_exec_df = pd.DataFrame()
            
            # Set the datetime as index
            app.temp_comm_df.set_index('datetime',inplace=True)
            # Erase the index name
            app.temp_comm_df.index.name = ''
            # Convert to NaN Realized PnL whose values are extremely high (due to IB mistake value)
            mask = app.temp_comm_df['Realized PnL'].astype(float) == 1.7976931348623157e+308
            app.temp_comm_df.loc[mask,'Realized PnL'] = np.nan
            # Concatenate the temporary dataframe with the main dataframe
            app.comm_df = pd.concat([app.comm_df,app.temp_comm_df])
            # Drop duplicates
            app.comm_df.drop_duplicates(inplace=True)
            # Sor the dataframe by index
            app.comm_df.sort_index(ascending=True, inplace=True)
            # Clear the temporary dataframe
            app.temp_comm_df = pd.DataFrame()

            
            # If the orders status and positions dataframes are not empty
            if (app.orders_status.empty == False) and (app.pos_df.empty == False):
                
                # If the previous stop-loss and take-profit target were filled or cancelled
                if (app.sl_filled_or_canceled_bool == True) and (app.tp_filled_or_canceled_bool == True):
                    # Set the stop-loss order execution time
                    sl_order_execution_time = dt.datetime.strptime(app.exec_df[app.exec_df['OrderId'] == app.sl_order_id]['Execution Time'].values[-1], '%Y-%m-%d %H:%M:%S')
                    # Set the take-profit order execution time
                    tp_order_execution_time = dt.datetime.strptime(app.exec_df[app.exec_df['OrderId'] == app.sl_order_id]['Execution Time'].values[-1], '%Y-%m-%d %H:%M:%S')
                    
                    # If the stop-loss execution time is later than than the take-profit execution time
                    if sl_order_execution_time > tp_order_execution_time:
                        # Update the remaining position based on the previous stop-loss order
                        update_remaining_position_based_on_risk_management(app, 'sl')
                    else:
                        # Update the remaining position based on the previous take-profit order
                        update_remaining_position_based_on_risk_management(app, 'tp')
                        
                # If the previous stop loss order is filled or canceled
                elif app.sl_filled_or_canceled_bool == True:                    
                    # Update the remaining position based on the previous stop-loss order
                    update_remaining_position_based_on_risk_management(app, 'sl')
                    
                # If the previous take profit order is filled or canceled
                elif app.tp_filled_or_canceled_bool == True:                       
                    # Update the remaining position based on the previous take-profit order
                    update_remaining_position_based_on_risk_management(app, 'tp')
                    
    print('The submitted orders were successfully updated...')
    app.logging.info('The submitted orders were successfully updated...')
    
def portfolio_allocation(app): 
    ''' Function to update the portfolio allocation'''

    print('Make the portfolio allocation ...')
    app.logging.info('Make the portfolio allocation ...')
    
    # If the app is connected
    if app.isConnected():
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
    
    # If the previous position sign is different from the current signal
    if (app.previous_quantity!=0) and (np.sign(app.previous_quantity)==app.signal) and (app.open_orders.empty==False):
        if app.trail:
            # Set a new stop-loss target price
            order_price = stra.set_stop_loss_price(app)
        else:
            # Set the previous stop-loss target price
            order_price = app.open_orders[app.open_orders["OrderId"]==app.sl_order_id]["AuxPrice"].values[-1]
        # Convert the quantity to an integer value
        quantity = int(abs(app.previous_quantity))
    # If they're equal
    else:
        # Set a new stop-loss target price
        order_price = stra.set_stop_loss_price(app)
        # Convert the quantity to an integer value
        quantity = int(abs(quantity))
   
    # If the signal tells you to go long
    if app.signal > 0:
        # Set the stop-loss direction to sell the position
        direction = 'SELL'
    # If the signal tells you to short-sell the asset
    elif app.signal < 0:
        # Set the stop-loss direction to buy the position
        direction = 'BUY'

    # Set the add decimal to zero
    add = 0.0
    # If the add value is less than or equal to 0.0001
    while add<=0.00010:
        # Send the stop-loss order
        app.placeOrder(order_id, app.contract, ibf.stopOrder(direction, quantity, order_price, app.trail))
        time.sleep(3)
        # Save the output errors in data as a boolean that corresponds to any error while sending the stop-loss order
        data = (321 in list(app.errors_dict.keys())) or \
                (110 in list(app.errors_dict.keys())) or \
                (463 in list(app.errors_dict.keys()))
        # If data is true
        if data == True:
            # Add 0.00001 to add
            add += 0.00001
            # Set the order price to 5 decimals
            order_price = round(order_price+add,5)
            
            print("Couldn't transmit the stop-loss order, the app will try again...")
            app.logging.info("Couldn't transmit the-stop loss order, the app will try again...")
            
            # Clean the errors dictionary
            app.errors_dict = {}
        else:
            print(f'Stop loss sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.logging.info(f'Stop loss sent with direction {direction}, quantity {quantity}, order price {order_price}')
            break
        # If the app is disconnected
        if 504 in list(app.errors_dict.keys()):
            break
        
def send_take_profit_order(app, order_id, quantity): 
    ''' Function to send a take profit order
        - The function has a while loop to incorporate the fact that sometimes
          the order is not sent due to decimal errors'''
    
    # If the previous position sign is different from the current signal
    if (app.previous_quantity!=0) and (np.sign(app.previous_quantity)==app.signal) and (app.open_orders.empty==False):
        # Set the previous take-profit target price
        order_price = app.open_orders[app.open_orders["OrderId"]==app.tp_order_id]["LmtPrice"].values[-1]
        # Convert the quantity to an integer value
        # quantity = int(abs(app.previous_quantity))
        quantity = int(abs(app.previous_quantity))
            
    # If they're equal
    else:
        # Set the take-profit target price
        order_price = stra.set_take_profit_price(app)
        # Convert the quantity to an integer value
        quantity = int(abs(quantity))

    # If the signal tells you to go long
    if app.signal > 0:
        # Set the take-profit direction to sell the position
        direction = 'SELL'
    # If the signal tells you to short-sell the asset
    elif app.signal < 0:
        # Set the take-profit direction to buy the position
        direction = 'BUY'
        
    # Set the add decimal to zero
    add = 0.0
    # If the add value is less than or equal to 0.0001
    while add<=0.00010:
        # Send the take-profit order
        app.placeOrder(order_id, app.contract, ibf.tpOrder(direction, quantity, order_price))
        time.sleep(3)
        # Save the output errors in data as a boolean that corresponds to any error while sending the take-profit order
        data = (321 in list(app.errors_dict.keys())) or \
                (110 in list(app.errors_dict.keys())) or \
                (463 in list(app.errors_dict.keys()))
        # If data is true
        if data == True:
            # Add 0.00001 to add
            add += 0.00001
            # Set the order price to 5 decimals
            order_price = round(order_price-add,5)
            
            print("Couldn't transmit the take-profit order, the app will try again...")
            app.logging.info("Couldn't transmit the take-profit order, the app will try again...")
            
            # Clean the errors dictionary
            app.errors_dict = {}
        else:
            print(f'Take profit sent with direction {direction}, quantity {quantity}, order price {order_price}')
            app.logging.info(f'Take profit sent with direction {direction}, quantity {quantity}, order price {order_price}')
            break
        # If the app is disconnected
        if 504 in list(app.errors_dict.keys()):
            return
        
def send_market_order(app, order_id, quantity):
    ''' Function to send a market order '''
    
    print('Sending the market order...')
    app.logging.info('Sending the market order...')
    
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
            app.placeOrder(order_id, app.contract, ibf.marketOrder(direction, int(abs(quantity))))                       
            time.sleep(3)                                                
            print("Market order sent...")
            app.logging.info("Market order sent...")
        else:
            return
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
            app.placeOrder(order_id, app.contract, ibf.marketOrder(direction, int(abs(quantity))))
            time.sleep(3)                                                
            print("Market order sent...")
            app.logging.info("Market order sent...")
        else:
            return
                    
def get_previous_quantity(app):
    ''' Function to get the previous position quantity'''
    
    # If the position dataframe is not empty
    if app.pos_df.empty==False:
        # Set the previous position quantity
        app.previous_quantity = app.pos_df[(app.pos_df['Symbol']==app.contract.symbol) & \
                                             (app.pos_df['Currency']==app.contract.currency)]["Position"].iloc[-1]
    # If it's empty
    else:
        # Set the previous position quantity to zero
        app.previous_quantity = 0
        
def get_current_quantity(app):
    ''' Function to get the current position quantity'''
            
    app.current_quantity = int(app.capital)

def get_previous_and_current_quantities(app):
    ''' Function to get the previous and current positions quantities'''
    
    print('Update the previous and current positions quantities...')
    app.logging.info('Update the previous and current positions quantities...')
    
    # If the app is connected
    if app.isConnected():
        # Update the portfolio allocation
        portfolio_allocation(app)
        # Update the last value of the asset
        update_asset_last_value(app)
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
            
    print('The previous and current positions quantities were successfully updated...')
    app.logging.info('The previous and current positions quantities were successfully updated...')
    
def update_trading_info(app):
    ''' Function to get the previous trading information'''

    print('Update the previous trading information...')
    app.logging.info('Update the previous trading information...')
    
    # Set the executors list
    executors_list = []
    # Append the functions to be used in parallel
    with ThreadPoolExecutor(3) as executor:
        executors_list.append(executor.submit(request_positions, app)) 
        executors_list.append(executor.submit(request_orders, app)) 
        executors_list.append(executor.submit(update_submitted_orders, app)) 

    # Run the functions in parallel
    for x in executors_list:
        x.result()
        
    
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
    
def send_orders_as_bracket(app, order_id, quantity, mkt_order, sl_order, tp_order, rm_quantity=None):
    ''' Function to send the orders as a bracket'''
    
    # Send a market and risk management orders
    if (mkt_order==True) and (sl_order==True) and (tp_order==True):
        send_market_order(app, order_id, quantity)
        if rm_quantity is None:
            send_stop_loss_order(app, order_id+1, quantity)
            send_take_profit_order(app, order_id+2, quantity)
        else:
            send_stop_loss_order(app, order_id+1, rm_quantity)
            send_take_profit_order(app, order_id+2, rm_quantity)
            
    # Send only the risk management orders
    elif (mkt_order==False) and (sl_order==True) and (tp_order==True):
        send_stop_loss_order(app, order_id, quantity)
        send_take_profit_order(app, order_id+1, quantity)
    # Send only the market order
    elif (mkt_order==True) and (sl_order==False) and (tp_order==False):
        send_market_order(app, order_id, quantity)
    else:
        pass
    
def set_new_and_rm_orders_quantities(app):
    
    # Set the signal
    signal = app.signal
    
    # Set the new quantity for the current market order
    if app.previous_leverage != 0:
        new_quantity = (app.leverage - app.previous_leverage)*app.previous_quantity/app.previous_leverage
    else:
        new_quantity = app.current_quantity

    if new_quantity < 0:
        new_quantity = math.floor(abs(new_quantity))
        if app.previous_leverage != 0:
            rm_quantity = int(app.previous_quantity - new_quantity)
        else:
            rm_quantity = None
        signal = -1.0
    elif new_quantity > 0:
        new_quantity = math.floor(abs(new_quantity))
        if app.previous_leverage != 0:
            rm_quantity = int(app.previous_quantity + new_quantity)
        else:
            rm_quantity = None
    else:
        new_quantity = app.previous_quantity
        rm_quantity = None
    
    return signal, new_quantity, rm_quantity 
                
def send_orders(app):
    ''' Function to send the orders if needed'''

    print('Sending the corresponding orders if needed...')
    app.logging.info('Sending the corresponding orders if needed...')
    
    if len(app.cash_balance.loc[:, 'leverage'].index) != 0:
        app.previous_leverage = app.cash_balance['leverage'].iloc[-1]
        app.previous_signal = app.cash_balance['signal'].iloc[-1]
    else:
        app.previous_leverage = 0.0
        app.previous_signal = 0.0
        
    # Update the previous trading information
    update_trading_info(app)  
    # Update the previous and current positions quantities
    get_previous_and_current_quantities(app)

    # Set an initial order id
    order_id = 0
    # If the app is connected
    if app.isConnected():
        # Update the order id
        app.reqIds(-1)
        time.sleep(2)        
        # Save the new order id in order_id
        order_id = app.nextValidOrderId
    else:
        return
    
    print('='*50)
    print('='*50)
    print(f'previous quantity is {app.previous_quantity}')
    print(f'previous signal is {app.previous_signal}')
    print(f'signal is {app.signal}')
    print(f'previous leverage is {app.previous_leverage}')
    print(f'leverage is {app.leverage}')
    print(f'current quantity is {app.signal*app.current_quantity}')
    print('='*50)
    print('='*50)
        
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
                
            new_quantity = int(abs(app.previous_quantity) + app.current_quantity)

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
                        
            new_quantity = int(abs(app.previous_quantity) + app.current_quantity)
    
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
                # Buy the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.current_quantity, True, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A new position was just opened together with new risk management orders...')
            app.logging.info('A new position was just opened together with new risk management orders...')
        
        # Update the signal and leverage values in the cash balance dataframe
        update_cash_balance_values_for_signals(app)
            
        # Update the trading information
        update_trading_info(app)  
                    
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
                
            new_quantity = int(abs(app.previous_quantity) + app.current_quantity)
    
            print(f'new quantity is {new_quantity}')
    
            # Set the executors list
            executors_list = []
            # Append the functions to be used in parallel
            with ThreadPoolExecutor(2) as executor:
                # Cancel the previous risk management orders
                executors_list.append(executor.submit(cancel_risk_management_previous_orders, app))
                # Short-sell the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, new_quantity, True, True, True, int(app.current_quantity)))
    
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
                        
            new_quantity = int(abs(app.previous_quantity) + app.current_quantity)
    
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
                # Buy the asset and send the risk management orders
                executors_list.append(executor.submit(send_orders_as_bracket, app, order_id, app.current_quantity, True, True, True))
    
            # Run the functions in parallel
            for x in executors_list:
                x.result()
    
            print('A new position was just opened together with new risk management orders...')
            app.logging.info('A new position was just opened together with new risk management orders...')

        # Update the signal and leverage values in the cash balance dataframe
        update_cash_balance_values_for_signals(app)
            
        # Update the trading information
        update_trading_info(app)  
                        
def strategy(app):
    ''' Function to get the strategy run'''

    print('Running the strategy for the period...')
    app.logging.info('Running the strategy for the period...')

    # Set a default dataframe
    base_df = pd.DataFrame()

    # Get the variables set in the main file (user_config/main.py)
    # Assumes main.py is in CWD or an otherwise accessible path for tf.extract_variables
    try:
        variables = tf.extract_variables('main.py')
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
        return_variables_prepare = tf.get_return_variable_names("strategy.py", "prepare_base_df")
    except FileNotFoundError:
        app.logging.error("strategy.py not found. Cannot inspect prepare_base_df.")
        print("Error: user_config/strategy.py not found.")
        return
    except AttributeError: # If prepare_base_df is not in strategy.py
        app.logging.error("Function prepare_base_df not found in strategy.py.")
        print("Error: Function prepare_base_df not found in user_config/strategy.py.")
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
                    err_msg = "'base_df' not found in return values of strategy.prepare_base_df. Check strategy.py."
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
                err_msg = "'base_df' not found in return values of strategy.prepare_base_df. Check strategy.py."
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
            return_variables_signal = tf.get_return_variable_names("strategy.py", "get_signal")
        except FileNotFoundError:
             app.logging.error("strategy.py not found. Cannot inspect get_signal.")
             return
        except AttributeError:
             app.logging.error("Function get_signal not found in strategy.py.")
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

        if 'signal' not in return_variables_signal:
             err_msg = "'signal' not found in return values of strategy.get_signal. Check strategy.py."
             app.logging.error(err_msg)
             print(f"Error: {err_msg}")
             return
        app.signal = results_signal[return_variables_signal.index('signal')]

        if 'leverage' in return_variables_signal:
            app.leverage = results_signal[return_variables_signal.index('leverage')]
        elif 'leverage' in effective_vars and effective_vars['leverage'] is not None:
            app.leverage = effective_vars['leverage'] # From main.py or app default
        elif app.leverage is None: # If not set by strategy and not in main.py (or was None)
             app.leverage = 1.0 # Fallback default

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
    
def save_week_open_and_close_datetimes(app):
    """ Function to fill all the dataframes with the week's open and close datetimes"""
    
    print("Saving the corresponding week's open and close datetimes in the corresponding dataframes...")
    app.logging.info("Saving the corresponding week's open and close datetimes in the corresponding dataframes...")
    
    # A for loop to iterate through each of the corresponding dataframes
    for dataframe in [app.open_orders, app.orders_status, app.exec_df, app.comm_df, \
                      app.pos_df, app.cash_balance]:
        # Get the rows which correspond to the week's datetimes
        mask = (dataframe.index>=app.market_open_time) & (dataframe.index<=app.market_close_time)
        # Set the corresponding market open time in each dataframe
        dataframe.loc[mask,'market_open_time'] = app.market_open_time
        # Set the corresponding market close time in each dataframe
        dataframe.loc[mask,'market_close_time'] = app.market_close_time
        # Drop duplicates if needed based only on the columns
        dataframe = dataframe[dataframe.duplicated(subset=dataframe.columns)]
        
    # Get the rows which correspond to the week's datetimes in the periods_traded dataframe
    mask = (app.periods_traded['trade_time']>=app.market_open_time) & (app.periods_traded['trade_time']<=app.market_close_time)
    # Set the corresponding market open time in the dataframe
    app.periods_traded.loc[mask,'market_open_time'] = app.market_open_time
    # Set the corresponding market close time in the dataframe
    app.periods_traded.loc[mask,'market_close_time'] = app.market_close_time

    print("The corresponding week's open and close datetimes were successfully added on the dataframes...")
    app.logging.info("The corresponding week's open and close datetimes were successfully added on the dataframes...")
    
def save_data(app):
    """ Function to save the data"""
    
    print("Saving all the data...")
    app.logging.info("Saving all the data...")
    
    # Forward-fill the cash balance dataframe values
    app.cash_balance.ffill(inplace=True)
    
    # Save the open and close market datetimes in all dataframes
    save_week_open_and_close_datetimes(app)

    # Group all the dataframes in a single dictionary
    dictfiles = {'open_orders':app.open_orders,\
                 'orders_status':app.orders_status,\
                 'executions':app.exec_df,\
                 'commissions':app.comm_df,\
                 'positions':app.pos_df,\
                 'cash_balance':app.cash_balance,\
                 'app_time_spent':app.app_time_spent,\
                 'periods_traded':app.periods_traded}
         
    # Save the dataframes in a single Excel workbook
    tf.save_xlsx(dict_df = dictfiles, path = 'data/database.xlsx')

    # Save the historical data
    app.historical_data.to_csv(app.historical_data_address)
    
    print("All data saved...")
    app.logging.info("All data saved...")

def save_data_and_send_email(app):
    """ Function to save the data and send email"""
    
    print("Saving the data and sending the email...")
    app.logging.info("Saving the data and sending the email...")
    
    # If the app is connected
    if app.isConnected():
        # Set the executors list
        executors_list = []
        # Append the functions to be used in parallel
        with ThreadPoolExecutor(2) as executor:
            # Save the data
            executors_list.append(executor.submit(save_data, app))
            # Send the email
            executors_list.append(executor.submit(send_email, app))

        # Run the functions in parallel
        for x in executors_list:
            x.result()

        print("The data was saved successfully...")
        app.logging.info("The data was saved successfully...")
    
def run_strategy(app):
    """ Function to run the whole strategy, including the signal"""

    print("Running the strategy, the signal and sending the orders if necessary...")
    app.logging.info("Running the strategy, the signal and sending the orders if necessary...")
    
    # Run the strategy
    strategy(app)
    
    # If the app is connected
    if app.isConnected():
        # Send the orders
        send_orders(app)
        print("The strategy, the signal and sending the orders were successfully run...")
        app.logging.info("The strategy, the signal and sending the orders were successfully run...")
    
        
    # Save the total seconds spent while trading in each period
    app.app_time_spent['seconds'].iloc[0] = (dt.datetime.now() - app.app_start_time).total_seconds() + 3
    
    # Set the current period as traded
    app.periods_traded['trade_done'].iloc[-1] = 1
    
    save_data_and_send_email(app)

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

    print('Update the trading info and closing the position...')
    app.logging.info('Update the trading info and closing the position...')
    
    # Update the trading info        
    update_trading_info(app)  
    
    # Cancel the previous risk management orders
    cancel_risk_management_previous_orders(app)                        

    # Signal and leverage are zero at the end of the day
    app.signal = app.leverage = 0
    
    # Get the previous and current quantities
    get_previous_and_current_quantities(app)
    
    # If the app is connected
    if app.isConnected():
        # Update the order id
        app.reqIds(-1)
        time.sleep(2)        
        order_id = app.nextValidOrderId
    
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
    app.periods_traded['trade_done'].iloc[-1] = 1

    # Save the data and send the email
    save_data_and_send_email(app)
    
    print('The trading info was updated and the position was closed successfully...')
    app.logging.info('The trading info was updated and the position was closed successfully...')
    
    if (app.next_period != app.market_close_time):
        print("Let's wait for the next trading day to start...")
        app.logging.info("Let's wait the next trading day to start...")
    else:
        print("Let's wait for the market to get closed...")
        app.logging.info("Let's wait for the market to get closed...")
        
    # Disconnect the app
    stop(app)
    
    # Wait until we arrive at the next trading period
    time.sleep(0 if (app.next_period-dt.datetime.now()).total_seconds()<0 else (app.next_period-dt.datetime.now()).total_seconds())

def send_email(app): 
    """ Function to send an email with relevant information of the trading current period"""

    if (app.open_orders.empty==False) and (app.orders_status.empty==False):
        
        try:
            # Get the market id 
            mkt_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='MKT')]["OrderId"].sort_values(ascending=True).values[-1])
            if app.trail:
                # Get the trailing stop loss id 
                sl_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='TRAIL')]["OrderId"].sort_values(ascending=True).values[-1])
            else:
                # Get the stop loss id 
                sl_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='STP')]["OrderId"].sort_values(ascending=True).values[-1])
            # Get the take profit id 
            tp_order_id = int(app.open_orders[(app.open_orders["Symbol"]==app.contract.symbol) & (app.open_orders["OrderType"]=='LMT')]["OrderId"].sort_values(ascending=True).values[-1])
            
            # Get the market order price
            market_order_price = float(app.orders_status[(app.orders_status['OrderId'] == mkt_order_id) & (app.orders_status['Status'] == 'Filled')]['AvgFillPrice'].sort_values(ascending=True).values[-1])
            # Get the stop loss price
            sl_order_price = float(app.open_orders[app.open_orders['OrderId'] == sl_order_id]['AuxPrice'].sort_values(ascending=True).values[-1])
            # Get the take profit price
            tp_order_price = float(app.open_orders[app.open_orders['OrderId'] == tp_order_id]['LmtPrice'].sort_values(ascending=True).values[-1])
                 
            # Import the email dataframe
            email_password = pd.read_excel('data/email_info.xlsx', index_col = 0)
            
            # Set the Gmail server
            smtp_server = 'smtp.gmail.com'
            # Set the Gmail port
            smtp_port = 587
            # Set the trader's email
            smtp_username = email_password['smtp_username'].iloc[0]
            # Set the trader's email password
            smtp_password = email_password['password'].iloc[0]
            
            # Set the trader's email to send the email
            from_email = email_password['smtp_username'].iloc[0]
            # Set the email to which we'll send the current period trading information
            to_email = email_password['to_email'].iloc[0]
                
            # Email subject
            subject = 'EPAT Trading App Status'
            # Body of the email per line
            body0 = f'- The period {app.current_period} was successfully traded'
            body1 = f'- The Forex pair is {app.ticker}'
            body2 = f'- The signal is {app.signal}'
            body3 = f'- The leverage is {app.leverage}'
            body4 = f"- The cash balance value is {round(app.cash_balance['value'].values[-1],2)} {app.account_currency}"
            body5 = f'- The current position quantity is {app.current_quantity} {app.contract.symbol}'
            body6 = f'- The stop-loss price is {sl_order_price}'
            body7 = f'- The market price is {market_order_price}'
            body8 = f'- The take-profit price is {tp_order_price}'
            
            # Concatenate the email message
            message = f'Subject: {subject}\n\n{body0}\n\n{body1}\n{body2}\n{body3}\n{body4}\n{body5}\n{body6}\n{body7}\n{body8}'
            
            # Send the email
            with smtplib.SMTP(smtp_server, smtp_port) as smtp:
                smtp.starttls()
                smtp.login(smtp_username, smtp_password)
                smtp.sendmail(from_email, to_email, message)
                
            print("The email was sent successfully...")
            app.logging.info("The email was sent successfully...")
    
        except:
            # Email subject
            subject = 'EPAT Trading app Status'
            # Body of the email per line
            body0 = f'- The period {app.current_period} was successfully traded'
            
            # Concatenate the email message
            message = f'Subject: {subject}\n\n{body0}'
            
            print("The email was sent successfully...")
            app.logging.info("The email was sent successfully...")
     
# Disconnect the app
def stop(app):
    print('Disconnecting...')
    app.disconnect()
