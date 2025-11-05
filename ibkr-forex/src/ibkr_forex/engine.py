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
import pickle
import inspect
import logging
import numpy as np
import datetime as dt
from threading import Thread
from ibkr_forex import create_database as cd
import strategy as stra
from ibkr_forex import trading_functions as tf
from ibkr_forex import setup_functions as sf
from ibkr_forex import setup_for_download_data as sdd
from ibkr_forex.setup import trading_app

now_ = dt.datetime.now()

# Set the month string to save the log file
if now_.month < 10:
    month = '0'+str(now_.month)
else:
    month = now_.month
# Set the day string to save the log file
if now_.day < 10:
    day = '0'+str(now_.day)
else:
    day = now_.day
# Set the hour string to save the log file
if now_.hour < 10:
    hour = '0'+str(now_.hour)
else:
    hour = now_.hour
# Set the minute string to save the log file
if now_.minute < 10:
    minute = '0'+str(now_.minute)
else:
    minute = now_.minute
# Set the second string to save the log file
if now_.second < 10:
    second = '0'+str(now_.second)
else:
    second = now_.second

# Save all the trading app info in the following log file
logging.basicConfig(filename=f'data/log/log_file_{now_.year}_{month}_{day}_{hour}_{minute}_{second}.log',
                    level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Function to run the app each period
def run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  
            historical_data_address, base_df_address, data_frequency, 
            trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail):
    
    print('='*100)
    print('='*100)
    print('='*100)
    logging.info('='*100)
    logging.info('='*100)

    print('Running the app...wish you the best!')
    logging.info('Running the app...wish you the best!')

    # Get the previous, current and next trading periods
    previous_period, current_period, next_period = tf.get_the_closest_periods(dt.datetime.now(), data_frequency, trading_day_end_datetime, previous_day_start_datetime, day_start_datetime, market_close_time)
    
    # A while loop to run the app, we will break the loop whenever we finish running the app for the current period
    while True:
        # Create an object of the app class
        app = trading_app(logging, account, account_currency, symbol, timezone, data_frequency, historical_data_address, base_df_address,  
                          market_open_time, market_close_time, 
                          previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, current_period, previous_period, next_period, train_span, test_span, trail)
                
        # Connect the app to the IB server
        print('Connecting the app to the IB server...')
        logging.info('Connecting the app to the IB server...')
        app.connect(host=host, port=port, clientId=client_id)
        
        # Set the app thread as the main one
        thread1 = Thread(target=app.run, daemon=True)
    
        # Start the app
        thread1.start()
            
        # Wait until the app is successfully connected
        time.sleep(5)
        
        print('='*100)
        print(f'Current period is {current_period}')
        logging.info(f'Current period is {current_period}')
        print(f'Trading day end datetime is {trading_day_end_datetime}')
        logging.info(f'Trading day end datetime is {trading_day_end_datetime}')
        print('='*100)
        
        # If now is before the market close datetime
        if dt.datetime.now() < market_close_time:
    
            # If now is before the trading day end datetime
            if dt.datetime.now() < trading_day_end_datetime:
                
                # If the current period hasn't been traded
                if app.periods_traded.loc[app.periods_traded['trade_time']==current_period]['trade_done'].values[0] == 0:
                    
                    # If the strategy time spent is filled
                    if app.previous_time_spent > 0:
                        if app.previous_time_spent >= (next_period - current_period).total_seconds():
                            app.previous_time_spent = 60
                        # If the previous time spent is less than the seconds left until the next trading period
                        if app.previous_time_spent < (next_period - dt.datetime.now()).total_seconds():
                            # Run the strategy, create the signal, and send orders if necessary
                            sf.run_strategy_for_the_period(app)
                            # If the strategy was successfully done
                            if app.strategy_end:
                                # Wait until we arrive at the next trading period
                                time.sleep(0 if (next_period-dt.datetime.now()).total_seconds()<0 else (next_period-dt.datetime.now()).total_seconds())
                                break
                            else:
                                # Couldn't connect to IB server, we'll try once again
                                print("Couldn't connect to the IB server, could be due to internet issues or the TWS/IB Gateway is not opened...")
                                logging.info("Couldn't connect to the IB server, could be due to internet issues or the TWS/IB Gateway is not opened...")
                        else:
                            print("Time up to the next period is not sufficient to run the strategy for the current period...")
                            logging.info("Time up to the next period is not sufficient to run the strategy for the current period...")
                            # Wait until we arrive at the next trading period
                            sf.wait_for_next_period(app)
                            time.sleep(0 if (next_period-dt.datetime.now()).total_seconds()<0 else (next_period-dt.datetime.now()).total_seconds())
                            break
                      
                    # If the strategy time spent is not fille, i.e., it's the first time we trade
                    else:
                        # Run the strategy, create the signal, and send orders if necessary
                        sf.run_strategy_for_the_period(app)
                        # If the strategy was successfully done
                        if app.strategy_end:
                            # Wait until we arrive at the next trading period
                            time.sleep(0 if (next_period-dt.datetime.now()).total_seconds()<0 else (next_period-dt.datetime.now()).total_seconds())
                            break
                        else:
                            # Couldn't connect to IB server, we'll try once again
                            print("Couldn't connect to the IB server, could be due to internet issues or the TWS/IB Gateway is not opened...")
                            logging.info("Couldn't connect to the IB server, could be due to internet issues or the TWS/IB Gateway is not opened...")
                
                # If the current period has already been traded
                else:
                    print("The current period has already been traded. Let's wait for the next period...")
                    logging.info("The current period has already been traded. Let's wait for the next period...")
                    # Wait until we arrive at the next trading period
                    sf.wait_for_next_period(app)
                    break
            # If now is after the trading day end datetime
            else:
                print("The trading end datetime has arrived. Let's close the existing position if exists and update the trading info...")
                logging.info("The trading end datetime has arrived. Let's close the existing position if exists and update the trading info...")
                # If the current period hasn't been traded
                if app.periods_traded.loc[app.periods_traded['trade_time']==trading_day_end_datetime]['trade_done'].values[0] == 0:
                    # Update the trading information and close the position if needed before the market closes
                    sf.update_and_close_positions(app)
                else:
                    print("The last position was already closed and the trading info was already updated...")
                    logging.info("The last position was already closed and the trading info was already updated...")
                    
                # Wait until we arrive at the next trading period
                print("Let's wait until the new trading day begins...")
                logging.info("Let's wait until the new trading day begins...")
                time.sleep(0 if (day_start_datetime-dt.datetime.now()).total_seconds()<0 else (day_start_datetime-dt.datetime.now()).total_seconds())
                break
        # If now is after the market close datetime
        else:   
            print('The market has closed...')       
            logging.info('The market has closed...')       
            break
                
# Run the trading all inside a loop for the whole week                        
def run_trading_setup_loop(host, port, account, client_id, data_frequency, london_start_hour, local_restart_hour, timezone, now_, account_currency, symbol, 
                           historical_data_address, base_df_address, train_span, test_span, trail):  
                  
    print('='*100)
    print('='*100)
    print('='*100)
    logging.info('='*100)
    logging.info('='*100)
    logging.info('='*100)

    # Get the local timezone hours that match the Easter timezone hours
    restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour = tf.get_end_hours(timezone, london_start_hour, local_restart_hour)
    # Get the market open and close datetimes of the current week
    market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute)
    
    # Get the corresponding auto-restart and day-end datetimes to be used while trading
    auto_restart_start_datetime, auto_restart_datetime_before_end, auto_restart_end_datetime, \
        day_start_datetime, day_datetime_before_end, trading_day_end_datetime, day_end_datetime, previous_day_start_datetime = \
            tf.get_restart_and_day_close_datetimes(data_frequency, dt.datetime.now(), day_end_hour, day_end_minute, restart_hour, restart_minute, trading_start_hour)

    print(f'market open time is {market_open_time}')
    logging.info(f'market open time is {market_open_time}')
    print(f'market close time is {market_close_time}')
    logging.info(f'market close time is {market_close_time}')
    
    print(f'\t - auto_restart_start_datetime is {auto_restart_start_datetime}')
    logging.info(f'\t - auto_restart_start_datetime is {auto_restart_start_datetime}')
    print(f'\t - auto_restart_datetime_before_end is {auto_restart_datetime_before_end}')
    logging.info(f'\t - auto_restart_datetime_before_end is {auto_restart_datetime_before_end}')
    print(f'\t - auto_restart_end_datetime is {auto_restart_end_datetime}')
    logging.info(f'\t - auto_restart_end_datetime is {auto_restart_end_datetime}')
    if dt.datetime.now()>=market_open_time:
       print(f'\t - previous_day_start_datetime is {previous_day_start_datetime}')
       logging.info(f'\t - previous_day_start_datetime is {previous_day_start_datetime}')
    print(f'\t - day_datetime_before_end is {day_datetime_before_end}')
    logging.info(f'\t - day_datetime_before_end is {day_datetime_before_end}')
    print(f'\t - trading_day_end_datetime is {trading_day_end_datetime}')
    logging.info(f'\t - trading_day_end_datetime is {trading_day_end_datetime}')
    print(f'\t - day_end_datetime is {day_end_datetime}')
    logging.info(f'\t - day_end_datetime is {day_end_datetime}')
    print(f'\t - day_start_datetime is {day_start_datetime}')
    logging.info(f'\t - day_start_datetime is {day_start_datetime}')

    # Check if now is sooner than the market opening datetime
    if dt.datetime.now() < market_open_time:
        print("Let's wait until the market opens...")
        logging.info("Let's wait until the market opens...")
        # If we are outside the week's market hours, we wait until we're in
        while dt.datetime.now() <= market_open_time: continue
    
    # Check if now is sooner than the day start datetime
    if dt.datetime.now() < previous_day_start_datetime:
        print("Let's wait until the trading day starts...")
        logging.info("Let's wait until the trading day starts...")
        # Start trading at the trading start datetime
        while dt.datetime.now() <= previous_day_start_datetime: continue
        
    # If we're inside the week's market hours
    while dt.datetime.now() >= market_open_time and dt.datetime.now() <= market_close_time:
        # Get the local timezone hours that match the Easter timezone hours
        restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour = tf.get_end_hours(timezone, london_start_hour, local_restart_hour)
        
        # Get the corresponding autorestart and day-end datetimes to be used while trading
        auto_restart_start_datetime, auto_restart_datetime_before_end, auto_restart_end_datetime, \
            day_start_datetime, day_datetime_before_end, trading_day_end_datetime, day_end_datetime, previous_day_start_datetime = \
                tf.get_restart_and_day_close_datetimes(data_frequency, dt.datetime.now(), day_end_hour, day_end_minute, restart_hour, restart_minute, trading_start_hour)
                
        # Set the highest hour
        highest_hour = restart_hour if restart_hour > day_end_hour else day_end_hour
        
        # Set the highest minute
        highest_minute = auto_restart_datetime_before_end.minute if highest_hour == restart_hour else day_end_datetime.minute
    
        # If now is sooner than the last day and the auto-restart hour
        if ((dt.datetime.now().weekday() <= (market_close_time.weekday()-1)) and (dt.datetime.now().hour < highest_hour) \
            and (dt.datetime.now().minute < highest_minute)): 
            
            # If the auto-restart datetime is sooner than the day start datetime
            if auto_restart_datetime_before_end < day_start_datetime:
                # A while loop to run the app
                while True:
                    # If now is less than the autorestart datetime
                    if (dt.datetime.now() < auto_restart_datetime_before_end):
                        # Run the app
                        run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                                    trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                    # If now is higher than the auto-restart datetime
                    else:
                        # Break the while loop
                        break
                # Wait until now is higher than auto-restart start datetime
                while (dt.datetime.now() >= auto_restart_datetime_before_end) and (dt.datetime.now() < auto_restart_start_datetime): continue
            # If the autorestart datetime is later than the day start datetime
            else:
                # A while loop to run the app
                while True:                
                    # If now is sooner than the day datetime before the day closes
                    if (dt.datetime.now() < day_datetime_before_end):
                        # Run the app
                        run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                                    trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                    # If now is later than the day datetime before the day closes
                    else:
                        # Break the while loop
                        break
                        
                # A while loop to run the app
                while True:                
                    # If now is later than the day datetime before the day closes and sooner than the trading day end datetime
                    if (dt.datetime.now() >= day_datetime_before_end) and (dt.datetime.now() < trading_day_end_datetime):
                        # Run the app
                        run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                                    trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                    # If now is later than the trading day end datetime
                    else:
                        # Break the while loop
                        break
                # A while loop to run the app
                while True:                
                    # If now is later than the trading day end datetime and sooner than the day end datetime
                    if (dt.datetime.now() >= trading_day_end_datetime) and (dt.datetime.now() < day_end_datetime):
                        # Run the app
                        run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                                    trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                    # If now is later than the day-end datetime
                    else:
                        # Break the while loop
                        break
                                        
                print("Let's wait until we start the trading day once again")
                logging.info("Let's wait until we start the trading day once again")
                while (dt.datetime.now() >= day_end_datetime) and (dt.datetime.now() < day_start_datetime): continue
            
        # If now is last day and later than the auto-restart hour                                                                                
        else:
            
            # A while loop to run the app
            while True:                
                # If now is sooner than the day datetime before the day closes
                if (dt.datetime.now() <= day_datetime_before_end):
                    # Run the app
                    run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                            trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                # If now is later than the day datetime before the day closes
                else:
                    # Break the while loop
                    break
            # A while loop to run the app
            while True:                
                # If now is later than the day datetime before the day closes and sooner than the trading day end datetime
                if (dt.datetime.now() >= day_datetime_before_end) and (dt.datetime.now() < trading_day_end_datetime): 
                    # Run the app
                    run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                            trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                # If now is later than the trading day end datetime
                else:
                    # Break the while loop
                    break
            # A while loop to run the app
            while True:                
                # If now is later than the trading day end datetime and sooner than the day end datetime
                if (dt.datetime.now() >= trading_day_end_datetime) and (dt.datetime.now() < day_end_datetime):
                    # Run the app
                    run_app(host, port, account, client_id, timezone, now_, account_currency, symbol,  historical_data_address, base_df_address, data_frequency, 
                            trading_day_end_datetime, day_end_datetime, previous_day_start_datetime, day_start_datetime, market_open_time, market_close_time, train_span, test_span, trail)
                    
                # If now is later than the day-end datetime
                else:
                    # Break the while loop
                    break
                            
            print("Let's wait until the trading week close datetime arrives")
            logging.info("Let's wait until the trading week close datetime arrives")
            while (dt.datetime.now() >= day_end_datetime) and (dt.datetime.now() < day_start_datetime): continue

def main():

    # Get the variables set in the main file (user_config/main.py)
    try:
        variables = tf.extract_variables('main.py') # Assumes main.py is in CWD relative to where engine.py is run
    except FileNotFoundError:
        print("CRITICAL ERROR: user_config/main.py not found. Please ensure it exists in the correct directory.")
        logging.critical("user_config/main.py not found. Setup cannot proceed.")
        return # Cannot proceed without main configurations

    # Essential variables with error handling
    try:
        host = variables['host']
        account = variables['account']
        timezone = variables['timezone']
        port = variables['port']
        account_currency = variables['account_currency']
        symbol = variables['symbol']
        data_frequency = variables['data_frequency']
        local_restart_hour = variables['local_restart_hour']
        # File names are expected from main.py, paths will be constructed
        historical_data_address_name = variables['historical_data_address']
        base_df_address_name = variables['base_df_address']
        train_span = variables['train_span']
        test_span_days = variables['test_span_days']
        client_id = variables['client_id']
        smtp_username = variables['smtp_username']
        to_email = variables['to_email']
        password = variables['password'] # Email password
        seed = variables['seed']
        trail = variables.get('trail', False) # Optional, defaults to False

    except KeyError as e:
        logging.error(f"Essential variable {e} not found in user_config/main.py. Please define it.")
        print(f"CRITICAL ERROR: Essential variable {e} not found in user_config/main.py. Exiting.")
        return

    # --- Path Construction ---
    # Assume 'data' and 'data/models' are subdirectories of where 'main.py' is (i.e., CWD)
    # This is a common convention for user-managed configurations.
    data_dir = 'data'
    models_dir = os.path.join(data_dir, 'models')
    log_dir = os.path.join(data_dir, 'log') # For logger path if needed, though logger already prepends data/log

    # Ensure directories exist
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir) # Ensure log directory also exists

    # Construct full paths for data files
    historical_minute_data_address = os.path.join(data_dir, f'app_{symbol}_df.csv') # Consistent naming
    # base_df_address should be constructed, main.py provides the name part
    if base_df_address_name.startswith(data_dir + os.path.sep): # If user provided 'data/file.csv'
        base_df_address = base_df_address_name
    else: # User provided 'file.csv'
        base_df_address = os.path.join(data_dir, base_df_address_name)

    # historical_data_address is for the *resampled* data file name
    if historical_data_address_name.startswith(data_dir + os.path.sep):
        resampled_historical_data_address = historical_data_address_name
    else:
        resampled_historical_data_address = os.path.join(data_dir, historical_data_address_name)


    # Set the London-timezone hour as the trading start hour
    london_start_hour = 23

    restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour = tf.get_end_hours(timezone, london_start_hour, local_restart_hour)
    market_open_time, market_close_time = tf.define_trading_week(timezone, trading_start_hour, day_end_minute)

    # Date stringing for model filenames (based on day *before* market_open_time)
    date_for_model_name = market_open_time - dt.timedelta(days=1)
    year_str_model = str(date_for_model_name.year)
    month_str_model = f"{date_for_model_name.month:02d}"
    day_str_model = f"{date_for_model_name.day:02d}"
    model_pickle_path = os.path.join(models_dir, f'stra_opt_{year_str_model}_{month_str_model}_{day_str_model}.pickle')


    test_span = test_span_days * tf.get_periods_per_day(data_frequency)
    # train_span from main.py is the one used for prepare_base_df and strategy_parameter_optimization
    # The original code `train_span += test_span` effectively meant that the `train_span` in `main.py` was the
    # span *before* adding test_span for the full optimization dataset.
    # For clarity, strategy_parameter_optimization should just receive the original train_span and test_span.

    # Prepare parameters for strategy_parameter_optimization
    stra_opt_pass_variables = variables.copy() # Start with all variables from main.py
    stra_opt_pass_variables.update({ # Add or override with derived/specific ones
        'market_open_time': market_open_time,
        'historical_minute_data_address': historical_minute_data_address, # Path to raw minute data
        'base_df_address': base_df_address, # Path for processed base_df
        'test_span': test_span,
        'train_span': train_span, # The train_span as defined in main.py
        'seed': seed,
        'data_frequency': data_frequency
        # Add other necessary parameters if strategy_parameter_optimization signature requires them
        # and they are not directly in `variables` from main.py.
    })


    try:
        signature_opt = inspect.signature(stra.strategy_parameter_optimization)
    except AttributeError:
        logging.critical("Function strategy_parameter_optimization not found in strategy.py. Cannot proceed with optimization logic.")
        return

    stra_func_params = []
    for name, param in signature_opt.parameters.items():
        if name in stra_opt_pass_variables:
            stra_func_params.append(stra_opt_pass_variables[name])
        elif param.default is not inspect.Parameter.empty:
            stra_func_params.append(param.default)
        else:
            logging.error(f"Parameter '{name}' for strategy.strategy_parameter_optimization not found or no default.")
            print(f"CRITICAL ERROR: Parameter '{name}' for strategy.strategy_parameter_optimization is missing. Exiting.")
            return

    np.random.seed(seed) # For any direct np.random use in strategy

    # Logic for historical data download and initial optimization
    if not os.path.exists(historical_minute_data_address):
        print('='*100 + "\nCreating the whole historical minute-frequency data...\n" + '='*100)
        logging.info('Creating the whole historical minute-frequency data...')
        # Note: sdd.run_hist_data_download_app's 2nd arg `historical_data_address` is for the *resampled* output name
        sdd.run_hist_data_download_app(historical_minute_data_address, resampled_historical_data_address, symbol, timezone, data_frequency, 'false', '10 D', train_span, market_open_time)

        print('='*100 + "\nOptimizing the strategy parameters (initial)...\n" + '='*100)
        logging.info('Optimizing the strategy parameters (initial)...')
        stra.strategy_parameter_optimization(*stra_func_params)
        with open(model_pickle_path, 'wb') as handle:
            pickle.dump({'stra_opt_done_for_week': True, 'date': date_for_model_name.isoformat()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Updating historical minute-frequency data...')
        logging.info('Updating historical minute-frequency data...')
        sdd.run_hist_data_download_app(historical_minute_data_address, resampled_historical_data_address, symbol, timezone, data_frequency, 'true', '10 D', train_span, market_open_time)

        if not os.path.exists(model_pickle_path):
            print('='*100 + "\nOptimizing the strategy parameters for the current week...\n" + '='*100)
            logging.info('Optimizing the strategy parameters for the current week...')
            stra.strategy_parameter_optimization(*stra_func_params)
            with open(model_pickle_path, 'wb') as handle:
                pickle.dump({'stra_opt_done_for_week': True, 'date': date_for_model_name.isoformat()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f"Strategy parameters already optimized for the week of {market_open_time.date()} (model file: {model_pickle_path}).")
            logging.info(f"Strategy parameters already optimized for week of {market_open_time.date()}.")

    # Database creation
    database_path = os.path.join(data_dir, "database.xlsx")
    if not os.path.exists(database_path):
        print('='*100 + "\nCreating the trading information database...\n" + '='*100)
        logging.info('Creating the trading information database...')
        cd.create_trading_info_workbook(smtp_username, to_email, password) # Assumes create_database writes to data/database.xlsx & data/email_info.xlsx

    print('='*100 + "\nRunning the trading app for the week...\n" + '='*100)
    logging.info('Running the trading app for the week...')
    run_trading_setup_loop(host, port, account, client_id, data_frequency, london_start_hour, local_restart_hour, timezone, dt.datetime.now(), account_currency, symbol,
                           resampled_historical_data_address, base_df_address, train_span, 1, trail) # test_span=1 for live trading

    # Post-week activities
    print('='*100 + "\nUpdating minute-frequency and resampled data post-trading week...\n" + '='*100)
    logging.info('Updating minute-frequency and resampled data post-trading week...')
    sdd.run_hist_data_download_app(historical_minute_data_address, resampled_historical_data_address, symbol, timezone, data_frequency, 'true', '10 D', train_span, market_open_time)

    # Prepare for next week's optimization check
    next_week_market_open_time, _ = tf.define_trading_week(timezone, trading_start_hour, day_end_minute, base_date=market_close_time + dt.timedelta(days=1))
    date_for_next_model_name = next_week_market_open_time - dt.timedelta(days=1)
    next_year_str = str(date_for_next_model_name.year)
    next_month_str = f"{date_for_next_model_name.month:02d}"
    next_day_str = f"{date_for_next_model_name.day:02d}"
    next_week_model_pickle_path = os.path.join(models_dir, f'stra_opt_{next_year_str}_{next_month_str}_{next_day_str}.pickle')

    if not os.path.exists(next_week_model_pickle_path):
        print('='*100 + "\nOptimizing strategy parameters for the upcoming week...\n" + '='*100)
        logging.info('Optimizing strategy parameters for the upcoming week...')

        stra_opt_pass_variables_next_week = stra_opt_pass_variables.copy()
        stra_opt_pass_variables_next_week['market_open_time'] = next_week_market_open_time

        stra_func_params_next_week = []
        for name, param in signature_opt.parameters.items():
            if name in stra_opt_pass_variables_next_week:
                stra_func_params_next_week.append(stra_opt_pass_variables_next_week[name])
            elif param.default is not inspect.Parameter.empty:
                stra_func_params_next_week.append(param.default)
            else:
                logging.error(f"Parameter '{name}' for next week's strategy.strategy_parameter_optimization not found.")
                print(f"CRITICAL ERROR: Parameter '{name}' for next week's optimization missing. Exiting.")
                return

        stra.strategy_parameter_optimization(*stra_func_params_next_week)
        with open(next_week_model_pickle_path, 'wb') as handle:
            pickle.dump({'stra_opt_done_for_week': True, 'date': date_for_next_model_name.isoformat()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
         print(f"Strategy parameters already optimized for upcoming week of {next_week_market_open_time.date()}.")
         logging.info(f"Strategy parameters already optimized for upcoming week of {next_week_market_open_time.date()}.")


main()
