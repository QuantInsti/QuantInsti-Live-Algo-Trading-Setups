"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

# Import the necessary libraries
import ast
import pytz
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from lightgbm import LGBMClassifier
from shaphypetune import BoostBoruta

def dropLabels(events,minPct=.05):
    """
    Removes under-represented classes from the target variable to address class imbalance.

    Iteratively drops the least frequent class in the 'y' column until:
    1. All remaining classes meet the minimum percentage threshold (minPct), OR
    2. Only two classes remain

    Args:
        events (pd.DataFrame): Input dataframe containing the 'y' target column
        minPct (float, optional): Minimum acceptable percentage for any class (0.05 = 5%)

    Returns:
        pd.DataFrame: Filtered dataframe with balanced classes in 'y' column

    Notes:
        - Operates on a copy of the input dataframe to avoid modifying original data
        - Preserves row indices after filtering
        - Particularly useful for ML models sensitive to class imbalance
        - Final classes will always have ≥ minPct representation or be reduced to 2 classes

    Example:
        If 'y' has classes distributed as [70%, 25%, 5%] with minPct=0.05:
        - First iteration: drops 5% class
        - Result: [73.7%, 26.3%] distribution
    """
    ''' Function to drop the lowest-percentage prediction feature class'''

    # apply weights, drop labels with insufficient examples
    while True:
        # Count the total number of observations per each prediction feature class
        df0=events['y'].value_counts(normalize=True)
        # If the class with the minimum number of observations or if there is only 2 prediction feature labels, then finish the loop
        if (df0.min()>minPct) or (df0.shape[0]<3):break
        # Drop the prediction feature label which has the lowest number of observations
        events = events[events['y']!=df0.index[df0.argmin()]]
    return events

def get_mid_series(dfraw):
    """
    Calculates mid-prices from bid/ask OHLC data for forex trading applications.

    Args:
        dfraw (pd.DataFrame): Raw dataframe containing bid/ask columns:
            - Format: bid_close, ask_close, bid_high, ask_high, etc.

    Returns:
        pd.DataFrame: Processed dataframe with:
            - Midpoint OHLC: (bid_price + ask_price)/2 for Close/High/Low/Open
            - Sorted chronological index
            - NaN-free values

    Notes:
        - Handles missing values via forward-filling before calculation
        - Ensures temporal order with ascending index sort
        - Removes residual NaN values post-calculation
        - Maintains original datetime index alignment
        - Designed for tick/quote data common in forex markets
    """

    # Copy the dataframe 
    dfraw = dfraw.copy(deep=True)
    # Set the OHLC names list
    prices = ['Close','High','Low','Open',]
    # Forward-fill the dataframe just in case
    dfraw.ffill(inplace=True)
    # Create a new dataframe with the previous dataframe index
    df = pd.DataFrame(index=dfraw.index)
    # Looping against the prices' names
    for price in prices:
        # Get the midpoint of each price
        df[f'{price}'] = ( dfraw[f'bid_{price.lower()}'] + dfraw[f'ask_{price.lower()}'] ) /2        
    # Sort the dataframe by index
    df.sort_index(ascending=True, inplace=True)
    # Drop NaN values just in case
    df.dropna(inplace=True)
    return df

def resample_df(dfraw,frequency,start='00h00min'):
    """
    Resamples OHLC data to specified frequency while preserving price extremes and their timestamps.

    Args:
        dfraw (pd.DataFrame): Raw OHLC data with datetime index
        frequency (str): Resampling frequency (e.g., '15min', '4h')
        start (str): Anchor time for first bar ('HHhMMmin' format)

    Returns:
        pd.DataFrame: Resampled bars with:
            - OHLC prices (first Open, last Close, max High, min Low)
            - Timestamps of price extremes (High_time, Low_time)
            - high_first flag (True if High occurred before Low in period)
            - Forward-projected last bar to avoid look-ahead bias

    Notes:
        - Aligns first bar to first occurrence of `start` time in data
        - Maintains temporal sequence with ascending index
        - Adds metadata for market structure analysis:
            * High_time/Low_time: Exact timing of price extremes
            * high_first: Indicates if High preceded Low in period
        - Handles final incomplete bar by forward projection
        - Designed for session-based trading strategies
    """

    # Validate frequency input
    if not isinstance(frequency, str) or \
       not (('h' in frequency and frequency[:-1].isdigit() and len(frequency) > 1) or \
            ('min' in frequency and frequency[:-3].isdigit() and len(frequency) > 3)):
        raise ValueError(f"Frequency string '{frequency}' is malformed. Expected format like '<number>h' or '<number>min'.")

    # Copy the dataframe
    df = dfraw.copy()
    # Get the start hour
    hour=int(start[0:2])
    # Get the start minute time
    minutes=int(start[3:5])

    # Set the first day of the new dataframe
    origin_candidates = df[(df.index.hour==hour) & (df.index.minute==minutes)]
    if origin_candidates.empty:
        # Attempt to find the earliest available time if the exact start is not present,
        # or raise an error if strict start time adherence is required.
        # For now, raising an error as per original intent.
        raise ValueError(f"Start time {start} not found in dataframe index. Consider checking data availability or adjusting start time.")
    origin = origin_candidates.index[0]

    # Subset the dataframe from the origin onwards
    df = df[df.index>=origin]

    # Create a datetime column based on the index
    df['datetime'] = df.index
    # Create a new dataframe
    df2 = (df.groupby(pd.Grouper(freq=frequency, origin=df.index[0]))
           # Resample the Open price
            .agg(Open=('Open','first'),
                 # Resample the Close price
                 Close=('Close','last'),
                 # Resample the High price
                 High=('High','max'),
                 # Resample the Low price
                 Low=('Low','min'),
                 # Get the High-price index
                 High_time=('High', lambda x : pd.NaT if x.empty or x.isnull().all() else x.idxmax()),
                 # Get the Low-price index
                 Low_time=('Low', lambda x : pd.NaT if x.empty or x.isnull().all() else x.idxmin()),
                 # Get the Open-price index
                 Open_time=('datetime','first'),
                 # Get the Close-price index
                 Close_time=('datetime','last'))
            # Create a column and set each row to True in case the high price index is sooner than the low price index
            .assign(high_first = lambda x: x["High_time"] < x["Low_time"])
            )

    final_df = df2.shift(1)

    if not df2.empty: # Ensure df2 is not empty before trying to access its last index
        if 'h' in frequency:
            final_df.loc[df2.index[-1]+dt.timedelta(hours=int(frequency[:frequency.find("h")])),:] = df2.loc[df2.index[-1],:]
        else: # Assumes 'min' if not 'h' based on prior validation
            final_df.loc[df2.index[-1]+dt.timedelta(minutes=int(frequency[:frequency.find("min")])),:] = df2.loc[df2.index[-1],:]
    else:
        # Handle empty df2 case, perhaps log a warning or return empty final_df
        # For now, it will result in an empty final_df after dropna if df2 was empty.
        pass

    final_df.dropna(inplace=True)

    return final_df

def directional_change_events(data, theta=0.004, columns=None):
    """
    Generates Directional Change (DC) indicators as defined by Chen and Tsang (2021).

    Implements event-driven market regime detection using price thresholds to identify:
    - Peak/trough points
    - Trend magnitude (TMV)
    - Trend duration (T)
    - Time-adjusted returns (R)

    Args:
        data (pd.DataFrame): OHLC price data with 'Close' column
        theta (float, optional): Threshold percentage for DC detection (0.004 = 0.4%)
        columns (list, optional): Subset of columns to return. None returns full dataframe

    Returns:
        pd.DataFrame: Original dataframe augmented with DC indicators:
            - Event: -1 (peak), 1 (trough), 0 (no event)
            - peak_trough_prices: Last detected peak/trough price
            - count: Bars since last DC event
            - TMV: Trend Magnitude Value (price change / (theta * previous peak/trough))
            - T: Time periods between events
            - R: Log-scaled time-adjusted returns

    Notes:
        - Initializes with upward trend assumption from first Close price
        - Events trigger when prices cross theta thresholds from last extremum
        - Forward-fills non-event periods for continuous regime tracking
        - Replaces infinite values with NaNs for robustness
        - Maintains temporal alignment through pandas index
    """
    
    # Copy the dataframe
    data = data.copy()

    # Create the necessary columns and variables
    data["Event"] = 0.0

    # Set the initial event variable value
    event = "upward" # initial event

    # Set the initial value for low and high prices
    ph = data['Close'].iloc[0] # highest price
    pl = data['Close'].iloc[0] # lowest price

    # Create loop to run through each date
    for t in range(0, len(data.index)):
        # Check if we're on a downward trend
        if event == "downward":
            # Check if the close price is higher than the low price by the theta threshold
            if data["Close"].iloc[t] >= pl * (1 + theta):
                # Set the event variable to upward
                event = "upward"
                # Set the high price as the current close price                
                ph = data["Close"].iloc[t]
            # If the close price is lower than the low price by the theta threshold
            else:
                # Check if the close price is less than the low price
                if data["Close"].iloc[t] < pl:
                    # Set the low price as the current close price
                    pl = data["Close"].iloc[t]
                    # Set the Event to upward for the current period
                    data["Event"].iloc[t] = 1
        # Check if we're on an upward trend
        elif event == "upward":
            # Check if the close price is less than the high price by the theta threshold
            if data["Close"].iloc[t] <= ph * (1 - theta):  
                # Set the event variable to downward
                event = "downward"
                # Set the low price as the current close price
                pl = data["Close"].iloc[t]
            # If the close price is higher than the high price by the theta threshold
            else:
                # Check if the close price is higher than the high price
                if data["Close"].iloc[t] > ph:
                    # Set the high price as the current close price
                    ph = data["Close"].iloc[t]
                    # Set the Event to downward for the current period
                    data["Event"].iloc[t] = -1

    # Set the peak and trough prices and forward-fill the column
    data['peak_trough_prices'] = np.where(data['Event']!=0, data['Close'],0)
    data['peak_trough_prices'].replace(to_replace=0, method='ffill', inplace=True)

    # Count the number of periods between a peak and a trough
    data['count'] = 0
    for i in range(1,len(data.index)):
        if data['Event'].iloc[(i-1)]!=0:
            data['count'].iloc[i] = 1+data['count'].iloc[(i-1)]
        else:
            data['count'].iloc[i] = 1

    # Compute the TMV indicator
    data['TMV'] = np.where(data['Event']!=0, abs(data['peak_trough_prices']-data['peak_trough_prices'].shift())/\
                          (data['peak_trough_prices'].shift()*theta),0)

    # Compute the time-completion-for-a-trend indicator
    data['T'] = np.where(data['Event']!=0, data['count'],0)

    # Compute the time-adjusted-return indicator and forward-fill it
    data['R'] = np.where(data['Event']!=0, np.log(data['TMV']/data['T']*theta),0)
    data['R'] = data['R'].replace(to_replace=0, method='ffill')

    # Drop NaN or infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if columns is None:
        return data
    else:
        return data[columns]

def library_boruta_shap(X, y, seed, max_iter, date_loc):
    """
    Performs feature selection using Boruta-Shap algorithm with LightGBM integration.

    Args:
        X (pd.DataFrame): Feature matrix with datetime index
        y (pd.Series): Target variable series
        seed (int): Random seed for reproducibility
        max_iter (int): Maximum Boruta iterations for feature confirmation
        date_loc (datetime): Split point for train/validation sets

    Returns:
        list: Selected feature names passing Boruta-Shap criteria. Returns all features if:
            - No features selected
            - Algorithm fails (handles exceptions)

    Notes:
        - Uses LightGBM classifier with fixed hyperparameter grid:
            * learning_rate: [0.2, 0.1]
            * num_leaves: [25]
            * max_depth: [12]
        - Implements BoostBoruta wrapper for SHAP-based feature importance
        - Validation set uses post-date_loc data for early stopping
        - Fallback mechanism returns original features on error
        - Parallel processing enabled (n_jobs=-2)
        - Early stopping after 6 non-improving Boruta rounds
    """

    X_train, X_test = X.loc[:date_loc,:], X.loc[date_loc:,:]
    y_train, y_test = y.loc[:date_loc, 'y'].values.reshape(-1,), y.loc[date_loc:, 'y'].values.reshape(-1,)
    
    # Parameters' range of values
    param_grid = {
                    'learning_rate': [0.2, 0.1],
                    'num_leaves': [25],#, 35],
                    'max_depth': [12]
                }

    clf_lgbm = LGBMClassifier(n_estimators=20, random_state=seed, n_jobs=-2)
    
    ### HYPERPARAM TUNING WITH GRID-SEARCH + BORUTA SHAP ###
    try:
        model = BoostBoruta(clf_lgbm, param_grid=param_grid, max_iter=max_iter, perc=100,
                            importance_type='shap_importances', train_importance=False, sampling_seed=seed, n_jobs=-2, early_stopping_boruta_rounds=6, verbose=0)
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        best_features = model.support_.tolist()
        
        if len(best_features)!=0:
            return best_features
        else:
            return X.columns.tolist()
    except:
        best_features = X.columns.tolist()
        return best_features

def create_Xy(indf, feature_cols, y_target_col): 
    """
    Splits a dataframe into feature matrix (X) and target variable (y) for machine learning.

    Args:
        indf (pd.DataFrame): Source dataframe containing both features and target
        feature_cols (list): Column names to use as input features
        y_target_col (str): Column name containing target variable to predict

    Returns:
        tuple: Contains two elements:
            - X (pd.DataFrame): Feature matrix with selected columns
            - y (pd.DataFrame): Target variable as single-column dataframe

    Notes:
        - Maintains index alignment between X and y
        - Converts y to DataFrame for sklearn compatibility
        - Does not validate column existence - ensure columns exist in indf
        - Preserves original dataframe ordering
    """
    """ Function to create the input and prediction features dataframes """
    # Create the input features and prediction features dataframes
    X, y = indf[feature_cols], indf[y_target_col].to_frame()
    return X, y

def roll_zscore(x, window):
    """ Function to create the rolling zscore versions of an array """
    # Create the rolling object
    r = x.rolling(window=window)
    # Set the rolling mean
    m = r.mean().shift(1)
    # Set the rolling standard deviation
    s = r.std().shift(1)
    # Compute the zscore values
    z = (x - m) / s
    return z

def rolling_zscore_function(data, scalable_features, window):
    """
    Applies rolling z-score normalization to specified features while preserving non-scalable columns.

    Processes input features by:
    1. Calculating rolling z-scores over a defined window period
    2. Handling infinite/nan values from initial window calculations
    3. Maintaining original feature values when insufficient historical data exists
    4. Merging scaled features with non-scaled columns

    Args:
        data (pd.DataFrame): Input dataframe containing features to scale
        scalable_features (list): Column names to apply rolling z-score normalization
        window (int): Lookback window for z-score calculation (μ and σ)

    Returns:
        tuple: Contains two elements:
            - pd.DataFrame: Processed dataframe with scaled/non-scaled features
            - list: Names of successfully scaled features

    Notes:
        - Features with >window NaN values after scaling retain original values
        - Infinite values are converted to NaN before processing
        - Final output drops all rows with NaN values
        - Preserves non-scalable features from original data
        - Requires `roll_zscore` utility function (assumes standard implementation)

    Example:
        scaled_df, scaled_cols = rolling_zscore_function(df, ['volatility', 'volume'], 30)
        # Result will have z-scores for features with sufficient history, original values otherwise
    """
    """ Function to create the rolling zscore versions of the feature inputs """

    # Create a scaled X dataframe based on the data index
    X_scaled_final = pd.DataFrame(index=data.index)
    # Create the scaled X data 
    X_scaled = roll_zscore(data[scalable_features], window=window)
    # Replace the infinite values with NaN values
    X_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    # Create a scaled features list
    scaled_features = list()
    
    # Loop through the scaled X data columns
    for feature in X_scaled.columns:
        # If the number of NaN values is higher than the max window used to compute the technical indicators
        if np.isnan(X_scaled[feature]).sum()>window:
            # Save in the scaled X data the same non-scaled feature column
            X_scaled_final[feature] = data[feature].copy()
        # If the number of NaN values is lower than the max window used to compute the technical indicators
        else:
            # Save the scaled column in the final dataframe
            X_scaled_final[feature] = X_scaled[feature].copy()
            # Save the scaled X column name in the list
            scaled_features.append(feature)
            
    # Concatenate the rest of the columns not used in the X_scaled dataframe
    X_scaled_final = pd.concat([X_scaled_final, data[data.columns.difference(scalable_features)]], axis=1)
            
    # Drop the NaN values
    X_scaled_final.dropna(inplace=True)
    
    return X_scaled_final, scaled_features

def train_test_split(X, y, split, purged_window_size, embargo_period):
    """
    Splits time-series data into training/test sets with purging and embargo periods to prevent look-ahead bias.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series/pd.DataFrame): Target variable
        split (int): Number of observations for test set (last `split` records)
        purged_window_size (int): Number of initial training observations to exclude (prevows model contamination)
        embargo_period (int): Number of final training observations to exclude (post-split buffer)

    Returns:
        tuple: Contains four elements:
            - X_train: Purged/embargoed training features
            - X_test: Test features (last `split` obs)
            - y_train: Purged/embargoed training targets
            - y_test: Test targets

    Notes:
        - Maintains temporal order - test set always uses most recent data
        - Preserves pandas index alignment
        - Purge window removes early potentially unreliable data
        - Embargo creates buffer between train/test to avoid leakage
        - Designed for financial time series where sequence matters
    """

    # If the split variable is an integer
    if isinstance(split, int):
        # Get the train data
        X_train, y_train = X.iloc[:-split], y.iloc[:-split]
        # Get the test data
        X_test, y_test = pd.DataFrame(X.iloc[-split:], index=X.index[-split:]), pd.DataFrame(y.iloc[-split:], index=y.index[-split:])
        
    # The purged start iloc value
    purged_start = max(0, len(y) - (len(y) - purged_window_size))
    # The embargo start iloc value
    embargo_start = max(0, len(y) - embargo_period)
       
    # Defining once again the X train data based on the purged window and embargo period
    X_train = X_train.iloc[purged_start:embargo_start]
    # Defining once again the y train data based on the purged window and embargo period
    y_train = y_train.iloc[purged_start:embargo_start]
    
    return X_train, X_test, y_train, y_test

def define_trading_week(local_timezone, trading_start_hour, day_end_minute):
    """ Function to get the current trading week start and end datetimes """
        
    # Set the now datetime
    today = dt.datetime.now().astimezone(pytz.timezone(local_timezone))
    
    # Set the Bogota timezone string
    bog = 'America/Bogota'
    # Set the Bogota-based today's datetime (naive)
    bogota_datetime = today.astimezone(pytz.timezone(bog))
    
    # Bogota-based start datetime
    bogota_trading_start_datetime = today.replace(hour=trading_start_hour, minute=day_end_minute, second=0, microsecond=0).astimezone(pytz.timezone(bog))
    # Bogota-based start hour
    bogota_trading_start_hour = bogota_trading_start_datetime.hour
    # Bogota-based start minute
    bogota_trading_start_minute = bogota_trading_start_datetime.minute
    
    # If we're out of trading hours (e.g., after market close on Friday, or on Saturday/Sunday)
    if (bogota_datetime.weekday() == 4 and bogota_datetime.time() >= bogota_trading_start_datetime.time()) or \
       (bogota_datetime.weekday() == 5) or \
       (bogota_datetime.weekday() == 6 and bogota_datetime.time() <= bogota_trading_start_datetime.time()):
        
        # The trading week has ended, so the next one starts on the upcoming Sunday.
        sunday = bogota_datetime + dt.timedelta(days=(6 - bogota_datetime.weekday()) % 7)
        # The end of that *next* trading week will be the Friday after that Sunday.
        friday = sunday + dt.timedelta(days=5) # This is the corrected line
    # If we're within the trading hours
    else:
        # The trading week ends on the upcoming Friday.
        friday = bogota_datetime + dt.timedelta(days=(4 - bogota_datetime.weekday()) % 7)
        # The trading week started on the previous Sunday.
        sunday = bogota_datetime - dt.timedelta(days=(bogota_datetime.weekday() + 1) % 7)
        
    # Set the trading week start datetime
    week_start = dt.datetime(sunday.year, sunday.month, sunday.day, bogota_trading_start_hour, bogota_trading_start_minute, 0)
    # Set the trading week end datetime
    week_end = dt.datetime(friday.year, friday.month, friday.day, bogota_trading_start_hour, bogota_trading_start_minute, 0)
    
    # Localize the week start datetime to Bogota's timezone
    week_start = pytz.timezone(bog).localize(week_start)
    # Localize the week end datetime to Bogota's timezone
    week_end = pytz.timezone(bog).localize(week_end)
    
    # Convert the week start datetime to the trader's timezone 
    week_start = week_start.astimezone(pytz.timezone(local_timezone))
    # Convert the week end datetime to the trader's timezone 
    week_end = week_end.astimezone(pytz.timezone(local_timezone))
    
    return week_start.replace(tzinfo=None), week_end.replace(tzinfo=None) 

def save_xlsx(dict_df, path):
    """
    Function to save a dictionary of dataframes to an Excel file, with each dataframe as a separate sheet
    """
    writer = pd.ExcelWriter(path)
    for key in dict_df:
        dict_df[key].to_excel(writer, key)
    writer.close()

def get_end_hours_old(timezone, london_start_hour, local_restart_hour):
    """ Function to get the end hours based on the Eastern timezone """
    
    # Set the easter timezone string
    est = 'US/Eastern'
    # Get today's datetime
    today_datetime = dt.datetime.now()
    # Set the eastern-timezone-based today's datetime
    eastern = today_datetime.astimezone(pytz.timezone(est))
    # Get the timezone difference hour and minute
    eastern_timestamp = eastern.strftime("%z")
    # Get the eastern timezone difference sign boolean
    eastern_negative_sign_bool = eastern_timestamp.startswith("-")
    # Set the eastern timezone difference sign number
    eastern_sign = -1 if eastern_negative_sign_bool else +1
    
    # Set the trader's timezone now datetime
    trader_datetime = today_datetime.astimezone(pytz.timezone(timezone))
    # Get the timezone difference hour and minute
    trader_datetime_timestamp = trader_datetime.strftime("%z")
    # Get the trader's timezone difference sign boolean
    trader_datetime_negative_sign_bool = trader_datetime_timestamp.startswith("-")
    # Set the trader's timezone difference sign number
    trader_datetime_sign = -1 if trader_datetime_negative_sign_bool else +1
    
    try:
        # Get the number of minutes of the difference between both datetimes
        minutes = int(str(abs(trader_datetime.replace(tzinfo=None) - eastern.replace(tzinfo=None)))[2:4])
    except:
        # Get the number of minutes of the difference between both datetimes
        minutes = int(str(abs(trader_datetime.replace(tzinfo=None) - eastern.replace(tzinfo=None)))[3:5])

    # If the trader's timezone sign is different from Eastern's
    if trader_datetime_sign != eastern_sign:
        # Set the restart hour
        restart_hour = local_restart_hour + int(eastern_timestamp[1:3])+int(trader_datetime_timestamp[1:3])
        restart_hour = restart_hour if restart_hour<=23 else restart_hour - 24
        
        # Set the day-end hour
        day_end_hour = 17 + int(eastern_timestamp[1:3])+int(trader_datetime_timestamp[1:3])
        day_end_hour = day_end_hour if day_end_hour<=23 else day_end_hour - 24
        
        # Set the restart minute
        restart_minute = day_end_minute = minutes
    # If the trader's timezone sign is equal to Eastern's
    else:
        # Set the restart hour
        restart_hour = local_restart_hour + int(eastern_timestamp[1:3])-int(trader_datetime_timestamp[1:3])
        restart_hour = restart_hour if restart_hour<=23 else restart_hour - 24
        
        # Set the day-end hour
        day_end_hour = 17 + int(eastern_timestamp[1:3])-int(trader_datetime_timestamp[1:3])
        day_end_hour = day_end_hour if day_end_hour<=23 else day_end_hour - 24
        
        # Set the restart minute
        restart_minute = day_end_minute = minutes
      
    # Set the trading start hour
    trading_start_hour = london_start_hour + trader_datetime_sign*int(trader_datetime_timestamp[1:3])
    trading_start_hour = trading_start_hour if trading_start_hour<=23 else (trading_start_hour - 24)
                    
    return restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour

def get_end_hours(timezone, london_start_hour, local_restart_hour):
    """ Function to get the end hours based on the Eastern timezone """

    # Set the easter timezone string
    est = 'US/Eastern'
    # Get today's datetime
    today_datetime = dt.datetime.now()
    # Set the eastern-timezone-based today's datetime
    eastern = today_datetime.astimezone(pytz.timezone(est))
    # Get the timezone difference hour and minute
    eastern_timestamp = eastern.strftime("%z")
    # Get the eastern timezone difference sign boolean
    eastern_negative_sign_bool = eastern_timestamp.startswith("-")
    # Set the eastern timezone difference sign number
    eastern_sign = -1 if eastern_negative_sign_bool else +1

    # Set the trader's timezone now datetime
    trader_datetime = today_datetime.astimezone(pytz.timezone(timezone))
    # Get the timezone difference hour and minute
    trader_datetime_timestamp = trader_datetime.strftime("%z")
    # Get the trader's timezone difference sign boolean
    trader_datetime_negative_sign_bool = trader_datetime_timestamp.startswith("-")
    # Set the trader's timezone difference sign number
    trader_datetime_sign = -1 if trader_datetime_negative_sign_bool else +1

    # Calculate minute difference robustly
    time_difference_seconds = abs((trader_datetime.replace(tzinfo=None) - eastern.replace(tzinfo=None)).total_seconds())
    minutes = int((time_difference_seconds % 3600) // 60)

    # If the trader's timezone sign is different from Eastern's
    if trader_datetime_sign != eastern_sign:
        # Set the restart hour
        restart_hour = local_restart_hour + int(eastern_timestamp[1:3])+int(trader_datetime_timestamp[1:3])
        restart_hour = restart_hour if restart_hour<=23 else restart_hour - 24

        # Set the day-end hour
        day_end_hour = 17 + int(eastern_timestamp[1:3])+int(trader_datetime_timestamp[1:3])
        day_end_hour = day_end_hour if day_end_hour<=23 else day_end_hour - 24

        # Set the restart minute
        restart_minute = day_end_minute = minutes
    # If the trader's timezone sign is equal to Eastern's
    else:
        # Set the restart hour
        restart_hour = local_restart_hour + int(eastern_timestamp[1:3])-int(trader_datetime_timestamp[1:3])
        restart_hour = restart_hour if restart_hour<=23 else restart_hour - 24

        # Set the day-end hour
        day_end_hour = 17 + int(eastern_timestamp[1:3])-int(trader_datetime_timestamp[1:3])
        day_end_hour = day_end_hour if day_end_hour<=23 else day_end_hour - 24

        # Set the restart minute
        restart_minute = day_end_minute = minutes

    # Set the trading start hour
    trading_start_hour = london_start_hour + trader_datetime_sign*int(trader_datetime_timestamp[1:3])
    trading_start_hour = trading_start_hour if trading_start_hour<=23 else (trading_start_hour - 24)

    return restart_hour, restart_minute, day_end_hour, day_end_minute, trading_start_hour
        
def get_data_frequency_values(data_frequency):
    """
    Parses frequency specification string into numerical and categorical components.

    Args:
        data_frequency (str): Frequency string in 'Xmin' or 'Yh' format
            Examples: '15min', '4h'

    Returns:
        tuple: Contains two elements:
            - frequency_number (int): Numerical value of frequency
            - frequency_string (str): Unit identifier ('min' or 'h')

    Notes:
        - Designed to support minute/hour frequencies only
        - Input format must contain either 'min' or 'h' substring
        - Used internally for period/day calculations in `get_periods_per_day`
        - Does not validate input format - assumes proper construction
    """
    
    # If the data frequency is in minutes
    if 'min' in data_frequency:
        # Set the frequency number
        frequency_number = int(data_frequency[:data_frequency.find("min")])
        # Set the frequency string
        frequency_string = data_frequency[data_frequency.find("min"):]
    # If the data frequency is in hours
    elif 'h' in data_frequency:
        # Set the frequency number
        frequency_number = int(data_frequency[:data_frequency.find("h")])
        # Set the frequency string
        frequency_string = data_frequency[data_frequency.find("h"):]
        
    return frequency_number, frequency_string 

def get_periods_per_day(data_frequency):
    """
    Calculates the number of data periods per trading day based on the specified frequency.

    Args:
        data_frequency (str): Frequency specification (e.g., '15min', '1h') 

    Returns:
        int: Number of periods in a 24-hour trading day

    Notes:
        - Requires `get_data_frequency_values` helper to parse frequency components
        - Supports minute ('min') and hour ('h') frequencies only
        - Assumes 24-hour trading calendar (no market closure periods)
        - Example: '15min' → 96 periods/day, '4h' → 6 periods/day
    """
    
    # Get the data frequency number and string
    frequency_number, frequency_string = get_data_frequency_values(data_frequency)
    
    # If the data frequency is in minutes
    if frequency_string == 'min':
        # Return the periods per day
        return 24*(60//frequency_number)
    # If the data frequency is in hours
    elif frequency_string == 'h':
        # Return the periods per day
        return 24//frequency_number

def get_restart_and_day_close_datetimes(data_frequency, now_datetime, day_end_hour, day_end_minute, restart_hour, restart_minute, trading_start_hour):
    """ Function to get the restart and day close datetimes """
    
    # If the now hour is sooner than the day-end hour
    if now_datetime.hour <= day_end_hour:
        # Set the start datetime
        start_datetime = (now_datetime - dt.timedelta(days=1)).replace(hour=trading_start_hour, minute=day_end_minute,second=0, microsecond=0)
    # If the now hour is later than the day-end hour
    else:
        # Set the start datetime
        start_datetime = now_datetime.replace(hour=trading_start_hour, minute=day_end_minute, second=0, microsecond=0)
        
    # If the now hour is later than the restart hour
    if now_datetime.hour >= restart_hour:
        # Set the restart end datetime
        auto_restart_end_datetime = (now_datetime + dt.timedelta(days=1)).replace(hour=restart_hour, minute=restart_minute,second=0, microsecond=0)
    # If the now hour is sooner than the restart hour
    else:
        # Set the restart end datetime
        auto_restart_end_datetime = now_datetime.replace(hour=restart_hour, minute=restart_minute,second=0, microsecond=0)
    
    # Set the day-end datetime
    day_end_datetime = day_datetime_before_end = (start_datetime + dt.timedelta(days=1)).replace(hour=day_end_hour)         
    # Set the day-end datetime in which we're going to close all positions
    trading_day_end_datetime = day_datetime_before_end = (start_datetime + dt.timedelta(days=1)).replace(hour=day_end_hour)   
    # Set the previous day start datetime
    previous_day_start_datetime = (trading_day_end_datetime - dt.timedelta(days=1)).replace(hour=trading_start_hour, minute=day_end_minute, microsecond=0)
    # Set the auto-restart start datetime
    auto_restart_start_datetime = auto_restart_datetime_before_end = auto_restart_end_datetime 

    # If the data frequency is in minutes
    if 'min' in data_frequency:
        # Set the frequency number
        frequency_number = int(data_frequency[:data_frequency.find("min")])
                
        # Create a frequency periods' list with the start datetime as the initial value
        frequency_periods = [start_datetime]
        
        # If the trading day-end datetime is later than the auto-restart end datetime
        if trading_day_end_datetime > auto_restart_end_datetime:

            # Fill the frequency_periods list up to the trading day-end datetime
            i = 0
            while frequency_periods[i] <= trading_day_end_datetime:
                frequency_periods.append(frequency_periods[i] + dt.timedelta(minutes=frequency_number))
                i += 1

            # Set the last trading datetime before the IB platform is auto-restarted
            for i in range(len(frequency_periods)):
                if frequency_periods[i] > auto_restart_end_datetime:
                    auto_restart_datetime_before_end = frequency_periods[i-1]
                    break
                
            # Loop to get the auto-restart start datetime
            for i in range(len(frequency_periods)):
                if frequency_periods[i] >= auto_restart_end_datetime:
                    if (frequency_periods[i] >= auto_restart_end_datetime.replace(minute=5)):
                        auto_restart_start_datetime = frequency_periods[i] 
                        break
    
            # Set last day datetime before the day is closed
            for i in range((len(frequency_periods)-1),0,-1):
                if (trading_day_end_datetime-frequency_periods[i]) > dt.timedelta(minutes=30):
                    day_datetime_before_end = frequency_periods[i]
                    break
            
        # If the trading day-end datetime is sooner than the auto-restart end datetime
        else:
            
            # Fill the frequency_periods list up to the trading day-end datetime
            i = 0
            while frequency_periods[i] <= trading_day_end_datetime:
                frequency_periods.append(frequency_periods[i] + dt.timedelta(minutes=frequency_number))
                i += 1

            # Set last day datetime before the day is closed
            for i in range((len(frequency_periods)-1),0,-1):
                if (trading_day_end_datetime-frequency_periods[i]) > dt.timedelta(minutes=30):
                    day_datetime_before_end = frequency_periods[i]
                    break

            # Create a second frequency periods' list with the start datetime as the initial value
            frequency_periods2 = [trading_day_end_datetime.replace(hour=day_end_hour+1)]
            i = 0
            # Fill the second frequency_periods list up to the auto-restart end datetime
            while frequency_periods2[i] >= auto_restart_end_datetime:
                frequency_periods2.append(frequency_periods2[i] + dt.timedelta(minutes=frequency_number))
                i += 1
            
            # Set the last trading datetime before the IB platform is auto-restarted
            for i in range(len(frequency_periods2)):
                if frequency_periods2[i] > auto_restart_end_datetime:                    
                    auto_restart_datetime_before_end = frequency_periods2[i-1]
                    break
                
            # Loop to get the auto-restart start datetime
            i = len(frequency_periods2)-1
            while auto_restart_start_datetime < auto_restart_end_datetime.replace(minute=5):
                if auto_restart_start_datetime >= auto_restart_end_datetime.replace(minute=5):
                    break
                auto_restart_start_datetime = frequency_periods2[i] + dt.timedelta(minutes=frequency_number)
                frequency_periods2.append(auto_restart_start_datetime)
                i += 1
                
    elif 'h' in data_frequency:
        # Set the frequency number
        frequency_number = int(data_frequency[:data_frequency.find("h")])
                
        # Create a frequency periods' list with the start datetime as the initial value
        frequency_periods = [start_datetime]
        
        # If the trading day-end datetime is later than the auto-restart end datetime
        if trading_day_end_datetime > auto_restart_end_datetime:

            # Fill the frequency_periods list up to the trading day-end datetime
            i = 0
            while frequency_periods[i] <= trading_day_end_datetime:
                frequency_periods.append(frequency_periods[i] + dt.timedelta(hours=frequency_number))
                i += 1

            # Set the last trading datetime before the IB platform is auto-restarted
            for i in range(len(frequency_periods)):
                if frequency_periods[i] > auto_restart_end_datetime:
                    auto_restart_datetime_before_end = frequency_periods[i-1]
                    break
                
            # Loop to get the auto-restart start datetime
            for i in range(len(frequency_periods)):
                if frequency_periods[i] >= auto_restart_end_datetime:
                    if (frequency_periods[i] >= auto_restart_end_datetime.replace(minute=5)):
                        auto_restart_start_datetime = frequency_periods[i]  
                        break
    
            # Set last day datetime before the day is closed
            for i in range((len(frequency_periods)-1),0,-1):
                if (trading_day_end_datetime-frequency_periods[i]) > dt.timedelta(minutes=30):
                    day_datetime_before_end = frequency_periods[i]
                    break

        # If the trading day-end datetime is sooner than the auto-restart end datetime
        else:
            
            # Fill the frequency_periods list up to the trading day-end datetime
            i = 0
            while frequency_periods[i] <= trading_day_end_datetime:
                frequency_periods.append(frequency_periods[i] + dt.timedelta(hours=frequency_number))
                i += 1

            # Set last day datetime before the day is closed
            for i in range((len(frequency_periods)-1),0,-1):
                if (trading_day_end_datetime-frequency_periods[i]) > dt.timedelta(minutes=30):
                    day_datetime_before_end = frequency_periods[i]
                    break

            # Create a second frequency periods' list with the start datetime as the initial value
            frequency_periods2 = [trading_day_end_datetime.replace(hour=day_end_hour+1)]
            # Fill the second frequency_periods list up to the auto restart end datetime
            i = 0
            while frequency_periods2[i] >= auto_restart_end_datetime:
                frequency_periods2.append(frequency_periods2[i] + dt.timedelta(minutes=frequency_number))
                i += 1

            # Set the last trading datetime before the IB platform is auto-restarted
            for i in range(len(frequency_periods2)):
                if frequency_periods2[i] > auto_restart_end_datetime:                    
                    auto_restart_datetime_before_end = frequency_periods2[i-1]
                    break
                
            # Loop to get the auto-restart start datetime
            i = len(frequency_periods2)-1
            while True:
                if auto_restart_start_datetime >= auto_restart_end_datetime.replace(minute=5):
                    break
                auto_restart_start_datetime = frequency_periods2[i] + dt.timedelta(hours=frequency_number)
                frequency_periods2.append(auto_restart_start_datetime)
                i += 1
            
    # Get the actual trading day-end datetime
    trading_day_end_datetime = trading_day_end_datetime.replace(hour=day_end_hour-1,minute=30,second=0)   

    # Set the day start datetime 
    day_start_datetime = previous_day_start_datetime + dt.timedelta(days=1)
    
    return auto_restart_start_datetime, auto_restart_datetime_before_end, auto_restart_end_datetime, \
            day_start_datetime, \
            day_datetime_before_end,  \
            trading_day_end_datetime, \
            day_end_datetime, previous_day_start_datetime

def get_frequency_change(data_frequency):
    """ Function to get data frequency timedelta """
    
    # If data frequency is in minutes
    if 'min' in data_frequency:
        # Define the data frequency timedelta
        time_change = dt.timedelta(minutes=int(data_frequency[:data_frequency.find("min")]))
    # If data frequency is in hours
    elif 'hour' in data_frequency:
        # Define the data frequency timedelta
        time_change = dt.timedelta(hour=int(data_frequency[:data_frequency.find("h")])) 

    return time_change

def get_todays_periods(now_, data_frequency, previous_day_start_datetime):
    """ Function to get all the trading periods from the previous-day start datetime up to now 
        - We set the previous day two days ago in case we're close to the previous day"""
    
    # Set the previous day to two days before
    previous_day_start_datetime = previous_day_start_datetime - dt.timedelta(days=1)
    # Create a list of trading periods where the first value is the previous-day start datetime
    periods = [previous_day_start_datetime]
    
    # Fill the periods' list up to the now datetime
    i = 0
    while True:
       if (periods[i] + get_frequency_change(data_frequency)) <= now_:
           periods.append(periods[i] + get_frequency_change(data_frequency))
       else:
           break
       i += 1
       
    # Set the last period of the list as the next period from now
    periods.append(periods[-1] + get_frequency_change(data_frequency))

    return periods

def get_the_closest_periods_old(now_, data_frequency, trading_day_end_datetime, previous_day_start_datetime, day_start_datetime, market_close_time):
    """ Function to get the closest trading periods to now """
    
    # Get the periods' list
    periods = get_todays_periods(now_, data_frequency, previous_day_start_datetime)
    
    # If now is sooner than the trading day-end datetime
    if now_ < trading_day_end_datetime:
        # If the last periods' list datetime is sooner than the trading day-end datetime
        if periods[-1] <= trading_day_end_datetime:
            # The next period is the last datetime in the periods' list
            next_period = periods[-1]
        # If the last periods' list datetime is later than the trading day-end datetime
        else:
            # The next period is the trading_day_end_datetime
            next_period = trading_day_end_datetime
         
        # Set the previous and current period
        previous_period, current_period = periods[-3], periods[-2]
        
    # If now is sooner than the trading day start datetime
    elif now_ < day_start_datetime:
        # If the last periods' list datetime is sooner than the trading day-end datetime
        if periods[-1] <= day_start_datetime:
            # The next period is the last datetime in the periods' list
            next_period = periods[-1]
        # If the last periods' list datetime is later than the trading day-end datetime
        else:
            # The next period is the trading_day_end_datetime
            next_period = day_start_datetime
         
        # Set the previous and current period
        previous_period, current_period = periods[-3], periods[-2]
        
    # If now is sooner than the market close datetime
    elif now_ < market_close_time:
        # If the last periods' list datetime is sooner than the market close datetime
        if periods[-1] <= market_close_time:
            # The next period is the last datetime in the periods' list
            next_period = periods[-1]
        # If the last periods' list datetime is later than the market close datetime
        else:
            # The next period is the market close datetime
            next_period = market_close_time
            
        # Set the previous and current period
        previous_period, current_period = periods[-3], trading_day_end_datetime
    
    return previous_period, current_period, next_period

def get_the_closest_periods(now_, data_frequency, trading_day_end_datetime, previous_day_start_datetime, day_start_datetime, market_close_time):
    """ Function to get the closest trading periods to now """
    
    # Get the periods' list
    periods = get_todays_periods(now_, data_frequency, previous_day_start_datetime)
    
    # If now is sooner than the trading day-end datetime
    if now_ < trading_day_end_datetime:
        # If the last periods' list datetime is sooner than the trading day-end datetime
        if periods[-1] <= trading_day_end_datetime:
            # The next period is the last datetime in the periods' list
            next_period = periods[-1]
        # If the last periods' list datetime is later than the trading day-end datetime
        else:
            # The next period is the trading_day_end_datetime
            next_period = trading_day_end_datetime
         
        # Set the previous and current period
        previous_period, current_period = periods[-3], periods[-2]
        
    # If now is sooner than the trading day start datetime
    elif now_ < day_start_datetime:
        # # If the last periods' list datetime is sooner than the trading day-end datetime
        # if periods[-1] <= day_start_datetime:
        #     # The next period is the last datetime in the periods' list
        #     next_period = periods[-1]
        # # If the last periods' list datetime is later than the trading day-end datetime
        # else:
        #     # The next period is the trading_day_end_datetime
        #     next_period = day_start_datetime
         
        # # Set the previous and current period
        # previous_period, current_period = periods[-3], periods[-2]
        
        if periods[-2] == trading_day_end_datetime:
            previous_period, current_period, next_period = periods[-3], trading_day_end_datetime, day_start_datetime
        else:
            previous_period, current_period, next_period = periods[-2], trading_day_end_datetime, day_start_datetime
        
    # If now is sooner than the market close datetime
    elif now_ < market_close_time:
        # If the last periods' list datetime is sooner than the market close datetime
        if periods[-1] <= market_close_time:
            # The next period is the last datetime in the periods' list
            next_period = periods[-1]
        # If the last periods' list datetime is later than the market close datetime
        else:
            # The next period is the market close datetime
            next_period = market_close_time
            
        # Set the previous and current period
        previous_period, current_period = periods[-3], trading_day_end_datetime
    
    return previous_period, current_period, next_period



def allsaturdays(date0):
    """ Function to get all the Saturday dates from 2005 to date0 """
    # Create d to be looped
    d = date0
    # Get the next Saturday
    d += dt.timedelta(days = 5 - d.weekday())
    # Loop from d backwards up to 2005 (arbitrary year)
    while d.year >= 2005:
        # Return d
        yield d
        # Go backwards to the previous Saturday
        d -= dt.timedelta(days = 7)

def saturdays_list(date0): 
    """ Function to get all the Saturday datetimes from 2005 to date0 for the historical data download app"""
    # Get the Saturdays list
    saturdays = list(allsaturdays(date0))
    # Get half of the Saturdays list
    saturdays = saturdays[::2][:-1]
    # Convert the Saturdays to datetimes with 23:59:00 (arbitrarily chosen)
    saturdays = [datetime(date0.year, date0.month, date0.day, 23,59,0) for date0 in saturdays]
    # Convert the Saturdays datetimes to datetime type
    saturdays = [date0.strftime("%Y%m%d-%H:%M:%S") for date0 in saturdays]
    return saturdays

def extract_variables(source_file):
    """
    Extracts variables and their values from a Python script and saves them
    to another Python script.
    """
    variables = {}
    try:
        with open(source_file, "r") as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variable_name = target.id
                        try:
                            variable_value = ast.literal_eval(node.value)
                            variables[variable_name] = variable_value
                        except (ValueError, SyntaxError):
                            # Handle cases where the value is not a literal
                            print(f"Warning: Could not extract literal value for {variable_name}")

    except FileNotFoundError:
        print(f"Error: Source file '{source_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return variables

def get_return_variable_names(filename, function_name):
    """
    Extracts return variable names from a function in a file.
    Args:
        filename (str): Path to the Python file (e.g., "strategy.py").
        function_name (str): Name of the function (e.g., "sum1").
    Returns:
        List of variable names (strings) or empty list if not found.
    """
    with open(filename, "r") as file:
        source_code = file.read()
    
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Find the return statement in the function body
            for body_node in node.body:
                if isinstance(body_node, ast.Return):
                    return_value = body_node.value
                    # Extract variable names from return statement
                    if isinstance(return_value, ast.Tuple):
                        return [n.id for n in return_value.elts if isinstance(n, ast.Name)]
                    elif isinstance(return_value, ast.Name):
                        return [return_value.id]
                    else:
                        return []  # Complex expression (not variables)
    return []
