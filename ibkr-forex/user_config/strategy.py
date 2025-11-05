"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
# Import the engine file
from ibkr_forex import engine
"""

# For data manipulation
import pickle 
import numpy as np
import pandas as pd
from hmmlearn import hmm
from datetime import datetime
from sklearn.utils import check_random_state
from ibkr_forex import trading_functions as tf
import featuretools as ft
from featuretools.primitives import Month, Weekday, Hour
from ta import add_all_ta_features
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.calibration import CalibratedClassifierCV as calibration

import warnings
warnings.filterwarnings("ignore")

def create_classifier_model(seed):
    """
    Creates a calibrated bagging classifier model with a Random Forest base estimator.

    This function constructs a classification model consisting of a BaggingClassifier that aggregates
    multiple RandomForestClassifier instances. The model is calibrated using isotonic regression to
    improve probability estimates.

    The base RandomForestClassifier is configured with 5 trees, all features considered for splits,
    and class weights adjusted per bootstrap sample. The BaggingClassifier ensembles 50 such random forests.
    Both components leverage parallel processing (n_jobs=-2). The final output is a calibrated model.

    Args:
        seed (int): Random seed for reproducibility. Controls randomness in the base estimator,
                   bagging process, and calibration.

    Returns:
        CalibratedClassifierCV: A classifier model wrapped with isotonic calibration. The model
                                combines bagged random forests and calibrated probabilities.

    Notes:
        - RandomForestClassifier uses "balanced_subsample" class weighting to handle imbalance.
        - n_jobs=-2 enables parallel processing using all except one CPU core.
        - Isotonic calibration fits a non-linear monotonic function to map probabilities to calibrated values.
    """
    
    # Create the model object
    model = calibration(BaggingClassifier(RFC(n_estimators=5, max_features=1.0, class_weight = "balanced_subsample",\
                                              n_jobs=-2, random_state=seed),
                                          n_estimators=50, random_state=seed, n_jobs=-2),
                        method='isotonic', n_jobs=-2)
    return model

def set_stop_loss_price(app):
    """
    Calculates and sets the stop-loss price based on the trading signal and predefined risk parameters.

    This function determines the stop-loss price for either a long or a short trading position.
    For a long position (where the signal is greater than 0), the stop-loss is set at a price
    below the entry price. Conversely, for a short position (where the signal is less than 0),
    the stop-loss is positioned above the entry price.

    The calculation of the stop-loss distance from the entry price relies on fixed risk
    parameters: a risk target of 0.3% and a stop-loss multiplier of 1. 
    
    You can modify the function as you wish as long as you do not change the function input and the output

    Args:
        app (object): The trading application object

    Returns:
        float: The calculated stop-loss price, rounded to five decimal places.

    Notes:
        - For long positions, the stop-loss is calculated as:
          `stop_loss = last_value * (1 - 0.003 * 1)`
        - For short positions, the stop-loss is calculated as:
          `stop_loss = last_value * (1 + 0.003 * 1)`
        - The result is rounded to five decimal places to ensure practical application
          on trading platforms.
        - The function utilizes fixed risk parameters: a 0.3% risk target and a
          stop-loss multiplier of 1.
    """
    # Set the signal
    signal = app.signal
    # Set the last tick value of the Forex contract
    last_value = app.last_value
    
    # Set the risk management price return target
    risk_management_target = 0.003   
    # Set the stop loss multiplier target
    stop_loss_multiplier = 1     
    # If the signal tells you to go long
    if signal > 0:
        # The stop loss price will be below the long position value
        order_price = round(last_value*(1-risk_management_target*stop_loss_multiplier),5)
    # If the signal tells you to short-sell the asset
    elif signal < 0:
        # The stop loss price will be above the short position value
        order_price = round(last_value*(1+risk_management_target*stop_loss_multiplier),5)
        
    return order_price

def set_take_profit_price(app):
    """
    Calculates and sets the take-profit price based on the trading signal and predefined risk parameters.

    This function determines the take-profit price for either a long or a short trading position.
    For a long position (where the signal is greater than 0), the take-profit is set at a price
    above the entry price. Conversely, for a short position (where the signal is less than 0),
    the take-profit is positioned below the entry price.

    The calculation of the take-profit distance from the entry price relies on fixed parameters:
    a 0.3% return target and a 1x multiplier.

    You can modify the function as you wish as long as you do not change the function input and the output

    Args:
        app (object): The trading application object

    Returns:
        float: The calculated take-profit price, rounded to five decimal places.

    Notes:
        - For long positions, the take-profit is calculated as:
          `take_profit = last_value * (1 + 0.003 * 1)` which simplifies to `last_value * 1.003`.
        - For short positions, the take-profit is calculated as:
          `take_profit = last_value * (1 - 0.003 * 1)` which simplifies to `last_value * 0.997`.
        - The function utilizes a fixed 0.3% (0.003) price target and a 1x take-profit multiplier.
        - The result is rounded to five decimal places to ensure practical application
          on trading platforms.
    """
            
    # Set the signal
    signal = app.signal
    # Set the last tick value of the Forex contract
    last_value = app.last_value
    
    # Set the risk management price return target
    risk_management_target = 0.003   
    # Set the stop loss multiplier target
    take_profit_multiplier = 1     
    # If the signal tells you to go long
    if signal > 0:
        # The stop loss price will be below the long position value
        order_price = round(last_value*(1+risk_management_target*take_profit_multiplier),5)
    # If the signal tells you to short-sell the asset
    elif signal < 0:
        # The stop loss price will be above the short position value
        order_price = round(last_value*(1-risk_management_target*take_profit_multiplier),5)
        
    return order_price

def prepare_base_df(historical_data, train_span=None):
    """
    Prepares a feature-engineered DataFrame for model training and analysis.

    This function undertakes a comprehensive feature engineering process on raw
    OHLC historical data. Key operations include:
    - Creation of a binary target variable ('y') based on next-period returns.
    - Generation of datetime features (one-hot encoded month, weekday, hour).
    - Calculation of various technical indicators using multiple window sizes
      (specifically windows of 3, 4, and 5 periods, as `max_window` is internally set to 6).
    - Creation of lagged percentage changes of OHLC data (up to 9 lags).
    - Enforcement of stationarity for technical indicator features using the
      Augmented Dickey-Fuller (ADF) test; non-stationary series are transformed
      using percentage changes.
    - Addition of volatility-based signals (e.g., moving average crosses,
      standard deviation thresholds).
    - Normalization of selected features using a rolling z-score (30-period window).
    - Cleaning of data, including handling missing or infinite values.

    Args:
        historical_data (pd.DataFrame): Raw input DataFrame containing at least
                                        'Open', 'High', 'Low', and 'Close' price data,
                                        and an index that can be used for time features.
        train_span (int, optional): If provided, truncates the processed DataFrame
                                    to the last N observations. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - pd.DataFrame: The processed DataFrame with engineered features,
              the target variable, and cleaned data, ready for model training
              or further analysis.
            - list: A list of strings containing the names of the final input
              features generated and selected by the function.

    Notes:
        - The target variable 'y' is categorical: 1 for an expected upward price
          movement in the next period, -1 for a downward movement. Rows with
          ambiguous or low-frequency labels might be dropped by an internal
          `tf.dropLabels` call.
        - Datetime features (month, weekday, hour) are derived using `featuretools`
          and then one-hot encoded. The 'Saturday' (day_5) dummy variable is
          explicitly dropped.
        - Technical indicators are calculated using the `add_all_ta_features`
          function (presumably from the `ta` library) for window sizes 3, 4, and 5.
          Volume-based indicators are excluded.
        - Stationarity for technical indicators is checked using the ADF test;
          if a series has a p-value > 0.05, it's transformed via `.pct_change()`.
        - OHLC data is transformed into percentage changes, and then lags (1 to 9)
          of these changes are created as features.
        - Additional signals include moving average crossover signals and standard
          deviation-based volatility signals.
        - Specified features (technical indicators and OHLC lags) undergo
          rolling z-score normalization with a 30-period window.
        - Missing data (NaNs) are handled primarily by forward-filling and then
          dropping any remaining NaNs. Infinite values are replaced with NaNs
          before this process.
        - The function internally sets `max_window = 6` for technical indicator
          window generation.
    """
    
    # The maximum window to create the technical indicators
    max_window = 6

    df = historical_data.copy()
        
    ###############################################################################
    # Section 1: Creating the first model prediction feature
    ###############################################################################
    # Compute the close-to-close log returns
    df['cc_returns'] = np.log(df.Close/df.Close.shift(1))
    # Compute the prediction feature for the first model
    df['y'] = np.where(df['cc_returns'].shift(-1)>0,1,0)
    df['y'] = np.where(df['cc_returns'].shift(-1)<0,-1,df['y'])
    # Drop the rows which have the prediction feature a label with very few observations
    df = tf.dropLabels(df)
    
    # Use the last number of observations
    if train_span is not None:
        df = df.iloc[-train_span:]

    ###############################################################################
    # Section 2: Creating the datetime input features
    ###############################################################################
    # Set the name for the dates dataframe index
    for_features_index = 'index1'
    # Create a column based on the index datetime
    df[for_features_index] = df.index
    
    # Create an entityset to form the dates-based features
    es = ft.EntitySet('My EntitySet')
    es.add_dataframe(
        dataframe_name = 'main_data_table',
        index = 'index',
        dataframe = df[['Close', for_features_index]])#,
        # time_index = 'index1')#, 
        # make_index=True)
        
    # Set the frequency to be used to get the dates time series
    time_features = [Month, Weekday, Hour]
    #time_features = [Minute, Hour, Weekday, Month]
    
    # Create the time series based on the time frequencies' features from above
    fm, features = ft.dfs(
        entityset = es,
        target_dataframe_name = 'main_data_table',
        trans_primitives = time_features)
    
    # Set the fm index as the df.index is
    fm.set_index(df.index, inplace=True, drop=True)
    
    # A function to obtain the fm columns names
    def get_var_name(variable):
         for name, value in globals().items():
            if value is variable:
                return name
            
    # Set the time features names in a list
    time_features_columns = [get_var_name(el).upper()+ '('+for_features_index+')' for el in time_features]
    
    # Obtain the dummies from the months' time series
    months_dummies = pd.get_dummies(fm[time_features_columns[0]])
    # Rename the month's columns
    months_dummies.columns = ['month_'+ str(month) for month in months_dummies.columns.to_list()]
    # Obtain the dummies from the days' time series
    days_dummies = pd.get_dummies(fm[time_features_columns[1]])
    # Rename the days columns
    days_dummies.columns = ['day_'+ str(day) for day in days_dummies.columns.to_list()]
    # Drop the Saturday's column
    days_dummies.drop('day_5', axis=1, inplace=True)
    # Obtain the dummies from the hours' time series
    hours_dummies = pd.get_dummies(fm[time_features_columns[2]])
    # Rename the hour's columns
    hours_dummies.columns = ['hour_'+ str(hour) for hour in hours_dummies.columns.to_list()]
    
    # Save the datetime features in a single dataframe
    datetime_features = pd.concat([months_dummies, days_dummies, hours_dummies],axis=1)
    
    ###############################################################################
    # Section 3: Creating the technical indicators features
    ###############################################################################
    # Create a Volume column with zeros
    df['Volume'] = 0.0
    
    # Set the list of window sizes        
    if max_window<=15:
        windows = list(range(3,max_window))
    elif max_window>=16:
        windows = list(range(3,11))+list(range(15,(max_window+1),10))
    
    # Create the technical indicators dataframe
    technical_features_df = pd.DataFrame(index=df.index)
    
    # Obtain the long-memory stationary OHLC data based on the optimal "d" previously estimated
    df[['Open_dif','High_dif','Low_dif','Close_dif']] = df[['Open','High','Low','Close']].pct_change()
        
    ohlc_lags_list = ['Open_dif','High_dif','Low_dif','Close_dif']
    for lag in list(range(1,10)):
        df[[f'Open_dif_{lag}',f'High_dif_{lag}',f'Low_dif_{lag}',f'Close_dif_{lag}']] = df[['Open_dif','High_dif','Low_dif','Close_dif']].shift(lag)
        ohlc_lags_list.extend([f'Open_dif_{lag}',f'High_dif_{lag}',f'Low_dif_{lag}',f'Close_dif_{lag}'])
    
    # Drop Nan values
    df.dropna(inplace=True)
    
    # Get all the possible technical indicators for each window size
    for window in windows:
        # Obtain the technical indicators
        technical_features = (
            add_all_ta_features(
                df[['Open','High','Low','Close','Volume']].copy(), \
                    open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
            .ffill()
        )
        
        # Drop the OHLCV columns
        technical_features.drop(['Open','High','Low','Close','Volume'], axis=1, inplace=True)
        # Save the names of the volume-based features as a list
        volume_indicators = technical_features.filter(like='volume', axis=1).columns.tolist()
        # Drop the volume-based features
        technical_features.drop(volume_indicators, axis=1, inplace=True)
        
        # Save the names of the volume-based features as a list
        volume_pvo_indicators = technical_features.filter(like='pvo', axis=1).columns.tolist()
        # Drop the volume-based features
        technical_features.drop(volume_pvo_indicators, axis=1, inplace=True)
        
        # Modify the dataframe columns to distinguish them from other features with different window sizes
        technical_features.columns = [f'{column}_{window}' for column in technical_features.columns.tolist()]
        # Concatenate these window-size-based technical indicators to the bigger dataframe
        technical_features_df.loc[technical_features.index, technical_features.columns.tolist()] = technical_features
        
    # Create a loop to make the technical indicators stationary
    for indicator in technical_features_df.columns:
        # If all observations are NaN values
        if technical_features_df[indicator].isna().all():
            # Drop the feature from the dataframe
            technical_features_df.drop(indicator, axis=1, inplace=True)  
            continue
        # If all observations are Infinite
        elif np.isinf(technical_features_df[indicator].values).all():
            # Drop the feature from the dataframe
            technical_features_df.drop(indicator, axis=1, inplace=True)  
            continue
        else:
            try:
                # Get the p-value of the adfuller applied to the technical indicator
                pvalue = adfuller(technical_features_df[indicator].dropna(), regression='c', autolag='AIC')[1]
                # If the p-value is higher than 0.05
                if pvalue > 0.05:
                    # Use the percentage returns of the technical indicator as the input feature
                    technical_features_df[indicator] = technical_features_df[indicator].pct_change()
                # If no p-value was obtained from the adfuller
                elif np.isnan(pvalue):
                    # Drop the feature from the dataframe
                    technical_features_df.drop(indicator, axis=1, inplace=True)
            except:
                # Drop the feature from the dataframe
                technical_features_df.drop(indicator, axis=1, inplace=True)
    
    # Creating more features
    ma_signal_names = [f'ma_signal_{i}' for i in windows]
    std_names = [f'std_{i}' for i in windows]
    std_mean_names = [f'std_mean_{i}' for i in windows]
    std_signal_names = [f'std_signal_{i1}_{i2}' for i1 in windows for i2 in windows]    
    df[ma_signal_names] = np.array([np.where(df['Close']>df['Close'].rolling(i).mean().values,1.0,-1.0) for i in windows]).T
    df[std_names] = np.array([df['Close'].rolling(i).std().values for i in windows]).T
    df[std_mean_names] = np.array([df[f'std_{i}'].rolling(i).mean() for i in windows]).T    
    df[std_signal_names] = np.array([np.where(df[f'std_{i1}']<df[f'std_mean_{i2}'],1.0,-1) for i1 in windows for i2 in windows]).T
    
    ###############################################################################
    # Section 4: Concatenating the necessary dataframes into a single dataframe
    ###############################################################################
    # Set the scalable features list
    scalable_features = technical_features_df.columns.tolist() + df[ohlc_lags_list].columns.tolist()
        
    # Create the base dataframe to be used for the ML model
    base_df = pd.concat([technical_features_df, df[ohlc_lags_list], datetime_features],axis=1)
    
    # Create the states' column to save the HMM-based regimes
    base_df['states'] = 0.0
    
    # Save the names of the final features as a list
    final_input_features = base_df.columns.tolist()
    
    # Create the dataframe to be used for estimating the model
    base_df = pd.concat([base_df, \
                         df[['y','cc_returns','Open','High','Low','Close','high_first']],\
                         df[ma_signal_names+std_signal_names]],axis=1) 
    
    # Save the high_first boolean values as integer numbers
    base_df['high_first_signal'] = np.where(base_df['high_first']==True,1.0,0.0)
    
    # Drop the high_first boolean-type column
    base_df.drop('high_first', axis=1, inplace=True)
        
    # Add the relevant features to the final features
    final_input_features.extend(ma_signal_names)
    final_input_features.extend(std_signal_names)
    final_input_features.append('high_first_signal')
    
    # Forward fill the NaN values
    base_df.ffill(inplace=True)  
    
    # Drop the NaN values
    base_df.dropna(inplace=True) 
    
    ###############################################################################
    # Section 5: Make the input features rolling-zscore-based
    ###############################################################################

    # Z-score the input features
    base_df, _ = tf.rolling_zscore_function(base_df, scalable_features, 30) 
        
    # Drop the Inf values
    base_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    base_df.ffill(inplace=True) 
    
    # Return only the base_df in case you're not working with an ML-based strategy
    return base_df, final_input_features

def get_signal(app):
    """
    Generates a trading signal and leverage using a pre-trained classifier and HMM-based regime detection.

    This function orchestrates the signal generation process by:
    1. Loading pre-trained Hidden Markov Model (HMM) and classifier model objects. These models
       are specific to the market open date (using year, month, and day-1 for file naming).
    2. Splitting the feature-engineered data into training and testing sets, applying purging
       (1 period) and embargo (1 period) techniques to prevent data leakage.
    3. Utilizing the HMM to predict market regimes on the training data based on directional
       change events (theta=0.00002).
    4. Forecasting the next market regime state for the test data using the HMM's transition matrix.
    5. Feeding the test data, augmented with the predicted regime state, into the pre-trained
       classifier to obtain the final trading signal.
    6. Returning the trading signal along with a fixed leverage value.

    Args:
        app (object): The trading application object that contain the following attributes:
                      - final_input_features (list): A list of strings representing the
                        column names of features to be used for model prediction. These
                        must match the features the loaded models were trained on.
                      - base_df (pd.DataFrame): A DataFrame containing feature-engineered
                        data, typically the output from a function like `prepare_base_df`.
                        It should include a 'Close' price column and columns for all
                        features listed in `final_input_features`.
                      - market_open_time (datetime.datetime): A datetime object representing
                        the market open time. This is used to construct the filenames
                        for loading the date-specific HMM and classifier models.

    Returns:
        tuple: A tuple containing two elements:
            - signal (float): The trading signal generated by the classifier. This is
              typically -1 (short), or 1 (long).
            - leverage (int): The leverage to be applied for the trade, currently
              hardcoded to 1. You can optionally create your leverage value based on the 
              base_df dataframe or any other computation

    Notes:
        - The function relies on pre-trained models stored in the 'data/models/' directory.
        - Model filenames follow a strict convention:
            - HMM model: `hmm_model_YYYY_MM_DD.pickle`
            - Classifier model: `model_object_YYYY_MM_DD.pickle`
          Where YYYY, MM are from `market_open_time`, and DD is `market_open_time.day - 1`.
        - Data splitting incorporates a purging window of 1 period and an embargo period of
          1 period to mitigate look-ahead bias.
        - HMM regime prediction is based on directional change events with a `theta` of 0.00002.
        - The next regime state for the test period is forecasted by sampling from the HMM's
          transition matrix based on the last observed state in the training data.
        - The leverage returned is currently fixed at 1.
    """
    
    # Set the features to be used 
    final_input_features = app.final_input_features
    # Set the feature-engineered dataframe
    base_df = app.base_df
    
    # The purged window and embargo period values
    purged_window_size = 1
    embargo_period = 1
    # Set the leverage for the trade size
    leverage = 1
    # Set the market open time
    market_open_time = app.market_open_time
    
    ''' Function to get the signal'''
    
    """ Change code from here """
    ###############################################################################
    # Section 1: Create the month and day strings to be used for calling the model objects
    ###############################################################################

    # Set the month and day strings to call the models
    month_string = str(market_open_time.month) if market_open_time.month>=10 else '0'+str(market_open_time.month)
    day_string = str(market_open_time.day-1) if (market_open_time.day-1)>=10 else '0'+str(market_open_time.day-1)

    ###############################################################################
    # Section 2: Split the data into train and test dataframes for the X and y features
    ###############################################################################

    # Create the input and prediction features
    X, y = tf.create_Xy(base_df, final_input_features, 'y')
    
    # Split the data
    X_train, X_test, y_train, _ = tf.train_test_split(X, y, 1, purged_window_size, embargo_period)
    
    ###############################################################################
    # Section 3: Create the HMM-based input feature
    ###############################################################################

    # Create the R indicator
    r_values = tf.directional_change_events(base_df.loc[X_train.index,['Close']], theta=0.00002, columns='R').dropna().values.reshape(-1,1)
    
    # Call the HMM model
    hmm_model = pickle.load(open(f'data/models/hmm_model_{market_open_time.year}_{month_string}_{day_string}.pickle', 'rb'))
        
    X_train['states'].iloc[-len(r_values):] = hmm_model.predict(r_values)
    
    # Forecast the next period state
    transmat_cdf = np.cumsum(hmm_model.transmat_, axis=1)
    random_state = check_random_state(hmm_model.random_state)
    X_test['states'].iloc[0] = (transmat_cdf[int(X_train['states'].iloc[-1])] > random_state.rand()).argmax()
    
    ###############################################################################
    # Section 4: Create the signal
    ###############################################################################
    # Call the random-forest first model object
    model_object = pickle.load(open(f'data/models/model_object_{market_open_time.year}_{month_string}_{day_string}.pickle', 'rb'))
    
    # Save the model test signal predictions
    signal = base_df.loc[X_test.index,'signal'] = float(model_object.predict(X_test[model_object.feature_names_in_.tolist()].astype("float32"))[0])
    
    """ Change code up to here """
    
    return signal, leverage

def strategy_parameter_optimization(seed, data_frequency,
                                    base_df_address,
                                    train_span, test_span, historical_minute_data_address, market_open_time):

    """
    Executes an end-to-end pipeline to train, select, and save a trading strategy model for a specific period.

    This function performs the following key steps:
    1.  Prepares a feature-engineered DataFrame (`base_df`) from raw OHLC minute data, including
        resampling, technical indicators, and datetime features.
    2.  Splits data into training and testing sets, applying purging and embargo (both set to 1 period).
    3.  Trains a Hidden Markov Model (HMM) for market regime detection using directional change events
        on the training data and forecasts regimes for the test data.
    4.  Performs feature selection on the training data using the Boruta-Shap algorithm to identify
        the most relevant features for the classifier.
    5.  Trains multiple classifier models (e.g., Random Forest) using different random seeds (3 seeds derived
        from the input `seed`) on the selected features.
    6.  Evaluates each trained classifier on the test data based on its annualized Sharpe ratio.
    7.  Saves the HMM, the best performing classifier model, and the initial list of features
        (before Boruta-Shap selection) to disk. The `base_df` is also saved.

    Args:
        seed (int): Random seed for reproducibility, used in HMM initialization,
                    Boruta-Shap, and for generating seeds for classifier training.
        data_frequency (str): Resampling frequency for the raw minute data (e.g., '15min', '1H').
        base_df_address (str): Filename (path relative to 'data/') to save the processed
                               feature-engineered DataFrame (`base_df`).
        train_span (int, optional): Number of periods from the end of the data to use for
                                    the initial `base_df` creation. If None, all data is used.
        test_span (int, optional): Number of periods to reserve for the test set.
                                   Defaults to 5 trading days worth of periods based on `data_frequency`.
        historical_minute_data_address (str): Path to the CSV file containing raw OHLC minute data.
        market_open_time (datetime.datetime): Market open timestamp used for:
                                          - Aligning data resampling.
                                          - Formatting filenames for saved models (using YYYY_MM_DD,
                                            where DD is `market_open_time.day - 1`).

    Returns:
        None: This function does not return any values directly. Instead, it saves the trained
              models, feature lists, and the processed DataFrame to disk.

    Notes:
        - Key artifacts are saved to the 'data/models/' directory (except `base_df`):
            1. HMM model: `hmm_model_YYYY_MM_DD.pickle`
            2. Best Classifier model: `model_object_YYYY_MM_DD.pickle`
            3. Initial Features list: `optimal_features_df.xlsx` (contains features generated by
               `prepare_base_df` before Boruta-Shap selection).
          The YYYY_MM_DD in filenames corresponds to `market_open_time.year`, `market_open_time.month`,
          and `market_open_time.day - 1`.
        - The processed `base_df` is saved to `data/` followed by `base_df_address`.
        - `purged_window_size` and `embargo_period` for train/test splitting are internally set to 1 period.
        - Three classifier models are trained using distinct random seeds derived from the input `seed`.
          The model achieving the highest annualized Sharpe ratio on the test set is saved.
        - Boruta-Shap feature selection is configured with 25 iterations and uses an 80/20 split
          of the training data for its internal validation.
        - Raw data preprocessing involves calculating mid-prices, resampling to `data_frequency`
          aligned with `market_open_time`, and feature engineering via `prepare_base_df`.
        - The HMM is a GaussianHMM with 2 hidden states (intended for bull/bear regimes),
          a diagonal covariance matrix, and trained for 100 EM algorithm iterations.
    """
    
    # Set the month and day strings to save the model objects
    month_string = str(market_open_time.month) if market_open_time.month>=10 else '0'+str(market_open_time.month)
    day_string = str(market_open_time.day-1) if (market_open_time.day-1)>=10 else '0'+str(market_open_time.day-1)

    """ Change code from here """
    purged_window_size = embargo_period = 1
    ###############################################################################
    # Section 1: Prepare the base_df dataframe
    ###############################################################################
    start_time = datetime.now()
    
    # Number of random seeds to generate from the above seed to create the models
    random_seeds = list(np.random.randint(low = 1, high = 10000001, size = 1000000))[:3]

    print('='*100)
    print('='*100)
    print(f"- Preparing the base_df dataframe starts at {start_time}")
    
    # Get the trading periods per day
    periods_per_day = tf.get_periods_per_day(data_frequency)
    # Set the number of rows to be used for the test data
    if test_span is None:
        test_span = 5*periods_per_day
    
    # Import the data
    df = pd.read_csv(historical_minute_data_address, index_col = 0)
    # Parse the index as datetime
    df.index = pd.to_datetime(df.index)
    # Get the midpoint of the OHLC data
    df = tf.get_mid_series(df)
    # Resample the data as the frequency string
    
    # Get the hour string from the market opening time
    hour_string = str(market_open_time.hour) if (market_open_time.hour)>=10 else '0'+str(market_open_time.hour)
    # Get the minute string from the market opening time
    minute_string = str(market_open_time.minute) if (market_open_time.minute)>=10 else '0'+str(market_open_time.minute)
    # Resample the data
    df2 = tf.resample_df(df,frequency=data_frequency,start=f'{hour_string}h{minute_string}min')
    
    # Prepare the dataframe to be used for fitting the model
    # Return None for the final_input_features in case you're not working with an ML-based strategy
    base_df, final_input_features = prepare_base_df(df2, train_span)
    
    # Save the final input features in case you're working with an ML-based strategy
    features_df = pd.DataFrame(data=final_input_features, columns=['final_features'], index=range(len(final_input_features)))
    features_df.to_excel('data/models/optimal_features_df.xlsx')
    
    start_time = datetime.now()
    print('='*100)
    print('='*100)
    print(f"- Backtesting starts at {start_time}")
    
    ###############################################################################
    # Section 2: Split the data into train and test dataframes for the X and y features
    ###############################################################################
    # Create the input and prediction features
    X, y = tf.create_Xy(base_df, final_input_features, 'y')
    
    # Split the data
    X_train, X_test, y_train, _ = tf.train_test_split(X, y, test_span, purged_window_size, embargo_period)
        
    ###############################################################################
    # Section 3: Create the HMM-based input feature
    ###############################################################################
    # Create the R indicator
    r_values = tf.directional_change_events(base_df.loc[X_train.index,['Close']], theta=0.00002, columns='R').values.reshape(-1,1)
    train_r_values = r_values[:-test_span]
    train_r_values = train_r_values[~np.isnan(train_r_values)].reshape(-1,1)
    test_r_values = r_values[-test_span:].reshape(-1,1)
    
    # Create an HMM object with two hidden states
    hmm_model = hmm.GaussianHMM(n_components = 2, covariance_type = "diag", n_iter = 100, random_state = seed)
    
    # Estimate the HMM model
    hmm_model.fit(train_r_values)
    
    # Use the Viterbi algorithm to find the fitted hidden states
    X_train['states'].iloc[(len(X_train)-len(train_r_values)):] = hmm_model.predict(train_r_values)
    
    # Compute the next state hidden state for the test data
    for i in range(len(train_r_values),(len(train_r_values)+len(test_r_values))):
        transmat_cdf = np.cumsum(hmm_model.transmat_, axis=1)
        random_state = check_random_state(hmm_model.random_state)
        if i == len(train_r_values):
            X_test['states'].iloc[(i-len(train_r_values))] = \
                (transmat_cdf[int(X_train['states'].iloc[-1])] > random_state.rand()).argmax()
        else:
            X_test['states'].iloc[(i-len(train_r_values))] = \
                (transmat_cdf[int(X_test['states'].iloc[(i-1-len(train_r_values))])] > random_state.rand()).argmax()
    
    ###############################################################################
    # Section 4: Do feature importance with the Boruta-Shap algorithm
    ###############################################################################
    # Set the date of train split to estimate the Boruta-Shap algorithm
    date_loc_split = X_train.index[int(len(X_train.index)*0.8)]
    
    # Select the best features based on the Boruta Shap algorithm
    print('Get the best features with the Boruta-Shap algorithm')
    selected_features = tf.library_boruta_shap(X_train, y_train.iloc[:,:], seed, 25, date_loc_split)
    print('The best features with the Boruta-Shap algorithm were obtained')
    
    ###############################################################################
    # Section 5: Estimate the models based on the list of seeds
    ###############################################################################
    # Set the start date of the backtesting seed loop
    loop_start_time = datetime.now()
    print('='*100)
    print(f"- Backtesting loop starts at {loop_start_time}")

    # A dictionary to save the model objects created per each seed
    model_objects = {}
    
    # A dictionary to save the test-data Sharpe ratios of each model
    models_sharpe = dict()

    # Annualize factor for the Sharpe ratio
    annualize_factor = 252*periods_per_day

    for i in range(len(random_seeds)):    
        print(f"\t- model number {i+1} estimation starts at {datetime.now()}")

        	# Create an random forest algo object
        model_objects[random_seeds[i]] = create_classifier_model(random_seeds[i]) 
        # Fit the model with the train data
        model_objects[random_seeds[i]].fit(X_train[selected_features].astype("float32"), y_train.astype("int32").values.ravel())
                
        	# Save the model train signal predictions
        base_df.loc[X_train.index,'signal'] = \
            model_objects[random_seeds[i]].predict(X_train[selected_features].astype("float32"))
            
        	# Save the model test signal predictions
        base_df.loc[X_test.index,'signal'] = \
            model_objects[random_seeds[i]].predict(X_test[selected_features].astype("float32"))
            
        # Compute the model's cumulative returns
        base_df.loc[X_train.index,'rets'] = base_df.loc[X_train.index,'cc_returns'] * \
            base_df.loc[X_train.index,'signal'].shift(1)
                                
        # Compute the model's cumulative returns
        base_df.loc[X_test.index,'rets'] = base_df.loc[X_test.index,'cc_returns'] * \
            base_df.loc[X_test.index,'signal'].shift(1)
                                
        # Compute the model's test data Sharpe ratio
        models_sharpe[random_seeds[i]] = np.round(base_df.loc[X_test.index,'rets'].mean() / \
                                                  base_df.loc[X_test.index,'rets'].std() * \
                                                  np.sqrt(annualize_factor),3)
                  
        print(f"\t\t- model number {i+1} with seed {random_seeds[i]} estimation ends at {datetime.now()}")
    
    loop_end_time = datetime.now()
    print('='*100)
    print(f"- Backtesting loop ends at {loop_end_time}")
    print(f"- Backtesting lasted {loop_end_time-start_time}")
      
    ###############################################################################
    # Section 6: Optimize the strategy based on the list of seeds
    ###############################################################################
    # Get the optimal model seed to trade the next month 
    optimal_seed = max(models_sharpe, key=models_sharpe.get)
            
    ###############################################################################
    # Section 7: Save all the model objects used while optimizating the strategy
    ###############################################################################
    # Save the HMM model
    with open(f'data/models/hmm_model_{market_open_time.year}_{month_string}_{day_string}.pickle', 'wb') as handle:
        pickle.dump(hmm_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the model object
    with open(f'data/models/model_object_{market_open_time.year}_{month_string}_{day_string}.pickle', 'wb') as handle:
        pickle.dump(model_objects[optimal_seed], handle, protocol=pickle.HIGHEST_PROTOCOL)

    """ Change code up to here """
    
    # Saving the base_df dataframe
    base_df.to_csv(base_df_address)    
