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
import datetime as dt
from ibkr_forex import ib_functions as ibf
from threading import Event
from ibkr_forex import trading_functions as tf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

# Define trading app class - inherits from EClient and EWrapper
class trading_app(EClient, EWrapper):
                    
    # Initialize the class - and inherited classes
    def __init__(self, logging, account, account_currency, symbol, timezone, data_frequency, historical_data_address, base_df_address, 
                 market_open_time, market_close_time, 
                 previous_day_start_datetime, trading_day_end_datetime, day_end_datetime, current_period, previous_period, next_period, train_span, test_span, trail):
        
        # Initialize the class from parents
        EClient.__init__(self, self)
        
        # Start time to get later the number of seconds used to run the whole strategy per period
        self.app_start_time = dt.datetime.now()
        
        # The account's base currency
        self.account_currency = account_currency
        
        # Set the account string depending on whether we're using a paper or real account
        if self.port == 4796 or self.port == 4001:
            self.account = 'account'
        else:
            # Paper trading account string
            self.account = account
            
        # Set the data frequency        
        self.data_frequency = data_frequency
        # Set the historical data file address
        self.historical_data_address = historical_data_address
        # Set the base_df file address
        self.base_df_address = base_df_address
        # Set the time zone of the trader 
        self.zone = timezone
        
        # Set the market open datetime of the current week
        self.market_open_time = market_open_time
        # Set the market close datetime of the current week
        self.market_close_time = market_close_time
        # Set the previous day start datetime 
        self.previous_day_start_datetime = previous_day_start_datetime
        # Set the trading day end datetime
        self.trading_day_end_datetime = trading_day_end_datetime
        # Set the day-end datetime
        self.day_end_datetime = day_end_datetime
        # Set the current period to trade
        self.current_period = current_period
        # Set the previous period we traded
        self.previous_period = previous_period
        # Set the next period we'll trade
        self.next_period = next_period
        # Set the train span to be used to create the base_df
        self.train_span = train_span
        # Set the test span for the trading app
        self.test_span = test_span
        
        # Get the data frequency number and time string from the data_frequency string
        self.frequency_number, self.frequency_string = tf.get_data_frequency_values(data_frequency)
        
        # Call the trading app database
        database = pd.ExcelFile('data/database.xlsx')
        # Load the open orders dataframe
        self.open_orders = pd.read_excel(database, "open_orders", index_col = 0)
        # Load the orders status dataframe
        self.orders_status = pd.read_excel(database, "orders_status", index_col = 0)
        # Load the executions dataframe
        self.exec_df = pd.read_excel(database, "executions", index_col = 0)
        # Load the commissions dataframe
        self.comm_df = pd.read_excel(database, "commissions", index_col = 0)
        # Load the positions dataframe
        self.pos_df = pd.read_excel(database, "positions", index_col = 0)
        # Load the cash balance dataframe
        self.cash_balance = database.parse("cash_balance", index_col = 0)

        # Convert to datetime the string index of each of the previous dataframes
        self.open_orders.index = pd.to_datetime(self.open_orders.index)
        self.orders_status.index = pd.to_datetime(self.orders_status.index)
        self.exec_df.index = pd.to_datetime(self.exec_df.index)
        self.comm_df.index = pd.to_datetime(self.comm_df.index)
        self.pos_df.index = pd.to_datetime(self.pos_df.index)
        self.cash_balance.index = pd.to_datetime(self.cash_balance.index)

        # Kiad the app time spent dataframe
        self.app_time_spent = database.parse("app_time_spent", index_col=0)
        # Convert to float previous_time_spent seconds column
        self.previous_time_spent = float(self.app_time_spent['seconds'].iloc[0])        

        # Load the periods traded dataframe
        self.periods_traded = database.parse("periods_traded", index_col = 0)
        
        # Convert to datetime type the trade_time column of the previous_traded dataframe
        self.periods_traded['trade_time'] = pd.to_datetime(self.periods_traded['trade_time'])
        
        # Add a new row to the periods_traded column with the current period
        self.periods_traded.loc[len(self.periods_traded.index),:] = [current_period, 0, market_open_time, market_close_time]
        
        # Create the new_df dataframe to save the downloaded historical BID and ASK data
        self.new_df = {}
        self.new_df['0'] = pd.DataFrame()
        self.new_df['1'] = pd.DataFrame()
       
        # Set the ticker
        self.ticker = symbol
        
        # Import the historical data 
        self.historical_data = pd.read_csv(historical_data_address, index_col=0)
        # Convert the historical data index to datetime
        self.historical_data.index = pd.to_datetime(self.historical_data.index)
        
        # Use the last train_span observations for the historical data
        self.historical_data = self.historical_data.tail(self.train_span)
                
        # Create the forex contract based on the ticker
        self.contract = ibf.ForexContract(self.ticker)
        
        # Create a dictionary to save the app output errors
        self.errors_dict = {}
        
        # Load the optimal features dataframe in case the trader chose to work with an ML-based strategy
        if os.path.exists('data/models/optimal_features_df.xlsx'):            
            features_df = pd.read_excel('data/models/optimal_features_df.xlsx', index_col=0)
            # Set all the features to prepare the data
            self.final_input_features = features_df['final_features'].dropna().tolist()
        else:
            print('There were no final features saved in the strategy_parameter_optimization function')
        
        # Set the stop loss order id to NaN             
        self.sl_order_id = np.nan
        # Set the take profit order id to NaN             
        self.tp_order_id = np.nan
        
        # Set the count values to zero to stop the app in case there's no way to keep the app running smoothly
        self.count = 0
        self.last_value_count = 0
        
        # Create the account update information dataframe
        self.acc_update = pd.DataFrame()        
        # Create a dictionary to save the historical data events to download it properly        
        self.hist_data_events = {}
        # Set the historical data threading event for the BID and ASK data request
        self.hist_data_events['0'] = Event()
        self.hist_data_events['1'] = Event()
        # Set a threading event to request the open orders and orders status
        self.orders_request_event = Event()
        # Set a threading event to request the trading positions
        self.positions_request_event = Event()
        # Set the threading event for the account update
        self.account_update_event = Event()
        # Set the threading event for the executions request
        self.executions_request_event = Event()

        # Create temporary dataframes to be used while requesting previous trading information
        self.acc_update = pd.DataFrame()
        self.temp_open_orders = pd.DataFrame()
        self.temp_orders_status = pd.DataFrame()
        self.temp_exec_df = pd.DataFrame()
        self.temp_comm_df = pd.DataFrame()  
        self.temp_pos_df = pd.DataFrame() 
        
        # Set the trailing stop loss boolean
        self.trail = trail
        
        # Set the strategy end to False
        self.strategy_end = False
        
        # Set the logging as part of the setup
        self.logging = logging
        
    def error(self, reqId, code, msg, advancedOrderRejectJson=''):
        ''' Called if an error occurs '''
        self.errors_dict[code] = msg
        print('Error: {} - {} - {}'.format(reqId, code, msg))
        self.logging.info('Error: {} - {} - {}'.format(reqId, code, msg))
                
    def nextValidId(self, orderId):
        ''' Set the next order id '''
        # If the app is connected
        if self.isConnected():
            super().nextValidId(orderId)
            self.nextValidOrderId = orderId
            print("NextValidId:", orderId)
            self.logging.info("NextValidId: {}".format(orderId))
            time.sleep(1)
        else:
            return
    
    def openOrder(self, orderId, contract, order, orderState):
        ''' Function to call the open orders '''
        super().openOrder(orderId, contract, order, orderState)
        # Create a dictionary to save the data
        dictionary = {"PermId":order.permId, \
                      "ClientId": order.clientId, \
                      "OrderId": orderId, 
                      "Account": order.account, \
                      "Symbol": contract.symbol, \
                      "SecType": contract.secType,
                      "Exchange": contract.exchange, \
                      "Action": order.action, \
                      "OrderType": order.orderType,
                      "TotalQty": float(order.totalQuantity), \
                      "CashQty": order.cashQty, 
                      "LmtPrice": order.lmtPrice, \
                      "AuxPrice": order.auxPrice,\
                      "Status": orderState.status,\
                      "datetime":dt.datetime.now().replace(microsecond=0)}
        # Save the data into the temporary dataframe
        self.temp_open_orders = pd.concat([self.temp_open_orders, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

        
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, \
                    permId, parentId, lastFillPrice, clientId, whyHeld, \
                    mktCapPrice):
        ''' Function to call the orders status'''
        super().orderStatus(orderId, status, filled, remaining, \
                            avgFillPrice, permId, parentId, lastFillPrice, \
                            clientId, whyHeld, mktCapPrice)
        
        # Create a dictionary to save the data
        dictionary = {"OrderId": orderId, \
                      "Status":status, \
                      "Filled":filled, \
                      "PermId":permId, \
                      "ClientId": clientId, \
                      "Remaining": float(remaining), \
                      "AvgFillPrice": avgFillPrice, \
                      "LastFillPrice": lastFillPrice, \
                      "datetime":dt.datetime.now().replace(microsecond=0)}
        # Save the data into the temporary dataframe
        self.temp_orders_status = pd.concat([self.temp_orders_status, pd.DataFrame(dictionary, index=[0])], ignore_index=True)
                
    def openOrderEnd(self):
        print("Open orders request was successfully completed")
        self.orders_request_event.set()

    # Receive details when orders are executed        
    def execDetails(self, reqId: int, contract, execution):
        ''' Function to call the trading executions'''
        print('Requesting the trading executions...')
        self.logging.info('Requesting the trading executions...')
        
        super().execDetails(reqId, contract, execution)
        # Create a dictionary to save the data
        dictionary = {"OrderId": execution.orderId,  
                      "PermId":execution.permId, \
                      "ExecutionId":execution.execId, \
                      "Symbol": contract.symbol, \
                      "Side":execution.side, \
                      "Price":execution.price, \
                      "AvPrice":execution.avgPrice, \
                      "cumQty":execution.cumQty, \
                      "Currency":contract.currency, \
                      "SecType": contract.secType, \
                      "Position": float(execution.shares), \
                      "Execution Time": execution.time, \
                      "Last Liquidity": execution.lastLiquidity, \
                      "OrderRef":execution.orderRef, \
                      "datetime":dt.datetime.now().replace(microsecond=0)}
            
        # Save the data into the temporary dataframe
        self.temp_exec_df = pd.concat([self.temp_exec_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)
        
    def commissionReport(self, commissionReport):
        ''' Function to call the trading commissions'''
        print('Requesting the trading commissions...')
        self.logging.info('Requesting the trading commissions...')
        
        super().commissionReport(commissionReport)
        # Create a dictionary to save the data
        dictionary = {"ExecutionId":commissionReport.execId, \
                      "Commission": commissionReport.commission, \
                      "Currency":commissionReport.currency, \
                      "Realized PnL": float(commissionReport.realizedPNL), \
                      "datetime":dt.datetime.now().replace(microsecond=0)}
            
        # Save the data into the temporary dataframe
        self.temp_comm_df = pd.concat([self.temp_comm_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)

    def execDetailsEnd(self, reqId: int):
        super().execDetailsEnd(reqId)
        print("Trading executions request was successfully finished. ReqId:", reqId)
        self.executions_request_event.set()
                
    # Receive the positions from the TWS
    def position(self, account, contract, position, avgCost):
        ''' Function to call the trading positions'''
        print('Requesting the trading positions...')
        self.logging.info('Requesting the trading positions...')
        super().position(account, contract, position, avgCost)
        # Create a dictionary to save the data
        dictionary = {"Account":account, "Symbol": contract.symbol, \
                      "SecType": contract.secType,
                      "Currency": contract.currency, "Position": float(position), \
                      "Avg cost": avgCost, "datetime":dt.datetime.now().replace(microsecond=0)}
            
        # Save the data into the temporary dataframe
        self.temp_pos_df = pd.concat([self.temp_pos_df, pd.DataFrame(dictionary, index=[0])], ignore_index=True)
        
    # Display message once the positions are retrieved
    def positionEnd(self):
        print('Positions Retrieved.')
        self.positions_request_event.set()
                
    # Receive historical bars from TWS
    def historicalData(self, reqId, bar):
        ''' Function to call the historical data'''
        # Save the data into the new_df dataframe
        self.new_df[f'{reqId}'].loc[bar.date,'close'] = bar.close
        self.new_df[f'{reqId}'].loc[bar.date,'open'] = bar.open
        self.new_df[f'{reqId}'].loc[bar.date,'high'] = bar.high
        self.new_df[f'{reqId}'].loc[bar.date,'low'] = bar.low
                        
    # Display a message once historical data is retrieved
    def historicalDataEnd(self,reqId,start,end):
        super().historicalDataEnd(reqId,start,end)
        print("Historical Data Download finished...")
        self.logging.info("Historical Data Download finished...")
        # Set the event and end the historical data download
        self.hist_data_events[f'{reqId}'].set()

    def tickByTickMidPoint(self, reqId, tick_time, midpoint):
        ''' Function to call in response to reqTickByTickData '''
        # Save midpoint price to last_value 
        self.last_value = midpoint
                        
    def updateAccountValue(self, key, value, currency, accountName):
        ''' Function to call the account values'''
        super().updateAccountValue(key, value, currency, accountName)
        # Save the data into a dictionary
        dictionary = {"key":key, "Account": accountName, "Value": value, \
                      "Currency": currency, "datetime":dt.datetime.now().replace(microsecond=0)}
                        
        # Save the data into a temporary dataframe
        self.acc_update = pd.concat([self.acc_update, pd.DataFrame(dictionary, index=[0])], ignore_index=True)
            
    def updateAccountTime(self, timeStamp: str):
        print("Account update time is:", timeStamp)
         
    def accountDownloadEnd(self, accountName: str):
        print("Account download was done for account:", accountName)
        self.logging.info(f"Account download was done for account: {accountName}")
        self.account_update_event.set()
