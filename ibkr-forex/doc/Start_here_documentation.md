## An end-to-end setup to trade forex algorithmically

#### This is your “Start here” document to set up your system for trading forex algorithmically.
###### QuantInsti Webpage: https://www.quantinsti.com/

**Version 1.0.0**
**Last Updated**: 2025-05-28

-----

## Disclaimer

#### This file is documentation only and you should not use it for live trading without appropriate backtesting and tweaking of the strategy parameters.

## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this document except in compliance with the License.
- You may obtain a copy of the License in `LICENSE.md` at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.

## Table of contents
1. [Introduction](#introduction)
2. [Crucial Attributes](#crucial_attributes)
3. [Setup Notes](#setup_notes)
4. [Interactive Brokers setup requirements](#ib_requirements)
5. [Setup of variables](#variables_setup)

<a id='introduction'></a>
## Introduction
This document provides a Python-based setup for algorithmic forex trading using the Interactive Brokers API. This script allows you to execute transactions in the forex market with a customizable strategy and interchangeable forex assets.

The script-based application is intended to demonstrate how you can use a ready-made, IB-API-based trading setup and its functionality during each trading period.

<a id='crucial_attributes'></a>
## Crucial Attributes:
- **End-to-End Forex Trading Setup**: A comprehensive solution for algorithmic forex trading, managing data from download and processing to strategy execution and order management.
- **Strategy Customization**: The setup is designed for full customization. You can modify the provided strategy or implement your own by editing the `strategy.py` file, tailoring the trading logic to your specific requirements and risk tolerance.
- **Flexible Asset Selection**: You can trade any forex pair available through Interactive Brokers. The setup can be easily configured for different assets.
- **Interactive Brokers API Integration**: Integration with the Interactive Brokers API provides reliable access to real-time market data and trade execution.
- **Script-Based Operation**: The setup is entirely script-based, without a graphical user interface, making it lightweight, fast, and easy to deploy and automate. This also facilitates understanding and modification of the code.

<a id='setup_notes'></a>
## Setup notes
1. If you want to use the same strategy to check how the setup works, you should only need to modify the “main.py” file . In case you want to modify the strategy at your convenience, you should also modify the “strategy_file.py” file (both files are located in the "samples" folder). Only forex contracts should be traded with this trading app.
2. If you run the trading setup for the first time, you'll see that you'll be downloading historical minute data. It will take like 3 to 5 days to complete the downloading (it will download from 2005 to 2024). This only happens at the very first time. Once you have the historical minute data up to date, you'll have the trading setup running.
3. The forex market closes from 5 pm to 6 pm Eastern time and the stop-loss and take-profit targets get discarded at 5 pm EST. Each day the setup will close all the existing positions half an hour before 5 pm EST. 
4. The setup will not leave any open positions on weekends. 
5. The strategy is based on bagging with a random forest algorithm. It creates long and short signals. To learn more about it, you can refer to the MLT-04 lecture.
6. The trading setup is designed to retrieve historical data from up to 10 previous days. If your historical data has missing data for more than 10 days, you’ll need to run the setup to download historical data and update the dataframe.
7. In case you want to get the live equity curve of the strategy, once you start trading, you should go to the “database.xlsx” Excel file, sheet “cash_balance”, column “value”. You can plot that column to see the equity curve graphically.
8. In case you want to make more changes to the setup so it can be better customized per your needs, you can modify all the other relevant files as needed.

<a id='ib_requirements'></a>
## Interactive Brokers setup requirements

1. You should install the **offline stable** version of the TWS. You should save the TWS file in “path_to/Jts/tws_platform"
2. You should install the **stable** version of the IB API. You should save the IB API files in "path_to/Jts/tws_api"
3. You should log in with your account credentials in the IB web platform and then go to Settings. Next, in the “Account Configuration” section, you should click on “Paper Trading Account”. Finally, you should click on the “Yes” button against the question “Share real-time market data subscriptions with paper trading account?” and click on “Save”. You should wait at least 2 hours to let IB make the paper account have market data subscriptions.
4. In the TWS or IB Gateway platform, you should do the following: Go to File, Global configuration, API, and in Settings:
    1. Check “ActiveX and Socket Clients”
    2. Uncheck the “Read-Only API”
    3. In the “send instrument-specific attributes for dual-mode API client in” box, you should select “operator timezone”.
    4. You can click on the “Reset API order ID sequence” whenever you need to restart paper or live trading.
5. In the TWS or IB Gateway platform, you should do the following: Go to File, Global configuration, Configuration and in “Lock and Exit”:
    - In the “Set Auto Logoff Timer” section, you should choose your local time to auto-log off or auto-restart the platform. Due to the IB platform specifications, in case you select auto-restart, it must restart at the specific hour you select. You should be careful with this. When selecting auto restart, sometimes it doesn’t work properly, so you might need to log in to the platform again manually. 
6. In Configuration, Export trade Reports, you should check the “Export trade reports periodically”.
7. In the same section from above, in the “Export filename”, you should type: “\path_to\Jts\setup\samples\data\reports\report.txt”. This file will give you a trade report of all the trading positions you took while trading.
8. In the same section from above, in the “Export directory”, you should type: “\path_to\Jts\setup\samples\data\reports”. This folder will be used to save the trade report from above.
9. In the same section above, you should specify the interval at which you would like the reports to be generated.
10. Depending on your initial equity and trading frequency, you will have different equity curves throughout time. If you first want to try paper trading, you should set the initial equity value. To do this, you should go to https://www.interactivebrokers.co.uk/sso/Login and do the following:
    1. Select “Paper”, instead of “Live”
    2. Login with your username and password
    3. Go to “Settings”
    4. Go to “Account Settings”
    5. In “Configuration”, you should click on the nut button of the “Paper Trading Account Reset”
    6. In the “Select Reset Amount” box, you should click on “Other Amount”. In the “Amount” box, you should write a specific amount you want to use as an initial equity value. You should read the below instructions and click on Continue.
11. In case you want to reset the paper trading account to default settings, you should do the following:
    1. You should drop all the created files saved in the "data" folder and sub-folders.
    2. You should go to the TWS platform, go to the “File” tab, click on “Global Configuration”, click on “API settings”, and click on the “Reset API order ID sequence” button. Finally, you should click on “Apply” and “Ok”. Then you can paper trade once again from the start. In case you have live traded, you should close any existing position on any asset before you restart live trading with the app.
12. To profit from forex leverage, you need to **ask IB to have a margin account for forex**. If you don’t do it, you will not be able to trade at all your capacity.

<a id='variables_setup'></a>
## Setup of variables
Inside the “main” file, you can change the following variables per your trading requirements. Each variable is explained and some extra information is added.

- **account**: The account name of your IB account. This account name starts with U followed by a number for a live trading account and starts with DU followed by a number for a paper trading account. You can learn more in the TBP-01 lecture.
- **timezone**: You should set the timezone of your country of residence. You should select the appropriate timezone as per the available Python time zones.
- **port**: The port number as per the live or paper trading account. You can learn more in the TBP-01 lecture.
- **account_currency**: The base currency that your IB account has. You set the base currency while creating your IB account. It can be USD, EUR, INR, etc.
- **symbol**: The forex symbol to be traded. You should choose as per the IB available forex assets to be traded. 
- **data_frequency**: The frequency used for trading. You should set this variable to ‘24h’ if you want to trade daily. The setup is not designed to trade with a frequency lower than daily (2-day, 3-day, etc.). You should be careful while deciding the data_frequency because the signal creation might take longer than your chosen trading frequency. To check how much time it takes to run the strategy, you should check for each period the “epat_trading_app_database” file, sheet “app_time_spent”, column name “seconds”, and the unique value. 
- **local_restart_hour**: The local timezone hour you previously selected to log off or auto-restart your IB TWS. If you log off or auto-restart at 11 pm in the TWS platform, you should set this variable to 23, and so on.
- **historical_data_address**: The string of the historical data file name and address. The data file is the resampled data per the frequency you set above.
- **base_df_address**: The string of the dataframe used to fit the machine learning model. You should set the file name and address at your convenience.
- **train_span**: You should set the train data number of observations to be used to fit the machine learning model. You should check the historical_data_address file to specify a number equal to or lower than the maximum data observations available in the historical dataframe file.
- **test_span_days**: To optimize the strategy, you should specify how many days you want to use as a validation dataset. The higher the trading frequency, the higher this number should be. For a daily frequency, you should set 22 days as a monthly validation dataset.
- **host**: You should set the host for the trading app. You can learn more in the TBP-01 lecture.
- **client_id**: You should set the client ID for the trading app. You can learn more in the TBP-01 lecture.
- **seed**: You should set the seed to create a list of random seeds for the machine learning strategy. Each seed provides a unique machine-learning model to be used for backtesting it. The best model is chosen based on the machine learning model that gives the best Sharpe ratio of its strategy returns.
- **smtp_username**: Your Gmail to be used from which you’ll send the trading information per the above trading data frequency.
- **to_email**: The email (it can be any email service: Gmail, Outlook, etc.) to send the trading information per the above trading data frequency.
- **password**: The app password that was obtained from Google Gmail. You need to allow the app password in Google: https://support.google.com/mail/answer/185833?hl=en. Once you access the link, you should click on the link “Create and manage your app passwords”. Then, you should type your email and password and you’ll be directed to the “App passwords” webpage. There, you should type an app name, it can be any name, and then you’ll be given a 12-letter-long password. You should copy that password and paste it into this variable.

In the same "main" file, you can add optional variables per your trading requirements. In this case we optionally added the following:
- **leverage**: You should set the fixed leverage you will use at your trading convenience. If you want to create a dynamic leverage position, you should change the strategy_file file. You can set this with values from 0 (no position) to any positive number. A high value needs to be evaluated as per the leverage limits IB sets for each forex asset.
