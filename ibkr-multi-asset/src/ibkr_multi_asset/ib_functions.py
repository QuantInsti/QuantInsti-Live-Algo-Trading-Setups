"""
## Licensed under the QuantInsti Open License (QOL) v1.1 (the "License").
- Copyright 2025 QuantInsti Quantitative Learning Pvt. Ltd.
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License in LICENSE.md at the repository root or at https://www.quantinsti.com.
- Non-Commercial use only; see the License for permitted use, attribution, and restrictions.
"""

# Import the necessary libraries
import datetime as dt
from ibapi.order import Order
from ibapi.client import Contract
from ibapi.execution import ExecutionFilter

def marketOrder(direction,quantity):
    ''' Function to set the market order object'''
    # Set the variable as an Order object
    order = Order()
    # Set the direction of the market order: Buy or Sell
    order.action = direction
    # Set the order as a market one
    order.orderType = "MKT"
    # Set the quantity
    order.totalQuantity = quantity
    # Transmit the order
    order.transmit = True
    # Trade with electronic quotes
    order.eTradeOnly = 0
    # Trade with firm quotes
    order.firmQuoteOnly = 0        
    return order

def cryptoMarketOrder(direction, quantity=None, cash_quantity=None, override=False):
    order = Order()
    order.action = direction
    order.orderType = "MKT"
    order.tif = "IOC"
    if quantity is not None:
        order.totalQuantity = quantity
    if cash_quantity is not None:
        order.cashQty = cash_quantity
    order.transmit = True
    order.eTradeOnly = 0
    order.firmQuoteOnly = 0
    if override:
        order.overridePercentageConstraints = True
    return order

def cryptoLimitOrder(direction, quantity, lmt_price, override=False):
    order = Order()
    order.action = direction
    order.orderType = "LMT"
    order.tif = "IOC"
    order.totalQuantity = quantity
    order.lmtPrice = lmt_price
    order.transmit = True
    order.eTradeOnly = 0
    order.firmQuoteOnly = 0
    if override:
        order.overridePercentageConstraints = True
    return order

def stopOrder(direction,quantity,st_price,trail,override=False):
    ''' Function to set the stop loss order object'''
    # Set the variable as an Order object
    order = Order()
    # Set the direction of the stop loss order: Buy or Sell
    order.action = direction
    # Set the quantity
    order.totalQuantity = quantity
    order.tif = "DAY"
    # Transmit the order
    order.transmit = True
    if trail == False:
        # Set the order as a stop loss one
        order.orderType = "STP"
        # Set the stop loss breach price
        order.auxPrice = st_price
    elif trail == True:
        # Set the order as a stop loss one
        order.orderType = "TRAIL"
        # Set the stop loss breach price
        order.auxPrice = st_price
        # Set the trailing stop loss breach price
        order.trailStopPrice = st_price
    # Trade with electronic quotes
    order.eTradeOnly = 0
    # Trade with firm quotes
    order.firmQuoteOnly = 0
    if override:
        order.overridePercentageConstraints = True
    return order

def tpOrder(direction,quantity,tp_price,override=False):
    ''' Function to set the take profit order object'''
    # Set the variable as an Order object
    order = Order()
    # Set the direction of the take profit order: Buy or Sell
    order.action = direction
    # Set the order as a limit order
    order.orderType = "LMT"
    # Set the quantity
    order.totalQuantity = quantity
    order.tif = "DAY"
    # Transmit the order
    order.transmit = True
    # Set the take profit breach price
    order.lmtPrice = tp_price
    # Trade with electronic quotes
    order.eTradeOnly = 0
    # Trade with firm quotes
    order.firmQuoteOnly = 0
    if override:
        order.overridePercentageConstraints = True
    return order

def ForexContract(symbol,sec_type="CASH",exchange="IDEALPRO"):
    ''' Function to set the Forex contract object'''
    if not (isinstance(symbol, str) and len(symbol) == 6 and symbol.isalpha()):
        raise ValueError(f"Forex symbol '{symbol}' must be a 6-character alphabetic string (e.g., EURUSD).")
    
    # Set the variable as a Contract object
    info = Contract()
    # Set the symbol of the contract
    info.symbol = symbol[:3]
    # Set the security type: Forex
    info.secType = sec_type
    # Set the exchange: IBKR's primary forex exchange
    info.exchange = exchange
    # Set the currency: Second half of the pair
    info.currency = symbol[3:]
    return info

def StockContract(symbol, sec_type="STK", exchange="SMART", currency="USD", primary_exchange=None):
    ''' Function to set the Stock contract object'''
    if not (isinstance(symbol, str) and 1 <= len(symbol) <= 10 and symbol.replace(".", "").replace("-", "").isalnum()):
        raise ValueError(f"Stock symbol '{symbol}' must be a valid equity ticker string.")

    info = Contract()
    info.symbol = symbol.upper()
    info.secType = sec_type
    info.exchange = exchange
    info.currency = currency
    if primary_exchange:
        info.primaryExchange = str(primary_exchange).upper()
    return info

def FuturesContract(symbol,sec_type="FUT",exchange="CME",multiplier=None,expiry=None):
    ''' Function to set the Futures contract object'''
    # Set the variable as a Contract object
    info = Contract()
    # Set the symbol of the contract
    info.symbol = symbol
    # Set the security type: Futures
    info.secType = sec_type
    # Set the exchange: CME
    info.exchange = exchange
    # Set the currency: USD
    info.currency = "USD"
    # Set the multiplier
    if multiplier:
        info.multiplier = str(multiplier)
    # Set the expiry
    if expiry:
        info.lastTradeDateOrContractMonth = expiry
    return info

def MetalsContract(symbol,sec_type="CMDTY",exchange="SMART",currency="USD"):
    ''' Function to set the Metals contract object'''
    # Set the variable as a Contract object
    info = Contract()
    # Set the symbol of the contract
    info.symbol = symbol
    # Set the security type: Commodity
    info.secType = sec_type
    # Set the exchange: SMART
    info.exchange = exchange
    # Set the currency: USD
    info.currency = currency
    return info

def CryptoContract(symbol,sec_type="CRYPTO",exchange="PAXOS",currency="USD"):
    ''' Function to set the Crypto contract object'''
    # Set the variable as a Contract object
    info = Contract()
    # Set the symbol of the contract
    info.symbol = symbol
    # Set the security type: Crypto
    info.secType = sec_type
    # Set the exchange: PAXOS
    info.exchange = exchange
    # Set the currency: USD
    info.currency = currency
    return info

def build_contract_from_spec(asset_spec):
    asset_spec = dict(asset_spec or {})
    symbol = str(asset_spec.get("symbol", "") or "").upper()
    asset_class = str(asset_spec.get("asset_class", "forex") or "forex").lower()
    exchange = str(asset_spec.get("exchange", "") or "").upper()
    currency = str(asset_spec.get("currency", "USD") or "USD").upper()
    contract_month = asset_spec.get("expiry") or asset_spec.get("contract_month")

    if asset_class == "forex":
        return ForexContract(symbol, sec_type="CASH", exchange=exchange or "IDEALPRO")

    if asset_class in {"stock", "stocks", "equity", "equities", "stk"}:
        return StockContract(
            symbol,
            sec_type="STK",
            exchange=exchange or "SMART",
            currency=currency,
            primary_exchange=asset_spec.get("primary_exchange") or asset_spec.get("primaryExchange"),
        )

    if asset_class in {"future", "futures", "fut"}:
        contract = FuturesContract(
            symbol,
            sec_type="FUT",
            exchange=exchange or "CME",
            expiry=contract_month,
        )
        # We don't set the multiplier in the seed contract to allow reqContractDetails
        # to find all matches (e.g. if IBKR expects '5' but we send '5.0').
        # The correct multiplier will be picked up during resolution.
        if asset_spec.get("multiplier") not in (None, ""):
            contract.multiplier = str(asset_spec.get("multiplier"))

        local_symbol = asset_spec.get("localSymbol")
        trading_class = asset_spec.get("tradingClass")
        if local_symbol:
            contract.localSymbol = str(local_symbol)
        if trading_class:
            contract.tradingClass = str(trading_class)
        return contract

    if asset_class in {"metal", "metals", "cmdty", "commodity"}:
        return MetalsContract(
            symbol,
            sec_type=str(asset_spec.get("secType", "CMDTY") or "CMDTY").upper(),
            exchange=exchange or "SMART",
            currency=currency,
        )

    if asset_class == "crypto":
        return CryptoContract(
            symbol,
            sec_type="CRYPTO",
            exchange=exchange or "PAXOS",
            currency=currency,
        )

    sec_type = str(asset_spec.get("secType", "") or "").upper()
    if sec_type == "CASH":
        return ForexContract(symbol, sec_type="CASH", exchange=exchange or "IDEALPRO")
    if sec_type == "STK":
        return StockContract(
            symbol,
            sec_type="STK",
            exchange=exchange or "SMART",
            currency=currency,
            primary_exchange=asset_spec.get("primary_exchange") or asset_spec.get("primaryExchange"),
        )
    if sec_type == "FUT":
        return FuturesContract(symbol, sec_type="FUT", exchange=exchange or "CME", multiplier=asset_spec.get("multiplier", 5), expiry=asset_spec.get("expiry"))
    if sec_type == "CMDTY":
        return MetalsContract(symbol, sec_type="CMDTY", exchange=exchange or "SMART", currency=currency)
    if sec_type == "CRYPTO":
        return CryptoContract(symbol, sec_type="CRYPTO", exchange=exchange or "PAXOS", currency=currency)

    raise ValueError(f"Unsupported asset specification: {asset_spec}")

def get_executions_filter(time_):
    ''' Function to set the executions filter object'''
    # Set the executions filter
    execFilter = ExecutionFilter()
    # Set the time to be used to request the executions data based on the filter
    execFilter.time = time_
    return execFilter

def executionFilter(time_):
    return get_executions_filter(time_)
