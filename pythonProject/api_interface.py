import math
import datetime
from threading import Thread
import time

from pytz import timezone, utc

import pytz

from datetime import timedelta

from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.common import BarData
from ibapi.order import Order
from ibapi import order_condition

import pandas as pd

Bid, Ask, Last, Close, High, Low, Open, Date, Volume = \
    'Bid', 'Ask', 'Last', 'Close', 'High', 'Low', 'Open', 'Date', 'Volume'

DelayedBid, DelayedAsk, DelayedLast, DelayedClose, DelayedHigh, DelayedLow, DelayedOpen, DelayedDate, DelayedVolume = \
    'DelayedBid', 'DelayedAsk', 'DelayedLast', 'DelayedClose', 'DelayedHigh', 'DelayedLow', 'DelayedOpen', 'DelayedDate', 'DelayedVolume'

HIDE_ERROR_CODES = [2104, 2106, 2158, 399]


class PriceInformation:
    def __init__(self, contract):
        self.contract = contract
        self.Bid = None
        self.Ask = None
        self.Last = None
        self.Close = None
        self.High = None
        self.Low = None
        self.Open = None
        self.Date = None
        self.Volume = None

        self.DelayedBid = None
        self.DelayedAsk = None
        self.DelayedLast = None
        self.DelayedClose = None
        self.DelayedHigh = None
        self.DelayedLow = None
        self.DelayedOpen = None
        self.DelayedDate = None
        self.DelayedVolume = None

        self.NotDefined = None

    def __str__(self):
        return f"Bid: {self.Bid}, Ask: {self.Ask}, Last: {self.Last}, Close: {self.Close}, High: {self.High}, " \
               f"Low: {self.Low}, Open: {self.Open}"

    # def __str__(self):
    #     report_string = ""
    #     report_string += self.contract.symbol if self.contract.localSymbol == "" else self.contract.localSymbol
    #
    #     for t in ['Bid', 'Ask', 'Last', 'Close', 'High', 'Low', 'Open', 'DelayedBid', 'DelayedAsk', 'DelayedLast',
    #               'DelayedClose', 'DelayedHigh', 'DelayedLow', 'DelayedOpen']:
    #         price = getattr(self, t, None)
    #         if price is not None:
    #             report_string += f", {t}: {str(price)}"
    #
    #     return report_string


class OpenOrderInfo:
    def __init__(self, contract=None, order=None, orderState=None, permId=None, clientId=None, orderId=None,
                 account=None, symbol=None, secType=None, exchange=None, action=None, orderType=None,
                 totalQty=None, cashQty=None, lmtPrice=None, auxPrice=None, status=None):

        self.contract = contract
        self.order = order
        self.orderstate = orderState
        self.permId = permId
        self.clientId = clientId
        self.orderId = orderId
        self.account = account
        self.symbol = symbol
        self.secType = secType
        self.exchange = exchange
        self.action = action
        self.orderType = orderType
        self.totalQty = totalQty
        self.cashQty = cashQty
        self.lmtPrice = lmtPrice
        self.auxprice = auxPrice
        self.status = status

    def __str__(self):
        if self.contract is None or self.order is None:
            return """Order State: {}, Status: {}, Filled: {}, Remaining: {}""". \
                format(self.orderstate, self.status, self.action, self.totalQty)
        else:
            return """Symbol: {}, Order: {}, Order State: {}, Status: {}, Filled: {}, Remaining: {}""". \
                format(self.contract.symbol, self.order.orderType, self.orderstate,
                       self.status, self.action, self.totalQty)


class OrderInformation:
    def __init__(self, contract=None, order=None, orderstate=None, status=None, filled=None, remaining=None,
                 avgFillPrice=None, permid=None, parentId=None, lastFillPrice=None, clientId=None, whyHeld=None,
                 mktCapPrice=None):
        self.contract = contract
        self.order = order
        self.orderstate = orderstate
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avgFillPrice
        self.permid = permid
        self.parentId = parentId
        self.lastFillPrice = lastFillPrice
        self.clientId = clientId
        self.whyHeld = whyHeld
        self.mktCapPrice = mktCapPrice

    def __str__(self):
        if self.contract is None or self.order is None:
            return """Order State: {}, Status: {}, Filled: {}, Remaining: {}""". \
                format(self.orderstate, self.status, self.filled, self.remaining)
        else:
            return """Symbol: {}, Order: {}, Order State: {}, Status: {}, Filled: {}, Remaining: {}""". \
                format(self.contract.symbol, self.order.orderType, self.orderstate,
                       self.status, self.filled, self.remaining)


class PositionInfo:
    def __init__(self, contract, pos, account, avgCost):
        self.contract = contract
        self.pos = pos
        self.account = account
        self.avgCost = avgCost

    def get(self, attribute):
        return getattr(self, attribute, None)


class Wrapper(EWrapper):
    """
    Inherited wrapper function to over(write) our own methods in.
    """
    FINISHED = "FINISHED"

    def __init__(self):
        EWrapper.__init__(self)
        self.market_data = {}
        self.contract_details = {}
        self.historical_market_data = {}
        self.next_valid_order_id = None
        self.positions_pnl = {}
        self.positions = []

        self.positions_end_flag = False
        self.next_valid_order_id = None

        self.df_dict = {}

        self.nyc = timezone('America/New_York')
        self.today = datetime.datetime.today().astimezone(self.nyc)

        self.current_dt = datetime.datetime.today().astimezone(self.nyc)

        self.df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        self.order_df = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                              'Account', 'Symbol', 'SecType',
                                              'Exchange', 'Action', 'OrderType',
                                              'TotalQty', 'CashQty', 'LmtPrice',
                                              'AuxPrice', 'Status'])

    def contractDetails(self, reqId, contractDetails):
        try:
            self.contract_details[reqId].append(contractDetails)
        except KeyError:
            pass

    def contractDetailsEnd(self, reqId):
        try:
            self.contract_details[reqId].append(self.FINISHED)
        except KeyError:
            pass

    def tickPrice(self, reqId, tickType, price: float, attrib):
        if tickType == 1:
            data_type = Bid
        elif tickType == 2:
            data_type = Ask
        elif tickType == 4:
            data_type = Last
        elif tickType == 9:
            data_type = Close
        elif tickType == 6:
            data_type = High
        elif tickType == 7:
            data_type = Low
        elif tickType == 14:
            data_type = Open
        elif tickType == 66:
            data_type = DelayedBid
        elif tickType == 67:
            data_type = DelayedAsk
        elif tickType == 68:
            data_type = DelayedLast
        elif tickType == 72:
            data_type = DelayedHigh
        elif tickType == 73:
            data_type = DelayedLow
        elif tickType == 75:
            data_type = DelayedClose
        elif tickType == 76:
            data_type = DelayedOpen
        else:
            data_type = "NotDefined"

        try:
            setattr(self.market_data[reqId], data_type, price)
        except KeyError:
            pass

    def historicalData(self, reqId: int, bar: BarData):
        """
        Function to recive all market data
        """
        self.historical_market_data[reqId].append(bar) if reqId in self.historical_market_data.keys() else None

        self.df.loc[len(self.df)] = [bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume]

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """
        Function to indicate market data has been received
        """
        self.historical_market_data[reqId].append(
            self.FINISHED) if reqId in self.historical_market_data.keys() else None

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)

        dictionary = {"PermId": order.permId, "ClientId": order.clientId, "OrderId": orderId,
                      "Account": order.account, "Symbol": contract.symbol, "SecType": contract.secType,
                      "Exchange": contract.exchange, "Action": order.action, "OrderType": order.orderType,
                      "TotalQty": order.totalQuantity, "CashQty": order.cashQty,
                      "LmtPrice": order.lmtPrice, "AuxPrice": order.auxPrice, "Status": orderState.status}

        if self.order_df is None:
            self.order_df = pd.DataFrame([dictionary])
        else:
            self.order_df = pd.concat([self.order_df, pd.DataFrame([dictionary])], ignore_index=True)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        order_details = OrderInformation(status=status, filled=filled, remaining=remaining, avgFillPrice=avgFillPrice,
                                         permid=permId, parentId=parentId, lastFillPrice=lastFillPrice,
                                         clientId=clientId,
                                         whyHeld=whyHeld, mktCapPrice=mktCapPrice)
        print(order_details)

    def pnlSingle(self, reqId: int, pos: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float, value: float):
        self.positions_pnl[
            reqId] = pos, dailyPnL, unrealizedPnL, realizedPnL if reqId in self.positions_pnl.keys() else None

    def position(self, account: str, contract: Contract, position: float,
                 avgCost: float):
        self.positions.append(PositionInfo(contract, position, account, avgCost))

    def positionEnd(self):
        self.positions_end_flag = True

    """ These functions below are called only when ReqAccountUpdates on
            EEClientSocket object has been called. """

    def updatePortfolioValue(self, value: str):
        self.portfolio_value = float(value)

    # Override the updatePortfolioValue method in the Main class to call it correctly
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        # print(f"Received account value update: Key: {key}, Value: {val}")
        super().updateAccountValue(key, val, currency, accountName)
        if key == "NetLiquidation":
            self.updatePortfolioValue(val)

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float, averageCost: float,
                        unrealizedPNL: float, realizedPNL: float, accountName: str):
        super().updatePortfolio(contract, position, marketPrice,
                                marketValue, averageCost, unrealizedPNL,
                                realizedPNL, accountName)

        print("UpdatePortfolio.", "Symbol:", contract.symbol, "SecType:",
              contract.secType, "Exchange:", contract.exchange,
              "Position:", position, "MarketPrice:",
              marketPrice, "MarketValue:", marketValue, "AverageCost:",
              averageCost, "UnrealizedPNL:", unrealizedPNL, "RealizedPNL:",
              realizedPNL, "AccountName:", accountName)

    def updateAccountTime(self, timeStamp: str):
        super().updateAccountTime(timeStamp)

        timeStamp = self.current_dt.strftime("%Y%m%d %H:%M:%S")
        print("UpdateAccountTime. Time:", timeStamp)

    def accountDownloadEnd(self, accountName: str):
        """This is called after a batch updateAccountValue() and
        updatePortfolio() is sent."""

        super().accountDownloadEnd(accountName)
        print("AccountDownloadEnd. Account:", accountName)


class Client(EClient):
    """
    Client class we can use to write our own functions in to request data from the API.
    """

    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

        self.nyc = timezone('America/New_York')
        self.today = datetime.datetime.today().astimezone(self.nyc)

        self.current_dt = datetime.datetime.today().astimezone(self.nyc)

    def AAPL_STK(self):
        contract = Contract()
        contract.symbol = "AAPL"
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        return contract

    def fxPair(self, pair, sec_type="CASH", exchange="IDEALPRO"):
        contract = Contract()
        contract.symbol = pair.split(".")[0]
        contract.secType = sec_type
        contract.currency = pair.split(".")[1]
        contract.exchange = exchange
        return contract

    def USD_EUR_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "EUR"
        contract.exchange = "IDEALPRO"
        return contract

    def USD_AUD_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def USD_GBP_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def USD_CAD_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def USD_JPY_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def USD_CHF_FX(self):
        contract = Contract()
        contract.symbol = "USD"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_USD_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_AUD_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_GBP_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_CAD_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_JPY_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def EUR_CHF_FX(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_USD_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_EUR_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "EUR"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_GBP_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_CAD_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_JPY_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def AUD_CHF_FX(self):
        contract = Contract()
        contract.symbol = "AUD"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_USD_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_EUR_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "EUR"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_AUD_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_CAD_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_JPY_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def GBP_CHF_FX(self):
        contract = Contract()
        contract.symbol = "GBP"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_USD_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_EUR_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "EUR"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_AUD_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_GBP_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_JPY_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def CAD_CHF_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_USD_FX(self):
        contract = Contract()
        contract.symbol = "JPY"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_EUR_FX(self):
        contract = Contract()
        contract.symbol = "CAD"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_AUD_FX(self):
        contract = Contract()
        contract.symbol = "JPY"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_GBP_FX(self):
        contract = Contract()
        contract.symbol = "JPY"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_CAD_FX(self):
        contract = Contract()
        contract.symbol = "JPY"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def JPY_CHF_FX(self):
        contract = Contract()
        contract.symbol = "JPY"
        contract.secType = "CASH"
        contract.currency = "CHF"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_USD_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_EUR_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "EUR"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_AUD_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "AUD"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_GBP_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "GBP"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_CAD_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "CAD"
        contract.exchange = "IDEALPRO"
        return contract

    def CHF_JPY_FX(self):
        contract = Contract()
        contract.symbol = "CHF"
        contract.secType = "CASH"
        contract.currency = "JPY"
        contract.exchange = "IDEALPRO"
        return contract

    def ind(self):
        contract = Contract()
        contract.symbol = "DAX"
        contract.secType = "IND"
        contract.currency = "EUR"
        contract.exchange = "DTB"
        return contract

    def fut(self):
        contract = Contract()
        contract.symbol = "ES"
        contract.secType = "FUT"
        contract.exchange = "GLOBEX"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = '%Y%m%d'
        return contract

    def cfd(self):
        contract = Contract()
        contract.symbol = "IBDE30"
        contract.secType = "CFD"
        contract.currency = "EUR"
        contract.exchange = "SMART"
        return contract

    def options(self):
        contract = Contract()
        contract.symbol = "GOOG"
        contract.secType = "FOP"
        contract.exchange = "BOX"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = '20220101'  # '%Y%m%d'
        contract.strike = 7.5
        contract.right = "C"
        contract.multiplier = "100"
        return contract

    def get_unique_id(self):
        self.unique_id = + 1
        return self.unique_id

    # def get_order_id(self):
    #     self.next_valid_order_id = + 1
    #     return self.next_valid_order_id

    def get_option_chain(self, ltd: str):
        contract = Contract()
        contract.symbol = "EOE"
        contract.secType = "OPT"
        contract.exchange = "FTA"
        contract.currency = "EUR"
        contract.lastTradeDateOrContractMonth = ltd
        contract.multiplier = "100"
        return contract

    def market_order(self, action, quantity):
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        return order

    def limit_order(self, action, quantity, limit_price):
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        return order

    def bracket_order(self, order_id, action: str, quantity, limit_price, take_profit_price, stop_loss_price):
        parent_order = Order()
        parent_order.orderId = order_id
        parent_order.action = action
        parent_order.orderType = "LMT"
        parent_order.totalQuantity = quantity
        parent_order.lmtPrice = limit_price
        parent_order.eTradeOnly = False
        parent_order.firmQuoteOnly = False
        parent_order.transmit = False

        take_profit_order = Order()
        take_profit_order.orderId = order_id + 1
        take_profit_order.action = "SELL" if action.upper() == "BUY" else "BUY"
        take_profit_order.orderType = "LMT"
        take_profit_order.totalQuantity = quantity
        take_profit_order.lmtPrice = take_profit_price
        take_profit_order.parentId = order_id
        take_profit_order.eTradeOnly = False
        take_profit_order.firmQuoteOnly = False
        take_profit_order.transmit = False

        stop_loss_order = Order()
        stop_loss_order.orderId = order_id + 2
        stop_loss_order.action = "SELL" if action.upper() == "BUY" else "BUY"
        stop_loss_order.orderType = "STP"
        stop_loss_order.auxPrice = stop_loss_price
        stop_loss_order.totalQuantity = quantity
        stop_loss_order.parentId = order_id
        stop_loss_order.eTradeOnly = False
        stop_loss_order.firmQuoteOnly = False
        stop_loss_order.transmit = True

        return [parent_order, take_profit_order, stop_loss_order]

    def TimeCondition(self, isMore, IsConjunction, time: datetime.datetime = None, delta: datetime.timedelta = None):
        if time is not None:
            condition_time = time.astimezone(pytz.utc).strftime("%Y%m%d-%H:%M:%S")
        elif delta is not None:
            condition_time = datetime.datetime.now() + delta
            condition_time = condition_time.astimezone(utc).strftime("%Y%m%d-%H:%M:%S")
        else:
            raise RuntimeError("Time is None and delta is None.")

        time_condition = order_condition.Create(order_condition.OrderCondition.Time)
        time_condition.isMore = isMore
        time_condition.time = condition_time
        time_condition.isConjunctionConnection = IsConjunction

        return time_condition

    def fetchAccountUpdates(self):
        # Call this function to start getting account values, portfolio, and last update time information via EWrapper.updateAccountValue(), EWrapperi.updatePortfolio() and Wrapper.updateAccountTime().
        # When you have a single account structure with LYNX there's no need to fill in an account number
        self.reqAccountUpdates(True, '')


class Main(Wrapper, Client):
    def __init__(self, ip_address, port_id, client_id):
        Wrapper.__init__(self)
        Client.__init__(self, wrapper=self)
        self.ip_address, self.port_id, self.client_id = ip_address, port_id, client_id

        # Connect the first time
        self.hide_error_codes = HIDE_ERROR_CODES
        self.reconnect()

    def reconnect(self):
        """
        Function that initializes the API and starts the connection.
        """
        self.connect(self.ip_address, self.port_id, self.client_id)
        thread = Thread(target=self.run).start()
        setattr(self, "_thread", thread)

    def complete_contract(self, contract: Contract, time_out=10) -> list:
        """
        This function requests the contract details from the api and waits for all the contracts to be received.
        """
        req_id = self.get_unique_id()
        self.contract_details[req_id] = []
        self.reqContractDetails(req_id, contract)

        for i in range(time_out * 100):
            time.sleep(0.01)
            if self.FINISHED in self.contract_details[req_id]:
                self.contract_details[req_id].remove(self.FINISHED)
                return [c.contract for c in self.contract_details[req_id]]
            else:
                continue
        else:
            return []

    def stream_market_data(self, ticker_id, contract):
        self.reqTickByTickData(ticker_id, contract, "MidPoint", 0, False)  # Request market data

    def get_market_data(self, contract: Contract, data_types: list, live_data=False, time_out=10) -> PriceInformation:
        """
        This function requests the market data and handles the data callback.
        """
        unique_id = self.get_unique_id()
        self.market_data[unique_id] = PriceInformation(contract)
        self.reqMarketDataType(1) if live_data else self.reqMarketDataType(3)
        self.reqMktData(unique_id, contract, "", True, False, [])  # select true because

        for i in range(100 * time_out):
            time.sleep(0.01)

            if None in [getattr(self.market_data[unique_id], dt) for dt in data_types]:
                continue
            else:
                break
        return self.market_data.pop(unique_id)

    def get_fx_market_data(self, pair, data_types: list, live_data=False, sec_type="CASH", exchange="IDEALPRO", time_out=10) -> PriceInformation:
        """
        This function requests the market data and handles the data callback.
        """
        contract = self.fxPair(pair, sec_type, exchange)

        unique_id = self.get_unique_id()
        self.market_data[unique_id] = PriceInformation(contract)
        self.reqMarketDataType(1) if live_data else self.reqMarketDataType(3)
        self.reqMktData(unique_id, contract, "", True, False, [])  # select true because

        for i in range(100 * time_out):
            time.sleep(0.01)

            if None in [getattr(self.market_data[unique_id], dt) for dt in data_types]:
                continue
            else:
                break
        return self.market_data.pop(unique_id)

    def get_historical_fx_data(self, pair, duration, bar_size, sec_type="CASH", exchange="IDEALPRO", time_out=50,
                               data_type="TRADES"):
        """
        Function to request historical market data for a given contract.
        """
        contract = self.fxPair(pair, sec_type, exchange)

        req_id = self.get_unique_id()
        self.historical_market_data[req_id] = []
        self.reqHistoricalData(req_id, contract, "", duration, bar_size, data_type, 1, 1, False, [])

        for i in range(time_out * 100):
            time.sleep(0.01)

            if self.FINISHED in self.historical_market_data[req_id]:
                self.historical_market_data[req_id].remove(self.FINISHED)
                return self.historical_market_data[req_id]
        else:
            self.cancelHistoricalData(req_id)
            print("failed to retrieve market data.")
            return []
        pass

    def get_historical_market_data(self, contract: Contract, duration: str = "1 D", bar_size: str = "1 day",
                                   time_out=50, data_type="TRADES") -> list:
        req_id = self.get_unique_id()
        self.historical_market_data[req_id] = []

        self.reqHistoricalData(req_id, contract, "", duration, bar_size, data_type, 1, 1, False, [])

        for i in range(time_out * 100):
            time.sleep(0.01)

            if self.FINISHED in self.historical_market_data[req_id]:
                self.historical_market_data[req_id].remove(self.FINISHED)
                return self.historical_market_data[req_id]
        else:
            self.cancelHistoricalData(req_id)
            print("failed to retrieve market data.")
            return []
        pass

    def bar_to_datetime(self, bar: BarData):
        try:
            dt_with_tz = self.nyc.localize(datetime.datetime.strptime(bar.date, "%Y%m%d %H:%M:%S"))
            return dt_with_tz
        except ValueError:
            dt_with_tz = self.nyc.localize(datetime.datetime.strptime(bar.date, "%Y%m%d"))
            return dt_with_tz

    def get_order_id(self, time_out=10):
        self.next_valid_order_id = None
        self.reqIds(-1)

        for i in range(time_out * 100):
            time.sleep(0.01)
            if self.next_valid_order_id is not None:
                return self.next_valid_order_id
            else:
                continue
        else:
            return None

    def reqOpenOrders(self):
        """
        Request all active orders submitted by the client application.
        """
        open_orders = []

        # Retrieve the open orders from the order_df DataFrame
        for _, row in self.order_df.iterrows():
            order = Order()
            contract = Contract()

            # Populate the order object
            order.permId = row['PermId']
            order.clientId = row['ClientId']
            order.orderId = row['OrderId']
            order.account = row['Account']
            order.action = row['Action']
            order.orderType = row['OrderType']
            order.totalQuantity = row['TotalQty']
            order.cashQty = row['CashQty']
            order.lmtPrice = row['LmtPrice']
            order.auxPrice = row['AuxPrice']

            # Populate the contract object
            contract.symbol = row['Symbol']
            contract.secType = row['SecType']
            contract.exchange = row['Exchange']

            open_orders.append((order, contract))

        return open_orders

    def cancel_order(self, orderId):
        """
        Cancel an order with the specified orderId.
        """
        self.cancelOrder(orderId)  # Call the cancelOrder method from the api_interface module

    def get_pnl(self, time_out=10):
        total_pnl = 0
        positions = self.get_positions()
        self.positions_pnl = {}
        contract_for_id = {}

        for position in positions:
            position: PositionInfo = position
            if position.pos == 0:
                continue
            req_id = self.get_unique_id()
            self.positions_pnl[req_id] = None
            self.reqPnLSingle(req_id, position.account, "", position.contract.conId)
            contract_for_id[req_id] = position.contract

        for i in range(time_out * 100):
            time.sleep(0.01)
            if None not in self.positions_pnl.values():
                break

        for key, pnl in self.positions_pnl.items():
            self.cancelPnLSingle(key)
            if pnl is not None:
                pos, dailyPnL, unrealizedPnL, realizedPnL = pnl
                msg = "PnL for contract: {}, is {}.".format(contract_for_id[key], dailyPnL)
                print(msg)
                total_pnl += dailyPnL

        print("total Pnl : {}.".format(total_pnl))

    def get_positions(self, time_out=10):
        self.positions = []
        self.positions_end_flag = False
        self.reqPositions()

        for i in range(time_out * 100):
            time.sleep(0.01)
            if self.positions_end_flag:
                break

        self.cancelPositions()

        return self.positions

    def error(self, reqId, errorCode, errorString):
        if errorCode not in self.hide_error_codes:
            print("Error Id: {}, Error Code: {}, String: {}".format(reqId, errorCode, errorString))