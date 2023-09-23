import csv
import config
import random
import requests
import json, os
import websocket
import pandas as pd
import asyncio, aiohttp
from pprint import pprint
from TechnicalAnalysis import MomentumHybrid
from Quant_Classifiers import XGBoostClassifier



class ConfigurationManager:
    def __init__(self):
        self.symbols = list(set(config.symbols))
        self.timeframes = list(set(config.timeframes))
        self.baseStreamURL = config.baseStreamURL
        self.updateStreamURL()
        
    def updateStreamURL(self):
        url = self.baseStreamURL
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                url = url + symbol.lower() + "@kline_" + timeframe + "/"   
        self.streamURL = url[:-1]


class InternalDataBase:

    def __init__(self):
        self.internalDataBase = {}

    async def fetchAndStoreCryptoData(self, symbols: list, timeframes: list):
        try:
            session = aiohttp.ClientSession()
            for symbol in symbols:
                for timeframe in timeframes:
                    url = 'https://api.binance.com/api/v3/klines?symbol={}&interval={}'.format(symbol, timeframe)
                    async with session.get(url, ssl=False) as response:
                        data = await response.json()
                        filtered_data = [[int(item) if i == 0 else float(item) for i, item in enumerate(inner_list[:6])] for inner_list in data]
                        df = pd.DataFrame(filtered_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                        df = df[:-1]
                        
                        if timeframe not in self.internalDataBase:
                            self.internalDataBase[timeframe] = {}
                        self.internalDataBase[timeframe][symbol] = df

            await session.close()

        except Exception as e:
            print("InternalData.fetchAndStoreCryptoData | ", e)

    def addToCryptoPriceData(self, candledata: dict):
        symbol = candledata['s'].upper()
        timeframe = candledata['i']
        try:
            new_row = pd.DataFrame([{ 
                'time':int(candledata['t']), 
                'open':float(candledata['o']), 
                'high':float(candledata['h']), 
                'low':float(candledata['l']), 
                'close':float(candledata['c']), 
                'volume':float(candledata['v'])
            }])

            if timeframe in self.internalDataBase:
                if symbol in self.internalDataBase[timeframe]:
                    self.internalDataBase[timeframe][symbol] = pd.concat([self.internalDataBase[timeframe][symbol], new_row], ignore_index=True)
                else:
                    self.internalDataBase[timeframe][symbol] = new_row
            else:
                self.internalDataBase[timeframe] = {symbol: new_row}

            return True

        except Exception as e:
            print("InternalDataBase.addToCryptoPriceData | ", e)
            return None


class WebSocketManager:
    def __init__(self, url: str, streamDataProcessor: callable, streamOpenProcessor: callable, streamCloseProcessor: callable):
    
        """
        WebSocketManager class constructor.

        Args:
        url (str): WebSocket URL to connect to.
        streamDataProcessor (callable): Function to process streaming data.
        streamOpenProcessor (callable): Function to process WebSocket open event.
        streamCloseProcessor (callable): Function to process WebSocket close event.
        params (dict): Dictionary containing parameters to pass to the stream processors.
        """
        self.url = url
        self.streamDataProcessor = streamDataProcessor
        self.streamOpenProcessor = streamOpenProcessor
        self.streamCloseProcessor = streamCloseProcessor
        self.ws = None

    def on_open(self, ws):

        """
        Function to be executed when WebSocket connection is opened.

        Args:
        ws (websocket.WebSocketApp): WebSocket object.
        """

        self.streamOpenProcessor()

    def on_message(self, ws, message):

        """
        Function to be executed when a message is received from the WebSocket connection.

        Args:
        ws (websocket.WebSocketApp): WebSocket object.
        message (str): Message received from WebSocket connection.
        """

        self.streamDataProcessor(message)

    def on_close(self, ws):

        """
        Function to be executed when WebSocket connection is closed.

        Args:
        ws (websocket.WebSocketApp): WebSocket object.
        """

        self.streamCloseProcessor()

    def stopSocket(self):

        """
        Function to stop WebSocket connection.
        """

        if self.ws:
            self.ws.close()
            print("stream stopped")

    def restartSocket(self, newUrl: str, streamDataProcessor: callable, streamOpenProcessor: callable, streamCloseProcessor: callable):

        """
        Function to restart WebSocket connection with a new URL.

        Args:
        newUrl (str): New WebSocket URL to connect to.
        streamDataProcessor (callable): Function to process streaming data.
        streamOpenProcessor (callable): Function to process WebSocket open event.
        streamCloseProcessor (callable): Function to process WebSocket close event.
        params (dict): Dictionary containing parameters to pass to the stream processors.

        """

        self.url = newUrl
        self.streamDataProcessor = streamDataProcessor
        self.streamOpenProcessor = streamOpenProcessor
        self.streamCloseProcessor = streamCloseProcessor

        if self.ws:
            self.ws.close()

        self.url = newUrl
        self.streamDataProcessor = streamDataProcessor
        self.streamOpenProcessor = streamOpenProcessor
        self.streamCloseProcessor = streamCloseProcessor
        self.startSocket()

    def startSocket(self):

        """
        Function to start WebSocket connection.
        """
        self.ws = websocket.WebSocketApp(
            self.url, on_open=self.on_open, 
            on_close=self.on_close, 
            on_message=self.on_message
        )
        self.ws.run_forever()


class TradeSimulator:
    def __init__(self, trade_capital:float, trade_fee:float, trade_log_path:str="trade_log.csv", stoploss: float=0.03, takeprofit: float=0.04):
        self.acc_bal = 0
        self.trade_capital = trade_capital
        self.trade_fee = trade_fee
        self.trade_log_path = trade_log_path
        self.stoploss_percent = stoploss
        self.takeprofit_percent = takeprofit

        self.symbol_timeframe = None
        self.in_position = False
        self.asset = 0
        self.stoploss = None
        self.takeprofit = None

        # Check if the trade log file exists, and create it if not
        if not os.path.isfile(self.trade_log_path):
            with open(self.trade_log_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Symbol', 'Interval', 'Time', 'Price', 'Action', 'Account Balance', 'Trade Fee', 'Stop Loss', 'Take Profit', 'PnL'])

    def log_to_csv(self, symbol, interval, time, price, action, trade_fee, pnl):
        # Log the trade details to the CSV file
        with open(self.trade_log_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([symbol, interval, time, price, action, self.acc_bal, trade_fee, self.stoploss, self.takeprofit, pnl])

    def simulate_trade(self, signal: dict):
        symbol = signal['symbol']
        interval = signal['interval']
        time = signal['time']
        price = signal['price']
        action = signal['action']

        # Perform trade calculations based on the action
        if action == 'buy' and not self.in_position and (self.symbol_timeframe==f"{symbol}_{interval}" or self.symbol_timeframe is None):
            trade_fee = self.trade_fee * self.trade_capital
            self.asset = self.trade_capital * ( (1/price) - self.trade_fee )   
            self.symbol_timeframe=f"{symbol}_{interval}"
            self.in_position = True
            self.stoploss = price - (price*self.stoploss_percent)
            self.takeprofit = price + (price*self.takeprofit_percent)
            self.log_to_csv(symbol, interval, time, price, action, trade_fee, 0)
            print(signal)

        elif self.in_position and self.symbol_timeframe==f"{symbol}_{interval}":
            # if action == 'sell':
            #     trade_fee = self.trade_fee * self.asset * price
            #     trade_revenue = (self.asset*price) * (1-self.trade_fee)
            #     pnl = (trade_revenue-self.trade_capital)
            #     self.acc_bal = self.acc_bal + pnl
            #     self.in_position = False
            #     self.symbol_timeframe = None
            #     self.stoploss = 0
            #     self.takeprofit = 0
            #     self.log_to_csv(symbol, interval, time, price, action, trade_fee, pnl)
            #     print(signal)
                
            if (price >= self.takeprofit) or (price <= self.stoploss):
                trade_fee = self.trade_fee * self.asset * price
                trade_revenue = (self.asset*price) * (1-self.trade_fee)
                pnl = (trade_revenue-self.trade_capital)
                self.acc_bal = self.acc_bal + pnl
                self.in_position = False
                self.symbol_timeframe = None
                self.stoploss = 0
                self.takeprofit = 0
                self.log_to_csv(symbol, interval, time, price, 'sell', trade_fee, pnl)
                signal["action"] = "stoploss or target hit"
                print(signal)

        return self.in_position


class SignalProcessor:
    def __init__(self, model_path: str):
        self.xgb = XGBoostClassifier().load_model(model_path)

    def transform_data(self, signal_data: pd.DataFrame):
        pct_change_between_rows = signal_data.pct_change(fill_method='pad').fillna(0).tail(1)
        latest_data = pct_change_between_rows.loc[:, ['atr', 'atrn', 'atrp', 'atr_avg_n', 'atr_avg_p', 'signal']]
        latest_data.columns = ['signal', 'atr_1', 'atrn_1', 'atrp_1', 'atr_avg_n_1', 'atr_avg_p_1']
        # Update the dtype check to use isinstance
        # if isinstance(latest_data['signal'].dtype, pd.CategoricalDtype):
        #     latest_data['signal'] = latest_data['signal'].astype(int)
        return latest_data

    def trade_action(self, data: pd.DataFrame, price: float, symbol: str, time: int, timeframe: str):

        signal_data = MomentumHybrid(data).generate_signal()
        signal = {
            'symbol': symbol,
            'interval': timeframe,
            'time': time,
            'price': price,
            'action': 'wait'
        }
        if signal_data["signal"].iloc[-1] == 0:
            signal['action'] = 'wait'
            return signal
        else:
            transformed_data_X = self.transform_data(signal_data)
            predicted_signal = self.xgb.predict(transformed_data_X)
            if predicted_signal == 1:
                if signal_data["signal"].iloc[-1] == -1:
                    signal['action'] = 'sell'
                else:
                    signal['action'] = 'buy'
            else:
                signal['action'] = 'wait'

        return signal


class DefaultDriver:
    def __init__(self, signal_generator: callable):
        self.config = ConfigurationManager()
        self.ws_manager = WebSocketManager
        self.db = InternalDataBase()
        self.simulator = TradeSimulator(
            trade_capital=500,
            trade_fee=0.0075
        )
        self.signal_generator = signal_generator
        self.state = False

    def stream_open_processor(self):
        print('Stream Connection Opened\n\n BOT Status : <Active>')

    def stream_data_processor(self, message): 

            json_message = json.loads(message)
            time = json_message['E']
            candle = json_message['k']
            is_candle_closed = candle['x']
            symbol = candle['s'].upper()
            timeframe = candle['i']
            price = float(candle['c'])

            if self.state:
                self.state = self.simulator.simulate_trade({
                    'symbol': symbol,
                    'interval': timeframe,
                    'time': time,
                    'price': price,
                    'action': 'wait'
                })

            if is_candle_closed:
                self.db.addToCryptoPriceData(candle)
                data = self.db.internalDataBase[timeframe][symbol]
                signal = self.signal_generator(
                    data=data, 
                    price=price,
                    symbol=symbol,
                    time=time,
                    timeframe=timeframe
                )
                if signal["action"] in ["buy", "sell"]:
                    self.state = self.simulator.simulate_trade(signal)

    def stream_close_processor(self):
        print('Stream Connection Closed\n BOT Status : <Inactive>')

    def run(self):
        print('\nStarting...', end="\r")
        asyncio.run(self.db.fetchAndStoreCryptoData(self.config.symbols, self.config.timeframes))

        # create an instance of the WebSocketManager class
        self.ws_manager = WebSocketManager(  
            self.config.streamURL,
            self.stream_data_processor,
            self.stream_open_processor,
            self.stream_close_processor
        )

        while True:
            try:
                self.ws_manager.startSocket()
            except Exception as e:
                print(f'WebSocket connection error: {str(e)}')




