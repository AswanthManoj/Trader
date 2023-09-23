import json
import asyncio
import warnings
from ArwenCore import TradeSimulator
from ArwenCore import SignalProcessor
from ArwenCore import WebSocketManager
from ArwenCore import InternalDataBase
from ArwenCore import ConfigurationManager


warnings.filterwarnings('ignore')



signal_generator = SignalProcessor("xgboost_classifier.joblib").trade_action
config = ConfigurationManager()
ws_manager = WebSocketManager
db = InternalDataBase()
simulator = TradeSimulator(
    trade_capital=500,
    trade_fee=0.0075,
    trade_log_path="trade_log.csv",
    stoploss=0.03,
    takeprofit=0.04
)
state = False



def stream_open_processor():
    print('Stream Connection Opened\n\n BOT Status : <Active>')

def stream_data_processor(message): 
        global state
        json_message = json.loads(message)
        time = json_message['E']
        candle = json_message['k']
        is_candle_closed = candle['x']
        symbol = candle['s'].upper()
        timeframe = candle['i']
        price = float(candle['c'])

        if state:
            state = simulator.simulate_trade({
                'symbol': symbol,
                'interval': timeframe,
                'time': time,
                'price': price,
                'action': 'wait'
            })

        if is_candle_closed:
            db.addToCryptoPriceData(candle)
            data = db.internalDataBase[timeframe][symbol]
            signal = signal_generator(
                data=data, 
                price=price,
                symbol=symbol,
                time=time,
                timeframe=timeframe
            )
            if signal["action"] in ["buy", "sell"]:
                state = simulator.simulate_trade(signal)

def stream_close_processor():
    print('Stream Connection Closed\n BOT Status : <Inactive>')

def run():
    print('\nStarting...', end="\r")
    asyncio.run(db.fetchAndStoreCryptoData(config.symbols, config.timeframes))
    ws_manager = WebSocketManager(  
        config.streamURL,
        stream_data_processor,
        stream_open_processor,
        stream_close_processor
    )
    while True:
        try:
            ws_manager.startSocket()
        except Exception as e:
            print(f'WebSocket connection error: {str(e)}')


if __name__ == "__main__":
    run()
