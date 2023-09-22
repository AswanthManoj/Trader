import requests, time, pandas as pd
from datetime import datetime
import numpy as np
from pprint import pprint
import random
import pandas_ta as ta
import matplotlib.pyplot as plt
import scipy 
import scipy.stats as stats


 
class TrendlineAnalysis():

    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data

    def check_trend_line(self, support: bool, pivot: int, slope: float, y: np.array):
        # compute sum of differences between line and prices, 
        # return negative val if invalid 
        
        # Find the intercept of the line going through pivot point with given slope
        intercept = -slope * pivot + y[pivot]
        line_vals = slope * np.arange(len(y)) + intercept
        
        diffs = line_vals - y
        
        # Check to see if the line is valid, return -1 if it is not valid.
        if support and diffs.max() > 1e-5:
            return -1.0
        elif not support and diffs.min() < -1e-5:
            return -1.0

        # Squared sum of diffs between data and line 
        err = (diffs ** 2.0).sum()
        return err

    def optimize_slope(self, support: bool, pivot:int , init_slope: float, y: np.array):
        
        # Amount to change slope by. Multiplyed by opt_step
        slope_unit = (y.max() - y.min()) / len(y) 
        
        # Optmization variables
        opt_step = 1.0
        min_step = 0.0001
        curr_step = opt_step # current step
        
        # Initiate at the slope of the line of best fit
        best_slope = init_slope
        best_err = self.check_trend_line(support, pivot, init_slope, y)
        assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

        get_derivative = True
        derivative = None
        while curr_step > min_step:

            if get_derivative:
                # Numerical differentiation, increase slope by very small amount
                # to see if error increases/decreases. 
                # Gives us the direction to change slope.
                slope_change = best_slope + slope_unit * min_step
                test_err = self.check_trend_line(support, pivot, slope_change, y)
                derivative = test_err - best_err
                
                # If increasing by a small amount fails, 
                # try decreasing by a small amount
                if test_err < 0.0:
                    slope_change = best_slope - slope_unit * min_step
                    test_err = self.check_trend_line(support, pivot, slope_change, y)
                    derivative = best_err - test_err

                if test_err < 0.0: # Derivative failed, give up
                    raise Exception("Derivative failed. Check your data. ")

                get_derivative = False

            if derivative > 0.0: # Increasing slope increased error
                test_slope = best_slope - slope_unit * curr_step
            else: # Increasing slope decreased error
                test_slope = best_slope + slope_unit * curr_step
            
            test_err = self.check_trend_line(support, pivot, test_slope, y)
            if test_err < 0 or test_err >= best_err: 
                # slope failed/didn't reduce error
                curr_step *= 0.5 # Reduce step size
            else: # test slope reduced error
                best_err = test_err 
                best_slope = test_slope
                get_derivative = True # Recompute derivative
        
        # Optimize done, return best slope and intercept
        return (best_slope, -best_slope * pivot + y[pivot])
    
    def fit_upper_trendline(self, data: np.array):
        x = np.arange(len(data))
        coefs = np.polyfit(x, data, 1)
        line_points = coefs[0] * x + coefs[1]
        upper_pivot = (data - line_points).argmax() 
        resist_coefs = self.optimize_slope(False, upper_pivot, coefs[0], data)
        return resist_coefs 

    def fit_lower_trendline(self, data: np.array):
        x = np.arange(len(data))
        coefs = np.polyfit(x, data, 1)
        line_points = coefs[0] * x + coefs[1]
        lower_pivot = (data - line_points).argmin() 
        support_coefs = self.optimize_slope(True, lower_pivot, coefs[0], data)
        return support_coefs 

    def fit_trendlines_single(self, data: np.array):
        # find line of best fit (least squared) 
        # coefs[0] = slope,  coefs[1] = intercept 
        x = np.arange(len(data))
        coefs = np.polyfit(x, data, 1)

        # Get points of line.
        line_points = coefs[0] * x + coefs[1]

        # Find upper and lower pivot points
        upper_pivot = (data - line_points).argmax() 
        lower_pivot = (data - line_points).argmin() 
    
        # Optimize the slope for both trend lines
        support_coefs = self.optimize_slope(True, lower_pivot, coefs[0], data)
        resist_coefs = self.optimize_slope(False, upper_pivot, coefs[0], data)

        return (support_coefs, resist_coefs) 

    def fit_trendlines_high_low(self, high: np.array, low: np.array, close: np.array):
        x = np.arange(len(close))
        coefs = np.polyfit(x, close, 1)
        # coefs[0] = slope,  coefs[1] = intercept
        line_points = coefs[0] * x + coefs[1]
        upper_pivot = (high - line_points).argmax() 
        lower_pivot = (low - line_points).argmin() 
        
        support_coefs = self.optimize_slope(True, lower_pivot, coefs[0], low)
        resist_coefs = self.optimize_slope(False, upper_pivot, coefs[0], high)

        return (support_coefs, resist_coefs)

    def get_trendline_data(self, data: pd.DataFrame, lookback_window:int=30)->pd.DataFrame:
        data = data.set_index('datetime')

        # Take natural log of data to resolve price scaling issues
        data[['open', 'high', 'low', 'close']] = np.log(data[['open', 'high', 'low', 'close']])

        support_slope = [np.nan] * len(data)
        resist_slope = [np.nan] * len(data)
        for i in range(lookback_window - 1, len(data)):
            candles = data.iloc[i - lookback_window + 1: i + 1]
            support_coefs, resist_coefs =  self.fit_trendlines_high_low(
                                                                candles['high'], 
                                                                candles['low'], 
                                                                candles['close'])
            support_slope[i] = support_coefs[0]
            support_intercept = support_coefs[1]

            resist_slope[i] = resist_coefs[0]
            resist_intercept = resist_coefs[1]

        data['support_slope'] = support_slope
        data['support_intercept'] = support_intercept
        data['resist_slope'] = resist_slope
        data['resist_intercept'] = resist_intercept

        return data

    def find_trendline(self, data: pd.DataFrame, lookback:int=72)->dict:
        data = data.set_index('datetime')
        # data = data[:-1]

        # Take natural log of data to resolve price scaling issues
        data[['open', 'high', 'low', 'close']] = np.log(data[['open', 'high', 'low', 'close']])
        
        x = [i for i in range(len(data[-lookback:]))]

        sl = np.zeros(len(data[:-lookback]))
        sl[:] = np.nan

        rl = np.zeros(len(data[:-lookback]))
        rl[:] = np.nan

        # Calculate trendlines for close prices and high-low-close prices
        candles = data.iloc[-lookback:-1]  # Last 72 candles in data except the recent candle
        support_coefs_c, resist_coefs_c = self.fit_trendlines_single(candles['close'])
        support_coefs, resist_coefs = self.fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])

        support_line_c = support_coefs_c[0] * np.arange(len(candles)+1) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(len(candles)+1) + resist_coefs_c[1]

        support_line = support_coefs[0] * np.arange(len(candles)+1) + support_coefs[1]
        resist_line = resist_coefs[0] * np.arange(len(candles)+1) + resist_coefs[1]

        data['support_line'] = sl.tolist() + support_line.tolist()
        data['resistance_line'] = rl.tolist() + resist_line.tolist()
        data['datetime'] = data.index
        data = data.reset_index(drop=True)

        return data

    def generate_signal(self, lookback:int=100, use_high_low:bool=True, slope_tolerance:float=0.15, use_atr:bool=False)->pd.DataFrame:
        data = self.data
        close = data['close'].to_numpy()
        high = data['high'].to_numpy()
        low = data['low'].to_numpy()
        dlength = len(data)

        s_tl = np.full(dlength, np.nan)
        r_tl = np.full(dlength, np.nan)
        signal = np.zeros(dlength)
        signal_b_price = np.full(dlength, np.nan)
        signal_s_price = np.full(dlength, np.nan)
        atr = ta.atr(data['high'], data['low'], data['close'], length=12)/2
        last_sig = 0

        for i in range(lookback, len(close)):
            # NOTE window does NOT include the current candle
            window_close = close[i - lookback: i]

            if use_high_low:
                window_high = high[i - lookback: i]
                window_low = low[i - lookback: i]
                s_coefs, r_coefs = self.fit_trendlines_high_low(window_high, window_low, window_close)
            else:
                s_coefs, r_coefs = self.fit_trendlines_single(window_close)

            # Find current value of line, projected forward to current bar
            s_val = s_coefs[1] + lookback * s_coefs[0] 
            r_val = r_coefs[1] + lookback * r_coefs[0] 

            s_tl[i] = s_val
            r_tl[i] = r_val

            if use_atr:
                if (close[i] > (r_val+atr[i-lookback])) and (r_coefs[0] < -slope_tolerance):
                    current_trend_signal = 1.0
                elif (close[i] < (s_val-atr[i-lookback])) and (s_coefs[0] > slope_tolerance):
                    current_trend_signal = -1.0
                else:
                    current_trend_signal = 0
            else:
                if (close[i] > r_val) and (r_coefs[0] < -slope_tolerance):
                    current_trend_signal = 1.0
                elif (close[i] < s_val) and (s_coefs[0] > slope_tolerance):
                    current_trend_signal = -1.0
                else:
                    current_trend_signal = 0

            if current_trend_signal == 1.0 and last_sig == -1.0:
                last_sig = 1.0
                signal[i] = 1.0
                signal_b_price[i] = close[i]
            elif current_trend_signal == -1.0 and last_sig == 1.0:
                last_sig = -1.0
                signal[i] = -1.0
                signal_s_price[i] = close[i]
            elif last_sig == 0:
                last_sig = current_trend_signal
                signal[i] = current_trend_signal 
                if current_trend_signal == 1.0:
                    signal_b_price[i] = close[i]
                elif current_trend_signal == -1.0:
                    signal_s_price[i] = close[i]
        
        signal_data = pd.DataFrame()
        signal_data['support'] = s_tl
        signal_data['resist'] = r_tl
        signal_data['signal'] = signal
        signal_data['signal_b_price'] = signal_b_price
        signal_data['signal_s_price'] = signal_s_price

        return signal_data

class HawkesProcess():
    """
    Class for generating signals based on the Hawkes process.

    Args:
        data (pd.DataFrame): Input data containing necessary columns.

    Attributes:
        data (pd.DataFrame): Input data containing necessary columns.

    Methods:
        hawkes_process(data, kappa)
        vol_signal(close, vol_hawkes, lookback)
        get_trades_from_signal(data, signal)
        generate_signal(lookback)
    """

    def __init__(self, data:pd.DataFrame):
        """
        Initialize the HawkesProcess class.

        Args:
            data (pd.DataFrame): Input data containing necessary columns.
        """
        self.data = data

    def hawkes_process(self, data: pd.Series, kappa: float):
        """
        Generate a Hawkes process based on input data and kappa.

        Args:
            data (pd.Series): Input data for the Hawkes process.
            kappa (float): Decay factor for the Hawkes process.

        Returns:
            pd.Series: Output Hawkes process series.
        """

        assert(kappa > 0.0)
        alpha = np.exp(-kappa)
        arr = data.to_numpy()
        output = np.zeros(len(data))
        output[:] = np.nan
        for i in range(1, len(data)):
            if np.isnan(output[i - 1]):
                output[i] = arr[i]
            else:
                output[i] = output[i - 1] * alpha + arr[i]
        return pd.Series(output, index=data.index) * kappa

    def vol_signal(self, close: pd.Series, vol_hawkes: pd.Series, lookback:int):
        """
        Generate volatility-based trading signals.

        Args:
            close (pd.Series): Close prices.
            vol_hawkes (pd.Series): Volatility Hawkes process.
            lookback (int): Lookback period for signals.

        Returns:
            tuple: Signal array, signal buy prices, signal sell prices.
        """
        dlength = len(close)
        signal = np.zeros(dlength)
        signal_b_price = np.full(dlength, np.nan)
        signal_s_price = np.full(dlength, np.nan)
        q05 = vol_hawkes.rolling(lookback).quantile(0.05)
        q95 = vol_hawkes.rolling(lookback).quantile(0.95)
        
        last_below = -1
        last_sig = 0

        for i in range(len(signal)):
            if vol_hawkes.iloc[i] < q05.iloc[i]:
                last_below = i

            if (vol_hawkes.iloc[i] > q95.iloc[i]) and (vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1]) and (last_below > 0):
                change = close.iloc[i] - close.iloc[last_below]

                if last_sig == 0:
                    if (change > 0.0):
                        signal_b_price[i] = close.iloc[i]
                        last_sig = -1
                        signal[i] = -1
                    elif (change < 0.0) and ():
                        signal_s_price[i] = close.iloc[i]
                        last_sig = 1
                        signal[i] = 1
                else:
                    if (change > 0.0) and (last_sig == 1):
                        signal_b_price[i] = close.iloc[i]
                        last_sig = -1
                        signal[i] = -1
                    elif (change < 0.0) and (last_sig == -1):
                        signal_s_price[i] = close.iloc[i]
                        last_sig = 1
                        signal[i] = 1

        return signal, signal_b_price, signal_s_price
 
    def get_trades_from_signal(self, data: pd.DataFrame, signal: np.array):
        """
        Get trade entry and exit times from a trading signal.

        Args:
            data (pd.DataFrame): Input data containing close prices.
            signal (np.array): Trading signal array.

        Returns:
            tuple: DataFrames of long and short trades.
        """
        # Gets trade entry and exit times from a signal
        # that has values of -1, 0, 1. Denoting short,flat,and long.
        # No position sizing.

        long_trades = []
        short_trades = []

        close_arr = data['close'].to_numpy()
        last_sig = 0.0
        open_trade = None
        idx = data.index
        for i in range(len(data)):
            if signal[i] == 1.0 and last_sig != 1.0: # Long entry
                if open_trade is not None:
                    open_trade[2] = idx[i]
                    open_trade[3] = close_arr[i]
                    short_trades.append(open_trade)

                open_trade = [idx[i], close_arr[i], -1, np.nan]
            if signal[i] == -1.0  and last_sig != -1.0: # Short entry
                if open_trade is not None:
                    open_trade[2] = idx[i]
                    open_trade[3] = close_arr[i]
                    long_trades.append(open_trade)

                open_trade = [idx[i], close_arr[i], -1, np.nan]
            
            if signal[i] == 0.0 and last_sig == -1.0: # Short exit
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)
                open_trade = None

            if signal[i] == 0.0  and last_sig == 1.0: # Long exit
                open_trade[2] = idx[i]
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)
                open_trade = None

            last_sig = signal[i]

        long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
        short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

        long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price']
        short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
        long_trades = long_trades.set_index('entry_time')
        short_trades = short_trades.set_index('entry_time')
        return long_trades, short_trades

    def generate_signal(self, lookback:int=336)->pd.DataFrame:
        """
        Generate trading signals based on the Hawkes process.

        Args:
            lookback (int): Lookback period for generating signals.

        Returns:
            pd.DataFrame: DataFrame containing generated signals.
        """

        data = self.data

        # Normalize volume
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], lookback) 
        data['norm_range'] = (data['high'] - data['low']) / data['atr']

        data['volume_hawk'] = self.hawkes_process(data['norm_range'], 0.1)
        signal, signal_b_price, signal_s_price = self.vol_signal(data['close'], data['volume_hawk'], 168)

        signal_data = pd.DataFrame()
        signal_data['atr'] = data['atr']
        signal_data['norm_range'] = data['norm_range']
        signal_data['volume_hawk'] = data['volume_hawk']
        signal_data['signal'] = signal
        signal_data['signal_b_price'] = signal_b_price
        signal_data['signal_s_price'] = signal_s_price

        return signal_data

class MomentumHybrid():

    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data

    def calculate_macd(self, data, macd_fast_length:int, macd_slow_length:int, macd_signal_smooth:int)->pd.DataFrame:
        try:
            data["ema_fast"] = data["close"].ewm(span=macd_fast_length, adjust=False).mean()
            data["ema_slow"] = data["close"].ewm(span=macd_slow_length, adjust=False).mean()
            data["macd"] = data["ema_fast"] - data["ema_slow"]
            data["macd_signal"] = data["macd"].ewm(span=macd_signal_smooth, adjust=False).mean()
            data["histogram"] = data["macd"] - data["macd_signal"]
            return data
        except:
            raise f"Error in MomentumHybrid.calculate_macd | {Exception}"
        
    def calculate_atrs(self, data, atr_length:int, b_atr_length:int, b_atr_avg_length:int, b_multiplier:float, s_atr_length:int, s_atr_avg_length:int, s_multiplier:float)->pd.DataFrame:
        try:
            data['tr'] = ta.true_range(data['high'], data['low'], data['close'])
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=atr_length)

            data['atrn'] = ta.atr(data['high'], data['low'], data['close'], length=b_atr_length)*(-1*b_multiplier)
            data["atr_avg_n"] = data["atrn"].ewm(span=b_atr_avg_length, adjust=True).mean()
            data["b_histogram"] = (data['macd_signal']-data["atr_avg_n"]).round(2)

            data['atrp'] = ta.atr(data['high'], data['low'], data['close'], length=s_atr_length)*(s_multiplier)
            data["atr_avg_p"] = data["atrp"].ewm(span=s_atr_avg_length, adjust=True).mean()
            data["s_histogram"] = (data['macd_signal']-data["atr_avg_p"]).round(2)
            return data
        except:
            raise f"Error in MomentumHybrid.calculate_atrs | {Exception}"

    def generate_signal(self, macd_fast_length:int=12, macd_slow_length:int=26, macd_signal_smooth:int=9, atr_length:int=12, b_atr_length:int=10, b_atr_avg_length:int=12, b_multiplier:float=1.25, s_atr_length:int=10, s_atr_avg_length:int=12, s_multiplier:float=1.25)->pd.DataFrame:
        try:
            data = self.data
            dlength = len(data)
            signal = np.full(dlength, 0)
            signal_b_price = np.full(dlength, np.nan)
            signal_s_price = np.full(dlength, np.nan)

            macd_data = self.calculate_macd(data, 
                                            macd_fast_length, 
                                            macd_slow_length, 
                                            macd_signal_smooth)
            atrs_data = self.calculate_atrs(macd_data,
                                            atr_length, 
                                            b_atr_length, 
                                            b_atr_avg_length, 
                                            b_multiplier, 
                                            s_atr_length, 
                                            s_atr_avg_length, 
                                            s_multiplier)

            for i in range(2, len(data)):
                if (atrs_data['b_histogram'].iloc[i-2] < 0) and (atrs_data['b_histogram'].iloc[i-1] < 0) and (atrs_data['b_histogram'].iloc[i] < 0):
                    if (atrs_data['b_histogram'].iloc[i-2] > atrs_data['b_histogram'].iloc[i-1]) and (atrs_data['b_histogram'].iloc[i-1] < atrs_data['b_histogram'].iloc[i]):
                        # buy signal
                        signal[i] = 1
                        signal_b_price[i] = atrs_data['close'].iloc[i]
                elif (atrs_data['s_histogram'].iloc[i-2] > 0) and (atrs_data['s_histogram'].iloc[i-1] > 0) and (atrs_data['s_histogram'].iloc[i] > 0):
                    if (atrs_data['s_histogram'].iloc[i-2] < atrs_data['s_histogram'].iloc[i-1]) and (atrs_data['s_histogram'].iloc[i-1] > atrs_data['s_histogram'].iloc[i]):
                        # sell signal
                        signal[i] = -1
                        signal_s_price[i] = atrs_data['close'].iloc[i]
                
            signal_data = pd.DataFrame()
            signal_data['macd_ema_fast'] = atrs_data['ema_fast']
            signal_data['macd_ema_slow'] = atrs_data['ema_slow']
            signal_data['macd'] = atrs_data['macd']
            signal_data['macd_signal'] = atrs_data['macd_signal']
            signal_data['macd_histogram'] = atrs_data['histogram']
            signal_data['tr'] = atrs_data['tr']
            signal_data['atr'] = atrs_data['atr']
            signal_data['atrn'] = atrs_data['atrn']
            signal_data['atr_avg_n'] = atrs_data['atr_avg_n']
            signal_data['b_histogram'] = atrs_data['b_histogram']
            signal_data['atrp'] = atrs_data['atrp']
            signal_data['atr_avg_p'] = atrs_data['atr_avg_p']
            signal_data['s_histogram'] = atrs_data['s_histogram']
            signal_data['signal'] = signal
            signal_data['signal_b_price'] = signal_b_price
            signal_data['signal_s_price'] = signal_s_price
            
            return signal_data
        except:
            raise f"Error in MomentumHybrid.generate_signal | {Exception}"

class VolumeSpreadHybrid():

    def __init__(self, data:pd.DataFrame):
        self.data = data

    def vsa_indicator(self, norm_lookback:int=168)->pd.DataFrame:
        try:
            # Norm lookback should be fairly large
            data = self.data
            
            atr = ta.atr(data['high'], data['low'], data['close'], norm_lookback)
            vol_med = data['volume'].rolling(norm_lookback).median()

            data['norm_range'] = (data['high'] - data['low']) / atr 
            data['norm_volume'] = data['volume'] / vol_med 

            norm_vol = data['norm_volume'].to_numpy()
            norm_range = data['norm_range'].to_numpy()

            range_dev = np.zeros(len(data))
            range_dev[:] = np.nan

            for i in range(norm_lookback * 2, len(data)):
                window = data.iloc[i - norm_lookback + 1: i+ 1]
                slope, intercept, r_val,_,_ = stats.linregress(window['norm_volume'], window['norm_range'])

                if slope <= 0.0 or r_val < 0.2:
                    range_dev[i] = 0.0
                    continue
            
                pred_range = intercept + slope * norm_vol[i]
                range_dev[i] = norm_range[i] - pred_range

            data['dev'] = range_dev
            df = pd.DataFrame({'norm_range':norm_range, 'norm_volume':norm_vol, 'vsa':range_dev})
            return df
        except:
            raise f"Error in VolumeSpread.generate_signal | {Exception}"

    def generate_signal(self, norm_lookback:int=168, threshold:float=-0.83)->pd.DataFrame:
        try:
            data = self.data
            dlength = len(data)
            signal = np.full(dlength, 0)
            signal_price = np.full(dlength, np.nan)
            vsa_data = self.vsa_indicator(norm_lookback)

            for i in range(dlength):
                if vsa_data['vsa'].iloc[i] < threshold:
                    signal[i] = 1
                    signal_price[i] = data['close'].iloc[i]
            
            vsa_data['signal'] = signal
            vsa_data['signal_price'] = signal_price
            return vsa_data
        except:
            raise f"Error in VolumeSpread.generate_signal | {Exception}"

    # def plot_around(data: pd.DataFrame, i: int, above: bool, threshold: float = 0.90):
    #     if above:
    #         extremes = data[data['dev'] > threshold]
    #     else:
    #         extremes = data[data['dev'] < -threshold]
        
    #     if i >= len(extremes):
    #         raise ValueError(f"i is too big, use less than {len(extremes)}")
    #     t =  extremes.index[i]
    #     td = pd.Timedelta(hours=24)
    #     surrounding = data.loc[t - td: t + td]

class PricePatternAnalysis():
    
    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.dlength = len(data)
 
    def rw_top(self, curr_index: int, order: int) -> bool: # Checks if there is a local top detected at curr index
        data = self.data['close'].to_numpy()
        if curr_index < order * 2 + 1:
            return False
        top = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] > v or data[k - i] > v:
                top = False
                break
        return top
  
    def rw_bottom(self, curr_index: int, order: int) -> bool: # Checks if there is a local top detected at curr index
        data = self.data['close'].to_numpy()
        if curr_index < order * 2 + 1:
            return False
        bottom = True
        k = curr_index - order
        v = data[k]
        for i in range(1, order + 1):
            if data[k + i] < v or data[k - i] < v:
                bottom = False
                break
        return bottom

    def generate_rolling_window_pattern(self, order:int=12):
        # Rolling window local tops and bottoms
        # tops = []
        # bottoms = []
        data = self.data['close'].to_numpy()
        signal = np.zeros_like(data, dtype=float)
        signal_bottom_price = np.full(self.dlength, np.nan)
        signal_top_price = np.full(self.dlength, np.nan)

        for i in range(self.dlength):
            if self.rw_top(i, order):
                # top[0] = confirmation index
                # top[1] = index of top
                # top[2] = price of top
                # top = [i, i - order, data[i - order]]
                signal[i - order] = -1
                signal_top_price[i - order] = data[i - order]
                # tops.append(top)
            
            if self.rw_bottom(i, order):
                # bottom[0] = confirmation index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                # bottom = [i, i - order, data[i - order]]
                signal[i - order] = 1
                signal_bottom_price[i - order] = data[i - order]
                # bottoms.append(bottom)

        signal_data = pd.DataFrame()
        signal_data['signal'] = signal
        signal_data['signal_bottom_price'] = signal_bottom_price
        signal_data['signal_top_price'] = signal_top_price
        return signal_data

    def generate_directional_change(self, sigma:float=0.02, enable_atr:bool=True, atr_multiplier:float=0.5):
        close, high, low = self.data['close'].to_numpy(), self.data['high'].to_numpy(), self.data['low'].to_numpy()   
        up_zig = True # Last extreme is a bottom. Next is a top. 
        tmp_max = high[0]
        tmp_min = low[0]
        tmp_max_i = 0
        tmp_min_i = 0
        signal = np.zeros_like(close, dtype=float)
        signal_bottom_price = np.full(self.dlength, np.nan)
        signal_top_price = np.full(self.dlength, np.nan)
        atr = ta.atr(self.data['high'], self.data['low'], self.data['close'], 1).to_numpy()
        # tops = []
        # bottoms = []

        for i in range(self.dlength):
            if enable_atr:
                sigma = atr[i] * atr_multiplier
            if up_zig: # Last extreme is a bottom
                if high[i] > tmp_max:
                    # New high, update 
                    tmp_max = high[i]
                    tmp_max_i = i
                elif close[i] < tmp_max - tmp_max * sigma: 
                    # Price retraced by sigma %. Top confirmed, record it
                    # top[0] = confirmation index
                    # top[1] = index of top
                    # top[2] = price of top
                    # top = [i, tmp_max_i, tmp_max]
                    # tops.append(top)
                    signal[tmp_max_i] = -1
                    signal_top_price[tmp_max_i] = tmp_max

                    # Setup for next bottom
                    up_zig = False
                    tmp_min = low[i]
                    tmp_min_i = i
            else: # Last extreme is a top
                if low[i] < tmp_min:
                    # New low, update 
                    tmp_min = low[i]
                    tmp_min_i = i
                elif close[i] > tmp_min + tmp_min * sigma: 
                    # Price retraced by sigma %. Bottom confirmed, record it
                    # bottom[0] = confirmation index
                    # bottom[1] = index of bottom
                    # bottom[2] = price of bottom
                    # bottom = [i, tmp_min_i, tmp_min]
                    # bottoms.append(bottom)
                    signal[tmp_min_i] = 1
                    signal_bottom_price[tmp_min_i] = tmp_min

                    # Setup for next top
                    up_zig = True
                    tmp_max = high[i]
                    tmp_max_i = i
        
        signal_data = pd.DataFrame()
        signal_data['signal'] = signal
        signal_data['signal_top_price'] = signal_top_price
        signal_data['signal_bottom_price'] = signal_bottom_price

        return signal_data



import random
import math

class GeneticAlgorithm:
    """
    ## Usage of Optimizer
    ```python
    # Define the function to be optimized
    def rosenbrock(x):
        return -x**2  # Negate the function since we're maximizing

    # Define the search space
    parameters = {
        'x': (10, 0)
    }

    # Define the genetic algorithm parameters
    population_size = 100
    num_generations = 500
    mutation_rate = 0.1

    # Run the genetic algorithm
    best_parameters = GeneticAlgorithm.run_genetic_algorithm(
        function_to_maximize=rosenbrock,
        parameters=parameters,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate
    )

    print(best_parameters)
    ```
    """
    @classmethod
    def generate_initial_population(cls, parameters, population_size):
        """
        Generates the initial population of individuals.
        """
        try:
            population = []
            for _ in range(population_size):
                individual = {
                    param: random.uniform(min_val, max_val)
                    for param, (min_val, max_val) in parameters.items()
                }
                population.append(individual)
            return population
        except:
            raise Exception

    @staticmethod
    def evaluate_fitness(population, function_to_maximize):
        """
        Evaluates the fitness scores for the population.
        """
        try:
            fitness_scores = []
            for individual in population:
                fitness_scores.append(function_to_maximize(**individual))
            return fitness_scores
        except:
            raise Exception

    @classmethod
    def tournament_selection(cls, population, fitness_scores, num_parents):
        """
        Performs tournament selection to choose parents from the population.
        """
        try:
            selected_parents = []
            population_size = len(population)
            for _ in range(num_parents):
                tournament = random.choices(list(range(population_size)), k=3)
                selected_parent = tournament[0]
                for competitor in tournament[1:]:
                    if fitness_scores[competitor] > fitness_scores[selected_parent]:
                        selected_parent = competitor
                selected_parents.append(population[selected_parent])
            return selected_parents
        except:
            raise Exception

    @staticmethod
    def crossover(parent1, parent2, parameters):
        """
        Performs crossover between two parents to generate offspring.
        """
        try:
            offspring = {}
            for param in parameters:
                if random.random() < 0.5:
                    offspring[param] = parent1[param]
                else:
                    offspring[param] = parent2[param]
            return offspring
        except:
            raise Exception

    @staticmethod
    def mutate(individual, parameters, mutation_rate):
        """
        Performs mutation on an individual by randomly modifying its parameter values.
        """
        try:
            for param in individual:
                if random.random() < mutation_rate:
                    min_val, max_val = parameters[param]
                    individual[param] = random.uniform(min_val, max_val)
            return individual
        except:
            raise Exception

    @classmethod
    def run_genetic_algorithm(cls, function_to_maximize, parameters, population_size, num_generations, mutation_rate):
        """
        Runs the genetic algorithm to optimize the parameters of a given function.
        """
        try:
            # dictionary to save parameters best in each generation
            best_of_gens = {key: [] for key in parameters.keys()}
            best_parameters = {'delta':-99999999, 'parameters':{}}

            # generate initial population
            population = cls.generate_initial_population(parameters, population_size)

            for generation in range(num_generations):
                fitness_scores = cls.evaluate_fitness(population, function_to_maximize)
                best_fitness = max(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

                if (not math.isnan(best_fitness)) and (best_fitness>best_parameters['delta']):
                    best_parameters['delta'] = best_fitness
                    best_parameters['parameters'] = best_individual

                print(f"Generation {generation+1}: \n\t Best Fitness = {best_fitness:.2f}, Best Individual = {best_individual}")

                selected_parents = cls.tournament_selection(population, fitness_scores, num_parents=2)
                offspring = []

                for i in range(0, population_size, 2):
                    parent1 = selected_parents[i % len(selected_parents)]
                    parent2 = selected_parents[(i + 1) % len(selected_parents)]
                    child1 = cls.crossover(parent1, parent2, parameters)
                    child2 = cls.crossover(parent2, parent1, parameters)
                    offspring.extend([cls.mutate(child1, parameters, mutation_rate), cls.mutate(child2, parameters, mutation_rate)])

                population = offspring

            # Best parameters after each generation
            fitness_scores = cls.evaluate_fitness(population, function_to_maximize)

            return best_parameters
        except:
            raise Exception



