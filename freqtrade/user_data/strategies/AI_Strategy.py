# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Import required indicators
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AI_Strategy(IStrategy):
    """
    AI-powered strategy that fetches dynamic configurations from core-app
    """
    INTERFACE_VERSION = 3

    # Strategy parameters
    can_short: bool = True
    
    # Define minimal_roi
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01
    }

    # Define stoploss
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Define timeframe
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Define buy parameters
    buy_rsi_enabled = BooleanParameter(default=True, space="buy")
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    
    buy_ema_enabled = BooleanParameter(default=True, space="buy")
    buy_ema_short = IntParameter(5, 50, default=10, space="buy")
    buy_ema_long = IntParameter(50, 200, default=50, space="buy")
    
    buy_bb_enabled = BooleanParameter(default=True, space="buy")
    buy_bb_width = DecimalParameter(0.5, 3.0, default=2.0, space="buy")

    # Define sell parameters  
    sell_rsi_enabled = BooleanParameter(default=True, space="sell")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    
    sell_ema_enabled = BooleanParameter(default=True, space="sell")
    
    sell_bb_enabled = BooleanParameter(default=True, space="sell")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # EMA - Exponential Moving Average
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=self.buy_bb_width.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=30).mean()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        conditions = []
        
        # RSI condition
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)
        
        # EMA condition
        if self.buy_ema_enabled.value:
            conditions.append(
                (dataframe['ema_10'] > dataframe['ema_50']) &
                (dataframe['close'] > dataframe['ema_10'])
            )
        
        # Bollinger Bands condition
        if self.buy_bb_enabled.value:
            conditions.append(
                (dataframe['close'] <= dataframe['bb_lowerband']) &
                (dataframe['bb_percent'] < 0.2)
            )

        # Volume condition
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        # Short entry conditions (opposite of long)
        short_conditions = []
        
        # RSI condition for short
        if self.buy_rsi_enabled.value:
            short_conditions.append(dataframe['rsi'] > (100 - self.buy_rsi.value))
        
        # EMA condition for short
        if self.buy_ema_enabled.value:
            short_conditions.append(
                (dataframe['ema_10'] < dataframe['ema_50']) &
                (dataframe['close'] < dataframe['ema_10'])
            )
        
        # Bollinger Bands condition for short
        if self.buy_bb_enabled.value:
            short_conditions.append(
                (dataframe['close'] >= dataframe['bb_upperband']) &
                (dataframe['bb_percent'] > 0.8)
            )

        # Volume condition
        short_conditions.append(dataframe['volume'] > 0)

        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_conditions),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        conditions = []
        
        # RSI condition
        if self.sell_rsi_enabled.value:
            conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        
        # EMA condition  
        if self.sell_ema_enabled.value:
            conditions.append(
                (dataframe['ema_10'] < dataframe['ema_50']) &
                (dataframe['close'] < dataframe['ema_10'])
            )
        
        # Bollinger Bands condition
        if self.sell_bb_enabled.value:
            conditions.append(
                (dataframe['close'] >= dataframe['bb_upperband']) &
                (dataframe['bb_percent'] > 0.8)
            )

        # Volume condition
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        # Short exit conditions (opposite of long exit)
        short_exit_conditions = []
        
        # RSI condition for short exit
        if self.sell_rsi_enabled.value:
            short_exit_conditions.append(dataframe['rsi'] < (100 - self.sell_rsi.value))
        
        # EMA condition for short exit
        if self.sell_ema_enabled.value:
            short_exit_conditions.append(
                (dataframe['ema_10'] > dataframe['ema_50']) &
                (dataframe['close'] > dataframe['ema_10'])
            )
        
        # Bollinger Bands condition for short exit
        if self.sell_bb_enabled.value:
            short_exit_conditions.append(
                (dataframe['close'] <= dataframe['bb_lowerband']) &
                (dataframe['bb_percent'] < 0.2)
            )

        # Volume condition
        short_exit_conditions.append(dataframe['volume'] > 0)

        if short_exit_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, short_exit_conditions),
                'exit_short'] = 1

        return dataframe


# Import reduce function
from functools import reduce