# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AI_Strategy(IStrategy):
    """
    AI-generated strategy template that gets populated by Jinja2 templating
    This strategy is dynamically generated based on LLM recommendations
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = {{ strategy.can_short | default(False) }}

    # Minimal ROI designed for the strategy.
    minimal_roi = {{ strategy.minimal_roi | default({"0": 0.10, "40": 0.04, "100": 0.02, "300": 0}) | tojson }}

    # Optimal stoploss designed for the strategy.
    stoploss = {{ strategy.stoploss | default(-0.10) }}

    # Trailing stoploss
    trailing_stop = {{ strategy.trailing_stop | default(False) }}
    trailing_stop_positive = {{ strategy.trailing_stop_positive | default(0.01) }}
    trailing_stop_positive_offset = {{ strategy.trailing_stop_positive_offset | default(0.0) }}
    trailing_only_offset_is_reached = {{ strategy.trailing_only_offset_is_reached | default(False) }}

    # Optimal timeframe for the strategy.
    timeframe = '{{ strategy.timeframe | default("15m") }}'

    # Run "populate_indicators" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # Strategy parameters
    buy_rsi_enabled = BooleanParameter(default={{ strategy.indicators.rsi.enabled | default(True) }}, space="buy", optimize=False)
    buy_rsi = IntParameter(10, 40, default={{ strategy.indicators.rsi.buy_threshold | default(30) }}, space="buy", optimize=False)
    
    sell_rsi_enabled = BooleanParameter(default={{ strategy.indicators.rsi.enabled | default(True) }}, space="sell", optimize=False)
    sell_rsi = IntParameter(60, 90, default={{ strategy.indicators.rsi.sell_threshold | default(70) }}, space="sell", optimize=False)
    
    buy_ema_enabled = BooleanParameter(default={{ strategy.indicators.ema.enabled | default(True) }}, space="buy", optimize=False)
    ema_period = IntParameter(10, 50, default={{ strategy.indicators.ema.period | default(20) }}, space="buy", optimize=False)
    
    buy_macd_enabled = BooleanParameter(default={{ strategy.indicators.macd.enabled | default(False) }}, space="buy", optimize=False)
    sell_macd_enabled = BooleanParameter(default={{ strategy.indicators.macd.enabled | default(False) }}, space="sell", optimize=False)
    
    buy_bb_enabled = BooleanParameter(default={{ strategy.indicators.bollinger.enabled | default(False) }}, space="buy", optimize=False)
    sell_bb_enabled = BooleanParameter(default={{ strategy.indicators.bollinger.enabled | default(False) }}, space="sell", optimize=False)

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will automatically be available in populate_indicators.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # EMA
        dataframe[f'ema_{self.ema_period.value}'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
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

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Volume indicators
        dataframe['ad'] = ta.AD(dataframe)
        dataframe['obv'] = ta.OBV(dataframe)

        # Additional indicators for AI strategy
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)

        # Commodity Channel Index
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        conditions = []

        # RSI condition
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # EMA condition  
        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] > dataframe[f'ema_{self.ema_period.value}'])

        # MACD condition
        if self.buy_macd_enabled.value:
            conditions.append(
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
            )

        # Bollinger Bands condition
        if self.buy_bb_enabled.value:
            conditions.append(
                (dataframe['close'] <= dataframe['bb_lowerband']) &
                (dataframe['bb_percent'] < 0.2)
            )

        # AI Strategy specific conditions (populated by template)
        {% for condition in strategy.buy_conditions %}
        conditions.append({{ condition.condition }})
        {% endfor %}

        # Volume condition
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        conditions = []

        # RSI condition
        if self.sell_rsi_enabled.value:
            conditions.append(dataframe['rsi'] > self.sell_rsi.value)

        # EMA condition
        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] < dataframe[f'ema_{self.ema_period.value}'])

        # MACD condition
        if self.sell_macd_enabled.value:
            conditions.append(
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
            )

        # Bollinger Bands condition
        if self.sell_bb_enabled.value:
            conditions.append(
                (dataframe['close'] >= dataframe['bb_upperband']) &
                (dataframe['bb_percent'] > 0.8)
            )

        # AI Strategy specific conditions (populated by template)
        {% for condition in strategy.sell_conditions %}
        conditions.append({{ condition.condition }})
        {% endfor %}

        # Volume condition
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic
        """
        # AI-based dynamic stoploss (placeholder)
        return {{ strategy.risk_management.stop_loss | default(-0.05) }}

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs):
        """
        Custom exit logic
        """
        # Take profit at target
        if current_profit >= {{ strategy.risk_management.take_profit | default(0.04) }}:
            return 'take_profit'
        
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                **kwargs) -> float:
        """
        Customize leverage for each new trade.
        """
        # Return leverage based on trading mode
        trading_mode = "{{ strategy.trading_mode | default('spot') }}"
        
        if trading_mode == 'margin':
            return min(3.0, max_leverage)
        elif trading_mode == 'futures':
            return min(2.0, max_leverage)
        else:
            return 1.0