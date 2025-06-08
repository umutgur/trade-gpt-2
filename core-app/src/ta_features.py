import pandas as pd
import numpy as np
import ta
from typing import Dict, List
from loguru import logger

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {
            'sma': self.sma,
            'ema': self.ema,
            'rsi': self.rsi,
            'macd': self.macd,
            'bollinger': self.bollinger_bands,
            'stochastic': self.stochastic,
            'atr': self.atr,
            'adx': self.adx,
            'williams_r': self.williams_r,
            'cci': self.cci
        }
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        try:
            df_copy = df.copy()
            
            # Trend indicators
            df_copy = self.add_trend_indicators(df_copy)
            
            # Momentum indicators
            df_copy = self.add_momentum_indicators(df_copy)
            
            # Volatility indicators
            df_copy = self.add_volatility_indicators(df_copy)
            
            # Volume indicators
            df_copy = self.add_volume_indicators(df_copy)
            
            # Price action features
            df_copy = self.add_price_action_features(df_copy)
            
            # Fill NaN values
            df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
            
            logger.info(f"Added technical features, shape: {df_copy.shape}")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators"""
        # Simple Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # ROC (Rate of Change)
        df['roc'] = ta.momentum.roc(df['close'], window=12)
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = keltner.keltner_channel_hband()
        df['kc_lower'] = keltner.keltner_channel_lband()
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        # Volume SMA
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=12)
        
        return df
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(periods=2)
        df['price_change_5'] = df['close'].pct_change(periods=5)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Upper and lower shadows
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Body size
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # Gap detection
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        return df
    
    def get_features_for_ml(self, df: pd.DataFrame, target_col: str = 'close') -> Dict:
        """Prepare features for machine learning"""
        try:
            # Add all technical features
            df_features = self.add_all_features(df)
            
            # Define feature columns (exclude raw OHLCV and metadata)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 'timeframe']
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            
            # Prepare features matrix
            X = df_features[feature_cols].values
            
            # Prepare target (next period return)
            y_price = df_features[target_col].shift(-1).dropna().values
            y_return = df_features[target_col].pct_change().shift(-1).dropna().values
            
            # Direction target (-1, 0, 1)
            y_direction = np.where(y_return > 0.001, 1, np.where(y_return < -0.001, -1, 0))
            
            # Align arrays
            min_len = min(len(X) - 1, len(y_price), len(y_return), len(y_direction))
            X = X[:min_len]
            y_price = y_price[:min_len]
            y_return = y_return[:min_len]
            y_direction = y_direction[:min_len]
            
            return {
                'features': X,
                'target_price': y_price,
                'target_return': y_return,
                'target_direction': y_direction,
                'feature_names': feature_cols,
                'timestamps': df_features['timestamp'].iloc[:min_len].values
            }
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return {}
    
    def sma(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    def ema(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window).mean()
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        return ta.momentum.rsi(series, window=window)
    
    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD Indicator"""
        macd_obj = ta.trend.MACD(series, window_fast=fast, window_slow=slow, window_sign=signal)
        return {
            'macd': macd_obj.macd(),
            'signal': macd_obj.macd_signal(),
            'histogram': macd_obj.macd_diff()
        }
    
    def bollinger_bands(self, series: pd.Series, window: int = 20, std: int = 2) -> Dict:
        """Bollinger Bands"""
        bb = ta.volatility.BollingerBands(series, window=window, window_dev=std)
        return {
            'upper': bb.bollinger_hband(),
            'middle': bb.bollinger_mavg(),
            'lower': bb.bollinger_lband()
        }
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3) -> Dict:
        """Stochastic Oscillator"""
        return {
            'k': ta.momentum.stoch(high, low, close, window=k_window, smooth_window=d_window),
            'd': ta.momentum.stoch_signal(high, low, close, window=k_window, smooth_window=d_window)
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        return ta.volatility.average_true_range(high, low, close, window=window)
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index"""
        return ta.trend.adx(high, low, close, window=window)
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        return ta.momentum.williams_r(high, low, close, lbp=window)
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        return ta.trend.cci(high, low, close, window=window)

# Global instance
ta_analyzer = TechnicalAnalysis()