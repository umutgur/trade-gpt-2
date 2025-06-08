import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add core-app src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core-app/src'))

from config import config
from ta_features import ta_analyzer
from metrics import metrics_calculator
from lstm_model import LSTMPredictor
from llm_strategy import LLMStrategyEngine

class TestConfig:
    """Test configuration module"""
    
    def test_config_validation(self):
        """Test config validation"""
        # Test with empty keys (should fail validation)
        original_openai_key = config.OPENAI_API_KEY
        config.OPENAI_API_KEY = ""
        
        assert not config.validate()
        
        # Restore original key
        config.OPENAI_API_KEY = original_openai_key
    
    def test_database_url(self):
        """Test database URL generation"""
        url = config.database_url
        assert "postgresql://" in url
        assert config.POSTGRES_USER in url
        assert config.POSTGRES_DB in url

class TestTechnicalAnalysis:
    """Test technical analysis module"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15T')
        np.random.seed(42)
        
        # Generate realistic price data
        price = 50000
        prices = [price]
        
        for _ in range(99):
            change = np.random.normal(0, 0.01)
            price = price * (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        return df
    
    def test_add_technical_features(self, sample_data):
        """Test adding technical features"""
        df_with_features = ta_analyzer.add_all_features(sample_data)
        
        # Check that features were added
        assert 'rsi' in df_with_features.columns
        assert 'sma_20' in df_with_features.columns
        assert 'ema_12' in df_with_features.columns
        assert 'macd' in df_with_features.columns
        assert 'bb_upper' in df_with_features.columns
        
        # Check no NaN values (should be filled)
        assert not df_with_features.isnull().any().any()
    
    def test_get_features_for_ml(self, sample_data):
        """Test ML feature preparation"""
        ml_features = ta_analyzer.get_features_for_ml(sample_data)
        
        assert 'features' in ml_features
        assert 'target_price' in ml_features
        assert 'target_return' in ml_features
        assert 'target_direction' in ml_features
        assert 'feature_names' in ml_features
        
        # Check array shapes
        features = ml_features['features']
        target_return = ml_features['target_return']
        
        assert features.shape[0] == target_return.shape[0]
        assert len(ml_features['feature_names']) == features.shape[1]
    
    def test_individual_indicators(self, sample_data):
        """Test individual indicators"""
        close_prices = sample_data['close']
        
        # Test RSI
        rsi = ta_analyzer.rsi(close_prices)
        assert len(rsi) == len(close_prices)
        assert rsi.max() <= 100
        assert rsi.min() >= 0
        
        # Test SMA
        sma = ta_analyzer.sma(close_prices, window=20)
        assert len(sma) == len(close_prices)
        
        # Test EMA
        ema = ta_analyzer.ema(close_prices, window=20)
        assert len(ema) == len(close_prices)

class TestMetricsCalculator:
    """Test metrics calculator module"""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data"""
        np.random.seed(42)
        trades = []
        for i in range(100):
            pnl = np.random.normal(10, 50)  # Random P&L
            trades.append({
                'profit_loss': pnl,
                'price': 50000 + np.random.normal(0, 1000),
                'amount': 0.1,
                'timestamp': datetime.now() - timedelta(days=i)
            })
        return trades
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation"""
        sharpe = metrics_calculator.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert -10 <= sharpe <= 10  # Reasonable range
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation"""
        sortino = metrics_calculator.calculate_sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create equity curve with known drawdown
        equity = pd.Series([1000, 1100, 1200, 900, 800, 1000, 1100])
        
        max_dd, duration = metrics_calculator.calculate_max_drawdown(equity)
        
        assert max_dd > 0  # Should have positive drawdown
        assert duration > 0  # Should have some duration
        assert isinstance(max_dd, float)
        assert isinstance(duration, int)
    
    def test_win_rate(self, sample_trades):
        """Test win rate calculation"""
        win_rate = metrics_calculator.calculate_win_rate(sample_trades)
        
        assert 0 <= win_rate <= 1
        assert isinstance(win_rate, float)
    
    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation"""
        profit_factor = metrics_calculator.calculate_profit_factor(sample_trades)
        
        assert profit_factor >= 0
        assert isinstance(profit_factor, float)
    
    def test_mape_calculation(self):
        """Test MAPE calculation"""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([98, 112, 118, 135, 142])
        
        mape = metrics_calculator.calculate_mape(actual, predicted)
        
        assert 0 <= mape <= 100
        assert isinstance(mape, float)
    
    def test_mae_calculation(self):
        """Test MAE calculation"""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([98, 112, 118, 135, 142])
        
        mae = metrics_calculator.calculate_mae(actual, predicted)
        
        assert mae >= 0
        assert isinstance(mae, float)

class TestLSTMPredictor:
    """Test LSTM predictor module"""
    
    @pytest.fixture
    def sample_features_target(self):
        """Create sample features and target data"""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        features = np.random.randn(n_samples, n_features)
        target = np.random.randn(n_samples)
        
        return features, target
    
    def test_sequence_preparation(self, sample_features_target):
        """Test sequence preparation for LSTM"""
        features, target = sample_features_target
        
        predictor = LSTMPredictor(seq_length=20)
        X, y = predictor.prepare_sequences(features, target)
        
        # Check shapes
        expected_samples = len(features) - predictor.seq_length
        assert X.shape[0] == expected_samples
        assert X.shape[1] == predictor.seq_length
        assert X.shape[2] == features.shape[1]
        assert y.shape[0] == expected_samples
    
    def test_model_building(self, sample_features_target):
        """Test LSTM model building"""
        features, target = sample_features_target
        
        predictor = LSTMPredictor(seq_length=20)
        X, y = predictor.prepare_sequences(features, target)
        
        if len(X) > 0:
            model = predictor.build_model((X.shape[1], X.shape[2]))
            
            assert model is not None
            assert len(model.layers) > 0
            assert model.input_shape[1:] == (X.shape[1], X.shape[2])

class TestLLMStrategyEngine:
    """Test LLM strategy engine module"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='15T')
        np.random.seed(42)
        
        prices = [50000]
        for _ in range(49):
            change = np.random.normal(0, 0.01)
            price = prices[-1] * (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 50)
        })
        
        return df
    
    def test_market_context_preparation(self, sample_market_data):
        """Test market context preparation"""
        engine = LLMStrategyEngine()
        context = engine._prepare_market_context(sample_market_data)
        
        assert 'current_price' in context
        assert 'price_change_1h' in context
        assert 'price_change_4h' in context
        assert 'price_change_24h' in context
        assert 'volatility_24h' in context
        assert 'volume_ratio' in context
        
        # Check data types
        assert isinstance(context['current_price'], (int, float))
        assert isinstance(context['price_change_1h'], (int, float))
    
    def test_default_strategy(self):
        """Test default strategy generation"""
        engine = LLMStrategyEngine()
        default_strategy = engine._get_default_strategy()
        
        required_keys = ['indicators', 'buy_conditions', 'sell_conditions', 
                        'risk_management', 'market_outlook', 'confidence']
        
        for key in required_keys:
            assert key in default_strategy
        
        # Check structure
        assert isinstance(default_strategy['indicators'], list)
        assert isinstance(default_strategy['buy_conditions'], list)
        assert isinstance(default_strategy['sell_conditions'], list)
        assert isinstance(default_strategy['risk_management'], dict)
    
    def test_strategy_validation(self):
        """Test strategy validation"""
        engine = LLMStrategyEngine()
        
        # Create invalid strategy (too many indicators)
        invalid_strategy = {
            'indicators': [{'name': f'IND{i}'} for i in range(10)],  # Too many
            'buy_conditions': [{'condition': 'test', 'threshold': 1, 'operator': '>'}],
            'sell_conditions': [{'condition': 'test', 'threshold': 1, 'operator': '<'}],
            'risk_management': {'stop_loss': 50, 'take_profit': 100},  # Invalid values
            'market_outlook': 'bullish',
            'confidence': 1.5  # Invalid confidence
        }
        
        validated = engine._validate_strategy(invalid_strategy)
        
        # Check limits are enforced
        assert len(validated['indicators']) <= engine.max_indicators
        assert 0 <= validated['confidence'] <= 1
        assert 0.5 <= validated['risk_management']['stop_loss'] <= 10

# Integration test
def test_api_ping():
    """Test basic API connectivity"""
    try:
        import ccxt
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        assert 'last' in ticker
        assert ticker['last'] > 0
    except Exception as e:
        pytest.skip(f"API test skipped due to: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])