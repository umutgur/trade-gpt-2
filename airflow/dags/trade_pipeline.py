from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os
# Heavy imports moved to function level to avoid DagBag timeout
# import pandas as pd  # Moved to functions
# import numpy as np   # Moved to functions
from loguru import logger

# Add core-app src to path (assuming mounted volume)
sys.path.append('/opt/airflow/dags')

# Lazy import function to avoid DagBag timeout
def get_core_modules():
    """Import core modules when needed (lazy loading)"""
    try:
        import sys
        sys.path.append('/app/src')
        from config import config
        from data_fetcher import data_fetcher
        from ta_features import ta_analyzer
        from llm_strategy import llm_engine
        from lstm_model import model_manager
        from backtest_runner import backtest_runner
        from metrics import metrics_calculator
        from db import db_manager
        
        return {
            'config': config,
            'data_fetcher': data_fetcher,
            'ta_analyzer': ta_analyzer,
            'llm_engine': llm_engine,
            'model_manager': model_manager,
            'backtest_runner': backtest_runner,
            'metrics_calculator': metrics_calculator,
            'db_manager': db_manager
        }
    except ImportError as e:
        logger.error(f"Import error: {e}")
        # Create dummy implementations for testing
        class DummyConfig:
            SYMBOLS = ["BTC/USDT", "ETH/USDT"]
            TIMEFRAME = "15m"
            SEQ_LENGTH = 60
            RETRAIN_EVERY = 96
        
        return {'config': DummyConfig()}
    
    # Create dummy functions
    def dummy_function(*args, **kwargs):
        logger.info("Using dummy function due to import issues")
        return {}
    
    data_fetcher = type('DummyDataFetcher', (), {
        'fetch_ohlcv': dummy_function,
        'save_to_database': dummy_function
    })()
    
    ta_analyzer = type('DummyTAAnalyzer', (), {
        'add_all_features': dummy_function,
        'get_features_for_ml': dummy_function
    })()
    
    llm_engine = type('DummyLLMEngine', (), {
        'generate_strategy': dummy_function,
        'provide_feedback': dummy_function
    })()
    
    model_manager = type('DummyModelManager', (), {
        'should_retrain': lambda *args: False,
        'train_model_for_symbol': dummy_function
    })()
    
    backtest_runner = type('DummyBacktestRunner', (), {
        'download_data': lambda *args: True,
        'run_backtest': dummy_function
    })()
    
    metrics_calculator = type('DummyMetricsCalculator', (), {
        'get_performance_summary': dummy_function,
        'calculate_trading_metrics': dummy_function,
        'save_metrics': dummy_function
    })()
    
    db_manager = type('DummyDBManager', (), {
        'get_latest_data': lambda *args: [],
        'save_strategy': lambda *args: 1
    })()

# Default arguments for the DAG
default_args = {
    'owner': 'trade-system',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'execution_timeout': timedelta(minutes=45),  # Reduced default timeout with optimized models
}

# Create main training DAG (every 4-6 hours)
dag = DAG(
    'crypto_training_pipeline',
    default_args=default_args,
    description='Crypto AI Model Training Pipeline',
    schedule_interval='0 */4 * * *',  # Every 4 hours
    catchup=False,
    max_active_runs=1,
    tags=['training', 'crypto', 'ai', 'lstm']
)

def fetch_market_data(**context):
    """Fetch latest market data for all symbols"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        data_fetcher = modules['data_fetcher']
        
        symbols = config.SYMBOLS
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            
            # Fetch latest OHLCV data
            df = data_fetcher.fetch_ohlcv(symbol, limit=500)
            
            if not df.empty:
                # Save to database
                success = data_fetcher.save_to_database(df)
                results[symbol] = {
                    'success': success,
                    'records': len(df),
                    'latest_price': df['close'].iloc[-1] if len(df) > 0 else 0
                }
                logger.info(f"Fetched {len(df)} records for {symbol}")
            else:
                results[symbol] = {'success': False, 'records': 0, 'latest_price': 0}
                logger.warning(f"No data fetched for {symbol}")
        
        # Store results in XCom for next task
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_market_data: {e}")
        raise

def prepare_features(**context):
    """Prepare technical analysis features"""
    import pandas as pd  # Import here to avoid DagBag timeout
    
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        db_manager = modules['db_manager']
        ta_analyzer = modules['ta_analyzer']
        
        fetch_results = context['task_instance'].xcom_pull(task_ids='fetch_data')
        symbols = config.SYMBOLS
        prepared_data = {}
        
        for symbol in symbols:
            if fetch_results.get(symbol, {}).get('success', False):
                logger.info(f"Preparing features for {symbol}")
                
                # Get latest data from database
                market_data = db_manager.get_latest_data(symbol, limit=500)
                
                if market_data:
                    # Convert to DataFrame
                    df = pd.DataFrame([{
                        'timestamp': item.timestamp.isoformat() if hasattr(item.timestamp, 'isoformat') else str(item.timestamp),
                        'open': item.open_price,
                        'high': item.high_price,
                        'low': item.low_price,
                        'close': item.close_price,
                        'volume': item.volume
                    } for item in market_data])
                    
                    # Convert timestamp column to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    df = df.sort_values('timestamp')
                    
                    # Add technical features
                    df_with_features = ta_analyzer.add_all_features(df)
                    
                    # Prepare features for ML
                    ml_features = ta_analyzer.get_features_for_ml(df_with_features)
                    
                    # Convert DataFrame to JSON-serializable format
                    df_records = df_with_features.copy()
                    # Convert timestamp to string for JSON serialization
                    if 'timestamp' in df_records.columns:
                        df_records['timestamp'] = df_records['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    prepared_data[symbol] = {
                        'dataframe': df_records.to_dict('records'),
                        'ml_features': {
                            'features': ml_features.get('features', []).tolist() if hasattr(ml_features.get('features', []), 'tolist') else ml_features.get('features', []),
                            'target_return': ml_features.get('target_return', []).tolist() if hasattr(ml_features.get('target_return', []), 'tolist') else ml_features.get('target_return', [])
                        } if ml_features else {},
                        'latest_price': float(df['close'].iloc[-1]),
                        'data_points': len(df)
                    }
                    
                    logger.info(f"Prepared {len(df_with_features)} feature records for {symbol}")
                else:
                    logger.warning(f"No market data found for {symbol}")
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Error in prepare_features: {e}")
        raise

def generate_strategies(**context):
    """Generate trading strategies using LLM"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        llm_engine = modules['llm_engine']
        metrics_calculator = modules['metrics_calculator']
        db_manager = modules['db_manager']
        
        # Import pandas here
        import pandas as pd
        
        prepared_data = context['task_instance'].xcom_pull(task_ids='prepare_data')
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        strategies = {}
        
        for symbol in symbols:
            if symbol in prepared_data:
                symbol_data = prepared_data[symbol]
                # Reconstruct DataFrame from dict
                df = pd.DataFrame(symbol_data['dataframe'])
                # Convert timestamp back to datetime if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                strategies[symbol] = {}
                
                for trading_mode in trading_modes:
                    logger.info(f"Generating strategy for {symbol} ({trading_mode})")
                    
                    # Get performance summary
                    performance_summary = metrics_calculator.get_performance_summary(
                        symbol, trading_mode, days=30
                    )
                    
                    # Generate strategy
                    strategy = llm_engine.generate_strategy(
                        symbol=symbol,
                        market_data=df,
                        performance_summary=performance_summary,
                        trading_mode=trading_mode
                    )
                    
                    strategies[symbol][trading_mode] = strategy
                    
                    logger.info(f"Generated strategy for {symbol} ({trading_mode})")
        
        return strategies
        
    except Exception as e:
        logger.error(f"Error in generate_strategies: {e}")
        raise

def run_backtests(**context):
    """Run backtests for generated strategies"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        backtest_runner = modules['backtest_runner']
        
        strategies = context['task_instance'].xcom_pull(task_ids='generate_strategy')
        backtest_results = {}
        
        for symbol, symbol_strategies in strategies.items():
            backtest_results[symbol] = {}
            
            for trading_mode, strategy in symbol_strategies.items():
                logger.info(f"Running backtest for {symbol} ({trading_mode})")
                
                # Download data for backtest
                backtest_runner.download_data(symbol, trading_mode, days=60)
                
                # Run backtest
                results = backtest_runner.run_backtest(
                    strategy_data=strategy,
                    symbol=symbol,
                    trading_mode=trading_mode
                )
                
                backtest_results[symbol][trading_mode] = results
                
                if results:
                    logger.info(f"Backtest completed for {symbol} ({trading_mode}): "
                              f"Return: {results.get('total_return', 0):.2f}%")
                else:
                    logger.warning(f"Backtest failed for {symbol} ({trading_mode})")
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in run_backtests: {e}")
        raise

def train_lstm_models(**context):
    """Train LSTM models for price prediction with optimizations"""
    import numpy as np  # Import here to avoid DagBag timeout
    
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        model_manager = modules['model_manager']
        
        prepared_data = context['task_instance'].xcom_pull(task_ids='prepare_data')
        model_results = {}
        
        # Smart training strategy - train one model at a time for quality
        max_concurrent_training = 1  # Focus on quality over speed
        trained_count = 0
        
        for symbol, symbol_data in prepared_data.items():
            logger.info(f"Checking LSTM model for {symbol}")
            
            ml_features = symbol_data.get('ml_features', {})
            
            if ml_features and len(ml_features.get('features', [])) > config.SEQ_LENGTH:
                features = np.array(ml_features['features'])
                target = np.array(ml_features['target_return'])
                
                # Smart data limiting for quality vs speed balance
                min_samples = config.MIN_TRAINING_SAMPLES
                max_samples = config.MAX_TRAINING_SAMPLES
                
                if len(features) < min_samples:
                    logger.warning(f"Insufficient data for {symbol}: {len(features)} < {min_samples}")
                    model_results[symbol] = {'status': 'insufficient_data'}
                    continue
                
                if len(features) > max_samples:
                    features = features[-max_samples:]
                    target = target[-max_samples:]
                    logger.info(f"Limited training data for {symbol} to {max_samples} samples for efficiency")
                
                # Check if we should retrain
                if model_manager.should_retrain(symbol, len(features)) and trained_count < max_concurrent_training:
                    logger.info(f"Retraining model for {symbol}")
                    
                    # Train model
                    training_results = model_manager.train_model_for_symbol(
                        symbol=symbol,
                        features=features,
                        target=target
                    )
                    
                    model_results[symbol] = training_results
                    trained_count += 1
                    logger.info(f"Model trained for {symbol}: MAPE {training_results.get('mape', 0):.2f}%")
                else:
                    logger.info(f"Using existing model for {symbol} (trained: {trained_count}/{max_concurrent_training})")
                    model_results[symbol] = {'status': 'existing_model_used'}
            else:
                logger.warning(f"Insufficient data for {symbol} LSTM training")
                model_results[symbol] = {'status': 'insufficient_data'}
        
        return model_results
        
    except Exception as e:
        logger.error(f"Error in train_lstm_models: {e}")
        raise

def execute_lstm_paper_trades(**context):
    """Execute LSTM-based paper trades using continuous trading system"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        
        # Import paper trader
        import sys
        sys.path.append('/app/src')
        from paper_trader import paper_trader
        
        trade_results = {}
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        
        logger.info("Starting LSTM-based paper trading execution...")
        
        for symbol in symbols:
            trade_results[symbol] = {}
            
            for trading_mode in trading_modes:
                logger.info(f"Executing LSTM paper trade for {symbol} ({trading_mode})")
                
                # Execute LSTM-based trade
                trade_result = paper_trader.execute_lstm_based_trade(symbol, trading_mode)
                
                if trade_result:
                    trade_results[symbol][trading_mode] = {
                        'action': trade_result.get('side', 'hold'),
                        'amount': trade_result.get('amount', 0),
                        'price': trade_result.get('price', 0),
                        'confidence': trade_result.get('confidence', 0),
                        'expected_return': trade_result.get('expected_return', 0),
                        'reason': trade_result.get('reason', ''),
                        'timestamp': trade_result.get('timestamp', datetime.now()).isoformat() if hasattr(trade_result.get('timestamp', datetime.now()), 'isoformat') else str(trade_result.get('timestamp', datetime.now())),
                        'is_paper': True
                    }
                    
                    logger.info(f"LSTM paper trade executed for {symbol} ({trading_mode}): {trade_result.get('side', 'hold')}")
                else:
                    trade_results[symbol][trading_mode] = {
                        'action': 'hold',
                        'reason': 'No LSTM signal',
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Get portfolio summary
        portfolio_summary = paper_trader.get_portfolio_summary()
        logger.info(f"Portfolio Summary: {portfolio_summary}")
        
        return {
            'trades': trade_results,
            'portfolio': portfolio_summary
        }
        
    except Exception as e:
        logger.error(f"Error in execute_lstm_paper_trades: {e}")
        raise

def calculate_and_log_metrics(**context):
    """Calculate and log performance metrics"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        metrics_calculator = modules['metrics_calculator']
        
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        
        all_metrics = {}
        
        for symbol in symbols:
            all_metrics[symbol] = {}
            
            for trading_mode in trading_modes:
                logger.info(f"Calculating metrics for {symbol} ({trading_mode})")
                
                # Calculate metrics
                metrics = metrics_calculator.calculate_trading_metrics(
                    symbol=symbol,
                    trading_mode=trading_mode,
                    period_days=7
                )
                
                # Save metrics to database
                if metrics:
                    metrics_calculator.save_metrics(
                        symbol=symbol,
                        trading_mode=trading_mode,
                        metrics=metrics,
                        period='daily'
                    )
                    
                    all_metrics[symbol][trading_mode] = metrics
                    
                    logger.info(f"Metrics for {symbol} ({trading_mode}): "
                              f"ROI: {metrics.get('total_return', 0):.2f}%, "
                              f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
        return all_metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_and_log_metrics: {e}")
        raise

def provide_llm_feedback(**context):
    """Provide feedback to LLM for strategy improvement"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        llm_engine = modules['llm_engine']
        
        backtest_results = context['task_instance'].xcom_pull(task_ids='backtest')
        strategies = context['task_instance'].xcom_pull(task_ids='generate_strategy')
        
        if not backtest_results or not strategies:
            logger.warning("No backtest results or strategies found for feedback")
            return {}
        
        feedback_results = {}
        
        for symbol, symbol_strategies in strategies.items():
            feedback_results[symbol] = {}
            
            for trading_mode, strategy in symbol_strategies.items():
                if symbol in backtest_results and trading_mode in backtest_results[symbol]:
                    performance_data = backtest_results[symbol][trading_mode]
                    
                    if performance_data and strategy.get('strategy_id'):
                        logger.info(f"Providing feedback for {symbol} ({trading_mode})")
                        
                        feedback = llm_engine.provide_feedback(
                            strategy_id=strategy['strategy_id'],
                            performance_data=performance_data
                        )
                        
                        feedback_results[symbol][trading_mode] = feedback
                        
                        logger.info(f"Feedback provided for {symbol} ({trading_mode})")
        
        return feedback_results
        
    except Exception as e:
        logger.error(f"Error in provide_llm_feedback: {e}")
        raise

# Define task dependencies
start_task = DummyOperator(
    task_id='start',
    dag=dag
)

fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_market_data,
    dag=dag
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_features,
    dag=dag
)

generate_strategy_task = PythonOperator(
    task_id='generate_strategy',
    python_callable=generate_strategies,
    dag=dag
)

backtest_task = PythonOperator(
    task_id='backtest',
    python_callable=run_backtests,
    dag=dag
)

train_lstm_task = PythonOperator(
    task_id='train_lstm',
    python_callable=train_lstm_models,
    execution_timeout=timedelta(hours=2),  # Adequate time for quality training
    dag=dag
)

# Paper trading moved to separate DAG (paper_trading_pipeline.py)

metrics_task = PythonOperator(
    task_id='log_metrics',
    python_callable=calculate_and_log_metrics,
    dag=dag
)

feedback_task = PythonOperator(
    task_id='feedback_to_llm',
    python_callable=provide_llm_feedback,
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    dag=dag
)

# Set task dependencies
start_task >> fetch_data_task >> prepare_data_task >> generate_strategy_task
generate_strategy_task >> backtest_task
prepare_data_task >> train_lstm_task
[backtest_task, train_lstm_task] >> metrics_task >> feedback_task >> end_task