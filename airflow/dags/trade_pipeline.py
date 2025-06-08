from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os
import pandas as pd
import numpy as np
from loguru import logger

# Add core-app src to path (assuming mounted volume)
sys.path.append('/opt/airflow/dags')

# Import our modules (these would be available through mounted volumes)
try:
    from config import config
    from data_fetcher import data_fetcher
    from ta_features import ta_analyzer
    from llm_strategy import llm_engine
    from lstm_model import model_manager
    from backtest_runner import backtest_runner
    from metrics import metrics_calculator
    from db import db_manager
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback imports or dummy implementations could go here

# Default arguments for the DAG
default_args = {
    'owner': 'trade-system',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Create DAG
dag = DAG(
    'crypto_trading_pipeline',
    default_args=default_args,
    description='Crypto Trading AI Pipeline',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    catchup=False,
    max_active_runs=1,
    tags=['trading', 'crypto', 'ai']
)

def fetch_market_data(**context):
    """Fetch latest market data for all symbols"""
    try:
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
    try:
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
                        'timestamp': item.timestamp,
                        'open': item.open_price,
                        'high': item.high_price,
                        'low': item.low_price,
                        'close': item.close_price,
                        'volume': item.volume
                    } for item in market_data])
                    
                    df = df.sort_values('timestamp')
                    
                    # Add technical features
                    df_with_features = ta_analyzer.add_all_features(df)
                    
                    # Prepare features for ML
                    ml_features = ta_analyzer.get_features_for_ml(df_with_features)
                    
                    prepared_data[symbol] = {
                        'dataframe': df_with_features,
                        'ml_features': ml_features,
                        'latest_price': df['close'].iloc[-1],
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
        prepared_data = context['task_instance'].xcom_pull(task_ids='prepare_data')
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        strategies = {}
        
        for symbol in symbols:
            if symbol in prepared_data:
                symbol_data = prepared_data[symbol]
                df = symbol_data['dataframe']
                
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
    """Train LSTM models for price prediction"""
    try:
        prepared_data = context['task_instance'].xcom_pull(task_ids='prepare_data')
        model_results = {}
        
        for symbol, symbol_data in prepared_data.items():
            logger.info(f"Training LSTM model for {symbol}")
            
            ml_features = symbol_data.get('ml_features', {})
            
            if ml_features and len(ml_features.get('features', [])) > config.SEQ_LENGTH:
                features = ml_features['features']
                target = ml_features['target_return']
                
                # Check if we should retrain
                if model_manager.should_retrain(symbol, len(features)):
                    logger.info(f"Retraining model for {symbol}")
                    
                    # Train model
                    training_results = model_manager.train_model_for_symbol(
                        symbol=symbol,
                        features=features,
                        target=target
                    )
                    
                    model_results[symbol] = training_results
                    logger.info(f"Model trained for {symbol}: MAPE {training_results.get('mape', 0):.2f}%")
                else:
                    logger.info(f"Using existing model for {symbol}")
                    model_results[symbol] = {'status': 'existing_model_used'}
            else:
                logger.warning(f"Insufficient data for {symbol} LSTM training")
                model_results[symbol] = {'status': 'insufficient_data'}
        
        return model_results
        
    except Exception as e:
        logger.error(f"Error in train_lstm_models: {e}")
        raise

def execute_paper_trades(**context):
    """Execute paper trades (placeholder for actual trading)"""
    try:
        strategies = context['task_instance'].xcom_pull(task_ids='generate_strategy')
        prepared_data = context['task_instance'].xcom_pull(task_ids='prepare_data')
        
        trade_results = {}
        
        for symbol, symbol_strategies in strategies.items():
            if symbol in prepared_data:
                latest_price = prepared_data[symbol]['latest_price']
                trade_results[symbol] = {}
                
                for trading_mode, strategy in symbol_strategies.items():
                    logger.info(f"Executing paper trade for {symbol} ({trading_mode})")
                    
                    # Simulate trade execution based on strategy
                    confidence = strategy.get('confidence', 0.5)
                    market_outlook = strategy.get('market_outlook', 'neutral')
                    
                    # Simple trading logic (placeholder)
                    if confidence > 0.7 and market_outlook == 'bullish':
                        trade_action = 'buy'
                    elif confidence > 0.7 and market_outlook == 'bearish':
                        trade_action = 'sell'
                    else:
                        trade_action = 'hold'
                    
                    trade_results[symbol][trading_mode] = {
                        'action': trade_action,
                        'price': latest_price,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"Paper trade result for {symbol} ({trading_mode}): {trade_action}")
        
        return trade_results
        
    except Exception as e:
        logger.error(f"Error in execute_paper_trades: {e}")
        raise

def calculate_and_log_metrics(**context):
    """Calculate and log performance metrics"""
    try:
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
        backtest_results = context['task_instance'].xcom_pull(task_ids='backtest')
        strategies = context['task_instance'].xcom_pull(task_ids='generate_strategy')
        
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
    dag=dag
)

paper_trade_task = PythonOperator(
    task_id='papertrade',
    python_callable=execute_paper_trades,
    dag=dag
)

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
[backtest_task, train_lstm_task] >> paper_trade_task >> metrics_task >> feedback_task >> end_task