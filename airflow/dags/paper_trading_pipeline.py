from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os
from loguru import logger

# Add core-app src to path
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
        from lstm_model import model_manager
        from metrics import metrics_calculator
        from db import db_manager
        
        return {
            'config': config,
            'data_fetcher': data_fetcher,
            'ta_analyzer': ta_analyzer,
            'model_manager': model_manager,
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

# Default arguments for the DAG
default_args = {
    'owner': 'paper-trading-system',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': days_ago(1),
}

# Create Paper Trading DAG (every 15 minutes)
dag = DAG(
    'paper_trading_pipeline',
    default_args=default_args,
    description='LSTM-based Paper Trading Pipeline',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    catchup=False,
    max_active_runs=1,
    tags=['paper-trading', 'lstm', 'crypto', 'live']
)

def fetch_current_market_data(**context):
    """Fetch current market data for paper trading"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        data_fetcher = modules['data_fetcher']
        
        symbols = config.SYMBOLS
        results = {}
        
        for symbol in symbols:
            logger.info(f"Fetching current data for {symbol}")
            
            # Fetch latest OHLCV data
            df = data_fetcher.fetch_ohlcv(symbol, limit=100)
            
            if not df.empty:
                # Convert timestamp to string for JSON serialization
                timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
                timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
                
                results[symbol] = {
                    'success': True,
                    'records': len(df),
                    'latest_price': float(df['close'].iloc[-1]) if len(df) > 0 else 0.0,
                    'timestamp': timestamp_str
                }
                logger.info(f"Fetched {len(df)} records for {symbol}")
            else:
                results[symbol] = {
                    'success': False, 
                    'records': 0, 
                    'latest_price': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                logger.warning(f"No data fetched for {symbol}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_current_market_data: {e}")
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
        
        logger.info("🚀 Starting LSTM-based paper trading execution...")
        
        total_trades = 0
        successful_trades = 0
        
        for symbol in symbols:
            trade_results[symbol] = {}
            
            for trading_mode in trading_modes:
                logger.info(f"📊 Executing LSTM paper trade for {symbol} ({trading_mode})")
                
                # Execute LSTM-based trade
                trade_result = paper_trader.execute_lstm_based_trade(symbol, trading_mode)
                total_trades += 1
                
                if trade_result and trade_result.get('side') != 'hold':
                    successful_trades += 1
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
                    
                    logger.info(f"✅ LSTM paper trade executed for {symbol} ({trading_mode}): {trade_result.get('side', 'hold')}")
                else:
                    trade_results[symbol][trading_mode] = {
                        'action': 'hold',
                        'reason': trade_result.get('reason', 'No LSTM signal') if trade_result else 'No LSTM signal',
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Get portfolio summary
        portfolio_summary = paper_trader.get_portfolio_summary()
        logger.info(f"📈 Portfolio Summary: {portfolio_summary}")
        
        # Log summary
        logger.info(f"📊 Paper Trading Summary: {successful_trades}/{total_trades} trades executed")
        
        return {
            'trades': trade_results,
            'portfolio': portfolio_summary,
            'summary': {
                'total_attempts': total_trades,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / total_trades if total_trades > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in execute_lstm_paper_trades: {e}")
        raise

def calculate_paper_trading_metrics(**context):
    """Calculate and log paper trading performance metrics"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        metrics_calculator = modules['metrics_calculator']
        
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        
        paper_metrics = {}
        
        for symbol in symbols:
            paper_metrics[symbol] = {}
            
            for trading_mode in trading_modes:
                logger.info(f"📊 Calculating paper trading metrics for {symbol} ({trading_mode})")
                
                # Calculate metrics specifically for paper trades
                metrics = metrics_calculator.calculate_trading_metrics(
                    symbol=symbol,
                    trading_mode=trading_mode,
                    period_days=7
                )
                
                # Filter for paper trades only
                if metrics:
                    # Add paper trading specific metrics
                    metrics['trading_type'] = 'paper'
                    metrics_calculator.save_metrics(
                        symbol=symbol,
                        trading_mode=trading_mode,
                        metrics=metrics,
                        period='paper_daily'
                    )
                    
                    paper_metrics[symbol][trading_mode] = metrics
                    
                    logger.info(f"📈 Paper metrics for {symbol} ({trading_mode}): "
                              f"ROI: {metrics.get('total_return', 0):.2f}%, "
                              f"Trades: {metrics.get('total_trades', 0)}")
        
        return paper_metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_paper_trading_metrics: {e}")
        raise

def monitor_paper_portfolio(**context):
    """Monitor and log paper trading portfolio status"""
    try:
        # Import paper trader
        import sys
        sys.path.append('/app/src')
        from paper_trader import paper_trader
        
        # Get current portfolio status
        portfolio = paper_trader.get_portfolio_summary()
        
        if portfolio:
            logger.info(f"💰 Portfolio Status:")
            logger.info(f"   Total Value: ${portfolio.get('total_value', 0):.2f}")
            logger.info(f"   Cash: ${portfolio.get('cash', 0):.2f}")
            logger.info(f"   Unrealized P&L: ${portfolio.get('unrealized_pnl', 0):.2f}")
            logger.info(f"   Active Positions: {portfolio.get('active_positions', 0)}")
            
            # Log individual positions
            positions = portfolio.get('positions', {})
            if positions:
                logger.info(f"🎯 Active Positions:")
                for position_key, position in positions.items():
                    logger.info(f"   {position_key}: {position}")
            else:
                logger.info(f"📋 No active positions")
        else:
            logger.warning("⚠️ Could not retrieve portfolio status")
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error in monitor_paper_portfolio: {e}")
        raise

# Define task dependencies
start_task = DummyOperator(
    task_id='start_paper_trading',
    dag=dag
)

fetch_data_task = PythonOperator(
    task_id='fetch_current_data',
    python_callable=fetch_current_market_data,
    dag=dag
)

paper_trade_task = PythonOperator(
    task_id='execute_paper_trades',
    python_callable=execute_lstm_paper_trades,
    dag=dag
)

metrics_task = PythonOperator(
    task_id='calculate_paper_metrics',
    python_callable=calculate_paper_trading_metrics,
    dag=dag
)

portfolio_task = PythonOperator(
    task_id='monitor_portfolio',
    python_callable=monitor_paper_portfolio,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_paper_trading',
    dag=dag
)

# Set task dependencies
start_task >> fetch_data_task >> paper_trade_task >> [metrics_task, portfolio_task] >> end_task