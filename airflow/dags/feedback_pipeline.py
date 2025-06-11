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
        from llm_strategy import llm_engine
        from lstm_model import model_manager
        from metrics import metrics_calculator
        from db import db_manager
        
        return {
            'config': config,
            'llm_engine': llm_engine,
            'model_manager': model_manager,
            'metrics_calculator': metrics_calculator,
            'db_manager': db_manager
        }
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return {}

# Default arguments for the DAG
default_args = {
    'owner': 'feedback-system',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# Create Feedback DAG (every 6 hours)
dag = DAG(
    'feedback_pipeline',
    default_args=default_args,
    description='AI Model Feedback and Strategy Optimization Pipeline',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=['feedback', 'optimization', 'ai', 'lstm', 'llm']
)

def analyze_paper_trading_performance(**context):
    """Analyze paper trading performance for feedback"""
    try:
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        db_manager = modules['db_manager']
        metrics_calculator = modules['metrics_calculator']
        
        symbols = config.SYMBOLS
        trading_modes = ['spot', 'margin', 'futures']
        analysis_results = {}
        
        logger.info("🔍 Analyzing paper trading performance for feedback...")
        
        for symbol in symbols:
            analysis_results[symbol] = {}
            
            for trading_mode in trading_modes:
                logger.info(f"📊 Analyzing {symbol} ({trading_mode})")
                
                # Get paper trading performance from last 24 hours
                session = db_manager.get_session()
                
                # Get paper trades
                paper_trades = session.query(db_manager.Trade)\
                    .filter(db_manager.Trade.symbol == symbol)\
                    .filter(db_manager.Trade.trading_mode == trading_mode)\
                    .filter(db_manager.Trade.is_paper == True)\
                    .filter(db_manager.Trade.timestamp >= datetime.now() - timedelta(hours=24))\
                    .all()
                
                if paper_trades:
                    # Calculate performance metrics
                    total_trades = len(paper_trades)
                    profitable_trades = len([t for t in paper_trades if (t.profit_loss or 0) > 0])
                    total_pnl = sum([t.profit_loss or 0 for t in paper_trades])
                    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                    
                    # Get LSTM prediction accuracy
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for trade in paper_trades:
                        if trade.side in ['buy', 'sell'] and trade.profit_loss is not None:
                            total_predictions += 1
                            # If trade was profitable and we bought, or trade was unprofitable and we sold
                            if (trade.side == 'buy' and trade.profit_loss > 0) or \
                               (trade.side == 'sell' and trade.profit_loss > 0):
                                correct_predictions += 1
                    
                    prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                    
                    # Analyze trading patterns
                    avg_trade_size = sum([t.amount for t in paper_trades]) / total_trades if total_trades > 0 else 0
                    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
                    
                    analysis_results[symbol][trading_mode] = {
                        'total_trades': total_trades,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl_per_trade': avg_pnl_per_trade,
                        'prediction_accuracy': prediction_accuracy,
                        'avg_trade_size': avg_trade_size,
                        'performance_score': (win_rate * 0.4) + (prediction_accuracy * 0.6),  # Weighted score
                        'needs_improvement': win_rate < 0.5 or prediction_accuracy < 0.6
                    }
                    
                    logger.info(f"📈 {symbol} ({trading_mode}): "
                              f"Trades: {total_trades}, Win Rate: {win_rate:.2%}, "
                              f"P&L: ${total_pnl:.2f}, Accuracy: {prediction_accuracy:.2%}")
                else:
                    analysis_results[symbol][trading_mode] = {
                        'total_trades': 0,
                        'needs_improvement': True,
                        'reason': 'No paper trades executed'
                    }
                
                session.close()
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in analyze_paper_trading_performance: {e}")
        raise

def provide_lstm_feedback(**context):
    """Provide feedback to LSTM models based on paper trading results"""
    try:
        # Get analysis results from previous task
        analysis_results = context['task_instance'].xcom_pull(task_ids='analyze_performance')
        
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        db_manager = modules['db_manager']
        
        feedback_results = {}
        
        logger.info("🤖 Providing feedback to LSTM models...")
        
        for symbol, symbol_data in analysis_results.items():
            feedback_results[symbol] = {}
            
            for trading_mode, performance in symbol_data.items():
                if performance.get('needs_improvement', False):
                    logger.info(f"🔄 {symbol} ({trading_mode}) needs improvement")
                    
                    # Create feedback record for LSTM model
                    session = db_manager.get_session()
                    
                    # Get latest LSTM model
                    latest_model = session.query(db_manager.LSTMModel)\
                        .filter(db_manager.LSTMModel.symbol == symbol)\
                        .filter(db_manager.LSTMModel.is_active == True)\
                        .order_by(db_manager.LSTMModel.created_at.desc())\
                        .first()
                    
                    if latest_model:
                        # Update model feedback
                        feedback_data = {
                            'win_rate': performance.get('win_rate', 0),
                            'prediction_accuracy': performance.get('prediction_accuracy', 0),
                            'total_pnl': performance.get('total_pnl', 0),
                            'performance_score': performance.get('performance_score', 0),
                            'feedback_timestamp': datetime.now(),
                            'needs_retraining': performance.get('performance_score', 0) < 0.5
                        }
                        
                        # Mark model for retraining if performance is poor
                        if feedback_data['needs_retraining']:
                            latest_model.is_active = False
                            session.commit()
                            logger.info(f"🔄 Marked {symbol} LSTM model for retraining due to poor performance")
                        
                        feedback_results[symbol][trading_mode] = feedback_data
                    
                    session.close()
                else:
                    logger.info(f"✅ {symbol} ({trading_mode}) performing well")
                    feedback_results[symbol][trading_mode] = {'status': 'performing_well'}
        
        return feedback_results
        
    except Exception as e:
        logger.error(f"Error in provide_lstm_feedback: {e}")
        raise

def provide_llm_strategy_feedback(**context):
    """Provide feedback to LLM for strategy improvement"""
    try:
        # Get analysis results from previous task
        analysis_results = context['task_instance'].xcom_pull(task_ids='analyze_performance')
        
        # Lazy import to avoid DagBag timeout
        modules = get_core_modules()
        config = modules['config']
        llm_engine = modules['llm_engine']
        db_manager = modules['db_manager']
        
        feedback_results = {}
        
        logger.info("🧠 Providing feedback to LLM for strategy optimization...")
        
        for symbol, symbol_data in analysis_results.items():
            feedback_results[symbol] = {}
            
            for trading_mode, performance in symbol_data.items():
                if performance.get('total_trades', 0) > 0:
                    # Prepare performance summary for LLM
                    performance_summary = {
                        'symbol': symbol,
                        'trading_mode': trading_mode,
                        'win_rate': performance.get('win_rate', 0),
                        'total_pnl': performance.get('total_pnl', 0),
                        'total_trades': performance.get('total_trades', 0),
                        'avg_pnl_per_trade': performance.get('avg_pnl_per_trade', 0),
                        'prediction_accuracy': performance.get('prediction_accuracy', 0),
                        'period': '24h paper trading',
                        'timestamp': datetime.now()
                    }
                    
                    # Get latest strategy
                    session = db_manager.get_session()
                    latest_strategy = session.query(db_manager.Strategy)\
                        .filter(db_manager.Strategy.symbol == symbol)\
                        .filter(db_manager.Strategy.trading_mode == trading_mode)\
                        .filter(db_manager.Strategy.is_active == True)\
                        .order_by(db_manager.Strategy.timestamp.desc())\
                        .first()
                    
                    if latest_strategy:
                        # Provide feedback to LLM
                        try:
                            feedback = llm_engine.provide_feedback(
                                strategy_id=latest_strategy.id,
                                performance_data=performance_summary
                            )
                            
                            feedback_results[symbol][trading_mode] = {
                                'feedback_provided': True,
                                'strategy_id': latest_strategy.id,
                                'performance_score': performance.get('performance_score', 0),
                                'feedback_summary': feedback
                            }
                            
                            logger.info(f"🧠 LLM feedback provided for {symbol} ({trading_mode})")
                        except Exception as e:
                            logger.error(f"Error providing LLM feedback for {symbol} ({trading_mode}): {e}")
                            feedback_results[symbol][trading_mode] = {'error': str(e)}
                    
                    session.close()
        
        return feedback_results
        
    except Exception as e:
        logger.error(f"Error in provide_llm_strategy_feedback: {e}")
        raise

def optimize_trading_parameters(**context):
    """Optimize trading parameters based on feedback"""
    try:
        # Get analysis and feedback results
        analysis_results = context['task_instance'].xcom_pull(task_ids='analyze_performance')
        lstm_feedback = context['task_instance'].xcom_pull(task_ids='lstm_feedback')
        
        # Import paper trader for parameter optimization
        import sys
        sys.path.append('/app/src')
        from paper_trader import paper_trader
        
        optimization_results = {}
        
        logger.info("⚙️ Optimizing trading parameters based on feedback...")
        
        # Analyze overall performance
        total_trades = 0
        total_profitable = 0
        total_pnl = 0
        
        for symbol, symbol_data in analysis_results.items():
            for trading_mode, performance in symbol_data.items():
                total_trades += performance.get('total_trades', 0)
                if performance.get('win_rate', 0) > 0.5:
                    total_profitable += performance.get('total_trades', 0)
                total_pnl += performance.get('total_pnl', 0)
        
        overall_win_rate = total_profitable / total_trades if total_trades > 0 else 0
        
        # Optimize parameters based on performance
        if overall_win_rate < 0.5:
            # Reduce position sizes and increase confidence threshold
            new_max_position_size = max(0.1, paper_trader.max_position_size * 0.8)
            paper_trader.max_position_size = new_max_position_size
            
            logger.info(f"📉 Reduced max position size to {new_max_position_size:.2%} due to poor performance")
            
            optimization_results['position_size_adjustment'] = {
                'action': 'reduced',
                'new_value': new_max_position_size,
                'reason': f'Overall win rate: {overall_win_rate:.2%}'
            }
        
        elif overall_win_rate > 0.7 and total_pnl > 0:
            # Increase position sizes for better performance
            new_max_position_size = min(0.3, paper_trader.max_position_size * 1.1)
            paper_trader.max_position_size = new_max_position_size
            
            logger.info(f"📈 Increased max position size to {new_max_position_size:.2%} due to good performance")
            
            optimization_results['position_size_adjustment'] = {
                'action': 'increased',
                'new_value': new_max_position_size,
                'reason': f'Overall win rate: {overall_win_rate:.2%}, P&L: ${total_pnl:.2f}'
            }
        
        # Adjust stop loss and take profit based on average trade performance
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        if avg_pnl_per_trade < -10:  # Losing more than $10 per trade on average
            # Tighten stop loss
            new_stop_loss = max(0.015, paper_trader.stop_loss_pct * 0.9)
            paper_trader.stop_loss_pct = new_stop_loss
            
            optimization_results['stop_loss_adjustment'] = {
                'action': 'tightened',
                'new_value': new_stop_loss,
                'reason': f'Average P&L per trade: ${avg_pnl_per_trade:.2f}'
            }
        
        logger.info(f"⚙️ Parameter optimization completed: {optimization_results}")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error in optimize_trading_parameters: {e}")
        raise

def log_feedback_summary(**context):
    """Log comprehensive feedback summary"""
    try:
        # Get all results from previous tasks
        analysis_results = context['task_instance'].xcom_pull(task_ids='analyze_performance')
        lstm_feedback = context['task_instance'].xcom_pull(task_ids='lstm_feedback')
        llm_feedback = context['task_instance'].xcom_pull(task_ids='llm_feedback')
        optimization_results = context['task_instance'].xcom_pull(task_ids='optimize_parameters')
        
        logger.info("📋 FEEDBACK SUMMARY REPORT")
        logger.info("=" * 50)
        
        # Performance summary
        total_symbols = len(analysis_results)
        symbols_needing_improvement = 0
        
        for symbol, symbol_data in analysis_results.items():
            for trading_mode, performance in symbol_data.items():
                if performance.get('needs_improvement', False):
                    symbols_needing_improvement += 1
        
        logger.info(f"📊 Performance Analysis:")
        logger.info(f"   Total Symbol/Mode Combinations: {total_symbols * 3}")
        logger.info(f"   Combinations Needing Improvement: {symbols_needing_improvement}")
        logger.info(f"   Performance Rate: {((total_symbols * 3 - symbols_needing_improvement) / (total_symbols * 3) * 100):.1f}%")
        
        # LSTM feedback summary
        if lstm_feedback:
            models_marked_for_retraining = 0
            for symbol, symbol_data in lstm_feedback.items():
                for trading_mode, feedback in symbol_data.items():
                    if feedback.get('needs_retraining', False):
                        models_marked_for_retraining += 1
            
            logger.info(f"🤖 LSTM Models:")
            logger.info(f"   Models Marked for Retraining: {models_marked_for_retraining}")
        
        # LLM feedback summary
        if llm_feedback:
            strategies_updated = 0
            for symbol, symbol_data in llm_feedback.items():
                for trading_mode, feedback in symbol_data.items():
                    if feedback.get('feedback_provided', False):
                        strategies_updated += 1
            
            logger.info(f"🧠 LLM Strategies:")
            logger.info(f"   Strategies Updated: {strategies_updated}")
        
        # Optimization summary
        if optimization_results:
            logger.info(f"⚙️ Parameter Optimizations:")
            for optimization, details in optimization_results.items():
                logger.info(f"   {optimization}: {details.get('action', 'unknown')} - {details.get('reason', '')}")
        
        logger.info("=" * 50)
        logger.info("✅ Feedback cycle completed successfully")
        
        return {
            'analysis_results': analysis_results,
            'lstm_feedback': lstm_feedback,
            'llm_feedback': llm_feedback,
            'optimization_results': optimization_results,
            'summary': {
                'total_combinations': total_symbols * 3,
                'needing_improvement': symbols_needing_improvement,
                'performance_rate': ((total_symbols * 3 - symbols_needing_improvement) / (total_symbols * 3) * 100) if total_symbols > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in log_feedback_summary: {e}")
        raise

# Define task dependencies
start_task = DummyOperator(
    task_id='start_feedback',
    dag=dag
)

analyze_task = PythonOperator(
    task_id='analyze_performance',
    python_callable=analyze_paper_trading_performance,
    dag=dag
)

lstm_feedback_task = PythonOperator(
    task_id='lstm_feedback',
    python_callable=provide_lstm_feedback,
    dag=dag
)

llm_feedback_task = PythonOperator(
    task_id='llm_feedback',
    python_callable=provide_llm_strategy_feedback,
    dag=dag
)

optimize_task = PythonOperator(
    task_id='optimize_parameters',
    python_callable=optimize_trading_parameters,
    dag=dag
)

summary_task = PythonOperator(
    task_id='log_summary',
    python_callable=log_feedback_summary,
    dag=dag
)

end_task = DummyOperator(
    task_id='end_feedback',
    dag=dag
)

# Set task dependencies
start_task >> analyze_task >> [lstm_feedback_task, llm_feedback_task] >> optimize_task >> summary_task >> end_task