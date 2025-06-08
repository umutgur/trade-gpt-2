import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from db import db_manager

class MetricsCalculator:
    def __init__(self):
        pass
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            
            sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            return float(sortino)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        try:
            if len(equity_curve) == 0:
                return 0.0, 0
            
            # Calculate running maximum
            running_max = equity_curve.cummax()
            
            # Calculate drawdown
            drawdown = (equity_curve - running_max) / running_max
            
            # Maximum drawdown
            max_dd = drawdown.min()
            
            # Drawdown duration
            is_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            
            for dd in is_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
            
            return float(abs(max_dd)), max_dd_duration
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, 0
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            if max_drawdown == 0:
                return 0.0
            
            annual_return = (1 + returns.mean()) ** 252 - 1
            calmar = annual_return / max_drawdown
            return float(calmar)
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        try:
            if not trades:
                return 0.0
            
            winning_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
            return winning_trades / len(trades)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        try:
            if not trades:
                return 0.0
            
            gross_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0)
            gross_loss = abs(sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """Calculate expectancy per trade"""
        try:
            if not trades:
                return 0.0
            
            win_rate = self.calculate_win_rate(trades)
            
            winning_trades = [trade for trade in trades if trade.get('profit_loss', 0) > 0]
            losing_trades = [trade for trade in trades if trade.get('profit_loss', 0) < 0]
            
            avg_win = np.mean([trade.get('profit_loss', 0) for trade in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([trade.get('profit_loss', 0) for trade in losing_trades]) if losing_trades else 0
            
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            return float(expectancy)
            
        except Exception as e:
            logger.error(f"Error calculating expectancy: {e}")
            return 0.0
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) == 0:
                return 0.0
            
            var = np.percentile(returns, confidence * 100)
            return float(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        try:
            if len(actual) == 0 or len(predicted) == 0:
                return 100.0
            
            # Avoid division by zero
            mask = actual != 0
            if not mask.any():
                return 100.0
            
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            return float(mape)
            
        except Exception as e:
            logger.error(f"Error calculating MAPE: {e}")
            return 100.0
    
    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        try:
            if len(actual) == 0 or len(predicted) == 0:
                return 0.0
            
            mae = np.mean(np.abs(actual - predicted))
            return float(mae)
            
        except Exception as e:
            logger.error(f"Error calculating MAE: {e}")
            return 0.0
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        try:
            if len(actual) == 0 or len(predicted) == 0:
                return 0.0
            
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            return float(rmse)
            
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            return 0.0
    
    def calculate_trading_metrics(self, symbol: str, trading_mode: str, 
                                period_days: int = 7) -> Dict:
        """Calculate comprehensive trading metrics"""
        try:
            # Get trades from database
            session = db_manager.get_session()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            trades = session.query(db_manager.Trade)\
                .filter(db_manager.Trade.symbol == symbol)\
                .filter(db_manager.Trade.trading_mode == trading_mode)\
                .filter(db_manager.Trade.timestamp >= start_date)\
                .all()
            
            if not trades:
                logger.warning(f"No trades found for {symbol} ({trading_mode})")
                return {}
            
            # Convert to list of dicts
            trade_data = []
            for trade in trades:
                trade_data.append({
                    'profit_loss': trade.profit_loss or 0,
                    'price': trade.price,
                    'amount': trade.amount,
                    'timestamp': trade.timestamp
                })
            
            # Calculate portfolio value over time
            portfolio_values = []
            cumulative_pnl = 0
            
            for trade in trade_data:
                cumulative_pnl += trade['profit_loss']
                portfolio_values.append(10000 + cumulative_pnl)  # Assuming $10k starting capital
            
            portfolio_series = pd.Series(portfolio_values)
            returns = self.calculate_returns(portfolio_series)
            
            # Calculate all metrics
            metrics = {
                'total_trades': len(trade_data),
                'win_rate': self.calculate_win_rate(trade_data),
                'profit_factor': self.calculate_profit_factor(trade_data),
                'expectancy': self.calculate_expectancy(trade_data),
                'total_pnl': sum(trade['profit_loss'] for trade in trade_data),
                'average_trade': np.mean([trade['profit_loss'] for trade in trade_data]),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'var_5': self.calculate_var(returns, 0.05),
                'period_days': period_days,
                'timestamp': datetime.now()
            }
            
            # Calculate drawdown
            if len(portfolio_series) > 1:
                max_dd, dd_duration = self.calculate_max_drawdown(portfolio_series)
                metrics['max_drawdown'] = max_dd
                metrics['max_drawdown_duration'] = dd_duration
                metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, max_dd)
            
            # Calculate total return
            if len(portfolio_series) > 0:
                metrics['total_return'] = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1) * 100
            
            session.close()
            
            logger.info(f"Calculated metrics for {symbol} ({trading_mode}): {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {}
    
    def save_metrics(self, symbol: str, trading_mode: str, metrics: Dict, 
                    period: str = "daily") -> bool:
        """Save metrics to database"""
        try:
            session = db_manager.get_session()
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and metric_name != 'timestamp':
                    metric_record = db_manager.Metrics(
                        symbol=symbol,
                        trading_mode=trading_mode,
                        metric_type=metric_name,
                        metric_value=float(metric_value),
                        period=period
                    )
                    session.add(metric_record)
            
            session.commit()
            session.close()
            
            logger.info(f"Saved metrics for {symbol} ({trading_mode})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    def get_performance_summary(self, symbol: str, trading_mode: str, 
                               days: int = 30) -> Dict:
        """Get performance summary for strategy generation"""
        try:
            metrics = self.calculate_trading_metrics(symbol, trading_mode, days)
            
            # Provide defaults for missing metrics
            summary = {
                'total_return': metrics.get('total_return', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'profit_factor': metrics.get('profit_factor', 1.0),
                'total_trades': metrics.get('total_trades', 0),
                'expectancy': metrics.get('expectancy', 0.0)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 1.0,
                'total_trades': 0,
                'expectancy': 0.0
            }

# Global instance
metrics_calculator = MetricsCalculator()