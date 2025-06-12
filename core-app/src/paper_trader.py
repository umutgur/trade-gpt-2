import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time

from config import config
from db import db_manager
from data_fetcher import data_fetcher
from lstm_model import model_manager
from ta_features import ta_analyzer

class PaperTrader:
    def __init__(self):
        self.positions = {}  # {symbol: {mode: position_info}}
        from .config import Config
        config = Config()
        self.portfolio_value = config.INITIAL_PORTFOLIO_VALUE  # Starting capital per mode
        self.min_trade_amount = 50  # Minimum trade size
        self.max_position_size = 0.2  # 20% of portfolio max per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
    def execute_lstm_based_trade(self, symbol: str, trading_mode: str) -> Dict:
        """Execute paper trade based on LSTM prediction"""
        try:
            logger.info(f"Executing LSTM-based paper trade for {symbol} ({trading_mode})")
            
            # Get latest market data
            market_data = data_fetcher.fetch_ohlcv(symbol, limit=100)
            if market_data.empty:
                logger.warning(f"No market data for {symbol}")
                return {}
            
            # Add technical features
            enhanced_data = ta_analyzer.add_all_features(market_data)
            ml_features = ta_analyzer.get_features_for_ml(enhanced_data)
            
            if not ml_features or len(ml_features.get('features', [])) < 60:
                logger.warning(f"Insufficient features for {symbol} prediction")
                return {}
            
            # Get LSTM prediction
            features = np.array(ml_features['features'])
            prediction_result = model_manager.get_prediction(symbol, features[-60:].reshape(1, 60, -1))
            
            if not prediction_result:
                logger.warning(f"No LSTM prediction available for {symbol}")
                return {}
            
            current_price = market_data['close'].iloc[-1]
            predicted_price = prediction_result.get('predicted_price', current_price)
            confidence = prediction_result.get('confidence', 0.5)
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            logger.info(f"LSTM Prediction - Current: ${current_price:.2f}, Predicted: ${predicted_price:.2f}, Return: {expected_return*100:.2f}%, Confidence: {confidence:.2f}")
            
            # Trading decision based on LSTM prediction
            trade_decision = self._make_trading_decision(
                symbol, trading_mode, current_price, expected_return, confidence
            )
            
            if trade_decision['action'] != 'hold':
                # Execute the trade
                trade_result = self._execute_trade(
                    symbol, trading_mode, trade_decision, current_price
                )
                
                # Save to database
                self._save_trade_to_db(symbol, trading_mode, trade_result)
                
                return trade_result
            
            return {'action': 'hold', 'reason': 'No strong signal from LSTM'}
            
        except Exception as e:
            logger.error(f"Error in LSTM-based trading for {symbol}: {e}")
            return {}
    
    def _make_trading_decision(self, symbol: str, trading_mode: str, 
                             current_price: float, expected_return: float, 
                             confidence: float) -> Dict:
        """Make trading decision based on LSTM prediction"""
        
        # Get current position
        position_key = f"{symbol}_{trading_mode}"
        current_position = self.positions.get(position_key, {})
        
        # Trading thresholds
        min_confidence = 0.6
        min_expected_return = 0.02  # 2% minimum expected return
        
        # Decision logic
        if confidence < min_confidence:
            return {'action': 'hold', 'reason': f'Low confidence: {confidence:.2f}'}
        
        # Buy signal: strong positive prediction
        if expected_return > min_expected_return and not current_position.get('side') == 'buy':
            position_size = min(
                self.max_position_size * self.portfolio_value,
                self.portfolio_value * confidence * 0.5  # Scale by confidence
            )
            
            return {
                'action': 'buy',
                'amount': position_size / current_price,
                'confidence': confidence,
                'expected_return': expected_return,
                'reason': f'LSTM bullish prediction: {expected_return*100:.1f}%'
            }
        
        # Sell signal: strong negative prediction or stop loss
        elif expected_return < -min_expected_return and current_position.get('side') == 'buy':
            return {
                'action': 'sell',
                'amount': current_position.get('amount', 0),
                'confidence': confidence,
                'expected_return': expected_return,
                'reason': f'LSTM bearish prediction: {expected_return*100:.1f}%'
            }
        
        # Stop loss check for existing positions
        elif current_position.get('side') == 'buy':
            entry_price = current_position.get('entry_price', current_price)
            current_return = (current_price - entry_price) / entry_price
            
            if current_return <= -self.stop_loss_pct:
                return {
                    'action': 'sell',
                    'amount': current_position.get('amount', 0),
                    'confidence': confidence,
                    'expected_return': current_return,
                    'reason': f'Stop loss triggered: {current_return*100:.1f}%'
                }
            elif current_return >= self.take_profit_pct:
                return {
                    'action': 'sell',
                    'amount': current_position.get('amount', 0),
                    'confidence': confidence,
                    'expected_return': current_return,
                    'reason': f'Take profit triggered: {current_return*100:.1f}%'
                }
        
        return {'action': 'hold', 'reason': 'No clear signal'}
    
    def _execute_trade(self, symbol: str, trading_mode: str, decision: Dict, 
                      current_price: float) -> Dict:
        """Execute the paper trade"""
        
        position_key = f"{symbol}_{trading_mode}"
        timestamp = datetime.now()
        
        trade_result = {
            'symbol': symbol,
            'trading_mode': trading_mode,
            'side': decision['action'],
            'amount': decision['amount'],
            'price': current_price,
            'timestamp': timestamp,
            'confidence': decision['confidence'],
            'expected_return': decision['expected_return'],
            'reason': decision['reason'],
            'is_paper': True
        }
        
        if decision['action'] == 'buy':
            # Open new position
            self.positions[position_key] = {
                'side': 'buy',
                'amount': decision['amount'],
                'entry_price': current_price,
                'entry_time': timestamp,
                'value': decision['amount'] * current_price
            }
            
            trade_result['profit_loss'] = 0
            logger.info(f"Paper BUY: {decision['amount']:.6f} {symbol} at ${current_price:.2f}")
            
        elif decision['action'] == 'sell':
            # Close position
            if position_key in self.positions:
                position = self.positions[position_key]
                entry_price = position['entry_price']
                profit_loss = (current_price - entry_price) * decision['amount']
                
                trade_result['profit_loss'] = profit_loss
                trade_result['entry_price'] = entry_price
                
                # Remove position
                del self.positions[position_key]
                
                logger.info(f"Paper SELL: {decision['amount']:.6f} {symbol} at ${current_price:.2f}, P&L: ${profit_loss:.2f}")
            else:
                trade_result['profit_loss'] = 0
        
        return trade_result
    
    def _save_trade_to_db(self, symbol: str, trading_mode: str, trade_result: Dict):
        """Save paper trade to database"""
        try:
            session = db_manager.get_session()
            
            trade = db_manager.Trade(
                strategy_id=1,  # Paper trading strategy
                symbol=symbol,
                trading_mode=trading_mode,
                side=trade_result['side'],
                amount=trade_result['amount'],
                price=trade_result['price'],
                fee=0.001 * trade_result['amount'] * trade_result['price'],  # 0.1% fee simulation
                timestamp=trade_result['timestamp'],
                is_paper=True,
                profit_loss=trade_result.get('profit_loss', 0)
            )
            
            session.add(trade)
            session.commit()
            logger.info(f"Paper trade saved to database: {symbol} {trade_result['side']}")
            
        except Exception as e:
            logger.error(f"Error saving paper trade: {e}")
        finally:
            session.close()
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            total_value = self.portfolio_value
            active_positions = len(self.positions)
            
            # Calculate unrealized P&L
            unrealized_pnl = 0
            for position_key, position in self.positions.items():
                symbol = position_key.split('_')[0]
                current_price = data_fetcher.get_current_price(symbol)
                if current_price:
                    unrealized_pnl += (current_price - position['entry_price']) * position['amount']
            
            return {
                'total_value': total_value + unrealized_pnl,
                'cash': self.portfolio_value,
                'unrealized_pnl': unrealized_pnl,
                'active_positions': active_positions,
                'positions': self.positions
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio summary: {e}")
            return {}
    
    def run_continuous_trading(self, symbols: List[str] = None, 
                             trading_modes: List[str] = None):
        """Run continuous paper trading for all symbols"""
        if symbols is None:
            symbols = config.SYMBOLS
        if trading_modes is None:
            trading_modes = ['spot', 'margin', 'futures']
        
        logger.info("Starting continuous LSTM-based paper trading...")
        
        for symbol in symbols:
            for trading_mode in trading_modes:
                try:
                    result = self.execute_lstm_based_trade(symbol, trading_mode)
                    if result:
                        logger.info(f"Trade executed: {result}")
                except Exception as e:
                    logger.error(f"Error in continuous trading {symbol} {trading_mode}: {e}")
                
                # Small delay between trades
                time.sleep(1)

# Global instance
paper_trader = PaperTrader()