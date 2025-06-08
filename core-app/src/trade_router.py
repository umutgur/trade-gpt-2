import subprocess
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from config import config

class TradeRouter:
    """Route paper trading commands to Freqtrade instances"""
    
    def __init__(self):
        self.freqtrade_endpoints = {
            'spot': 'http://freqtrade-spot:8080',
            'margin': 'http://freqtrade-margin:8080', 
            'futures': 'http://freqtrade-futures:8080'
        }
        
    def get_status(self, trading_mode: str) -> Dict:
        """Get status from Freqtrade instance"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            response = requests.get(f"{endpoint}/api/v1/status", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error getting status for {trading_mode}: {e}")
            return {'error': str(e)}
    
    def force_entry(self, trading_mode: str, symbol: str, side: str = 'long') -> Dict:
        """Force entry trade"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            data = {
                'pair': symbol,
                'side': side
            }
            
            response = requests.post(
                f"{endpoint}/api/v1/forceentry",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Force entry executed: {trading_mode} {symbol} {side}")
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error forcing entry: {e}")
            return {'error': str(e)}
    
    def force_exit(self, trading_mode: str, trade_id: int) -> Dict:
        """Force exit trade"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            data = {'tradeid': trade_id}
            
            response = requests.post(
                f"{endpoint}/api/v1/forceexit",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Force exit executed: {trading_mode} trade {trade_id}")
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error forcing exit: {e}")
            return {'error': str(e)}
    
    def get_open_trades(self, trading_mode: str) -> List[Dict]:
        """Get open trades"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return []
            
            response = requests.get(f"{endpoint}/api/v1/status", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('open_trades', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []
    
    def get_trade_history(self, trading_mode: str, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return []
            
            response = requests.get(
                f"{endpoint}/api/v1/trades",
                params={'limit': limit},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('trades', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def update_strategy(self, trading_mode: str, strategy_name: str) -> Dict:
        """Update strategy for trading mode"""
        try:
            # Stop the bot
            stop_result = self.stop_trading(trading_mode)
            
            if 'error' in stop_result:
                return stop_result
            
            # Update strategy in config (this would need to modify the config file)
            # For now, we'll restart with the new strategy
            
            # Start the bot with new strategy
            start_result = self.start_trading(trading_mode)
            
            return start_result
            
        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
            return {'error': str(e)}
    
    def start_trading(self, trading_mode: str) -> Dict:
        """Start trading for mode"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            response = requests.post(f"{endpoint}/api/v1/start", timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Started trading for {trading_mode}")
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            return {'error': str(e)}
    
    def stop_trading(self, trading_mode: str) -> Dict:
        """Stop trading for mode"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            response = requests.post(f"{endpoint}/api/v1/stop", timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Stopped trading for {trading_mode}")
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            return {'error': str(e)}
    
    def get_profit_summary(self, trading_mode: str) -> Dict:
        """Get profit summary"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {}
            
            response = requests.get(f"{endpoint}/api/v1/profit", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting profit summary: {e}")
            return {}
    
    def get_performance(self, trading_mode: str) -> Dict:
        """Get performance metrics"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {}
            
            response = requests.get(f"{endpoint}/api/v1/performance", timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {}
    
    def reload_config(self, trading_mode: str) -> Dict:
        """Reload configuration"""
        try:
            endpoint = self.freqtrade_endpoints.get(trading_mode)
            if not endpoint:
                return {'error': f'Invalid trading mode: {trading_mode}'}
            
            response = requests.post(f"{endpoint}/api/v1/reload_config", timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Reloaded config for {trading_mode}")
                return response.json()
            else:
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return {'error': str(e)}

# Global instance
trade_router = TradeRouter()