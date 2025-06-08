import subprocess
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from jinja2 import Template
from loguru import logger
from config import config
from db import db_manager

class BacktestRunner:
    def __init__(self):
        self.freqtrade_path = "/freqtrade"
        self.config_base_path = "/freqtrade/user_data/configs"
        self.strategy_path = "/freqtrade/user_data/strategies"
    
    def generate_strategy_file(self, strategy_data: Dict, trading_mode: str) -> str:
        """Generate Freqtrade strategy file from LLM strategy data"""
        try:
            # Read the template
            template_path = os.path.join(self.strategy_path, "AI_Strategy.py")
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Prepare template variables
            template_vars = self._prepare_template_vars(strategy_data, trading_mode)
            
            # Render template
            template = Template(template_content)
            strategy_content = template.render(**template_vars)
            
            # Write strategy file
            strategy_filename = f"AI_Strategy_{trading_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            strategy_file_path = os.path.join(self.strategy_path, strategy_filename)
            
            with open(strategy_file_path, 'w') as f:
                f.write(strategy_content)
            
            logger.info(f"Generated strategy file: {strategy_file_path}")
            return strategy_filename.replace('.py', '')
            
        except Exception as e:
            logger.error(f"Error generating strategy file: {e}")
            return "AI_Strategy"
    
    def _prepare_template_vars(self, strategy_data: Dict, trading_mode: str) -> Dict:
        """Prepare variables for Jinja2 template"""
        try:
            # Extract indicators and their parameters
            indicators = {}
            for indicator in strategy_data.get('indicators', []):
                name = indicator['name'].lower()
                indicators[name] = {
                    'enabled': True,
                    'parameters': indicator.get('parameters', {})
                }
            
            # Convert conditions to dataframe operations
            buy_conditions = []
            for condition in strategy_data.get('buy_conditions', []):
                buy_conditions.append(self._convert_condition_to_dataframe(condition))
            
            sell_conditions = []
            for condition in strategy_data.get('sell_conditions', []):
                sell_conditions.append(self._convert_condition_to_dataframe(condition))
            
            # Risk management
            risk = strategy_data.get('risk_management', {})
            
            template_vars = {
                'strategy': {
                    'trading_mode': trading_mode,
                    'timeframe': config.TIMEFRAME,
                    'can_short': trading_mode == 'futures',
                    'stoploss': -abs(risk.get('stop_loss', 5.0)) / 100,
                    'minimal_roi': self._generate_roi_table(risk.get('take_profit', 10.0)),
                    'trailing_stop': trading_mode in ['margin', 'futures'],
                    'trailing_stop_positive': risk.get('take_profit', 10.0) / 200,
                    'trailing_stop_positive_offset': risk.get('take_profit', 10.0) / 100,
                    'trailing_only_offset_is_reached': True,
                    'indicators': indicators,
                    'buy_conditions': buy_conditions,
                    'sell_conditions': sell_conditions,
                    'risk_management': risk
                }
            }
            
            return template_vars
            
        except Exception as e:
            logger.error(f"Error preparing template variables: {e}")
            return {'strategy': {}}
    
    def _convert_condition_to_dataframe(self, condition: Dict) -> Dict:
        """Convert LLM condition to dataframe operation"""
        try:
            condition_str = condition['condition']
            threshold = condition['threshold']
            operator = condition['operator']
            
            # Map common conditions to dataframe operations
            condition_mapping = {
                'rsi': f"dataframe['rsi'] {operator} {threshold}",
                'close > ema_20': f"dataframe['close'] > dataframe['ema_20']",
                'close < ema_20': f"dataframe['close'] < dataframe['ema_20']",
                'macd > signal': f"dataframe['macd'] > dataframe['macdsignal']",
                'macd < signal': f"dataframe['macd'] < dataframe['macdsignal']",
                'bb_lower': f"dataframe['close'] <= dataframe['bb_lowerband']",
                'bb_upper': f"dataframe['close'] >= dataframe['bb_upperband']",
            }
            
            if condition_str in condition_mapping:
                return {'condition': condition_mapping[condition_str]}
            else:
                # Generic condition
                return {'condition': f"dataframe['{condition_str}'] {operator} {threshold}"}
                
        except Exception as e:
            logger.error(f"Error converting condition: {e}")
            return {'condition': "True"}
    
    def _generate_roi_table(self, take_profit_pct: float) -> Dict:
        """Generate ROI table based on take profit percentage"""
        take_profit = take_profit_pct / 100
        return {
            "0": take_profit,
            "40": take_profit * 0.6,
            "100": take_profit * 0.3,
            "300": 0
        }
    
    def run_backtest(self, strategy_data: Dict, symbol: str, trading_mode: str,
                    start_date: str = None, end_date: str = None) -> Dict:
        """Run backtest with given strategy"""
        try:
            # Generate strategy file
            strategy_name = self.generate_strategy_file(strategy_data, trading_mode)
            
            # Prepare config
            config_file = os.path.join(self.config_base_path, f"{trading_mode}.json")
            
            # Update config for specific symbol and strategy
            updated_config = self._update_config_for_backtest(
                config_file, strategy_name, symbol, start_date, end_date
            )
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(updated_config, f, indent=2)
                temp_config_path = f.name
            
            try:
                # Run backtest
                cmd = [
                    "freqtrade", "backtesting",
                    "--config", temp_config_path,
                    "--strategy", strategy_name,
                    "--timerange", self._get_timerange(start_date, end_date),
                    "--enable-protections",
                    "--cache", "none"
                ]
                
                logger.info(f"Running backtest command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=self.freqtrade_path,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout
                )
                
                if result.returncode == 0:
                    # Parse backtest results
                    backtest_results = self._parse_backtest_output(result.stdout)
                    
                    # Save to database
                    strategy_id = strategy_data.get('strategy_id')
                    if strategy_id and backtest_results:
                        db_manager.save_backtest_results(
                            strategy_id=strategy_id,
                            symbol=symbol,
                            trading_mode=trading_mode,
                            start_date=datetime.strptime(start_date, '%Y%m%d') if start_date else datetime.now() - timedelta(days=30),
                            end_date=datetime.strptime(end_date, '%Y%m%d') if end_date else datetime.now(),
                            results=backtest_results
                        )
                    
                    logger.info(f"Backtest completed successfully for {symbol} ({trading_mode})")
                    return backtest_results
                else:
                    logger.error(f"Backtest failed: {result.stderr}")
                    return {}
                    
            finally:
                # Clean up temporary config file
                os.unlink(temp_config_path)
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _update_config_for_backtest(self, config_file: str, strategy_name: str, 
                                  symbol: str, start_date: str = None, 
                                  end_date: str = None) -> Dict:
        """Update config for backtest"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update strategy
            config_data['strategy'] = strategy_name
            
            # Update pair whitelist to single symbol
            config_data['exchange']['pair_whitelist'] = [symbol]
            
            # Set dry run to false for backtest
            config_data['dry_run'] = False
            
            # Add backtest specific settings
            config_data['dataformat_ohlcv'] = 'json'
            
            return config_data
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return {}
    
    def _get_timerange(self, start_date: str = None, end_date: str = None) -> str:
        """Get timerange string for backtest"""
        if start_date and end_date:
            return f"{start_date}-{end_date}"
        elif config.BT_RANGE:
            return config.BT_RANGE.replace('-', '-')
        else:
            # Default to last 30 days
            end = datetime.now()
            start = end - timedelta(days=30)
            return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    
    def _parse_backtest_output(self, output: str) -> Dict:
        """Parse backtest output to extract metrics"""
        try:
            lines = output.split('\n')
            results = {}
            
            for line in lines:
                line = line.strip()
                
                # Parse key metrics
                if 'Total Return' in line:
                    results['total_return'] = self._extract_percentage(line)
                elif 'Sharpe' in line:
                    results['sharpe_ratio'] = self._extract_float(line)
                elif 'Max Drawdown' in line:
                    results['max_drawdown'] = self._extract_percentage(line)
                elif 'Win Rate' in line:
                    results['win_rate'] = self._extract_percentage(line)
                elif 'Profit Factor' in line:
                    results['profit_factor'] = self._extract_float(line)
                elif 'Total trades' in line:
                    results['total_trades'] = self._extract_integer(line)
                elif 'Starting balance' in line:
                    results['initial_balance'] = self._extract_float(line)
                elif 'Final balance' in line:
                    results['final_balance'] = self._extract_float(line)
            
            # Calculate missing values
            if 'initial_balance' in results and 'final_balance' in results:
                if 'total_return' not in results:
                    results['total_return'] = ((results['final_balance'] / results['initial_balance']) - 1) * 100
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing backtest output: {e}")
            return {}
    
    def _extract_percentage(self, line: str) -> float:
        """Extract percentage value from line"""
        try:
            import re
            match = re.search(r'([-+]?\d*\.?\d+)%', line)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _extract_float(self, line: str) -> float:
        """Extract float value from line"""
        try:
            import re
            match = re.search(r'([-+]?\d*\.?\d+)', line.split(':')[-1])
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _extract_integer(self, line: str) -> int:
        """Extract integer value from line"""
        try:
            import re
            match = re.search(r'(\d+)', line.split(':')[-1])
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0
    
    def download_data(self, symbol: str, trading_mode: str, days: int = 30) -> bool:
        """Download historical data for backtesting"""
        try:
            config_file = os.path.join(self.config_base_path, f"{trading_mode}.json")
            
            cmd = [
                "freqtrade", "download-data",
                "--config", config_file,
                "--pairs", symbol,
                "--timeframes", config.TIMEFRAME,
                "--days", str(days)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.freqtrade_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Downloaded data for {symbol}")
                return True
            else:
                logger.error(f"Failed to download data: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return False

# Global instance
backtest_runner = BacktestRunner()