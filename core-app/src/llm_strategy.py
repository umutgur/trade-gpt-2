import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from openai import OpenAI
from loguru import logger
from config import config
from db import db_manager

class LLMStrategyEngine:
    def __init__(self):
        # Strategy caching for cost optimization
        self._strategy_cache = {}
        self._cache_duration_hours = 6  # Cache strategies for 6 hours (longer cache for performance)
        try:
            # Initialize OpenAI client with minimal parameters
            if config.OPENAI_API_KEY and config.OPENAI_API_KEY.strip():
                # Try different initialization methods to avoid httpx conflicts
                try:
                    import httpx
                    # Create httpx client with optimized settings
                    http_client = httpx.Client(
                        timeout=httpx.Timeout(60.0, connect=10.0),
                        limits=httpx.Limits(max_keepalive_connections=1, max_connections=1),
                        headers={"Connection": "close"}
                    )
                    self.client = OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        http_client=http_client,
                        max_retries=1  # Reduce retries
                    )
                    self.llm_available = True
                    logger.info("OpenAI client initialized with custom httpx client")
                except Exception as httpx_error:
                    logger.warning(f"Custom httpx failed: {httpx_error}, trying basic initialization")
                    # Fallback: try basic initialization with reduced retries
                    self.client = OpenAI(
                        api_key=config.OPENAI_API_KEY,
                        max_retries=1,
                        timeout=60.0
                    )
                    self.llm_available = True
            else:
                logger.warning("OpenAI API key not available, using fallback strategies")
                self.client = None
                self.llm_available = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # Create a dummy client for development
            self.client = None
            self.llm_available = False
        self.max_indicators = 3
        self.max_conditions = 3
    
    def generate_strategy(self, symbol: str, market_data: pd.DataFrame, 
                         performance_summary: Dict, trading_mode: str = "spot") -> Dict:
        """Generate trading strategy using GPT-4 with smart caching"""
        try:
            # Check cache first to save costs
            cache_key = f"{symbol}_{trading_mode}"
            if cache_key in self._strategy_cache:
                cached_time, cached_strategy = self._strategy_cache[cache_key]
                hours_since_cache = (datetime.now() - cached_time).total_seconds() / 3600
                
                if hours_since_cache < self._cache_duration_hours:
                    logger.info(f"Using cached strategy for {symbol} ({trading_mode}) - {hours_since_cache:.1f}h old")
                    return cached_strategy
            # If LLM is not available, use default strategy
            if not self.llm_available:
                logger.info(f"Using default strategy for {symbol} ({trading_mode}) - LLM not available")
                strategy_data = self._get_default_strategy()
                
                # Save to database
                strategy_id = db_manager.save_strategy(
                    symbol=symbol,
                    trading_mode=trading_mode,
                    strategy_data=strategy_data,
                    reasoning="Default strategy - LLM not available"
                )
                
                strategy_data['strategy_id'] = strategy_id
                return strategy_data
            
            # Prepare market context
            market_context = self._prepare_market_context(market_data)
            
            # Create prompt
            prompt = self._create_strategy_prompt(
                symbol, market_context, performance_summary, trading_mode
            )
            
            # Enhanced prompt for direct JSON response
            enhanced_prompt = prompt + """

IMPORTANT: Respond with valid JSON in this exact format:
{
  "indicators": [
    {"name": "EMA", "parameters": {"period": 20}, "reasoning": "Trend identification"},
    {"name": "RSI", "parameters": {"period": 14}, "reasoning": "Momentum analysis"}
  ],
  "buy_conditions": [
    {"condition": "close > ema_20", "threshold": 0, "operator": ">"},
    {"condition": "rsi", "threshold": 30, "operator": "<"}
  ],
  "sell_conditions": [
    {"condition": "close < ema_20", "threshold": 0, "operator": "<"},
    {"condition": "rsi", "threshold": 70, "operator": ">"}
  ],
  "risk_management": {
    "stop_loss": 2.0,
    "take_profit": 4.0,
    "position_size": 10.0,
    "max_positions": 2
  },
  "market_outlook": "bullish",
  "confidence": 0.75
}

Respond ONLY with valid JSON, no other text."""

            # Call GPT-4 for better strategy quality
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            # Parse JSON response directly
            strategy_text = response.choices[0].message.content.strip()
            if strategy_text.startswith('```json'):
                strategy_text = strategy_text.split('```json')[1].split('```')[0].strip()
            elif strategy_text.startswith('```'):
                strategy_text = strategy_text.split('```')[1].strip()
            
            strategy_data = json.loads(strategy_text)
            
            # Validate and clean strategy
            strategy_data = self._validate_strategy(strategy_data)
            
            # Save to database
            strategy_id = db_manager.save_strategy(
                symbol=symbol,
                trading_mode=trading_mode,
                strategy_data=strategy_data,
                reasoning=response.choices[0].message.content or "Generated by GPT-4"
            )
            
            strategy_data['strategy_id'] = strategy_id
            
            # Cache the strategy for cost optimization
            self._strategy_cache[cache_key] = (datetime.now(), strategy_data.copy())
            
            logger.info(f"Generated NEW strategy for {symbol} ({trading_mode}) - Cost: $0.029")
            return strategy_data
            
        except Exception as e:
            logger.error(f"Error generating strategy for {symbol}: {e}")
            return self._get_default_strategy()
    
    def _prepare_market_context(self, df: pd.DataFrame) -> Dict:
        """Prepare market context from data"""
        try:
            if df.empty:
                return {}
            
            latest = df.iloc[-1]
            
            # Price action
            price_change_1h = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
            price_change_4h = ((df['close'].iloc[-1] / df['close'].iloc[-16]) - 1) * 100 if len(df) >= 16 else 0
            price_change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-96]) - 1) * 100 if len(df) >= 96 else 0
            
            # Volatility
            volatility_24h = df['close'].pct_change().rolling(96).std().iloc[-1] * 100 if len(df) >= 96 else 0
            
            # Volume analysis
            avg_volume = df['volume'].rolling(24).mean().iloc[-1] if len(df) >= 24 else latest['volume']
            volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 1
            
            # Support/Resistance levels
            recent_high = df['high'].rolling(48).max().iloc[-1] if len(df) >= 48 else latest['high']
            recent_low = df['low'].rolling(48).min().iloc[-1] if len(df) >= 48 else latest['low']
            
            return {
                'current_price': latest['close'],
                'price_change_1h': round(price_change_1h, 2),
                'price_change_4h': round(price_change_4h, 2),
                'price_change_24h': round(price_change_24h, 2),
                'volatility_24h': round(volatility_24h, 2),
                'volume_ratio': round(volume_ratio, 2),
                'recent_high': recent_high,
                'recent_low': recent_low,
                'timestamp': latest['timestamp'] if 'timestamp' in latest else datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error preparing market context: {e}")
            return {}
    
    def _create_strategy_prompt(self, symbol: str, market_context: Dict, 
                              performance_summary: Dict, trading_mode: str) -> str:
        """Create strategy generation prompt"""
        prompt = f"""
Analyze the current market conditions for {symbol} and generate a trading strategy.

MARKET CONTEXT:
- Current Price: ${market_context.get('current_price', 'N/A')}
- 1H Change: {market_context.get('price_change_1h', 0):.2f}%
- 4H Change: {market_context.get('price_change_4h', 0):.2f}%
- 24H Change: {market_context.get('price_change_24h', 0):.2f}%
- 24H Volatility: {market_context.get('volatility_24h', 0):.2f}%
- Volume Ratio: {market_context.get('volume_ratio', 1):.2f}x
- Recent High: ${market_context.get('recent_high', 'N/A')}
- Recent Low: ${market_context.get('recent_low', 'N/A')}

TRADING MODE: {trading_mode.upper()}
{"- Leverage: 3x (Margin)" if trading_mode == "margin" else ""}
{"- Leverage: 2x (Futures)" if trading_mode == "futures" else ""}

PREVIOUS PERFORMANCE:
- Total Return: {performance_summary.get('total_return', 0):.2f}%
- Win Rate: {performance_summary.get('win_rate', 0):.2f}%
- Sharpe Ratio: {performance_summary.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance_summary.get('max_drawdown', 0):.2f}%

REQUIREMENTS:
1. Use maximum 3 technical indicators from: EMA, RSI, MACD, Bollinger Bands, ADX, Stochastic
2. Create maximum 3 buy conditions and 3 sell conditions
3. Include risk management parameters
4. Consider the trading mode's leverage and margin requirements
5. Adapt to current market volatility and trend

Generate a strategy that balances profit potential with risk management.
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for strategy generation"""
        return """
You are an expert cryptocurrency trading strategist. Your role is to analyze market data and generate precise, actionable trading strategies.

KEY PRINCIPLES:
1. Risk Management First: Always prioritize capital preservation
2. Simplicity: Use maximum 3 indicators and 3 conditions
3. Adaptability: Adjust to market conditions and volatility
4. Precision: Provide specific numerical thresholds
5. Mode Awareness: Consider leverage implications for margin/futures

AVAILABLE INDICATORS:
- EMA (Exponential Moving Average): Trend following
- RSI (Relative Strength Index): Momentum oscillator (0-100)
- MACD: Trend and momentum combination
- Bollinger Bands: Volatility and mean reversion
- ADX: Trend strength (0-100)
- Stochastic: Momentum oscillator (0-100)

RISK MANAGEMENT:
- Spot: Conservative position sizing
- Margin (3x): Moderate leverage with tight stops
- Futures (2x): Lower leverage with wider stops

Always provide specific, actionable strategies with clear entry/exit criteria.
"""
    
    def _get_strategy_function_schema(self) -> Dict:
        """Get simplified function schema for cost optimization"""
        return {
            "name": "generate_trading_strategy",
            "description": "Generate trading strategy",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {"type": "array", "items": {"type": "object"}},
                    "buy_conditions": {"type": "array", "items": {"type": "object"}},
                    "sell_conditions": {"type": "array", "items": {"type": "object"}},
                    "risk_management": {"type": "object"},
                    "market_outlook": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["indicators", "buy_conditions", "sell_conditions", "risk_management", "market_outlook", "confidence"]
            }
        }
    
    def _validate_strategy(self, strategy: Dict) -> Dict:
        """Validate and clean strategy data"""
        try:
            # Ensure max limits
            if len(strategy.get('indicators', [])) > self.max_indicators:
                strategy['indicators'] = strategy['indicators'][:self.max_indicators]
            
            if len(strategy.get('buy_conditions', [])) > self.max_conditions:
                strategy['buy_conditions'] = strategy['buy_conditions'][:self.max_conditions]
            
            if len(strategy.get('sell_conditions', [])) > self.max_conditions:
                strategy['sell_conditions'] = strategy['sell_conditions'][:self.max_conditions]
            
            # Validate risk management
            risk = strategy.get('risk_management', {})
            risk['stop_loss'] = max(0.5, min(10, risk.get('stop_loss', 2)))
            risk['take_profit'] = max(1, min(20, risk.get('take_profit', 5)))
            risk['position_size'] = max(1, min(50, risk.get('position_size', 10)))
            risk['max_positions'] = max(1, min(5, risk.get('max_positions', 2)))
            
            # Ensure confidence is valid
            strategy['confidence'] = max(0, min(1, strategy.get('confidence', 0.5)))
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error validating strategy: {e}")
            return self._get_default_strategy()
    
    def _get_default_strategy(self) -> Dict:
        """Get default conservative strategy"""
        return {
            "indicators": [
                {"name": "EMA", "parameters": {"period": 20}, "reasoning": "Trend following"},
                {"name": "RSI", "parameters": {"period": 14}, "reasoning": "Momentum confirmation"}
            ],
            "buy_conditions": [
                {"condition": "close > ema_20", "threshold": 0, "operator": ">"},
                {"condition": "rsi", "threshold": 70, "operator": "<"}
            ],
            "sell_conditions": [
                {"condition": "close < ema_20", "threshold": 0, "operator": "<"},
                {"condition": "rsi", "threshold": 30, "operator": ">"}
            ],
            "risk_management": {
                "stop_loss": 2.0,
                "take_profit": 4.0,
                "position_size": 10.0,
                "max_positions": 2
            },
            "market_outlook": "neutral",
            "confidence": 0.5
        }
    
    def provide_feedback(self, strategy_id: int, performance_data: Dict) -> str:
        """Provide feedback to improve future strategies"""
        try:
            if not self.llm_available:
                return self._get_default_feedback(performance_data)
            
            feedback_prompt = f"""
Analyze the performance of trading strategy ID {strategy_id}:

PERFORMANCE METRICS:
- Total Return: {performance_data.get('total_return', 0):.2f}%
- Win Rate: {performance_data.get('win_rate', 0):.2f}%
- Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance_data.get('max_drawdown', 0):.2f}%
- Total Trades: {performance_data.get('total_trades', 0)}
- Average P&L per Trade: ${performance_data.get('avg_pnl_per_trade', 0):.2f}
- LSTM Prediction Accuracy: {performance_data.get('prediction_accuracy', 0):.2%}
- Period: {performance_data.get('period', 'Unknown')}
- Source: {performance_data.get('source', 'Backtest')} Trading

What improvements can be made to the strategy? Focus on:
1. Risk management adjustments
2. Entry/exit condition optimization
3. Indicator parameter tuning
4. Position sizing modifications

Provide specific, actionable recommendations.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading strategy analyst providing improvement feedback."},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            feedback = response.choices[0].message.content
            logger.info(f"Generated feedback for strategy {strategy_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return "Unable to generate feedback at this time."
    
    def _get_default_feedback(self, performance_data: Dict) -> str:
        """Generate simple rule-based feedback when LLM is unavailable"""
        total_return = performance_data.get('total_return', 0)
        win_rate = performance_data.get('win_rate', 0)
        max_drawdown = performance_data.get('max_drawdown', 0)
        
        feedback = "Performance Analysis (Rule-based):\n\n"
        
        if total_return < 0:
            feedback += "• Strategy is underperforming. Consider tightening entry conditions.\n"
        elif total_return > 10:
            feedback += "• Strategy shows good returns. Monitor for sustainability.\n"
        else:
            feedback += "• Strategy shows moderate performance. Room for improvement.\n"
        
        if win_rate < 50:
            feedback += "• Low win rate detected. Review entry signals and market conditions.\n"
        elif win_rate > 70:
            feedback += "• High win rate is excellent. Monitor position sizing.\n"
        
        if max_drawdown > 10:
            feedback += "• High drawdown risk. Consider reducing position sizes or tightening stop losses.\n"
        elif max_drawdown < 5:
            feedback += "• Low drawdown indicates good risk management.\n"
        
        return feedback

# Global instance
llm_engine = LLMStrategyEngine()