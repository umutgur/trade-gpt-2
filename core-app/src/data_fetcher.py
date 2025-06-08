import ccxt
import pandas as pd
import asyncio
import websocket
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from loguru import logger
from config import config
from db import db_manager

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'sandbox': False,  # Set to True for testnet
            'enableRateLimit': True,
        })
        self.ws_clients: Dict[str, websocket.WebSocketApp] = {}
        self.price_callbacks: List[Callable] = []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        if timeframe is None:
            timeframe = config.TIMEFRAME
        
        try:
            logger.info(f"Fetching {symbol} {timeframe} data, limit: {limit}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                            timeframe: str = None) -> pd.DataFrame:
        """Fetch historical data for a date range"""
        if timeframe is None:
            timeframe = config.TIMEFRAME
        
        try:
            start_timestamp = int(datetime.strptime(start_date, '%Y%m%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y%m%d').timestamp() * 1000)
            
            all_data = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_timestamp, limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    current_timestamp = ohlcv[-1][0] + 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error in batch fetch: {e}")
                    break
            
            if all_data:
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                logger.info(f"Fetched {len(df)} historical candles for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame) -> bool:
        """Save market data to database"""
        if df.empty:
            return False
        
        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    'symbol': row['symbol'],
                    'timeframe': row['timeframe'],
                    'timestamp': row['timestamp'],
                    'open_price': row['open'],
                    'high_price': row['high'],
                    'low_price': row['low'],
                    'close_price': row['close'],
                    'volume': row['volume']
                })
            
            return db_manager.insert_market_data(records)
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return False
    
    def start_websocket(self, symbol: str, callback: Optional[Callable] = None):
        """Start WebSocket for real-time price data"""
        stream_name = f"{symbol.lower().replace('/', '').replace('-', '')}@kline_{config.TIMEFRAME}"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                kline_data = data['k']
                
                price_data = {
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(kline_data['t'] / 1000),
                    'open': float(kline_data['o']),
                    'high': float(kline_data['h']),
                    'low': float(kline_data['l']),
                    'close': float(kline_data['c']),
                    'volume': float(kline_data['v']),
                    'is_closed': kline_data['x']  # True if kline is closed
                }
                
                # Call custom callback if provided
                if callback:
                    callback(price_data)
                
                # Call all registered callbacks
                for cb in self.price_callbacks:
                    cb(price_data)
                
                # Save closed candles to database
                if price_data['is_closed']:
                    df = pd.DataFrame([{
                        'symbol': price_data['symbol'],
                        'timeframe': config.TIMEFRAME,
                        'timestamp': price_data['timestamp'],
                        'open': price_data['open'],
                        'high': price_data['high'],
                        'low': price_data['low'],
                        'close': price_data['close'],
                        'volume': price_data['volume']
                    }])
                    self.save_to_database(df)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error for {symbol}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed for {symbol}")
        
        def on_open(ws):
            logger.info(f"WebSocket opened for {symbol}")
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_clients[symbol] = ws
        
        # Run in separate thread
        def run_ws():
            ws.run_forever()
        
        thread = threading.Thread(target=run_ws, daemon=True)
        thread.start()
        
        logger.info(f"Started WebSocket for {symbol}")
    
    def stop_websocket(self, symbol: str):
        """Stop WebSocket for symbol"""
        if symbol in self.ws_clients:
            self.ws_clients[symbol].close()
            del self.ws_clients[symbol]
            logger.info(f"Stopped WebSocket for {symbol}")
    
    def register_price_callback(self, callback: Callable):
        """Register callback for real-time price updates"""
        self.price_callbacks.append(callback)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def fetch_multiple_symbols(self, symbols: List[str], timeframe: str = None, 
                                   limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        if timeframe is None:
            timeframe = config.TIMEFRAME
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._fetch_async(symbol, timeframe, limit))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbols[i]}: {result}")
                data_dict[symbols[i]] = pd.DataFrame()
            else:
                data_dict[symbols[i]] = result
        
        return data_dict
    
    async def _fetch_async(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Async wrapper for fetch_ohlcv"""
        return self.fetch_ohlcv(symbol, timeframe, limit)

# Global instance
data_fetcher = DataFetcher()