import os
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "trade")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "tradepass")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "tradedb")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    
    # Trading Parameters
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")
    SEQ_LENGTH: int = int(os.getenv("SEQ_LENGTH", "60"))
    BT_RANGE: str = os.getenv("BT_RANGE", "20240101-20240601")
    RETRAIN_EVERY: int = int(os.getenv("RETRAIN_EVERY", "96"))
    
    # Symbols to trade
    SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    
    # Model parameters
    LSTM_UNITS: int = 50
    DROPOUT_RATE: float = 0.2
    EPOCHS: int = 50
    BATCH_SIZE: int = 32
    
    # Risk management
    MAX_POSITIONS: int = 3
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def validate(self) -> bool:
        """Validate required configuration"""
        required_fields = [
            "OPENAI_API_KEY",
            "BINANCE_API_KEY", 
            "BINANCE_API_SECRET"
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                logger.error(f"Missing required configuration: {field}")
                return False
        
        return True

config = Config()

# Configure logging
logger.add(
    "/app/logs/trade_system_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    serialize=True
)