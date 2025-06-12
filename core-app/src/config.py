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
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5433"))
    
    # Trading Parameters
    TIMEFRAME: str = os.getenv("TIMEFRAME", "15m")
    SEQ_LENGTH: int = int(os.getenv("SEQ_LENGTH", "60"))
    BT_RANGE: str = os.getenv("BT_RANGE", "20240101-20240601")
    RETRAIN_EVERY: int = int(os.getenv("RETRAIN_EVERY", "96"))
    
    # Symbols to trade
    SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    
    # Model parameters - Balanced for quality and speed
    LSTM_UNITS: int = 50  # Restored for better capacity
    DROPOUT_RATE: float = 0.2
    EPOCHS: int = 100  # Increased with early stopping for quality
    BATCH_SIZE: int = 32  # Balanced for stability
    
    # Training optimization parameters
    EARLY_STOPPING_PATIENCE: int = 15  # Allow more time for convergence
    MIN_TRAINING_SAMPLES: int = 500  # Minimum data for quality training
    MAX_TRAINING_SAMPLES: int = 2000  # Increased limit for better models
    
    # Risk management
    MAX_POSITIONS: int = 3
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    
    # Portfolio management
    INITIAL_PORTFOLIO_VALUE: float = float(os.getenv("INITIAL_PORTFOLIO_VALUE", "10000"))
    
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
import os
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

logger.add(
    os.path.join(log_dir, "trade_system_{time}.log"),
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    serialize=True
)