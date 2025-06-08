from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
from loguru import logger
from config import config

Base = declarative_base()

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())

class Strategy(Base):
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    trading_mode = Column(String(20), nullable=False)  # spot, margin, futures
    strategy_data = Column(JSON, nullable=False)
    llm_reasoning = Column(Text)
    timestamp = Column(DateTime, default=func.now(), index=True)
    is_active = Column(Boolean, default=True)

class LSTMModel(Base):
    __tablename__ = "lstm_models"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_path = Column(String(255), nullable=False)
    training_data_size = Column(Integer, nullable=False)
    train_loss = Column(Float)
    val_loss = Column(Float)
    mape = Column(Float)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)

class Backtest(Base):
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    trading_mode = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    trading_mode = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    is_paper = Column(Boolean, default=True)
    profit_loss = Column(Float)
    created_at = Column(DateTime, default=func.now())

class Metrics(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    trading_mode = Column(String(20), nullable=False)
    metric_type = Column(String(50), nullable=False)  # roi, sharpe, drawdown, mape, etc.
    metric_value = Column(Float, nullable=False)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    timestamp = Column(DateTime, default=func.now(), index=True)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(config.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def insert_market_data(self, data: list[dict]) -> bool:
        """Insert market data batch"""
        session = self.get_session()
        try:
            for row in data:
                market_data = MarketData(**row)
                session.add(market_data)
            session.commit()
            logger.info(f"Inserted {len(data)} market data records")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting market data: {e}")
            return False
        finally:
            session.close()
    
    def get_latest_data(self, symbol: str, limit: int = 1000) -> list[MarketData]:
        """Get latest market data for symbol"""
        session = self.get_session()
        try:
            return session.query(MarketData)\
                .filter(MarketData.symbol == symbol)\
                .order_by(MarketData.timestamp.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    def save_strategy(self, symbol: str, trading_mode: str, strategy_data: dict, reasoning: str) -> int:
        """Save strategy and return ID"""
        session = self.get_session()
        try:
            strategy = Strategy(
                symbol=symbol,
                trading_mode=trading_mode,
                strategy_data=strategy_data,
                llm_reasoning=reasoning
            )
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            return strategy.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving strategy: {e}")
            raise
        finally:
            session.close()
    
    def save_model_info(self, symbol: str, model_path: str, training_size: int, 
                       train_loss: float, val_loss: float, mape: float) -> int:
        """Save LSTM model information"""
        session = self.get_session()
        try:
            # Deactivate old models
            session.query(LSTMModel)\
                .filter(LSTMModel.symbol == symbol)\
                .update({"is_active": False})
            
            # Create new model record
            model = LSTMModel(
                symbol=symbol,
                model_path=model_path,
                training_data_size=training_size,
                train_loss=train_loss,
                val_loss=val_loss,
                mape=mape
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return model.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving model info: {e}")
            raise
        finally:
            session.close()
    
    def save_backtest_results(self, strategy_id: int, symbol: str, trading_mode: str,
                            start_date: datetime, end_date: datetime, results: dict) -> int:
        """Save backtest results"""
        session = self.get_session()
        try:
            backtest = Backtest(
                strategy_id=strategy_id,
                symbol=symbol,
                trading_mode=trading_mode,
                start_date=start_date,
                end_date=end_date,
                initial_balance=results.get("initial_balance", 10000),
                final_balance=results.get("final_balance", 0),
                total_return=results.get("total_return", 0),
                sharpe_ratio=results.get("sharpe_ratio"),
                max_drawdown=results.get("max_drawdown"),
                win_rate=results.get("win_rate"),
                profit_factor=results.get("profit_factor"),
                total_trades=results.get("total_trades", 0)
            )
            session.add(backtest)
            session.commit()
            session.refresh(backtest)
            return backtest.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving backtest results: {e}")
            raise
        finally:
            session.close()

db_manager = DatabaseManager()