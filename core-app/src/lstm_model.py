import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
from typing import Tuple, Dict, Optional
from datetime import datetime
from loguru import logger
from config import config
from db import db_manager

class LSTMPredictor:
    def __init__(self, seq_length: int = None):
        self.seq_length = seq_length or config.SEQ_LENGTH
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        try:
            # Scale features and target
            features_scaled = self.scaler_features.fit_transform(features)
            target_scaled = self.scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
            
            X, y = [], []
            
            for i in range(self.seq_length, len(features_scaled)):
                X.append(features_scaled[i-self.seq_length:i])
                y.append(target_scaled[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        try:
            model = Sequential([
                LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
                Dropout(config.DROPOUT_RATE),
                
                LSTM(config.LSTM_UNITS // 2, return_sequences=True),
                Dropout(config.DROPOUT_RATE),
                
                LSTM(config.LSTM_UNITS // 4, return_sequences=False),
                Dropout(config.DROPOUT_RATE),
                
                Dense(25, activation='relu'),
                Dropout(config.DROPOUT_RATE),
                
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"Built LSTM model with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, features: np.ndarray, target: np.ndarray, 
              validation_split: float = 0.2) -> Dict:
        """Train LSTM model"""
        try:
            # Prepare sequences
            X, y = self.prepare_sequences(features, target)
            
            if len(X) == 0:
                raise ValueError("No sequences prepared for training")
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate metrics
            train_loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            
            # Calculate MAPE on validation set
            val_size = int(len(X) * validation_split)
            X_val = X[-val_size:]
            y_val = y[-val_size:]
            
            y_pred_scaled = self.model.predict(X_val)
            y_pred = self.scaler_target.inverse_transform(y_pred_scaled)
            y_true = self.scaler_target.inverse_transform(y_val.reshape(-1, 1))
            
            mape = mean_absolute_percentage_error(y_true, y_pred)
            
            self.is_trained = True
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mape': mape,
                'training_samples': len(X)
            }
            
            logger.info(f"Training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, features: np.ndarray, return_direction: bool = True) -> Dict:
        """Make predictions with the trained model"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model is not trained")
            
            # Scale features
            features_scaled = self.scaler_features.transform(features)
            
            # Create sequence for prediction (last seq_length points)
            if len(features_scaled) < self.seq_length:
                raise ValueError(f"Need at least {self.seq_length} data points for prediction")
            
            X = features_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
            
            # Make prediction
            pred_scaled = self.model.predict(X, verbose=0)
            pred_price = self.scaler_target.inverse_transform(pred_scaled)[0][0]
            
            # Calculate direction and confidence
            current_price = features[-1, 0] if features.shape[1] > 0 else pred_price
            price_change = (pred_price - current_price) / current_price
            
            if return_direction:
                if price_change > 0.001:  # > 0.1%
                    direction = 1
                elif price_change < -0.001:  # < -0.1%
                    direction = -1
                else:
                    direction = 0
            else:
                direction = None
            
            confidence = min(1.0, abs(price_change) * 10)  # Scale confidence
            
            result = {
                'predicted_price': float(pred_price),
                'current_price': float(current_price),
                'price_change_pct': float(price_change * 100),
                'direction': direction,
                'confidence': float(confidence),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Prediction: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {}
    
    def save_model(self, symbol: str, model_dir: str = "/app/models") -> str:
        """Save trained model and scalers"""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained")
            
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_{symbol.replace('/', '_')}_{timestamp}"
            model_path = os.path.join(model_dir, model_name)
            
            # Save model
            self.model.save(f"{model_path}.h5")
            
            # Save scalers
            joblib.dump(self.scaler_features, f"{model_path}_scaler_features.pkl")
            joblib.dump(self.scaler_target, f"{model_path}_scaler_target.pkl")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'seq_length': self.seq_length,
                'feature_names': self.feature_names,
                'created_at': datetime.now().isoformat()
            }
            joblib.dump(metadata, f"{model_path}_metadata.pkl")
            
            logger.info(f"Model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model and scalers"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{model_path}.h5")
            
            # Load scalers
            self.scaler_features = joblib.load(f"{model_path}_scaler_features.pkl")
            self.scaler_target = joblib.load(f"{model_path}_scaler_target.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{model_path}_metadata.pkl")
            self.seq_length = metadata['seq_length']
            self.feature_names = metadata['feature_names']
            
            self.is_trained = True
            
            logger.info(f"Model loaded from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_performance(self, features: np.ndarray, target: np.ndarray, 
                            test_size: int = 100) -> Dict:
        """Evaluate model performance on recent data"""
        try:
            if not self.is_trained:
                return {}
            
            # Use last test_size samples
            test_features = features[-test_size:]
            test_target = target[-test_size:]
            
            # Make predictions
            predictions = []
            for i in range(self.seq_length, len(test_features)):
                features_seq = test_features[i-self.seq_length:i]
                pred = self.predict(features_seq, return_direction=False)
                predictions.append(pred.get('predicted_price', 0))
            
            if len(predictions) == 0:
                return {}
            
            # Calculate metrics
            actual = test_target[self.seq_length:]
            predicted = np.array(predictions)
            
            mape = mean_absolute_percentage_error(actual, predicted)
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            direction_accuracy = np.mean(actual_direction == predicted_direction)
            
            return {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'direction_accuracy': float(direction_accuracy),
                'test_samples': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}

class ModelManager:
    def __init__(self):
        self.models: Dict[str, LSTMPredictor] = {}
    
    def train_model_for_symbol(self, symbol: str, features: np.ndarray, 
                              target: np.ndarray) -> Dict:
        """Train LSTM model for specific symbol"""
        try:
            predictor = LSTMPredictor()
            predictor.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
            
            # Train model
            metrics = predictor.train(features, target)
            
            # Save model
            model_path = predictor.save_model(symbol)
            
            # Save to database
            model_id = db_manager.save_model_info(
                symbol=symbol,
                model_path=model_path,
                training_size=metrics['training_samples'],
                train_loss=metrics['train_loss'],
                val_loss=metrics['val_loss'],
                mape=metrics['mape']
            )
            
            # Store in memory
            self.models[symbol] = predictor
            
            metrics['model_id'] = model_id
            metrics['model_path'] = model_path
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {}
    
    def get_prediction(self, symbol: str, features: np.ndarray) -> Dict:
        """Get prediction for symbol"""
        try:
            if symbol not in self.models:
                logger.warning(f"No model available for {symbol}")
                return {}
            
            return self.models[symbol].predict(features)
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return {}
    
    def should_retrain(self, symbol: str, current_data_size: int) -> bool:
        """Check if model should be retrained"""
        # Retrain every RETRAIN_EVERY candles or if no model exists
        return symbol not in self.models or current_data_size % config.RETRAIN_EVERY == 0

# Global instance
model_manager = ModelManager()