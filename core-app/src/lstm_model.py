import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
from typing import Tuple, Dict, Optional
from datetime import datetime
from loguru import logger
from config import config
from db import db_manager

# TensorFlow/Keras imports with fallback and optimization
try:
    import tensorflow as tf
    # Configure TensorFlow for optimal performance
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    # Disable GPU if not available to avoid warnings
    tf.config.set_visible_devices([], 'GPU')
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow initialized with CPU optimization")
except ImportError:
    logger.warning("TensorFlow not available, using sklearn fallback")
    TENSORFLOW_AVAILABLE = False

class LSTMPredictor:
    def __init__(self, seq_length: int = None):
        self.seq_length = seq_length or config.SEQ_LENGTH
        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_names = None
        self.is_trained = False
        self.use_tensorflow = TENSORFLOW_AVAILABLE
        
        if not self.use_tensorflow:
            # Fallback to Random Forest when TensorFlow is not available
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
    def prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare high-quality sequences for LSTM training"""
        try:
            # Data quality checks
            if len(features) < self.seq_length + 10:
                logger.warning(f"Insufficient data for quality sequences: {len(features)}")
                return np.array([]), np.array([])
            
            # Remove outliers from target for better training
            target_q99 = np.percentile(np.abs(target), 99)
            valid_indices = np.abs(target) <= target_q99
            features_clean = features[valid_indices]
            target_clean = target[valid_indices]
            
            if len(features_clean) < len(features) * 0.8:
                logger.warning(f"Too many outliers removed: {len(features_clean)}/{len(features)}")
                features_clean, target_clean = features, target
            
            # Scale features and target
            features_scaled = self.scaler_features.fit_transform(features_clean)
            target_scaled = self.scaler_target.fit_transform(target_clean.reshape(-1, 1)).flatten()
            
            X, y = [], []
            
            # Create sequences with stride for better coverage
            stride = max(1, len(features_scaled) // (config.MAX_TRAINING_SAMPLES // 2))
            
            for i in range(self.seq_length, len(features_scaled), stride):
                X.append(features_scaled[i-self.seq_length:i])
                y.append(target_scaled[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} quality sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build optimized LSTM model architecture for faster training"""
        try:
            if not self.use_tensorflow:
                logger.info("Using Random Forest model as fallback")
                return self.model
                
            # Balanced architecture for quality and efficiency
            model = Sequential([
                LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
                Dropout(config.DROPOUT_RATE),
                
                LSTM(config.LSTM_UNITS // 2, return_sequences=True),
                Dropout(config.DROPOUT_RATE),
                
                LSTM(config.LSTM_UNITS // 4, return_sequences=False),
                Dropout(config.DROPOUT_RATE),
                
                Dense(32, activation='relu'),  # Increased for better learning
                Dropout(config.DROPOUT_RATE),
                
                Dense(16, activation='relu'),
                Dropout(0.1),
                
                Dense(1)
            ])
            
            # Adaptive learning rate for quality training
            model.compile(
                optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                loss='huber',  # More robust to outliers than MSE
                metrics=['mae', 'mape']
            )
            
            logger.info(f"Built optimized LSTM model with input shape: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, features: np.ndarray, target: np.ndarray, 
              validation_split: float = 0.2) -> Dict:
        """Train LSTM model"""
        try:
            if not self.use_tensorflow:
                # Train Random Forest model
                return self._train_sklearn_model(features, target, validation_split)
            
            # Prepare sequences
            X, y = self.prepare_sequences(features, target)
            
            if len(X) == 0:
                raise ValueError("No sequences prepared for training")
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Smart callbacks for quality training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    min_delta=0.0001,  # Very small improvement threshold
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,  # Less aggressive reduction
                    patience=8,  # More patience for learning rate reduction
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Train model with progress logging
            logger.info(f"Starting training with {len(X)} samples, {config.EPOCHS} epochs, batch size {config.BATCH_SIZE}")
            
            history = self.model.fit(
                X, y,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info(f"Training completed successfully in {len(history.history['loss'])} epochs")
            
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
    
    def _train_sklearn_model(self, features: np.ndarray, target: np.ndarray, 
                           validation_split: float = 0.2) -> Dict:
        """Train sklearn model as fallback"""
        try:
            # Scale features and target
            features_scaled = self.scaler_features.fit_transform(features)
            target_scaled = self.scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
            
            # Split data
            split_idx = int(len(features_scaled) * (1 - validation_split))
            X_train, X_val = features_scaled[:split_idx], features_scaled[split_idx:]
            y_train, y_val = target_scaled[:split_idx], target_scaled[split_idx:]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)
            
            # Calculate metrics
            train_loss = np.mean((y_train - y_pred_train) ** 2)
            val_loss = np.mean((y_val - y_pred_val) ** 2)
            
            # MAPE on validation set
            y_pred_original = self.scaler_target.inverse_transform(y_pred_val.reshape(-1, 1))
            y_true_original = self.scaler_target.inverse_transform(y_val.reshape(-1, 1))
            mape = mean_absolute_percentage_error(y_true_original, y_pred_original)
            
            self.is_trained = True
            
            metrics = {
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'mape': float(mape),
                'training_samples': len(features_scaled)
            }
            
            logger.info(f"Sklearn training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training sklearn model: {e}")
            raise
    
    def predict(self, features: np.ndarray, return_direction: bool = True) -> Dict:
        """Make predictions with the trained model"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model is not trained")
            
            # Input validation: check for NaN and Inf values
            if np.isnan(features).any():
                logger.warning("Input features contain NaN values, replacing with zeros")
                features = np.nan_to_num(features, nan=0.0)
                
            if np.isinf(features).any():
                logger.warning("Input features contain Inf values, replacing with finite values")
                features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
            
            # Handle different input shapes
            if features.ndim == 3:
                # Already shaped for LSTM (1, seq_length, n_features)
                if features.shape[0] != 1:
                    raise ValueError("Batch size must be 1 for prediction")
                
                # Reshape to 2D for scaling: (seq_length, n_features)
                features_2d = features[0]
                features_scaled = self.scaler_features.transform(features_2d)
                
                if self.use_tensorflow:
                    # Reshape back to 3D for LSTM
                    X = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
                else:
                    # Use last sample for sklearn
                    X = features_scaled[-1:]
                
                # Get current price from original features
                current_price = features[0, -1, 0]  # Last timestep, first feature (price)
                
            elif features.ndim == 2:
                # 2D input (n_samples, n_features) - traditional format
                features_scaled = self.scaler_features.transform(features)
                current_price = features[-1, 0]  # Last sample, first feature (price)
            else:
                raise ValueError(f"Unsupported input shape: {features.shape}")
            
            # Make prediction based on model type
            if not self.use_tensorflow:
                # Use sklearn model with last sample
                if features.ndim == 3:
                    X = features_scaled[-1:]
                else:
                    X = features_scaled[-1:]
                pred_scaled = self.model.predict(X)
                pred_price = self.scaler_target.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            else:
                # Use TensorFlow LSTM model
                if features.ndim == 3:
                    # Already prepared for LSTM
                    X = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
                else:
                    # Create sequence for prediction (last seq_length points)
                    if len(features_scaled) < self.seq_length:
                        raise ValueError(f"Need at least {self.seq_length} data points for prediction")
                    X = features_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
                
                # Make prediction
                pred_scaled = self.model.predict(X, verbose=0)
                pred_price = self.scaler_target.inverse_transform(pred_scaled)[0][0]
            
            # Validate prediction result
            if np.isnan(pred_price) or np.isinf(pred_price):
                logger.error(f"Model produced invalid prediction: {pred_price}")
                return {}
            
            if np.isnan(current_price) or np.isinf(current_price):
                logger.error(f"Current price is invalid: {current_price}")
                return {}
            
            # Calculate direction and confidence
            price_change = (pred_price - current_price) / current_price
            
            # Validate price change
            if np.isnan(price_change) or np.isinf(price_change):
                logger.error(f"Price change calculation resulted in invalid value: {price_change}")
                return {}
            
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
    
    def save_model(self, symbol: str, model_dir: str = None) -> str:
        """Save trained model and scalers"""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained")
            
            # Set default model directory in the project
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{'sklearn' if not self.use_tensorflow else 'lstm'}_{symbol.replace('/', '_')}_{timestamp}"
            model_path = os.path.join(model_dir, model_name)
            
            # Save model
            if self.use_tensorflow:
                self.model.save(f"{model_path}.h5")
            else:
                joblib.dump(self.model, f"{model_path}_model.pkl")
            
            # Save scalers
            joblib.dump(self.scaler_features, f"{model_path}_scaler_features.pkl")
            joblib.dump(self.scaler_target, f"{model_path}_scaler_target.pkl")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'seq_length': self.seq_length,
                'feature_names': self.feature_names,
                'use_tensorflow': self.use_tensorflow,
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
            # Load metadata first
            metadata = joblib.load(f"{model_path}_metadata.pkl")
            self.seq_length = metadata['seq_length']
            self.feature_names = metadata['feature_names']
            self.use_tensorflow = metadata.get('use_tensorflow', True)
            
            # Load model
            if self.use_tensorflow and TENSORFLOW_AVAILABLE:
                self.model = tf.keras.models.load_model(f"{model_path}.h5")
            else:
                self.model = joblib.load(f"{model_path}_model.pkl")
                self.use_tensorflow = False
            
            # Load scalers
            self.scaler_features = joblib.load(f"{model_path}_scaler_features.pkl")
            self.scaler_target = joblib.load(f"{model_path}_scaler_target.pkl")
            
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
            # If model not in memory, try to load from database
            if symbol not in self.models:
                if self._load_model_from_db(symbol):
                    logger.info(f"Loaded model for {symbol} from database")
                else:
                    logger.warning(f"No model available for {symbol}")
                    return {}
            
            return self.models[symbol].predict(features)
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return {}
    
    def _load_model_from_db(self, symbol: str) -> bool:
        """Load model from database into memory"""
        try:
            session = db_manager.get_session()
            
            # Get latest model for symbol
            model_record = session.query(db_manager.LSTMModel)\
                .filter(db_manager.LSTMModel.symbol == symbol)\
                .filter(db_manager.LSTMModel.is_active == True)\
                .order_by(db_manager.LSTMModel.created_at.desc())\
                .first()
            
            if model_record and model_record.model_path:
                # Load model
                predictor = LSTMPredictor()
                if predictor.load_model(model_record.model_path):
                    self.models[symbol] = predictor
                    session.close()
                    return True
            
            session.close()
            return False
            
        except Exception as e:
            logger.error(f"Error loading model from DB for {symbol}: {e}")
            return False
    
    def should_retrain(self, symbol: str, current_data_size: int) -> bool:
        """Check if model should be retrained"""
        # Retrain every RETRAIN_EVERY candles or if no model exists
        return symbol not in self.models or current_data_size % config.RETRAIN_EVERY == 0

# Global instance
model_manager = ModelManager()