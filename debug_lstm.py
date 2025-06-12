#!/usr/bin/env python3
"""Debug script to understand LSTM signal generation"""

import sys
import os
sys.path.append('/Users/umutgur/trade-gpt-2/core-app/src')

# Mock the database dependency to avoid connection issues
import unittest.mock

# Mock the database manager
class MockDatabaseManager:
    def get_session(self):
        return None
    
    def save_model_info(self, *args, **kwargs):
        return 1

# Mock the session and all database operations
mock_session = unittest.mock.MagicMock()
mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.first.return_value = None
mock_session.close.return_value = None

with unittest.mock.patch('db.db_manager', MockDatabaseManager()):
    with unittest.mock.patch('db.db_manager.get_session', return_value=mock_session):
        try:
            from data_fetcher import data_fetcher
            from lstm_model import model_manager
            from ta_features import ta_analyzer
            import numpy as np
            from loguru import logger
            
            # Test the full pipeline to understand where "No LSTM signal" comes from
            symbol = 'BTC/USDT'
            
            print(f"=== Testing LSTM Signal Generation for {symbol} ===\n")
            
            # Step 1: Test data fetching
            print("1. Testing data fetching...")
            try:
                market_data = data_fetcher.fetch_ohlcv(symbol, limit=100)
                if market_data.empty:
                    print("❌ No market data retrieved")
                    exit(1)
                else:
                    print(f"✅ Retrieved {len(market_data)} data points")
                    print(f"   Latest close price: ${market_data['close'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"❌ Data fetching failed: {e}")
                exit(1)
            
            # Step 2: Test technical analysis features
            print("\n2. Testing technical analysis features...")
            try:
                enhanced_data = ta_analyzer.add_all_features(market_data)
                ml_features = ta_analyzer.get_features_for_ml(enhanced_data)
                
                if not ml_features or len(ml_features.get('features', [])) < 60:
                    print(f"❌ Insufficient features: {len(ml_features.get('features', []))} < 60")
                    exit(1)
                else:
                    print(f"✅ Generated {len(ml_features['features'])} feature points")
                    print(f"   Feature shape: {np.array(ml_features['features']).shape}")
                    print(f"   Target shape: {np.array(ml_features['target']).shape}")
            except Exception as e:
                print(f"❌ Feature generation failed: {e}")
                exit(1)
            
            # Step 3: Test model training (to create a model)
            print("\n3. Testing model training...")
            try:
                features = np.array(ml_features['features'])
                target = np.array(ml_features['target'])
                
                # Train a model for this symbol
                training_metrics = model_manager.train_model_for_symbol(symbol, features, target)
                
                if training_metrics:
                    print(f"✅ Model training completed")
                    print(f"   Training samples: {training_metrics.get('training_samples', 'N/A')}")
                    print(f"   MAPE: {training_metrics.get('mape', 'N/A'):.4f}")
                else:
                    print("❌ Model training failed - no metrics returned")
                    exit(1)
            except Exception as e:
                print(f"❌ Model training failed: {e}")
                exit(1)
            
            # Step 4: Test prediction
            print("\n4. Testing LSTM prediction...")
            try:
                # Prepare input for prediction (last 60 time steps)
                features_for_prediction = features[-60:].reshape(1, 60, -1)
                
                print(f"   Input shape for prediction: {features_for_prediction.shape}")
                
                prediction_result = model_manager.get_prediction(symbol, features_for_prediction)
                
                if not prediction_result:
                    print("❌ No prediction result returned")
                    print("   This is likely the source of 'No LSTM signal'")
                    
                    # Debug: Check if model exists in memory
                    if symbol in model_manager.models:
                        print(f"   ✅ Model exists in memory for {symbol}")
                        print(f"   Model type: {type(model_manager.models[symbol])}")
                        print(f"   Model is_trained: {model_manager.models[symbol].is_trained}")
                    else:
                        print(f"   ❌ No model in memory for {symbol}")
                        
                    # Try to manually call the predict method
                    if symbol in model_manager.models:
                        predictor = model_manager.models[symbol]
                        try:
                            manual_prediction = predictor.predict(features_for_prediction)
                            print(f"   Manual prediction result: {manual_prediction}")
                        except Exception as e:
                            print(f"   Manual prediction error: {e}")
                            
                else:
                    print(f"✅ Prediction successful: {prediction_result}")
                    
                    current_price = market_data['close'].iloc[-1]
                    predicted_price = prediction_result.get('predicted_price', current_price)
                    confidence = prediction_result.get('confidence', 0.5)
                    expected_return = (predicted_price - current_price) / current_price
                    
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   Predicted Price: ${predicted_price:.2f}")
                    print(f"   Expected Return: {expected_return*100:.2f}%")
                    print(f"   Confidence: {confidence:.3f}")
                    
                    # Test trading decision logic
                    print("\n5. Testing trading decision logic...")
                    min_confidence = 0.6
                    min_expected_return = 0.02
                    
                    if confidence < min_confidence:
                        print(f"   ❌ Would result in 'No LSTM signal' - Low confidence: {confidence:.3f} < {min_confidence}")
                        print(f"   This is likely why you see 'No LSTM signal'")
                    elif abs(expected_return) < min_expected_return:
                        print(f"   ❌ Would result in 'No LSTM signal' - Low expected return: {abs(expected_return):.3f} < {min_expected_return}")
                        print(f"   This is likely why you see 'No LSTM signal'")
                    else:
                        print(f"   ✅ Trading signal would be generated!")
                        print(f"   Signal: {'BUY' if expected_return > 0 else 'SELL'}")
                        
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                import traceback
                traceback.print_exc()
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Make sure all dependencies are installed")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()