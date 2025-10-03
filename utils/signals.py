"""
Signal generation utilities for BUY/SELL/HOLD signals
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import csv
from typing import Dict, List, Tuple, Optional
import joblib
from config import Config

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Class to generate trading signals based on model predictions"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the signal generator
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.buy_threshold = Config.BUY_THRESHOLD
        self.sell_threshold = Config.SELL_THRESHOLD
        
        # Load model if exists
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                # Persist expected feature list if available
                try:
                    if hasattr(self.model, 'feature_names_in_'):
                        features_path = os.path.join(os.path.dirname(self.model_path), 'feature_columns.json')
                        if not os.path.exists(features_path):
                            import json
                            with open(features_path, 'w', encoding='utf-8') as f:
                                json.dump(list(self.model.feature_names_in_), f, ensure_ascii=False)
                except Exception as _:
                    pass
            else:
                logger.warning(f"Model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def predict_probability(self, features: pd.DataFrame) -> float:
        """
        Predict probability of price increase
        
        Args:
            features: Feature DataFrame for prediction
            
        Returns:
            Probability of price increase (0-1)
        """
        try:
            if self.model is None:
                logger.warning("No model loaded, returning default probability")
                return 0.5
            
            # Ensure features match model expectations
            if hasattr(self.model, 'feature_names_in_'):
                # Harmonize naming differences with legacy checkpoint
                try:
                    legacy_map = {
                        'bb_upper': 'bollinger_hband',
                        'bb_lower': 'bollinger_lband',
                        'bb_middle': 'bollinger_mband',
                        'volume_ratio': 'vol_ratio',
                        'sma_20': 'ema50',  # best-effort alias if model used SMA
                        'sma_50': 'ema200'  # best-effort alias if model used SMA
                    }
                    for legacy_name, current_name in legacy_map.items():
                        if legacy_name in getattr(self.model, 'feature_names_in_', []) and legacy_name not in features.columns:
                            if current_name in features.columns:
                                features[legacy_name] = features[current_name]
                except Exception as _:
                    pass

                # Select only the features the model was trained on, preserving order
                expected = list(self.model.feature_names_in_)
                # Build exact-length array and fill
                missing = [f for f in expected if f not in features.columns]
                extra = [f for f in features.columns if f not in expected]
                if missing or extra:
                    logger.warning(f"Feature mismatch: expected {len(expected)}, available {len(features.columns)}")
                    logger.warning(f"Missing features ({len(missing)}): {missing}")
                    logger.warning(f"Extra features ({len(extra)}): {extra}")
                    # Persist debug info next to model
                    try:
                        import json
                        debug_dir = os.path.dirname(self.model_path)
                        debug_json = {
                            "expected": expected,
                            "available": list(features.columns),
                            "missing": missing,
                            "extra": extra
                        }
                        with open(os.path.join(debug_dir, 'last_prediction_features_info.json'), 'w', encoding='utf-8') as f:
                            json.dump(debug_json, f, ensure_ascii=False, indent=2)
                        # Dump last row values for available features
                        if len(features) > 0:
                            features.tail(1).to_csv(os.path.join(debug_dir, 'last_prediction_row.csv'), index=False)
                    except Exception as _:
                        pass
                n_rows = len(features)
                import numpy as _np
                X = _np.zeros((n_rows, len(expected)), dtype=float)
                for idx, fname in enumerate(expected):
                    if fname in features.columns:
                        X[:, idx] = features[fname].astype(float).to_numpy()
                if X.shape[1] != len(expected):
                    logger.error("Constructed feature array has wrong shape; returning 0.5.")
                    return 0.5
            else:
                # Try to load expected features from persisted file
                try:
                    import json, os
                    features_path = os.path.join(os.path.dirname(self.model_path), 'feature_columns.json')
                    if os.path.exists(features_path):
                        expected = json.load(open(features_path, 'r', encoding='utf-8'))
                        n_rows = len(features)
                        import numpy as _np
                        X = _np.zeros((n_rows, len(expected)), dtype=float)
                        missing = []
                        extra = [f for f in features.columns if f not in expected]
                        for idx, fname in enumerate(expected):
                            if fname in features.columns:
                                X[:, idx] = features[fname].astype(float).to_numpy()
                            else:
                                missing.append(fname)
                        if missing or extra:
                            logger.warning(f"Feature mismatch vs JSON: expected {len(expected)}, available {len(features.columns)}")
                            logger.warning(f"Missing features ({len(missing)}): {missing}")
                            logger.warning(f"Extra features ({len(extra)}): {extra}")
                            # Persist debug info next to model
                            try:
                                import json
                                debug_dir = os.path.dirname(self.model_path)
                                debug_json = {
                                    "expected": expected,
                                    "available": list(features.columns),
                                    "missing": missing,
                                    "extra": extra
                                }
                                with open(os.path.join(debug_dir, 'last_prediction_features_info.json'), 'w', encoding='utf-8') as f:
                                    json.dump(debug_json, f, ensure_ascii=False, indent=2)
                                if len(features) > 0:
                                    features.tail(1).to_csv(os.path.join(debug_dir, 'last_prediction_row.csv'), index=False)
                            except Exception as _:
                                pass
                        
                except Exception as _:
                    X = features.astype(float).to_numpy()

            # Predict probability
            # Ensure X has the exact number of features the model expects
            try:
                expected_n = getattr(self.model, 'n_features_in_', X.shape[1])
                if X.shape[1] != expected_n:
                    import numpy as _np
                    old_n = X.shape[1]
                    if old_n < expected_n:
                        logger.warning(f"Adjusting features: padding from {old_n} to {expected_n} with zeros")
                        pad = _np.zeros((X.shape[0], expected_n - old_n), dtype=X.dtype)
                        X = _np.concatenate([X, pad], axis=1)
                    else:
                        logger.warning(f"Adjusting features: truncating from {old_n} to {expected_n}")
                        X = X[:, :expected_n]
            except Exception as _:
                pass

            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X)[0][1]  # Probability of class 1
            else:
                # For models without predict_proba, use predict and convert to probability
                prediction = self.model.predict(X)[0]
                prob = float(prediction)
            
            return prob
            
        except Exception as e:
            logger.error(f"Error predicting probability: {e}")
            return 0.5
    
    def generate_signal(self, probability: float, current_price: float = None, 
                       technical_indicators: Dict = None) -> Dict:
        """
        Generate BUY/SELL/HOLD signal based on probability and technical indicators
        
        Args:
            probability: Predicted probability of price increase
            current_price: Current stock price
            technical_indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with signal information
        """
        try:
            signal_info = {
                'timestamp': datetime.now(),
                'probability': probability,
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': '',
                'price': current_price
            }
            
            # Basic signal generation based on probability
            if probability >= self.buy_threshold:
                signal_info['signal'] = 'BUY'
                signal_info['confidence'] = (probability - self.buy_threshold) / (1 - self.buy_threshold)
                signal_info['reason'] = f'High probability of price increase: {probability:.3f}'
            elif probability <= self.sell_threshold:
                signal_info['signal'] = 'SELL'
                signal_info['confidence'] = (self.sell_threshold - probability) / self.sell_threshold
                signal_info['reason'] = f'Low probability of price increase: {probability:.3f}'
            else:
                signal_info['signal'] = 'HOLD'
                signal_info['confidence'] = 1 - abs(probability - 0.5) * 2
                signal_info['reason'] = f'Moderate probability: {probability:.3f}'
            
            # Apply technical analysis filters if available
            if technical_indicators:
                signal_info = self.apply_technical_filters(signal_info, technical_indicators)
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                'timestamp': datetime.now(),
                'probability': 0.5,
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'price': current_price
            }
    
    def apply_technical_filters(self, signal_info: Dict, indicators: Dict) -> Dict:
        """
        Apply technical analysis filters to refine signals
        
        Args:
            signal_info: Current signal information
            indicators: Dictionary of technical indicators
            
        Returns:
            Updated signal information
        """
        try:
            # RSI filter
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if signal_info['signal'] == 'BUY' and rsi > 70:
                    # Overbought condition, downgrade signal
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (RSI overbought)'
                    signal_info['confidence'] *= 0.5
                elif signal_info['signal'] == 'SELL' and rsi < 30:
                    # Oversold condition, downgrade signal
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (RSI oversold)'
                    signal_info['confidence'] *= 0.5
            
            # MACD filter
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                
                if signal_info['signal'] == 'BUY' and macd < macd_signal:
                    # MACD bearish crossover
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (MACD bearish)'
                    signal_info['confidence'] *= 0.7
                elif signal_info['signal'] == 'SELL' and macd > macd_signal:
                    # MACD bullish crossover
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (MACD bullish)'
                    signal_info['confidence'] *= 0.7
            
            # Bollinger Bands filter
            if all(key in indicators for key in ['bb_upper', 'bb_lower', 'close']):
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                close = indicators['close']
                
                if signal_info['signal'] == 'BUY' and close > bb_upper:
                    # Price above upper band
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (Price above BB upper)'
                    signal_info['confidence'] *= 0.6
                elif signal_info['signal'] == 'SELL' and close < bb_lower:
                    # Price below lower band
                    signal_info['signal'] = 'HOLD'
                    signal_info['reason'] += ' (Price below BB lower)'
                    signal_info['confidence'] *= 0.6
            
            # Volume filter
            if 'volume_ratio' in indicators:
                volume_ratio = indicators['volume_ratio']
                if volume_ratio < 0.5:  # Low volume
                    signal_info['confidence'] *= 0.8
                    signal_info['reason'] += ' (Low volume)'
            
            return signal_info
            
        except Exception as e:
            logger.error(f"Error applying technical filters: {e}")
            return signal_info
    
    def log_signal(self, ticker: str, signal_info: Dict):
        """
        Log signal to file and console
        
        Args:
            ticker: Stock ticker symbol
            signal_info: Signal information dictionary
        """
        try:
            # Console logging
            logger.info(f"{ticker}: {signal_info['signal']} - {signal_info['reason']} "
                       f"(Confidence: {signal_info['confidence']:.3f})")
            
            # File logging
            log_entry = {
                'timestamp': signal_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'signal': signal_info['signal'],
                'probability': signal_info['probability'],
                'confidence': signal_info['confidence'],
                'reason': signal_info['reason'],
                'price': signal_info.get('price', 'N/A')
            }
            
            # Write to CSV
            csv_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_entry)
            
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    def get_signal_summary(self, ticker: str, days: int = 7) -> Dict:
        """
        Get signal summary for a ticker over specified days
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with signal summary
        """
        try:
            csv_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
            
            if not os.path.exists(csv_file):
                return {'error': 'No signal log found'}
            
            # Read signal log
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for ticker and recent days
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            ticker_signals = df[(df['ticker'] == ticker) & (df['timestamp'] >= cutoff_date)]
            
            if ticker_signals.empty:
                return {'error': f'No signals found for {ticker} in last {days} days'}
            
            # Calculate summary
            signal_counts = ticker_signals['signal'].value_counts()
            avg_confidence = ticker_signals['confidence'].mean()
            latest_signal = ticker_signals.iloc[-1]
            
            summary = {
                'ticker': ticker,
                'period_days': days,
                'total_signals': len(ticker_signals),
                'signal_counts': signal_counts.to_dict(),
                'average_confidence': avg_confidence,
                'latest_signal': {
                    'timestamp': latest_signal['timestamp'],
                    'signal': latest_signal['signal'],
                    'probability': latest_signal['probability'],
                    'confidence': latest_signal['confidence'],
                    'reason': latest_signal['reason']
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {'error': str(e)}
    
    def get_all_signals(self, limit: int = 100) -> pd.DataFrame:
        """
        Get all recent signals
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            DataFrame with recent signals
        """
        try:
            csv_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
            
            if not os.path.exists(csv_file):
                return pd.DataFrame()
            
            # Read signal log
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp and get recent signals
            df = df.sort_values('timestamp', ascending=False).head(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting all signals: {e}")
            return pd.DataFrame()
    
    def clear_old_signals(self, days_to_keep: int = 30):
        """
        Clear old signals from log file
        
        Args:
            days_to_keep: Number of days of signals to keep
        """
        try:
            csv_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
            
            if not os.path.exists(csv_file):
                return
            
            # Read signal log
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter recent signals
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            recent_signals = df[df['timestamp'] >= cutoff_date]
            
            # Write back to file
            recent_signals.to_csv(csv_file, index=False)
            
            logger.info(f"Cleared old signals, kept {len(recent_signals)} recent signals")
            
        except Exception as e:
            logger.error(f"Error clearing old signals: {e}")
