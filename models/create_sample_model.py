"""
Script to create a sample model for demonstration purposes
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.fetch_data import DataFetcher
from utils.features import TechnicalIndicators
from utils.preprocess import DataPreprocessor

def create_sample_model():
    """Create a sample model using the new data structure"""
    
    try:
        print("🚀 Creating sample model with new data structure...")
        
        # Initialize components
        data_fetcher = DataFetcher()
        preprocessor = DataPreprocessor()
        
        # Create sample data with new structure
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        tickers = ['VCB'] * n_samples
        
        # Generate realistic price data
        base_price = 100
        price_changes = np.random.normal(0, 0.02, n_samples)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 10))  # Minimum price of 10
        
        sample_data = pd.DataFrame({
            'Ticker': tickers,
            'TradingDate': dates,
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'MatchVolume': np.random.uniform(1000000, 10000000, n_samples),
            'PB': np.random.uniform(0.5, 3.0, n_samples),
            'PE': np.random.uniform(5, 30, n_samples),
            'ROE': np.random.uniform(0.05, 0.25, n_samples),
            'EPS': np.random.uniform(1000, 5000, n_samples),
            'RevenueGrowth': np.random.uniform(-0.1, 0.3, n_samples),
            'ProfitGrowth': np.random.uniform(-0.2, 0.4, n_samples)
        })
        
        # Initialize technical indicators calculator
        technical_calc = TechnicalIndicators()
        
        # Calculate all indicators
        print("📊 Calculating technical indicators...")
        processed_data = technical_calc.calculate_all_indicators(sample_data)
        
        # Prepare features for model
        print("🔧 Preparing features for model...")
        X, y = technical_calc.prepare_features_for_model(processed_data, target_period=10)
        
        if X.empty or y.empty:
            print("❌ No valid data for training")
            return None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"✅ Training accuracy: {train_score:.3f}")
        print(f"✅ Test accuracy: {test_score:.3f}")
        
        # Save model and components
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
        
        joblib.dump(model, Config.MODEL_PATH)
        joblib.dump(scaler, os.path.join(os.path.dirname(Config.MODEL_PATH), 'scaler.pkl'))
        joblib.dump(X.columns.tolist(), os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_names.pkl'))
        
        print(f"💾 Model saved to {Config.MODEL_PATH}")
        print(f"💾 Scaler saved to scaler.pkl")
        print(f"💾 Feature names saved to feature_names.pkl")
        
        return model, scaler, X.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error creating sample model: {e}")
        return None, None, None

def create_simple_fallback_model():
    """Create a simple fallback model if main method fails"""
    
    print("🔄 Creating simple fallback model...")
    
    # Create simple sample data
    np.random.seed(42)
    n_samples = 500
    
    # Define feature names based on new structure
    feature_names = [
        'rsi', 'mfi', 'macd', 'macd_signal', 'macd_diff',
        'bollinger_hband', 'bollinger_lband', 'bb_pos',
        'ema50', 'ema200', 'vol_ratio',
        'price_vs_ema50', 'price_vs_ema200', 'ema_ratio',
        'macd_signal_strength', 'macd_trend', 'trend_score',
        'rsi_overbought', 'rsi_oversold', 'mfi_momentum',
        'volume_momentum', 'momentum_score',
        'PB', 'PE', 'ROE', 'EPS', 'RevenueGrowth', 'ProfitGrowth',
        'pe_normalized', 'pb_normalized', 'growth_score',
        'roe_quality', 'value_score'
    ]
    
    # Generate realistic feature values
    features = np.random.randn(n_samples, len(feature_names))
    
    # Make features more realistic
    features[:, 0] = np.random.uniform(20, 80, n_samples)  # RSI
    features[:, 2] = np.random.uniform(-0.5, 0.5, n_samples)  # MACD
    features[:, 7] = np.random.uniform(0, 1, n_samples)  # BB position
    features[:, 10] = np.random.uniform(0.5, 2.0, n_samples)  # Volume ratio
    
    # Create target with some logic
    target = np.zeros(n_samples)
    for i in range(n_samples):
        prob = 0.5
        if features[i, 0] < 70:  # RSI not overbought
            prob += 0.1
        if features[i, 2] > 0:  # MACD bullish
            prob += 0.1
        if features[i, 10] > 1:  # Volume above average
            prob += 0.1
        
        prob += np.random.normal(0, 0.1)
        prob = max(0, min(1, prob))
        target[i] = 1 if prob > 0.6 else 0
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    df['target'] = target
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Training accuracy: {train_score:.3f}")
    print(f"✅ Test accuracy: {test_score:.3f}")
    
    # Save model and components
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    
    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(scaler, os.path.join(os.path.dirname(Config.MODEL_PATH), 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_names.pkl'))
    
    print(f"💾 Fallback model saved to {Config.MODEL_PATH}")
    
    return model, scaler, feature_names

if __name__ == "__main__":
    # Try to create model with new structure first
    model, scaler, features = create_sample_model()
    
    # If that fails, create simple fallback model
    if model is None:
        print("\n⚠️ Main model creation failed, trying fallback method...")
        model, scaler, features = create_simple_fallback_model()
    
    if model is not None:
        print("\n🎉 Sample model created successfully!")
        print(f"📊 Features: {len(features)}")
        print("✅ Ready to use with the dashboard")
    else:
        print("\n❌ Failed to create any model")
