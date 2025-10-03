"""
Settings page for configuration and model management
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import DataFetcher
from utils.features import TechnicalIndicators
from utils.preprocess import DataPreprocessor
from config import Config

# Page configuration
st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'technical_indicators' not in st.session_state:
    st.session_state.technical_indicators = TechnicalIndicators()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()

def save_config_to_file(config_dict, filename):
    """Save configuration to JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', filename)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False

def load_config_from_file(filename):
    """Load configuration from JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', filename)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

def train_new_model(tickers, days_back=365):
    """Train a new model with historical data"""
    try:
        st.info("Starting model training...")
        
        all_data = []
        all_features = []
        all_targets = []
        
        # Collect data from multiple tickers
        for ticker in tickers:
            with st.spinner(f"Processing {ticker}..."):
                # Load historical data
                historical_data = st.session_state.data_fetcher.get_cached_historical_data(ticker)
                if historical_data.empty:
                    historical_data = st.session_state.data_fetcher.fetch_historical_data(ticker, days_back)
                
                if historical_data.empty:
                    st.warning(f"No data available for {ticker}")
                    continue
                
                # Load financial data
                financial_data = st.session_state.data_fetcher.fetch_financial_ratios([ticker])
                
                # Prepare training data
                X, y = st.session_state.preprocessor.prepare_training_data(
                    historical_data, financial_data, ticker
                )
                
                if not X.empty and not y.empty:
                    all_features.append(X)
                    all_targets.append(y)
                    all_data.append((ticker, len(X)))
        
        if not all_features:
            st.error("No training data available")
            return None
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        st.info(f"Combined dataset: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        # Handle missing values
        combined_features = combined_features.fillna(combined_features.median())
        combined_targets = combined_targets.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, combined_targets, test_size=0.2, random_state=42, stratify=combined_targets
        )
        
        # Scale features
        X_train_scaled = st.session_state.preprocessor.scale_features(X_train, fit_scaler=True)
        X_test_scaled = st.session_state.preprocessor.scale_features(X_test, fit_scaler=False)
        
        # Train model
        with st.spinner("Training Random Forest model..."):
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        model_path = Config.MODEL_PATH
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Save feature names
        feature_names_path = os.path.join(os.path.dirname(model_path), 'feature_names.pkl')
        joblib.dump(combined_features.columns.tolist(), feature_names_path)
        
        st.success("Model training completed successfully!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Training Samples", len(X_train))
            st.metric("Test Samples", len(X_test))
        
        with col2:
            st.metric("Features", len(combined_features.columns))
            st.metric("Tickers Used", len(all_data))
        
        # Display classification report
        st.subheader("📊 Model Performance")
        
        if isinstance(report, dict):
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # Display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        st.subheader("📈 Confusion Matrix")
        st.dataframe(cm_df)
        
        return model
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def display_current_config():
    """Display current configuration"""
    st.subheader("📋 Current Configuration")
    
    config_data = {
        'Parameter': [
            'BUY Threshold',
            'SELL Threshold',
            'RSI Period',
            'EMA Periods',
            'SMA Periods',
            'Bollinger Period',
            'Bollinger Std',
            'Prediction Days',
            'Initial Capital',
            'Commission Rate',
            'Refresh Interval',
            'Max Tickers'
        ],
        'Value': [
            Config.BUY_THRESHOLD,
            Config.SELL_THRESHOLD,
            Config.RSI_PERIOD,
            Config.EMA_PERIODS,
            Config.SMA_PERIODS,
            Config.BOLLINGER_PERIOD,
            Config.BOLLINGER_STD,
            Config.PREDICTION_DAYS,
            f"{Config.INITIAL_CAPITAL:,} VND",
            f"{Config.COMMISSION_RATE:.3f}",
            f"{Config.REFRESH_INTERVAL} seconds",
            Config.MAX_TICKERS
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)

def display_model_info():
    """Display current model information"""
    st.subheader("🤖 Model Information")
    
    model_path = Config.MODEL_PATH
    
    if os.path.exists(model_path):
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Model details
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", type(model).__name__)
                st.metric("Model File", os.path.basename(model_path))
                
                # File size
                file_size = os.path.getsize(model_path) / 1024  # KB
                st.metric("File Size", f"{file_size:.1f} KB")
            
            with col2:
                # Last modified
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                st.metric("Last Modified", mod_time.strftime("%Y-%m-%d %H:%M:%S"))
                
                # Model attributes
                if hasattr(model, 'n_estimators'):
                    st.metric("Estimators", model.n_estimators)
                if hasattr(model, 'max_depth'):
                    st.metric("Max Depth", model.max_depth)
            
            # Feature names
            feature_names_path = os.path.join(os.path.dirname(model_path), 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                feature_names = joblib.load(feature_names_path)
                st.metric("Features", len(feature_names))
                
                # Display feature list
                if st.checkbox("Show Feature List"):
                    feature_df = pd.DataFrame({'Feature': feature_names})
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("No model file found. Please train a new model.")

def display_data_status():
    """Display data storage status"""
    st.subheader("💾 Data Storage Status")
    
    # Check directories
    directories = {
        'Historical Data': Config.HISTORICAL_DATA_DIR,
        'Realtime Data': Config.REALTIME_DATA_DIR,
        'Models': os.path.dirname(Config.MODEL_PATH),
        'Main Data': Config.DATA_DIR
    }
    
    status_data = []
    
    for name, path in directories.items():
        if os.path.exists(path):
            # Count files
            try:
                files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                file_count = len(files)
                
                # Calculate total size
                total_size = 0
                for file in files:
                    file_path = os.path.join(path, file)
                    total_size += os.path.getsize(file_path)
                
                size_mb = total_size / (1024 * 1024)
                
                status_data.append({
                    'Directory': name,
                    'Path': path,
                    'Files': file_count,
                    'Size (MB)': f"{size_mb:.2f}",
                    'Status': '✅ Available'
                })
            except Exception as e:
                status_data.append({
                    'Directory': name,
                    'Path': path,
                    'Files': 'Error',
                    'Size (MB)': 'Error',
                    'Status': f'❌ Error: {e}'
                })
        else:
            status_data.append({
                'Directory': name,
                'Path': path,
                'Files': 'N/A',
                'Size (MB)': 'N/A',
                'Status': '❌ Not Found'
            })
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

def main():
    """Main settings page function"""
    st.title("⚙️ Settings & Configuration")
    st.markdown("Configure system parameters and manage models")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Model Management", "Data Management", "System Status"])
    
    with tab1:
        st.header("📋 Configuration Settings")
        
        display_current_config()
        
        # Configuration editor
        st.subheader("✏️ Edit Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Signal Thresholds**")
            buy_threshold = st.slider(
                "BUY Threshold",
                min_value=0.5,
                max_value=0.9,
                value=Config.BUY_THRESHOLD,
                step=0.05
            )
            
            sell_threshold = st.slider(
                "SELL Threshold",
                min_value=0.1,
                max_value=0.5,
                value=Config.SELL_THRESHOLD,
                step=0.05
            )
        
        with col2:
            st.write("**Technical Indicators**")
            rsi_period = st.number_input(
                "RSI Period",
                min_value=5,
                max_value=50,
                value=Config.RSI_PERIOD
            )
            
            bollinger_period = st.number_input(
                "Bollinger Period",
                min_value=10,
                max_value=50,
                value=Config.BOLLINGER_PERIOD
            )
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            config_dict = {
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'rsi_period': rsi_period,
                'bollinger_period': bollinger_period,
                'last_updated': datetime.now().isoformat()
            }
            
            if save_config_to_file(config_dict, 'user_config.json'):
                st.success("Configuration saved successfully!")
    
    with tab2:
        st.header("🤖 Model Management")
        
        display_model_info()
        
        # Model training section
        st.subheader("🔄 Train New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ticker selection for training
            tickers_input = st.text_area(
                "Training Tickers (one per line)",
                value="VCB\nVIC\nHPG\nCTG\nBID",
                help="Enter stock tickers to use for training, one per line"
            )
            
            tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        
        with col2:
            days_back = st.number_input(
                "Days of Historical Data",
                min_value=100,
                max_value=1000,
                value=365,
                step=50
            )
            
            if st.button("🚀 Train New Model", type="primary"):
                if tickers:
                    train_new_model(tickers, days_back)
                else:
                    st.error("Please enter at least one ticker symbol")
        
        # Model backup/restore
        st.subheader("💾 Model Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Backup Current Model"):
                model_path = Config.MODEL_PATH
                if os.path.exists(model_path):
                    backup_path = f"{model_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    try:
                        import shutil
                        shutil.copy2(model_path, backup_path)
                        st.success(f"Model backed up to {backup_path}")
                    except Exception as e:
                        st.error(f"Error creating backup: {e}")
                else:
                    st.error("No model file found to backup")
        
        with col2:
            st.file_uploader(
                "📤 Restore Model",
                type=['pkl'],
                help="Upload a model file to restore"
            )
    
    with tab3:
        st.header("💾 Data Management")
        
        display_data_status()
        
        # Data operations
        st.subheader("🔧 Data Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Historical Data"):
                try:
                    historical_dir = Config.HISTORICAL_DATA_DIR
                    if os.path.exists(historical_dir):
                        files = [f for f in os.listdir(historical_dir) if f.endswith('.csv')]
                        for file in files:
                            os.remove(os.path.join(historical_dir, file))
                        st.success(f"Cleared {len(files)} historical data files")
                    else:
                        st.warning("Historical data directory not found")
                except Exception as e:
                    st.error(f"Error clearing historical data: {e}")
        
        with col2:
            if st.button("🗑️ Clear Realtime Data"):
                try:
                    realtime_dir = Config.REALTIME_DATA_DIR
                    if os.path.exists(realtime_dir):
                        files = [f for f in os.listdir(realtime_dir) if f.endswith('.json')]
                        for file in files:
                            os.remove(os.path.join(realtime_dir, file))
                        st.success(f"Cleared {len(files)} realtime data files")
                    else:
                        st.warning("Realtime data directory not found")
                except Exception as e:
                    st.error(f"Error clearing realtime data: {e}")
        
        with col3:
            if st.button("🗑️ Clear Signal Logs"):
                try:
                    signals_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
                    if os.path.exists(signals_file):
                        os.remove(signals_file)
                        st.success("Cleared signal logs")
                    else:
                        st.warning("No signal log file found")
                except Exception as e:
                    st.error(f"Error clearing signal logs: {e}")
        
        # Data refresh
        st.subheader("🔄 Refresh Data")
        
        refresh_ticker = st.text_input("Ticker to refresh", value="VCB").upper()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"📥 Refresh {refresh_ticker} Historical Data"):
                try:
                    data = st.session_state.data_fetcher.fetch_historical_data(refresh_ticker, 365)
                    if not data.empty:
                        st.success(f"Refreshed historical data for {refresh_ticker}")
                    else:
                        st.error(f"No data retrieved for {refresh_ticker}")
                except Exception as e:
                    st.error(f"Error refreshing data: {e}")
        
        with col2:
            if st.button(f"📊 Refresh {refresh_ticker} Financial Data"):
                try:
                    data = st.session_state.data_fetcher.fetch_financial_ratios([refresh_ticker])
                    if not data.empty:
                        st.success(f"Refreshed financial data for {refresh_ticker}")
                    else:
                        st.error(f"No financial data retrieved for {refresh_ticker}")
                except Exception as e:
                    st.error(f"Error refreshing financial data: {e}")
        
        # Ticker management
        st.subheader("📋 Ticker Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Watchlist**")
            for i, ticker in enumerate(Config.TICKERS_WATCHLIST, 1):
                st.write(f"{i}. {ticker}")
        
        with col2:
            st.write("**Get All Tickers**")
            if st.button("📥 Fetch All Tickers from Exchanges"):
                try:
                    all_tickers = st.session_state.data_fetcher.get_all_tickers()
                    if all_tickers:
                        st.success(f"Found {len(all_tickers)} tickers from all exchanges")
                        st.write("Sample tickers:", all_tickers[:10])
                    else:
                        st.error("No tickers found")
                except Exception as e:
                    st.error(f"Error fetching tickers: {e}")
            
            if st.button("🏢 Get Tickers by Exchange"):
                try:
                    tickers_by_exchange = st.session_state.data_fetcher.get_tickers_by_exchange()
                    if tickers_by_exchange:
                        for exchange, tickers in tickers_by_exchange.items():
                            st.write(f"**{exchange}**: {len(tickers)} tickers")
                            if tickers:
                                st.write(f"Sample: {tickers[:5]}")
                    else:
                        st.error("No tickers found")
                except Exception as e:
                    st.error(f"Error fetching tickers by exchange: {e}")
        
        # Debug section
        st.subheader("🔍 Debug & Troubleshooting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Status Check**")
            if st.button("📊 Check Market Status"):
                try:
                    market_status = st.session_state.data_fetcher.check_market_status()
                    if "error" not in market_status:
                        st.success(f"Market Status: {market_status['status']}")
                        st.write(f"Current Time: {market_status['current_time']}")
                        st.write(f"Day: {market_status['weekday']}")
                        st.write(f"Is Weekend: {market_status['is_weekend']}")
                        st.write(f"Is Trading Time: {market_status['is_trading_time']}")
                        st.write(f"Trading Hours: {market_status['trading_hours']}")
                    else:
                        st.error(f"Error: {market_status['error']}")
                except Exception as e:
                    st.error(f"Error checking market status: {e}")
        
        with col2:
            st.write("**Debug Ticker Data**")
            debug_ticker = st.selectbox(
                "Select ticker to debug",
                options=Config.TICKERS_WATCHLIST,
                key="debug_ticker"
            )
            
            if st.button("🔍 Debug Ticker Data"):
                try:
                    debug_info = st.session_state.data_fetcher.debug_ticker_data(debug_ticker)
                    
                    if "error" not in debug_info:
                        st.write(f"**Ticker**: {debug_info['ticker']}")
                        st.write(f"**Client Status**: {debug_info['client_status']}")
                        st.write(f"**Data Status**: {debug_info.get('data_status', 'Unknown')}")
                        
                        if 'data_shape' in debug_info:
                            st.write(f"**Data Shape**: {debug_info['data_shape']}")
                        if 'data_columns' in debug_info:
                            st.write(f"**Columns**: {debug_info['data_columns']}")
                        if 'fetch_error' in debug_info:
                            st.error(f"**Fetch Error**: {debug_info['fetch_error']}")
                        
                        # Market status details
                        market_status = debug_info.get('market_status', {})
                        if market_status:
                            st.write(f"**Market**: {market_status.get('status', 'Unknown')}")
                            st.write(f"**Time**: {market_status.get('current_time', 'Unknown')}")
                    else:
                        st.error(f"Debug Error: {debug_info['error']}")
                        
                except Exception as e:
                    st.error(f"Error debugging ticker: {e}")
    
    with tab4:
        st.header("📊 System Status")
        
        # System information
        st.subheader("💻 System Information")
        
        import platform
        import psutil
        
        sys_info = {
            'Platform': platform.platform(),
            'Python Version': platform.python_version(),
            'CPU Count': psutil.cpu_count(),
            'Memory Total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'Memory Available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            'Disk Usage': f"{psutil.disk_usage('/').percent:.1f}%"
        }
        
        sys_df = pd.DataFrame(list(sys_info.items()), columns=['Metric', 'Value'])
        st.dataframe(sys_df, use_container_width=True, hide_index=True)
        
        # Application status
        st.subheader("📱 Application Status")
        
        app_status = {
            'Data Fetcher': '✅ Initialized' if 'data_fetcher' in st.session_state else '❌ Not Initialized',
            'Signal Generator': '✅ Initialized' if 'signal_generator' in st.session_state else '❌ Not Initialized',
            'Technical Indicators': '✅ Initialized' if 'technical_indicators' in st.session_state else '❌ Not Initialized',
            'Preprocessor': '✅ Initialized' if 'preprocessor' in st.session_state else '❌ Not Initialized',
            'Model Available': '✅ Available' if os.path.exists(Config.MODEL_PATH) else '❌ Not Available'
        }
        
        status_df = pd.DataFrame(list(app_status.items()), columns=['Component', 'Status'])
        st.dataframe(status_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
