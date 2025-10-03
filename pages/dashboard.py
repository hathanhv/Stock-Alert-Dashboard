"""
Main dashboard page for real-time stock monitoring and signals
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
import os
import sys
import pandas_ta as ta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import DataFetcher
from utils.features import TechnicalIndicators
from utils.preprocess import DataPreprocessor
from utils.signals import SignalGenerator
from config import Config

# Page configuration
st.set_page_config(
    page_title="Stock Alert Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'signal_generator' not in st.session_state:
    st.session_state.signal_generator = SignalGenerator()
if 'technical_indicators' not in st.session_state:
    st.session_state.technical_indicators = TechnicalIndicators(fi_indicator=ta)
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()

def load_historical_data(ticker, days_back=365):
    """Load historical data for a ticker"""
    try:
        # Try to get cached data first
        cached_data = st.session_state.data_fetcher.get_cached_historical_data(ticker)
        if cached_data.empty:
            # Fetch new data if no cache
            cached_data = st.session_state.data_fetcher.fetch_historical_data(ticker, days_back)
        return cached_data
    except Exception as e:
        st.error(f"Error loading historical data for {ticker}: {e}")
        return pd.DataFrame()

def load_financial_data(tickers):
    """Load financial ratios data"""
    try:
        return st.session_state.data_fetcher.fetch_financial_ratios(tickers)
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

def load_realtime_snapshot(ticker):
    """Load latest cached realtime snapshot for a ticker"""
    try:
        return st.session_state.data_fetcher.get_cached_realtime_data(ticker)
    except Exception as _:
        return {}

def create_price_chart(data, technical_data=None):
    """Create interactive price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Technical indicators
    if technical_data is not None:
        # Exponential moving averages
        if 'ema50' in technical_data.columns:
            fig.add_trace(
                go.Scatter(x=technical_data.index, y=technical_data['ema50'], 
                          name='EMA 50', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        if 'ema200' in technical_data.columns:
            fig.add_trace(
                go.Scatter(x=technical_data.index, y=technical_data['ema200'], 
                          name='EMA 200', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'bollinger_hband' in technical_data.columns and 'bollinger_lband' in technical_data.columns:
            fig.add_trace(
                go.Scatter(x=technical_data.index, y=technical_data['bollinger_hband'], 
                          name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            if 'bollinger_mband' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(x=technical_data.index, y=technical_data['bollinger_mband'], 
                              name='BB Middle', line=dict(color='gray', width=1, dash='dot')),
                    row=1, col=1
                )
            fig.add_trace(
                go.Scatter(x=technical_data.index, y=technical_data['bollinger_lband'], 
                          name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
    
    # Volume
    colors = ['green' if close >= open else 'red' 
              for close, open in zip(data['close'], data['open'])]
    
    fig.add_trace(
        go.Bar(x=data.index, y=data['volume'], name='Volume', 
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if technical_data is not None and 'rsi' in technical_data.columns:
        fig.add_trace(
            go.Scatter(x=technical_data.index, y=technical_data['rsi'], 
                      name='RSI', line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        title="Stock Price Chart",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def display_signal_info(ticker, signal_info):
    """Display signal information in a formatted way"""
    signal = signal_info['signal']
    confidence = signal_info['confidence']
    probability = signal_info['probability']
    reason = signal_info['reason']
    
    # Color coding
    if signal == 'BUY':
        color = 'green'
        icon = '🟢'
    elif signal == 'SELL':
        color = 'red'
        icon = '🔴'
    else:
        color = 'orange'
        icon = '🟡'
    
    st.markdown(f"""
    <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0;">
        <h3>{icon} {signal} Signal for {ticker}</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Probability:</strong> {probability:.1%}</p>
        <p><strong>Reason:</strong> {reason}</p>
    </div>
    """, unsafe_allow_html=True)

def display_technical_indicators(data):
    """Display technical indicators in a formatted table"""
    if data.empty:
        return
    
    # Get latest values
    latest = data.iloc[-1]
    
    indicators = {
        'RSI': latest.get('rsi', 'N/A'),
        'MACD': latest.get('macd', 'N/A'),
        'MACD Signal': latest.get('macd_signal', 'N/A'),
        'EMA 50': latest.get('ema50', 'N/A'),
        'EMA 200': latest.get('ema200', 'N/A'),
        'BB Upper': latest.get('bollinger_hband', 'N/A'),
        'BB Middle': latest.get('bollinger_mband', 'N/A'),
        'BB Lower': latest.get('bollinger_lband', 'N/A'),
        'Volume Ratio': latest.get('vol_ratio', 'N/A')
    }
    
    # Format values
    formatted_indicators = {}
    for key, value in indicators.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            if 'Ratio' in key or 'RSI' in key:
                formatted_indicators[key] = f"{value:.2f}"
            else:
                formatted_indicators[key] = f"{value:.4f}"
        else:
            formatted_indicators[key] = 'N/A'
    
    # Create DataFrame for display
    df = pd.DataFrame(list(formatted_indicators.items()), columns=['Indicator', 'Value'])
    
    st.subheader("📊 Technical Indicators")
    st.dataframe(df, use_container_width=True)

def display_fundamental_ratios(financial_data, ticker):
    """Display fundamental ratios"""
    print(financial_data)
    if financial_data is None or len(financial_data) == 0:
        st.warning("No fundamental data available")
        return
    col_name = 'Ticker' if 'Ticker' in financial_data.columns else 'ticker'
    ticker_data = financial_data[financial_data[col_name] == ticker]
    
    if ticker_data.empty:
        st.warning(f"No fundamental data available for {ticker}")
        return
    
    latest = ticker_data.iloc[0]
    
    ratios = {
        'P/E Ratio': latest.get('PE', latest.get('pe_ratio', 'N/A')),
        'P/B Ratio': latest.get('PB', latest.get('pb_ratio', 'N/A')),
        'ROE': latest.get('ROE', latest.get('roe', 'N/A')),
        'EPS': latest.get('EPS', latest.get('eps', 'N/A')),
        'Revenue Growth': latest.get('RevenueGrowth', latest.get('revenue_growth', 'N/A')),
        'Profit Growth': latest.get('ProfitGrowth', latest.get('profit_growth', 'N/A'))
    }
    
    # Format values
    formatted_ratios = {}
    for key, value in ratios.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            if 'Growth' in key:
                formatted_ratios[key] = f"{value:.1%}"
            else:
                formatted_ratios[key] = f"{value:.2f}"
        else:
            formatted_ratios[key] = 'N/A'
    
    # Create DataFrame for display
    df = pd.DataFrame(list(formatted_ratios.items()), columns=['Ratio', 'Value'])
    
    st.subheader("💰 Fundamental Ratios")
    st.dataframe(df, use_container_width=True)

def main():
    """Main dashboard function"""
    st.title("📈 Stock Alert Dashboard")
    st.markdown("Real-time stock monitoring and trading signals")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Stock Ticker", 
        options=Config.TICKERS_WATCHLIST,
        index=0,
        help="Select stock ticker from watchlist"
    )
    
    # Signal thresholds
    st.sidebar.subheader("Signal Thresholds")
    buy_threshold = st.sidebar.slider(
        "BUY Threshold", 
        min_value=0.5, 
        max_value=0.9, 
        value=Config.BUY_THRESHOLD, 
        step=0.05
    )
    sell_threshold = st.sidebar.slider(
        "SELL Threshold", 
        min_value=0.1, 
        max_value=0.5, 
        value=Config.SELL_THRESHOLD, 
        step=0.05
    )
    
    # Update signal generator thresholds
    st.session_state.signal_generator.buy_threshold = buy_threshold
    st.session_state.signal_generator.sell_threshold = sell_threshold
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)
    
    # Main content
    if ticker:
        # Load data
        with st.spinner(f"Loading data for {ticker}..."):
            # Load historical data
            historical_data = load_historical_data(ticker, days_back=365*2)
            
            if historical_data.empty:
                st.error(f"No data available for ticker {ticker}")
                return
            
            # Load financial data
            financial_data = load_financial_data([ticker])
            
            # Calculate technical indicators
            technical_data = st.session_state.technical_indicators.calculate_all_indicators(historical_data)
            
            # Prepare data for prediction
            prediction_data = st.session_state.preprocessor.prepare_prediction_data(
                historical_data, financial_data, ticker
            )
            
            if prediction_data.empty:
                st.warning("Unable to prepare data for prediction")
                return
        
        # Try realtime snapshot
        rt = load_realtime_snapshot(ticker)
        has_rt = bool(rt) and 'price' in rt and rt['price'] is not None
        # Display current price
        if has_rt:
            current_price = rt['price']
            last_close = historical_data['close'].iloc[-1]
            price_change = (current_price - last_close) / last_close if last_close else 0.0
        else:
            current_price = historical_data['close'].iloc[-1]
            price_change = historical_data['close'].pct_change().iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"{current_price:,.0f} VND",
                delta=f"{price_change:.2%}"
            )
        
        with col2:
            volume = rt.get('volume') if has_rt else historical_data['volume'].iloc[-1]
            st.metric(
                label="Volume",
                value=f"{volume:,.0f}"
            )
        
        with col3:
            # Generate signal
            probability = st.session_state.signal_generator.predict_probability(prediction_data)
            technical_indicators = technical_data.iloc[-1].to_dict()
            
            signal_info = st.session_state.signal_generator.generate_signal(
                probability, current_price, technical_indicators
            )
            
            # Log signal
            st.session_state.signal_generator.log_signal(ticker, signal_info)
            
            st.metric(
                label="Signal",
                value=signal_info['signal'],
                delta=f"Confidence: {signal_info['confidence']:.1%}"
            )
        
        # Display signal information
        display_signal_info(ticker, signal_info)
        
        # Charts
        st.subheader("📊 Price Chart")
        fig = create_price_chart(historical_data, technical_data)
        # Overlay realtime point if available
        if has_rt:
            fig.add_trace(
                go.Scatter(x=[pd.Timestamp.now()], y=[rt['price']], name='Realtime',
                           mode='markers', marker=dict(color='red', size=8)),
                row=1, col=1
            )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical and fundamental data
        col1, col2 = st.columns(2)
        
        with col1:
            display_technical_indicators(technical_data)
        
        with col2:
            display_fundamental_ratios(financial_data, ticker)
        
        # Recent signals
        st.subheader("📋 Recent Signals")
        recent_signals = st.session_state.signal_generator.get_all_signals(limit=10)
        
        if not recent_signals.empty:
            # Filter for current ticker
            ticker_signals = recent_signals[recent_signals['ticker'] == ticker]
            
            if not ticker_signals.empty:
                # Display recent signals for this ticker
                display_columns = ['timestamp', 'signal', 'probability', 'confidence', 'reason']
                st.dataframe(
                    ticker_signals[display_columns].head(5),
                    use_container_width=True
                )
            else:
                st.info(f"No recent signals found for {ticker}")
        else:
            st.info("No signals logged yet")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
