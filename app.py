"""
Main Streamlit application for Stock Alert Dashboard
"""
import streamlit as st
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

# Page configuration
st.set_page_config(
    page_title="Stock Alert Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #28a745;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background-color: #dc3545;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_system_status():
    """Check system status and dependencies"""
    status = {
        'config_valid': False,
        'model_available': False,
        'directories_exist': False,
        'credentials_set': False
    }
    
    try:
        # Check configuration
        Config.validate_config()
        status['config_valid'] = True
        
        # Check model
        if os.path.exists(Config.MODEL_PATH):
            status['model_available'] = True
        
        # Check directories
        if (os.path.exists(Config.HISTORICAL_DATA_DIR) and 
            os.path.exists(Config.REALTIME_DATA_DIR)):
            status['directories_exist'] = True
        
        # Check credentials
        if Config.USERNAME1 and Config.PASSWORD1:
            status['credentials_set'] = True
            
    except Exception as e:
        st.error(f"Configuration error: {e}")
    
    return status

def display_welcome_page():
    """Display welcome page with system overview"""
    st.markdown('<h1 class="main-header">📈 Stock Alert Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Welcome to Stock Alert Dashboard</h3>
        <p>This application provides real-time stock monitoring, trading signals, and backtesting capabilities 
        using machine learning models trained on historical data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    st.subheader("📊 System Status")
    
    status = check_system_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Configuration Status**")
        
        config_items = [
            ("Configuration Valid", status['config_valid']),
            ("Model Available", status['model_available']),
            ("Directories Created", status['directories_exist']),
            ("Credentials Set", status['credentials_set'])
        ]
        
        for item, is_ok in config_items:
            status_class = "status-online" if is_ok else "status-offline"
            st.markdown(f'<span class="status-indicator {status_class}"></span>{item}', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Quick Actions**")
        
        if st.button("🏠 Go to Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard.py")
        
        if st.button("📊 Run Backtest", use_container_width=True):
            st.switch_page("pages/backtest_page.py")
        
        if st.button("⚙️ Open Settings", use_container_width=True):
            st.switch_page("pages/settings.py")
    
    # Features overview
    st.subheader("🚀 Key Features")
    
    features = [
        {
            "icon": "📈",
            "title": "Real-time Monitoring",
            "description": "Monitor stock prices and generate trading signals in real-time"
        },
        {
            "icon": "🤖",
            "title": "AI-Powered Signals",
            "description": "Machine learning models predict price movements with confidence scores"
        },
        {
            "icon": "📊",
            "title": "Technical Analysis",
            "description": "Comprehensive technical indicators including RSI, MACD, Bollinger Bands"
        },
        {
            "icon": "💰",
            "title": "Fundamental Analysis",
            "description": "Financial ratios integration (P/E, P/B, ROE, EPS, Growth rates)"
        },
        {
            "icon": "🔄",
            "title": "Backtesting",
            "description": "Test strategies on historical data with detailed performance metrics"
        },
        {
            "icon": "⚙️",
            "title": "Customizable",
            "description": "Adjustable thresholds, parameters, and model retraining capabilities"
        }
    ]
    
    # Display features in a grid
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-container">
                <h4>{feature['icon']} {feature['title']}</h4>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Getting started guide
    st.subheader("🚀 Getting Started")
    
    if not all(status.values()):
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Setup Required</h4>
            <p>Before using the dashboard, please complete the following setup steps:</p>
            <ol>
                <li><strong>Set Credentials:</strong> Copy <code>env_example.txt</code> to <code>.env</code> and add your FiinQuantX credentials</li>
                <li><strong>Install Dependencies:</strong> Run <code>pip install -r requirements.txt</code></li>
                <li><strong>Train Model:</strong> Go to Settings page to train your first model</li>
                <li><strong>Configure Parameters:</strong> Adjust signal thresholds and technical indicators</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>✅ System Ready</h4>
            <p>Your system is properly configured and ready to use! You can now:</p>
            <ul>
                <li>Monitor real-time stock prices on the Dashboard</li>
                <li>Run backtests to evaluate trading strategies</li>
                <li>Adjust settings and retrain models as needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity (if available)
    st.subheader("📋 Recent Activity")
    
    try:
        signals_file = os.path.join(Config.DATA_DIR, 'signals_log.csv')
        if os.path.exists(signals_file):
            import pandas as pd
            signals_df = pd.read_csv(signals_file)
            
            if not signals_df.empty:
                # Get recent signals
                recent_signals = signals_df.tail(5)
                
                st.markdown("**Latest Trading Signals**")
                for _, signal in recent_signals.iterrows():
                    signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
                    st.markdown(f"{signal_color} **{signal['ticker']}** - {signal['signal']} ({signal['timestamp']})")
            else:
                st.info("No trading signals generated yet")
        else:
            st.info("No signal log found - start monitoring stocks to generate signals")
            
    except Exception as e:
        st.warning(f"Could not load recent activity: {e}")

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📈 Dashboard", "📊 Backtest", "⚙️ Settings"]
    )
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Info**")
    st.sidebar.markdown(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if model exists
    model_status = "✅ Available" if os.path.exists(Config.MODEL_PATH) else "❌ Not Found"
    st.sidebar.markdown(f"🤖 Model: {model_status}")
    
    # Navigation logic
    if page == "🏠 Home":
        display_welcome_page()
    elif page == "📈 Dashboard":
        st.switch_page("pages/dashboard.py")
    elif page == "📊 Backtest":
        st.switch_page("pages/backtest_page.py")
    elif page == "⚙️ Settings":
        st.switch_page("pages/settings.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>📈 Stock Alert Dashboard v1.0 | Built with Streamlit & Machine Learning</p>
        <p>⚠️ This tool is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
