"""
Backtesting page for strategy evaluation
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_data import DataFetcher
from utils.features import TechnicalIndicators
from utils.preprocess import DataPreprocessor
from utils.signals import SignalGenerator
from utils.backtest import Backtester
from config import Config

# Page configuration
st.set_page_config(
    page_title="Backtest Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'signal_generator' not in st.session_state:
    st.session_state.signal_generator = SignalGenerator()
if 'technical_indicators' not in st.session_state:
    st.session_state.technical_indicators = TechnicalIndicators()
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()

def load_historical_data(ticker, days_back=365):
    """Load historical data for a ticker"""
    try:
        cached_data = st.session_state.data_fetcher.get_cached_historical_data(ticker)
        if cached_data.empty:
            cached_data = st.session_state.data_fetcher.fetch_historical_data(ticker, days_back)
        return cached_data
    except Exception as e:
        st.error(f"Error loading historical data for {ticker}: {e}")
        return pd.DataFrame()

def generate_signals_for_backtest(data, financial_data, ticker):
    """Generate signals for backtesting"""
    try:
        signals = []
        
        # Calculate technical indicators
        technical_data = st.session_state.technical_indicators.calculate_all_indicators(data)
        
        # Generate signals for each day
        for i in range(len(data)):
            # Prepare data for prediction (use data up to current point)
            current_data = data.iloc[:i+1].copy()
            current_technical = technical_data.iloc[:i+1].copy()
            
            # Calculate indicators for current data
            current_technical = st.session_state.technical_indicators.calculate_all_indicators(current_data)
            
            if current_technical.empty or len(current_technical) < 50:  # Need enough data
                signals.append('HOLD')
                continue
            
            # Prepare prediction data
            prediction_data = st.session_state.preprocessor.prepare_prediction_data(
                current_data, financial_data, ticker
            )
            
            if prediction_data.empty:
                signals.append('HOLD')
                continue
            
            # Get technical indicators for signal generation
            tech_indicators = current_technical.iloc[-1].to_dict()
            price_col = 'close' if 'close' in current_data.columns else ('Close' if 'Close' in current_data.columns else None)
            current_price = current_data[price_col].iloc[-1] if price_col else np.nan
            
            # Predict probability
            probability = st.session_state.signal_generator.predict_probability(prediction_data)
            
            # Generate signal
            signal_info = st.session_state.signal_generator.generate_signal(
                probability, current_price, tech_indicators
            )
            
            signals.append(signal_info['signal'])
        
        return pd.Series(signals, index=data.index)
        
    except Exception as e:
        st.error(f"Error generating signals: {e}")
        return pd.Series(['HOLD'] * len(data), index=data.index)

def create_backtest_charts(results):
    """Create interactive charts for backtest results"""
    if 'equity_curve' not in results:
        return None
    
    equity_df = results['equity_curve']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Equity Curve', 'Daily Returns', 'Drawdown', 'Position'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_hline(
        y=results['initial_capital'],
        line_dash="dash",
        line_color="red",
        annotation_text="Initial Capital",
        row=1, col=1
    )
    
    # Daily returns
    if 'returns' in equity_df.columns:
        returns = equity_df['returns'].dropna()
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name='Daily Returns',
                line=dict(color='green', width=1)
            ),
            row=1, col=2
        )
    
    # Drawdown
    peak = equity_df['total_value'].expanding().max()
    drawdown = (equity_df['total_value'] - peak) / peak
    
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Position
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df['position'],
            mode='lines',
            name='Position',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Backtest Results",
        height=800,
        showlegend=True
    )
    
    return fig

def display_performance_metrics(results):
    """Display performance metrics in a formatted way"""
    if 'error' in results:
        st.error(f"Error in backtest: {results['error']}")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{results['total_return_pct']:.2f}%",
            delta=f"{results['final_value'] - results['initial_capital']:,.0f} VND"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{results['sharpe_ratio']:.2f}"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown_pct']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{results['win_rate_pct']:.1f}%"
        )
    
    # Detailed metrics table
    st.subheader("📊 Detailed Performance Metrics")
    
    metrics_data = {
        'Metric': [
            'Initial Capital',
            'Final Value',
            'Total Return',
            'Annualized Return',
            'Volatility',
            'Sharpe Ratio',
            'Maximum Drawdown',
            'Total Trades',
            'Buy Trades',
            'Sell Trades',
            'Win Rate',
            'Average Trade Return'
        ],
        'Value': [
            f"{results['initial_capital']:,.0f} VND",
            f"{results['final_value']:,.0f} VND",
            f"{results['total_return_pct']:.2f}%",
            f"{results['total_return_pct']:.2f}%",  # Simplified annualized
            f"{results['volatility']:.2f}",
            f"{results['sharpe_ratio']:.2f}",
            f"{results['max_drawdown_pct']:.2f}%",
            f"{results['total_trades']}",
            f"{results['buy_trades']}",
            f"{results['sell_trades']}",
            f"{results['win_rate_pct']:.1f}%",
            f"{results['avg_trade_return_pct']:.2f}%"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

def display_trade_analysis(results):
    """Display trade analysis"""
    if 'trades' not in results:
        return
    
    trades = results['trades']
    
    if not trades:
        st.info("No trades executed during backtest period")
        return
    
    st.subheader("📋 Trade Analysis")
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    if not trades_df.empty:
        # Display executed trades only
        executed_trades = trades_df[trades_df.get('executed', False)]
        
        if not executed_trades.empty:
            # Format for display
            display_cols = ['date', 'signal', 'price', 'quantity', 'value', 'commission']
            available_cols = [col for col in display_cols if col in executed_trades.columns]
            
            st.dataframe(
                executed_trades[available_cols],
                use_container_width=True
            )
            
            # Trade statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Trade Statistics")
                
                total_commission = executed_trades['commission'].sum()
                avg_commission = executed_trades['commission'].mean()
                
                st.metric("Total Commission", f"{total_commission:,.0f} VND")
                st.metric("Average Commission", f"{avg_commission:,.0f} VND")
            
            with col2:
                # Trade frequency
                if 'date' in executed_trades.columns:
                    executed_trades['date'] = pd.to_datetime(executed_trades['date'])
                    trades_by_month = executed_trades.groupby(
                        executed_trades['date'].dt.to_period('M')
                    ).size()
                    
                    st.subheader("Trades by Month")
                    st.bar_chart(trades_by_month)

def main():
    """Main backtest page function"""
    st.title("📊 Backtest Analysis")
    st.markdown("Evaluate trading strategy performance on historical data")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Backtest Configuration")
    
    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Stock Ticker", 
        options=Config.TICKERS_WATCHLIST,
        index=0,
        help="Select stock ticker from watchlist"
    )
    
    # Date range
    st.sidebar.subheader("Date Range")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=start_date,
        max_value=end_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=end_date,
        max_value=datetime.date.today()
    )
    
    # Backtest parameters
    st.sidebar.subheader("Strategy Parameters")
    initial_capital = st.sidebar.number_input(
        "Initial Capital (VND)",
        min_value=100000,
        max_value=10000000,
        value=Config.INITIAL_CAPITAL,
        step=100000
    )
    
    commission_rate = st.sidebar.slider(
        "Commission Rate (%)",
        min_value=0.0,
        max_value=1.0,
        value=Config.COMMISSION_RATE * 100,
        step=0.01
    ) / 100
    
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
    
    # Update signal generator and backtester
    st.session_state.signal_generator.buy_threshold = buy_threshold
    st.session_state.signal_generator.sell_threshold = sell_threshold
    st.session_state.backtester.initial_capital = initial_capital
    st.session_state.backtester.commission_rate = commission_rate
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        if ticker:
            with st.spinner("Running backtest..."):
                # Load historical data
                historical_data = load_historical_data(ticker, days_back=730)
                
                if historical_data.empty:
                    st.error(f"No data available for ticker {ticker}")
                    return
                
                # Ensure 'TradingDate' exists and set as datetime index for filtering and indicators
                if 'TradingDate' not in historical_data.columns:
                    if 'timestamp' in historical_data.columns:
                        historical_data['TradingDate'] = pd.to_datetime(historical_data['timestamp'], errors='coerce')
                    else:
                        # Fallback: try to build from index
                        historical_data['TradingDate'] = pd.to_datetime(historical_data.index, errors='coerce')
                else:
                    historical_data['TradingDate'] = pd.to_datetime(historical_data['TradingDate'], errors='coerce')
                historical_data = historical_data.set_index('TradingDate')

                # Filter by date range using Timestamps to avoid dtype mismatch
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                historical_data = historical_data[(historical_data.index >= start_ts) & (historical_data.index <= end_ts)]

                # Ensure OHLCV are numeric (avoid all-zero due to dtype/object)
                for col in ['open','high','low','close','volume','Open','High','Low','Close','MatchVolume']:
                    if col in historical_data.columns:
                        historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')

                # Normalize columns: ensure both lowercase and uppercase OHLCV available
                # Lowercase from uppercase
                if 'close' not in historical_data.columns and 'Close' in historical_data.columns:
                    historical_data['close'] = historical_data['Close']
                if 'open' not in historical_data.columns and 'Open' in historical_data.columns:
                    historical_data['open'] = historical_data['Open']
                if 'high' not in historical_data.columns and 'High' in historical_data.columns:
                    historical_data['high'] = historical_data['High']
                if 'low' not in historical_data.columns and 'Low' in historical_data.columns:
                    historical_data['low'] = historical_data['Low']
                if 'volume' not in historical_data.columns and 'MatchVolume' in historical_data.columns:
                    historical_data['volume'] = historical_data['MatchVolume']
                # Uppercase from lowercase (for indicators expecting uppercase)
                if 'Close' not in historical_data.columns and 'close' in historical_data.columns:
                    historical_data['Close'] = historical_data['close']
                if 'Open' not in historical_data.columns and 'open' in historical_data.columns:
                    historical_data['Open'] = historical_data['open']
                if 'High' not in historical_data.columns and 'high' in historical_data.columns:
                    historical_data['High'] = historical_data['high']
                if 'Low' not in historical_data.columns and 'low' in historical_data.columns:
                    historical_data['Low'] = historical_data['low']
                if 'MatchVolume' not in historical_data.columns and 'volume' in historical_data.columns:
                    historical_data['MatchVolume'] = historical_data['volume']
                
                if historical_data.empty:
                    st.error("No data available for selected date range")
                    return

                # DEBUG: preview OHLCV after normalization
                debug_cols = [c for c in ['Open','High','Low','Close','MatchVolume','open','high','low','close','volume'] if c in historical_data.columns]
                st.caption("Preview OHLCV after filtering (tail 5):")
                st.dataframe(historical_data[debug_cols].tail(5))
                
                # Load financial data
                financial_data = st.session_state.data_fetcher.fetch_financial_ratios([ticker])
                
                # Generate signals
                st.subheader("🔄 Generating Signals...")
                signals = generate_signals_for_backtest(historical_data, financial_data, ticker)

                # DEBUG: signal distribution
                vc = signals.value_counts(dropna=False)
                st.caption("Signal distribution:")
                st.write(vc)
                num_trades = int(vc.get('BUY', 0)) + int(vc.get('SELL', 0))
                if num_trades == 0:
                    st.warning("No BUY/SELL signals generated in the selected range. Adjust thresholds or date range.")
                    return
                
                # Run backtest
                st.subheader("📈 Running Backtest...")
                results = st.session_state.backtester.run_backtest(
                    historical_data, signals, start_ts, end_ts
                )
                
                # Store results in session state
                st.session_state.backtest_results = results
                st.session_state.backtest_ticker = ticker
                
                st.success("Backtest completed successfully!")
        else:
            st.error("Please enter a valid ticker symbol")
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        ticker = st.session_state.backtest_ticker
        
        st.header(f"📊 Backtest Results for {ticker}")
        
        # Performance metrics
        display_performance_metrics(results)
        
        # Charts
        st.subheader("📈 Performance Charts")
        fig = create_backtest_charts(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        display_trade_analysis(results)
        
        # Export results
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Trade Log"):
                if 'trades' in results:
                    trades_df = pd.DataFrame(results['trades'])
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_backtest_trades.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("Download Equity Curve"):
                if 'equity_curve' in results:
                    equity_df = results['equity_curve']
                    csv = equity_df.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_equity_curve.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
