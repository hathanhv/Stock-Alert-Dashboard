"""
Test script to verify FiinQuantX connection and data fetching
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.fetch_data import DataFetcher
from config import Config

def test_connection():
    """Test FiinQuantX connection and data fetching"""
    print("🚀 Testing FiinQuantX Connection...")
    
    try:
        # Initialize data fetcher
        print("1. Initializing DataFetcher...")
        fetcher = DataFetcher()
        
        if fetcher.client is None:
            print("❌ Failed to initialize FiinQuantX client")
            return False
        
        print("✅ FiinQuantX client initialized successfully")
        
        # Check market status
        print("\n2. Checking market status...")
        market_status = fetcher.check_market_status()
        print(f"Market Status: {market_status.get('status', 'Unknown')}")
        print(f"Current Time: {market_status.get('current_time', 'Unknown')}")
        print(f"Is Trading Time: {market_status.get('is_trading_time', 'Unknown')}")
        
        # Test ticker list
        print("\n3. Testing ticker list...")
        try:
            tickers_by_exchange = fetcher.get_tickers_by_exchange()
            if tickers_by_exchange:
                for exchange, tickers in tickers_by_exchange.items():
                    print(f"{exchange}: {len(tickers)} tickers")
            else:
                print("❌ No tickers found")
        except Exception as e:
            print(f"❌ Error getting tickers: {e}")
        
        # Test historical data fetching
        print("\n4. Testing historical data fetching...")
        test_ticker = "VCB"
        debug_info = fetcher.debug_ticker_data(test_ticker)
        
        print(f"Ticker: {debug_info.get('ticker', 'Unknown')}")
        print(f"Client Status: {debug_info.get('client_status', 'Unknown')}")
        print(f"Data Status: {debug_info.get('data_status', 'Unknown')}")
        
        if 'data_shape' in debug_info:
            print(f"Data Shape: {debug_info['data_shape']}")
        if 'fetch_error' in debug_info:
            print(f"❌ Fetch Error: {debug_info['fetch_error']}")
        
        # Test actual data fetch
        print("\n5. Testing actual data fetch...")
        try:
            # Test direct API call first
            print("Testing direct API call...")
            data_result = fetcher.client.Fetch_Trading_Data(
                realtime=False,
                tickers=test_ticker,
                fields=['open','high','low','close','volume'],
                from_date='2024-01-01',
                to_date='2024-12-31',
                by='1d'
            )
            
            print(f"API result type: {type(data_result)}")
            print(f"API result methods: {[method for method in dir(data_result) if not method.startswith('_')]}")
            
            # Try to get data
            if hasattr(data_result, 'get_data'):
                data = data_result.get_data()
                print(f"✅ Successfully got DataFrame with shape: {data.shape}")
                print(f"Columns: {list(data.columns)}")
            else:
                print("❌ No get_data() method found")
                
        except Exception as e:
            print(f"❌ Error in direct API test: {e}")
        
        # Test using fetcher method
        try:
            data = fetcher.fetch_historical_data(test_ticker, days_back=30)
            if not data.empty:
                print(f"✅ Successfully fetched {len(data)} records for {test_ticker}")
                print(f"Columns: {list(data.columns)}")
                print(f"Date range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"❌ No data returned for {test_ticker}")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
        
        # Test financial ratios
        print("\n6. Testing financial ratios...")
        try:
            financial_data = fetcher.fetch_financial_ratios(['VCB', 'VIC'])
            if not financial_data.empty:
                print(f"✅ Successfully fetched financial data for {len(financial_data)} tickers")
                print(f"Financial data columns: {list(financial_data.columns)}")
                print(financial_data.head())
            else:
                print("❌ No financial data returned")
        except Exception as e:
            print(f"❌ Error fetching financial data: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Connection test completed!")
    else:
        print("\n💥 Connection test failed!")
