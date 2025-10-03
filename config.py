"""
Configuration module for Stock Alert Dashboard
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # FiinQuantX credentials
    USERNAME1 = os.getenv('USERNAME1')
    PASSWORD1 = os.getenv('PASSWORD1')
    
    # Model configuration
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'checkpoint.pkl')
    
    # Data paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, 'historical')
    REALTIME_DATA_DIR = os.path.join(DATA_DIR, 'realtime')
    
    # Signal thresholds
    BUY_THRESHOLD = 0.7  # Probability threshold for BUY signal
    SELL_THRESHOLD = 0.3  # Probability threshold for SELL signal
    
    # Technical indicators parameters
    RSI_PERIOD = 14
    EMA_PERIODS = [5, 10, 20, 50]
    SMA_PERIODS = [20, 50, 100, 200]
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Prediction horizon (days)
    PREDICTION_DAYS = 10
    
    # Backtest configuration
    INITIAL_CAPITAL = 1000000  # 1M VND
    COMMISSION_RATE = 0.0015  # 0.15% commission
    
    # Streamlit configuration
    REFRESH_INTERVAL = 30  # seconds
    MAX_TICKERS = 50
    
    # Watchlist for testing (10 tickers)
    TICKERS_WATCHLIST = [
        'VCB',  # Vietcombank
        'VIC',  # Vingroup
        'HPG',  # Hoa Phat Group
        'CTG',  # VietinBank
        'BID',  # BIDV
        'GAS',  # PetroVietnam Gas
        'VHM',  # Vinhomes
        'MSN',  # Masan Group
        'PLX',  # Petrolimex
        'VRE',   # Vincom Retail
        'AAM',
        'TVH',
        'POM',
        'VNM',
        'FPT',
        'CTR',
        'SCS'
    ]
    
    # Logging configuration
    LOG_LEVEL = 'DEBUG'
    LOG_FILE = 'stock_alert.log'
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.USERNAME1 or not cls.PASSWORD1:
            raise ValueError("USERNAME1 and PASSWORD1 must be set in .env file")
        
        # Create directories if they don't exist
        os.makedirs(cls.HISTORICAL_DATA_DIR, exist_ok=True)
        os.makedirs(cls.REALTIME_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        
        return True
