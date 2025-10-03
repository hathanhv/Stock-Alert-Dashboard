"""
Data fetching utilities for historical, realtime, and fundamental data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Callable, Optional
import time
import os
import json
import csv
from FiinQuantX import FiinSession
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DataFetcher:
    """Class to handle data fetching from various sources"""
    
    def __init__(self):
        """Initialize the data fetcher"""
        try:
            self.client = FiinSession(
                username=Config.USERNAME1,
                password=Config.PASSWORD1
            ).login()
            self.fi = self.client.FiinIndicator()
            logger.info("FiinQuantX client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FiinQuantX client: {e}")
            self.client = None
            self.fi = None
    
    def fetch_historical_data(self, ticker: str, days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical EOD data for a ticker with new structure
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to fetch back
            
        Returns:
            DataFrame with OHLCV and additional data
        """
        try:
            logger.info(f"Fetching historical data for {ticker} ({days_back} days back)")
            
            # Check if client is initialized
            if self.client is None:
                logger.error("FiinQuantX client not initialized")
                return pd.DataFrame()
            
            # Calculate start and end dates (ensure we go back enough to get trading days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)  # Add buffer for weekends/holidays
            
            logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Fetch data using FiinQuantX with new structure
            data_result = self.client.Fetch_Trading_Data(
                realtime=False,
                tickers=ticker,
                fields=['open','high','low','close','volume','bu','sd','fn','fs','fb'],
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d'),
                by='1d'
            )
            
            # Get DataFrame from result object
            data = data_result.get_data() if hasattr(data_result, 'get_data') else data_result
        
            
            # Check if we have required columns
            logger.info(f"Raw data columns: {list(data.columns)}")
            logger.info(f"Raw data shape: {data.shape}")
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Rename columns to match expected structure
            if 'TradingDate' not in data.columns:
                data['TradingDate'] = data['timestamp']
            if 'Open' not in data.columns:
                data['Open'] = data['open'] if 'open' in data.columns else data['Open']
            if 'Close' not in data.columns:
                data['Close'] = data['close'] if 'close' in data.columns else data['Close']
            if 'High' not in data.columns:
                data['High'] = data['high'] if 'high' in data.columns else data['High']
            if 'Low' not in data.columns:
                data['Low'] = data['low'] if 'low' in data.columns else data['Low']
            if 'MatchVolume' not in data.columns:
                data['MatchVolume'] = data['volume'] if 'volume' in data.columns else data['MatchVolume']
            
            # Save to cache
            cache_file = os.path.join(Config.HISTORICAL_DATA_DIR, f"{ticker}_historical.csv")
            data.to_csv(cache_file)
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_realtime_data(self, tickers: List[str], callback: Callable = None, run_forever: bool = True) -> Dict[str, Dict]:
        """
        Fetch realtime data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            callback: Callback function to handle realtime data
            
        Returns:
            Dictionary with realtime data for each ticker
        """
        try:
            logger.info(f"Fetching realtime data for {len(tickers)} tickers")
            
            realtime_data = {}

            def on_event(data):
                """Handle realtime data events from FiinQuantX RealTimeData"""
                try:
                    # Convert to DataFrame per vendor sample
                    df = data.to_dataFrame() if hasattr(data, 'to_dataFrame') else None
                    if df is None or df.empty:
                        return
                    # Iterate rows (usually one row per tick)
                    for _, row in df.iterrows():
                        ticker = str(row.get('Ticker') or row.get('ticker') or '').upper()
                        if not ticker or ticker not in [t.upper() for t in tickers]:
                            continue
                        # Helper to fetch and coerce numeric field from possible names
                        def get_num(names, default=0.0):
                            for name in names:
                                if name in row and pd.notna(row[name]):
                                    try:
                                        return float(pd.to_numeric(row[name], errors='coerce'))
                                    except Exception:
                                        continue
                            return float(default)
                        snapshot = {
                            'price': get_num(['Close','close','Last','last','MatchedPrice']),
                            'volume': get_num(['MatchVolume','TotalMatchVolume','volume','Vol','TotalVol']),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'open': get_num(['Open','open']),
                            'high': get_num(['High','high']),
                            'low': get_num(['Low','low'])
                        }
                        realtime_data[ticker] = snapshot
                        # Save to cache
                        os.makedirs(Config.REALTIME_DATA_DIR, exist_ok=True)
                        cache_file = os.path.join(Config.REALTIME_DATA_DIR, f"{ticker}_realtime.json")
                        with open(cache_file, 'w') as f:
                            json.dump(snapshot, f, default=str)
                        logger.debug(f"Wrote realtime snapshot: {cache_file}")
                        # Append to consolidated CSV log
                        try:
                            log_path = os.path.join(Config.REALTIME_DATA_DIR, 'realtime_log.csv')
                            file_exists = os.path.exists(log_path)
                            with open(log_path, 'a', newline='', encoding='utf-8') as lf:
                                writer = csv.DictWriter(lf, fieldnames=['timestamp','ticker','price','volume','open','high','low'])
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow({
                                    'timestamp': snapshot['timestamp'],
                                    'ticker': ticker,
                                    'price': snapshot['price'],
                                    'volume': snapshot['volume'],
                                    'open': snapshot['open'],
                                    'high': snapshot['high'],
                                    'low': snapshot['low']
                                })
                        except Exception as _:
                            pass
                        # User callback
                        if callback:
                            try:
                                callback(snapshot)
                            except Exception as _:
                                pass
                except Exception as e:
                    logger.error(f"Error processing realtime data: {e}")

            # Create stream per vendor API and start it
            Events = self.client.Trading_Data_Stream(tickers=tickers, callback=on_event)
            Events.start()

            if run_forever:
                try:
                    while not getattr(Events, '_stop', False):
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Realtime stream stopped by user")
                    try:
                        Events.stop()
                    except Exception:
                        pass
            return realtime_data
            
        except Exception as e:
            logger.error(f"Error fetching realtime data: {e}")
            return {}
    
    def fetch_financial_ratios(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch financial ratios for tickers (most recent quarterly report)
        Gets the latest financial data for real-time analysis
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with financial ratios from the most recent quarter
        """
        try:
            logger.info(f"Fetching financial ratios for {len(tickers)} tickers")
            
            # Get financial ratios (vendor may return dict keyed by ticker -> list of reports)
            fi_dict = self.client.FundamentalAnalysis().get_ratios(
                # tickers=tickers,
                # TimeFilter="Quarterly",
                # LatestYear=2025,
                # NumberOfPeriod=1,
                # Consolidated=True
                tickers=tickers,
                years=[2025],
                quarters=[2],
                type="consolidated"
            )
            
            if not fi_dict:
                logger.warning("No financial ratios data found")
                return pd.DataFrame()
            
            # Debug: log the structure of fi_dict
            logger.info(f"Financial data structure: {type(fi_dict)}")
            
            # Convert to DataFrame
            ratios_data = []
            
            def append_item(ticker_code: str, item: dict):
                ratios = item.get("ratios", {}) if isinstance(item, dict) else {}
                profitability = ratios.get("ProfitabilityRatio", {}) if isinstance(ratios, dict) else {}
                valuation = ratios.get("ValuationRatios", {}) if isinstance(ratios, dict) else {}
                growth = ratios.get("Growth", {}) if isinstance(ratios, dict) else {}
                ratios_data.append({
                    "Ticker": ticker_code,
                    "PB": valuation.get("PriceToBook", np.nan),
                    "PE": valuation.get("PriceToEarning", np.nan),
                    "ROE": profitability.get("ROE", np.nan),
                    "EPS": valuation.get("BasicEPS", np.nan),
                    "RevenueGrowth": growth.get("NetRevenueGrowthYoY", np.nan),
                    "ProfitGrowth": growth.get("GrossProfitGrowthYoY", np.nan),
                    "ReportYear": item.get("year"),
                    "ReportQuarter": item.get("quarter")
                })

            # Vendor may return dict: { 'AAM': [ {year, quarter, ratios: {...}}, ... ], ... }
            if isinstance(fi_dict, dict):
                for ticker_code, reports in fi_dict.items():
                    if isinstance(reports, list):
                        for report in reports:
                            append_item(ticker_code, report)
                    elif isinstance(reports, dict):
                        append_item(ticker_code, reports)
            elif isinstance(fi_dict, list):
                for item in fi_dict:
                    ticker_code = item.get("ticker") if isinstance(item, dict) else None
                    if ticker_code:
                        append_item(ticker_code, item)
            else:
                logger.warning("Unrecognized financial ratios format")
            
            ratios_df = pd.DataFrame(ratios_data)
            
            # if not ratios_df.empty:
            #     # Save to cache
            #     cache_file = os.path.join(Config.DATA_DIR, 'financial_ratios.csv')
            #     ratios_df.to_csv(cache_file, index=False)
            #     logger.info(f"Successfully fetched financial ratios for {len(ratios_df)} tickers")
            
            return ratios_df
            
        except Exception as e:
            logger.error(f"Error fetching financial ratios: {e}")
            return pd.DataFrame()
    
    def get_cached_historical_data(self, ticker: str) -> pd.DataFrame:
        """Get cached historical data if available"""
        cache_file = os.path.join(Config.HISTORICAL_DATA_DIR, f"{ticker}_historical.csv")
        if os.path.exists(cache_file):
            try:
                # Try reading without assuming an index column
                data = pd.read_csv(cache_file)
                # Ensure expected columns exist or try common fallbacks
                if 'timestamp' in data.columns and 'TradingDate' not in data.columns:
                    data['TradingDate'] = pd.to_datetime(data['timestamp'], errors='coerce')
                if 'open' in data.columns and 'Open' not in data.columns:
                    data['Open'] = data['open']
                if 'high' in data.columns and 'High' not in data.columns:
                    data['High'] = data['high']
                if 'low' in data.columns and 'Low' not in data.columns:
                    data['Low'] = data['low']
                if 'close' in data.columns and 'Close' not in data.columns:
                    data['Close'] = data['close']
                if 'volume' in data.columns and 'MatchVolume' not in data.columns:
                    data['MatchVolume'] = data['volume']
                logger.info(f"Loaded cached historical data for {ticker}")
                return data
            except Exception as e:
                logger.error(f"Error loading cached data for {ticker}: {e}")
        return pd.DataFrame()
    
    def get_cached_realtime_data(self, ticker: str) -> Dict:
        """Get cached realtime data if available"""
        cache_file = os.path.join(Config.REALTIME_DATA_DIR, f"{ticker}_realtime.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded cached realtime data for {ticker}")
                return data
            except Exception as e:
                logger.error(f"Error loading cached realtime data for {ticker}: {e}")
        return {}
    
    def update_historical_data(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """
        Update historical data with recent data
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of recent days to fetch
            
        Returns:
            Updated DataFrame
        """
        try:
            # Get existing data
            existing_data = self.get_cached_historical_data(ticker)
            
            # if existing_data.empty:
            #     # If no existing data, fetch full dataset
            #     return self.fetch_historical_data(ticker, days_back=365)
            
            # Get recent data
            recent_data = self.fetch_historical_data(ticker, days_back)
            
            # if recent_data.empty:
            #     return existing_data
            
            # Merge data
            combined_data = pd.concat([existing_data, recent_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            
            # Save updated data
            cache_file = os.path.join(Config.HISTORICAL_DATA_DIR, f"{ticker}_historical.csv")
            combined_data.to_csv(cache_file)
            
            logger.info(f"Updated historical data for {ticker}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error updating historical data for {ticker}: {e}")
            return self.get_cached_historical_data(ticker)
    
    def get_all_tickers(self) -> List[str]:
        """
        Get all tickers from 3 exchanges (HOSE, HNX, UPCOM)
        
        Returns:
            List of all ticker symbols
        """
        try:
            if self.client is None:
                logger.error("FiinQuantX client not initialized")
                return []
            
            logger.info("Fetching all tickers from 3 exchanges...")
            
            # Get tickers from each exchange
            hose = list(self.client.TickerList(tickers="VNINDEX"))
            hnx = list(self.client.TickerList(tickers="HNXINDEX"))
            upcom = list(self.client.TickerList(tickers="UPCOMINDEX"))
            
            # Combine all tickers
            all_tickers = hose + hnx + upcom
            
            logger.info(f"Found {len(all_tickers)} total tickers:")
            logger.info(f"  HOSE: {len(hose)} tickers")
            logger.info(f"  HNX: {len(hnx)} tickers")
            logger.info(f"  UPCOM: {len(upcom)} tickers")
            
            return all_tickers
            
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return []
    
    def get_tickers_by_exchange(self) -> Dict[str, List[str]]:
        """
        Get tickers grouped by exchange
        
        Returns:
            Dictionary with exchange names as keys and ticker lists as values
        """
        try:
            if self.client is None:
                logger.error("FiinQuantX client not initialized")
                return {}
            
            logger.info("Fetching tickers by exchange...")
            
            # Get tickers from each exchange
            hose = list(self.client.TickerList(tickers="VNINDEX"))
            hnx = list(self.client.TickerList(tickers="HNXINDEX"))
            upcom = list(self.client.TickerList(tickers="UPCOMINDEX"))
            
            tickers_by_exchange = {
                'HOSE': hose,
                'HNX': hnx,
                'UPCOM': upcom
            }
            
            logger.info(f"Retrieved tickers from {len(tickers_by_exchange)} exchanges")
            return tickers_by_exchange
            
        except Exception as e:
            logger.error(f"Error fetching tickers by exchange: {e}")
            return {}
    
    def check_market_status(self) -> Dict:
        """
        Check market status and trading hours
        
        Returns:
            Dictionary with market status information
        """
        try:
            if self.client is None:
                return {"error": "FiinQuantX client not initialized"}
            
            current_time = datetime.now()
            current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Vietnamese stock market trading hours
            market_info = {
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": current_time.strftime("%A"),
                "is_weekend": current_weekday >= 5,  # Saturday or Sunday
                "trading_hours": {
                    "morning_session": "09:00-11:30",
                    "afternoon_session": "13:00-15:00"
                },
                "is_trading_time": False
            }
            
            # Check if it's trading time (Monday-Friday, 9:00-11:30 or 13:00-15:00)
            if current_weekday < 5:  # Monday to Friday
                current_hour_minute = current_time.hour * 60 + current_time.minute
                
                # Morning session: 9:00-11:30 (540-690 minutes)
                # Afternoon session: 13:00-15:00 (780-900 minutes)
                if (540 <= current_hour_minute <= 690) or (780 <= current_hour_minute <= 900):
                    market_info["is_trading_time"] = True
            
            market_info["status"] = "OPEN" if market_info["is_trading_time"] and not market_info["is_weekend"] else "CLOSED"
            
            return market_info
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return {"error": str(e)}
    
    def debug_ticker_data(self, ticker: str) -> Dict:
        """
        Debug function to check ticker data availability
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with debug information
        """
        try:
            debug_info = {
                "ticker": ticker,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_status": self.check_market_status(),
                "client_status": "OK" if self.client is not None else "NOT_INITIALIZED"
            }
            
            if self.client is None:
                debug_info["error"] = "FiinQuantX client not initialized"
                return debug_info
            
            # Try to fetch a small amount of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Just 7 days
            
            try:
                data_result = self.client.Fetch_Trading_Data(
                    realtime=False,
                    tickers=ticker,
                    fields=['open','high','low','close','volume'],
                    from_date=start_date.strftime('%Y-%m-%d'),
                    to_date=end_date.strftime('%Y-%m-%d'),
                    by='1d'
                )
                
                # Get DataFrame from result object
                data = data_result.get_data() if hasattr(data_result, 'get_data') else data_result
                    
            except Exception as fetch_error:
                debug_info["data_status"] = "ERROR"
                debug_info["fetch_error"] = str(fetch_error)
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error in debug_ticker_data: {e}")
            return {"error": str(e), "ticker": ticker}
