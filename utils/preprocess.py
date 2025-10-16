"""
Data preprocessing and feature engineering utilities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from config import Config
from utils.features import TechnicalIndicators

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class to handle data preprocessing and feature engineering"""
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.scaler_fitted = False
        # Load canonical feature order from JSON if present, else default
        default_order = [
            "price_vs_ema50", "price_vs_ema200", "ema_ratio",
            "macd_signal_strength", "macd_trend", "trend_score",
            "rsi", "mfi", "bb_pos", "vol_ratio",
            "rsi_overbought", "rsi_oversold", "mfi_momentum",
            "volume_momentum", "momentum_score",
            "PE", "PB", "ROE", "EPS", "RevenueGrowth", "ProfitGrowth",
            "pe_normalized", "pb_normalized", "growth_score",
            "roe_quality", "value_score"
        ]
        self.FEATURE_ORDER = default_order
        try:
            import json
            features_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_columns.json')
            if os.path.exists(features_path):
                with open(features_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list) and loaded:
                        self.FEATURE_ORDER = loaded
        except Exception as _:
            self.FEATURE_ORDER = default_order
    
    def merge_financial_data(self, price_data: pd.DataFrame, financial_data: pd.DataFrame, 
                            ticker: str) -> pd.DataFrame:
        """
        Merge price data with financial ratios data
        
        Args:
            price_data: OHLCV price data
            financial_data: Financial ratios data
            ticker: Stock ticker symbol
            
        Returns:
            Merged DataFrame
        """
        try:
            result = price_data.copy()

            # Filter financial data for the specific ticker
            fin = financial_data[financial_data['Ticker'] == ticker].copy()

            # Prepare return with empty columns if no financials
            financial_cols = ['PB', 'PE', 'ROE', 'EPS', 'RevenueGrowth', 'ProfitGrowth']
            for col in financial_cols:
                if col not in result.columns:
                    result[col] = np.nan

            if fin.empty:
                logger.warning(f"No financial data found for {ticker}")
                return result

            # Build report end date from ReportYear/ReportQuarter (Q1=03-31, Q2=06-30, Q3=09-30, Q4=12-31)
            fin = fin.copy()
            if 'ReportYear' in fin.columns and 'ReportQuarter' in fin.columns:
                q_to_month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
                def to_report_end(y, q):
                    try:
                        m, d = q_to_month_day.get(int(q), (12, 31))
                        return pd.Timestamp(int(y), m, d)
                    except Exception:
                        return pd.NaT
                fin['ReportEnd'] = fin.apply(lambda r: to_report_end(r.get('ReportYear'), r.get('ReportQuarter')), axis=1)
            elif 'TradingDate' in fin.columns:
                fin['ReportEnd'] = pd.to_datetime(fin['TradingDate'], errors='coerce')
            else:
                fin['ReportEnd'] = pd.NaT

            # Keep only needed columns for merge
            keep_cols = ['Ticker', 'ReportEnd'] + [c for c in financial_cols if c in fin.columns]
            fin = fin[keep_cols].dropna(subset=['ReportEnd']).sort_values('ReportEnd')

            # Ensure price dates are datetime and sorted
            if 'TradingDate' in result.columns:
                result['TradingDate'] = pd.to_datetime(result['TradingDate'], errors='coerce')
            result = result.sort_values('TradingDate')

            # As-of merge: for each TradingDate, take latest previous report
            merged = pd.merge_asof(
                result.sort_values('TradingDate'),
                fin.sort_values('ReportEnd'),
                left_on='TradingDate',
                right_on='ReportEnd',
                by=None,  # fin already filtered by ticker
                direction='backward'
            )

            # If multiple tickers in price_data (unlikely here), ensure only the current ticker rows are affected
            merged.loc[merged.get('Ticker') != ticker, financial_cols] = merged.loc[merged.get('Ticker') != ticker, financial_cols]

            logger.info(f"Successfully merged financial data for {ticker} using merge_asof")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging financial data for {ticker}: {e}")
            return price_data
    
    def prepare_quantitative_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare quantitative data for processing
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Processed DataFrame
        """
        try:
            logger.info("📊 Starting data preprocessing...")
            
            # Create copy to avoid modifying original data
            data = df.copy()
            
            # Ensure timestamp is datetime
            if 'TradingDate' in data.columns:
                data['TradingDate'] = pd.to_datetime(data['TradingDate'])
            
            # Sort by ticker and timestamp
            data = data.sort_values(['Ticker', 'TradingDate']).reset_index(drop=True)
            
            logger.info(f"📈 Total records: {len(data):,}")
            logger.info(f"🏢 Number of tickers: {data['Ticker'].nunique()}")
            if 'TradingDate' in data.columns:
                logger.info(f"📅 Date range: {data['TradingDate'].min()} to {data['TradingDate'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing quantitative data: {e}")
            return df
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            logger.info("🔍 Handling missing values...")
            
            # Check missing values
            missing_summary = data.isnull().sum()
            missing_pct = (missing_summary / len(data)) * 100
            
            logger.info("Missing values by column:")
            for col in missing_summary[missing_summary > 0].index:
                logger.info(f"  {col}: {missing_summary[col]:,} ({missing_pct[col]:.2f}%)")
            
            result = data.copy()
            
            # Forward fill for technical indicators
            technical_cols = ['rsi', 'mfi', 'macd', 'macd_signal', 'macd_diff',
                             'bollinger_hband', 'bollinger_lband', 'bb_pos', 'vol_ratio',
                             'ema50', 'ema200', 'vol_ma20']
            
            for col in technical_cols:
                if col in result.columns:
                    result[col] = result.groupby('Ticker')[col].fillna(method='ffill')
            
            # Handle fundamental data with multi-step approach
            fundamental_cols = ['PB', 'PE', 'ROE', 'EPS', 'RevenueGrowth', 'ProfitGrowth']
            
            for col in fundamental_cols:
                if col in result.columns:
                    # Step 1: Forward fill by ticker (keep latest period value)
                    result[col] = result.groupby("Ticker")[col].ffill()
                    
                    # Step 2: Fill with ticker median
                    result[col] = result.groupby("Ticker")[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    
                    # Step 3: Fallback with market median
                    median_val = result[col].median()
                    result[col] = result[col].fillna(median_val)
            
            logger.info("✅ Missing values handled")
            return result
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return data
    
    def reorder_and_rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder and rename columns to standard format
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with reordered and renamed columns
        """
        # print("TÉT")
        # print(data.columns)
        try:
            # Define robust mapping from raw to standardized names
            mapping_pairs = {
                'ticker': 'ticker', 'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
                'rsi': 'rsi', 'mfi': 'mfi', 'macd': 'macd', 'macd_signal': 'macd_signal', 'macd_diff': 'macd_diff',
                'bollinger_hband': 'bollinger_hband', 'bollinger_lband': 'bollinger_lband', 'bollinger_mband': 'bollinger_mband', 'bb_pos': 'bb_pos',
                'ema50': 'ema50', 'ema200': 'ema200', 'vol_ma20': 'vol_ma20', 'vol_ratio': 'vol_ratio', 'Volume Ratio': 'vol_ratio',
                'PB': 'PB', 'PE': 'PE', 'ROE': 'ROE', 'EPS': 'EPS', 'RevenueGrowth': 'RevenueGrowth', 'ProfitGrowth': 'ProfitGrowth',
                'return_5d': 'return_5d', 'target_binary_5d': 'target_binary_5d', 'target_multi_5d': 'target_multi_5d',
                'return_10d': 'return_10d', 'target_binary_10d': 'target_binary_10d', 'target_multi_10d': 'target_multi_10d',
                'return_20d': 'return_20d', 'target_binary_20d': 'target_binary_20d', 'target_multi_20d': 'target_multi_20d',
                'price_vs_ema50': 'price_vs_ema50', 'price_vs_ema200': 'price_vs_ema200', 'ema_ratio': 'ema_ratio',
                'macd_signal_strength': 'macd_signal_strength', 'macd_trend': 'macd_trend', 'trend_score': 'trend_score',
                'rsi_overbought': 'rsi_overbought', 'rsi_oversold': 'rsi_oversold', 'rsi_neutral': 'rsi_neutral',
                'mfi_momentum': 'mfi_momentum', 'volume_momentum': 'volume_momentum', 'momentum_score': 'momentum_score',
                'pe_normalized': 'pe_normalized', 'pb_normalized': 'pb_normalized', 'growth_score': 'growth_score', 'roe_quality': 'roe_quality', 'value_score': 'value_score',
                # Raw vendor columns kept only for intermediate steps; will drop
                'bu': 'bu', 'sd': 'sd', 'fn': 'fn', 'fs': 'fs', 'fb': 'fb',
                'Ticker': 'Ticker', 'TradingDate': 'TradingDate', 'Open': 'Open', 'Close': 'Close', 'High': 'High', 'Low': 'Low', 'MatchVolume': 'MatchVolume'
            }

            present_old = [c for c in mapping_pairs.keys() if c in data.columns]
            col_mapping = {c: mapping_pairs[c] for c in present_old}
            result = data[present_old].rename(columns=col_mapping)
            drop_cols = [
                'bu', 'sd', 'fn', 'fs', 'fb', 'Ticker', 'TradingDate',
                'Open', 'Close', 'High', 'Low', 'MatchVolume'
            ]
            result = result.drop(columns=[c for c in drop_cols if c in result.columns])

            logger.info(f"Reordered and renamed {len(present_old)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Error reordering and renaming columns: {e}")
            return data
    
    def select_features(self, data: pd.DataFrame, feature_list: List[str] = None) -> pd.DataFrame:
        """
        Select relevant features for model training
        
        Args:
            data: Input DataFrame
            feature_list: List of features to select (if None, uses default)
            
        Returns:
            DataFrame with selected features
        """
        try:
            if feature_list is None:
                # Use canonical order
                feature_list = self.FEATURE_ORDER.copy()
            
            # Filter available features
            available_features = [f for f in feature_list if f in data.columns]
            
            # Add target variables
            target_cols = ['target_binary_5d', 'target_binary_10d', 'target_binary_20d',
                          'return_5d', 'return_10d', 'return_20d']
            for col in target_cols:
                if col in data.columns:
                    available_features.append(col)
            
            result = data[available_features].copy()
            
            # Store feature columns for later use
            self.feature_columns = [f for f in available_features if f not in target_cols]
            
            logger.info(f"Selected {len(self.feature_columns)} features for training")
            return result
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return data
    
    def scale_features(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale features for model training
        
        Args:
            data: Input DataFrame
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            result = data.copy()
            
            # Get numeric columns
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            
            if fit_scaler:
                # Fit scaler on training data
                result[numeric_columns] = self.scaler.fit_transform(result[numeric_columns])
                self.scaler_fitted = True
                
                # Save scaler
                scaler_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                
                logger.info("Fitted and saved scaler")
            else:
                # Use existing scaler for prediction
                if self.scaler_fitted:
                    result[numeric_columns] = self.scaler.transform(result[numeric_columns])
                else:
                    # Try to load existing scaler
                    scaler_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'scaler.pkl')
                    if os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)
                        result[numeric_columns] = self.scaler.transform(result[numeric_columns])
                        self.scaler_fitted = True
                        logger.info("Loaded and applied existing scaler")
                    else:
                        logger.warning("No scaler found, using unscaled data")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return data
    
    def prepare_training_data(self, data: pd.DataFrame, financial_data: pd.DataFrame = None,
                            ticker: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare complete training dataset
        
        Args:
            data: OHLCV price data
            financial_data: Financial ratios data
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (features, target)
        """
        try:
            # Merge financial data if provided
            if financial_data is not None and ticker is not None:
                data = self.merge_financial_data(data, financial_data, ticker)
            
            # Prepare quantitative data
            data = self.prepare_quantitative_data(data)

            # Compute technical indicators to ensure required feature columns exist
            technical_indicators = TechnicalIndicators()
            data = technical_indicators.calculate_all_indicators(data)
            
            # Handle missing values
            data = self.handle_missing_values(data)
            
            # Reorder and rename columns
            data = self.reorder_and_rename_columns(data)
            print("TÉT0")
            print(data.columns)
            # Select features
            data = self.select_features(data)

            # Persist selected feature columns for inference
            try:
                import json
                features_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_columns.json')
                with open(features_path, 'w', encoding='utf-8') as f:
                    json.dump(self.feature_columns, f, ensure_ascii=False)
            except Exception as _:
                pass

            # Scale features (fit scaler on training data and save)
            data[self.feature_columns] = self.scale_features(data[self.feature_columns], fit_scaler=True)
            
            # Remove rows with missing target (last prediction_days rows)
            target_cols = [col for col in data.columns if col.startswith('target_binary_')]
            if target_cols:
                data = data.dropna(subset=target_cols)
            
            # Separate features and target
            feature_cols = self.feature_columns
            X = data[feature_cols]
            y = data['target_binary_10d']  # Default to 10-day prediction
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def prepare_prediction_data(self, data: pd.DataFrame, financial_data: pd.DataFrame = None,
                              ticker: str = None) -> pd.DataFrame:
        """
        Prepare data for prediction
        
        Args:
            data: OHLCV price data
            financial_data: Financial ratios data
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame ready for prediction
        """
        try:
            # Make a working copy and ensure required structural columns exist
            df = data.copy()
            debug_steps = {}
            debug_steps['start_cols'] = list(df.columns)
            # Restore TradingDate from index if missing
            if 'TradingDate' not in df.columns:
                try:
                    df['TradingDate'] = pd.to_datetime(df.index)
                except Exception:
                    pass
            # Ensure Ticker column exists for group-based indicators
            if 'Ticker' not in df.columns and ticker is not None:
                df['Ticker'] = ticker
            # Normalize OHLCV column names to expected uppercase if only lowercase exist
            name_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'MatchVolume'
            }
            for src, dst in name_map.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
            debug_steps['after_normalize_cols'] = list(df.columns)

            # Compute technical indicators to ensure required feature columns exist
            technical_indicators = TechnicalIndicators()
            df = technical_indicators.calculate_all_indicators(df)
            debug_steps['after_indicators_cols'] = list(df.columns)

            # If indicators did not materialize (common if a lib error occurred),
            # compute a minimal single-ticker feature set inline as a fallback.
            try:
                base_needed = ['ema50', 'ema200', 'rsi', 'macd_diff', 'bollinger_hband', 'bollinger_lband', 'vol_ma20', 'vol_ratio']
                if not any(c in df.columns for c in base_needed):
                    close_series = pd.to_numeric(df['Close'], errors='coerce')
                    vol_series = pd.to_numeric(df['MatchVolume'], errors='coerce') if 'MatchVolume' in df.columns else None
                    import pandas_ta as _ta
                    df['ema50'] = _ta.ema(close_series, length=50)
                    df['ema200'] = _ta.ema(close_series, length=200)
                    macd_df = _ta.macd(close_series, fast=12, slow=26, signal=9)
                    if macd_df is not None and hasattr(macd_df, 'columns'):
                        # MACDh is histogram (diff)
                        macd_cols = [c for c in macd_df.columns if str(c).startswith('MACDh_')]
                        if macd_cols:
                            df['macd_diff'] = macd_df[macd_cols[0]]
                    df['rsi'] = _ta.rsi(close_series, length=14)
                    bb_df = _ta.bbands(close_series, length=20, std=2.0)
                    if bb_df is not None and hasattr(bb_df, 'columns'):
                        upper = [c for c in bb_df.columns if str(c).startswith('BBU_')]
                        lower = [c for c in bb_df.columns if str(c).startswith('BBL_')]
                        if upper:
                            df['bollinger_hband'] = bb_df[upper[0]]
                        if lower:
                            df['bollinger_lband'] = bb_df[lower[0]]
                    if vol_series is not None:
                        df['vol_ma20'] = vol_series.rolling(20).mean()
                        df['vol_ratio'] = vol_series / df['vol_ma20']
                    # Derived basic engineered columns
                    if 'ema50' in df.columns:
                        df['price_vs_ema50'] = (close_series - df['ema50']) / df['ema50']
                    if 'ema200' in df.columns:
                        df['price_vs_ema200'] = (close_series - df['ema200']) / df['ema200']
                    if 'ema50' in df.columns and 'ema200' in df.columns:
                        df['ema_ratio'] = df['ema50'] / df['ema200']
                    if 'macd_diff' in df.columns and 'Close' in df.columns:
                        df['macd_signal_strength'] = pd.to_numeric(df['macd_diff'], errors='coerce') / close_series.replace(0, np.nan)
                        df['macd_trend'] = (pd.to_numeric(df['macd_diff'], errors='coerce') > 0).astype(int)
                    if 'bollinger_hband' in df.columns and 'bollinger_lband' in df.columns:
                        denom = (df['bollinger_hband'] - df['bollinger_lband'])
                        df['bb_pos'] = np.where(denom != 0, (close_series - df['bollinger_lband']) / denom, np.nan)
                    # Momentum approximations
                    df['rsi_overbought'] = (pd.to_numeric(df.get('rsi', pd.Series(np.nan, index=df.index)), errors='coerce') > 70).astype(int)
                    df['rsi_oversold'] = (pd.to_numeric(df.get('rsi', pd.Series(np.nan, index=df.index)), errors='coerce') < 30).astype(int)
                    if 'vol_ratio' in df.columns:
                        df['volume_momentum'] = df['vol_ratio'] - 1
                    # Trend/momentum scores
                    conds_trend = []
                    if 'ema50' in df.columns:
                        conds_trend.append(close_series > df['ema50'])
                    if 'ema200' in df.columns:
                        conds_trend.append(close_series > df['ema200'])
                    if 'macd_diff' in df.columns:
                        conds_trend.append(pd.to_numeric(df['macd_diff'], errors='coerce') > -0.1)
                    if 'ema50' in df.columns and 'ema200' in df.columns:
                        conds_trend.append(df['ema50'] > df['ema200'])
                    if conds_trend:
                        df['trend_score'] = np.mean(conds_trend, axis=0)
                    conds_mom = []
                    if 'bb_pos' in df.columns:
                        conds_mom.append(pd.to_numeric(df['bb_pos'], errors='coerce') >= 0.3)
                    if 'vol_ratio' in df.columns:
                        conds_mom.append(pd.to_numeric(df['vol_ratio'], errors='coerce') >= 0.8)
                    if 'rsi' in df.columns:
                        rsi_vals = pd.to_numeric(df['rsi'], errors='coerce')
                        conds_mom.append(rsi_vals > 30)
                        conds_mom.append(rsi_vals < 80)
                    if conds_mom:
                        df['momentum_score'] = np.mean(conds_mom, axis=0)
                    debug_steps['after_minimal_fallback_cols'] = list(df.columns)
            except Exception:
                pass
            # Merge financial data if provided
            if financial_data is not None and ticker is not None:
                df = self.merge_financial_data(df, financial_data, ticker)
            debug_steps['after_merge_cols'] = list(df.columns)
            
            # Prepare quantitative data
            df = self.prepare_quantitative_data(df)
            debug_steps['after_prepare_cols'] = list(df.columns)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            debug_steps['after_missing_cols'] = list(df.columns)
                      
            # Determine expected feature columns (prefer JSON)
            feature_cols = []
            try:
                import json
                features_json_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_columns.json')
                if os.path.exists(features_json_path):
                    with open(features_json_path, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list) and loaded:
                            feature_cols = loaded
            except Exception:
                feature_cols = []

            # Important: For prediction, skip aggressive reorder/rename to avoid dropping engineered columns.
            # We'll select features directly from the current frame.
            df_before_rename = df.copy()
            # print("TÉT2")
            # pd.set_option("display.max_columns", None)
            # print(data.head(10))
            # Determine feature columns strictly from feature_columns.json if present
            feature_cols = []
            try:
                # feature_cols may have been read above; if still empty, load again or fallback
                if not feature_cols:
                    import json
                    features_json_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_columns.json')
                    if os.path.exists(features_json_path):
                        with open(features_json_path, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                            if isinstance(loaded, list) and loaded:
                                feature_cols = loaded
                # If JSON missing, fall back to model metadata or pkls
                if not feature_cols and os.path.exists(Config.MODEL_PATH):
                    model = joblib.load(Config.MODEL_PATH)
                    if hasattr(model, 'feature_names_in_'):
                        feature_cols = list(getattr(model, 'feature_names_in_'))
                if not feature_cols:
                    fn_path = os.path.join(os.path.dirname(Config.MODEL_PATH), 'feature_names.pkl')
                    if os.path.exists(fn_path):
                        feature_cols = joblib.load(fn_path)
                if not feature_cols:
                    feature_cols = [c for c in self.FEATURE_ORDER if c in df.columns]
            except Exception:
                feature_cols = [c for c in self.FEATURE_ORDER if c in df.columns]

            # Ensure all expected features exist; create missing with 0.0, preserve exact order
            # Debug snapshot before filling zeros
            df_before_fill = df.copy()
            missing_list = [c for c in feature_cols if c not in df_before_fill.columns]

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[feature_cols].copy()
            debug_steps['selected_feature_cols'] = feature_cols

            # Scale features with saved scaler
            df = self.scale_features(df, fit_scaler=False)
            
            # Get the latest row for prediction
            latest_data = df.iloc[-1:].copy()

            # Save debug info to Streamlit session if available
            try:
                import streamlit as _st
                _st.session_state.prediction_feature_debug = {
                    'available_before_fill': list(df_before_fill.columns),
                    'missing_from_json': missing_list,
                    'feature_order_json': feature_cols,
                    'steps': debug_steps
                }
            except Exception:
                pass
            
            logger.info("Prepared prediction data")
            return latest_data
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return pd.DataFrame()
    
    def full_data_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete data preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Fully processed DataFrame
        """
        try:
            logger.info("🚀 Starting full data preprocessing pipeline...")
            
            # Step 1: Prepare quantitative data
            data = self.prepare_quantitative_data(df)
            
            # Step 2: Handle missing values
            data = self.handle_missing_values(data)
            
            # Step 3: Reorder and rename columns
            data = self.reorder_and_rename_columns(data)
            # print("TÉT")
            # print(data.columns)
            # Step 4: Remove rows with missing target variables
            target_cols = [col for col in data.columns if col.startswith('target_binary_')]
            if target_cols:
                data = data.dropna(subset=target_cols)
            
            logger.info(f"📊 Final processed data: {len(data):,} records")
            return data
            
        except Exception as e:
            logger.error(f"Error in full data preprocessing: {e}")
            return df