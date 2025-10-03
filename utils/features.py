"""
Technical indicators and feature engineering utilities using FiinIndicator
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from config import Config
import pandas_ta as ta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class to calculate technical indicators using FiinIndicator"""
    
    def __init__(self, fi_indicator=None):
        """Initialize the technical indicators calculator"""
        # self.fi = fi_indicator
        if fi_indicator is None:
            self.fi = ta
        else:
            self.fi = fi_indicator
        self.rsi_period = Config.RSI_PERIOD
        self.ema_periods = Config.EMA_PERIODS
        self.sma_periods = Config.SMA_PERIODS
        self.bollinger_period = Config.BOLLINGER_PERIOD
        self.bollinger_std = Config.BOLLINGER_STD
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Calculate RSI using FiinIndicator
        
        Args:
            data: DataFrame with OHLCV data
            period: RSI period (default from config)
            
        Returns:
            DataFrame with RSI column added
        """
        if period is None:
            period = self.rsi_period
        
        try:
            result = data.copy()
            # Ensure numeric types for TA calculations
            for col in ["Open", "High", "Low", "Close", "MatchVolume"]:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
            
            if self.fi is not None:
                # result["rsi"] = result.groupby("Ticker")["Close"].transform(
                #     lambda x: self.fi.rsi(x, window=period)
                # )
                result["rsi"] = result.groupby("Ticker")["Close"].transform(
                    lambda x: ta.rsi(x, length=period)
                )
            else:
                # Fallback calculation if FiinIndicator not available
                result["rsi"] = np.nan
                logger.warning("FiinIndicator not available, using fallback RSI calculation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            result = data.copy()
            result["rsi"] = np.nan
            return result
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate MFI (Money Flow Index) using FiinIndicator
        
        Args:
            data: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            DataFrame with MFI column added
        """
        try:
            result = data.copy()
            # Ensure numeric types
            for col in ["High", "Low", "Close", "MatchVolume"]:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
            
            if self.fi is not None:
                # result["mfi"] = result.groupby("Ticker").apply(
                #     lambda g: self.fi.mfi(g["High"], g["Low"], g["Close"], g["MatchVolume"], window=period)
                # ).reset_index(level=0, drop=True)
                mfi_vals = result.groupby("Ticker").apply(
                    lambda g: ta.mfi(g["High"], g["Low"], g["Close"], g["MatchVolume"], length=period)
                )
                if isinstance(mfi_vals, pd.DataFrame):
                    mfi_vals = mfi_vals.iloc[:, 0]
                if isinstance(mfi_vals.index, pd.MultiIndex):
                    mfi_vals.index = mfi_vals.index.droplevel(0)
                result["mfi"] = mfi_vals.values
            else:
                result["mfi"] = np.nan
                logger.warning("FiinIndicator not available, using fallback MFI calculation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MFI: {e}")
            result = data.copy()
            result["mfi"] = np.nan
            return result
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD using FiinIndicator
        
        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD columns added
        """
        try:
            result = data.copy()
            
            if self.fi is not None:
                # result['macd'] = result.groupby("Ticker")["Close"].transform(
                #     lambda x: self.fi.macd(x, window_fast=fast, window_slow=slow)
                # )
                # result['macd_signal'] = result.groupby("Ticker")["Close"].transform(
                #     lambda x: self.fi.macd_signal(x, window_fast=fast, window_slow=slow, window_sign=signal)
                # )
                # result['macd_diff'] = result.groupby("Ticker")["Close"].transform(
                #     lambda x: self.fi.macd_diff(x, window_fast=fast, window_slow=slow, window_sign=signal)
                # )
                macd_df = result.groupby("Ticker").apply(
                    lambda g: ta.macd(g["Close"], fast=fast, slow=slow, signal=signal)
                )
                if isinstance(macd_df.index, pd.MultiIndex):
                    macd_df.index = macd_df.index.droplevel(0)
                result["macd"] = macd_df[f"MACD_{fast}_{slow}_{signal}"].values
                result["macd_signal"] = macd_df[f"MACDs_{fast}_{slow}_{signal}"].values
                result["macd_diff"] = macd_df[f"MACDh_{fast}_{slow}_{signal}"].values
            else:
                result['macd'] = np.nan
                result['macd_signal'] = np.nan
                result['macd_diff'] = np.nan
                logger.warning("FiinIndicator not available, using fallback MACD calculation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            result = data.copy()
            result['macd'] = np.nan
            result['macd_signal'] = np.nan
            result['macd_diff'] = np.nan
            return result
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = None, std: float = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using FiinIndicator
        
        Args:
            data: DataFrame with OHLCV data
            period: BB period (default from config)
            std: Standard deviation multiplier (default from config)
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        if period is None:
            period = self.bollinger_period
        if std is None:
            std = self.bollinger_std
        
        try:
            result = data.copy()
            
            if self.fi is not None:
                # result['bollinger_hband'] = result.groupby("Ticker")['Close'].transform(
                #     lambda x: self.fi.bollinger_hband(x, window=period, window_dev=std)
                # )
                # result['bollinger_lband'] = result.groupby("Ticker")['Close'].transform(
                #     lambda x: self.fi.bollinger_lband(x, window=period, window_dev=std)
                # )
                bb_df = result.groupby("Ticker").apply(
                    lambda g: ta.bbands(g["Close"], length=period, std=std)
                )
                if isinstance(bb_df.index, pd.MultiIndex):
                    bb_df.index = bb_df.index.droplevel(0)
                # Find upper/middle/lower band columns robustly
                upper_candidates = [c for c in bb_df.columns if c.startswith(f"BBU_{period}_") or c.startswith(f"BBU_{period}")]
                middle_candidates = [c for c in bb_df.columns if c.startswith(f"BBM_{period}_") or c.startswith(f"BBM_{period}")]
                lower_candidates = [c for c in bb_df.columns if c.startswith(f"BBL_{period}_") or c.startswith(f"BBL_{period}")]
                if not upper_candidates or not lower_candidates:
                    # Fallback: any BBU_/BBL_ if period-embedded names differ
                    if not upper_candidates:
                        upper_candidates = [c for c in bb_df.columns if c.startswith("BBU_")]
                    if not middle_candidates:
                        middle_candidates = [c for c in bb_df.columns if c.startswith("BBM_")]
                    if not lower_candidates:
                        lower_candidates = [c for c in bb_df.columns if c.startswith("BBL_")]
                if upper_candidates and lower_candidates:
                    upper_vals = bb_df[upper_candidates[0]].values
                    lower_vals = bb_df[lower_candidates[0]].values
                    result["bollinger_hband"] = upper_vals
                    result["bollinger_lband"] = lower_vals
                    if middle_candidates:
                        result["bollinger_mband"] = bb_df[middle_candidates[0]].values
                    else:
                        result["bollinger_mband"] = (result["bollinger_hband"] + result["bollinger_lband"]) / 2
                else:
                    result["bollinger_hband"] = np.nan
                    result["bollinger_lband"] = np.nan
                    result["bollinger_mband"] = np.nan

                # Calculate Bollinger position
                denom = (result['bollinger_hband'] - result['bollinger_lband'])
                result['bb_pos'] = np.where(denom != 0, (result['Close'] - result['bollinger_lband']) / denom, np.nan)

                # Aliases for dashboard
                result['BB Upper'] = result['bollinger_hband']
                result['BB Middle'] = result['bollinger_mband']
                result['BB Lower'] = result['bollinger_lband']
            else:
                result['bollinger_hband'] = np.nan
                result['bollinger_lband'] = np.nan
                result['bb_pos'] = np.nan
                logger.warning("FiinIndicator not available, using fallback Bollinger Bands calculation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            result = data.copy()
            result['bollinger_hband'] = np.nan
            result['bollinger_lband'] = np.nan
            result['bb_pos'] = np.nan
            return result
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate EMA using FiinIndicator
        
        Args:
            data: DataFrame with OHLCV data
            periods: List of EMA periods (default from config)
            
        Returns:
            DataFrame with EMA columns added
        """
        if periods is None:
            periods = self.ema_periods
        
        try:
            result = data.copy()
            
            if self.fi is not None:
                # for period in periods:
                #     result[f"ema{period}"] = result.groupby("Ticker")["Close"].transform(
                #         lambda x: self.fi.ema(x, window=period)
                #     )
                for p in periods:
                    result[f"ema{p}"] = result.groupby("Ticker")["Close"].transform(
                        lambda x: ta.ema(x, length=p)
                    )
            else:
                for period in periods:
                    result[f"ema{period}"] = np.nan
                logger.warning("FiinIndicator not available, using fallback EMA calculation")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            result = data.copy()
            for period in periods:
                result[f"ema{period}"] = np.nan
            return result
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators added
        """
        try:
            result = data.copy()
            
            # Volume moving average
            result['vol_ma20'] = result.groupby("Ticker")['MatchVolume'].transform(
                lambda x: x.rolling(20).mean()
            )
            
            # Volume ratio
            result['vol_ratio'] = result['MatchVolume'] / result['vol_ma20']
            # Alias for dashboard
            result['Volume Ratio'] = result['vol_ratio']
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            result = data.copy()
            result['vol_ma20'] = np.nan
            result['vol_ratio'] = np.nan
            return result
    
    def create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for ML model
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with target variables added
        """
        try:
            result = data.copy()
            
            # Sort by ticker and timestamp
            result = result.sort_values(['Ticker', 'TradingDate']).reset_index(drop=True)
            
            # Calculate returns for different periods
            for period in [5, 10, 20]:
                result[f'return_{period}d'] = result.groupby('Ticker')['Close'].pct_change(periods=period).shift(-period)
                
                # Binary target (increase/decrease)
                result[f'target_binary_{period}d'] = (result[f'return_{period}d'] > 0).astype(int)
                
                # Multi-class target
                result[f'target_multi_{period}d'] = pd.cut(
                    result[f'return_{period}d'],
                    bins=[-np.inf, -0.05, 0.05, np.inf],
                    labels=['Giảm', 'Sideway', 'Tăng']
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return data
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical features
        
        Args:
            data: DataFrame with OHLCV and indicators data
            
        Returns:
            DataFrame with technical features added
        """
        try:
            result = data.copy()
            
            # === TREND INDICATORS ===
            # Price vs EMA ratios
            result['price_vs_ema50'] = (result['Close'] - result['ema50']) / result['ema50']
            result['price_vs_ema200'] = (result['Close'] - result['ema200']) / result['ema200']
            result['ema_ratio'] = result['ema50'] / result['ema200']
            
            # MACD signals
            result['macd_signal_strength'] = result['macd_diff'] / result['Close']
            result['macd_trend'] = (result['macd_diff'] > 0).astype(int)
            
            # Trend Score (0-1)
            trend_conditions = [
                result['Close'] > result['ema50'],
                result['Close'] > result['ema200'],
                result['macd_diff'] > -0.1,
                result['ema50'] > result['ema200']
            ]
            result['trend_score'] = np.mean(trend_conditions, axis=0)
            
            # === MOMENTUM INDICATORS ===
            # RSI levels
            result['rsi_overbought'] = (result['rsi'] > 70).astype(int)
            result['rsi_oversold'] = (result['rsi'] < 30).astype(int)
            result['rsi_neutral'] = ((result['rsi'] >= 30) & (result['rsi'] <= 70)).astype(int)
            
            # MFI momentum
            result['mfi_momentum'] = (result['mfi'] - 50) / 50
            
            # Volume momentum
            result['volume_momentum'] = result['vol_ratio'] - 1
            
            # Momentum Score (0-1)
            momentum_conditions = [
                result['bb_pos'] >= 0.3,
                result['vol_ratio'] >= 0.8,
                result['rsi'] > 30,
                result['rsi'] < 80
            ]
            result['momentum_score'] = np.mean(momentum_conditions, axis=0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return data
    
    def create_value_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create value-based features from fundamental data
        
        Args:
            data: DataFrame with fundamental ratios
            
        Returns:
            DataFrame with value features added
        """
        try:
            result = data.copy()
            
            # P/E ratio normalized
            result['pe_normalized'] = np.where(result['PE'] > 0, 1 / (1 + result['PE']/10), 0)
            
            # P/B ratio normalized
            result['pb_normalized'] = np.where(result['PB'] > 0, 1 / (1 + result['PB']/2), 0)
            
            # Growth metrics
            result['growth_score'] = (
                (result['RevenueGrowth'] > 0).astype(int) +
                (result['ProfitGrowth'] > 0).astype(int)
            ) / 2
            
            # ROE quality
            result['roe_quality'] = np.where(result['ROE'] > 0.15, 2,
                                  np.where(result['ROE'] > 0.10, 1, 0)) / 2
            
            # Value Score (0-1)
            value_conditions = [
                result['PE'] <= 40,
                result['PB'] <= 6,
                result['ROE'] >= 0.10,
                result['EPS'] > 0,
                (result['RevenueGrowth'] > 0) | (result['ProfitGrowth'] > 0)
            ]
            result['value_score'] = np.mean(value_conditions, axis=0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating value features: {e}")
            return data
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataset
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        try:
            result = data.copy()
            
            # Ensure we have the required columns
            required_cols = ['Ticker', 'TradingDate', 'Open', 'High', 'Low', 'Close', 'MatchVolume']
            for col in required_cols:
                if col not in result.columns:
                    logger.error(f"Missing required column: {col}")
                    return result
            
            # Sort data
            result = result.sort_values(['Ticker', 'TradingDate']).reset_index(drop=True)
            # Ensure numeric dtypes before passing to TA
            for col in ['Open','High','Low','Close','MatchVolume']:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
            
            # Calculate basic indicators
            result = self.calculate_rsi(result)
            result = self.calculate_mfi(result)
            result = self.calculate_macd(result)
            result = self.calculate_bollinger_bands(result)
            result = self.calculate_ema(result, [50, 200])
            result = self.calculate_volume_indicators(result)
            
            # Create target variables
            result = self.create_target_variables(result)
            
            # Create technical features
            result = self.create_technical_features(result)
            
            # Create value features (if fundamental data available)
            fundamental_cols = ['PB', 'PE', 'ROE', 'EPS', 'RevenueGrowth', 'ProfitGrowth']
            if all(col in result.columns for col in fundamental_cols):
                result = self.create_value_features(result)
            
            logger.info(f"Calculated technical indicators for {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return data
    
    def prepare_features_for_model(self, data: pd.DataFrame, target_period: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML model training
        
        Args:
            data: DataFrame with all indicators
            target_period: Target prediction period in days
            
        Returns:
            Tuple of (features, target)
        """
        try:
            # Define feature columns
            trend_features = [
                'price_vs_ema50', 'price_vs_ema200', 'ema_ratio',
                'macd_signal_strength', 'macd_trend', 'trend_score'
            ]
            
            momentum_features = [
                'rsi', 'mfi', 'bb_pos', 'vol_ratio',
                'rsi_overbought', 'rsi_oversold', 'mfi_momentum',
                'volume_momentum', 'momentum_score'
            ]
            
            value_features = [
                'PE', 'PB', 'ROE', 'EPS', 'RevenueGrowth', 'ProfitGrowth',
                'pe_normalized', 'pb_normalized', 'growth_score',
                'roe_quality', 'value_score'
            ]
            
            # Combine all features
            feature_columns = trend_features + momentum_features + value_features
            
            # Filter available features
            available_features = [f for f in feature_columns if f in data.columns]
            
            # Get features and target
            X = data[available_features].copy()
            y = data[f'target_binary_{target_period}d'].copy()
            
            # Remove rows with missing target
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()