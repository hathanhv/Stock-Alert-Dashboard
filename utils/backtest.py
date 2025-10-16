"""
Backtesting utilities for trading strategy evaluation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

logger = logging.getLogger(__name__)

class Backtester:
    """Class to perform backtesting of trading strategies"""
    
    def __init__(self, initial_capital: float = None, commission_rate: float = None):
        """
        Initialize the backtester
        
        Args:
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate for trades
        """
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.commission_rate = commission_rate or Config.COMMISSION_RATE
        self.allow_short = False
        self.reset()
    
    def reset(self):
        """Reset backtesting state"""
        self.capital = self.initial_capital
        self.position = 0  # Number of shares held
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.peak_capital = self.initial_capital
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        Calculate commission for a trade
        
        Args:
            trade_value: Value of the trade
            
        Returns:
            Commission amount
        """
        return abs(trade_value) * self.commission_rate
    
    def execute_trade(self, signal: str, price: float, date: datetime, 
                     quantity: int = None) -> Dict:
        """
        Execute a trade based on signal
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            price: Current stock price
            date: Trade date
            quantity: Number of shares to trade (if None, uses all available capital)
            
        Returns:
            Trade execution details
        """
        try:
            trade = {
                'date': date,
                'signal': signal,
                'price': price,
                'quantity': 0,
                'value': 0,
                'commission': 0,
                'capital_before': self.capital,
                'position_before': self.position
            }
            
            if signal == 'BUY' and self.position == 0:
                # Buy signal and no current position
                if quantity is None:
                    # Use all available capital
                    quantity = int(self.capital / price)
                
                trade_value = quantity * price
                commission = self.calculate_commission(trade_value)
                
                if trade_value + commission <= self.capital:
                    self.position = quantity
                    self.capital -= (trade_value + commission)
                    
                    trade.update({
                        'quantity': quantity,
                        'value': -trade_value,
                        'commission': commission,
                        'executed': True
                    })
                else:
                    trade['executed'] = False
                    trade['reason'] = 'Insufficient capital'
            
            elif signal == 'SELL' and self.position > 0:
                # Sell signal and have position
                quantity = self.position
                trade_value = quantity * price
                commission = self.calculate_commission(trade_value)
                
                self.capital += (trade_value - commission)
                
                trade.update({
                    'quantity': -quantity,
                    'value': trade_value,
                    'commission': commission,
                    'executed': True
                })
                
                self.position = 0
            elif signal == 'SELL' and self.position == 0 and self.allow_short:
                # Open short position: borrow shares and sell now
                if quantity is None:
                    quantity = int(self.capital / price)
                trade_value = quantity * price
                commission = self.calculate_commission(trade_value)
                # Receive proceeds from short sale
                self.capital += (trade_value - commission)
                self.position = -quantity
                trade.update({
                    'quantity': -quantity,
                    'value': trade_value,
                    'commission': commission,
                    'executed': True,
                    'short_open': True
                })
            elif signal == 'BUY' and self.position < 0 and self.allow_short:
                # Cover short position
                quantity = -self.position
                trade_value = quantity * price
                commission = self.calculate_commission(trade_value)
                # Pay to buy back shares
                self.capital -= (trade_value + commission)
                trade.update({
                    'quantity': quantity,
                    'value': -trade_value,
                    'commission': commission,
                    'executed': True,
                    'short_close': True
                })
                self.position = 0
            
            else:
                # HOLD or invalid signal
                trade['executed'] = False
                trade['reason'] = f'No action needed (signal: {signal}, position: {self.position})'
            
            trade.update({
                'capital_after': self.capital,
                'position_after': self.position,
                'total_value': self.capital + (self.position * price)
            })
            
            self.trades.append(trade)
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'error': str(e)}
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    start_date: datetime = None, end_date: datetime = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data (OHLCV)
            signals: Trading signals series
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results dictionary
        """
        try:
            self.reset()
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Ensure signals are aligned with data
            signals = signals.reindex(data.index, method='ffill')
            
            # Run backtest
            for date, row in data.iterrows():
                price = row['close']
                signal = signals.loc[date] if date in signals.index else 'HOLD'
                
                # Execute trade
                self.execute_trade(signal, price, date)
                
                # Update equity curve
                total_value = self.capital + (self.position * price)
                self.equity_curve.append({
                    'date': date,
                    'capital': self.capital,
                    'position': self.position,
                    'price': price,
                    'total_value': total_value
                })
                
                # Update drawdown
                if total_value > self.peak_capital:
                    self.peak_capital = total_value
                else:
                    drawdown = (self.peak_capital - total_value) / self.peak_capital
                    self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Calculate performance metrics
            results = self.calculate_performance_metrics(data)
            
            logger.info(f"Backtest completed: {len(self.trades)} trades, "
                       f"Final value: {results['final_value']:,.0f} VND")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Args:
            data: Historical price data
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.equity_curve:
                return {'error': 'No equity curve data'}
            
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # Calculate daily returns
            equity_df['returns'] = equity_df['total_value'].pct_change()
            self.daily_returns = equity_df['returns'].dropna()
            
            # Basic metrics
            initial_value = self.initial_capital
            final_value = equity_df['total_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Risk metrics
            if len(self.daily_returns) > 0:
                volatility = self.daily_returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (self.daily_returns.mean() * 252) / volatility if volatility > 0 else 0
                max_drawdown = self.max_drawdown
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Trade analysis
            executed_trades = [t for t in self.trades if t.get('executed', False)]
            buy_trades = [t for t in executed_trades if t['signal'] == 'BUY']
            sell_trades = [t for t in executed_trades if t['signal'] == 'SELL']
            
            # Win/loss ratio
            if len(sell_trades) > 0:
                profitable_trades = sum(1 for t in sell_trades if t['value'] > 0)
                win_rate = profitable_trades / len(sell_trades)
            else:
                win_rate = 0
            
            # Average trade metrics
            avg_trade_return = 0
            if len(executed_trades) > 0:
                trade_returns = []
                for i in range(0, len(executed_trades) - 1, 2):
                    if i + 1 < len(executed_trades):
                        buy_trade = executed_trades[i]
                        sell_trade = executed_trades[i + 1]
                        if buy_trade['signal'] == 'BUY' and sell_trade['signal'] == 'SELL':
                            trade_return = (sell_trade['value'] - abs(buy_trade['value'])) / abs(buy_trade['value'])
                            trade_returns.append(trade_return)
                
                if trade_returns:
                    avg_trade_return = np.mean(trade_returns)
            
            results = {
                'initial_capital': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': len(executed_trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'avg_trade_return': avg_trade_return,
                'avg_trade_return_pct': avg_trade_return * 100,
                'equity_curve': equity_df,
                'trades': executed_trades
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def plot_equity_curve(self, results: Dict, save_path: str = None):
        """
        Plot equity curve and performance charts
        
        Args:
            results: Backtest results dictionary
            save_path: Path to save the plot
        """
        try:
            if 'equity_curve' not in results:
                logger.error("No equity curve data available for plotting")
                return
            
            equity_df = results['equity_curve']
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtest Results', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(equity_df.index, equity_df['total_value'], label='Portfolio Value', linewidth=2)
            axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value (VND)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Daily returns
            if 'returns' in equity_df.columns:
                axes[0, 1].hist(equity_df['returns'].dropna(), bins=30, alpha=0.7, color='skyblue')
                axes[0, 1].set_title('Daily Returns Distribution')
                axes[0, 1].set_xlabel('Daily Return')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Drawdown
            peak = equity_df['total_value'].expanding().max()
            drawdown = (equity_df['total_value'] - peak) / peak
            axes[1, 0].fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Position over time
            axes[1, 1].plot(equity_df.index, equity_df['position'], label='Position', linewidth=2)
            axes[1, 1].set_title('Position Over Time')
            axes[1, 1].set_ylabel('Shares Held')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
    
    def generate_trade_report(self, results: Dict) -> pd.DataFrame:
        """
        Generate detailed trade report
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            DataFrame with trade details
        """
        try:
            if 'trades' not in results:
                return pd.DataFrame()
            
            trades_df = pd.DataFrame(results['trades'])
            
            # Add additional columns
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df['trade_type'] = trades_df['signal'].apply(
                lambda x: 'Entry' if x == 'BUY' else 'Exit' if x == 'SELL' else 'Hold'
            )
            
            return trades_df
            
        except Exception as e:
            logger.error(f"Error generating trade report: {e}")
            return pd.DataFrame()
    
    def compare_with_benchmark(self, results: Dict, benchmark_data: pd.DataFrame) -> Dict:
        """
        Compare strategy performance with benchmark
        
        Args:
            results: Backtest results dictionary
            benchmark_data: Benchmark price data
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            if 'equity_curve' not in results:
                return {'error': 'No equity curve data available'}
            
            equity_df = results['equity_curve']
            
            # Align benchmark data with equity curve
            benchmark_aligned = benchmark_data.reindex(equity_df.index, method='ffill')
            
            if benchmark_aligned.empty:
                return {'error': 'No benchmark data available for comparison'}
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_aligned['close'].pct_change()
            strategy_returns = equity_df['returns']
            
            # Calculate metrics
            benchmark_total_return = (benchmark_aligned['close'].iloc[-1] / benchmark_aligned['close'].iloc[0]) - 1
            strategy_total_return = results['total_return']
            
            # Beta calculation (simplified)
            if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
                correlation = strategy_returns.corr(benchmark_returns)
                strategy_vol = strategy_returns.std()
                benchmark_vol = benchmark_returns.std()
                beta = correlation * (strategy_vol / benchmark_vol) if benchmark_vol > 0 else 0
            else:
                beta = 0
            
            comparison = {
                'strategy_return': strategy_total_return,
                'benchmark_return': benchmark_total_return,
                'excess_return': strategy_total_return - benchmark_total_return,
                'strategy_volatility': results['volatility'],
                'benchmark_volatility': benchmark_returns.std() * np.sqrt(252),
                'strategy_sharpe': results['sharpe_ratio'],
                'benchmark_sharpe': (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)) if benchmark_returns.std() > 0 else 0,
                'beta': beta,
                'correlation': correlation if 'correlation' in locals() else 0
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with benchmark: {e}")
            return {'error': str(e)}
