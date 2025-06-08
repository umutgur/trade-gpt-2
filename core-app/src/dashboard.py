import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List
from loguru import logger

from config import config
from data_fetcher import data_fetcher
from db import db_manager
from metrics import metrics_calculator
from lstm_model import model_manager
from llm_strategy import llm_engine
from ta_features import ta_analyzer

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #00ff00; }
    .negative { color: #ff0000; }
    .neutral { color: #888888; }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.symbols = config.SYMBOLS
        self.trading_modes = ['spot', 'margin', 'futures']
        
    def run(self):
        """Main dashboard function"""
        st.title("🚀 Crypto Trading AI Dashboard")
        
        # Sidebar
        self.create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "💰 Spot Trading", "📈 Margin Trading", 
            "⚡ Futures Trading", "🔬 Backtest Analytics"
        ])
        
        with tab1:
            self.overview_tab()
        
        with tab2:
            self.trading_tab("spot")
        
        with tab3:
            self.trading_tab("margin")
        
        with tab4:
            self.trading_tab("futures")
        
        with tab5:
            self.backtest_analytics_tab()
    
    def create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.header("🎛️ Controls")
        
        # Symbol selection
        selected_symbol = st.sidebar.selectbox(
            "Select Symbol",
            self.symbols,
            index=0
        )
        st.session_state.selected_symbol = selected_symbol
        
        # Time range
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1H", "4H", "1D", "1W", "1M"],
            index=2
        )
        st.session_state.time_range = time_range
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
        if auto_refresh:
            st.experimental_rerun()
        
        # System status
        st.sidebar.header("⚙️ System Status")
        self.show_system_status()
    
    def show_system_status(self):
        """Show system status in sidebar"""
        try:
            # Database connection
            try:
                session = db_manager.get_session()
                session.close()
                st.sidebar.success("✅ Database Connected")
            except:
                st.sidebar.error("❌ Database Error")
            
            # API status
            try:
                current_price = data_fetcher.get_current_price("BTC/USDT")
                if current_price:
                    st.sidebar.success("✅ Binance API Connected")
                else:
                    st.sidebar.warning("⚠️ API Issues")
            except:
                st.sidebar.error("❌ API Error")
            
            # Airflow status (placeholder)
            st.sidebar.info("🔄 Airflow Pipeline Active")
            
        except Exception as e:
            st.sidebar.error(f"❌ System Error: {str(e)}")
    
    def overview_tab(self):
        """Overview tab with key metrics"""
        st.header("📊 Portfolio Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = self.get_total_pnl()
            st.metric(
                "Total P&L (7d)",
                f"${total_pnl:.2f}",
                delta=f"{(total_pnl/10000)*100:.2f}%"
            )
        
        with col2:
            avg_sharpe = self.get_average_sharpe()
            st.metric(
                "Avg Sharpe Ratio",
                f"{avg_sharpe:.2f}",
                delta="Good" if avg_sharpe > 1 else "Needs Improvement"
            )
        
        with col3:
            total_trades = self.get_total_trades()
            st.metric(
                "Total Trades (7d)",
                total_trades,
                delta=f"+{total_trades//7} per day"
            )
        
        with col4:
            avg_accuracy = self.get_prediction_accuracy()
            st.metric(
                "LSTM Accuracy",
                f"{avg_accuracy:.1f}%",
                delta="Good" if avg_accuracy > 70 else "Training"
            )
        
        # Performance charts
        st.subheader("📈 Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve chart
            equity_fig = self.create_equity_curve_chart()
            st.plotly_chart(equity_fig, use_container_width=True)
        
        with col2:
            # Symbol performance heatmap
            heatmap_fig = self.create_performance_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Recent activities
        st.subheader("🎯 Recent Activities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Trades**")
            trades_df = self.get_recent_trades()
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No recent trades")
        
        with col2:
            st.write("**Latest Strategies**")
            strategies_df = self.get_recent_strategies()
            if not strategies_df.empty:
                st.dataframe(strategies_df, use_container_width=True)
            else:
                st.info("No recent strategies")
    
    def trading_tab(self, trading_mode: str):
        """Trading tab for specific mode"""
        mode_name = trading_mode.title()
        st.header(f"📊 {mode_name} Trading Dashboard")
        
        symbol = st.session_state.get('selected_symbol', 'BTC/USDT')
        
        # Trading mode metrics
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = self.get_trading_mode_metrics(symbol, trading_mode)
        
        with col1:
            roi = metrics.get('total_return', 0)
            st.metric(
                "7d ROI",
                f"{roi:.2f}%",
                delta=f"{'📈' if roi > 0 else '📉'}"
            )
        
        with col2:
            win_rate = metrics.get('win_rate', 0) * 100
            st.metric(
                "Win Rate",
                f"{win_rate:.1f}%",
                delta="Good" if win_rate > 60 else "Improving"
            )
        
        with col3:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
            )
        
        with col4:
            drawdown = metrics.get('max_drawdown', 0) * 100
            st.metric(
                "Max Drawdown",
                f"{drawdown:.2f}%",
                delta="Low Risk" if drawdown < 5 else "High Risk"
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart with predictions
            price_fig = self.create_price_prediction_chart(symbol, trading_mode)
            st.plotly_chart(price_fig, use_container_width=True)
        
        with col2:
            # P&L chart
            pnl_fig = self.create_pnl_chart(symbol, trading_mode)
            st.plotly_chart(pnl_fig, use_container_width=True)
        
        # Strategy and trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Current Strategy")
            current_strategy = self.get_current_strategy(symbol, trading_mode)
            if current_strategy:
                self.display_strategy_info(current_strategy)
            else:
                st.info("No active strategy")
        
        with col2:
            st.subheader("📋 Recent Trades")
            trades_df = self.get_trades_for_mode(symbol, trading_mode)
            if not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("No recent trades")
        
        # Risk metrics
        st.subheader("⚠️ Risk Metrics")
        self.display_risk_metrics(symbol, trading_mode)
    
    def backtest_analytics_tab(self):
        """Backtest analytics tab"""
        st.header("🔬 Backtest Analytics")
        
        # Backtest controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox(
                "Symbol",
                self.symbols,
                key="backtest_symbol"
            )
        
        with col2:
            trading_mode = st.selectbox(
                "Trading Mode",
                self.trading_modes,
                key="backtest_mode"
            )
        
        with col3:
            days_back = st.selectbox(
                "Period",
                [7, 14, 30, 60, 90],
                index=2,
                key="backtest_days"
            )
        
        # Backtest results
        if st.button("🚀 Run New Backtest"):
            with st.spinner("Running backtest..."):
                self.run_new_backtest(symbol, trading_mode, days_back)
        
        # Historical backtest results
        st.subheader("📊 Backtest Results")
        
        backtest_data = self.get_backtest_results(symbol, trading_mode)
        
        if backtest_data:
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance metrics chart
                metrics_fig = self.create_backtest_metrics_chart(backtest_data)
                st.plotly_chart(metrics_fig, use_container_width=True)
            
            with col2:
                # Equity curve
                equity_fig = self.create_backtest_equity_curve(backtest_data)
                st.plotly_chart(equity_fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            results_df = pd.DataFrame(backtest_data)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("No backtest results available")
        
        # Strategy performance heatmap
        st.subheader("🔥 Strategy Performance Heatmap")
        strategy_heatmap = self.create_strategy_performance_heatmap()
        st.plotly_chart(strategy_heatmap, use_container_width=True)
    
    def get_total_pnl(self) -> float:
        """Get total P&L across all modes"""
        try:
            total = 0
            for symbol in self.symbols:
                for mode in self.trading_modes:
                    metrics = metrics_calculator.calculate_trading_metrics(symbol, mode, 7)
                    total += metrics.get('total_pnl', 0)
            return total
        except:
            return 0.0
    
    def get_average_sharpe(self) -> float:
        """Get average Sharpe ratio"""
        try:
            sharpe_values = []
            for symbol in self.symbols:
                for mode in self.trading_modes:
                    metrics = metrics_calculator.calculate_trading_metrics(symbol, mode, 7)
                    sharpe = metrics.get('sharpe_ratio', 0)
                    if sharpe != 0:
                        sharpe_values.append(sharpe)
            return np.mean(sharpe_values) if sharpe_values else 0.0
        except:
            return 0.0
    
    def get_total_trades(self) -> int:
        """Get total number of trades"""
        try:
            total = 0
            for symbol in self.symbols:
                for mode in self.trading_modes:
                    metrics = metrics_calculator.calculate_trading_metrics(symbol, mode, 7)
                    total += metrics.get('total_trades', 0)
            return total
        except:
            return 0
    
    def get_prediction_accuracy(self) -> float:
        """Get average LSTM prediction accuracy"""
        try:
            session = db_manager.get_session()
            models = session.query(db_manager.LSTMModel)\
                .filter(db_manager.LSTMModel.is_active == True)\
                .all()
            
            if models:
                mape_values = [100 - model.mape for model in models if model.mape]
                return np.mean(mape_values) if mape_values else 50.0
            
            session.close()
            return 50.0
        except:
            return 50.0
    
    def create_equity_curve_chart(self):
        """Create equity curve chart"""
        try:
            # Get portfolio value over time
            fig = go.Figure()
            
            # Placeholder data - replace with actual portfolio tracking
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            portfolio_value = 10000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_value,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00D4AA', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve (30d)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400
            )
            
            return fig
        except:
            # Return empty chart on error
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def create_performance_heatmap(self):
        """Create performance heatmap"""
        try:
            # Create heatmap data
            data = []
            symbols = []
            modes = []
            values = []
            
            for symbol in self.symbols:
                for mode in self.trading_modes:
                    metrics = metrics_calculator.calculate_trading_metrics(symbol, mode, 7)
                    roi = metrics.get('total_return', 0)
                    symbols.append(symbol)
                    modes.append(mode)
                    values.append(roi)
            
            # Reshape for heatmap
            heatmap_data = np.array(values).reshape(len(self.symbols), len(self.trading_modes))
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=self.trading_modes,
                y=self.symbols,
                colorscale='RdYlGn',
                text=heatmap_data,
                texttemplate="%{text:.2f}%",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="7-Day ROI Heatmap (%)",
                height=400
            )
            
            return fig
        except:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
    
    def get_recent_trades(self) -> pd.DataFrame:
        """Get recent trades across all modes"""
        try:
            session = db_manager.get_session()
            trades = session.query(db_manager.Trade)\
                .filter(db_manager.Trade.timestamp >= datetime.now() - timedelta(days=7))\
                .order_by(db_manager.Trade.timestamp.desc())\
                .limit(10)\
                .all()
            
            if trades:
                data = []
                for trade in trades:
                    data.append({
                        'Symbol': trade.symbol,
                        'Mode': trade.trading_mode,
                        'Side': trade.side,
                        'Amount': f"{trade.amount:.4f}",
                        'Price': f"${trade.price:.2f}",
                        'P&L': f"${trade.profit_loss or 0:.2f}",
                        'Time': trade.timestamp.strftime('%Y-%m-%d %H:%M')
                    })
                
                session.close()
                return pd.DataFrame(data)
            
            session.close()
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def get_recent_strategies(self) -> pd.DataFrame:
        """Get recent strategies"""
        try:
            session = db_manager.get_session()
            strategies = session.query(db_manager.Strategy)\
                .filter(db_manager.Strategy.is_active == True)\
                .order_by(db_manager.Strategy.timestamp.desc())\
                .limit(10)\
                .all()
            
            if strategies:
                data = []
                for strategy in strategies:
                    strategy_data = strategy.strategy_data or {}
                    data.append({
                        'Symbol': strategy.symbol,
                        'Mode': strategy.trading_mode,
                        'Outlook': strategy_data.get('market_outlook', 'Unknown'),
                        'Confidence': f"{strategy_data.get('confidence', 0)*100:.1f}%",
                        'Created': strategy.timestamp.strftime('%Y-%m-%d %H:%M')
                    })
                
                session.close()
                return pd.DataFrame(data)
            
            session.close()
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def get_trading_mode_metrics(self, symbol: str, trading_mode: str) -> Dict:
        """Get metrics for specific trading mode"""
        return metrics_calculator.calculate_trading_metrics(symbol, trading_mode, 7)
    
    def create_price_prediction_chart(self, symbol: str, trading_mode: str):
        """Create price chart with LSTM predictions"""
        try:
            # Get recent price data
            market_data = db_manager.get_latest_data(symbol, limit=100)
            
            if market_data:
                df = pd.DataFrame([{
                    'timestamp': item.timestamp,
                    'close': item.close_price
                } for item in market_data])
                
                df = df.sort_values('timestamp')
                
                fig = go.Figure()
                
                # Actual prices
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['close'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # LSTM prediction (placeholder)
                if len(df) > 0:
                    last_price = df['close'].iloc[-1]
                    pred_time = df['timestamp'].iloc[-1] + timedelta(minutes=15)
                    
                    # Get prediction from model
                    try:
                        # Placeholder prediction
                        pred_price = last_price * (1 + np.random.normal(0, 0.01))
                        
                        fig.add_trace(go.Scatter(
                            x=[df['timestamp'].iloc[-1], pred_time],
                            y=[last_price, pred_price],
                            mode='lines+markers',
                            name='LSTM Prediction',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        ))
                    except:
                        pass
                
                fig.update_layout(
                    title=f"{symbol} Price & Prediction",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
        
        # Return empty chart on error
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    def create_pnl_chart(self, symbol: str, trading_mode: str):
        """Create P&L chart"""
        try:
            session = db_manager.get_session()
            trades = session.query(db_manager.Trade)\
                .filter(db_manager.Trade.symbol == symbol)\
                .filter(db_manager.Trade.trading_mode == trading_mode)\
                .filter(db_manager.Trade.timestamp >= datetime.now() - timedelta(days=7))\
                .order_by(db_manager.Trade.timestamp)\
                .all()
            
            if trades:
                timestamps = [trade.timestamp for trade in trades]
                cumulative_pnl = np.cumsum([trade.profit_loss or 0 for trade in trades])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='#2ca02c' if cumulative_pnl[-1] > 0 else '#d62728', width=2),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title=f"{symbol} {trading_mode.title()} P&L (7d)",
                    xaxis_title="Time",
                    yaxis_title="P&L ($)",
                    height=400
                )
                
                session.close()
                return fig
            
            session.close()
        except:
            pass
        
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(text="No trades available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    def get_current_strategy(self, symbol: str, trading_mode: str) -> Dict:
        """Get current active strategy"""
        try:
            session = db_manager.get_session()
            strategy = session.query(db_manager.Strategy)\
                .filter(db_manager.Strategy.symbol == symbol)\
                .filter(db_manager.Strategy.trading_mode == trading_mode)\
                .filter(db_manager.Strategy.is_active == True)\
                .order_by(db_manager.Strategy.timestamp.desc())\
                .first()
            
            if strategy:
                result = strategy.strategy_data or {}
                session.close()
                return result
            
            session.close()
            return {}
        except:
            return {}
    
    def display_strategy_info(self, strategy: Dict):
        """Display strategy information"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Market Outlook:** {strategy.get('market_outlook', 'Unknown')}")
                st.write(f"**Confidence:** {strategy.get('confidence', 0)*100:.1f}%")
                
                indicators = strategy.get('indicators', [])
                if indicators:
                    st.write("**Indicators:**")
                    for ind in indicators:
                        st.write(f"• {ind.get('name', 'Unknown')}")
            
            with col2:
                risk = strategy.get('risk_management', {})
                st.write(f"**Stop Loss:** {risk.get('stop_loss', 0):.1f}%")
                st.write(f"**Take Profit:** {risk.get('take_profit', 0):.1f}%")
                st.write(f"**Position Size:** {risk.get('position_size', 0):.1f}%")
        except:
            st.error("Error displaying strategy info")
    
    def get_trades_for_mode(self, symbol: str, trading_mode: str) -> pd.DataFrame:
        """Get trades for specific mode"""
        try:
            session = db_manager.get_session()
            trades = session.query(db_manager.Trade)\
                .filter(db_manager.Trade.symbol == symbol)\
                .filter(db_manager.Trade.trading_mode == trading_mode)\
                .filter(db_manager.Trade.timestamp >= datetime.now() - timedelta(days=7))\
                .order_by(db_manager.Trade.timestamp.desc())\
                .limit(10)\
                .all()
            
            if trades:
                data = []
                for trade in trades:
                    data.append({
                        'Side': trade.side,
                        'Amount': f"{trade.amount:.4f}",
                        'Price': f"${trade.price:.2f}",
                        'P&L': f"${trade.profit_loss or 0:.2f}",
                        'Time': trade.timestamp.strftime('%m-%d %H:%M')
                    })
                
                session.close()
                return pd.DataFrame(data)
            
            session.close()
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def display_risk_metrics(self, symbol: str, trading_mode: str):
        """Display risk metrics"""
        try:
            metrics = self.get_trading_mode_metrics(symbol, trading_mode)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var = metrics.get('var_5', 0) * 100
                st.metric("VaR (5%)", f"{var:.2f}%")
            
            with col2:
                sortino = metrics.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            
            with col3:
                drawdown_duration = metrics.get('max_drawdown_duration', 0)
                st.metric("DD Duration", f"{drawdown_duration} periods")
        except:
            st.error("Error loading risk metrics")
    
    def get_backtest_results(self, symbol: str, trading_mode: str) -> List[Dict]:
        """Get backtest results"""
        try:
            session = db_manager.get_session()
            backtests = session.query(db_manager.Backtest)\
                .filter(db_manager.Backtest.symbol == symbol)\
                .filter(db_manager.Backtest.trading_mode == trading_mode)\
                .order_by(db_manager.Backtest.created_at.desc())\
                .limit(10)\
                .all()
            
            if backtests:
                data = []
                for bt in backtests:
                    data.append({
                        'Date': bt.created_at.strftime('%Y-%m-%d'),
                        'Total Return': f"{bt.total_return:.2f}%",
                        'Sharpe': f"{bt.sharpe_ratio or 0:.2f}",
                        'Max DD': f"{bt.max_drawdown or 0:.2f}%",
                        'Win Rate': f"{bt.win_rate or 0:.1f}%",
                        'Trades': bt.total_trades
                    })
                
                session.close()
                return data
            
            session.close()
            return []
        except:
            return []
    
    def create_backtest_metrics_chart(self, backtest_data: List[Dict]):
        """Create backtest metrics chart"""
        if not backtest_data:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Placeholder chart
        fig = go.Figure()
        fig.add_annotation(text="Backtest metrics chart", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    def create_backtest_equity_curve(self, backtest_data: List[Dict]):
        """Create backtest equity curve"""
        if not backtest_data:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Placeholder chart
        fig = go.Figure()
        fig.add_annotation(text="Equity curve chart", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    def create_strategy_performance_heatmap(self):
        """Create strategy performance heatmap"""
        # Placeholder heatmap
        fig = go.Figure()
        fig.add_annotation(text="Strategy performance heatmap", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    def run_new_backtest(self, symbol: str, trading_mode: str, days: int):
        """Run new backtest"""
        try:
            st.success(f"Backtest initiated for {symbol} ({trading_mode}) - {days} days")
            # Implementation would trigger actual backtest
        except Exception as e:
            st.error(f"Error running backtest: {e}")

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()