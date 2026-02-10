import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VaR & ES Calculator",
)

# Title and Introduction
st.title("Value at Risk (VaR) and Expected Shortfall (ES) Calculator")
st.markdown("---")

# Use tabs for navigation instead of radio buttons
tab1, tab2 = st.tabs(["Introduction & Theory", "Calculator"])

def calculate_parametric_var_es(investment, mean_return, std_dev, n_days, confidence_level):
    """Calculate parametric VaR and ES for n-day horizon"""
    alpha = 1 - (confidence_level / 100)
    z_score = stats.norm.ppf(alpha)
    
    # Correct VaR formula for n-day horizon
    var = investment * (mean_return * n_days - z_score * std_dev * np.sqrt(n_days))
    
    # Correct ES formula for n-day horizon
    es = investment * (mean_return * n_days - 
                      std_dev * np.sqrt(n_days) * 
                      stats.norm.pdf(z_score) / alpha)
    
    return var, es, z_score

def calculate_monte_carlo_var_es(investment, mean_return, std_dev, n_days, confidence_level, n_simulations):
    """Calculate VaR and ES using Monte Carlo simulation for n-day horizon"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random daily returns for n_days across n_simulations
    daily_returns = np.random.normal(
        mean_return, 
        std_dev, 
        (n_simulations, n_days)
    )
    
    # Calculate cumulative returns for each simulation path
    cumulative_returns = np.prod(1 + daily_returns, axis=1) - 1
    
    # Calculate portfolio values
    portfolio_values = investment * (1 + cumulative_returns)
    portfolio_returns = portfolio_values - investment
    
    # Sort returns for VaR and ES calculation
    sorted_returns = np.sort(portfolio_returns)
    alpha = 1 - (confidence_level / 100)
    var_idx = int(alpha * n_simulations)
    
    var_mc = sorted_returns[var_idx]
    es_mc = sorted_returns[:var_idx].mean()
    
    return var_mc, es_mc, portfolio_returns, daily_returns

with tab1:  # Introduction & Theory
    st.header("Introduction to Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### What is Value at Risk (VaR)?")
        st.markdown("""
        **Value at Risk (VaR)** is a statistical measure that quantifies the level of financial risk 
        within a firm, portfolio, or position over a specific time frame. It estimates the maximum 
        potential loss with a given confidence level.
        """)
        
        st.markdown("#### Formula:")
        st.latex(r"VaR = P \times (\mu \times n - Z_{\alpha} \times \sigma \times \sqrt{n})")
        
        st.markdown("**Where:**")
        st.markdown(r"""
        - $P$ = Initial investment
        - $Z_{\alpha}$ = Z-score for confidence level $\alpha$
        - $\mu$ = Mean daily return
        - $\sigma$ = Daily standard deviation
        - $n$ = Time horizon in days
        """)
    
    with col2:
        st.markdown("### What is Expected Shortfall (ES)?")
        st.markdown("""
        **Expected Shortfall (ES)**, also known as Conditional VaR (CVaR), measures the average loss 
        that occurs in the worst-case scenarios beyond the VaR level. It provides a more 
        comprehensive view of tail risk.
        """)
        
        st.markdown("#### Formula:")
        st.latex(r"ES = P \times (\mu \times n - \sigma \times \sqrt{n} \times \frac{\phi(Z_{\alpha})}{1-\alpha})")
        
        st.markdown("**Where:**")
        st.markdown(r"""
        - $\phi(Z_{\alpha})$ = Normal density at the VaR quantile
        - $\alpha$ = Confidence level (e.g., 0.05 for 95%)
        - $\sigma \times \sqrt{n}$ = Volatility scaled by time horizon
        """)
    
    st.markdown("---")
    
    st.header("Time Horizon Scaling")
    
    st.markdown("### How Time Horizon Affects VaR and ES")
    
    st.markdown("#### Volatility Scaling:")
    st.latex(r"\sigma_n = \sigma_{daily} \times \sqrt{n}")
    
    st.markdown("#### Mean Return Scaling:")
    st.latex(r"\mu_n = \mu_{daily} \times n")
    
    st.markdown("#### Impact on VaR and ES:")
    st.markdown("""
    1. **Longer horizons** → Higher absolute VaR/ES values
    2. **Volatility increases** at $\sqrt{n}$ rate
    3. **Mean return increases** at linear rate
    4. **Distribution widens** with longer horizons
    """)

with tab2:  # Calculator page
    st.header("VaR and ES Calculator")
    
    # Create input section in a container
    with st.container():
        st.subheader("Input Parameters")
        
        # Create columns for inputs
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            # Investment amount
            investment = st.number_input(
                "Initial Investment Amount ($)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                help="The initial portfolio value"
            )
            
            # Stock selection
            stock_symbol = st.text_input(
                "Stock Symbol (e.g., AAPL, MSFT, GOOGL)", 
                "AAPL",
                help="Enter a valid stock ticker symbol"
            ).upper()
        
        with input_col2:
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Start date for historical data"
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End date for historical data"
            )
        
        with input_col3:
            # Time horizon and confidence level
            forecast_days = st.slider(
                "Time Horizon (Days)",
                min_value=1,
                value=10,
                step=1,
                help="Number of days for VaR/ES calculation"
            )
            
            confidence_level = st.slider(
                "Confidence Level (%)",
                min_value=0,
                max_value=100,
                value=95,
                step=1,
                help="Confidence level for risk calculation (e.g., 95% means 5% worst cases)"
            )
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                st.markdown("#### Statistical Parameters")
                
                use_custom_params = st.checkbox(
                    "Use Custom Statistical Parameters",
                    help="Override automatic calculation from historical data"
                )
            
            with adv_col2:
                if use_custom_params:
                    mean_return = st.number_input(
                        "Daily Mean Return (%)",
                        value=0.05,
                        step=0.01,
                        format="%.2f",
                        help="Expected average daily return"
                    ) / 100
                else:
                    mean_return = None
            
            with adv_col3:
                if use_custom_params:
                    std_dev = st.number_input(
                        "Daily Standard Deviation (%)",
                        value=1.5,
                        step=0.1,
                        format="%.2f",
                        help="Daily volatility (standard deviation of returns)"
                    ) / 100
                else:
                    std_dev = None
        
        # Simulation parameters
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            n_simulations = st.number_input(
                "Number of Monte Carlo Simulations",
                min_value=1,
                value=1000,
                step=1000,
                help="More simulations = more accurate but slower computation"
            )
        
        with sim_col2:
            calculate_button = st.button("Calculate VaR & ES", use_container_width=True)
    
    # Results section - appears below when button is clicked
    if calculate_button:
        try:
            # Display loading
            with st.spinner(f"Fetching data and calculating {forecast_days}-day VaR & ES..."):
                # Fetch historical data using yfinance
                ticker = yf.Ticker(stock_symbol)
                
                # Get data without filling NaN values
                stock_data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    auto_adjust=True
                )
                
                # Check if data was retrieved
                if stock_data.empty:
                    st.error(f"No data found for {stock_symbol}. Please check the symbol and date range.")
                    st.stop()
                
                # Get basic stock info for validation
                try:
                    info = ticker.info
                    st.success(f"✓ Retrieved data for {info.get('longName', stock_symbol)}")
                except:
                    st.success(f"✓ Retrieved data for {stock_symbol}")
                
                # Select price column (use Close if available)
                if 'Close' in stock_data.columns:
                    price_series = stock_data['Close']
                else:
                    # Fall back to the first numeric column
                    numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
                    price_series = stock_data[numeric_cols[0]]
                
                # Drop NaN values from price series
                price_series_clean = price_series.dropna()
                
                if len(price_series_clean) == 0:
                    st.error("No valid price data after removing NaN values. Please try a different date range.")
                    st.stop()
                
                # Calculate returns and drop NaN values from returns
                returns = price_series_clean.pct_change().dropna()
                
                if len(returns) < 10:
                    st.warning(f"Only {len(returns)} valid return data points. For more reliable results, use a longer date range.")
                
                if len(returns) < 2:
                    st.error("Insufficient return data for calculations. Please use a longer date range.")
                    st.stop()
                
                # Use custom parameters or calculate from data
                if mean_return is None:
                    mean_return = float(returns.mean())
                    std_dev = float(returns.std())
                
                # Check for valid calculations
                if pd.isna(mean_return) or pd.isna(std_dev):
                    st.error("Could not calculate mean or standard deviation. Please check your data.")
                    st.stop()
                
                if std_dev == 0:
                    st.warning("Standard deviation is zero. This may indicate insufficient price variation in the selected period.")
                
                # Display statistical parameters (rounded to 2 decimal places)
                st.markdown("---")
                st.subheader("Statistical Summary")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Daily Mean", f"{mean_return*100:.2f}%")
                with stats_col2:
                    st.metric("Daily Std Dev", f"{std_dev*100:.2f}%")
                with stats_col3:
                    annualized_vol = std_dev * np.sqrt(252)
                    st.metric("Annual Volatility", f"{annualized_vol*100:.2f}%")
                with stats_col4:
                    st.metric("Data Points", f"{len(returns)}")
                
                # Time-scaled parameters (show only final values)
                st.markdown(f"#### {forecast_days}-Day Scaled Parameters")
                
                scaled_mean = mean_return * forecast_days
                scaled_vol = std_dev * np.sqrt(forecast_days)
                
                scale_col1, scale_col2 = st.columns(2)
                with scale_col1:
                    st.metric(f"{forecast_days}-Day Mean Return", f"{scaled_mean*100:.2f}%")
                with scale_col2:
                    st.metric(f"{forecast_days}-Day Volatility", f"{scaled_vol*100:.2f}%")
                
                # Calculate VaR and ES using time-scaled formulas
                var_parametric, es_parametric, z_score = calculate_parametric_var_es(
                    investment, mean_return, std_dev, forecast_days, confidence_level
                )
                
                var_monte_carlo, es_monte_carlo, portfolio_returns, daily_returns = calculate_monte_carlo_var_es(
                    investment, mean_return, std_dev, forecast_days, confidence_level, n_simulations
                )
                
                # Display results
                st.markdown("---")
                st.subheader(f"{forecast_days}-Day Risk Metrics")
                
                # Create metrics
                results_col1, results_col2 = st.columns(2)
                
                with results_col1:
                    st.markdown("##### Parametric Method")
                    st.metric(
                        f"Value at Risk ({confidence_level}%)",
                        f"-${abs(var_parametric):,.2f}",
                        delta=None
                    )
                    st.metric(
                        "Expected Shortfall",
                        f"-${abs(es_parametric):,.2f}",
                        delta=None
                    )
                    st.caption(f"Z-score: {z_score:.4f}")
                
                with results_col2:
                    st.markdown("##### Monte Carlo Simulation")
                    st.metric(
                        f"Value at Risk ({confidence_level}%)",
                        f"-${abs(var_monte_carlo):,.2f}",
                        delta=None
                    )
                    st.metric(
                        "Expected Shortfall",
                        f"-${abs(es_monte_carlo):,.2f}",
                        delta=None
                    )
                    st.caption(f"Based on {n_simulations:,} simulations")
                
                # Display percentage of investment
                st.markdown("#### As Percentage of Investment")
                
                percent_col1, percent_col2 = st.columns(2)
                with percent_col1:
                    var_percent_para = abs(var_parametric) / investment * 100
                    es_percent_para = abs(es_parametric) / investment * 100
                    st.metric("Parametric VaR", f"{var_percent_para:.2f}%")
                    st.metric("Parametric ES", f"{es_percent_para:.2f}%")
                
                with percent_col2:
                    var_percent_mc = abs(var_monte_carlo) / investment * 100
                    es_percent_mc = abs(es_monte_carlo) / investment * 100
                    st.metric("Monte Carlo VaR", f"{var_percent_mc:.2f}%")
                    st.metric("Monte Carlo ES", f"{es_percent_mc:.2f}%")
                
                # Visualization
                st.markdown("---")
                st.subheader("Visualizations")
                
                # Explanation of Probability Density
                with st.expander("📊 What does 'Probability Density' mean in the return distribution graph?"):
                    st.markdown("""
                    **Probability Density** in the return distribution graph shows:
                    
                    ### Understanding Probability Density:
                    1. **Y-axis (Density)**: Represents how likely returns are to occur in a given range
                    2. **Higher density** = More likely returns in that range
                    3. **Area under the curve** = Total probability (always equals 1)
                    
                    ### Interpretation:
                    - **Tall, narrow peak** → Returns are concentrated around the mean
                    - **Wide, flat distribution** → Returns are more spread out (higher volatility)
                    - **Area under curve** between two points = Probability of returns in that range
                    
                    ### Example:
                    If the density at $10,000 return is 0.0002, it means:
                    - Returns around $10,000 have a certain likelihood
                    - To get actual probability, you'd multiply density by the bin width
                    
                    **Key point**: Density values can be greater than 1 for very narrow distributions - what matters is the area under the curve, not the height.
                    """)
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    f"{forecast_days}-Day Return Distribution", 
                    "Monte Carlo Paths", 
                    "Time Horizon Analysis"
                ])
                
                with viz_tab1:
                    # Histogram of Monte Carlo returns with VaR/ES
                    fig1 = go.Figure()
                    
                    # Add histogram
                    fig1.add_trace(go.Histogram(
                        x=portfolio_returns,
                        nbinsx=50,
                        name=f'Simulated Returns',
                        marker_color='#3498db',
                        opacity=0.7,
                        histnorm='probability density',
                        hovertemplate='<b>Return</b>: $%{x:,.0f}<br><b>Density</b>: %{y:.4f}<extra></extra>'
                    ))
                    
                    # Add normal distribution overlay for comparison
                    x_norm = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
                    y_norm = stats.norm.pdf(
                        x_norm, 
                        loc=investment * scaled_mean,
                        scale=investment * scaled_vol
                    )
                    
                    fig1.add_trace(go.Scatter(
                        x=x_norm,
                        y=y_norm,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='#2c3e50', width=2.5, dash='dash'),
                        hovertemplate='<b>Normal Distribution</b><br>Return: $%{x:,.0f}<br>Density: %{y:.4f}<extra></extra>'
                    ))
                    
                    # Add vertical lines for VaR and ES with better positioning
                    fig1.add_vline(
                        x=var_monte_carlo,
                        line_dash="dash",
                        line_color="#e74c3c",
                        line_width=3,
                        annotation=dict(
                            text=f"<b>VaR ({confidence_level}%)</b><br>-${abs(var_monte_carlo):,.0f}",
                            font=dict(size=11, color="#e74c3c"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=2,
                            bordercolor="#e74c3c",
                            yanchor="top",
                            y=0.98,
                            xanchor="right",
                            x=0.98,
                            showarrow=False
                        )
                    )
                    
                    fig1.add_vline(
                        x=es_monte_carlo,
                        line_dash="dot",
                        line_color="#f39c12",
                        line_width=3,
                        annotation=dict(
                            text=f"<b>Expected Shortfall</b><br>-${abs(es_monte_carlo):,.0f}",
                            font=dict(size=11, color="#f39c12"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=2,
                            bordercolor="#f39c12",
                            yanchor="top",
                            y=0.88,
                            xanchor="right",
                            x=0.98,
                            showarrow=False
                        )
                    )
                    
                    # Shade the tail region
                    tail_returns = portfolio_returns[portfolio_returns <= var_monte_carlo]
                    if len(tail_returns) > 0:
                        x_tail = np.linspace(portfolio_returns.min(), var_monte_carlo, 100)
                        # Get approximate density values for tail region
                        y_tail = np.zeros_like(x_tail)
                        
                        fig1.add_trace(go.Scatter(
                            x=x_tail,
                            y=y_tail,
                            fill='tozeroy',
                            fillcolor='rgba(231, 76, 60, 0.25)',
                            line=dict(width=0),
                            name=f'Worst {100-confidence_level}% Cases',
                            hovertemplate='<b>Tail Risk Region</b><br>Return ≤ VaR<extra></extra>',
                            showlegend=True
                        ))
                    
                    fig1.update_layout(
                        title=dict(
                            text=f"{forecast_days}-Day Return Distribution",
                            font=dict(size=20, color='#2c3e50'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text=f"{forecast_days}-Day Return ($)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        yaxis_title=dict(
                            text="Probability Density",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        template="plotly_white",
                        height=600,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='rgba(0, 0, 0, 0.2)',
                            borderwidth=1,
                            font=dict(size=12),
                            itemsizing='constant'
                        ),
                        margin=dict(l=50, r=50, t=80, b=50),
                        plot_bgcolor='rgba(240, 240, 240, 0.1)'
                    )
                    
                    # Add grid for better readability
                    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Add summary statistics below the chart
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        st.metric("Simulation Mean", f"${portfolio_returns.mean():,.0f}")
                    with summary_col2:
                        st.metric("Simulation Std Dev", f"${portfolio_returns.std():,.0f}")
                    with summary_col3:
                        st.metric("Minimum Return", f"${portfolio_returns.min():,.0f}")
                    with summary_col4:
                        st.metric("Maximum Return", f"${portfolio_returns.max():,.0f}")
                
                with viz_tab2:
                    # Plot sample of Monte Carlo paths
                    n_sample_paths = min(50, n_simulations)
                    sample_indices = np.random.choice(n_simulations, n_sample_paths, replace=False)
                    
                    fig2 = go.Figure()
                    
                    for idx in sample_indices:
                        # Calculate cumulative portfolio value over time
                        cumulative_return = np.cumprod(1 + daily_returns[idx])
                        portfolio_path = investment * cumulative_return
                        
                        fig2.add_trace(go.Scatter(
                            x=list(range(forecast_days)),
                            y=portfolio_path,
                            mode='lines',
                            line=dict(width=1, color='rgba(52, 152, 219, 0.15)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Add mean path
                    mean_path = investment * np.cumprod(1 + daily_returns.mean(axis=0))
                    fig2.add_trace(go.Scatter(
                        x=list(range(forecast_days)),
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color='#e74c3c'),
                        name='Mean Path'
                    ))
                    
                    # Add initial investment line
                    fig2.add_hline(
                        y=investment,
                        line_dash="dash",
                        line_color="green",
                        line_width=2,
                        annotation=dict(
                            text=f"Initial Investment: ${investment:,.0f}",
                            font=dict(size=11, color="green"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=1,
                            bordercolor="green",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.02,
                            showarrow=False
                        )
                    )
                    
                    fig2.update_layout(
                        title=dict(
                            text=f"Monte Carlo Simulation Paths ({forecast_days} Days)",
                            font=dict(size=20, color='#2c3e50'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text="Days",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        yaxis_title=dict(
                            text="Portfolio Value ($)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        template="plotly_white",
                        height=600,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='rgba(0, 0, 0, 0.2)',
                            borderwidth=1,
                            font=dict(size=12)
                        ),
                        margin=dict(l=50, r=50, t=80, b=50),
                        plot_bgcolor='rgba(240, 240, 240, 0.1)'
                    )
                    
                    # Add grid for better readability
                    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add path statistics
                    path_col1, path_col2, path_col3 = st.columns(3)
                    with path_col1:
                        st.metric("Final Mean Value", f"${mean_path[-1]:,.0f}")
                    with path_col2:
                        max_final_value = max([investment * np.prod(1 + daily_returns[i]) for i in sample_indices])
                        st.metric("Max Final Value", f"${max_final_value:,.0f}")
                    with path_col3:
                        min_final_value = min([investment * np.prod(1 + daily_returns[i]) for i in sample_indices])
                        st.metric("Min Final Value", f"${min_final_value:,.0f}")
                
                with viz_tab3:
                    # Analyze how VaR scales with different time horizons
                    horizons_to_analyze = [1, 5, 10, 20, forecast_days]
                    if forecast_days not in horizons_to_analyze:
                        horizons_to_analyze.append(forecast_days)
                        horizons_to_analyze.sort()
                    
                    parametric_vars = []
                    monte_carlo_vars = []
                    
                    for horizon in horizons_to_analyze:
                        # Parametric VaR
                        var_para, _, _ = calculate_parametric_var_es(
                            investment, mean_return, std_dev, horizon, confidence_level
                        )
                        parametric_vars.append(abs(var_para))
                        
                        # Monte Carlo VaR (use fewer simulations for speed)
                        var_mc, _, _, _ = calculate_monte_carlo_var_es(
                            investment, mean_return, std_dev, horizon, confidence_level, min(5000, n_simulations)
                        )
                        monte_carlo_vars.append(abs(var_mc))
                    
                    # Create comparison plot
                    fig3 = go.Figure()
                    
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=parametric_vars,
                        mode='lines+markers',
                        name='Parametric VaR',
                        line=dict(color='#2c3e50', width=3),
                        marker=dict(size=10, symbol='circle')
                    ))
                    
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=monte_carlo_vars,
                        mode='lines+markers',
                        name='Monte Carlo VaR',
                        line=dict(color='#3498db', width=3, dash='dash'),
                        marker=dict(size=10, symbol='square')
                    ))
                    
                    # Add square root scaling reference
                    sqrt_scaling = [parametric_vars[0] * np.sqrt(h) for h in horizons_to_analyze]
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=sqrt_scaling,
                        mode='lines',
                        name='√n Scaling Reference',
                        line=dict(color='#95a5a6', width=2, dash='dot'),
                        opacity=0.6
                    ))
                    
                    fig3.update_layout(
                        title=dict(
                            text=f"VaR Scaling with Time Horizon ({confidence_level}% Confidence)",
                            font=dict(size=20, color='#2c3e50'),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text="Time Horizon (Days)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        yaxis_title=dict(
                            text="VaR ($)",
                            font=dict(size=14, color='#2c3e50')
                        ),
                        template="plotly_white",
                        height=600,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='rgba(0, 0, 0, 0.2)',
                            borderwidth=1,
                            font=dict(size=12)
                        ),
                        margin=dict(l=50, r=50, t=80, b=50),
                        plot_bgcolor='rgba(240, 240, 240, 0.1)'
                    )
                    
                    # Add grid for better readability
                    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Add scaling statistics
                    scaling_col1, scaling_col2, scaling_col3 = st.columns(3)
                    with scaling_col1:
                        actual_scaling = parametric_vars[-1] / parametric_vars[0]
                        st.metric("Actual Scaling Factor", f"{actual_scaling:.2f}x")
                    with scaling_col2:
                        expected_scaling = np.sqrt(forecast_days)
                        st.metric("Expected √n Scaling", f"{expected_scaling:.2f}x")
                    with scaling_col3:
                        scaling_difference = ((actual_scaling / expected_scaling) - 1) * 100
                        st.metric("Deviation from √n", f"{scaling_difference:+.1f}%")
                    
                    st.info(f"""
                    **Time Scaling Analysis:**
                    - {forecast_days}-day VaR is **{parametric_vars[-1]/parametric_vars[0]:.1f} times** the 1-day VaR
                    - Expected scaling with √{forecast_days} = **{np.sqrt(forecast_days):.1f} times**
                    - Differences show the impact of mean return and compounding effects
                    """)
                
                # Interpretation with formulas
                st.markdown("---")
                st.subheader("Risk Interpretation")
                
                interp_col1, interp_col2 = st.columns(2)
                
                with interp_col1:
                    st.markdown(f"""
                    ### Parametric Method
                    
                    **For a {forecast_days}-day holding period:**
                    
                    - **{confidence_level}% confidence** that losses won't exceed **${abs(var_parametric):,.0f}**
                    - **Expected average loss** in worst {100-confidence_level}% scenarios: **${abs(es_parametric):,.0f}**
                    
                    **Assumptions:**
                    - Returns follow normal distribution
                    - Daily returns are independent
                    - Constant volatility over time
                    """)
                
                with interp_col2:
                    worst_case_loss = portfolio_returns.min()
                    probability_below_var = np.mean(portfolio_returns <= var_monte_carlo) * 100
                    
                    st.markdown(f"""
                    ### Monte Carlo Simulation
                    
                    **Based on {n_simulations:,} simulated {forecast_days}-day paths:**
                    
                    - **{probability_below_var:.1f}%** of scenarios exceeded VaR (target: {100-confidence_level}%)
                    - **Maximum simulated loss**: **${abs(worst_case_loss):,.0f}**
                    - **Median {forecast_days}-day return**: **${np.median(portfolio_returns):,.0f}**
                    
                    **Advantages:**
                    - Captures compounding effects
                    - No normality assumption needed
                    - Simulates actual return paths
                    """)
                
                # Download results
                st.markdown("---")
                st.subheader("Export Results")
                
                # Create comprehensive results dataframe
                results_df = pd.DataFrame({
                    'Parameter': [
                        'Stock Symbol',
                        'Initial Investment ($)',
                        'Time Horizon (Days)',
                        'Confidence Level (%)',
                        'Daily Mean Return (%)',
                        'Daily Standard Deviation (%)',
                        f'{forecast_days}-Day Mean Return (%)',
                        f'{forecast_days}-Day Volatility (%)',
                        'Parametric VaR ($)',
                        'Parametric VaR (%)',
                        'Parametric ES ($)',
                        'Parametric ES (%)',
                        'Monte Carlo VaR ($)',
                        'Monte Carlo VaR (%)',
                        'Monte Carlo ES ($)',
                        'Monte Carlo ES (%)',
                        'Z-score',
                        'Number of Simulations',
                        'Maximum Simulated Loss ($)'
                    ],
                    'Value': [
                        stock_symbol,
                        f'{investment:,.0f}',
                        forecast_days,
                        f'{confidence_level}',
                        f'{mean_return*100:.2f}',
                        f'{std_dev*100:.2f}',
                        f'{scaled_mean*100:.2f}',
                        f'{scaled_vol*100:.2f}',
                        f'-{abs(var_parametric):,.2f}',
                        f'{abs(var_parametric)/investment*100:.2f}',
                        f'-{abs(es_parametric):,.2f}',
                        f'{abs(es_parametric)/investment*100:.2f}',
                        f'-{abs(var_monte_carlo):,.2f}',
                        f'{abs(var_monte_carlo)/investment*100:.2f}',
                        f'-{abs(es_monte_carlo):,.2f}',
                        f'{abs(es_monte_carlo)/investment*100:.2f}',
                        f'{z_score:.4f}',
                        f'{n_simulations:,}',
                        f'-{abs(worst_case_loss):,.2f}'
                    ]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results as CSV",
                    data=csv,
                    file_name=f"var_es_{stock_symbol}_{forecast_days}d_{confidence_level}pc_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("""
            **Troubleshooting steps:**
            1. Check your internet connection
            2. Verify the stock symbol exists (try AAPL, MSFT, GOOGL)
            3. Try a much longer date range (e.g., 2 years)
            4. Check if the stock trades during your selected period
            5. Try without custom parameters first
            """)
    
    else:
        # Display placeholder when no calculation has been performed
        st.markdown("---")
        st.info("Enter your parameters above and click 'Calculate VaR & ES' to begin analysis.")
        
            st.markdown("""
        ### How to Use This Calculator
        
        1. **Enter Investment Details**: Set your initial investment and select a stock
        2. **Choose Date Range**: Select historical data period for analysis
        3. **Set Risk Parameters**: Define time horizon and confidence level
        4. **Optional Advanced Settings**: Customize statistical parameters if needed
        5. **Click Calculate**: Generate VaR and ES calculations
        
        The calculator will:
        - Fetch historical stock price data
        - Calculate daily return statistics
        - Compute Value at Risk and Expected Shortfall using both parametric and Monte Carlo methods
        - Generate interactive visualizations of risk metrics
        """)
