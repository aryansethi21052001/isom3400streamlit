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
st.set_page_config(page_title="VaR & ES Calculator", layout="wide")

# Title and Introduction
st.title("Value at Risk (VaR) and Expected Shortfall (ES) Calculator")
st.markdown("---")

# Use tabs for navigation instead of radio buttons
tab1, tab2 = st.tabs(["Introduction & Theory", "Calculator"])

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data with caching for better performance"""
    ticker = yf.Ticker(symbol)
    stock_data = ticker.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=False
    )
    # Return only the data, not the ticker object
    return stock_data

@st.cache_data
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

@st.cache_data
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

def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5 + 0.2 * np.sin(2 * np.pi * i / n)  # Vary lightness for better distinction
        saturation = 0.7 + 0.2 * np.cos(2 * np.pi * i / n)  # Vary saturation
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.3)')
    return colors

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
    
    # Create input form for cleaner layout
    with st.form("input_form"):
        st.subheader("Input Parameters")
        
        # Create columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic Parameters
            st.markdown("#### Basic Parameters")
            investment = st.number_input(
                "Initial Investment Amount (USD)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                help="The initial portfolio value"
            )
            
            stock_symbol = st.text_input(
                "Stock Symbol (e.g., AAPL, MSFT, GOOGL)", 
                "AAPL",
                help="Enter a valid stock ticker symbol"
            ).upper()
            
            # Date range in same column
            st.markdown("#### Date Range")
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
        
        with col2:
            # Risk Parameters
            st.markdown("#### Risk Parameters")
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
            
            # Simulation parameters
            st.markdown("#### Simulation Settings")
            n_simulations = st.number_input(
                "Number of Monte Carlo Simulations",
                min_value=1,
                value=1000,
                step=1000,
                help="More simulations = more accurate but slower computation"
            )
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            adv_col1, adv_col2 = st.columns(2)
            
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
                    
                    std_dev = st.number_input(
                        "Daily Standard Deviation (%)",
                        value=1.5,
                        step=0.1,
                        format="%.2f",
                        help="Daily volatility (standard deviation of returns)"
                    ) / 100
                else:
                    mean_return = None
                    std_dev = None
        
        # Submit button
        calculate_button = st.form_submit_button("Calculate VaR & ES", use_container_width=True)
    
    # Results section - appears below when form is submitted
    if calculate_button:
        try:
            # Display loading
            with st.spinner(f"Fetching data and calculating {forecast_days}-day VaR & ES..."):
                # Fetch historical data using cached function
                stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
                
                # Check if data was retrieved
                if stock_data.empty:
                    st.error(f"No data found for {stock_symbol}. Please check the symbol and date range.")
                    st.stop()
                
                # Get basic stock info for validation (create new ticker object but don't cache it)
                ticker = yf.Ticker(stock_symbol)
                try:
                    info = ticker.info
                    st.success(f"✓ Retrieved data for {info.get('longName', stock_symbol)}")
                except:
                    st.success(f"✓ Retrieved data for {stock_symbol}")
                
                # Select price column (use Adj Close if available, otherwise use Close)
                if 'Adj Close' in stock_data.columns:
                    price_series = stock_data['Adj Close']
                else:
                    price_series = stock_data['Close']
                
                # Drop NaN values from price series
                price_series_clean = price_series.dropna()
                
                if len(price_series_clean) == 0:
                    st.error("No valid price data after removing NaN values. Please try a different date range.")
                    st.stop()
                
                # Calculate returns and drop NaN values from returns
                returns = price_series_clean.pct_change().dropna()
                
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
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    f"{forecast_days}-Day Return Distribution", 
                    "Monte Carlo Simulation", 
                    "Time Horizon Analysis"
                ])
                
                with viz_tab1:
                    # Calculate histogram data
                    hist, bin_edges = np.histogram(portfolio_returns, bins=100)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    fig1 = go.Figure()
                    
                    # Add histogram bars
                    fig1.add_trace(go.Bar(
                        x=bin_centers,
                        y=hist,
                        width=[(bin_edges[i+1] - bin_edges[i]) * 0.9 for i in range(len(bin_edges)-1)],
                        name='Simulated Returns',
                        marker_color='rgba(31, 119, 180, 0.7)',
                        opacity=0.7,
                        hovertemplate='<b>Return</b>: $%{x:,.0f}<br><b>Frequency</b>: %{y}<extra></extra>'
                    ))
                    
                    # Calculate the appropriate y-position based on histogram height at VaR position
                    # Find the histogram bin that contains the VaR value
                    var_bin_idx = np.digitize(var_monte_carlo, bin_edges) - 1
                    if var_bin_idx >= 0 and var_bin_idx < len(hist):
                        var_y_pos = hist[var_bin_idx] if var_bin_idx < len(hist) else 0
                    else:
                        var_y_pos = max(hist) * 0.8  # Fallback position
                    
                    # Find the histogram bin that contains the ES value
                    es_bin_idx = np.digitize(es_monte_carlo, bin_edges) - 1
                    if es_bin_idx >= 0 and es_bin_idx < len(hist):
                        es_y_pos = hist[es_bin_idx] if es_bin_idx < len(hist) else 0
                    else:
                        es_y_pos = max(hist) * 0.7  # Fallback position
                    
                    # Ensure positions don't overlap
                    if abs(var_bin_idx - es_bin_idx) < 2:  # If they're in same or adjacent bins
                        es_y_pos = var_y_pos * 0.8  # Place ES lower
                    
                    # Add VaR line
                    fig1.add_vline(
                        x=var_monte_carlo,
                        line_dash="dash",
                        line_color="#d62728",
                        line_width=2.5
                    )
                    
                    # Add ES line
                    fig1.add_vline(
                        x=es_monte_carlo,
                        line_dash="dot",
                        line_color="#ff7f0e",
                        line_width=2.5
                    )
                    
                    # Shade the tail region
                    tail_mask = portfolio_returns <= var_monte_carlo
                    tail_returns = portfolio_returns[tail_mask]
                    
                    if len(tail_returns) > 0:
                        # Find which histogram bins are in the tail
                        tail_bin_indices = [i for i, center in enumerate(bin_centers) 
                                          if center <= var_monte_carlo]
                        
                        if tail_bin_indices:
                            # Add shaded bars for tail region
                            for idx in tail_bin_indices:
                                fig1.add_trace(go.Bar(
                                    x=[bin_centers[idx]],
                                    y=[hist[idx]],
                                    width=[(bin_edges[idx+1] - bin_edges[idx]) * 0.9],
                                    marker_color='rgba(214, 39, 40, 0.3)',
                                    name='Tail Risk Region' if idx == tail_bin_indices[0] else '',
                                    showlegend=True if idx == tail_bin_indices[0] else False,
                                    hovertemplate='<b>Tail Return</b>: $%{x:,.0f}<br><b>Frequency</b>: %{y}<extra></extra>'
                                ))
                    
                    fig1.update_layout(
                        title=dict(
                            text=f"{forecast_days}-Day Return Distribution",
                            font=dict(size=16),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text=f"{forecast_days}-Day Return ($)",
                            font=dict(size=12)
                        ),
                        yaxis_title=dict(
                            text="Frequency",
                            font=dict(size=12)
                        ),
                        xaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=1.15,
                            bgcolor="rgba(255, 255, 255, 0.9)",
                            bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1,
                            font=dict(size=10, color="black"),
                            itemsizing="constant",
                            title=dict(
                                text=f"<b>Risk Metrics</b><br>VaR ({confidence_level}%): -${abs(var_monte_carlo):,.2f}<br>ES: -${abs(es_monte_carlo):,.2f}",
                                font=dict(size=10)
                            )
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        bargap=0.1,
                        bargroupgap=0.1
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Add summary statistics below the chart
                    st.caption("Distribution Statistics")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        sim_mean = portfolio_returns.mean()
                        st.metric("Simulation Mean", f"${sim_mean:,.0f}")
                    with summary_col2:
                        sim_median = np.median(portfolio_returns)
                        st.metric("Simulation Median", f"${sim_median:,.0f}")
                    with summary_col3:
                        st.metric("Simulation Std Dev", f"${portfolio_returns.std():,.0f}")
                    with summary_col4:
                        st.metric("Tail Probability", f"{len(tail_returns)/len(portfolio_returns)*100:.1f}%")
                
                with viz_tab2:
                    # Show ALL individual paths with unique colors
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Generate distinct colors for all paths
                    status_text.text(f"Generating {n_simulations:,} simulations...")
                    colors = generate_distinct_colors(n_simulations)
                    
                    fig2 = go.Figure()
                    
                    # Plot ALL paths with progress indicator
                    for i in range(n_simulations):
                        # Calculate cumulative portfolio value over time
                        cumulative_return = np.cumprod(1 + daily_returns[i])
                        portfolio_path = investment * cumulative_return
                        
                        # Add path to figure
                        fig2.add_trace(go.Scatter(
                            x=list(range(forecast_days)),
                            y=portfolio_path,
                            mode='lines',
                            line=dict(width=1, color=colors[i]),
                            name=f'Path {i+1}',
                            showlegend=False, 
                            hovertemplate=f'<b>Path {i+1}</b><br>Day %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
                        ))
                        
                        # Update progress every 10 paths
                        if i % 10 == 0:
                            progress = (i + 1) / n_simulations
                            progress_bar.progress(progress)
                    
                    # Add mean path with thicker line
                    mean_path = investment * np.cumprod(1 + daily_returns.mean(axis=0))
                    fig2.add_trace(go.Scatter(
                        x=list(range(forecast_days)),
                        y=mean_path,
                        mode='lines',
                        line=dict(width=3, color='#000000'),
                        name='Mean Path',
                        hovertemplate='<b>Mean Path</b><br>Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Add initial investment line
                    fig2.add_hline(
                        y=investment,
                        line_dash="dash",
                        line_color="#2ca02c",
                        line_width=2,
                        annotation=dict(
                            text=f"Initial: ${investment:,.2f}",
                            font=dict(size=10, color="#2ca02c"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=1,
                            bordercolor="#2ca02c",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.02,
                            showarrow=False
                        )
                    )
                    
                    # Add median path
                    median_path = investment * np.cumprod(1 + np.median(daily_returns, axis=0))
                    fig2.add_trace(go.Scatter(
                        x=list(range(forecast_days)),
                        y=median_path,
                        mode='lines',
                        line=dict(width=2, color='#ff7f0e', dash='dash'),
                        name='Median Path',
                        hovertemplate='<b>Median Path</b><br>Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Add VaR threshold line (at VaR level portfolio value)
                    var_portfolio_value = investment + var_monte_carlo
                    fig2.add_hline(
                        y=var_portfolio_value,
                        line_dash="dot",
                        line_color="#d62728",
                        line_width=2,
                        annotation=dict(
                            text=f"VaR ({confidence_level}%): ${var_portfolio_value:,.0f}",
                            font=dict(size=10, color="#d62728"),
                            bgcolor="rgba(255,255,255,0.9)",
                            borderwidth=1,
                            bordercolor="#d62728",
                            yanchor="top",
                            y=-0.02,
                            xanchor="left",
                            x=0.02,
                            showarrow=False
                        )
                    )
                    
                    fig2.update_layout(
                        title=dict(
                            text=f"Monte Carlo Simulation ({n_simulations:,} Paths) for {forecast_days} Days",
                            font=dict(size=16),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text="Days",
                            font=dict(size=12)
                        ),
                        yaxis_title=dict(
                            text="Portfolio Value ($)",
                            font=dict(size=12)
                        ),
                        xaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white",
                        height=600,
                        hovermode="closest",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.02,
                            bgcolor="rgba(255, 255, 255, 0.9)",
                            bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1,
                            font=dict(size=10, color="black")
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)'
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    status_text.success(f"✓ Generated {n_simulations:,} paths!")
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add path statistics
                    st.caption("Path Statistics")
                    path_col1, path_col2, path_col3, path_col4 = st.columns(4)
                    with path_col1:
                        st.metric("Final Mean Value", f"${mean_path[-1]:,.0f}")
                    with path_col2:
                        max_final_value = investment * np.max(np.prod(1 + daily_returns, axis=1))
                        st.metric("Max Final Value", f"${max_final_value:,.0f}")
                    with path_col3:
                        min_final_value = investment * np.min(np.prod(1 + daily_returns, axis=1))
                        st.metric("Min Final Value", f"${min_final_value:,.0f}")
                    with path_col4:
                        median_final = investment * np.median(np.prod(1 + daily_returns, axis=1))
                        st.metric("Median Final", f"${median_final:,.0f}")
                    
                    # Add path density information
                    st.info(f"**Visualization Note:** Showing all {n_simulations:,} paths.")
                
                with viz_tab3:
                    # Analyze how VaR scales with different time horizons
                    horizons_to_analyze = [1, 5, 10, 20, forecast_days]
                    if forecast_days not in horizons_to_analyze:
                        horizons_to_analyze.append(forecast_days)
                        horizons_to_analyze.sort()
                    
                    parametric_vars = []
                    parametric_es_list = []
                    monte_carlo_vars = []
                    monte_carlo_es_list = []
                    
                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()
                    
                    for idx, horizon in enumerate(horizons_to_analyze):
                        status_text2.text(f"Calculating for {horizon}-day horizon...")
                        
                        # Parametric VaR & ES
                        var_para, es_para, _ = calculate_parametric_var_es(
                            investment, mean_return, std_dev, horizon, confidence_level
                        )
                        parametric_vars.append(abs(var_para))
                        parametric_es_list.append(abs(es_para))
                        
                        # Monte Carlo VaR & ES
                        var_mc, es_mc, _, _ = calculate_monte_carlo_var_es(
                            investment, mean_return, std_dev, horizon, confidence_level, n_simulations
                        )
                        monte_carlo_vars.append(abs(var_mc))
                        monte_carlo_es_list.append(abs(es_mc))
                        
                        progress_bar2.progress((idx + 1) / len(horizons_to_analyze))
                    
                    # Create comparison plot
                    fig3 = go.Figure()
                    
                    # Add traces for VaR
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=parametric_vars,
                        mode='lines+markers',
                        name='Parametric VaR',
                        line=dict(color='#1f77b4', width=2.5),
                        marker=dict(size=8, symbol='circle')
                    ))
                    
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=monte_carlo_vars,
                        mode='lines+markers',
                        name='Monte Carlo VaR',
                        line=dict(color='#ff7f0e', width=2.5, dash='dash'),
                        marker=dict(size=8, symbol='square')
                    ))
                    
                    # Add traces for ES
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=parametric_es_list,
                        mode='lines+markers',
                        name='Parametric ES',
                        line=dict(color='#2ca02c', width=2.5),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=monte_carlo_es_list,
                        mode='lines+markers',
                        name='Monte Carlo ES',
                        line=dict(color='#d62728', width=2.5, dash='dot'),
                        marker=dict(size=8, symbol='cross')
                    ))
                    
                    # Add square root scaling reference
                    sqrt_scaling = [parametric_vars[0] * np.sqrt(h) for h in horizons_to_analyze]
                    fig3.add_trace(go.Scatter(
                        x=horizons_to_analyze,
                        y=sqrt_scaling,
                        mode='lines',
                        name='√n Scaling',
                        line=dict(color='#7f7f7f', width=1.5, dash='dot'),
                        opacity=0.6
                    ))
                    
                    fig3.update_layout(
                        title=dict(
                            text=f"Risk Metric Scaling with Time Horizon ({confidence_level}% Confidence)",
                            font=dict(size=16),
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title=dict(
                            text="Time Horizon (Days)",
                            font=dict(size=12)
                        ),
                        yaxis_title=dict(
                            text="Risk Metric ($)",
                            font=dict(size=12)
                        ),
                        xaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        yaxis=dict(fixedrange=True, showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
                        template="plotly_white",
                        height=500,
                        hovermode="x unified",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.02,
                            bgcolor="rgba(255, 255, 255, 0.9)",
                            bordercolor="rgba(0, 0, 0, 0.2)",
                            borderwidth=1,
                            font=dict(size=10, color="black")
                        ),
                        margin=dict(l=50, r=50, t=60, b=50),
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)'
                    )
                    
                    progress_bar2.empty()
                    status_text2.empty()
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Add scaling statistics
                    st.caption("Scaling Analysis")
                    scaling_col1, scaling_col2, scaling_col3, scaling_col4 = st.columns(4)
                    with scaling_col1:
                        actual_scaling = parametric_vars[-1] / parametric_vars[0]
                        st.metric("VaR Scaling", f"{actual_scaling:.2f}x")
                    with scaling_col2:
                        expected_scaling = np.sqrt(forecast_days)
                        st.metric("Expected √n", f"{expected_scaling:.2f}x")
                    with scaling_col3:
                        scaling_difference = ((actual_scaling / expected_scaling) - 1) * 100
                        st.metric("Deviation", f"{scaling_difference:+.1f}%")
                    with scaling_col4:
                        es_scaling = parametric_es_list[-1] / parametric_es_list[0]
                        st.metric("ES Scaling", f"{es_scaling:.2f}x")
                    
                    st.info(f"""
                    **Time Scaling Analysis:**
                    - {forecast_days}-day VaR is **{parametric_vars[-1]/parametric_vars[0]:.1f} times** the 1-day VaR
                    - {forecast_days}-day ES is **{parametric_es_list[-1]/parametric_es_list[0]:.1f} times** the 1-day ES
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
                    median_return = np.median(portfolio_returns)
                    
                    st.markdown(f"""
                    ### Monte Carlo Simulation
                    
                    **Based on {n_simulations:,} simulated {forecast_days}-day paths:**
                    
                    - **{probability_below_var:.1f}%** of scenarios exceeded VaR (target: {100-confidence_level}%)
                    - **Maximum simulated loss**: **${abs(worst_case_loss):,.0f}**
                    - **Median {forecast_days}-day return**: **${median_return:,.0f}**
                    
                    **Advantages:**
                    - Captures compounding effects
                    - No normality assumption needed
                    - Simulates actual return paths
                    - Shows full distribution of outcomes
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
                        'Maximum Simulated Loss ($)',
                        'Median Return ($)',
                        'Simulation Mean Return ($)'
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
                        f'-{abs(worst_case_loss):,.2f}',
                        f'{np.median(portfolio_returns):,.2f}',
                        f'{portfolio_returns.mean():,.2f}'
                    ]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Complete Results as CSV",
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
