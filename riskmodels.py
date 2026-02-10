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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction & Theory", "Calculator"])

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

if page == "Introduction & Theory":
    st.header("Introduction to Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Value at Risk (VaR)?
        
        **Value at Risk (VaR)** is a statistical measure that quantifies the level of financial risk 
        within a firm, portfolio, or position over a specific time frame. It estimates the maximum 
        potential loss with a given confidence level.
        
        #### Time-Adjusted Formula:
        ```
        VaR = Initial Investment × (μ × n - Zα × σ × √n)
        ```
        
        Where:
        - **Zα** = Z-score corresponding to confidence level α
        - **μ** = Mean daily return
        - **σ** = Daily standard deviation
        - **n** = Time horizon in days
        """)
    
    with col2:
        st.markdown("""
        ### What is Expected Shortfall (ES)?
        
        **Expected Shortfall (ES)**, also known as Conditional VaR, measures the average loss 
        that occurs in the worst-case scenarios beyond the VaR level. It provides a more 
        comprehensive view of tail risk.
        
        #### Time-Adjusted Formula (Parametric):
        ```
        ES = Initial Investment × (μ × n - σ × √n × φ(Zα) / (1-α))
        ```
        
        Where:
        - **φ(Zα)** = Normal density at the VaR quantile
        - **α** = Confidence level (e.g., 0.05 for 95%)
        - **σ × √n** = Volatility scaled by time horizon
        """)
    
    st.markdown("---")
    
    st.header("Time Horizon Scaling")
    
    st.markdown("""
    ### How Time Horizon Affects Risk Metrics
    
    #### Volatility Scaling:
    For independent returns, volatility scales with the square root of time:
    ```
    σ_n = σ_daily × √n
    ```
    
    #### Mean Return Scaling:
    Expected return scales linearly with time:
    ```
    μ_n = μ_daily × n
    ```
    
    #### Impact on VaR and ES:
    1. **Longer horizons** → Higher absolute VaR/ES values
    2. **Volatility increases** at √n rate
    3. **Mean return increases** at linear rate
    4. **Distribution widens** with longer horizons
    """)
    
    # Create a visualization of time scaling
    st.subheader("Time Scaling Visualization")
    
    # Example parameters
    example_investment = 100000
    example_mean = 0.0005  # 0.05% daily
    example_std = 0.015  # 1.5% daily
    example_conf = 95
    
    # Calculate for different horizons
    horizons = [1, 5, 10, 20, 30]
    vars_parametric = []
    vars_monte_carlo = []
    
    for n in horizons:
        var_para, es_para, z_score = calculate_parametric_var_es(
            example_investment, example_mean, example_std, n, example_conf
        )
        vars_parametric.append(abs(var_para))
        
        var_mc, es_mc, _, _ = calculate_monte_carlo_var_es(
            example_investment, example_mean, example_std, n, example_conf, 10000
        )
        vars_monte_carlo.append(abs(var_mc))
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=horizons,
        y=vars_parametric,
        mode='lines+markers',
        name='Parametric VaR',
        line=dict(color='#2c3e50', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=horizons,
        y=vars_monte_carlo,
        mode='lines+markers',
        name='Monte Carlo VaR',
        line=dict(color='#3498db', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="VaR Scaling with Time Horizon",
        xaxis_title="Time Horizon (Days)",
        yaxis_title="VaR ($)",
        template="plotly_white",
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.header("Calculation Methods")
    
    method_col1, method_col2 = st.columns(2)
    
    with method_col1:
        st.markdown("""
        ### Parametric Method (Variance-Covariance)
        
        #### Time Scaling:
        ```python
        # Daily to n-day conversion
        n_day_mean = daily_mean * n
        n_day_vol = daily_vol * sqrt(n)
        
        # VaR calculation
        VaR_n = investment * (n_day_mean - Z * n_day_vol)
        ```
        
        #### Assumptions:
        1. Returns follow normal distribution
        2. Independent daily returns
        3. Constant volatility over time
        """)
    
    with method_col2:
        st.markdown("""
        ### Historical Simulation (Monte Carlo)
        
        #### Time Scaling:
        ```python
        # Generate n-day returns
        daily_returns = normal(mean, std, (simulations, n))
        n_day_returns = prod(1 + daily_returns, axis=1) - 1
        
        # Calculate portfolio values
        portfolio_values = investment * (1 + n_day_returns)
        ```
        
        #### Key Features:
        1. Captures compounding effects
        2. No normality assumption needed
        3. Can simulate path-dependent scenarios
        """)

else:  # Calculator page
    st.header("VaR and ES Calculator")
    
    # Create two main columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Investment amount
        investment = st.number_input(
            "Initial Investment Amount ($)",
            min_value=1000.0,
            max_value=10000000.0,
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
        
        # Date range
        col1a, col1b = st.columns(2)
        with col1a:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Start date for historical data"
            )
        with col1b:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End date for historical data"
            )
        
        # Time horizon and confidence level
        st.markdown("#### Risk Parameters")
        
        forecast_days = st.slider(
            "Time Horizon (Days)",
            min_value=1,
            max_value=252,
            value=10,
            step=1,
            help="Number of days for VaR/ES calculation"
        )
        
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="Confidence level for risk calculation (e.g., 95% means 5% worst cases)"
        )
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            st.markdown("#### Statistical Parameters")
            
            use_custom_params = st.checkbox(
                "Use Custom Statistical Parameters",
                help="Override automatic calculation from historical data"
            )
            
            if use_custom_params:
                mean_return = st.number_input(
                    "Daily Mean Return (%)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    format="%.3f",
                    help="Expected average daily return"
                ) / 100
                
                std_dev = st.number_input(
                    "Daily Standard Deviation (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.5,
                    step=0.1,
                    format="%.2f",
                    help="Daily volatility (standard deviation of returns)"
                ) / 100
            else:
                mean_return = None
                std_dev = None
        
        # Simulation parameters
        st.markdown("#### Simulation Settings")
        n_simulations = st.selectbox(
            "Number of Monte Carlo Simulations",
            options=[1000, 5000, 10000, 50000],
            index=1,
            help="More simulations = more accurate but slower computation"
        )
        
        calculate_button = st.button("Calculate VaR & ES")
    
    with col2:
        if calculate_button:
            try:
                # Display loading
                with st.spinner(f"Fetching data and calculating {forecast_days}-day VaR & ES..."):
                    # Fetch historical data
                    stock_data = yf.download(
                        stock_symbol,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    
                    if len(stock_data) == 0:
                        st.error(f"No data found for {stock_symbol}. Please check the symbol and date range.")
                    else:
                        # Calculate returns
                        stock_data['Returns'] = stock_data['Adj Close'].pct_change().dropna()
                        returns = stock_data['Returns'].values
                        
                        # Use custom parameters or calculate from data
                        if mean_return is None:
                            mean_return = returns.mean()
                            std_dev = returns.std()
                        
                        # Display statistical parameters
                        st.markdown("#### Statistical Summary")
                        
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        with stats_col1:
                            st.metric("Daily Mean", f"{mean_return*100:.4f}%")
                        with stats_col2:
                            st.metric("Daily Std Dev", f"{std_dev*100:.4f}%")
                        with stats_col3:
                            annualized_vol = std_dev * np.sqrt(252)
                            st.metric("Annual Volatility", f"{annualized_vol*100:.2f}%")
                        with stats_col4:
                            st.metric("Data Points", f"{len(returns)}")
                        
                        # Time-scaled parameters
                        st.markdown(f"#### {forecast_days}-Day Scaled Parameters")
                        
                        scaled_mean = mean_return * forecast_days
                        scaled_vol = std_dev * np.sqrt(forecast_days)
                        
                        scale_col1, scale_col2 = st.columns(2)
                        with scale_col1:
                            st.info(f"**Scaled Mean Return:** {scaled_mean*100:.3f}%")
                        with scale_col2:
                            st.info(f"**Scaled Volatility:** {scaled_vol*100:.3f}%")
                        
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
                                name=f'{forecast_days}-Day Returns',
                                marker_color='#3498db',
                                opacity=0.7,
                                histnorm='probability density'
                            ))
                            
                            # Add vertical lines for VaR and ES
                            fig1.add_vline(
                                x=var_monte_carlo,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"VaR ({confidence_level}%): -${abs(var_monte_carlo):,.0f}",
                                annotation_position="top left"
                            )
                            
                            fig1.add_vline(
                                x=es_monte_carlo,
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"ES: -${abs(es_monte_carlo):,.0f}",
                                annotation_position="top right"
                            )
                            
                            # Shade the tail region
                            tail_returns = portfolio_returns[portfolio_returns <= var_monte_carlo]
                            if len(tail_returns) > 0:
                                fig1.add_trace(go.Scatter(
                                    x=np.sort(tail_returns),
                                    y=np.zeros_like(tail_returns),
                                    fill='tozeroy',
                                    fillcolor='rgba(231, 76, 60, 0.3)',
                                    line=dict(width=0),
                                    name='Tail Risk Region',
                                    showlegend=True
                                ))
                            
                            fig1.update_layout(
                                title=f"Monte Carlo {forecast_days}-Day Return Distribution",
                                xaxis_title=f"{forecast_days}-Day Return ($)",
                                yaxis_title="Density",
                                template="plotly_white",
                                height=400,
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                hovermode="x unified"
                            )
                            
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
                                line=dict(color='#2c3e50', width=2, dash='dash')
                            ))
                            
                            st.plotly_chart(fig1, use_container_width=True)
                        
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
                                    line=dict(width=1, color='rgba(52, 152, 219, 0.3)'),
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
                                annotation_text=f"Initial: ${investment:,.0f}"
                            )
                            
                            fig2.update_layout(
                                title=f"Monte Carlo Simulation Paths ({forecast_days} Days)",
                                xaxis_title="Days",
                                yaxis_title="Portfolio Value ($)",
                                template="plotly_white",
                                height=400,
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig2, use_container_width=True)
                        
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
                                
                                # Monte Carlo VaR
                                var_mc, _, _, _ = calculate_monte_carlo_var_es(
                                    investment, mean_return, std_dev, horizon, confidence_level, 5000
                                )
                                monte_carlo_vars.append(abs(var_mc))
                            
                            # Create comparison plot
                            fig3 = go.Figure()
                            
                            fig3.add_trace(go.Scatter(
                                x=horizons_to_analyze,
                                y=parametric_vars,
                                mode='lines+markers',
                                name='Parametric VaR',
                                line=dict(color='#2c3e50', width=3)
                            ))
                            
                            fig3.add_trace(go.Scatter(
                                x=horizons_to_analyze,
                                y=monte_carlo_vars,
                                mode='lines+markers',
                                name='Monte Carlo VaR',
                                line=dict(color='#3498db', width=3, dash='dash')
                            ))
                            
                            # Add square root scaling reference
                            sqrt_scaling = [parametric_vars[0] * np.sqrt(h) for h in horizons_to_analyze]
                            fig3.add_trace(go.Scatter(
                                x=horizons_to_analyze,
                                y=sqrt_scaling,
                                mode='lines',
                                name='√n Scaling Reference',
                                line=dict(color='#95a5a6', width=2, dash='dot'),
                                opacity=0.5
                            ))
                            
                            fig3.update_layout(
                                title="VaR Scaling with Different Time Horizons",
                                xaxis_title="Time Horizon (Days)",
                                yaxis_title=f"VaR ({confidence_level}%) ($)",
                                template="plotly_white",
                                height=400,
                                hovermode="x unified",
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig3, use_container_width=True)
                            
                            # Add explanation
                            st.info(f"""
                            **Time Scaling Analysis:**
                            - VaR scales approximately with √n (square root of time)
                            - {forecast_days}-day VaR is about {np.sqrt(forecast_days):.1f} times 1-day VaR
                            - Differences between methods increase with longer horizons due to compounding effects
                            """)
                        
                        # Interpretation
                        st.markdown("---")
                        st.subheader("Risk Interpretation")
                        
                        interp_col1, interp_col2 = st.columns(2)
                        
                        with interp_col1:
                            st.markdown(f"""
                            ### Parametric Method
                            
                            **For a {forecast_days}-day holding period:**
                            
                            - **{confidence_level}% confidence** that losses won't exceed **${abs(var_parametric):,.0f}**
                            - **Expected average loss** in worst {100-confidence_level}% scenarios: **${abs(es_parametric):,.0f}**
                            - **Daily volatility scaled by √{forecast_days} = {np.sqrt(forecast_days):.2f}**
                            
                            *Assumes normally distributed, independent daily returns.*
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
                            
                            *Captures compounding and non-normal distribution effects.*
                            """)
                        
                        # Download results
                        st.markdown("---")
                        st.subheader("Export Results")
                        
                        # Create comprehensive results dataframe
                        results_df = pd.DataFrame({
                            'Parameter': [
                                'Stock Symbol',
                                'Initial Investment',
                                'Time Horizon (Days)',
                                'Confidence Level',
                                'Daily Mean Return',
                                'Daily Standard Deviation',
                                f'{forecast_days}-Day Scaled Mean',
                                f'{forecast_days}-Day Scaled Volatility',
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
                                investment,
                                forecast_days,
                                f'{confidence_level}%',
                                f'{mean_return*100:.4f}%',
                                f'{std_dev*100:.4f}%',
                                f'{scaled_mean*100:.3f}%',
                                f'{scaled_vol*100:.3f}%',
                                -abs(var_parametric),
                                f'{abs(var_parametric)/investment*100:.2f}%',
                                -abs(es_parametric),
                                f'{abs(es_parametric)/investment*100:.2f}%',
                                -abs(var_monte_carlo),
                                f'{abs(var_monte_carlo)/investment*100:.2f}%',
                                -abs(es_monte_carlo),
                                f'{abs(es_monte_carlo)/investment*100:.2f}%',
                                f'{z_score:.4f}',
                                n_simulations,
                                -abs(worst_case_loss)
                            ]
                        })
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Results as CSV",
                            data=csv,
                            file_name=f"var_es_{stock_symbol}_{forecast_days}d_{confidence_level}pc_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("""
                **Common issues:**
                - Invalid stock symbol
                - Insufficient historical data for selected date range
                - Network connectivity issues
                - Extreme parameter values causing calculation errors
                
                **Suggestions:**
                1. Verify the stock symbol is valid
                2. Try a longer date range
                3. Check if the stock was trading during your selected period
                4. Try different statistical parameters
                """)
        
        else:
            # Display placeholder when no calculation has been performed
            st.info("👈 Enter your parameters on the left and click 'Calculate VaR & ES' to begin.")
            
            st.markdown(f"""
            ### Multi-Period Risk Calculation
            
            This calculator computes **{forecast_days if calculate_button else 'N'}-day Value at Risk and Expected Shortfall** using:
            
            1. **Time-scaled parametric formulas:**
               ```
               VaR_n = Investment × (μ×n - Z×σ×√n)
               ES_n = Investment × (μ×n - σ×√n × φ(Z)/(1-α))
               ```
            
            2. **Multi-period Monte Carlo simulation:**
               - Generates {n_simulations if calculate_button else 'thousands'} of {forecast_days if calculate_button else 'N'}-day return paths
               - Accounts for compounding effects
               - No normality assumption required
            
            **Key time scaling concepts:**
            - Volatility scales with √n (square root of time)
            - Expected return scales linearly with n
            - VaR typically increases with longer time horizons
            """)
            
            # Example calculation display
            with st.expander("Example Calculation Preview"):
                st.markdown("""
                **For a $100,000 investment with:**
                - Daily mean: 0.05%
                - Daily volatility: 1.5%
                - 10-day horizon
                - 95% confidence
                
                **Time-scaled parameters:**
                - 10-day mean: 0.5% (0.05% × 10)
                - 10-day volatility: 4.74% (1.5% × √10)
                
                **Approximate 10-day VaR:**
                ```
                VaR ≈ $100,000 × (0.005 - 1.645 × 0.0474) ≈ -$7,300
                ```
                
                This means with 95% confidence, losses won't exceed $7,300 over 10 days.
                """)
