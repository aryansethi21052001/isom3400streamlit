import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Calculator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .formula {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #1E88E5;
        text-align: center;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Calculator", "Monte Carlo Simulation"])

# Black-Scholes Formula Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def monte_carlo_simulation(S, K, T, r, sigma, option_type='call', n_simulations=10000, n_steps=252):
    """Perform Monte Carlo simulation for option pricing"""
    dt = T / n_steps
    simulations = np.zeros((n_simulations, n_steps + 1))
    simulations[:, 0] = S
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        simulations[:, t] = simulations[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    if option_type == 'call':
        payoffs = np.maximum(simulations[:, -1] - K, 0)
    else:  # put option
        payoffs = np.maximum(K - simulations[:, -1], 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price, simulations, payoffs

# Home Page
if page == "Home":
    st.markdown("<h1 class='main-header'>Black-Scholes Option Pricing Calculator</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Introduction</h2>", unsafe_allow_html=True)
        st.write("""
        Welcome to the Black-Scholes Option Pricing Calculator! This application allows you to:
        
        - Calculate theoretical prices for European call and put options
        - Understand the Black-Scholes model
        - Perform Monte Carlo simulations to estimate option prices
        - Visualize the simulation results
        
        Options are financial derivatives that give the holder the right, but not the obligation, 
        to buy or sell an underlying asset at a specified price on or before a certain date.
        """)
        
        st.markdown("<h2 class='sub-header'>About the Black-Scholes Model</h2>", unsafe_allow_html=True)
        st.write("""
        The Black-Scholes model, developed by Fischer Black and Myron Scholes in 1973, 
        is a mathematical model for pricing options. It assumes:
        
        1. The option is European and can only be exercised at expiration
        2. No dividends are paid during the option's life
        3. Markets are efficient (no arbitrage opportunities)
        4. Risk-free rate and volatility are constant
        5. Returns are log-normally distributed
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/en/7/74/Black%E2%80%93Scholes_formula.jpg", 
                caption="Black-Scholes Formula", use_column_width=True)
    
    st.markdown("<h2 class='sub-header'>The Black-Scholes Formula</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Call Option:</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class='formula'>
        C = S × N(d₁) - K × e^(-rT) × N(d₂)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4>Put Option:</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class='formula'>
        P = K × e^(-rT) × N(-d₂) - S × N(-d₁)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Where:</h4>", unsafe_allow_html=True)
        st.markdown("""
        - **C** = Call option price
        - **P** = Put option price
        - **S** = Current stock price
        - **K** = Strike price
        - **T** = Time to expiration (years)
        - **r** = Risk-free interest rate
        - **σ** = Volatility of returns
        - **N(·)** = Cumulative standard normal distribution
        - **d₁** = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        - **d₂** = d₁ - σ√T
        """)
    
    st.markdown("<h2 class='sub-header'>How to Use This Calculator</h2>", unsafe_allow_html=True)
    st.write("""
    1. Navigate to the **Calculator** page from the sidebar
    2. Select your option type (Call or Put)
    3. Enter the required parameters
    4. View the calculated option price
    5. Go to **Monte Carlo Simulation** to run simulations and visualize results
    """)

# Calculator Page
elif page == "Calculator":
    st.markdown("<h1 class='main-header'>Option Pricing Calculator</h1>", unsafe_allow_html=True)
    
    # Input parameters in sidebar
    with st.sidebar:
        st.markdown("### Input Parameters")
        
        option_type = st.selectbox(
            "Option Type",
            ["Call", "Put"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input(
                "Stock Price (S)",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=1.0
            )
            K = st.number_input(
                "Strike Price (K)",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=1.0
            )
        
        with col2:
            T = st.number_input(
                "Time to Expiration (T in years)",
                min_value=0.01,
                max_value=50.0,
                value=1.0,
                step=0.25
            )
            r = st.number_input(
                "Risk-Free Rate (r as decimal)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.3f"
            )
        
        sigma = st.slider(
            "Volatility (σ as decimal)",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Annualized volatility of the underlying asset"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Parameters Summary</h2>", unsafe_allow_html=True)
        
        # Display parameters in a nice format
        params = {
            "Stock Price (S)": f"${S:.2f}",
            "Strike Price (K)": f"${K:.2f}",
            "Time to Expiration (T)": f"{T:.2f} years",
            "Risk-Free Rate (r)": f"{r*100:.2f}%",
            "Volatility (σ)": f"{sigma*100:.2f}%",
            "Option Type": option_type
        }
        
        for param, value in params.items():
            st.write(f"**{param}:** {value}")
        
        # Calculate and display option price
        if st.button("Calculate Option Price", type="primary"):
            with st.spinner("Calculating..."):
                
                if option_type == "Call":
                    price = black_scholes_call(S, K, T, r, sigma)
                else:
                    price = black_scholes_put(S, K, T, r, sigma)
                
                # Display result
                st.markdown(f"""
                <div class='result-box'>
                    <h3 style='color: #0D47A1; margin: 0;'>{option_type} Option Price</h3>
                    <h2 style='color: #1E88E5; margin: 10px 0;'>${price:.4f}</h2>
                    <p style='color: #666; margin: 0;'>Theoretical price based on Black-Scholes model</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Greeks (simplified calculation)
                st.markdown("<h3 class='sub-header'>Option Greeks (Approximate)</h3>", unsafe_allow_html=True)
                
                # Simplified Greeks calculation
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                
                if option_type == "Call":
                    delta = norm.cdf(d1)
                    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))) / 365
                else:
                    delta = norm.cdf(d1) - 1
                    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d1 + sigma * np.sqrt(T))) / 365
                
                vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
                rho = K * T * np.exp(-r * T) * (norm.cdf(d1 - sigma * np.sqrt(T)) if option_type == "Call" else norm.cdf(-d1 + sigma * np.sqrt(T))) * 0.01
                
                greeks = {
                    "Delta": f"{delta:.4f}",
                    "Gamma": f"{gamma:.4f}",
                    "Theta": f"{theta:.4f} per day",
                    "Vega": f"{vega:.4f} per 1% vol change",
                    "Rho": f"{rho:.4f} per 1% rate change"
                }
                
                cols = st.columns(5)
                for (greek, value), col in zip(greeks.items(), cols):
                    with col:
                        st.metric(label=greek, value=value)
    
    with col2:
        st.markdown("<h2 class='sub-header'>Visualization</h2>", unsafe_allow_html=True)
        
        # Create payoff diagram
        spot_prices = np.linspace(S * 0.5, S * 1.5, 100)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if option_type == "Call":
            intrinsic_values = np.maximum(spot_prices - K, 0)
            label = "Call Option Payoff"
            color = "green"
        else:
            intrinsic_values = np.maximum(K - spot_prices, 0)
            label = "Put Option Payoff"
            color = "red"
        
        ax.plot(spot_prices, intrinsic_values, color=color, linewidth=2, label=label)
        ax.axvline(x=S, color='blue', linestyle='--', alpha=0.7, label=f'Current Price (${S})')
        ax.axvline(x=K, color='orange', linestyle='--', alpha=0.7, label=f'Strike Price (${K})')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Stock Price at Expiration ($)')
        ax.set_ylabel('Option Payoff ($)')
        ax.set_title(f'{option_type} Option Payoff Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Monte Carlo Simulation Page
elif page == "Monte Carlo Simulation":
    st.markdown("<h1 class='main-header'>Monte Carlo Simulation</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Simulation Parameters</h2>", unsafe_allow_html=True)
        
        # Input parameters
        option_type_mc = st.selectbox(
            "Option Type",
            ["Call", "Put"],
            key="mc_option_type"
        )
        
        col1a, col2a = st.columns(2)
        with col1a:
            S_mc = st.number_input(
                "Stock Price (S)",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                key="mc_S"
            )
            K_mc = st.number_input(
                "Strike Price (K)",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                key="mc_K"
            )
        
        with col2a:
            T_mc = st.number_input(
                "Time to Expiration (T in years)",
                min_value=0.01,
                max_value=50.0,
                value=1.0,
                step=0.25,
                key="mc_T"
            )
            r_mc = st.number_input(
                "Risk-Free Rate (r as decimal)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.3f",
                key="mc_r"
            )
        
        sigma_mc = st.slider(
            "Volatility (σ as decimal)",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            key="mc_sigma"
        )
        
        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Higher number = more accurate but slower"
        )
        
        n_steps = st.slider(
            "Time Steps per Simulation",
            min_value=10,
            max_value=500,
            value=252,
            step=10,
            help="Number of time steps in each simulation (trading days)"
        )
        
        if st.button("Run Monte Carlo Simulation", type="primary"):
            st.session_state.run_simulation = True
    
    with col2:
        if 'run_simulation' in st.session_state and st.session_state.run_simulation:
            with st.spinner(f"Running {n_simulations:,} simulations..."):
                # Run Monte Carlo simulation
                progress_bar = st.progress(0)
                
                # Simulate progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                
                # Actual simulation
                mc_price, simulations, payoffs = monte_carlo_simulation(
                    S_mc, K_mc, T_mc, r_mc, sigma_mc,
                    option_type_mc.lower(),
                    n_simulations,
                    n_steps
                )
                
                # Calculate Black-Scholes price for comparison
                if option_type_mc == "Call":
                    bs_price = black_scholes_call(S_mc, K_mc, T_mc, r_mc, sigma_mc)
                else:
                    bs_price = black_scholes_put(S_mc, K_mc, T_mc, r_mc, sigma_mc)
                
                # Display results
                st.markdown("<h2 class='sub-header'>Simulation Results</h2>", unsafe_allow_html=True)
                
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.markdown(f"""
                    <div class='result-box'>
                        <h4>Monte Carlo Price</h4>
                        <h3>${mc_price:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown(f"""
                    <div class='result-box'>
                        <h4>Black-Scholes Price</h4>
                        <h3>${bs_price:.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write(f"**Difference:** ${abs(mc_price - bs_price):.4f} ({abs((mc_price - bs_price)/bs_price*100):.2f}%)")
                
                # Create visualizations
                st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)
                
                # Plot 1: Sample price paths
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                n_sample_paths = min(50, n_simulations)
                for i in range(n_sample_paths):
                    ax1.plot(simulations[i, :], alpha=0.5, linewidth=0.5)
                
                ax1.axhline(y=K_mc, color='red', linestyle='--', label=f'Strike Price (${K_mc})')
                ax1.set_xlabel('Time Steps')
                ax1.set_ylabel('Stock Price ($)')
                ax1.set_title(f'Sample Monte Carlo Simulation Paths (First {n_sample_paths} paths)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # Plot 2: Payoff distribution
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.hist(payoffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(x=np.mean(payoffs), color='red', linestyle='--', 
                          label=f'Mean Payoff: ${np.mean(payoffs):.2f}')
                ax2.set_xlabel('Payoff at Expiration ($)')
                ax2.set_ylabel('Frequency')
                ax2.set_title(f'Distribution of Option Payoffs ({n_simulations:,} simulations)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Plot 3: Convergence of Monte Carlo price
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                cumulative_mean = np.cumsum(payoffs) / np.arange(1, n_simulations + 1)
                discounted_cumulative_mean = cumulative_mean * np.exp(-r_mc * T_mc)
                
                ax3.plot(discounted_cumulative_mean, label='Monte Carlo Estimate', color='blue')
                ax3.axhline(y=bs_price, color='red', linestyle='--', 
                          label=f'Black-Scholes Price: ${bs_price:.4f}')
                ax3.set_xlabel('Number of Simulations')
                ax3.set_ylabel('Option Price ($)')
                ax3.set_title('Convergence of Monte Carlo Estimate')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Add confidence interval (95%)
                if n_simulations > 100:
                    cumulative_std = np.std(payoffs) / np.sqrt(np.arange(1, n_simulations + 1))
                    ci_upper = discounted_cumulative_mean + 1.96 * cumulative_std * np.exp(-r_mc * T_mc)
                    ci_lower = discounted_cumulative_mean - 1.96 * cumulative_std * np.exp(-r_mc * T_mc)
                    ax3.fill_between(range(n_simulations), ci_lower, ci_upper, alpha=0.2, color='blue')
                
                st.pyplot(fig3)
                
                # Display simulation statistics
                st.markdown("<h2 class='sub-header'>Simulation Statistics</h2>", unsafe_allow_html=True)
                
                stats_cols = st.columns(4)
                stats = {
                    "Mean Payoff": f"${np.mean(payoffs):.2f}",
                    "Std Dev": f"${np.std(payoffs):.2f}",
                    "Min Payoff": f"${np.min(payoffs):.2f}",
                    "Max Payoff": f"${np.max(payoffs):.2f}",
                    "Probability of Exercise": f"{(payoffs > 0).sum() / n_simulations * 100:.2f}%",
                    "Standard Error": f"${np.std(payoffs) / np.sqrt(n_simulations) * np.exp(-r_mc * T_mc):.4f}"
                }
                
                for i, (stat_name, stat_value) in enumerate(stats.items()):
                    with stats_cols[i % 4]:
                        st.metric(label=stat_name, value=stat_value)
        
        else:
            st.info("👈 Configure the simulation parameters and click 'Run Monte Carlo Simulation' to start")
            
            # Placeholder image/explanation
            st.markdown("""
            <div style='text-align: center; padding: 40px; background-color: #f5f5f5; border-radius: 10px;'>
                <h3 style='color: #666;'>Monte Carlo Simulation Ready</h3>
                <p>Configure the parameters on the left and run the simulation to see results here.</p>
                <p>The simulation will show:</p>
                <ul style='text-align: left; display: inline-block;'>
                    <li>Sample price paths</li>
                    <li>Payoff distribution</li>
                    <li>Convergence analysis</li>
                    <li>Comparison with Black-Scholes price</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This application demonstrates the Black-Scholes option pricing model and Monte Carlo simulation for educational purposes.

**Assumptions:**
- European options only
- No dividends
- Constant volatility
- Risk-free rate is constant
""")
