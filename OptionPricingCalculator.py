import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Formula Functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price, d1, d2

# Page configuration
st.set_page_config(layout="wide")
st.title("Black-Scholes Option Pricing Calculator", text_alignment="center")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")

     st.markdown("""
    <style>
        /* Disable typing in selectbox */
        div[data-testid="stSelectbox"] input {
            pointer-events: none;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create the navigation menu
    page = st.selectbox(
        "Select Page:",
        ["Home", "Calculator"],
        label_visibility="collapsed")

# Home Page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Introduction</h2>", unsafe_allow_html=True)
    st.write("""
    Welcome to the **Black-Scholes Option Pricing Calculator**! This professional tool allows you to:
    
    - Calculate theoretical prices for European call and put options  
    - Understand the Black-Scholes model in detail  
    - Visualise option payoffs and Greeks  
    - Make informed financial decisions  
    
    Options are financial derivatives that give the holder the right, but not the obligation, 
    to buy or sell an underlying asset at a specified price (strike price) on or before a certain date (expiration).
    """)
    
    st.markdown("<h2 class='sub-header'>About the Black-Scholes Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    The **Black-Scholes model**, developed by Fischer Black and Myron Scholes in 1973, is a groundbreaking 
    mathematical model for pricing European-style options. It revolutionized the field of quantitative finance 
    and earned Scholes and Robert Merton the 1997 Nobel Prize in Economics.
    """)
    
    st.markdown("""
    The model is based on these key assumptions:
    
    1. **European Exercise**: Options can only be exercised at expiration.
    2. **No Dividends**: No dividends are paid during the option's life.
    3. **Efficient Markets**: No arbitrage opportunities exist.
    4. **Constant Parameters**: Risk-free rate and volatility are constant.
    5. **Lognormal Returns**: Stock returns follow a log-normal distribution.
    6. **Frictionless Markets**: No transaction costs or taxes.
    7. **Continuous Trading**: Trading occurs continuously.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='section-header'>Call Option Formula</h3>", unsafe_allow_html=True)
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("📐 Variables Explained", expanded=False):
            st.markdown("""
            Where:
            - $C$ = Call option price
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $T$ = Time to expiration (years)
            - $r$ = Risk-free interest rate
            - $N(·)$ = Cumulative standard normal distribution
            - $d_1 = \dfrac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
            - $d_2 = d_1 - \sigma\sqrt{T}$
            """)
    
    with col2:
        st.markdown("<h3 class='section-header'>Put Option Formula</h3>", unsafe_allow_html=True)
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("📐 Variables Explained", expanded=False):
            st.markdown("""
            Where:
            - $P$ = Put option price
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $T$ = Time to expiration (years)
            - $r$ = Risk-free interest rate
            - $N(·)$ = Cumulative standard normal distribution
            - $d_1 = \dfrac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
            - $d_2 = d_1 - \sigma\sqrt{T}$
            """)

    st.markdown("<h2 class='sub-header'>Understanding Option Moneyness</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    **Moneyness** indicates whether an option would be profitable if exercised immediately:
    """)

    moneyness_table = """
    | Status | Call Condition | Put Condition | Description |
    |--------|----------------|---------------|-------------|
    | *In-The-Money (ITM)* | $S > K$ | $S < K$ | Option has intrinsic value |
    | *At-The-Money (ATM)* | $S = K$ | $S = K$ | Stock price equals strike price |
    | *Out-of-The-Money (OTM)* | $S < K$ | $S > K$ | Option has no intrinsic value |
    """
    
    st.markdown(moneyness_table)
    
    # Add formulas with st.latex()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### **Intrinsic Value Formulas:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Call Option:**")
        st.latex(r"C_{\text{IV}} = \max(0, S - K)")
        
    with col2:
        st.markdown("**Put Option:**")
        st.latex(r"P_{\text{IV}} = \max(0, K - S)")
    
    st.markdown("""
    **Where:**
    - **$S$** = Current Stock Price
    - **$K$** = Strike Price
    - **$C_{\t{IV}}$** = Call Intrinsic Value
    - **$P_{\t{IV}}$** = Put Intrinsic Value
    """)

    st.markdown("<h2 class='sub-header'>Option Greeks</h2>", unsafe_allow_html=True)
    
    st.markdown('The "Greeks" measure the sensitivity of the option price to various factors:', unsafe_allow_html=True)
    
    greek_cols = st.columns(5)
    
    greeks = [
        ("Δ (Delta)", r"\frac{\partial V}{\partial S}", "Price sensitivity to underlying asset"),
        ("Γ (Gamma)", r"\frac{\partial^2 V}{\partial S^2}", "Delta's sensitivity to price changes"),
        ("Θ (Theta)", r"\frac{\partial V}{\partial t}", "Time decay of option value"),
        ("ν (Vega)", r"\frac{\partial V}{\partial \sigma}", "Sensitivity to volatility changes"),
        ("ρ (Rho)", r"\frac{\partial V}{\partial r}", "Sensitivity to interest rate changes")
    ]
    
    for col, (name, formula, desc) in zip(greek_cols, greeks):
        with col:
            st.markdown(f"<div style='text-align: center; font-weight: bold;'>{name}</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.latex(formula)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align: center; font-size: 0.9rem; color: #666;'>{desc}</div>", unsafe_allow_html=True)

# Calculator Page
elif page == "Calculator":
    
    st.markdown("**Instructions:** Fill in all parameters in the sidebar and click 'Calculate Option Price' to compute the theoretical option value.")
    
    with st.sidebar:
        st.markdown("### Input Parameters")
        
        # Option Type
        option_type = st.selectbox(
            "**Option Type**",
            ["Call", "Put"],
            help="Select 'Call' for right to buy, 'Put' for right to sell."
        )
        
        # Stock Price
        S = st.number_input(
            "**Asset Price (S)**",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Current market price of the underlying asset."
        )
        
        # Strike Price
        K = st.number_input(
            "**Strike Price (K)**",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Price at which the option can be exercised."
        )
        
        # Time to Expiration
        T = st.number_input(
            "**Time to Expiration (T in years)**",
            min_value=0.01,
            value=1.0,
            step=0.25,
            help="Time remaining until the option expires."
        )
        
        # Risk-Free Rate
        r = st.number_input(
            "**Risk-Free Rate (r)**",
            min_value=0.0,
            value=0.05,
            step=0.01,
            help="Annual risk-free interest rate (e.g., 0.05 for 5%)."
        )
        
        # Volatility
        sigma = st.slider(
            "**Volatility (σ)**",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Annualised volatility of the underlying asset (e.g., 0.2 for 20%)."
        )
        
        # Calculate button in sidebar
        calculate_button = st.button(
            "Calculate Option Price",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if calculate_button:
        st.markdown("<h2 class='section-header'>Calculation Results</h2>", unsafe_allow_html=True)
        
        # Calculate option price
        if option_type == "Call":
            price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
            option_formula = r"C = S_0 N(d_1) - K e^{-rT} N(d_2)"
        else:
            price, d1, d2 = black_scholes_put(S, K, T, r, sigma)
            option_formula = r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)"
        
        # Display result
        st.markdown(f"""
        <div style='text-align: center; background: linear-gradient(135deg, #1E88E5, #0D47A1); padding: 30px; border-radius: 12px; margin: 20px 0;'>
            <div style='color: white; font-size: 1.1rem; margin-bottom: 10px;'>
                {option_type} Option Price
            </div>
            <div style='color: white; font-size: 4rem; font-weight: bold; margin: 20px 0;'>
                ${price:.2f}
            </div>
            <div style='color: rgba(255,255,255,0.9); font-size: 0.9rem;'>
                Theoretical Price based on the Black-Scholes Model
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Greeks
        st.markdown("<h3 class='section-header'>Option Greeks</h3>", unsafe_allow_html=True)
        
        # Calculate Greeks
        if option_type == "Call":
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01
        rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == "Call" else norm.cdf(-d2)) * 0.01
        
        # Display Greeks in columns
        greek_cols = st.columns(5)
        greeks_data = [
            ("Δ (Delta)", f"{delta:.2f}", "Price sensitivity"),
            ("Γ (Gamma)", f"{gamma:.2f}", "Delta sensitivity"),
            ("Θ (Theta)", f"{theta:.2f}", "Daily time decay"),
            ("ν (Vega)", f"{vega:.2f}", "Volatility sensitivity"),
            ("ρ (Rho)", f"{rho:.2f}", "Interest rate sensitivity")
        ]
        
        for col, (name, value, desc) in zip(greek_cols, greeks_data):
            with col:
                st.metric(label=name, value=value)
                st.caption(desc)

        st.markdown("<h2 class='sub-header'>Payoff Visualisation</h2>", unsafe_allow_html=True)

        # Create payoff diagram
        spot_prices = np.linspace(S * 0.5, S * 1.5, 200)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Payoff at expiration (intrinsic value only)
        if option_type == "Call":
            payoffs = np.maximum(spot_prices - K, 0)
            label = "Call Option Payoff"
            color = "green"
            # For long call: profit = payoff - premium paid
            profit_loss = payoffs - price
            # Breakeven point: Spot price where payoff = premium
            breakeven_price = K + price
        else:
            payoffs = np.maximum(K - spot_prices, 0)
            label = "Put Option Payoff"
            color = "red"
            # For long put: profit = payoff - premium paid
            profit_loss = payoffs - price
            # Breakeven point: Spot price where payoff = premium
            breakeven_price = K - price
        
        # Plot 1: Payoff Diagram (Intrinsic Value)
        ax1.plot(spot_prices, payoffs, color=color, linewidth=3, label='Payoff at Expiration', alpha=0.8)
        ax1.fill_between(spot_prices, payoffs, alpha=0.2, color=color)
        ax1.axvline(x=S, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'Current Price (${S:.2f})')
        ax1.axvline(x=K, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Strike Price (${K:.2f})')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add breakeven line to payoff diagram
        ax1.axvline(x=breakeven_price, color='purple', linestyle=':', alpha=0.7, linewidth=1.5, 
                   label=f'Breakeven (${breakeven_price:.2f})')
        
        ax1.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax1.set_ylabel('Option Payoff ($)', fontsize=12)
        ax1.set_title(f'Long {option_type} Option: Payoff Diagram', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
       # Plot 2: Profit/Loss Diagram (Accounting for Premium)
        ax2.plot(spot_prices, profit_loss, color='purple', linewidth=3, label='Net Profit/Loss', alpha=0.8)
        ax2.fill_between(spot_prices, profit_loss, where=profit_loss>=0, alpha=0.2, color='green', label='Profit Zone')
        ax2.fill_between(spot_prices, profit_loss, where=profit_loss<0, alpha=0.2, color='red', label='Loss Zone')
        
        # Key reference lines for profit/loss
        ax2.axvline(x=S, color='blue', linestyle='--', alpha=0.7, linewidth=2, label=f'Current Price (${S:.2f})')
        ax2.axvline(x=K, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Strike Price (${K:.2f})')
        ax2.axvline(x=breakeven_price, color='purple', linestyle=':', alpha=0.7, linewidth=1.5, 
                   label=f'Breakeven (${breakeven_price:.2f})')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.axhline(y=-price, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, 
                   label=f'Max Loss: -${price:.2f}')
        
        # Add maximum profit line and include in legend
        if option_type == "Call":
            # For calls, max profit is theoretically unlimited, so show as asymptotic
            ax2.axhline(y=spot_prices[-1] - K - price, color='darkgreen', linestyle=':', alpha=0.3, linewidth=1,
                       label='Max Profit: Unlimited')
        else:  # Put option
            max_profit_value = K - spot_prices[0] - price
            ax2.axhline(y=max_profit_value, color='darkgreen', linestyle=':', alpha=0.3, linewidth=1,
                       label=f'Max Profit: ${max_profit_value:.2f}')
        
        ax2.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax2.set_ylabel('Net Profit/Loss ($)', fontsize=12)
        ax2.set_title(f'Long {option_type} Option: Profit/Loss Diagram', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add detailed explanation with key metrics
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown(f"""
            **Payoff Diagram (Intrinsic Value):**
            
            - **Current Price**: ${S:.2f}
            - **Strike Price**: ${K:.2f}
            - **Breakeven Price**: ${breakeven_price:.2f}
            - **Option Premium**: ${price:.2f}
            """)
            st.markdown("*The payoff shows the option's value at expiration without considering the premium paid.*")
        
        with col_exp2:
            max_loss = price
            if option_type == "Call":
                max_profit = "Unlimited"
                breakeven_explanation = f"Stock must rise above ${breakeven_price:.2f} to profit"
            else:
                max_profit = f"\\${K - 0 - price:.2f} (if stock goes to \\$0)"
                breakeven_explanation = f"Stock must fall below ${breakeven_price:.2f} to profit"
            
            st.markdown(f"""
            **Profit/Loss Analysis:**
            
            - **Maximum Loss**: -${max_loss:.2f} (premium paid)
            - **Maximum Profit**: {max_profit}
            - **Breakeven**: {breakeven_explanation}
            - **Moneyness**: {"In-The-Money (ITM)" if (option_type == "Call" and S > K) or (option_type == "Put" and S < K) else "Out-of-The-Money (OTM)" if (option_type == "Call" and S < K) or (option_type == "Put" and S > K) else "At-The-Money (ATM)"}
            
            *Net profit/loss accounts for the ${price:.2f} premium paid.*
            """)
    


    
