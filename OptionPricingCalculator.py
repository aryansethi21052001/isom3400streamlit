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
st.title("Black-Scholes Option Pricing Calculator", text_alignment="center")

# Navigation
tab1, tab2 = st.tabs(["Home", "Calculator"])

# Home Page (unchanged)
with tab1:
    st.subheader("Introduction")
    st.write("""
    Welcome to the **Black-Scholes Option Pricing Calculator**! This professional tool allows you to:
    
    - Calculate theoretical prices for European call and put options  
    - Understand the Black-Scholes model in detail  
    - Visualise option payoffs and Greeks  
    - Make informed financial decisions  
    
    Options are financial derivatives that give the holder the right, but not the obligation, 
    to buy or sell an underlying asset at a specified price (strike price) on or before a certain date (expiration).
    """)
    
    st.subheader("About the Black-Scholes Model")
    
    st.write("""
    The **Black-Scholes model**, developed by Fischer Black and Myron Scholes in 1973, is a groundbreaking 
    mathematical model for pricing European-style options. It revolutionized the field of quantitative finance 
    and earned Scholes and Robert Merton the 1997 Nobel Prize in Economics.
    """)
    
    st.write("""
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
    
    st.markdown(moneyness_table, text_alignment="center")
    
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

    st.subheader("Option Greeks")
    
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
with tab2:
    
    st.write("**Instructions:** Fill in all parameters in the sidebar and click 'Calculate Option Price' to compute the theoretical option value.")
    
    with st.form("Option_Pricing_Calculator"):
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
            max_value=1.0,
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
        calculate_button = st.form_submit_button(
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
        
        # Add position selection
        position_type = st.radio(
            "**Select Position to Visualize:**",
            ["Long Position", "Short Position", "Both Positions"],
            horizontal=True,
            help="Long = Buying the option, Short = Selling/writing the option"
        )
        
        # Create payoff diagram
        spot_prices = np.linspace(S * 0.5, S * 1.5, 200)
        
        # Calculate payoffs for call and put options
        if option_type == "Call":
            intrinsic_value = np.maximum(spot_prices - K, 0)
            option_color = "green"
            option_label = "Call Option"
            # Breakeven points
            breakeven_long = K + price
            breakeven_short = K + price  # Same breakeven but opposite profit/loss
        else:
            intrinsic_value = np.maximum(K - spot_prices, 0)
            option_color = "red"
            option_label = "Put Option"
            # Breakeven points
            breakeven_long = K - price
            breakeven_short = K - price  # Same breakeven but opposite profit/loss
        
        # Calculate profit/loss for long and short positions
        # Long position: profit = payoff - premium paid
        profit_loss_long = intrinsic_value - price
        # Short position: profit = premium received - payoff (if exercised)
        profit_loss_short = price - intrinsic_value
        
        # Determine how many plots to show
        if position_type == "Both Positions":
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            axes = [(ax1, ax2), (ax3, ax4)]
            positions = [("Long", profit_loss_long, "Buyer"), ("Short", profit_loss_short, "Seller")]
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if position_type == "Long Position":
                positions = [("Long", profit_loss_long, "Buyer")]
                axes = [(ax1, ax2)]
            else:  # Short Position
                positions = [("Short", profit_loss_short, "Seller")]
                axes = [(ax1, ax2)]
        
        # Plot for each position
        for (pos_name, profit_loss, role), (ax_payoff, ax_profit) in zip(positions, axes):
            # Colors for long vs short
            if pos_name == "Long":
                pos_color = option_color
                pos_prefix = "Long"
            else:
                pos_color = "blue" if option_type == "Call" else "orange"
                pos_prefix = "Short"
            
            # Plot 1: Payoff Diagram (Intrinsic Value)
            if pos_name == "Long":
                ax_payoff.plot(spot_prices, intrinsic_value, color=pos_color, linewidth=3, 
                              label=f'{pos_prefix} {option_label} Payoff', alpha=0.8)
            else:
                ax_payoff.plot(spot_prices, -intrinsic_value, color=pos_color, linewidth=3, 
                              label=f'{pos_prefix} {option_label} Payoff', alpha=0.8)
            
            ax_payoff.fill_between(spot_prices, intrinsic_value if pos_name == "Long" else -intrinsic_value, 
                                  alpha=0.2, color=pos_color)
            
            # Key reference lines
            ax_payoff.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                            label=f'Current Price (${S:.2f})')
            ax_payoff.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                            label=f'Strike Price (${K:.2f})')
            ax_payoff.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            ax_payoff.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
            ax_payoff.set_ylabel('Option Payoff ($)', fontsize=12)
            ax_payoff.set_title(f'{pos_prefix} {option_type} Option: Payoff Diagram\n({role} Perspective)', 
                              fontsize=14, fontweight='bold')
            ax_payoff.legend(loc='best')
            ax_payoff.grid(True, alpha=0.3)
            
            # Plot 2: Profit/Loss Diagram (Accounting for Premium)
            ax_profit.plot(spot_prices, profit_loss, color='purple', linewidth=3, 
                          label='Net Profit/Loss', alpha=0.8)
            
            # Fill profit and loss zones
            profit_zone = profit_loss >= 0
            loss_zone = profit_loss < 0
            
            ax_profit.fill_between(spot_prices, profit_loss, where=profit_zone, 
                                  alpha=0.2, color='green', label='Profit Zone')
            ax_profit.fill_between(spot_prices, profit_loss, where=loss_zone, 
                                  alpha=0.2, color='red', label='Loss Zone')
            
            # Key reference lines for profit/loss
            ax_profit.axvline(x=S, color='black', linestyle='--', alpha=0.7, linewidth=2, 
                            label=f'Current Price (${S:.2f})')
            ax_profit.axvline(x=K, color='gray', linestyle='--', alpha=0.7, linewidth=2, 
                            label=f'Strike Price (${K:.2f})')
            
            # Breakeven line
            breakeven = breakeven_long if pos_name == "Long" else breakeven_short
            ax_profit.axvline(x=breakeven, color='purple', linestyle=':', alpha=0.7, 
                            linewidth=1.5, label=f'Breakeven (${breakeven:.2f})')
            
            ax_profit.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # Maximum loss and profit lines
            if pos_name == "Long":
                max_loss = -price
                ax_profit.axhline(y=max_loss, color='red', linestyle=':', alpha=0.5, 
                                linewidth=1.5, label=f'Max Loss: ${max_loss:.2f}')
                
                if option_type == "Call":
                    # For long calls, max profit is unlimited
                    ax_profit.axhline(y=spot_prices[-1] - K - price, color='darkgreen', 
                                    linestyle=':', alpha=0.3, linewidth=1,
                                    label='Max Profit: Unlimited')
                else:
                    # For long puts, max profit is K - price (if stock goes to 0)
                    max_profit_value = K - price
                    ax_profit.axhline(y=max_profit_value, color='darkgreen', linestyle=':', 
                                    alpha=0.3, linewidth=1,
                                    label=f'Max Profit: ${max_profit_value:.2f}')
            else:
                # Short position
                max_profit = price
                ax_profit.axhline(y=max_profit, color='darkgreen', linestyle=':', alpha=0.5, 
                                linewidth=1.5, label=f'Max Profit: ${max_profit:.2f}')
                
                if option_type == "Call":
                    # For short calls, max loss is unlimited
                    ax_profit.axhline(y=-(spot_prices[-1] - K) + price, color='red', 
                                    linestyle=':', alpha=0.3, linewidth=1,
                                    label='Max Loss: Unlimited')
                else:
                    # For short puts, max loss is K - price (if stock goes to 0)
                    max_loss_value = -(K) + price
                    ax_profit.axhline(y=max_loss_value, color='red', linestyle=':', 
                                    alpha=0.3, linewidth=1,
                                    label=f'Max Loss: ${max_loss_value:.2f}')
            
            ax_profit.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
            ax_profit.set_ylabel('Net Profit/Loss ($)', fontsize=12)
            ax_profit.set_title(f'{pos_prefix} {option_type} Option: Profit/Loss Diagram\n({role} Perspective)', 
                              fontsize=14, fontweight='bold')
            ax_profit.legend(loc='best')
            ax_profit.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add detailed explanation based on selected position(s)
        st.markdown("### **Position Analysis**")
        
        if position_type == "Both Positions":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### **Long Position (Buyer)**")
                if option_type == "Call":
                    st.markdown(f"""
                    **For Long Call:**
                    - **Cost**: Pay ${price:.2f} premium
                    - **Breakeven**: Stock > ${breakeven_long:.2f}
                    - **Max Loss**: -${price:.2f} (premium paid)
                    - **Max Profit**: Unlimited
                    - **When to use**: Bullish outlook, expect stock to rise significantly
                    """)
                else:
                    st.markdown(f"""
                    **For Long Put:**
                    - **Cost**: Pay ${price:.2f} premium
                    - **Breakeven**: Stock < ${breakeven_long:.2f}
                    - **Max Loss**: -${price:.2f} (premium paid)
                    - **Max Profit**: ${K - price:.2f} (if stock goes to $0)
                    - **When to use**: Bearish outlook, expect stock to fall
                    """)
            
            with col2:
                st.markdown("#### **Short Position (Seller/Writer)**")
                if option_type == "Call":
                    st.markdown(f"""
                    **For Short Call:**
                    - **Income**: Receive ${price:.2f} premium
                    - **Breakeven**: Stock < ${breakeven_short:.2f}
                    - **Max Profit**: ${price:.2f} (premium received)
                    - **Max Loss**: Unlimited
                    - **When to use**: Neutral to bearish, expect stock to stay flat or fall
                    - **Risk**: Must deliver shares if exercised (naked call)
                    """)
                else:
                    st.markdown(f"""
                    **For Short Put:**
                    - **Income**: Receive ${price:.2f} premium
                    - **Breakeven**: Stock > ${breakeven_short:.2f}
                    - **Max Profit**: ${price:.2f} (premium received)
                    - **Max Loss**: ${K - price:.2f} (if stock goes to $0)
                    - **When to use**: Neutral to bullish, expect stock to stay flat or rise
                    - **Risk**: Must buy shares if exercised
                    """)
        
        else:
            # Single position analysis
            if position_type == "Long Position":
                st.markdown("#### **Long Position Analysis (Buyer)**")
                if option_type == "Call":
                    st.markdown(f"""
                    - **Premium Paid**: ${price:.2f}
                    - **Breakeven Price**: ${breakeven_long:.2f}
                    - **Maximum Loss**: -${price:.2f} (limited to premium paid)
                    - **Maximum Profit**: Unlimited (stock can rise indefinitely)
                    - **Profit Condition**: Stock price > ${breakeven_long:.2f}
                    - **Strategy**: Bullish - You profit if the stock rises significantly
                    """)
                else:
                    st.markdown(f"""
                    - **Premium Paid**: ${price:.2f}
                    - **Breakeven Price**: ${breakeven_long:.2f}
                    - **Maximum Loss**: -${price:.2f} (limited to premium paid)
                    - **Maximum Profit**: ${K - price:.2f} (if stock goes to $0)
                    - **Profit Condition**: Stock price < ${breakeven_long:.2f}
                    - **Strategy**: Bearish - You profit if the stock falls
                    """)
            else:  # Short Position
                st.markdown("#### **Short Position Analysis (Seller/Writer)**")
                if option_type == "Call":
                    st.markdown(f"""
                    - **Premium Received**: ${price:.2f}
                    - **Breakeven Price**: ${breakeven_short:.2f}
                    - **Maximum Profit**: ${price:.2f} (limited to premium received)
                    - **Maximum Loss**: Unlimited (stock can rise indefinitely)
                    - **Profit Condition**: Stock price < ${breakeven_short:.2f}
                    - **Strategy**: Neutral to bearish - You profit if stock stays flat or falls
                    - **Risk**: Naked call - Must deliver shares if exercised (infinite risk)
                    """)
                else:
                    st.markdown(f"""
                    - **Premium Received**: ${price:.2f}
                    - **Breakeven Price**: ${breakeven_short:.2f}
                    - **Maximum Profit**: ${price:.2f} (limited to premium received)
                    - **Maximum Loss**: ${K - price:.2f} (if stock goes to $0)
                    - **Profit Condition**: Stock price > ${breakeven_short:.2f}
                    - **Strategy**: Neutral to bullish - You profit if stock stays flat or rises
                    - **Risk**: Must buy shares at strike price if exercised
                    """)
        
        # Moneyness information
        st.markdown("### **Current Option Status**")
        if option_type == "Call":
            moneyness = "In-The-Money (ITM)" if S > K else "Out-of-The-Money (OTM)" if S < K else "At-The-Money (ATM)"
            intrinsic_val = max(S - K, 0)
            time_val = price - intrinsic_val
        else:
            moneyness = "In-The-Money (ITM)" if S < K else "Out-of-The-Money (OTM)" if S > K else "At-The-Money (ATM)"
            intrinsic_val = max(K - S, 0)
            time_val = price - intrinsic_val
        
        st.markdown(f"""
        - **Moneyness**: {moneyness}
        - **Intrinsic Value**: ${intrinsic_val:.2f}
        - **Time Value**: ${time_val:.2f}
        - **Total Premium**: ${price:.2f}
        """)
