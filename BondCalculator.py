import streamlit as st
import math
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Bond Price Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3B82F6;
    }
    .price-difference {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .positive-diff {
        color: #10B981;
    }
    .negative-diff {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Constants
FREQUENCY_OPTIONS = {
    "Annual": 1,
    "Semi-annual": 2,
    "Quarterly": 4,
    "Monthly": 12,
    "Daily": 365
}

class BondCalculator:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'bond_type' not in st.session_state:
            st.session_state.bond_type = "Zero Coupon Bond"
        if 'calculation_done' not in st.session_state:
            st.session_state.calculation_done = False
    
    def calculate_discrete_price(self, params):
        """Calculate bond price using discrete compounding"""
        if params['bond_type'] == "Zero Coupon Bond":
            # Zero Coupon Bond discrete model: P = F / (1 + r/n)^(n*t)
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            price = params['principal'] / ((1 + r_per_period) ** n_periods)
            return price
        else:
            # Coupon Bond discrete model
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            # Calculate coupon payment per period
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            # Calculate PV of coupon payments
            pv_coupons = 0
            for k in range(1, int(n_periods) + 1):
                pv_coupons += coupon_payment / ((1 + r_per_period) ** k)
            
            # Calculate PV of principal
            pv_principal = params['principal'] / ((1 + r_per_period) ** n_periods)
            
            price = pv_coupons + pv_principal
            return price
    
    def calculate_continuous_price(self, params):
        """Calculate bond price using continuous compounding"""
        if params['bond_type'] == "Zero Coupon Bond":
            # Zero Coupon Bond continuous model: P = F * e^(-r*t)
            price = params['principal'] * math.exp(-params['interest_rate'] * params['maturity'])
            return price
        else:
            # Coupon Bonds with continuous discounting
            price = 0
            # Calculate coupon payment
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            # Calculate present value of coupon payments
            for i in range(1, int(params['maturity'] * params['payments_per_year']) + 1):
                t = i / params['payments_per_year']
                if t <= params['maturity']:
                    price += coupon_payment * math.exp(-params['interest_rate'] * t)
            
            # Add present value of principal
            price += params['principal'] * math.exp(-params['interest_rate'] * params['maturity'])
            return price
    
    def calculate_macaulay_duration(self, params, discrete_price):
        """Calculate Macaulay Duration for the bond"""
        if params['bond_type'] == "Zero Coupon Bond":
            # For zero coupon bonds, Macaulay Duration = time to maturity
            macaulay_duration = params['maturity']
            return macaulay_duration
        else:
            # For coupon bonds
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            # Calculate weighted average time of cash flows
            weighted_sum = 0
            
            # Weight coupon payments
            for k in range(1, int(n_periods) + 1):
                time_years = k / params['compounding_periods']
                cash_flow = coupon_payment
                present_value = cash_flow / ((1 + r_per_period) ** k)
                weighted_sum += time_years * present_value
            
            # Weight principal payment at maturity
            time_years = params['maturity']
            cash_flow = params['principal']
            present_value = cash_flow / ((1 + r_per_period) ** n_periods)
            weighted_sum += time_years * present_value
            
            # Macaulay Duration = weighted_sum / bond price
            macaulay_duration = weighted_sum / discrete_price
            return macaulay_duration
    
    def calculate_modified_duration(self, params, macaulay_duration):
        """Calculate Modified Duration for the bond"""
        if params['bond_type'] == "Zero Coupon Bond":
            # For zero coupon bonds
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['compounding_periods']
            
            # Modified Duration = Macaulay Duration / (1 + yield per period)
            modified_duration = macaulay_duration / (1 + r_per_period)
            return modified_duration
        else:
            # For coupon bonds
            r_per_period = params['interest_rate'] / params['compounding_periods']
            
            # Modified Duration = Macaulay Duration / (1 + yield per period)
            modified_duration = macaulay_duration / (1 + r_per_period)
            return modified_duration
    
    def calculate_convexity(self, params, discrete_price):
        """Calculate Convexity for the bond"""
        if params['bond_type'] == "Zero Coupon Bond":
            # For zero coupon bonds
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            
            # Convexity = [n*(n+1)] / [(1+r_per_period)^2]
            convexity = (n_periods * (n_periods + 1)) / ((1 + r_per_period) ** 2)
            convexity = convexity / (params['compounding_periods'] ** 2)  # Convert to years²
            return convexity
        else:
            # For coupon bonds
            r_per_period = params['interest_rate'] / params['compounding_periods']
            n_periods = params['maturity'] * params['compounding_periods']
            coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
            
            # Calculate convexity
            convexity_sum = 0
            
            # Convexity contributions from coupon payments
            for k in range(1, int(n_periods) + 1):
                time_years = k / params['compounding_periods']
                cash_flow = coupon_payment
                present_value = cash_flow / ((1 + r_per_period) ** k)
                convexity_sum += time_years * (time_years + 1/params['compounding_periods']) * present_value
            
            # Convexity contribution from principal payment
            time_years = params['maturity']
            cash_flow = params['principal']
            present_value = cash_flow / ((1 + r_per_period) ** n_periods)
            convexity_sum += time_years * (time_years + 1/params['compounding_periods']) * present_value
            
            # Convexity = convexity_sum / [bond_price * (1+r_per_period)^2]
            convexity = convexity_sum / (discrete_price * ((1 + r_per_period) ** 2))
            return convexity
    
    def create_input_section(self):
        """Create the input section in sidebar"""
        st.sidebar.header("📊 Bond Parameters")
        
        # Bond type selection
        bond_type = st.sidebar.selectbox(
            "Bond Type",
            ["Zero Coupon Bond", "Coupon Bond"],
            index=0 if st.session_state.bond_type == "Zero Coupon Bond" else 1,
            key="bond_type_select"
        )
        st.session_state.bond_type = bond_type
        
        # Principal amount
        principal = st.sidebar.number_input(
            "Principal/Face Value ($)",
            min_value=0.01,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
        
        # Maturity in years
        maturity = st.sidebar.slider(
            "Time to Maturity (years)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            format="%.1f"
        )
        
        # Interest rate (yield)
        interest_rate = st.sidebar.slider(
            "Annual Interest Rate (Yield)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            format="%.1f%%"
        ) / 100  # Convert to decimal
        
        # Coupon rate (only for coupon bonds)
        coupon_rate = 0.0
        if bond_type == "Coupon Bond":
            coupon_rate = st.sidebar.slider(
                "Annual Coupon Rate",
                min_value=0.0,
                max_value=100.0,
                value=3.5,
                step=0.1,
                format="%.1f%%"
            ) / 100  # Convert to decimal
        
        # Frequency selection
        frequency = st.sidebar.selectbox(
            "Compounding/Payment Frequency",
            list(FREQUENCY_OPTIONS.keys()),
            index=0  # Default to Annual
        )
        compounding_periods = FREQUENCY_OPTIONS[frequency]
        payments_per_year = FREQUENCY_OPTIONS[frequency] if bond_type == "Coupon Bond" else 1
        
        # Calculate button
        if st.sidebar.button("Calculate Bond Price", type="primary", use_container_width=True):
            params = {
                'bond_type': bond_type,
                'principal': principal,
                'maturity': maturity,
                'interest_rate': interest_rate,
                'coupon_rate': coupon_rate,
                'compounding_periods': compounding_periods,
                'payments_per_year': payments_per_year,
                'frequency_name': frequency
            }
            st.session_state.calculation_params = params
            st.session_state.calculation_done = True
        
        # Reset button
        if st.sidebar.button("Reset", use_container_width=True):
            st.session_state.calculation_done = False
    
    def display_results(self):
        """Display calculation results"""
        params = st.session_state.calculation_params
        
        # Calculate prices
        discrete_price = self.calculate_discrete_price(params)
        continuous_price = self.calculate_continuous_price(params)
        
        # Calculate differences
        price_diff = continuous_price - discrete_price
        price_diff_pct = (price_diff / discrete_price) * 100 if discrete_price != 0 else 0
        
        # Main results header
        st.markdown(f'<h1 class="main-header">📈 Bond Price Calculator</h1>', unsafe_allow_html=True)
        
        # Bond Information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sub-header">📋 Bond Information</div>', unsafe_allow_html=True)
            
            info_data = {
                "Parameter": ["Bond Type", "Principal/Face Value", "Time to Maturity", 
                             "Annual Interest Rate", "Compounding Frequency"],
                "Value": [
                    params['bond_type'],
                    f"${params['principal']:,.2f}",
                    f"{params['maturity']:.1f} years",
                    f"{params['interest_rate']*100:.2f}%",
                    params['frequency_name']
                ]
            }
            
            if params['bond_type'] == "Coupon Bond":
                info_data["Parameter"].insert(4, "Annual Coupon Rate")
                info_data["Value"].insert(4, f"{params['coupon_rate']*100:.2f}%")
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<div class="sub-header">💰 Price Results</div>', unsafe_allow_html=True)
            
            # Display prices in metric boxes
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    label="Discrete Model",
                    value=f"${discrete_price:,.2f}",
                    delta=None
                )
            with cols[1]:
                st.metric(
                    label="Continuous Model",
                    value=f"${continuous_price:,.2f}",
                    delta=None
                )
            
            # Price difference
            diff_color = "positive-diff" if price_diff >= 0 else "negative-diff"
            diff_sign = "+" if price_diff >= 0 else ""
            st.markdown(
                f'<div class="result-box">'
                f'<h4>Price Difference</h4>'
                f'<p class="price-difference {diff_color}">'
                f'Continuous - Discrete: {diff_sign}${price_diff:,.2f} '
                f'({diff_sign}{price_diff_pct:.2f}%)'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Create tabs for additional views
        tab1, tab2, tab3 = st.tabs(["Detailed Analysis", "Duration & Risk", "Explanation"])
        
        with tab1:
            self.display_detailed_analysis(params, discrete_price, continuous_price)
        
        with tab2:
            self.display_duration_analysis(params, discrete_price)
        
        with tab3:
            self.display_explanation(params)
    
    def display_detailed_analysis(self, params, discrete_price, continuous_price):
        """Display detailed analysis including cash flows"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📅 Payment Schedule (Coupon Bonds Only)")
            if params['bond_type'] == "Coupon Bond":
                # Generate payment schedule
                num_payments = int(params['maturity'] * params['payments_per_year'])
                coupon_payment = (params['coupon_rate'] * params['principal']) / params['payments_per_year']
                
                payments = []
                for i in range(1, num_payments + 1):
                    payment_time = i / params['payments_per_year']
                    if payment_time <= params['maturity']:
                        # Discrete discount factor
                        r_per_period = params['interest_rate'] / params['compounding_periods']
                        discount_factor_discrete = 1 / ((1 + r_per_period) ** (i))
                        pv_discrete = coupon_payment * discount_factor_discrete
                        
                        # Continuous discount factor
                        discount_factor_continuous = math.exp(-params['interest_rate'] * payment_time)
                        pv_continuous = coupon_payment * discount_factor_continuous
                        
                        payments.append({
                            "Payment #": i,
                            "Time (years)": f"{payment_time:.2f}",
                            "Coupon Payment": f"${coupon_payment:.2f}",
                            "PV (Discrete)": f"${pv_discrete:.2f}",
                            "PV (Continuous)": f"${pv_continuous:.2f}"
                        })
                
                # Add principal payment at maturity
                discount_factor_discrete_principal = 1 / ((1 + params['interest_rate']/params['compounding_periods']) ** 
                                                        (params['maturity'] * params['compounding_periods']))
                discount_factor_continuous_principal = math.exp(-params['interest_rate'] * params['maturity'])
                
                payments.append({
                    "Payment #": "Principal",
                    "Time (years)": f"{params['maturity']:.2f}",
                    "Coupon Payment": f"${params['principal']:.2f}",
                    "PV (Discrete)": f"${params['principal'] * discount_factor_discrete_principal:.2f}",
                    "PV (Continuous)": f"${params['principal'] * discount_factor_continuous_principal:.2f}"
                })
                
                payments_df = pd.DataFrame(payments)
                st.dataframe(payments_df, use_container_width=True)
            else:
                st.info("Payment schedule is only applicable for Coupon Bonds.")
        
        with col2:
            st.markdown("##### 🔍 Key Metrics")
            
            # Calculate additional metrics
            if params['bond_type'] == "Coupon Bond":
                # Current yield
                coupon_payment_annual = params['coupon_rate'] * params['principal']
                current_yield = (coupon_payment_annual / discrete_price) * 100
                
                # Yield to maturity approximation
                ytm = params['interest_rate'] * 100
                
                metrics_data = {
                    "Metric": ["Current Yield", "Yield to Maturity", "Duration (approx.)"],
                    "Value": [
                        f"{current_yield:.2f}%",
                        f"{ytm:.2f}%",
                        f"{params['maturity']:.1f} years"
                    ]
                }
            else:
                metrics_data = {
                    "Metric": ["Discount Rate", "Compounding Periods"],
                    "Value": [
                        f"{params['interest_rate']*100:.2f}%",
                        f"{params['compounding_periods']}"
                    ]
                }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Price comparison using Streamlit's built-in chart
            st.markdown("##### Price Comparison")
            price_data = pd.DataFrame({
                'Model': ['Discrete', 'Continuous'],
                'Price': [discrete_price, continuous_price]
            })
            
            st.bar_chart(price_data.set_index('Model'))
    
    def display_duration_analysis(self, params, discrete_price):
        """Display duration and risk analysis"""
        st.markdown('<div class="sub-header">📊 Duration & Risk Analysis</div>', unsafe_allow_html=True)
        
        # Calculate Macaulay Duration
        macaulay_duration = self.calculate_macaulay_duration(params, discrete_price)
        
        # Calculate Modified Duration
        modified_duration = self.calculate_modified_duration(params, macaulay_duration)
        
        # Calculate Convexity
        convexity = self.calculate_convexity(params, discrete_price)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Macaulay Duration",
                value=f"{macaulay_duration:.2f} years",
                delta=None,
                help="Weighted average time to receive cash flows"
            )
        
        with col2:
            st.metric(
                label="Modified Duration",
                value=f"{modified_duration:.2f} years",
                delta=None,
                help="Price sensitivity to interest rate changes"
            )
        
        with col3:
            st.metric(
                label="Convexity",
                value=f"{convexity:.2f}",
                delta=None,
                help="Curvature of price-yield relationship"
            )
        
        # Display what duration means
        st.markdown("### 📈 Understanding Duration")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            #### **Macaulay Duration**
            - **Definition**: Weighted average time to receive cash flows
            - **Formula**: Σ(t × PV(CF_t)) / Price
            - **Interpretation**: 
                - Higher duration = More price sensitivity
                - Zero-coupon bond duration = Maturity
                - Coupon bond duration < Maturity
            """)
        
        with col_info2:
            st.markdown("""
            #### **Modified Duration**
            - **Definition**: Price sensitivity to yield changes
            - **Formula**: Macaulay Duration / (1 + yield/periods)
            - **Application**: 
                - ΔPrice ≈ -Modified Duration × ΔYield × Price
                - For 1% yield increase: Price ↓ by Modified Duration %
                - Key measure for interest rate risk
            """)
        
        with col_info3:
            st.markdown("""
            #### **Convexity**
            - **Definition**: Measures curvature of price-yield curve
            - **Importance**: 
                - Adjusts duration for large yield changes
                - Positive convexity = Price increases more than duration predicts
                - Higher convexity = Better risk-return profile
                - Always positive for non-callable bonds
            """)
        
        # Price change calculations
        st.markdown("### 🎯 Price Sensitivity Estimates")
        
        # Create input for yield change
        col_yield1, col_yield2 = st.columns(2)
        
        with col_yield1:
            yield_change = st.slider(
                "Yield Change (percentage points)",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.1f%%"
            ) / 100  # Convert to decimal
        
        with col_yield2:
            # Calculate estimated price changes
            price_change_duration = -modified_duration * yield_change * discrete_price
            price_change_convexity = 0.5 * convexity * (yield_change ** 2) * discrete_price
            total_price_change = price_change_duration + price_change_convexity
            new_price_estimate = discrete_price + total_price_change
            
            st.metric(
                label="Estimated New Price",
                value=f"${new_price_estimate:,.2f}",
                delta=f"{total_price_change:,.2f}",
                delta_color="inverse"
            )
        
        # Display detailed calculations
        st.markdown("##### 📝 Detailed Price Change Calculation")
        
        calc_data = {
            "Component": [
                "Current Bond Price",
                "Modified Duration",
                "Convexity",
                "Yield Change",
                "Duration Effect",
                "Convexity Effect",
                "Total Price Change",
                "Estimated New Price"
            ],
            "Value": [
                f"${discrete_price:,.2f}",
                f"{modified_duration:.4f}",
                f"{convexity:.4f}",
                f"{yield_change*100:+.2f}%",
                f"${price_change_duration:,.2f}",
                f"${price_change_convexity:,.2f}",
                f"${total_price_change:,.2f}",
                f"${new_price_estimate:,.2f}"
            ],
            "Formula": [
                "-",
                "Macaulay Duration / (1 + yield/periods)",
                "Σ[t(t+1)PV(CF_t)] / [Price × (1+yield)²]",
                "Given",
                "-Modified Duration × ΔYield × Price",
                "0.5 × Convexity × (ΔYield)² × Price",
                "Duration Effect + Convexity Effect",
                "Current Price + Total Price Change"
            ]
        }
        
        calc_df = pd.DataFrame(calc_data)
        st.dataframe(calc_df, use_container_width=True, hide_index=True)
        
        # Duration insights
        st.markdown("### 💡 Key Insights")
        
        insights = [
            f"**📊 Interest Rate Sensitivity**: A 1% increase in yield would decrease the bond price by approximately **{modified_duration:.1f}%**",
            f"**⏳ Cash Flow Timing**: The weighted average time to receive all cash flows is **{macaulay_duration:.1f} years**",
            f"**📈 Convexity Benefit**: Positive convexity means the bond's price increases more when yields fall than it decreases when yields rise",
            f"**🔍 Accuracy**: For small yield changes (±1%), duration alone provides a good estimate. For larger changes, convexity adjustment is important"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def display_explanation(self, params):
        """Display explanation of the calculations"""
        st.markdown("##### 📚 How Bond Prices Are Calculated")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###### Discrete Compounding Model")
            if params['bond_type'] == "Zero Coupon Bond":
                st.latex(r'''
                P = \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
                st.markdown("""
                Where:
                - **P** = Price of the bond
                - **F** = Face value/principal
                - **r** = Annual interest rate (decimal)
                - **n** = Compounding periods per year
                - **t** = Time to maturity in years
                """)
            else:
                st.latex(r'''
                P = \sum_{k=1}^{n \cdot t} \frac{C}{(1 + \frac{r}{n})^{k}} + \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
                st.markdown("""
                Where:
                - **P** = Price of the bond
                - **C** = Coupon payment per period = (coupon rate × F) / payments per year
                - **F** = Face value/principal
                - **r** = Annual interest rate (decimal)
                - **n** = Compounding periods per year
                - **t** = Time to maturity in years
                """)
        
        with col2:
            st.markdown("###### Continuous Compounding Model")
            if params['bond_type'] == "Zero Coupon Bond":
                st.latex(r'''
                P = F \cdot e^{-r \cdot t}
                ''')
                st.markdown("""
                Where:
                - **P** = Price of the bond
                - **F** = Face value/principal
                - **r** = Annual interest rate (decimal)
                - **t** = Time to maturity in years
                - **e** = Euler's number (≈ 2.71828)
                """)
            else:
                st.latex(r'''
                P = \sum_{i=1}^{m \cdot t} C \cdot e^{-r \cdot t_i} + F \cdot e^{-r \cdot t}
                ''')
                st.markdown("""
                Where:
                - **P** = Price of the bond
                - **C** = Coupon payment per period
                - **F** = Face value/principal
                - **r** = Annual interest rate (decimal)
                - **tᵢ** = Time of i-th coupon payment
                - **t** = Time to maturity in years
                - **m** = Payments per year
                """)
        
        # Duration formulas
        st.markdown("##### 📐 Duration Formulas")
        
        col_dur1, col_dur2 = st.columns(2)
        
        with col_dur1:
            st.markdown("###### Macaulay Duration")
            st.latex(r'''
            D_{mac} = \frac{\sum_{t=1}^{T} t \cdot PV(CF_t)}{P}
            ''')
            st.markdown("""
            Where:
            - **Dₘₐ꜀** = Macaulay Duration (years)
            - **t** = Time of cash flow (years)
            - **PV(CFₜ)** = Present value of cash flow at time t
            - **P** = Bond price
            - **T** = Maturity
            """)
        
        with col_dur2:
            st.markdown("###### Modified Duration")
            st.latex(r'''
            D_{mod} = \frac{D_{mac}}{1 + \frac{y}{m}}
            ''')
            st.markdown("""
            Where:
            - **Dₘₒₔ** = Modified Duration (years)
            - **Dₘₐ꜀** = Macaulay Duration (years)
            - **y** = Yield to maturity (annual)
            - **m** = Number of compounding periods per year
            
            **Price Sensitivity:**
            """)
            st.latex(r'''
            \frac{\Delta P}{P} \approx -D_{mod} \cdot \Delta y
            ''')
        
        st.markdown("---")
        st.markdown("##### 💡 Key Insights")
        
        insights = [
            "📊 **Interest Rate Sensitivity**: Bond prices move inversely to interest rates",
            "⏳ **Time Value**: Longer maturity bonds are more sensitive to rate changes",
            "🔄 **Compounding Effect**: More frequent compounding leads to slightly lower prices",
            "📈 **Coupon Effect**: Higher coupon bonds are less sensitive to rate changes",
            "🔍 **Model Difference**: Continuous compounding typically gives slightly different results than discrete"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def display_welcome(self):
        """Display welcome message when no calculation has been done"""
        st.markdown(f'<h1 class="main-header">📈 Bond Price Calculator</h1>', unsafe_allow_html=True)
        
        st.markdown("""
            ## Welcome to the Bond Calculator!
            
            This tool helps you calculate bond prices using two different models:
            
            ### 🔷 **Discrete Compounding Model**
            - Uses periodic compounding (annual, semi-annual, etc.)
            - More common in practice
            - Formula:""")
        
        st.latex(r'''
                P = \sum_{k=1}^{n \cdot t} \frac{C}{(1 + \frac{r}{n})^{k}} + \frac{F}{(1 + \frac{r}{n})^{n \cdot t}}
                ''')
        
        st.markdown("""    
            ### 🔶 **Continuous Compounding Model**
            - Assumes continuous compounding
            - Uses Euler's number (e)
            - Often used in theoretical finance
            - Formula:""")  
        
        st.latex(r'''
                P = \sum_{i=1}^{m \cdot t} C \cdot e^{-r \cdot t_i} + F \cdot e^{-r \cdot t}
                ''')
        
        st.markdown("""    
            ### 🎯 **Features:**
            1. **Zero Coupon Bonds**: Calculate prices for bonds with no periodic interest payments
            2. **Coupon Bonds**: Calculate prices for bonds with regular interest payments
            3. **Detailed Breakdown**: View payment schedules and present values
            4. **Visual Comparisons**: Charts to understand relationships 
            5. **Duration Analysis**: Calculate Macaulay and Modified Duration
            6. **Risk Assessment**: Measure interest rate sensitivity
            7. **Convexity**: Account for non-linear price-yield relationship
            8. **Price Sensitivity**: Estimate price changes for yield shifts
            
            ### 🚀 **How to Use:**
            1. Adjust parameters in the **sidebar**
            2. Click **"Calculate Bond Price"**
            3. Explore results in the **tabs below**
            """)
        
        st.markdown("""
            ### 📝 Example Parameters
            - **Principal**: $1,000
            - **Maturity**: 5 years
            - **Interest Rate**: 5%
            - **Frequency**: Annual
            
            Try these values to get started!
            """)
        
        st.markdown("---")
        st.markdown("### Bond Pricing Concepts")
        
        concepts_col1, concepts_col2, concepts_col3 = st.columns(3)
        
        with concepts_col1:
            st.markdown("""
            #### **Zero Coupon Bonds**
            - No periodic interest payments
            - Issued at a discount to face value
            - Price = Present value of face value
            - Higher price sensitivity to rate changes
            - Duration = Maturity
            """)
        
        with concepts_col2:
            st.markdown("""
            #### **Coupon Bonds**
            - Regular interest payments
            - Face value repaid at maturity
            - Price = PV of coupons + PV of face value
            - Lower duration than zero-coupon bonds
            - Duration < Maturity
            """)
        
        with concepts_col3:
            st.markdown("""
            #### **Duration Concepts**
            1. **Macaulay Duration**: Weighted average cash flow time
            2. **Modified Duration**: Price sensitivity to yield
            3. **Convexity**: Curvature adjustment
            4. **Immunization**: Duration matching strategy
            """)
    
    def run(self):
        """Main application runner"""
        # Create sidebar
        with st.sidebar:
            self.create_input_section()
        
        # Main content area
        if st.session_state.get('calculation_done', False):
            self.display_results()
        else:
            self.display_welcome()

# Run the application
if __name__ == "__main__":
    calculator = BondCalculator()
    calculator.run()
