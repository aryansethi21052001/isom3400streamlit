import streamlit as st
import math
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Bond Price Calculator",
    page_icon="📈",
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

class BondCalculatorStreamlit:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'bond_type' not in st.session_state:
            st.session_state.bond_type = "Zero Coupon"
        if 'calculation_done' not in st.session_state:
            st.session_state.calculation_done = False
    
    def calculate_discrete_price(self, params):
        """Calculate bond price using discrete compounding"""
        if params['bond_type'] == "Zero Coupon":
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
        if params['bond_type'] == "Zero Coupon":
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
    
    def create_input_section(self):
        """Create the input section in sidebar"""
        st.sidebar.header("📊 Bond Parameters")
        
        # Bond type selection
        bond_type = st.sidebar.selectbox(
            "Bond Type",
            ["Zero Coupon", "Coupon"],
            index=0 if st.session_state.bond_type == "Zero Coupon" else 1,
            key="bond_type_select"
        )
        st.session_state.bond_type = bond_type
        
        # Principal amount
        principal = st.sidebar.number_input(
            "Principal/Face Value ($)",
            min_value=0.01,
            max_value=1000000000.0,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
        
        # Maturity in years
        maturity = st.sidebar.slider(
            "Time to Maturity (years)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5,
            format="%.1f years"
        )
        
        # Interest rate (yield)
        interest_rate = st.sidebar.slider(
            "Annual Interest Rate (Yield)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            format="%.1f%%"
        ) / 100  # Convert to decimal
        
        # Coupon rate (only for coupon bonds)
        coupon_rate = 0.0
        if bond_type == "Coupon":
            coupon_rate = st.sidebar.slider(
                "Annual Coupon Rate",
                min_value=0.0,
                max_value=20.0,
                value=3.5,
                step=0.1,
                format="%.1f%%"
            ) / 100  # Convert to decimal
        
        # Frequency selection
        frequency = st.sidebar.selectbox(
            "Compounding/Payment Frequency",
            list(FREQUENCY_OPTIONS.keys()),
            index=1  # Default to Semi-annual
        )
        compounding_periods = FREQUENCY_OPTIONS[frequency]
        payments_per_year = FREQUENCY_OPTIONS[frequency] if bond_type == "Coupon" else 1
        
        # Calculate button
        if st.sidebar.button("🚀 Calculate Bond Price", type="primary", use_container_width=True):
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
        if st.sidebar.button("🔄 Reset", use_container_width=True):
            st.session_state.calculation_done = False
        
        st.sidebar.markdown("---")
        st.sidebar.info("💡 **Tip**: Adjust sliders and see real-time effects on bond pricing.")
    
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
                    f"{params['maturity']:.2f} years",
                    f"{params['interest_rate']*100:.2f}%",
                    params['frequency_name']
                ]
            }
            
            if params['bond_type'] == "Coupon":
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
        tab1, tab2 = st.tabs(["📊 Detailed Analysis", "ℹ️ Explanation"])
        
        with tab1:
            self.display_detailed_analysis(params, discrete_price, continuous_price)
        
        with tab2:
            self.display_explanation(params)
    
    def display_detailed_analysis(self, params, discrete_price, continuous_price):
        """Display detailed analysis including cash flows"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📅 Payment Schedule (Coupon Bonds Only)")
            if params['bond_type'] == "Coupon":
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
            if params['bond_type'] == "Coupon":
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
            st.markdown("##### 📊 Price Comparison")
            price_data = pd.DataFrame({
                'Model': ['Discrete', 'Continuous'],
                'Price': [discrete_price, continuous_price]
            })
            
            st.bar_chart(price_data.set_index('Model'))
    
    def display_explanation(self, params):
        """Display explanation of the calculations"""
        st.markdown("##### 📚 How Bond Prices Are Calculated")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###### Discrete Compounding Model")
            if params['bond_type'] == "Zero Coupon":
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
            if params['bond_type'] == "Zero Coupon":
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
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ## Welcome to the Interactive Bond Calculator!
            
            This tool helps you calculate bond prices using two different models:
            
            ### 🔷 **Discrete Compounding Model**
            - Uses periodic compounding (annual, semi-annual, etc.)
            - More common in practice
            - Formula varies based on compounding frequency
            
            ### 🔶 **Continuous Compounding Model**
            - Assumes continuous compounding
            - Uses Euler's number (e)
            - Often used in theoretical finance
            
            ### 🎯 **Features:**
            1. **Zero Coupon Bonds**: Calculate prices for bonds with no periodic interest payments
            2. **Coupon Bonds**: Calculate prices for bonds with regular interest payments
            3. **Detailed Breakdown**: View payment schedules and present values
            4. **Visual Comparisons**: Charts to understand relationships
            
            ### 🚀 **How to Use:**
            1. Adjust parameters in the **sidebar**
            2. Click **"Calculate Bond Price"**
            3. Explore results in the **tabs below**
            """)
        
        with col2:
            st.markdown("""
            ### Quick Start Guide
            
            | Step | Action | Location |
            |------|--------|----------|
            | 1️⃣ | Select Bond Type | Sidebar |
            | 2️⃣ | Set Parameters | Sidebar Sliders |
            | 3️⃣ | Click Calculate | Calculate Button |
            | 4️⃣ | Explore Results | Main Area Tabs |
            
            ### 📝 Example Parameters
            - **Principal**: $1,000
            - **Maturity**: 5 years
            - **Interest Rate**: 5%
            - **Frequency**: Semi-annual
            
            Try these values to get started!
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Bond Pricing Concepts")
        
        concepts_col1, concepts_col2, concepts_col3 = st.columns(3)
        
        with concepts_col1:
            st.markdown("""
            #### **Zero Coupon Bonds**
            - No periodic interest payments
            - Issued at a discount to face value
            - Price = Present value of face value
            - Higher price sensitivity to rate changes
            """)
        
        with concepts_col2:
            st.markdown("""
            #### **Coupon Bonds**
            - Regular interest payments
            - Face value repaid at maturity
            - Price = PV of coupons + PV of face value
            - Lower duration than zero-coupon bonds
            """)
        
        with concepts_col3:
            st.markdown("""
            #### **Key Factors**
            1. **Interest Rates**: Inverse relationship
            2. **Time to Maturity**: Longer = more sensitive
            3. **Coupon Rate**: Higher = less sensitive
            4. **Compounding**: More frequent = lower price
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
    calculator = BondCalculatorStreamlit()
    calculator.run()