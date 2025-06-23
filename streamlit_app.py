import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Import our simulation classes
from sim import DPoSSimulation, SimulationParams, StakingTier

# Clean up session state on code reload to prevent KeyErrors with enum objects
def clean_session_state():
    """Clean up old simulation results that may have stale enum references"""
    keys_to_clear = []
    for key in st.session_state.keys():
        if 'results_' in key or 'params_' in key or 'sim_' in key:
            # Check if we have results with StakingTier keys that might be stale
            if key.startswith('results_'):
                try:
                    results = st.session_state[key]
                    if isinstance(results, dict) and 'apy_track' in results:
                        # Try to access a StakingTier key - if it fails, clear this session state
                        _ = results['apy_track'][StakingTier.LIQUID]
                except (KeyError, TypeError, AttributeError):
                    keys_to_clear.append(key)
    
    # Clear problematic keys
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    if keys_to_clear:
        st.info(f"Cleaned up {len(keys_to_clear)} stale simulation results due to code reload.")

# Run cleanup on app start
clean_session_state()

# Page config
st.set_page_config(
    page_title="DPoS Policy Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("DPoS Staking Policy Simulator")
st.markdown("**Evaluate staking policies and predict user participation rates**")

# Add key business questions upfront
st.info(
    "**Key Questions This Tool Answers:**\n"
    "• What APY do we need for 70% user participation?\n"
    "• How does user risk tolerance affect staking adoption?\n"
    "• Can our reserves sustain this participation level?\n"
    "• Should we optimize for user adoption or total staking volume?"
)

# Create tabs for Option 1 and Option 2
tab1, tab2 = st.tabs(["Option 1: Fixed Budget", "Option 2: Dynamic Emissions"])

with tab1:
    st.markdown("**Fixed budget approach**: Set APYs directly, rewards come from a fixed reserve")

with tab2:
    st.markdown("**Dynamic emissions approach**: APYs determined by emission schedule and total staking")

# Shared sidebar configuration for both options
st.sidebar.header("Simulation Configuration")

# Common Economic Environment (shared by both options)
st.sidebar.subheader("Economic Environment")
years = st.sidebar.slider("Simulation Duration (years)", 1.0, 10.0, 5.0, 0.5)
T = int(years * 365)

# Market Scenario (shared)
st.sidebar.subheader("Market Scenario")

# Market scenario presets
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Bearish", help="Declining market conditions"):
        st.session_state.update({
            'market_drift': -0.3,  # -30% annual
            'market_volatility': 0.4  # 40% volatility
        })
        st.rerun()

with col2:
    if st.button("Neutral", help="Stable market conditions"):
        st.session_state.update({
            'market_drift': 0.0,   # 0% annual
            'market_volatility': 0.3  # 30% volatility
        })
        st.rerun()

with col3:
    if st.button("Bullish", help="Rising market conditions"):
        st.session_state.update({
            'market_drift': 0.5,   # +50% annual
            'market_volatility': 0.35  # 35% volatility
        })
        st.rerun()

# Manual market parameters
with st.sidebar.expander("Manual Market Settings"):
    market_drift = st.slider("Annual Price Drift (%)", -50, 100, 
                            int(st.session_state.get('market_drift', 0.0) * 100), 5,
                            help="Expected annual price change") / 100
    
    market_volatility = st.slider("Annual Price Volatility (%)", 10, 80,
                                 int(st.session_state.get('market_volatility', 0.3) * 100), 5,
                                 help="Price fluctuation intensity") / 100
    
    st.info(f"**Current Market Scenario:**\n"
           f"• Expected annual return: {market_drift:+.0%}\n"
           f"• Price volatility: {market_volatility:.0%}")

# Agent Psychology (shared)
st.sidebar.subheader("User Psychology")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    bearish_pct = st.slider("Bearish %", 0, 100, 30, 5, key="bearish_pct")
with col2:
    neutral_pct = st.slider("Neutral %", 0, 100, 50, 5, key="neutral_pct")
with col3:
    bullish_pct = st.slider("Bullish %", 0, 100, 20, 5, key="bullish_pct")

# Validate percentages sum to 100
total_pct = bearish_pct + neutral_pct + bullish_pct
if total_pct != 100:
    st.sidebar.warning(f"Warning: Must sum to 100% (currently {total_pct}%)")

# Sentiment configuration expanders
with st.sidebar.expander("Bearish Users"):
    bearish_hurdle = st.slider("Hurdle Rate (%)", 5, 25, 15, 1, key="bearish_hurdle") / 100
    bearish_illiquidity = st.slider("Illiquidity Cost (%)", 0.5, 5.0, 3.0, 0.1, key="bearish_illiq") / 100
    bearish_price_exp = st.slider("Price Expectation (%)", -50, 10, -30, 5, key="bearish_price") / 100

with st.sidebar.expander("Neutral Users"):
    neutral_hurdle = st.slider("Hurdle Rate (%)", 3, 20, 8, 1, key="neutral_hurdle") / 100
    neutral_illiquidity = st.slider("Illiquidity Cost (%)", 0.5, 3.0, 1.5, 0.1, key="neutral_illiq") / 100
    neutral_price_exp = st.slider("Price Expectation (%)", -20, 30, 0, 5, key="neutral_price") / 100

with st.sidebar.expander("Bullish Users"):
    bullish_hurdle = st.slider("Hurdle Rate (%)", 1, 15, 3, 1, key="bullish_hurdle") / 100
    bullish_illiquidity = st.slider("Illiquidity Cost (%)", 0.1, 2.0, 0.5, 0.1, key="bullish_illiq") / 100
    bullish_price_exp = st.slider("Price Expectation (%)", 0, 100, 50, 5, key="bullish_price") / 100

# Supply & Economics (shared)
with st.sidebar.expander("Supply & Economics"):
    C0 = st.sidebar.number_input("Initial Supply (millions)", 10.0, 200.0, 50.0, 5.0) * 1e6
    daily_vesting = st.sidebar.number_input("Daily Vesting (thousands)", 100.0, 5000.0, 1000.0, 100.0) * 1e3
    max_supply = st.sidebar.number_input("Max Supply (billions)", 1.0, 10.0, 2.0, 0.1) * 1e9
    
    agent_holdings_ratio = st.sidebar.slider("User Holdings Ratio", 0.2, 0.8, 0.4, 0.05, 
                                            help="What % of supply do users hold?")
    wealth_concentration = st.sidebar.slider("Wealth Concentration", 1.0, 3.0, 1.5, 0.1, 
                                            help="1.0=equal wealth, 3.0=very concentrated")

# Simulation Settings (shared)
with st.sidebar.expander("Simulation Settings"):
    N = st.sidebar.slider("Number of Simulated Users", 500, 2000, 1000, 100, 
                         help="More users = smoother estimates")
    initial_price = st.sidebar.number_input("Initial Token Price ($)", 0.01, 100.0, 0.1, 0.01)
    staking_price_impact = st.sidebar.slider("Staking Price Impact (%)", 0.0, 20.0, 10.0, 1.0) / 100

# OPTION-SPECIFIC CONTROLS
st.sidebar.markdown("---")
st.sidebar.subheader("Reward Mechanism")

# Option 1: Fixed Budget Controls
with st.sidebar.expander("Option 1: Fixed APY", expanded=False):
    st.write("**Set APY rates directly**")
    
    # Direct APY sliders for each tier
    liquid_apy = st.slider("Liquid APY (%)", 1.0, 20.0, 
                          st.session_state.get('liquid_apy', 5.0), 0.5, 
                          help="No lock-up required - immediate liquidity", key="opt1_liquid") / 100
    
    one_year_apy = st.slider("1-Year Lock APY (%)", 2.0, 25.0, 
                            st.session_state.get('one_year_apy', 8.0), 0.5, 
                            help="1 year commitment - higher rewards", key="opt1_1yr") / 100
    
    two_year_apy = st.slider("2-Year Lock APY (%)", 3.0, 30.0, 
                            st.session_state.get('two_year_apy', 10.0), 0.5, 
                            help="2 year commitment - highest rewards", key="opt1_2yr") / 100
    
    # Calculate multipliers for backwards compatibility
    R_base = liquid_apy  # Use liquid APY as the base
    liquid_mult = 1.0  # Liquid is always 1x the base
    one_year_mult = one_year_apy / R_base if R_base > 0 else 1.2
    two_year_mult = two_year_apy / R_base if R_base > 0 else 1.5
    
    # Reserve budget
    B0 = st.number_input("Reserve Budget (millions)", 50.0, 500.0, 
                        st.session_state.get('B0', 200e6) / 1e6, 10.0, key="opt1_budget") * 1e6
    
    st.info(f"**APY Structure:**\n"
           f"• Liquid: {liquid_apy*100:.1f}%\n"
           f"• 1-Year: {one_year_apy*100:.1f}% ({one_year_mult:.1f}x)\n"
           f"• 2-Year: {two_year_apy*100:.1f}% ({two_year_mult:.1f}x)")

# Option 2: Dynamic Emissions Controls  
with st.sidebar.expander("Option 2: Dynamic Multipliers", expanded=False):
    st.write("**Set multipliers, APY varies with participation**")
    st.caption("APY = (Emission × Multiplier) ÷ (365 × Total Weighted Stake)")
    
    liquid_multiplier = st.slider("Liquid Multiplier", 0.5, 2.0, 1.0, 0.1, 
                                 help="Base multiplier for liquid staking", key="opt2_liquid") 
    
    one_year_multiplier = st.slider("1-Year Lock Multiplier", 0.8, 3.0, 1.2, 0.1, 
                                   help="Multiplier for 1-year commitment", key="opt2_1yr")
    
    two_year_multiplier = st.slider("2-Year Lock Multiplier", 1.0, 4.0, 1.5, 0.1, 
                                   help="Multiplier for 2-year commitment", key="opt2_2yr")
    
    # Emission parameters
    E0 = st.number_input("Initial Annual Emission (millions)", 10.0, 200.0, 50.0, 5.0, 
                        help="Starting emission rate per year", key="opt2_emission") * 1e6
    k = st.slider("Decay Rate", 0.1, 2.0, 0.5, 0.1, 
                 help="Higher = faster emission decay", key="opt2_decay")
    
    # Baseline staking (existing stakers)
    baseline_staking = st.number_input("Baseline Staking (millions)", 5.0, 100.0, 10.0, 1.0,
                                     help="Initial staking amount from existing users (not modeled as agents)", 
                                     key="opt2_baseline") * 1e6
    
    # Show current multiplier relationships
    st.info(f"**Multiplier Ratios:**\n"
           f"• 1-Year vs Liquid: {one_year_multiplier/liquid_multiplier:.1f}x\n"
           f"• 2-Year vs Liquid: {two_year_multiplier/liquid_multiplier:.1f}x\n"
           f"• 2-Year vs 1-Year: {two_year_multiplier/one_year_multiplier:.1f}x")

# Performance info
st.sidebar.info("**Optimized Performance**\n• Daily time steps\n• Vectorized calculations")

# Option 1 simulation button and results
with tab1:
    # Run simulation button for Option 1
    if st.sidebar.button("Run Option 1 Simulation", type="primary", key="run_opt1"):
        
        # Optimized simulation parameters
        time_step = 1  # Daily for accuracy
        fast_mode = False  
        vectorized = True
        
        # Create parameters with sentiment-based approach
        params = SimulationParams(
            T=T,
            N=N,
            R_base=R_base,
            B0=B0,
            # Sentiment distribution
            bearish_pct=bearish_pct/100,
            neutral_pct=neutral_pct/100,
            bullish_pct=bullish_pct/100,
            # Bearish characteristics
            bearish_hurdle_rate=bearish_hurdle,
            bearish_illiquidity_cost=bearish_illiquidity,
            bearish_price_expectation=bearish_price_exp,
            # Neutral characteristics
            neutral_hurdle_rate=neutral_hurdle,
            neutral_illiquidity_cost=neutral_illiquidity,
            neutral_price_expectation=neutral_price_exp,
            # Bullish characteristics
            bullish_hurdle_rate=bullish_hurdle,
            bullish_illiquidity_cost=bullish_illiquidity,
            bullish_price_expectation=bullish_price_exp,
            # Other parameters
            C0=C0,
            daily_vesting=daily_vesting,
            max_supply=max_supply,
            agent_holdings_ratio=agent_holdings_ratio,
            wealth_concentration=wealth_concentration,
            holdings_scale_with_agents=False,  # Fixed user base for policy evaluation
            initial_price=initial_price,
            staking_price_impact=staking_price_impact,
            market_drift=market_drift,
            market_volatility=market_volatility,
            time_step=time_step,
            fast_mode=fast_mode,
            vectorized=vectorized,
            multipliers={
                StakingTier.LIQUID: liquid_mult,
                StakingTier.ONE_YEAR: one_year_mult,
                StakingTier.TWO_YEAR: two_year_mult
            }
        )
        
        # Run simulation
        with st.spinner("Running Option 1 simulation..."):
            sim = DPoSSimulation(params)
            results = sim.simulate_option1(debug=False)
            
        # Store results in session state
        st.session_state['results_opt1'] = results
        st.session_state['params_opt1'] = params
        st.session_state['sim_opt1'] = sim
    
    # Display Option 1 results if available
    if 'results_opt1' in st.session_state:
        st.header("Option 1 Results: Fixed APY")
        
        try:
            results = st.session_state['results_opt1']
            params = st.session_state['params_opt1']
            sim = st.session_state['sim_opt1']
            
            # Validate results structure
            if not all(key in results for key in ['apy_track', 'staked_track', 'total_staked', 'reserve_track', 'price_track']):
                st.error("❌ Invalid results structure. Please re-run the simulation.")
                st.stop()
            
            # Additional validation: check if we can access StakingTier keys
            try:
                _ = results['apy_track'][StakingTier.LIQUID]
            except KeyError:
                st.error("Stale simulation results detected (likely due to code reload). Please re-run the simulation.")
                # Clear the problematic session state
                for key in ['results_opt1', 'params_opt1', 'sim_opt1']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop()
            
        except Exception as e:
            st.error(f"Error accessing simulation results: {e}")
            st.info("Please run a new simulation.")
            # Clear potentially corrupted session state
            for key in ['results_opt1', 'params_opt1', 'sim_opt1']:
                if key in st.session_state:
                    del st.session_state[key]
            st.stop()
        
        # REDESIGNED METRICS - Policy Evaluation Focus
        st.subheader("Policy Effectiveness")
        
        # Calculate user participation metrics
        total_agent_holdings_over_time = np.sum(sim.agent_holdings_track, axis=0)
        
        # Calculate participation rates
        users_with_tokens = np.sum(sim.agent_holdings_track[:, 0] > 0)
        avg_staking = np.mean(results['total_staked'])
        supply_participation = (avg_staking / np.mean(sim.C_track)) * 100
        
        # Estimate user participation (users who choose to stake)
        # This is approximate - for exact count we'd need to track individual decisions
        agent_participation_rate = np.mean([
            (results['total_staked'][t] / total_agent_holdings_over_time[t]) * 100 
            if total_agent_holdings_over_time[t] > 0 else 0
            for t in range(len(results['total_staked']))
        ])
        
        # Approximate user participation based on staking behavior
        estimated_user_participation = min(100, agent_participation_rate / 0.8 * 100)  # Adjust for 80% max stake ratio
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User Participation", f"{estimated_user_participation:.1f}%", 
                     help="Estimated % of token holders who choose to stake")
        
        with col2:
            st.metric("Supply Staked", f"{supply_participation:.1f}%", 
                     help="% of total token supply being staked")
        
        with col3:
            final_reserve = results['reserve_track'][-1]
            reserve_pct = (final_reserve / params.B0) * 100
            st.metric("Reserve Health", f"{reserve_pct:.1f}%", 
                     delta=f"{reserve_pct-100:.1f}%" if reserve_pct != 100 else "Stable",
                     help="Remaining reserve budget")
        
        with col4:
            # Policy sustainability check
            depletion_day = None
            if any(results['reserve_track'] <= 0):
                depletion_day = np.argmax(results['reserve_track'] <= 0)
                sustainability = f"Day {depletion_day}"
                delta_color = "off"
            else:
                sustainability = "Sustainable"
                delta_color = "normal"
            st.metric("Policy Duration", sustainability,
                     help="How long this policy can be maintained")
        
        # Policy insights
        if estimated_user_participation < 50:
            st.warning("**Low User Adoption**: Consider increasing APY or targeting more risk-tolerant users")
        elif estimated_user_participation > 90:
            st.success("**Excellent Adoption**: Policy is very attractive to users")
        else:
            st.info("**Good Adoption**: Solid user participation rate")
        
        if supply_participation > 60:
            st.info("**High Security**: Large portion of supply is staked and securing the network")
        elif supply_participation < 20:
            st.warning("**Security Risk**: Low staking participation may reduce network security")

        # Charts
        st.header("Results")
        
        # Create time arrays (both days and years)
        days = np.arange(params.T)
        years_array = days / 365.0  # Convert to years for plotting
        
        # Calculate participation percentage
        participation_pct = (results['total_staked'] / sim.C_track) * 100
        
        # Calculate user engagement metrics
        total_agent_holdings_over_time = np.sum(sim.agent_holdings_track, axis=0)
        user_engagement_rate = np.zeros(len(days))
        
        for t in range(len(days)):
            if total_agent_holdings_over_time[t] > 0:
                user_engagement_rate[t] = (results['total_staked'][t] / total_agent_holdings_over_time[t]) * 100
            else:
                user_engagement_rate[t] = 0
        
        # Create policy-focused dataframe for charts
        liquid_apy = results['apy_track'][StakingTier.LIQUID] * 100
        one_year_apy = results['apy_track'][StakingTier.ONE_YEAR] * 100
        two_year_apy = results['apy_track'][StakingTier.TWO_YEAR] * 100
        liquid_staked = results['staked_track'][StakingTier.LIQUID] / 1e6
        one_year_staked = results['staked_track'][StakingTier.ONE_YEAR] / 1e6
        two_year_staked = results['staked_track'][StakingTier.TWO_YEAR] / 1e6

        policy_df = pd.DataFrame({
            'Day': days,
            'Year': years_array,
            'Total Staked (M)': results['total_staked'] / 1e6,
            'Supply Participation (%)': participation_pct,
            'User Engagement (%)': user_engagement_rate,
            'Reserve Health (M)': results['reserve_track'] / 1e6,
            'Reserve Sustainability (%)': (results['reserve_track'] / params.B0) * 100,
            'Policy Cost (M)': (params.B0 - results['reserve_track']) / 1e6,
            'APY Liquid (%)': liquid_apy,
            'APY 1-Year (%)': one_year_apy,
            'APY 2-Year (%)': two_year_apy,
            'Adoption Liquid (M)': liquid_staked,
            'Adoption 1-Year (M)': one_year_staked,
            'Adoption 2-Year (M)': two_year_staked,
            'User Holdings (M)': total_agent_holdings_over_time / 1e6,
            'Liquid Holdings (M)': (total_agent_holdings_over_time - results['total_staked']) / 1e6,
        })
        
        # Create policy-focused chart data
        tier_adoption_df = pd.DataFrame({
            'Year': np.concatenate([years_array, years_array, years_array]),
            'Adoption (M)': np.concatenate([policy_df['Adoption Liquid (M)'], 
                                           policy_df['Adoption 1-Year (M)'], 
                                           policy_df['Adoption 2-Year (M)']]),
            'Tier': ['Liquid'] * len(days) + ['1-Year Lock'] * len(days) + ['2-Year Lock'] * len(days)
        })
        
        user_allocation_df = pd.DataFrame({
            'Year': np.concatenate([years_array, years_array]),
            'Amount (M)': np.concatenate([policy_df['Total Staked (M)'], 
                                         policy_df['Liquid Holdings (M)']]),
            'Allocation': ['Staked (Earning Rewards)'] * len(days) + ['Liquid (Available)'] * len(days)
        })
        
        # Policy-Focused Layout: 3 Rows x 2 Columns for Better Visibility
        # Row 1: Key Policy Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Engagement", help="Percentage of user token holdings that are actively staked (earning rewards)")
            
            chart1_base = alt.Chart(policy_df).mark_line(strokeWidth=3, color='#2E86AB').encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('User Engagement (%):Q', title='User Engagement (%)', scale=alt.Scale(domain=[0, 100]))
            )
            
            # Reference lines for engagement targets
            ref_90 = alt.Chart(pd.DataFrame({'y': [90]})).mark_rule(
                color='green', strokeDash=[2, 2], opacity=0.7
            ).encode(y='y:Q')
            
            ref_70 = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
                color='orange', strokeDash=[3, 3], opacity=0.6
            ).encode(y='y:Q')
            
            ref_50 = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(
                color='red', strokeDash=[5, 5], opacity=0.5
            ).encode(y='y:Q')
            
            chart1 = (chart1_base + ref_90 + ref_70 + ref_50).properties(
                width=400, height=300
            ).resolve_scale(color='independent')
            st.altair_chart(chart1, use_container_width=True)
        
        with col2:
            st.subheader("Policy Costs", help="Shows cumulative policy costs (purple area) and remaining reserve budget (orange line)")
            
            chart2_area = alt.Chart(policy_df).mark_area(
                color='#A23B72',
                opacity=0.6,
                line={'color': '#A23B72', 'strokeWidth': 2}
            ).encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Policy Cost (M):Q', title='Cost (M tokens)')
            )
            
            chart2_line = alt.Chart(policy_df).mark_line(strokeWidth=2, color='#F18F01').encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Reserve Health (M):Q', title='Reserve (M tokens)')
            )
            
            chart2 = (chart2_area + chart2_line).properties(
                width=400, height=300
            ).resolve_scale(color='independent')
            st.altair_chart(chart2, use_container_width=True)
        
        # Row 2: Network Security & User Behavior
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Network Security", help="Percentage of total token supply that is staked and securing the network")
            
            chart3_base = alt.Chart(policy_df).mark_line(strokeWidth=3, color='#C73E1D').encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Supply Participation (%):Q', title='Supply Participation (%)')
            )
            
            # Security level benchmarks
            ref_high = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(
                color='green', strokeDash=[2, 2], opacity=0.7
            ).encode(y='y:Q')
            
            ref_medium = alt.Chart(pd.DataFrame({'y': [40]})).mark_rule(
                color='orange', strokeDash=[3, 3], opacity=0.6
            ).encode(y='y:Q')
            
            ref_low = alt.Chart(pd.DataFrame({'y': [20]})).mark_rule(
                color='red', strokeDash=[5, 5], opacity=0.5
            ).encode(y='y:Q')
            
            chart3 = (chart3_base + ref_high + ref_medium + ref_low).properties(
                width=400, height=300
            ).resolve_scale(color='independent')
            st.altair_chart(chart3, use_container_width=True)
        
        with col4:
            st.subheader("Tier Preferences", help="Shows which lock-up periods users prefer: Liquid (no lock), 1-Year, or 2-Year commitments")
            
            chart4 = alt.Chart(tier_adoption_df).mark_area().encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Adoption (M):Q', title='Tokens (M)'),
                color=alt.Color('Tier:N', 
                               scale=alt.Scale(range=['#87CEEB', '#32CD32', '#FF6347']),
                               legend=None),
                order=alt.Order('Tier:N', sort='ascending')
            ).properties(width=400, height=300)
            st.altair_chart(chart4, use_container_width=True)
        
        # Row 3: Capital Allocation & Supply Analysis
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Capital Allocation", help="How users split their token holdings between staking (earning rewards) vs keeping liquid (available for trading)")
            
            chart5 = alt.Chart(user_allocation_df).mark_area().encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Amount (M):Q', title='Holdings (M)'),
                color=alt.Color('Allocation:N', 
                               scale=alt.Scale(range=['#FF6B6B', '#4ECDC4']),
                               legend=None),
                order=alt.Order('Allocation:N', sort='descending')
            ).properties(width=400, height=300)
            st.altair_chart(chart5, use_container_width=True)
        
        with col6:
            st.subheader("Staked vs Supply", help="Compares total staked tokens (solid line) against circulating supply (dashed line) over time")
            
            # Create supply comparison data
            supply_comparison_df = pd.DataFrame({
                'Year': np.concatenate([years_array, years_array]),
                'Tokens (M)': np.concatenate([policy_df['Total Staked (M)'], sim.C_track / 1e6]),
                'Type': ['Total Staked'] * len(years_array) + ['Circulating Supply'] * len(years_array)
            })
            
            chart6 = alt.Chart(supply_comparison_df).mark_line(strokeWidth=3).encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Tokens (M):Q', title='Tokens (M)'),
                color=alt.Color('Type:N', 
                               scale=alt.Scale(domain=['Total Staked', 'Circulating Supply'], 
                                             range=['#1f77b4', '#ff7f0e']),
                               legend=None),
                strokeDash=alt.StrokeDash('Type:N',
                                         scale=alt.Scale(domain=['Total Staked', 'Circulating Supply'],
                                                       range=[[1,0], [5,5]]))
            ).properties(width=400, height=300)
            st.altair_chart(chart6, use_container_width=True)

        # Add token price chart in a new row
        st.subheader("Token Price Evolution", help="Shows how token price changes over time based on market scenario and staking impact")
        
        # Create price evolution data - ensure arrays match length
        price_data = results['price_track']
        if len(price_data) == len(years_array) + 1:
            # Price track includes initial price, skip it
            price_values = price_data[1:]
        else:
            # Price track matches years exactly
            price_values = price_data[:len(years_array)]
        
        price_df = pd.DataFrame({
            'Year': years_array,
            'Token Price ($)': price_values,
            'Market Scenario': f"{market_drift:+.0%} drift, {market_volatility:.0%} vol"
        })
        
        chart_price = alt.Chart(price_df).mark_line(strokeWidth=3, color='#ff7f0e').encode(
            x=alt.X('Year:Q', title='Years'),
            y=alt.Y('Token Price ($):Q', title='Token Price ($)', scale=alt.Scale(type='log')),
            tooltip=['Year:Q', 'Token Price ($):Q', 'Market Scenario:N']
        ).properties(width=600, height=300)
        
        st.altair_chart(chart_price, use_container_width=True)
        
        # Show price statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Price", f"${params.initial_price:.3f}")
        with col2:
            final_price = price_values[-1]
            st.metric("Final Price", f"${final_price:.3f}", 
                     delta=f"{((final_price/params.initial_price-1)*100):+.1f}%")
        with col3:
            max_price = np.max(price_values)
            st.metric("Max Price", f"${max_price:.3f}")
        with col4:
            min_price = np.min(price_values)
            st.metric("Min Price", f"${min_price:.3f}")
        
        # Determine policy type for export filename
        input_liquid_apy = params.R_base  # This is the liquid APY we set as base
        if input_liquid_apy <= 0.08:
            policy_type = "Conservative"
        elif input_liquid_apy <= 0.12:
            policy_type = "Balanced"
        else:
            policy_type = "Aggressive"
        
        # Export section
        st.header("Export Results")
        
        # Create policy-focused export data
        export_data = {
            'Year': years_array,
            'User Engagement (%)': user_engagement_rate,
            'Supply Participation (%)': participation_pct,
            'Total Staked (M)': results['total_staked'] / 1e6,
            'Reserve Health (%)': (results['reserve_track'] / params.B0) * 100,
            'Policy Cost (M)': (params.B0 - results['reserve_track']) / 1e6,
            'Liquid APY (%)': results['apy_track'][StakingTier.LIQUID] * 100,
            '1-Year APY (%)': results['apy_track'][StakingTier.ONE_YEAR] * 100,
            '2-Year APY (%)': results['apy_track'][StakingTier.TWO_YEAR] * 100,
            'Liquid Adoption (M)': results['staked_track'][StakingTier.LIQUID] / 1e6,
            '1-Year Adoption (M)': results['staked_track'][StakingTier.ONE_YEAR] / 1e6,
            '2-Year Adoption (M)': results['staked_track'][StakingTier.TWO_YEAR] / 1e6,
            'User Holdings (M)': total_agent_holdings_over_time / 1e6,
            'Liquid Holdings (M)': (total_agent_holdings_over_time - results['total_staked']) / 1e6,
        }
        
        export_df = pd.DataFrame(export_data)
        
        # Policy summary for filename
        policy_name = f"{params.R_base*100:.1f}pct_APY_{policy_type.split()[0].lower()}"
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Results",
                data=csv,
                file_name=f"dpos_policy_analysis_{policy_name}.csv",
                mime="text/csv",
                help="Download complete time-series data for further analysis"
            )
        
        with col2:
            # Policy summary export
            avg_engagement = np.mean(user_engagement_rate)
            avg_supply_participation = np.mean(participation_pct)
            final_reserve_pct = (results['reserve_track'][-1] / params.B0) * 100
            
            summary_data = {
                'Metric': [
                    'Liquid APY (%)',
                    '1-Year APY (%)',
                    '2-Year APY (%)',
                    'Policy Type',
                    'Average User Engagement (%)',
                    'Average Supply Participation (%)',
                    'Final Reserve Health (%)'
                ],
                'Value': [
                    f"{params.R_base*100:.1f}%",
                    f"{(params.R_base * params.multipliers[StakingTier.ONE_YEAR])*100:.1f}%",
                    f"{(params.R_base * params.multipliers[StakingTier.TWO_YEAR])*100:.1f}%",
                    policy_type,
                    f"{avg_engagement:.1f}%",
                    f"{avg_supply_participation:.1f}%",
                    f"{final_reserve_pct:.1f}%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="Download Policy Summary",
                data=summary_csv,
                file_name=f"policy_summary_{policy_name}.csv",
                mime="text/csv",
                help="Download key policy metrics summary"
            )
        
        # Preview policy data
        st.subheader("Policy Data Preview")
        show_full_table = st.checkbox("Show complete time series", value=False)
        
        if show_full_table:
            st.dataframe(export_df, use_container_width=True)
        else:
            # Show quarterly snapshots for better overview
            quarterly_indices = [i for i in range(0, len(export_df), max(1, len(export_df)//8))][:8]
            preview_df = export_df.iloc[quarterly_indices]
            st.dataframe(preview_df, use_container_width=True)
            st.info(f"Showing quarterly snapshots. Check 'Show complete time series' to see all {len(export_df)} data points.")
        
        # Additional insights
        st.header("Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reserve Analysis")
            if depletion_day:
                st.error(f"Reserve depleted on day {depletion_day}")
                st.write(f"Reserve lasted {depletion_day/365:.1f} years")
            else:
                st.success("Reserve maintained throughout simulation")
                reserve_burn_rate = (params.B0 - final_reserve) / params.T
                st.write(f"Daily burn rate: {reserve_burn_rate/1e6:.2f}M tokens")
                
            # Calculate sustainability
            if depletion_day:
                sustainability_score = max(0, (depletion_day / params.T) * 100)
            else:
                sustainability_score = 100
            st.metric("Sustainability Score", f"{sustainability_score:.0f}/100")
        
        with col2:
            st.subheader("Participation Analysis")
            max_participation = np.max(participation_pct)
            min_participation = np.min(participation_pct)
            participation_volatility = np.std(participation_pct)
            
            st.metric("Max Participation", f"{max_participation:.1f}%")
            st.metric("Min Participation", f"{min_participation:.1f}%")
            st.metric("Participation Volatility", f"{participation_volatility:.2f}%")
            
            # Tier preference analysis
            final_liquid = results['staked_track'][StakingTier.LIQUID][-1]
            final_1yr = results['staked_track'][StakingTier.ONE_YEAR][-1]
            final_2yr = results['staked_track'][StakingTier.TWO_YEAR][-1]
            total_final = final_liquid + final_1yr + final_2yr
            
            if total_final > 0:
                st.write("**Final Tier Distribution:**")
                st.write(f"- Liquid: {(final_liquid/total_final)*100:.1f}%")
                st.write(f"- 1-Year: {(final_1yr/total_final)*100:.1f}%")
                st.write(f"- 2-Year: {(final_2yr/total_final)*100:.1f}%")
    else:
        st.info("Configure Option 1 parameters and click 'Run Option 1 Simulation' to see results!")
        
        # Show some example configurations
        st.subheader("Example Configurations")
        
        examples = {
            "Conservative": {
                "Base APY": "5%",
                "Reserve": "300M tokens",
                "Description": "Lower rewards, larger reserve for long-term sustainability"
            },
            "Aggressive": {
                "Base APY": "15%", 
                "Reserve": "100M tokens",
                "Description": "High rewards, smaller reserve for rapid growth"
            },
            "Balanced": {
                "Base APY": "8%",
                "Reserve": "200M tokens", 
                "Description": "Moderate rewards and reserve size"
            }
        }
        
        for name, config in examples.items():
            with st.expander(f"{name} Configuration"):
                st.write(f"**Base APY:** {config['Base APY']}")
                st.write(f"**Reserve:** {config['Reserve']}")
                st.write(f"**Description:** {config['Description']}")

with tab2:
    # Option 2 simulation button and results
    if st.sidebar.button("Run Option 2 Simulation", type="primary", key="run_opt2"):
        
        # Optimized simulation parameters
        time_step = 1  # Daily for accuracy
        fast_mode = False  
        vectorized = True
        
        # Create parameters with sentiment-based approach for Option 2
        params = SimulationParams(
            T=T,
            N=N,
            E0=E0,
            k=k,
            baseline_staking=baseline_staking,
            # Sentiment distribution
            bearish_pct=bearish_pct/100,
            neutral_pct=neutral_pct/100,
            bullish_pct=bullish_pct/100,
            # Bearish characteristics
            bearish_hurdle_rate=bearish_hurdle,
            bearish_illiquidity_cost=bearish_illiquidity,
            bearish_price_expectation=bearish_price_exp,
            # Neutral characteristics
            neutral_hurdle_rate=neutral_hurdle,
            neutral_illiquidity_cost=neutral_illiquidity,
            neutral_price_expectation=neutral_price_exp,
            # Bullish characteristics
            bullish_hurdle_rate=bullish_hurdle,
            bullish_illiquidity_cost=bullish_illiquidity,
            bullish_price_expectation=bullish_price_exp,
            # Other parameters
            C0=C0,
            daily_vesting=daily_vesting,
            max_supply=max_supply,
            agent_holdings_ratio=agent_holdings_ratio,
            wealth_concentration=wealth_concentration,
            holdings_scale_with_agents=False,  # Fixed user base for policy evaluation
            initial_price=initial_price,
            staking_price_impact=staking_price_impact,
            market_drift=market_drift,
            market_volatility=market_volatility,
            time_step=time_step,
            fast_mode=fast_mode,
            vectorized=vectorized,
            multipliers={
                StakingTier.LIQUID: liquid_multiplier,
                StakingTier.ONE_YEAR: one_year_multiplier,
                StakingTier.TWO_YEAR: two_year_multiplier
            }
        )
        
        # Run simulation
        with st.spinner("Running Option 2 simulation..."):
            sim = DPoSSimulation(params)
            results = sim.simulate_option2(debug=False)
            
        # Store results in session state
        st.session_state['results_opt2'] = results
        st.session_state['params_opt2'] = params
        st.session_state['sim_opt2'] = sim
    
    # Display Option 2 results if available
    if 'results_opt2' in st.session_state:
        st.header("Option 2 Results: Dynamic Emissions")
        
        try:
            results = st.session_state['results_opt2']
            params = st.session_state['params_opt2']
            sim = st.session_state['sim_opt2']
            
            # Validate results structure
            if not all(key in results for key in ['apy_track', 'staked_track', 'total_staked', 'emission_track', 'price_track']):
                st.error("Invalid results structure. Please re-run the simulation.")
                st.stop()
            
            # Additional validation: check if we can access StakingTier keys
            try:
                _ = results['apy_track'][StakingTier.LIQUID]
            except KeyError:
                st.error("Stale simulation results detected (likely due to code reload). Please re-run the simulation.")
                # Clear the problematic session state
                for key in ['results_opt2', 'params_opt2', 'sim_opt2']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop()
            
        except Exception as e:
            st.error(f"Error accessing simulation results: {e}")
            st.info("Please run a new simulation.")
            # Clear potentially corrupted session state
            for key in ['results_opt2', 'params_opt2', 'sim_opt2']:
                if key in st.session_state:
                    del st.session_state[key]
            st.stop()
        
        # OPTION 2 METRICS - Dynamic Emissions Focus
        st.subheader("Dynamic Policy Effectiveness")
        
        # Calculate user participation metrics
        total_agent_holdings_over_time = np.sum(sim.agent_holdings_track, axis=0)
        
        # Calculate participation rates
        avg_staking = np.mean(results['total_staked'])
        supply_participation = (avg_staking / np.mean(sim.C_track)) * 100
        
        # Estimate user participation based on staking behavior
        agent_participation_rate = np.mean([
            (results['total_staked'][t] / total_agent_holdings_over_time[t]) * 100 
            if total_agent_holdings_over_time[t] > 0 else 0
            for t in range(len(results['total_staked']))
        ])
        
        estimated_user_participation = min(100, agent_participation_rate / 0.8 * 100)
        
        # Calculate APY statistics (excluding Day 0 artifacts)
        liquid_apy_avg = np.mean(results['apy_track'][StakingTier.LIQUID][1:]) * 100
        one_year_apy_avg = np.mean(results['apy_track'][StakingTier.ONE_YEAR][1:]) * 100
        two_year_apy_avg = np.mean(results['apy_track'][StakingTier.TWO_YEAR][1:]) * 100
        
        # Calculate total emissions used
        total_emissions = np.sum(results['emission_track'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("User Participation", f"{estimated_user_participation:.1f}%", 
                     help="Estimated % of token holders who choose to stake")
        
        with col2:
            st.metric("Supply Staked", f"{supply_participation:.1f}%", 
                     help="% of total token supply being staked")
        
        with col3:
            st.metric("Avg APY Range", f"{liquid_apy_avg:.1f}%-{two_year_apy_avg:.1f}%", 
                     help="Average APY range from Liquid to 2-Year tiers")
        
        with col4:
            st.metric("Total Emissions", f"{total_emissions/1e6:.1f}M", 
                     help="Total tokens emitted as rewards over simulation period")
        
        # Policy insights for Option 2
        if estimated_user_participation < 50:
            st.warning("**Low User Adoption**: Consider adjusting multipliers or emission schedule")
        elif estimated_user_participation > 90:
            st.success("**Excellent Adoption**: Dynamic policy is very attractive to users")
        else:
            st.info("**Good Adoption**: Solid user participation with dynamic rewards")
        
        if supply_participation > 60:
            st.info("**High Security**: Large portion of supply is staked and securing the network")
        elif supply_participation < 20:
            st.warning("**Security Risk**: Low staking participation may reduce network security")

        # Charts - Focus on Option 2 specific visualizations
        st.header("Dynamic Emissions Results")
        
        # Create time arrays
        days = np.arange(params.T)
        years_array = days / 365.0
        
        # THE KEY CHART: Variable APY by Bucket (what the user specifically requested)
        st.subheader("Variable APY by Staking Tier", help="Shows how APY changes over time for each staking tier based on participation levels")
        
        # Add log scale toggle
        col_toggle, col_info = st.columns([1, 3])
        with col_toggle:
            use_log_scale = st.checkbox("Log Scale", value=False, help="Use logarithmic scale for APY axis (helpful when APY values vary widely)")
        with col_info:
            st.info("**Note**: Chart starts from Day 1 to exclude artificial Day 0 APY spikes caused by baseline staking setup.")
        
        # Create APY dataframe for the key chart (skip Day 0 due to setup artifacts)
        days_from_1 = days[1:]  # Skip day 0
        years_from_1 = years_array[1:]  # Skip day 0
        
        apy_df = pd.DataFrame({
            'Year': np.concatenate([years_from_1, years_from_1, years_from_1]),
            'APY (%)': np.concatenate([
                results['apy_track'][StakingTier.LIQUID][1:] * 100,
                results['apy_track'][StakingTier.ONE_YEAR][1:] * 100,
                results['apy_track'][StakingTier.TWO_YEAR][1:] * 100
            ]),
            'Tier': (['Liquid (No Lock)'] * len(years_from_1) + 
                    ['1-Year Lock'] * len(years_from_1) + 
                    ['2-Year Lock'] * len(years_from_1)),
            'Multiplier': ([f"{liquid_multiplier}x"] * len(years_from_1) + 
                          [f"{one_year_multiplier}x"] * len(years_from_1) + 
                          [f"{two_year_multiplier}x"] * len(years_from_1))
        })
        
        # Create the APY chart with different colors for each tier
        # Conditional y-axis scale based on toggle
        if use_log_scale:
            y_scale = alt.Scale(type='log', zero=False)
            chart_title = "Variable APY by Staking Tier (Day 1+, Log Scale) - Higher Participation = Lower APY"
        else:
            y_scale = alt.Scale(zero=False)
            chart_title = "Variable APY by Staking Tier (Day 1+) - Higher Participation = Lower APY"
        
        apy_chart = alt.Chart(apy_df).mark_line(strokeWidth=3).encode(
            x=alt.X('Year:Q', title='Years'),
            y=alt.Y('APY (%):Q', title='APY (%)', scale=y_scale),
            color=alt.Color('Tier:N', 
                           scale=alt.Scale(domain=['Liquid (No Lock)', '1-Year Lock', '2-Year Lock'],
                                         range=['#87CEEB', '#32CD32', '#FF6347']),
                           legend=alt.Legend(title="Staking Tier", orient="right")),
            tooltip=['Year:Q', 'APY (%):Q', 'Tier:N', 'Multiplier:N']
        ).properties(
            width=700, 
            height=400,
            title=chart_title
        ).resolve_scale(color='independent')
        
        st.altair_chart(apy_chart, use_container_width=True)
        
        # Show APY statistics (Day 1+ to exclude setup artifacts)
        col1, col2, col3 = st.columns(3)
        with col1:
            liquid_max = np.max(results['apy_track'][StakingTier.LIQUID][1:]) * 100
            liquid_min = np.min(results['apy_track'][StakingTier.LIQUID][1:]) * 100
            st.metric("Liquid APY Range", f"{liquid_min:.1f}%-{liquid_max:.1f}%")
        
        with col2:
            one_yr_max = np.max(results['apy_track'][StakingTier.ONE_YEAR][1:]) * 100
            one_yr_min = np.min(results['apy_track'][StakingTier.ONE_YEAR][1:]) * 100
            st.metric("1-Year APY Range", f"{one_yr_min:.1f}%-{one_yr_max:.1f}%")
        
        with col3:
            two_yr_max = np.max(results['apy_track'][StakingTier.TWO_YEAR][1:]) * 100
            two_yr_min = np.min(results['apy_track'][StakingTier.TWO_YEAR][1:]) * 100
            st.metric("2-Year APY Range", f"{two_yr_min:.1f}%-{two_yr_max:.1f}%")
        
        # Additional Option 2 specific charts - Reorganized to 2 columns for better visibility
        # Row 1: Core Dynamic Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emission Schedule", help="Shows how token emissions decay over time")
            
            emission_df = pd.DataFrame({
                'Year': years_array,
                'Annual Emission (M)': results['emission_track'] / 1e6
            })
            
            emission_chart = alt.Chart(emission_df).mark_area(
                color='#FF6B6B',
                opacity=0.6,
                line={'color': '#FF6B6B', 'strokeWidth': 2}
            ).encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Annual Emission (M):Q', title='Annual Emission (M tokens)')
            ).properties(width=400, height=300)
            
            st.altair_chart(emission_chart, use_container_width=True)
        
        with col2:
            st.subheader("Participation vs APY", help="Shows relationship between staking participation and average APY")
            
            # Calculate participation percentage and average APY
            participation_pct = (results['total_staked'] / sim.C_track) * 100
            avg_apy_per_day = np.mean([
                results['apy_track'][StakingTier.LIQUID],
                results['apy_track'][StakingTier.ONE_YEAR], 
                results['apy_track'][StakingTier.TWO_YEAR]
            ], axis=0) * 100
            
            participation_apy_df = pd.DataFrame({
                'Year': years_array,
                'Participation (%)': participation_pct,
                'Average APY (%)': avg_apy_per_day
            })
            
            # Create dual-axis chart
            base = alt.Chart(participation_apy_df)
            
            participation_line = base.mark_line(strokeWidth=2, color='blue').encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Participation (%):Q', title='Participation (%)', scale=alt.Scale(zero=False))
            )
            
            apy_line = base.mark_line(strokeWidth=2, color='red', strokeDash=[5,5]).encode(
                x=alt.X('Year:Q'),
                y=alt.Y('Average APY (%):Q', title='Average APY (%)', scale=alt.Scale(zero=False))
            )
            
            dual_chart = (participation_line + apy_line).resolve_scale(
                y='independent'
            ).properties(width=400, height=300)
            
            st.altair_chart(dual_chart, use_container_width=True)
        
        # Row 2: Tier & Market Analysis
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Staking by Tier", help="Shows how much is staked in each tier over time")
            
            tier_staking_df = pd.DataFrame({
                'Year': np.concatenate([years_array, years_array, years_array]),
                'Staked (M)': np.concatenate([
                    results['staked_track'][StakingTier.LIQUID] / 1e6,
                    results['staked_track'][StakingTier.ONE_YEAR] / 1e6,
                    results['staked_track'][StakingTier.TWO_YEAR] / 1e6
                ]),
                'Tier': (['Liquid'] * len(years_array) + 
                        ['1-Year'] * len(years_array) + 
                        ['2-Year'] * len(years_array))
            })
            
            tier_chart = alt.Chart(tier_staking_df).mark_area().encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Staked (M):Q', title='Tokens Staked (M)'),
                color=alt.Color('Tier:N', 
                               scale=alt.Scale(range=['#87CEEB', '#32CD32', '#FF6347'])),
                order=alt.Order('Tier:N', sort='ascending')
            ).properties(width=400, height=300)
            
            st.altair_chart(tier_chart, use_container_width=True)
        
        with col4:
            st.subheader("Token Price Evolution", help="Shows how token price changes under dynamic emissions")
            
            # Handle price data length
            price_data = results['price_track']
            if len(price_data) == len(years_array) + 1:
                price_values = price_data[1:]
            else:
                price_values = price_data[:len(years_array)]
            
            price_df = pd.DataFrame({
                'Year': years_array,
                'Token Price ($)': price_values
            })
            
            price_chart = alt.Chart(price_df).mark_line(strokeWidth=3, color='#ff7f0e').encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Token Price ($):Q', title='Token Price ($)', scale=alt.Scale(type='log'))
            ).properties(width=400, height=300)
            
            st.altair_chart(price_chart, use_container_width=True)
        
        # Row 3: Supply Analysis
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Staked vs Supply", help="Compares total staked tokens (solid line) against circulating supply (dashed line) over time")
            
            # Create supply comparison data
            supply_comparison_df = pd.DataFrame({
                'Year': np.concatenate([years_array, years_array]),
                'Tokens (M)': np.concatenate([results['total_staked'] / 1e6, sim.C_track / 1e6]),
                'Type': ['Total Staked'] * len(years_array) + ['Circulating Supply'] * len(years_array)
            })
            
            supply_chart = alt.Chart(supply_comparison_df).mark_line(strokeWidth=3).encode(
                x=alt.X('Year:Q', title='Years'),
                y=alt.Y('Tokens (M):Q', title='Tokens (M)'),
                color=alt.Color('Type:N', 
                               scale=alt.Scale(domain=['Total Staked', 'Circulating Supply'], 
                                             range=['#1f77b4', '#ff7f0e']),
                               legend=None),
                strokeDash=alt.StrokeDash('Type:N',
                                         scale=alt.Scale(domain=['Total Staked', 'Circulating Supply'],
                                                       range=[[1,0], [5,5]]))
            ).properties(width=400, height=300)
            
            st.altair_chart(supply_chart, use_container_width=True)
        
        with col6:
            # Add a placeholder or another useful chart here if needed
            st.info("**Space for additional analysis**\n\nThis area could show baseline staking impact, cumulative emissions, or other dynamic metrics specific to Option 2.")
        
        # Export section for Option 2
        st.header("Export Option 2 Results")
        
        # Create Option 2 export data
        export_data_opt2 = {
            'Year': years_array,
            'User Engagement (%)': agent_participation_rate,
            'Supply Participation (%)': participation_pct,
            'Total Staked (M)': results['total_staked'] / 1e6,
            'Emission (M/year)': results['emission_track'] / 1e6,
            'Liquid APY (%)': results['apy_track'][StakingTier.LIQUID] * 100,
            '1-Year APY (%)': results['apy_track'][StakingTier.ONE_YEAR] * 100,
            '2-Year APY (%)': results['apy_track'][StakingTier.TWO_YEAR] * 100,
            'Liquid Staked (M)': results['staked_track'][StakingTier.LIQUID] / 1e6,
            '1-Year Staked (M)': results['staked_track'][StakingTier.ONE_YEAR] / 1e6,
            '2-Year Staked (M)': results['staked_track'][StakingTier.TWO_YEAR] / 1e6,
            'Token Price ($)': price_values,
            'Weighted Stake (M)': results['weighted_stake'] / 1e6,
        }
        
        export_df_opt2 = pd.DataFrame(export_data_opt2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_opt2 = export_df_opt2.to_csv(index=False)
            st.download_button(
                label="Download Option 2 Results",
                data=csv_opt2,
                file_name=f"dpos_option2_dynamic_emissions.csv",
                mime="text/csv",
                help="Download complete Option 2 time-series data"
            )
        
        with col2:
            # Option 2 summary
            summary_data_opt2 = {
                'Metric': [
                    'Liquid Multiplier',
                    '1-Year Multiplier', 
                    '2-Year Multiplier',
                    'Initial Emission (M/year)',
                    'Decay Rate',
                    'Avg User Engagement (%)',
                    'Avg Supply Participation (%)',
                    'Total Emissions (M)'
                ],
                'Value': [
                    f"{liquid_multiplier}x",
                    f"{one_year_multiplier}x", 
                    f"{two_year_multiplier}x",
                    f"{E0/1e6:.0f}M",
                    f"{k}",
                    f"{agent_participation_rate:.1f}%",
                    f"{supply_participation:.1f}%",
                    f"{total_emissions/1e6:.1f}M"
                ]
            }
            
            summary_df_opt2 = pd.DataFrame(summary_data_opt2)
            summary_csv_opt2 = summary_df_opt2.to_csv(index=False)
            
            st.download_button(
                label="Download Option 2 Summary",
                data=summary_csv_opt2,
                file_name=f"option2_summary_dynamic_emissions.csv", 
                mime="text/csv",
                help="Download Option 2 policy summary"
            )
        
        # Data Preview
        st.subheader("Option 2 Data Preview")
        show_full_table_opt2 = st.checkbox("Show complete Option 2 time series", value=False)
        
        if show_full_table_opt2:
            st.dataframe(export_df_opt2, use_container_width=True)
        else:
            quarterly_indices = [i for i in range(0, len(export_df_opt2), max(1, len(export_df_opt2)//8))][:8]
            preview_df_opt2 = export_df_opt2.iloc[quarterly_indices]
            st.dataframe(preview_df_opt2, use_container_width=True)
            st.info(f"Showing quarterly snapshots. Check 'Show complete Option 2 time series' to see all {len(export_df_opt2)} data points.")
            
    else:
        st.info("Configure Option 2 parameters and click 'Run Option 2 Simulation' to see results!")
        
        # Show current configuration
        st.subheader("Current Option 2 Configuration")
        st.write("**Current Option 2 Settings:**")
        st.write(f"• Liquid Multiplier: {liquid_multiplier}x")
        st.write(f"• 1-Year Multiplier: {one_year_multiplier}x") 
        st.write(f"• 2-Year Multiplier: {two_year_multiplier}x")
        st.write(f"• Initial Emission: {E0/1e6:.0f}M tokens/year")
        st.write(f"• Decay Rate: {k}")
        
        st.markdown("**Key differences from Option 1:**")
        st.markdown("• **Variable APY**: APY changes based on total staking participation")
        st.markdown("• **Dynamic Rewards**: Higher participation = lower APY for everyone") 
        st.markdown("• **Decaying Emissions**: Emission schedule determines total rewards available")
        st.markdown("• **Sustainable**: More economically sustainable but less predictable returns")
        st.markdown("• **Self-Balancing**: System naturally balances participation vs rewards")
