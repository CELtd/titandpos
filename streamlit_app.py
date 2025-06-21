import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Import our simulation classes
from sim import DPoSSimulation, SimulationParams, StakingTier

# Page config
st.set_page_config(
    page_title="DPoS Policy Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 DPoS Staking Policy Simulator")
st.markdown("**Evaluate staking policies and predict user participation rates**")

# Add key business questions upfront
st.info(
    "💡 **Key Questions This Tool Answers:**\n"
    "• What APY do we need for 70% user participation?\n"
    "• How does user risk tolerance affect staking adoption?\n"
    "• Can our reserves sustain this participation level?\n"
    "• Should we optimize for user adoption or total staking volume?"
)

# Sidebar for parameters - REDESIGNED for policy focus
st.sidebar.header("🎯 Policy Configuration")

# Staking Policy (most important section)
st.sidebar.subheader("💰 Staking Incentives")
st.sidebar.write("**APY by Tier**")

# Direct APY sliders for each tier
liquid_apy = st.sidebar.slider("Liquid APY (%)", 1.0, 20.0, 
                              st.session_state.get('liquid_apy', 8.0), 0.5, 
                              help="No lock-up required - immediate liquidity") / 100

one_year_apy = st.sidebar.slider("1-Year Lock APY (%)", 2.0, 25.0, 
                                st.session_state.get('one_year_apy', 10.0), 0.5, 
                                help="1 year commitment - higher rewards") / 100

two_year_apy = st.sidebar.slider("2-Year Lock APY (%)", 3.0, 30.0, 
                                st.session_state.get('two_year_apy', 12.0), 0.5, 
                                help="2 year commitment - highest rewards") / 100

# Calculate base rate and multipliers for the simulation (backwards compatibility)
R_base = liquid_apy  # Use liquid APY as the base
liquid_mult = 1.0  # Liquid is always 1x the base
one_year_mult = one_year_apy / R_base if R_base > 0 else 1.2
two_year_mult = two_year_apy / R_base if R_base > 0 else 1.5

# User Behavior Parameters (second most important)
st.sidebar.subheader("👥 User Population Characteristics")
st.sidebar.write("*These parameters model your expected user base*")

rho_min = st.sidebar.slider("Min Required Return (%)", 0.5, 10.0, 
                           st.session_state.get('rho_min', 0.02) * 100, 0.5, 
                           help="Lowest return rate users will accept") / 100
rho_max = st.sidebar.slider("Max Required Return (%)", 5.0, 30.0, 
                           st.session_state.get('rho_max', 0.15) * 100, 0.5, 
                           help="Highest return rate demanded by picky users") / 100
lambda_max = st.sidebar.slider("Max Illiquidity Tolerance (%)", 0.5, 8.0, 
                              st.session_state.get('lambda_max', 0.02) * 100, 0.1, 
                              help="How much users hate lock-ups") / 100

wealth_concentration = st.sidebar.slider("Wealth Concentration", 1.0, 3.0, 1.5, 0.1, 
                                        help="1.0=equal wealth, 3.0=very concentrated")

# Economic Context
st.sidebar.subheader("🏛️ Economic Environment")
years = st.sidebar.slider("Simulation Duration (years)", 1.0, 10.0, 5.0, 0.5)
T = int(years * 365)

price_scenario = st.sidebar.selectbox(
    "Market Scenario",
    ["bearish", "neutral", "bullish"],
    index=1,
    help="Affects user expectations: Bearish=-30% drift, Neutral=0%, Bullish=+50%"
)

# Reserve & Supply (less prominent)
with st.sidebar.expander("💰 Reserve & Supply Parameters"):
    B0 = st.sidebar.number_input("Reserve Budget (millions)", 50.0, 500.0, 
                                st.session_state.get('B0', 100e6) / 1e6, 10.0) * 1e6
    C0 = st.sidebar.number_input("Initial Supply (millions)", 10.0, 200.0, 50.0, 5.0) * 1e6
    daily_vesting = st.sidebar.number_input("Daily Vesting (thousands)", 100.0, 5000.0, 1000.0, 100.0) * 1e3
    max_supply = st.sidebar.number_input("Max Supply (billions)", 1.0, 10.0, 2.0, 0.1) * 1e9
    
    agent_holdings_ratio = st.sidebar.slider("User Holdings Ratio", 0.2, 0.8, 0.4, 0.05, 
                                            help="What % of supply do users hold?")

# Simulation Settings (least prominent)
with st.sidebar.expander("⚙️ Simulation Settings"):
    N = st.sidebar.slider("Number of Simulated Users", 500, 2000, 1000, 100, 
                         help="More users = smoother estimates")
    initial_price = st.sidebar.number_input("Initial Token Price ($)", 0.1, 100.0, 1.0, 0.1)
    staking_price_impact = st.sidebar.slider("Staking Price Impact (%)", 0.0, 20.0, 10.0, 1.0) / 100

# Performance info
st.sidebar.info("⚡ **Optimized Performance**\n• Daily time steps\n• Vectorized calculations")

# Policy Presets
st.sidebar.subheader("📋 Policy Presets")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button("🎯 Conservative", help="Low risk, broad appeal"):
        st.session_state.update({
            'liquid_apy': 6.0, 'one_year_apy': 7.0, 'two_year_apy': 8.0,
            'B0': 150e6, 'price_scenario': 'neutral',
            'rho_min': 0.01, 'rho_max': 0.10, 'lambda_max': 0.015
        })
        st.rerun()

with col2:
    if st.button("💎 Balanced", help="Moderate rewards and risk"):
        st.session_state.update({
            'liquid_apy': 8.0, 'one_year_apy': 10.0, 'two_year_apy': 12.0,
            'B0': 100e6, 'price_scenario': 'neutral',
            'rho_min': 0.02, 'rho_max': 0.15, 'lambda_max': 0.02
        })
        st.rerun()

with col3:
    if st.button("🚀 Aggressive", help="High rewards, risk-seeking users"):
        st.session_state.update({
            'liquid_apy': 12.0, 'one_year_apy': 16.0, 'two_year_apy': 20.0,
            'B0': 75e6, 'price_scenario': 'bullish',
            'rho_min': 0.015, 'rho_max': 0.20, 'lambda_max': 0.025
        })
        st.rerun()

# Run simulation button
if st.sidebar.button("🚀 Run Policy Simulation", type="primary"):
    
    # Optimized simulation parameters
    time_step = 1  # Daily for accuracy
    fast_mode = False  
    vectorized = True
    
    # Create parameters
    params = SimulationParams(
        T=T,
        N=N,
        R_base=R_base,
        B0=B0,
        rho_min=rho_min,
        rho_max=rho_max,
        lambda_max=lambda_max,
        C0=C0,
        daily_vesting=daily_vesting,
        max_supply=max_supply,
        agent_holdings_ratio=agent_holdings_ratio,
        wealth_concentration=wealth_concentration,
        holdings_scale_with_agents=False,  # Fixed user base for policy evaluation
        price_scenario=price_scenario,
        initial_price=initial_price,
        staking_price_impact=staking_price_impact,
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
    with st.spinner("Evaluating policy effectiveness..."):
        sim = DPoSSimulation(params)
        results = sim.simulate_option1(debug=False)
        
    # Store results in session state
    st.session_state['results'] = results
    st.session_state['params'] = params
    st.session_state['sim'] = sim

# Display results if available
if 'results' in st.session_state and 'params' in st.session_state and 'sim' in st.session_state:
    try:
        results = st.session_state['results']
        params = st.session_state['params']
        sim = st.session_state['sim']
        
        # Validate results structure
        if not all(key in results for key in ['apy_track', 'staked_track', 'total_staked', 'reserve_track', 'price_track']):
            st.error("❌ Invalid results structure. Please re-run the simulation.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error accessing simulation results: {e}")
        st.info("Please run a new simulation.")
        st.stop()
    
    # REDESIGNED METRICS - Policy Evaluation Focus
    st.header("📊 Policy Effectiveness")
    
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
        st.metric("🎯 User Participation", f"{estimated_user_participation:.1f}%", 
                 help="Estimated % of token holders who choose to stake")
        
    with col2:
        st.metric("💎 Supply Staked", f"{supply_participation:.1f}%", 
                 help="% of total token supply being staked")
        
    with col3:
        final_reserve = results['reserve_track'][-1]
        reserve_pct = (final_reserve / params.B0) * 100
        st.metric("🏛️ Reserve Health", f"{reserve_pct:.1f}%", 
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
        st.metric("⏱️ Policy Duration", sustainability,
                 help="How long this policy can be maintained")
    
    # Policy insights
    if estimated_user_participation < 50:
        st.warning("⚠️ **Low User Adoption**: Consider increasing APY or targeting more risk-tolerant users")
    elif estimated_user_participation > 90:
        st.success("🎉 **Excellent Adoption**: Policy is very attractive to users")
    else:
        st.info("✅ **Good Adoption**: Solid user participation rate")
    
    if supply_participation > 60:
        st.info("🔐 **High Security**: Large portion of supply is staked and securing the network")
    elif supply_participation < 20:
        st.warning("🔓 **Security Risk**: Low staking participation may reduce network security")

    # Charts
    st.header("📈 Results")
    
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
    
    # Policy-Focused 2x3 Grid Layout
    # Row 1: Key Policy Metrics
    col1, col2, col3 = st.columns(3)
    
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
            width=280, height=280
        ).resolve_scale(color='independent')
        st.altair_chart(chart1, use_container_width=False)
    
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
            width=280, height=280
        ).resolve_scale(color='independent')
        st.altair_chart(chart2, use_container_width=False)
    
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
            width=280, height=280
        ).resolve_scale(color='independent')
        st.altair_chart(chart3, use_container_width=False)
    
    # Row 2: Policy Design Analysis
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("Tier Preferences", help="Shows which lock-up periods users prefer: Liquid (no lock), 1-Year, or 2-Year commitments")
        
        chart4 = alt.Chart(tier_adoption_df).mark_area().encode(
            x=alt.X('Year:Q', title='Years'),
            y=alt.Y('Adoption (M):Q', title='Tokens (M)'),
            color=alt.Color('Tier:N', 
                           scale=alt.Scale(range=['#87CEEB', '#32CD32', '#FF6347']),
                           legend=None),
            order=alt.Order('Tier:N', sort='ascending')
        ).properties(width=280, height=280)
        st.altair_chart(chart4, use_container_width=False)
    
    with col5:
        st.subheader("Capital Allocation", help="How users split their token holdings between staking (earning rewards) vs keeping liquid (available for trading)")
        
        chart5 = alt.Chart(user_allocation_df).mark_area().encode(
            x=alt.X('Year:Q', title='Years'),
            y=alt.Y('Amount (M):Q', title='Holdings (M)'),
            color=alt.Color('Allocation:N', 
                           scale=alt.Scale(range=['#FF6B6B', '#4ECDC4']),
                           legend=None),
            order=alt.Order('Allocation:N', sort='descending')
        ).properties(width=280, height=280)
        st.altair_chart(chart5, use_container_width=False)
    
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
        ).properties(width=280, height=280)
        st.altair_chart(chart6, use_container_width=False)
    
    # Determine policy type for export filename
    input_liquid_apy = params.R_base  # This is the liquid APY we set as base
    if input_liquid_apy <= 0.08:
        policy_type = "Conservative"
    elif input_liquid_apy <= 0.12:
        policy_type = "Balanced"
    else:
        policy_type = "Aggressive"
    
    # Export section
    st.header("📋 Export Results")
    
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
            label="📥 Download Detailed Results",
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
            label="📋 Download Policy Summary",
            data=summary_csv,
            file_name=f"policy_summary_{policy_name}.csv",
            mime="text/csv",
            help="Download key policy metrics summary"
        )
    
    # Preview policy data
    st.subheader("📊 Policy Data Preview")
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
    st.header("🔍 Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reserve Analysis")
        if depletion_day:
            st.error(f"⚠️ Reserve depleted on day {depletion_day}")
            st.write(f"Reserve lasted {depletion_day/365:.1f} years")
        else:
            st.success("✅ Reserve maintained throughout simulation")
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
    st.info("👈 Configure parameters in the sidebar and click 'Run Policy Simulation' to see results!")
    
    # Show some example configurations
    st.header("💡 Example Configurations")
    
    examples = {
        "Conservative": {
            "Base APY": "8%",
            "Reserve": "200M tokens",
            "Description": "Lower rewards, larger reserve for long-term sustainability"
        },
        "Aggressive": {
            "Base APY": "20%", 
            "Reserve": "50M tokens",
            "Description": "High rewards, smaller reserve for rapid growth"
        },
        "Balanced": {
            "Base APY": "12%",
            "Reserve": "100M tokens", 
            "Description": "Moderate rewards and reserve size"
        }
    }
    
    for name, config in examples.items():
        with st.expander(f"{name} Configuration"):
            st.write(f"**Base APY:** {config['Base APY']}")
            st.write(f"**Reserve:** {config['Reserve']}")
            st.write(f"**Description:** {config['Description']}") 