# DPoS Staking Simulation Framework Reference

## Overview

This framework simulates Delegated Proof of Stake (DPoS) staking economics with realistic agent-based modeling. It evaluates different staking policy designs by modeling user behavior, token economics, and network security dynamics.

## Core Architecture

### 1. Simulation Classes

#### `StakingTier` (Enum)
Defines the three staking commitment levels:
- **LIQUID**: No lock-up period, immediate liquidity
- **ONE_YEAR**: 1-year commitment, higher rewards
- **TWO_YEAR**: 2-year commitment, highest rewards

#### `SimulationParams` (Dataclass)
Configuration container for all simulation parameters with performance optimizations.

#### `DPoSSimulation` (Main Class)
Core simulation engine that orchestrates agent behavior, token economics, and policy evaluation.

---

## Key Parameters

### Time & Performance Parameters
```python
T: int = 365 * 2          # Simulation duration in days (default: 2 years)
E: int = 365              # Epochs per year (daily epochs)
time_step: int = 1        # Time step size (1=daily, 7=weekly, 30=monthly)
```

### Staking Policy Parameters
```python
R_base: float = 0.12      # Base annual APY (12%)
B0: float = 100e6         # Initial reserve budget (100M tokens - default in UX)
multipliers: Dict = {     # Tier reward multipliers
    LIQUID: 1.0,          # 1x base rate
    ONE_YEAR: 1.2,        # 1.2x base rate  
    TWO_YEAR: 1.5         # 1.5x base rate
}
```

### Agent Behavior Parameters (Sentiment-Based)
```python
N: int = 10000            # Number of simulated users

# Market sentiment distribution (must sum to 1.0)
bearish_pct: float = 0.3    # 30% bearish agents
neutral_pct: float = 0.5    # 50% neutral agents  
bullish_pct: float = 0.2    # 20% bullish agents

# Bearish agent characteristics (risk-averse, pessimistic)
bearish_hurdle_rate: float = 0.15      # 15% hurdle rate
bearish_illiquidity_cost: float = 0.03 # 3% illiquidity cost
bearish_price_expectation: float = -0.3 # -30% annual price expectation

# Neutral agent characteristics (moderate expectations)
neutral_hurdle_rate: float = 0.08      # 8% hurdle rate
neutral_illiquidity_cost: float = 0.015 # 1.5% illiquidity cost
neutral_price_expectation: float = 0.0  # 0% annual price expectation

# Bullish agent characteristics (risk-seeking, optimistic)
bullish_hurdle_rate: float = 0.03      # 3% hurdle rate
bullish_illiquidity_cost: float = 0.005 # 0.5% illiquidity cost
bullish_price_expectation: float = 0.5  # 50% annual price expectation

# Wealth distribution and market structure
wealth_concentration: float = 1.5  # Power law exponent - controls wealth inequality
holdings_scale_with_agents: bool = True  # Scale total holdings with agent count
```

**Wealth Concentration Explained:**
The `wealth_concentration` parameter controls how unequally tokens are distributed among agents using a power law (Pareto) distribution:
- **1.0**: Equal wealth distribution (all agents have similar holdings)
- **1.5**: Moderate inequality (realistic for most crypto markets)
- **2.0-3.0**: High inequality (small number of "whales" hold most tokens)

Higher values create more realistic simulations where a few large holders dominate staking decisions, similar to real cryptocurrency markets.

**Holdings Scaling Explained:**
The `holdings_scale_with_agents` parameter determines how total market participation scales with the number of simulated agents:
- **True (recommended)**: Fewer agents = proportionally less total market participation
  - 1,000 agents = 10% of normal market participation
  - 10,000 agents = 100% of normal market participation
  - This creates realistic scenarios where reducing agents for faster simulation doesn't artificially inflate participation rates
- **False**: Fixed total market participation regardless of agent count
  - Can lead to unrealistic scenarios where 100 agents control 40% of token supply

### Token Economics Parameters
```python
C0: float = 50e6          # Initial circulating supply (50M)
daily_vesting: float = 1e6 # Daily token vesting (1M/day)
max_supply: float = 2e9   # Maximum supply cap (2B tokens)

# Market participation structure
agent_holdings_ratio: float = 0.4  # % of circulating supply held by users (40%)
max_stake_ratio: float = 0.8       # % of user holdings willing to stake (80%)

# Combined effect: Maximum possible staking = 40% × 80% = 32% of circulating supply
```

### Price Model Parameters (Enhanced GBM)
```python
initial_price: float = 0.1       # Starting token price ($)
staking_price_impact: float = 0.1 # Maximum price impact from staking (10%)

# Market scenario parameters (GBM)
market_drift: float = 0.0        # Annual price drift (0=neutral, +0.5=50% bull, -0.3=30% bear)
market_volatility: float = 0.3   # Annual price volatility (0.3 = 30%)
```

**Market Scenario Presets:**
- **🐻 Bearish**: -30% annual drift, 40% volatility
- **😐 Neutral**: 0% annual drift, 30% volatility  
- **🚀 Bullish**: +50% annual drift, 35% volatility

**Parameter Interaction:**
- `agent_holdings_ratio`: Defines market structure - what portion of tokens are held by potential stakers
- `max_stake_ratio`: Defines individual behavior - users keep 20% liquid for trading/fees
- `holdings_scale_with_agents`: If True, fewer agents = proportionally less total market participation
- **True staking ceiling**: `agent_holdings_ratio × max_stake_ratio` = 32% of total supply

---

## Agent Behavior Model

### Enhanced Sentiment-Based Heterogeneity
Each agent is assigned to one of three sentiment categories with distinct characteristics:

1. **Bearish Agents** (30% default): Risk-averse, pessimistic
   - High hurdle rates (15% default)
   - High illiquidity costs (3% default)
   - Negative price expectations (-30% default)

2. **Neutral Agents** (50% default): Moderate expectations
   - Medium hurdle rates (8% default)
   - Medium illiquidity costs (1.5% default)
   - Zero price expectations (0% default)

3. **Bullish Agents** (20% default): Risk-seeking, optimistic
   - Low hurdle rates (3% default)
   - Low illiquidity costs (0.5% default)
   - Positive price expectations (50% default)

4. **Token Holdings**: Power-law distributed wealth with realistic agent scaling

### Enhanced Decision-Making Process

#### Improved Utility Function
For each staking tier, agents calculate utility considering price expectations:
```
utility = daily_yield + individual_price_expectation - illiquidity_cost
```

Where:
- **daily_yield**: APY converted to daily rate (tier-specific)
- **individual_price_expectation**: Agent's personal price expectation (sentiment-based)
- **illiquidity_cost**: Penalty for lock-up duration based on agent type

#### Staking Decision Logic
1. Calculate utility for each tier (Liquid, 1-Year, 2-Year) for all agents simultaneously
2. Compare best staking utility vs. utility of NOT staking (0)
3. Only stake if:
   - Best utility > hurdle rate (`rho`)
   - Best utility > 0 (better than not staking)
   - Agent has tokens available

#### Realistic Constraints
- **Liquidity preference**: Agents only stake up to 80% of holdings (keep 20% liquid)
- **Economic rationality**: No staking when APY = 0% 
- **Dynamic response**: Agents adjust behavior based on changing reward conditions
- **Market structure**: Only 40% of circulating supply is held by potential stakers
- **Agent scaling**: Holdings scale realistically with number of simulated agents

---

## Economic Models

### 1. Enhanced Token Supply Model
```python
def calculate_circulating_supply(t: int) -> float:
    uncapped_supply = C0 + t * daily_vesting
    return min(uncapped_supply, max_supply)  # Supply cap prevents infinite inflation
```

**Features:**
- Linear vesting schedule with supply cap
- Realistic token release mechanism
- Prevents infinite inflation scenarios

### 2. Advanced Price Model (Market Scenario GBM)
```python
# Use explicit market scenario parameters
mu = market_drift  # Annual drift from market scenario preset
sigma = market_volatility  # Annual volatility from market scenario preset

# Base GBM evolution
dS = S * (mu*dt + sigma*dW)

# Staking impact with sigmoid function
staking_impact = price_impact * sigmoid(staking_ratio - 0.5)
final_price = base_gbm_price * (1 + staking_impact)
```

**Market Scenarios** (explicit presets):
- **Bearish**: -30% annual drift, 40% volatility (declining market)
- **Neutral**: 0% annual drift, 30% volatility (stable market)
- **Bullish**: +50% annual drift, 35% volatility (rising market)

### 3. Reserve Depletion Model (Option 1)
```python
# Daily reserve burn calculation
delta_B = daily_rate * weighted_stake
B_current = max(0, B_current - delta_B)

# APY calculation with depletion handling
if B_current > 0:
    daily_yield = multiplier * base_rate
else:
    daily_yield = 0.0  # No rewards when depleted
```

---

## Simulation Algorithms

### Core Simulation Loop
```python
for t in range(simulation_duration):
    1. update_agent_holdings(t)          # Distribute vesting (with scaling)
    2. calculate_circulating_supply(t)   # Update token supply (with cap)
    3. calculate_daily_yields(t)         # Determine APY by tier
    4. make_agent_decisions(t)           # Agent staking choices (optimized)
    5. apply_demand_constraints(t)       # Cap to available holdings
    6. update_reserves(t)                # Burn rewards from budget
    7. calculate_price_impact(t)         # Update token price (GBM + staking)
    8. track_metrics(t)                  # Record all tracking data
```

### Key Algorithms

#### 1. Realistic Agent Holdings Distribution
```python
def setup_agents():
    # Scale total holdings with agent count for realism
    if holdings_scale_with_agents:
        base_agent_count = 10000
        scaling_factor = N / base_agent_count
        total_agent_holdings = circulating_supply * agent_holdings_ratio * scaling_factor
        total_agent_holdings = min(total_agent_holdings, circulating_supply * 0.8)
    else:
        total_agent_holdings = circulating_supply * agent_holdings_ratio
    
    # Power law wealth distribution (realistic inequality)
    pareto_samples = np.random.pareto(wealth_concentration, N)
    normalized_wealth = pareto_samples / np.sum(pareto_samples)
    individual_holdings = normalized_wealth * total_agent_holdings
```

#### 2. Enhanced Demand Capping Algorithm
```python
def apply_demand_constraints():
    total_demand = sum(demand_by_tier.values())
    
    # Maximum stakeable = agent holdings × max_stake_ratio (80%)
    total_stakeable = sum(agent_holdings) * max_stake_ratio
    
    if total_demand > total_stakeable:
        # Scale down proportionally to respect liquidity preferences
        scale_factor = total_stakeable / total_demand
        for tier in tiers:
            actual_stake[tier] = demand[tier] * scale_factor
```

---

## Output & Analysis

### Enhanced Results Dictionary
```python
results = {
    'reserve_track': np.array,      # Reserve budget over time
    'staked_track': Dict[Tier, np.array],  # Staking by tier over time
    'apy_track': Dict[Tier, np.array],     # APY by tier over time  
    'total_staked': np.array,       # Total staked tokens over time
    'price_track': np.array,        # Token price over time (GBM + staking impact)
}
```

### New Analysis Methods

#### `create_comparison_dataframe()`
Generates comprehensive DataFrame with:
- Agent participation rates (% of agent holdings staked)
- Market participation (% of circulating supply staked)
- Unstaked agent holdings
- Daily stake changes and reserve burn rates
- APY tracking by tier

#### Enhanced Plotting (`plot_results()`)
- **Agent participation rate**: % of user holdings staked
- **Market participation rate**: % of circulating supply staked  
- **Holdings breakdown**: Staked vs liquid agent holdings
- **Reserve health**: Depletion tracking with alerts
- **Price evolution**: GBM path with staking impact
- **Tier distribution**: Preference analysis across lock periods

### Key Performance Indicators
1. **User Engagement**: % of user holdings actively staked
2. **Supply Participation**: % of circulating supply staked
3. **Reserve Health**: % of original budget remaining
4. **Policy Duration**: Days until reserve depletion
5. **Tier Preferences**: Distribution across lock-up periods
6. **Price Impact**: Token price evolution from staking

---

## Usage Examples

### Policy Simulation with Sentiment Analysis
```python
from sim import DPoSSimulation, SimulationParams

# Configure sentiment-based simulation
params = SimulationParams(
    T=365*5,              # 5 years
    N=2000,               # 2000 agents
    R_base=0.12,          # 12% base APY
    B0=100e6,             # 100M reserve (matches UX default)
    
    # Market sentiment (must sum to 1.0)
    bearish_pct=0.4,      # 40% bearish
    neutral_pct=0.4,      # 40% neutral
    bullish_pct=0.2,      # 20% bullish
    
    # Market scenario (GBM parameters)
    initial_price=0.1,    # Start at $0.10
    market_drift=0.0,     # Neutral market (0% annual drift)
    market_volatility=0.3, # 30% annual volatility
    
    # Wealth distribution and scaling
    wealth_concentration=1.5,     # Moderate inequality
    holdings_scale_with_agents=True  # Realistic scaling
)

# Run simulation
sim = DPoSSimulation(params)
results = sim.simulate_option1()

# Create analysis DataFrame
df = sim.create_comparison_dataframe(results)

# Key metrics
agent_participation = results['total_staked'] / sim.agent_holdings_track.sum(axis=0)
market_participation = results['total_staked'] / sim.C_track
reserve_health = results['reserve_track'] / params.B0
price_performance = results['price_track'][-1] / params.initial_price

print(f"Average agent participation: {agent_participation.mean():.1%}")
print(f"Average market participation: {market_participation.mean():.1%}")
print(f"Token price performance: {price_performance:.2f}x")
```

### Performance Comparison
```python
# Fast exploration mode
fast_params = SimulationParams(
    fast_mode=True,       # Enable optimizations
    time_step=7,          # Weekly simulation
    vectorized=True,      # Use vectorized calculations
    N=500                 # Fewer agents
)

# Production analysis mode
production_params = SimulationParams(
    fast_mode=False,      # Maximum accuracy
    time_step=1,          # Daily simulation
    vectorized=True,      # Still use vectorization
    N=5000               # More agents for precision
)
```

---

## Model Validation & Performance

### Economic Realism Checks
1. **Zero APY → Zero Participation**: ✅ Agents don't stake when rewards = 0%
2. **Agent Count Scaling**: ✅ Fewer agents = less total market participation
3. **Wealth Inequality**: ✅ Power-law distribution matches real-world token holdings
4. **Liquidity Preference**: ✅ Agents keep portion of holdings liquid
5. **Sentiment Impact**: ✅ Price expectations affect staking decisions
6. **Supply Cap Enforcement**: ✅ Prevents infinite inflation scenarios

### Performance Benchmarks
- **Daily, 1000 agents, 2 years**: ~0.04 seconds
- **Daily, 10000 agents, 2 years**: ~0.3 seconds  
- **Weekly, 10000 agents, 5 years**: ~0.1 seconds

### New Validation Results
- ✅ GBM price modeling reflects market sentiment
- ✅ Agent sentiment affects participation patterns
- ✅ Supply cap prevents unrealistic inflation
- ✅ Holdings scaling improves realism with different agent counts

---

## Extension Points

### Adding New Features
1. **New Staking Tiers**: Extend `StakingTier` enum and multipliers
2. **Custom Utility Functions**: Override agent utility calculations
3. **Alternative Price Models**: Replace price path generation
4. **Dynamic Parameters**: Make parameters time-dependent
5. **Custom Sentiment Models**: Extend agent behavior classification

### Research Applications
- **Tokenomics Design**: Optimize APY and reserve allocation
- **User Adoption Modeling**: Predict participation rates by sentiment
- **Economic Security Analysis**: Balance rewards vs. network security
- **Market Impact Studies**: Analyze price effects of staking policies
- **Behavioral Economics**: Study sentiment-based decision making

---

## Technical Notes

### Dependencies
```python
import numpy as np          # Numerical computations
import pandas as pd         # Data analysis & DataFrame creation
import matplotlib.pyplot as plt  # Enhanced visualization
from dataclasses import dataclass  # Configuration management
from enum import Enum       # Type safety
from typing import Dict, List, Tuple  # Type hints
```

### Memory & Performance Considerations
- **Agent tracking**: O(N × T) memory usage
- **Large simulations**: Use time stepping for N > 10,000
- **Long durations**: Consider weekly/monthly steps for T > 365×5

### Numerical Stability & Robustness
- Prevents negative prices (minimum 0.01)
- Handles division by zero in APY calculations
- Caps demand to available holdings
- Uses stable sigmoid for price impact
- GBM price model prevents extreme price movements
- Supply cap enforcement prevents overflow scenarios

---

This enhanced framework provides a comprehensive foundation for DPoS staking policy evaluation with realistic agent behavior, advanced economic modeling, sentiment-based analysis, and performance optimizations suitable for both research and production use cases. 