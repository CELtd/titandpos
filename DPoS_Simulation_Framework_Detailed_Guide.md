# DPoS Staking Policy Simulator - Technical Guide

## Overview

This simulation models a **Delegated Proof of Stake (DPoS) ecosystem** where individual token holders (agents) make rational economic decisions about whether to stake their tokens based on offered rewards, personal risk tolerance, and market expectations.

The simulator helps answer policy questions:
- **What APY is needed to achieve target participation rates?**
- **How do different user risk profiles affect adoption?**
- **Can reward budgets sustain desired participation levels?**
- **What's the balance between user adoption and protocol costs?**

---

## Agent-Based Modeling Framework

### 1. Agent Types & Psychology

The simulation models **three distinct user archetypes** based on market sentiment:

#### 🐻 **Bearish Agents**
- **Risk Profile**: Highly risk-averse, pessimistic about token price
- **Hurdle Rate**: 15% (requires high compensation for risk)
- **Illiquidity Cost**: 3% (strongly dislikes being locked up)
- **Price Expectation**: -30% annual (expects token to lose value)
- **Behavior**: Only stakes when rewards significantly exceed risk perception

#### 😐 **Neutral Agents**
- **Risk Profile**: Moderate risk tolerance, realistic expectations
- **Hurdle Rate**: 8% (reasonable return requirements)
- **Illiquidity Cost**: 1.5% (moderate preference for liquidity)
- **Price Expectation**: 0% annual (expects stable token price)
- **Behavior**: Stakes when rewards provide fair compensation

#### 🚀 **Bullish Agents**
- **Risk Profile**: Risk-seeking, optimistic about token appreciation
- **Hurdle Rate**: 3% (willing to accept lower immediate returns)
- **Illiquidity Cost**: 0.5% (comfortable with lock-ups)
- **Price Expectation**: +50% annual (expects significant appreciation)
- **Behavior**: Stakes readily, sees staking as enhancing overall returns

### 2. Utility Calculation Framework

Each agent evaluates staking opportunities using a **multi-factor utility function**:

```
Utility = Daily_Staking_Yield + Daily_Price_Expectation - Daily_Illiquidity_Cost
```

#### **Daily Staking Yield**
- Direct rewards from staking (APY ÷ 365)
- Varies by tier: Liquid (1x), 1-Year (1.2x), 2-Year (1.5x)
- In Option 2, this changes dynamically based on participation

#### **Daily Price Expectation**
- Agent's personal belief about token appreciation
- Only applied when staking yields are positive (no rewards = no price exposure)
- Bullish agents see staking as amplifying price gains
- Bearish agents factor in expected price declines

#### **Daily Illiquidity Cost**
- Economic cost of having tokens locked up
- Scales with lock duration: Liquid (0 days), 1-Year (365 days), 2-Year (730 days)
- Formula: `Annual_Illiquidity_Cost × Duration_Years ÷ 365`

### 3. Decision-Making Process

**Step 1: Evaluate All Tiers**
- Agent calculates utility for Liquid, 1-Year, and 2-Year staking
- Compares against utility of not staking (always 0)

**Step 2: Apply Hurdle Rate Filter**
- Best staking utility must exceed agent's daily hurdle rate
- Hurdle rate represents minimum acceptable return for taking any risk

**Step 3: Stake Allocation**
- If conditions are met, agent stakes up to 80% of holdings (keeps 20% liquid)
- Chooses the tier with highest utility
- If no tier meets criteria, keeps all tokens liquid

### 4. Wealth Distribution & Staking Capacity

Agents have realistic **power-law distributed wealth** with multiple factors limiting total staking capacity:

#### **Market Structure Constraints**
- **Agent Holdings Ratio**: Agents collectively hold only 40% of circulating supply (configurable 20-80%)
- **Remaining 60%**: Held by institutions, exchanges, locked tokens, or inactive users not modeled as agents
- **Wealth Concentration**: Power-law distribution creates realistic inequality (small number of "whales")

#### **Individual Liquidity Preferences**
- **Max Stake Ratio**: Each agent stakes maximum 80% of their holdings (keeps 20% liquid)
- **Liquidity Buffer**: Agents need liquid tokens for trading, fees, and unexpected opportunities
- **Risk Management**: Even bullish agents maintain some portfolio liquidity

#### **Theoretical Maximum Staking**
```
Max Total Staking = Circulating Supply × Agent Holdings Ratio × Max Stake Ratio
                  = 100% × 40% × 80% = 32% of total supply
```

#### **Agent Scaling (Simulation Efficiency)**
- **Holdings Scale with Agents**: Fewer simulated agents = proportionally less total market participation
- **Base Reference**: 10,000 agents = 100% of normal market participation
- **Scaling Example**: 1,000 agents = 10% of normal participation (prevents unrealistic concentration)
- **Maximum Cap**: Agent holdings capped at 80% of circulating supply (prevents unrealistic scenarios)

#### **Dynamic Constraints During Simulation**
- **Behavioral Filters**: Agents only stake if utility exceeds their personal hurdle rate
- **Economic Rationality**: No staking when APY = 0% or negative expected returns
- **Available Holdings**: Agents can only stake tokens they actually possess at each time step
- **Demand Capping**: Total stake demand limited by total agent holdings (can't stake more than they own)

#### **Practical Staking Limits in Real Scenarios**

**Typical Participation Rates by Market Conditions:**
- **Bear Market**: 10-20% of supply staked (high hurdle rates, pessimistic expectations)
- **Neutral Market**: 25-35% of supply staked (moderate participation)
- **Bull Market**: 15-25% of supply staked (opportunity cost of missing price gains)

**Why Maximum (32%) is Rarely Reached:**
1. **Agent Psychology**: Not all agents find staking attractive even with good APY
2. **Market Dynamics**: Price expectations affect staking decisions
3. **Hurdle Rates**: Risk-averse agents require high compensation
4. **Timing**: Agents enter/exit staking based on changing conditions

---

## Option 1: Fixed Budget Policy

### Mechanism Overview

Option 1 simulates a **traditional rewards program** with a fixed budget and static APY rates.

#### **Core Components:**

1. **Fixed Reserve Budget** (e.g., 200M tokens)
2. **Static APY Structure** (e.g., 5% liquid, 8% 1-year, 10% 2-year)
3. **Budget Depletion Mechanics**
4. **Predictable Returns**

### Daily Simulation Flow

```
Day N:
├── Calculate fixed daily yields for each tier
├── Agents evaluate staking opportunities
├── Aggregate staking demand by tier
├── Deduct rewards from reserve budget
├── Update reserve health
└── Continue to Day N+1 (if budget remains)
```

#### **APY Calculation:**
```
Daily_Yield = (1 + Annual_APY)^(1/365) - 1
Tier_Yield = Base_Daily_Yield × Multiplier
```

#### **Budget Mechanics:**
```
Daily_Cost = Sum(Staked_Amount[tier] × Daily_Yield[tier] × Multiplier[tier])
New_Reserve = Previous_Reserve - Daily_Cost
```

### Key Characteristics

✅ **Predictable**: Users know exact returns upfront  
✅ **Simple**: Easy to understand and communicate  
❌ **Finite**: Budget depletes over time  
❌ **Rigid**: Can't adapt to changing participation  

### Policy Questions Answered

- **Budget Sustainability**: How long can this APY be maintained?
- **Participation Forecasting**: What participation rate will this APY achieve?
- **Reserve Optimization**: What's the optimal budget size for target participation?

---

## Option 2: Dynamic Emissions Policy

### Mechanism Overview

Option 2 simulates an **adaptive tokenomics system** where APY varies inversely with participation levels.

#### **Core Components:**

1. **Decaying Emission Schedule** (e.g., 50M tokens/year, 0.5 decay rate)
2. **Baseline Staking Amount** (e.g., 10M tokens from existing stakers)
3. **Dynamic APY Calculation**
4. **Self-Balancing Participation**

### Dynamic APY Formula

The key insight: **More participation = Lower APY for everyone**

```
Current_Weighted_Stake = Baseline_Staking + Agent_Weighted_Staking
Base_Daily_Yield = Daily_Emission ÷ Current_Weighted_Stake
Tier_APY = Base_Daily_Yield × Tier_Multiplier × 365
```

#### **Emission Decay:**
```
Daily_Emission = Initial_Annual_Emission × exp(-decay_rate × days/365)
```

#### **Weighted Staking:**
```
Weighted_Stake = Liquid_Stake×1.0 + OneYear_Stake×1.2 + TwoYear_Stake×1.5
```

### Daily Simulation Flow

```
Day N:
├── Calculate current emission rate (decaying)
├── Determine weighted stake (baseline + agents)
├── Calculate dynamic APY for each tier
├── Agents evaluate new staking opportunities
├── Update participation levels
├── APY adjusts based on new participation
└── Continue to Day N+1
```

### Feedback Loop Dynamics

**High APY → More Staking → Lower APY → Equilibrium**

1. **Initial State**: Few stakers, high APY
2. **Attraction Phase**: High APY attracts more stakers
3. **Saturation Phase**: More stakers reduce APY for everyone
4. **Equilibrium**: APY stabilizes at level that balances supply/demand

<!-- ### Baseline Staking Importance

**Purpose**: Prevents unrealistic APY spikes at launch

Without baseline: `50M emission ÷ 2M agent staking = 2,500% APY` 😱  
With 10M baseline: `50M emission ÷ 12M total staking = 417% APY` ✅ -->

### Key Characteristics

✅ **Self-Balancing**: APY naturally adjusts to participation  
✅ **Sustainable**: Emissions can continue indefinitely  
✅ **Market-Responsive**: Adapts to changing user behavior  
❌ **Unpredictable**: Users can't know future APY  
❌ **Complex**: Harder to understand and communicate  

### Policy Questions Answered

- **Natural Participation Rate**: What participation level will this system reach?
- **APY Range Forecasting**: What APY range can users expect?
- **Emission Optimization**: What emission schedule achieves target participation?
- **Market Adaptation**: How quickly does the system respond to changes?

---

## Simulation Implementation Details

### Time Structure

- **Daily Time Steps**: Each simulation day represents 24 hours
- **Multi-Year Horizons**: Typically 2-5 years for policy evaluation
- **Agent Persistence**: Same agents throughout simulation (no entry/exit)

### Market Dynamics

#### **Token Price Evolution**
```
Price_t+1 = Price_t × exp(Market_Drift + Market_Volatility × Random_Shock + Staking_Impact)
```

- **Market Scenarios**: Bearish (-30%), Neutral (0%), Bullish (+50%)
- **Staking Impact**: Higher staking can positively influence price
- **Volatility**: Realistic price fluctuations affect agent decisions

#### **Supply Mechanics**
```
Circulating_Supply_t = Initial_Supply + Daily_Vesting × Days_Elapsed
```

- **Daily Vesting**: New tokens enter circulation (e.g., 1M/day)
- **Max Supply Cap**: Vesting stops at predetermined maximum
- **Agent Holdings Growth**: Agents receive proportional share of new tokens

### Staking Capacity Implementation

The simulator enforces realistic staking limits through multiple mechanisms:

#### **Market Structure Setup**
```python
# Agent holdings = circulating_supply × agent_holdings_ratio × scaling_factor
total_agent_holdings = C0 * 0.4 * (N / 10000)  # 40% of supply, scaled by agent count
total_agent_holdings = min(total_agent_holdings, C0 * 0.8)  # Cap at 80% of supply
```

#### **Individual Staking Decisions**
```python
# Each agent stakes maximum 80% of their holdings
if utility_exceeds_hurdle_rate:
    stake_amount = available_tokens * 0.8  # Keep 20% liquid
else:
    stake_amount = 0  # Don't stake if not profitable
```

#### **Demand Capping**
```python
# Total staking demand cannot exceed what agents actually hold
total_demand = sum(individual_stake_demands)
total_agent_holdings = sum(agent_holdings_at_time_t)
if total_demand > total_agent_holdings:
    # Scale down all demands proportionally
    scale_factor = total_agent_holdings / total_demand
    actual_staking = total_demand * scale_factor
```

### Performance Optimizations

The simulator uses **vectorized calculations** for speed:

- **Batch Utility Calculation**: All agents evaluated simultaneously
- **NumPy Operations**: Efficient mathematical operations
- **Vectorized Decision Making**: Thousands of agents processed in milliseconds

---

## Interpreting Results

### Key Metrics

#### **User Engagement (%)**
- Percentage of agent token holdings actively staked
- Higher = more effective policy

#### **Supply Participation (%)**
- Percentage of total token supply being staked
- Higher = better network security

#### **Reserve Health** (Option 1)
- Remaining budget as percentage of initial
- Tracks policy sustainability

#### **APY Volatility** (Option 2)
- How much APY fluctuates over time
- Lower = more predictable for users

### Comparative Analysis

| Aspect | Option 1 (Fixed) | Option 2 (Dynamic) |
|--------|------------------|---------------------|
| **Predictability** | High | Low |
| **Sustainability** | Limited | Infinite |
| **User Experience** | Simple | Complex |
| **Policy Control** | Direct | Indirect |
| **Market Adaptation** | None | Automatic |

---

## Limitations & Assumptions

### Agent Behavior
- **Rational Actors**: Agents always maximize utility (real users may not)
- **Perfect Information**: Agents see all available options instantly
- **No Learning**: Agent preferences don't evolve over time
- **No Social Effects**: Agents don't influence each other
- **Fixed Liquidity Preference**: All agents maintain exactly 20% liquid (no variation)
- **Instant Execution**: Staking/unstaking happens immediately (no delays or penalties)

### Market Dynamics
- **Simplified Price Model**: Basic geometric Brownian motion
- **No External Factors**: Doesn't model regulatory changes, competitor actions
- **Fixed User Base**: No new users enter or existing users leave

### Technical Limitations
- **Daily Resolution**: Can't model intraday behavior
- **Deterministic Preferences**: Agent types are fixed categories
- **No Gas Costs**: Transaction fees not considered
- **Perfect Execution**: All staking actions execute instantly

---

## Conclusion

This simulation provides a **powerful framework for evaluating DPoS staking policies** by modeling realistic user behavior and economic dynamics. While simplified compared to real-world complexity, it captures the essential tradeoffs between:

- **User Adoption vs. Protocol Costs**
- **Predictability vs. Sustainability** 
- **Immediate Rewards vs. Long-term Viability**

By understanding these mechanics, projects can design more effective tokenomics that balance user incentives with protocol sustainability, ultimately leading to healthier and more resilient DPoS ecosystems. 