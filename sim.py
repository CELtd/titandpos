import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import pandas as pd

class StakingTier(Enum):
    LIQUID = "L"
    ONE_YEAR = "1"
    TWO_YEAR = "2"

@dataclass
class SimulationParams:
    # Time parameters
    T: int = 365 * 2  # 2 years in days
    E: int = 365      # Epochs per year
    time_step: int = 1  # NEW: Time step in days (1=daily, 7=weekly, 30=monthly)
    
    # Option 1 parameters
    R_base: float = 0.12  # Base annual APY
    B0: float = 1e9       # Initial reserve budget
    
    # Option 2 parameters  
    E0: float = 1e8       # Initial annual emission
    k: float = 0.5        # Decay rate
    baseline_staking: float = 10e6  # Initial baseline staking amount (existing stakers)
    
    # Tier multipliers
    multipliers: Dict[StakingTier, float] = None
    durations: Dict[StakingTier, float] = None
    
    # Agent parameters
    N: int = 10000        # Number of agents
    
    # Market sentiment distribution (percentages must sum to 100%)
    bearish_pct: float = 0.3    # 30% bearish agents
    neutral_pct: float = 0.5    # 50% neutral agents  
    bullish_pct: float = 0.2    # 20% bullish agents
    
    # Bearish agent characteristics
    bearish_hurdle_rate: float = 0.15      # 15% hurdle rate (very risk averse)
    bearish_illiquidity_cost: float = 0.03 # 3% illiquidity cost (hate being locked up)
    bearish_price_expectation: float = -0.3 # -30% annual price expectation
    
    # Neutral agent characteristics  
    neutral_hurdle_rate: float = 0.08      # 8% hurdle rate (moderate)
    neutral_illiquidity_cost: float = 0.015 # 1.5% illiquidity cost (moderate)
    neutral_price_expectation: float = 0.0  # 0% annual price expectation
    
    # Bullish agent characteristics
    bullish_hurdle_rate: float = 0.03      # 3% hurdle rate (risk seeking)
    bullish_illiquidity_cost: float = 0.005 # 0.5% illiquidity cost (don't mind locks)
    bullish_price_expectation: float = 0.5  # 50% annual price expectation
    
    # Performance parameters
    fast_mode: bool = False  # NEW: Enable fast mode optimizations
    vectorized: bool = True  # NEW: Use vectorized calculations when possible
    
    # Supply parameters
    C0: float = 50e6      # Initial circulating supply (50M tokens)
    daily_vesting: float = 1e6  # Daily vesting amount (1M tokens/day)
    max_supply: float = 2e9  # NEW: Maximum circulating supply cap (2B tokens)
    
    # Agent token holdings
    agent_holdings_ratio: float = 0.4  # Agents collectively hold 40% of circulating supply
    wealth_concentration: float = 1.5   # Power law exponent (higher = more concentrated)
    holdings_scale_with_agents: bool = True  # NEW: If True, fewer agents = proportionally less total holdings
    max_stake_ratio: float = 0.8  # NEW: Maximum % of holdings each agent will stake (keep 20% liquid)
    
    # Stake distribution
    stake_per_agent_ratio: float = 0.001  # Each agent stakes 0.1% of circulating supply on average
    
    # Supply parameters
    C0: float = 1e9       # Initial circulating supply
    
    # Price model parameters (GBM)
    initial_price: float = 0.1       # Starting token price
    staking_price_impact: float = 0.1  # How much staking affects price (0.1 = 10% max impact)
    
    # Market scenario parameters (GBM)
    market_drift: float = 0.0        # Annual price drift (0 = neutral, +0.5 = 50% bull, -0.3 = 30% bear)
    market_volatility: float = 0.3   # Annual price volatility (0.3 = 30%)
    
    def __post_init__(self):
        if self.multipliers is None:
            self.multipliers = {
                StakingTier.LIQUID: 1.0,
                StakingTier.ONE_YEAR: 1.2, 
                StakingTier.TWO_YEAR: 1.5
            }
        if self.durations is None:
            self.durations = {
                StakingTier.LIQUID: 0.0,
                StakingTier.ONE_YEAR: 1.0,
                StakingTier.TWO_YEAR: 2.0
            }
        
        # Adjust parameters for time stepping
        if self.time_step > 1:
            self.daily_vesting *= self.time_step  # Scale vesting for larger time steps
            
        # Fast mode adjustments
        if self.fast_mode:
            self.N = min(self.N, 1000)  # Cap agents at 1000 for fast mode
            self.time_step = max(self.time_step, 7)  # Minimum weekly simulation

class DPoSSimulation:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.setup_agents()
        self.setup_tracking()
        
    def setup_agents(self):
        """Initialize agent characteristics based on market sentiment distribution"""
        np.random.seed(42)  # For reproducibility
        
        # Validate sentiment percentages sum to 1.0
        total_pct = self.params.bearish_pct + self.params.neutral_pct + self.params.bullish_pct
        if abs(total_pct - 1.0) > 0.001:
            raise ValueError(f"Sentiment percentages must sum to 1.0, got {total_pct}")
        
        # Determine agent types based on percentages
        n_bearish = int(self.params.N * self.params.bearish_pct)
        n_neutral = int(self.params.N * self.params.neutral_pct)
        n_bullish = self.params.N - n_bearish - n_neutral  # Ensure exact total
        
        # Create agent type assignments
        agent_types = (['bearish'] * n_bearish + 
                      ['neutral'] * n_neutral + 
                      ['bullish'] * n_bullish)
        np.random.shuffle(agent_types)  # Randomize order
        self.agent_types = np.array(agent_types)
        
        # Initialize arrays for agent characteristics
        self.rho = np.zeros(self.params.N)           # Annual hurdle rates
        self.lambda_agent = np.zeros(self.params.N)  # Annual illiquidity costs
        self.price_expectations = np.zeros(self.params.N)  # Annual price expectations
        
        # Assign characteristics based on agent type
        for i, agent_type in enumerate(agent_types):
            if agent_type == 'bearish':
                self.rho[i] = self.params.bearish_hurdle_rate
                self.lambda_agent[i] = self.params.bearish_illiquidity_cost
                self.price_expectations[i] = self.params.bearish_price_expectation
            elif agent_type == 'neutral':
                self.rho[i] = self.params.neutral_hurdle_rate
                self.lambda_agent[i] = self.params.neutral_illiquidity_cost
                self.price_expectations[i] = self.params.neutral_price_expectation
            else:  # bullish
                self.rho[i] = self.params.bullish_hurdle_rate
                self.lambda_agent[i] = self.params.bullish_illiquidity_cost
                self.price_expectations[i] = self.params.bullish_price_expectation
        
        # Convert annual rates to daily equivalents for simulation
        self.delta = (1 + self.rho) ** (1/self.params.E) - 1  # Daily hurdle rates
        
        # Generate wealth distribution using power law (Pareto distribution)
        # This creates realistic wealth inequality
        initial_circ_supply = self.params.C0
        
        # FIX: Make agent holdings scale realistically with number of agents
        if self.params.holdings_scale_with_agents:
            # Realistic scaling: fewer agents = proportionally less total market participation
            # Base calculation assumes 10,000 agents as reference point
            base_agent_count = 10000
            scaling_factor = self.params.N / base_agent_count
            total_agent_holdings = initial_circ_supply * self.params.agent_holdings_ratio * scaling_factor
            
            # Cap at maximum circulating supply to prevent unrealistic scenarios
            total_agent_holdings = min(total_agent_holdings, initial_circ_supply * 0.8)  # Max 80% held by agents
        else:
            # Original behavior: fixed percentage regardless of agent count
            total_agent_holdings = initial_circ_supply * self.params.agent_holdings_ratio
        
        # Generate power law distributed wealth
        pareto_samples = np.random.pareto(self.params.wealth_concentration, self.params.N)
        pareto_normalized = pareto_samples / np.sum(pareto_samples)
        
        # Scale to total agent holdings
        self.agent_token_holdings = pareto_normalized * total_agent_holdings
        
        # Initialize tracking for agent holdings over time (they receive new tokens from vesting)
        self.agent_holdings_track = np.zeros((self.params.N, self.params.T))
        
    def update_agent_holdings(self, t: int):
        """Update agent holdings based on new vesting"""
        if not hasattr(self, 'agent_holdings_track'):
            raise ValueError("agent_holdings_track not initialized!")
            
        if t == 0:
            # Initial holdings
            for i in range(self.params.N):
                self.agent_holdings_track[i, t] = self.agent_token_holdings[i]
        else:
            # Previous holdings plus share of new vesting
            # Handle time stepping by using the most recent available data
            prev_t = max(0, t - self.params.time_step)
            if prev_t >= self.params.T:
                prev_t = self.params.T - 1
                
            # FIX: Vesting should scale with agent holdings scaling
            base_new_vesting = self.params.daily_vesting * self.params.agent_holdings_ratio
            
            if self.params.holdings_scale_with_agents:
                # Scale vesting proportionally to agent count (same as holdings scaling)
                base_agent_count = 10000
                scaling_factor = self.params.N / base_agent_count
                new_vesting = base_new_vesting * scaling_factor
                # Cap vesting to prevent unrealistic scenarios
                max_vesting = self.params.daily_vesting * 0.8  # Max 80% of daily vesting
                new_vesting = min(new_vesting, max_vesting)
            else:
                # Original behavior: fixed percentage regardless of agent count  
                new_vesting = base_new_vesting
            
            # For time stepping, scale the vesting appropriately
            if self.params.time_step > 1:
                new_vesting *= self.params.time_step
            
            # Distribute new vesting proportionally to existing holdings (rich get richer)
            total_existing = np.sum(self.agent_holdings_track[:, prev_t])
            
            for i in range(self.params.N):
                if prev_t < self.agent_holdings_track.shape[1]:
                    prev_holdings = self.agent_holdings_track[i, prev_t]
                else:
                    prev_holdings = self.agent_token_holdings[i]
                    
                # Remove any staked tokens from available holdings
                available_holdings = prev_holdings  # For now, assume unstaking is instant
                
                # Add proportional share of new vesting
                if total_existing > 0:
                    vesting_share = (prev_holdings / total_existing) * new_vesting
                else:
                    vesting_share = new_vesting / self.params.N
                    
                self.agent_holdings_track[i, t] = available_holdings + vesting_share
        
    def setup_tracking(self):
        """Initialize tracking arrays"""
        self.tiers = list(StakingTier)
        
        # Common tracking
        self.C_track = np.zeros(self.params.T)
        self.price_track = np.ones(self.params.T + 1) * self.params.initial_price  # Start at initial price
        self.total_staked_track = np.zeros(self.params.T)
        
        # Option-specific tracking  
        self.S_track = {tier: np.zeros(self.params.T) for tier in self.tiers}
        self.apy_track = {tier: np.zeros(self.params.T) for tier in self.tiers}
        
        # Option 1 specific
        self.B_track_opt1 = np.zeros(self.params.T)
        
        # Option 2 specific
        self.emission_track_opt2 = np.zeros(self.params.T)
        self.weighted_stake_track = np.zeros(self.params.T)
        
        # Generate GBM price path
        self._generate_price_path()
    
    def _generate_price_path(self):
        """Generate realistic price path using Geometric Brownian Motion"""
        # Use explicit market scenario parameters
        mu = self.params.market_drift  # Annual drift from market scenario
        sigma = self.params.market_volatility  # Annual volatility from market scenario
        
        # Convert to daily parameters
        dt = 1/365  # Daily time step
        mu_daily = mu * dt
        sigma_daily = sigma * np.sqrt(dt)
        
        # Generate random shocks (reproducible with seed)
        np.random.seed(42)  # For reproducibility
        shocks = np.random.normal(0, 1, self.params.T)
        
        # Generate base GBM path
        base_price_track = np.zeros(self.params.T + 1)
        base_price_track[0] = self.params.initial_price
        
        for t in range(self.params.T):
            # Standard GBM formula: dS = S * (mu*dt + sigma*dW)
            base_price_track[t+1] = base_price_track[t] * np.exp(
                mu_daily - 0.5 * sigma_daily**2 + sigma_daily * shocks[t]
            )
        
        # Store the base price path (will be modified by staking impact)
        self.base_price_track = base_price_track.copy()
        self.price_track = base_price_track.copy()
        
    def calculate_circulating_supply(self, t: int) -> float:
        """Calculate circulating supply with linear vesting schedule and maximum cap"""
        uncapped_supply = self.params.C0 + t * self.params.daily_vesting
        return min(uncapped_supply, self.params.max_supply)
        
    def calculate_price_impact(self, t: int, total_staked: float) -> float:
        """Realistic price model: GBM base + staking impact"""
        circ_supply = self.calculate_circulating_supply(t)
        staking_ratio = total_staked / circ_supply if circ_supply > 0 else 0
        
        # Base GBM price from the pre-generated path
        base_price = self.base_price_track[t+1]
        
        # Staking impact: higher staking ratio -> price premium
        # Use a sigmoid-like function to prevent extreme impacts
        staking_impact = self.params.staking_price_impact * (
            2 / (1 + np.exp(-10 * (staking_ratio - 0.5))) - 1
        )
        
        # Apply staking impact to base price
        final_price = base_price * (1 + staking_impact)
        
        return max(final_price, 0.01)  # Prevent negative prices
        
    def calculate_agent_utility(self, tier: StakingTier, daily_yield: float, t: int) -> np.ndarray:
        """Calculate utility for each agent for a given tier"""
        duration = self.params.durations[tier]
        # Convert annual illiquidity cost to daily
        daily_illiquidity_cost = self.lambda_agent * duration / self.params.E
        
        # Use individual agent price expectations (converted to daily)
        daily_price_expectations = np.zeros(self.params.N)
        if daily_yield > 0:  # Only apply price appreciation if there are staking rewards
            daily_price_expectations = self.price_expectations / 365  # Convert annual to daily
        
        # Agents consider both yield AND their individual expected token appreciation
        total_expected_return = daily_yield + daily_price_expectations
        
        # Net utility after illiquidity cost
        utility = total_expected_return - daily_illiquidity_cost
        
        # Add small noise if there are actual rewards
        if daily_yield > 0:
            noise = np.random.normal(0, 0.0001, len(self.lambda_agent))
            utility = utility + noise
        
        return utility
        
    def calculate_agent_utility_vectorized(self, daily_yields: Dict[StakingTier, float], t: int) -> Dict[StakingTier, np.ndarray]:
        """Vectorized utility calculation for all agents and tiers at once"""
        utilities = {}
        
        # Calculate utilities for each tier (vectorized)
        for tier in self.tiers:
            duration = self.params.durations[tier]
            daily_illiquidity_cost = self.lambda_agent * duration / self.params.E
            daily_yield = daily_yields[tier]
            
            # Use individual agent price expectations (vectorized)
            daily_price_expectations = np.zeros(self.params.N)
            if daily_yield > 0:  # Only apply price appreciation if there are staking rewards
                daily_price_expectations = self.price_expectations / 365  # Convert annual to daily
            
            total_expected_return = daily_yield + daily_price_expectations
            
            # Vectorized utility calculation for all agents
            utilities[tier] = total_expected_return - daily_illiquidity_cost
            
            # Add noise if there are actual rewards and not in fast mode
            if daily_yield > 0 and not self.params.fast_mode:
                noise = np.random.normal(0, 0.0001, self.params.N)
                utilities[tier] += noise
                
        return utilities
    
    def make_agent_decisions_vectorized(self, daily_yields: Dict[StakingTier, float], t: int) -> Dict:
        """Vectorized agent decision making - much faster than individual loops"""
        # Get agent holdings for this period
        available_tokens = self.agent_holdings_track[:, t]
        
        # Agents with no tokens can't stake
        has_tokens_mask = available_tokens > 0
        
        if not np.any(has_tokens_mask):
            return {
                'demand': {tier: 0.0 for tier in self.tiers},
                'agent_count': {tier: 0 for tier in self.tiers},
                'unstaking_agents': self.params.N
            }
        
        # Calculate utilities for all agents and tiers
        utilities = self.calculate_agent_utility_vectorized(daily_yields, t)
        
        # FIX: Add the utility of NOT staking (which is 0)
        # This is critical - agents should compare staking utilities against doing nothing
        utility_not_staking = np.zeros(self.params.N)  # Utility of not staking = 0
        
        # Find best tier for each agent (vectorized)
        utility_matrix = np.array([utilities[tier] for tier in self.tiers])  # Shape: (3, N)
        best_tier_indices = np.argmax(utility_matrix, axis=0)  # Shape: (N,)
        best_staking_utilities = np.max(utility_matrix, axis=0)  # Shape: (N,)
        
        # FIX: Only stake if best staking utility exceeds hurdle rate AND is better than not staking
        # This fixes the bug where agents stake even with 0% APY
        should_stake_mask = (
            (best_staking_utilities >= self.delta) &  # Exceeds hurdle rate
            (best_staking_utilities > utility_not_staking) &  # Better than not staking
            has_tokens_mask  # Has tokens to stake
        )
        
        # Calculate demand by tier
        demand = {tier: 0.0 for tier in self.tiers}
        agent_count = {tier: 0 for tier in self.tiers}
        
        for tier_idx, tier in enumerate(self.tiers):
            # Find agents who chose this tier and should stake
            tier_agents_mask = (best_tier_indices == tier_idx) & should_stake_mask
            
            if np.any(tier_agents_mask):
                demand[tier] = np.sum(available_tokens[tier_agents_mask])
                agent_count[tier] = np.sum(tier_agents_mask)
        
        unstaking_agents = self.params.N - np.sum(should_stake_mask)
        
        return {
            'demand': demand,
            'agent_count': agent_count,
            'unstaking_agents': int(unstaking_agents)
        }
    
    def simulate_option1(self, debug=False) -> Dict:
        """Simulate Option 1: Fixed budget with static APYs"""
        if debug:
            print("Simulating Option 1: Fixed Budget")
            if self.params.time_step > 1:
                print(f"Using time step: {self.params.time_step} days")
            if self.params.vectorized:
                print("Using vectorized calculations")
            if self.params.fast_mode:
                print("Fast mode enabled")
        
        # Initialize
        B_current = self.params.B0
        r_daily = (1 + self.params.R_base) ** (1/self.params.E) - 1
        
        # Adjust for time stepping
        if self.params.time_step > 1:
            r_step = (1 + self.params.R_base) ** (self.params.time_step/365) - 1
        else:
            r_step = r_daily
        
        if debug:
            print(f"Daily base rate: {r_daily:.6f}")
            if self.params.time_step > 1:
                print(f"Step rate ({self.params.time_step} days): {r_step:.6f}")
            print(f"Agent hurdle rates (daily): min={self.delta.min():.6f}, max={self.delta.max():.6f}")
            print(f"Illiquidity costs (annual): min={self.lambda_agent.min():.4f}, max={self.lambda_agent.max():.4f}")
        
        # Adjust simulation steps for time stepping
        simulation_steps = self.params.T // self.params.time_step
        
        for step in range(simulation_steps):
            t = step * self.params.time_step  # Actual time in days
            
            # Update agent holdings for this period
            self.update_agent_holdings(t)
            
            # Update circulating supply
            self.C_track[t] = self.calculate_circulating_supply(t)
            
            # Calculate daily yields for each tier
            daily_yields = {}
            for tier in self.tiers:
                multiplier = self.params.multipliers[tier]
                # FIX: APY should be zero if reserve is depleted
                if B_current > 0:
                    daily_yields[tier] = multiplier * r_daily
                else:
                    daily_yields[tier] = 0.0  # No yield when reserve is depleted
                self.apy_track[tier][t] = daily_yields[tier] * 365
            
            if debug and step == 0:
                print("Daily yields by tier:")
                for tier in self.tiers:
                    print(f"  {tier.value}: {daily_yields[tier]:.6f} (APY: {self.apy_track[tier][t]:.2%})")
                print(f"Agent holdings: min={self.agent_holdings_track[:, t].min()/1e3:.1f}K, max={self.agent_holdings_track[:, t].max()/1e6:.1f}M, total={self.agent_holdings_track[:, t].sum()/1e6:.1f}M")
            
            # Agent decision making - USE VECTORIZED VERSION
            if self.params.vectorized:
                decision_results = self.make_agent_decisions_vectorized(daily_yields, t)
                demand = decision_results['demand']
                agent_count = decision_results['agent_count']
                unstaking_agents = decision_results['unstaking_agents']
            else:
                # Fallback to original (slower) method
                demand = {tier: 0.0 for tier in self.tiers}
                agent_count = {tier: 0 for tier in self.tiers}
                unstaking_agents = 0
                
                for i in range(self.params.N):
                    available_tokens = self.agent_holdings_track[i, t]
                    
                    if available_tokens <= 0:
                        continue
                    
                    utilities = {}
                    for tier in self.tiers:
                        utilities[tier] = self.calculate_agent_utility(tier, daily_yields[tier], t)[i]
                    
                    best_tier = max(utilities.keys(), key=lambda x: utilities[x])
                    best_staking_utility = utilities[best_tier]
                    
                    # FIX: Compare against utility of NOT staking (which is 0)
                    utility_not_staking = 0.0
                    
                    # Only stake if staking utility exceeds hurdle rate AND is better than not staking
                    if (best_staking_utility >= self.delta[i] and 
                        best_staking_utility > utility_not_staking):
                        # FIX: Agents only stake a portion of their holdings (keep some liquid)
                        stake_amount = available_tokens * self.params.max_stake_ratio
                        demand[best_tier] += stake_amount
                        agent_count[best_tier] += 1
                    else:
                        unstaking_agents += 1
            
            # Cap demand to available agent holdings (can't stake more than they own!)
            total_demand = sum(demand.values())
            total_agent_holdings = np.sum(self.agent_holdings_track[:, t])
            cap = total_agent_holdings  # Agents can only stake what they actually hold
            
            if debug and step == 0:
                print(f"Total stake demand: {sum(demand.values())/1e6:.1f}M tokens")
                print(f"Agent participation by tier:")
                for tier in self.tiers:
                    print(f"  {tier.value}: {agent_count[tier]} agents, {demand[tier]/1e6:.1f}M tokens")
                print(f"Agents not staking (unfavorable conditions): {unstaking_agents}")
                print(f"Total agent holdings: {total_agent_holdings/1e6:.1f}M tokens")
                print(f"Agent holdings cap: {cap/1e6:.1f}M tokens")
                print(f"Reserve remaining: {B_current/1e6:.1f}M tokens")
            
            if total_demand <= cap:
                for tier in self.tiers:
                    self.S_track[tier][t] = demand[tier]
                if debug and step == 0:
                    print("✓ All demand satisfied (demand <= circulating supply)")
            else:
                scale = cap / total_demand
                for tier in self.tiers:
                    self.S_track[tier][t] = demand[tier] * scale
                if debug and step == 0:
                    print(f"⚠ Demand capped: scale factor = {scale:.3f}")
                    print(f"  Actual stakes after scaling:")
                    for tier in self.tiers:
                        print(f"    {tier.value}: {self.S_track[tier][t]/1e6:.1f}M tokens")
            
            # Calculate total staked and weighted stake
            total_staked = sum(self.S_track[tier][t] for tier in self.tiers)
            weighted_stake = sum(
                self.params.multipliers[tier] * self.S_track[tier][t] 
                for tier in self.tiers
            )
            
            self.total_staked_track[t] = total_staked
            
            # Update reserve (if positive) - scale for time step
            if B_current > 0:
                if self.params.time_step > 1:
                    delta_B = r_step * weighted_stake  # Use step rate for larger time steps
                else:
                    delta_B = r_daily * weighted_stake
                B_current = max(0, B_current - delta_B)
            
            self.B_track_opt1[t] = B_current
            
            # Fill in intermediate days if using time stepping > 1
            if self.params.time_step > 1:
                for fill_day in range(t + 1, min(t + self.params.time_step, self.params.T)):
                    # Interpolate values for intermediate days
                    self.C_track[fill_day] = self.calculate_circulating_supply(fill_day)
                    for tier in self.tiers:
                        self.S_track[tier][fill_day] = self.S_track[tier][t]
                        self.apy_track[tier][fill_day] = self.apy_track[tier][t]
                    self.total_staked_track[fill_day] = total_staked
                    self.B_track_opt1[fill_day] = B_current
            
            # Update price for next period
            if t < self.params.T - 1:
                self.price_track[t+1] = self.calculate_price_impact(t, total_staked)
        
        return {
            'reserve_track': self.B_track_opt1,
            'staked_track': self.S_track,
            'apy_track': self.apy_track,
            'total_staked': self.total_staked_track,
            'price_track': self.price_track[:-1]
        }
    
    def simulate_option2(self, debug=False) -> Dict:
        """Simulate Option 2: Decaying emission with dynamic APYs"""
        if debug:
            print("Simulating Option 2: Dynamic Emissions")
            if self.params.time_step > 1:
                print(f"Using time step: {self.params.time_step} days")
            if self.params.vectorized:
                print("Using vectorized calculations")
            if self.params.fast_mode:
                print("Fast mode enabled")
        
        # Reset tracking for clean comparison
        self.setup_tracking()
        
        for t in range(self.params.T):
            # Update agent holdings for this period
            self.update_agent_holdings(t)
            
            # Update circulating supply
            self.C_track[t] = self.calculate_circulating_supply(t)
            
            # Calculate emission for this period
            emission_t = self.params.E0 * np.exp(-self.params.k * t / 365)
            self.emission_track_opt2[t] = emission_t
            
            # Calculate current weighted stake: baseline + agent staking
            if t == 0:
                # Start with baseline staking (existing stakers not modeled as agents)
                current_weighted_stake = self.params.baseline_staking
            else:
                # Baseline + previous period's agent staking
                current_weighted_stake = self.params.baseline_staking + self.weighted_stake_track[t-1]
            
            # Calculate dynamic APYs for each tier
            daily_yields = {}
            for tier in self.tiers:
                multiplier = self.params.multipliers[tier]
                if current_weighted_stake > 0:
                    base_daily_yield = emission_t / (365 * current_weighted_stake)
                    daily_yields[tier] = multiplier * base_daily_yield
                    
                    # FIX: If emissions become very low (less than 1% of initial), 
                    # set APY to zero to simulate economic unviability
                    if emission_t < (self.params.E0 * 0.01):  # Less than 1% of initial emission
                        daily_yields[tier] = 0.0
                else:
                    daily_yields[tier] = 0
                
                self.apy_track[tier][t] = daily_yields[tier] * 365
            
            # Agent decision making - USE VECTORIZED VERSION LIKE OPTION 1
            if self.params.vectorized:
                decision_results = self.make_agent_decisions_vectorized(daily_yields, t)
                demand = decision_results['demand']
                agent_count = decision_results['agent_count']
                unstaking_agents = decision_results['unstaking_agents']
            else:
                # Fallback to original (slower) method
                demand = {tier: 0.0 for tier in self.tiers}  # Now tracks total stake demand, not agent count
                agent_count = {tier: 0 for tier in self.tiers}  # Track agent count for debugging
                utilities_debug = {tier: [] for tier in self.tiers}
                
                # Track agents who decide to unstake (new feature)
                unstaking_agents = 0
                
                for i in range(self.params.N):
                    available_tokens = self.agent_holdings_track[i, t]
                    
                    if available_tokens <= 0:
                        continue  # Agent has no tokens to stake
                    
                    utilities = {}
                    for tier in self.tiers:
                        utilities[tier] = self.calculate_agent_utility(tier, daily_yields[tier], t)[i]
                        if debug and t == 0 and i < 5:  # Debug first 5 agents on first day
                            utilities_debug[tier].append(utilities[tier])
                    
                    # Choose best tier if utility exceeds hurdle rate
                    best_tier = max(utilities.keys(), key=lambda x: utilities[x])
                    best_utility = utilities[best_tier]
                    
                    # FIX: More dynamic staking behavior
                    # Only stake if utility is positive AND exceeds hurdle rate
                    # This makes agents more responsive to changing conditions
                    if best_utility >= self.delta[i] and best_utility >= 0:
                        # FIX: Stake only a portion of available tokens (keep some liquid)
                        stake_amount = available_tokens * self.params.max_stake_ratio
                        demand[best_tier] += stake_amount
                        agent_count[best_tier] += 1
                    else:
                        # Agent chooses not to stake (keeps tokens liquid)
                        # This creates more realistic, dynamic behavior
                        unstaking_agents += 1
            
            # Cap demand to available agent holdings (can't stake more than they own!)
            total_demand = sum(demand.values())
            total_agent_holdings = np.sum(self.agent_holdings_track[:, t])
            cap = total_agent_holdings  # Agents can only stake what they actually hold
            
            if debug and t == 0:
                print("Daily yields by tier:")
                for tier in self.tiers:
                    print(f"  {tier.value}: {daily_yields[tier]:.6f} (APY: {self.apy_track[tier][t]:.2%})")
                print(f"Agent holdings: min={self.agent_holdings_track[:, t].min()/1e3:.1f}K, max={self.agent_holdings_track[:, t].max()/1e6:.1f}M, total={self.agent_holdings_track[:, t].sum()/1e6:.1f}M")
                
                # Only show utilities debug if not using vectorized mode
                if not self.params.vectorized:
                    print("Sample agent utilities (first 5 agents):")
                    for tier in self.tiers:
                        print(f"  {tier.value}: {utilities_debug[tier]}")
                
                print(f"Total stake demand: {sum(demand.values())/1e6:.1f}M tokens")
                print(f"Agent participation by tier:")
                for tier in self.tiers:
                    print(f"  {tier.value}: {agent_count[tier]} agents, {demand[tier]/1e6:.1f}M tokens")
                print(f"Agents not staking (unfavorable conditions): {unstaking_agents}")
                print(f"Total agent holdings: {total_agent_holdings/1e6:.1f}M tokens")
                print(f"Agent holdings cap: {cap/1e6:.1f}M tokens")
            
            if total_demand <= cap:
                for tier in self.tiers:
                    self.S_track[tier][t] = demand[tier]
                if debug and t == 0:
                    print("✓ All demand satisfied (demand <= circulating supply)")
            else:
                scale = cap / total_demand
                for tier in self.tiers:
                    self.S_track[tier][t] = demand[tier] * scale
                if debug and t == 0:
                    print(f"⚠ Demand capped: scale factor = {scale:.3f}")
                    print(f"  Actual stakes after scaling:")
                    for tier in self.tiers:
                        print(f"    {tier.value}: {self.S_track[tier][t]/1e6:.1f}M tokens")
            
            # Calculate actual weighted stake
            total_staked = sum(self.S_track[tier][t] for tier in self.tiers)
            weighted_stake = sum(
                self.params.multipliers[tier] * self.S_track[tier][t] 
                for tier in self.tiers
            )
            
            self.total_staked_track[t] = total_staked
            self.weighted_stake_track[t] = weighted_stake
            
            # Update price for next period
            if t < self.params.T - 1:
                self.price_track[t+1] = self.calculate_price_impact(t, total_staked)
        
        return {
            'emission_track': self.emission_track_opt2,
            'staked_track': self.S_track,
            'apy_track': self.apy_track,
            'total_staked': self.total_staked_track,
            'weighted_stake': self.weighted_stake_track,
            'price_track': self.price_track[:-1]
        }
    
    def compare_options(self, debug=False) -> Dict:
        """Run both simulations and return comparison metrics"""
        # Run Option 1
        results_opt1 = self.simulate_option1(debug=debug)
        
        # DON'T reset everything - just run Option 2 with the same agent setup
        results_opt2 = self.simulate_option2(debug=debug)
        
        # Calculate comparison metrics
        comparison = {
            'option1': results_opt1,
            'option2': results_opt2,
            'metrics': {
                'avg_participation_opt1': np.mean(results_opt1['total_staked']) / np.mean(self.C_track),
                'avg_participation_opt2': np.mean(results_opt2['total_staked']) / np.mean(self.C_track),
                'apy_volatility_opt1': {tier.value: np.std(results_opt1['apy_track'][tier]) for tier in self.tiers},
                'apy_volatility_opt2': {tier.value: np.std(results_opt2['apy_track'][tier]) for tier in self.tiers},
                'reserve_depletion_day': np.argmax(results_opt1['reserve_track'] <= 0) if any(results_opt1['reserve_track'] <= 0) else None
            }
        }
        
        return comparison
    
    def plot_results(self, results: Dict, title_prefix: str = ""):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 for new plot
        days = np.arange(self.params.T)
        
        # Plot 1: Total Staked Over Time + Circulating Supply
        axes[0, 0].plot(days, results['total_staked'] / 1e6, label='Total Staked', linewidth=2, color='blue')
        axes[0, 0].plot(days, self.C_track / 1e6, label='Circulating Supply', linewidth=2, linestyle='--', alpha=0.8, color='gray')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Tokens (Millions)')
        axes[0, 0].set_title(f'{title_prefix}Total Staked vs Circulating Supply')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: APY by Tier
        for tier in self.tiers:
            axes[0, 1].plot(days, results['apy_track'][tier] * 100, 
                          label=f'{tier.value} ({self.params.multipliers[tier]}x)', linewidth=2)
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('APY (%)')
        axes[0, 1].set_title(f'{title_prefix}APY by Tier')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: NEW - Agent Staking Participation Rate
        # Calculate total agent holdings over time
        total_agent_holdings_over_time = np.sum(self.agent_holdings_track, axis=0)
        
        # Calculate agent participation rate (% of agent holdings that are staked)
        agent_participation_rate = np.zeros(self.params.T)
        for t in range(self.params.T):
            if total_agent_holdings_over_time[t] > 0:
                agent_participation_rate[t] = (results['total_staked'][t] / total_agent_holdings_over_time[t]) * 100
            else:
                agent_participation_rate[t] = 0
        
        axes[0, 2].plot(days, agent_participation_rate, linewidth=2, color='green')
        axes[0, 2].set_xlabel('Days')
        axes[0, 2].set_ylabel('Participation Rate (%)')
        axes[0, 2].set_title(f'{title_prefix}Agent Staking Participation\n(% of Agent Holdings Staked)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add reference lines
        axes[0, 2].axhline(y=100, color='red', linestyle=':', alpha=0.5, label='100% (All Holdings Staked)')
        axes[0, 2].axhline(y=75, color='orange', linestyle=':', alpha=0.5, label='75%')
        axes[0, 2].axhline(y=50, color='blue', linestyle=':', alpha=0.5, label='50%')
        axes[0, 2].legend(fontsize=8)
        
        # Plot 4: Market Participation Rate (% of circulating supply staked)
        market_participation_pct = (results['total_staked'] / self.C_track) * 100
        axes[1, 0].plot(days, market_participation_pct, label='Market Participation %', linewidth=2, color='purple')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Participation (%)')
        axes[1, 0].set_title(f'{title_prefix}Market Staking Participation\n(% of Circulating Supply)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        axes[1, 0].axhline(y=50, color='red', linestyle=':', alpha=0.5, label='50% Reference')
        axes[1, 0].axhline(y=25, color='orange', linestyle=':', alpha=0.5, label='25% Reference')
        axes[1, 0].legend()
        
        # Plot 5: Reserve Depletion
        axes[1, 1].plot(days, results['reserve_track'] / 1e6, 'b-', label='Reserve (M tokens)', linewidth=2)
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Reserve Remaining (M tokens)', color='b')
        axes[1, 1].set_title(f'{title_prefix}Reserve Depletion')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Highlight reserve depletion point if it occurs
        zero_points = np.where(results['reserve_track'] <= 0)[0]
        if len(zero_points) > 0:
            depletion_day = zero_points[0]
            axes[1, 1].axvline(x=depletion_day, color='red', linestyle='--', alpha=0.7, 
                             label=f'Depleted Day {depletion_day}')
            axes[1, 1].legend()
        
        # Plot 6: Agent Holdings vs Staking Over Time
        axes[1, 2].plot(days, total_agent_holdings_over_time / 1e6, 
                       label='Total Agent Holdings', linewidth=2, color='blue', alpha=0.7)
        axes[1, 2].plot(days, results['total_staked'] / 1e6, 
                       label='Total Staked', linewidth=2, color='red')
        
        # Fill area between to show unstaked holdings
        axes[1, 2].fill_between(days, total_agent_holdings_over_time / 1e6, 
                               results['total_staked'] / 1e6, 
                               alpha=0.3, color='lightblue', label='Unstaked Holdings')
        
        axes[1, 2].set_xlabel('Days')
        axes[1, 2].set_ylabel('Tokens (Millions)')
        axes[1, 2].set_title(f'{title_prefix}Agent Holdings: Staked vs Liquid')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def create_comparison_dataframe(self, results: Dict) -> pd.DataFrame:
        """Create a detailed dataframe for analyzing daily simulation results"""
        days = np.arange(self.params.T)
        
        # Calculate agent participation metrics
        total_agent_holdings_over_time = np.sum(self.agent_holdings_track, axis=0)
        agent_participation_rate = np.zeros(self.params.T)
        
        for t in range(self.params.T):
            if total_agent_holdings_over_time[t] > 0:
                agent_participation_rate[t] = (results['total_staked'][t] / total_agent_holdings_over_time[t]) * 100
            else:
                agent_participation_rate[t] = 0
        
        # Build comprehensive dataframe
        data = {
            'day': days,
            'circ_supply': self.C_track,
            
            # Agent holdings and participation data
            'total_agent_holdings': total_agent_holdings_over_time,
            'agent_participation_rate': agent_participation_rate,  # % of agent holdings staked
            'unstaked_agent_holdings': total_agent_holdings_over_time - results['total_staked'],
            
            # Staking data
            'total_staked': results['total_staked'],
            'market_participation': results['total_staked'] / self.C_track * 100,  # % of circulating supply staked
            'reserve': results['reserve_track'],
            
            # APY data by tier
            'liquid_apy': results['apy_track'][StakingTier.LIQUID] * 100,
            '1yr_apy': results['apy_track'][StakingTier.ONE_YEAR] * 100,
            '2yr_apy': results['apy_track'][StakingTier.TWO_YEAR] * 100,
            
            # Staking by tier
            'liquid_staked': results['staked_track'][StakingTier.LIQUID],
            '1yr_staked': results['staked_track'][StakingTier.ONE_YEAR],
            '2yr_staked': results['staked_track'][StakingTier.TWO_YEAR],
        }
        
        df = pd.DataFrame(data)
        
        # Add derived metrics
        df['stake_change'] = df['total_staked'].diff()
        df['agent_participation_change'] = df['agent_participation_rate'].diff()
        df['reserve_burn_rate'] = -df['reserve'].diff()  # Daily reserve consumption
        
        return df

# Example usage
if __name__ == "__main__":
    params = SimulationParams(
        T=365,  # 1 year simulation
        N=1000,  # 1000 agents
        R_base=0.12,  # 12% base APY
        B0=1e8,  # 100M token reserve
        E0=5e6,  # 5M initial annual emission (reduced for smaller supply)
        k=0.5,   # Decay rate
        rho_min=0.02,  # 2% min hurdle
        rho_max=0.15,  # 15% max hurdle
        lambda_max=0.02,  # 2% max illiquidity cost
        C0=50e6,  # 50M initial circulating supply
        daily_vesting=1e6,  # 1M tokens/day vesting
        agent_holdings_ratio=0.4,  # Agents hold 40% of circulating supply
        wealth_concentration=1.5   # Power law wealth distribution
    )
    
    sim = DPoSSimulation(params)
    results = sim.compare_options(debug=True)
    
    print("\nComparison Metrics:")
    print(f"Average Participation - Option 1: {results['metrics']['avg_participation_opt1']:.2%}")
    print(f"Average Participation - Option 2: {results['metrics']['avg_participation_opt2']:.2%}")
    
    if results['metrics']['reserve_depletion_day']:
        print(f"Reserve depleted on day: {results['metrics']['reserve_depletion_day']}")
    else:
        print("Reserve maintained throughout simulation period")
    
    # Create detailed dataframe analysis
    df = sim.create_comparison_dataframe(results)
    
    print("\n" + "="*80)
    print("DETAILED DAILY ANALYSIS (First 30 days)")
    print("="*80)
    
    # Show first 30 days with key columns
    analysis_cols = [
        'day', 'circ_supply', 
        'total_staked', 'market_participation', 'stake_change',
        '2yr_apy', 'reserve'
    ]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print(df[analysis_cols].head(30).to_string(index=False))
    
    print("\n" + "="*80)
    print("UNSTAKING ANALYSIS")
    print("="*80)
    
    print("Neither Option 1 nor Option 2 currently implement proper unstaking mechanics:")
    print("- Line 118 shows: 'available_holdings = prev_holdings  # For now, assume unstaking is instant'")
    print("- Both options assume agents can instantly unstake and restake every day")
    print("- This means there's no lock-up period enforcement despite tier durations being defined")
    print("\nTier durations are only used for illiquidity cost calculations:")
    print("- Liquid: 0 years (no illiquidity cost)")
    print("- 1-Year: 1 year lock-up (illiquidity cost applied)")  
    print("- 2-Year: 2 year lock-up (illiquidity cost applied)")
    
    # print("\n" + "="*80)
    # print("OPTION 2 OSCILLATION ANALYSIS")
    # print("="*80)
    
    # Analyze oscillations in Option 2
    # opt2_changes = df['opt2_stake_change'].dropna()
    # print(f"Option 2 stake changes - Mean: {opt2_changes.mean():.2f}, Std: {opt2_changes.std():.2f}")
    # print(f"Number of positive changes: {(opt2_changes > 0).sum()}")
    # print(f"Number of negative changes: {(opt2_changes < 0).sum()}")
    # print(f"Largest increase: {opt2_changes.max():.2f}")
    # print(f"Largest decrease: {opt2_changes.min():.2f}")
    
    # Plot results
    sim.plot_results(results, "Titan DPoS Simulation: ")