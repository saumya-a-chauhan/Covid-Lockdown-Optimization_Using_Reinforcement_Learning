import numpy as np
from scipy.spatial.distance import cdist

NUM_AGENTS = 2500
DAYS = 200

class CovidEnvSIR:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.grid_size = 1.0
        self.r_avg = 1.0 / np.sqrt(NUM_AGENTS)
        self.inf_radius = self.r_avg / 2.0  
        
        # 3-Tier Action Space: Modulating Monte Carlo particle step sizes
        self.l_lockdown = 0.016
        self.l_partial = 0.028
        self.l_open = 0.040
        
        # ANTI-EXPLOIT: Guarantee an outbreak begins so the AI cannot gamble
        self.initial_infected_count = 1
        
        # Literature-Backed Case Fatality Rates (CFR)
        self.cfr_base = 0.014   # Verity et al., 2020 (Baseline)
        self.cfr_breach = 0.058 # Ji et al., 2020 (Healthcare collapse)

    def reset(self):
        self.positions = np.random.rand(NUM_AGENTS, 2)
        
        # Status Map -> 0: Susceptible, 1: Infected, 2: Recovered, 3: Dead
        self.status = np.zeros(NUM_AGENTS)
        self.timer = np.zeros(NUM_AGENTS)
        
        dists = cdist(self.positions, [[0.5, 0.5]])
        idx = np.argsort(dists[:, 0])[:self.initial_infected_count]
        self.status[idx] = 1
        
        self.current_inf = self.initial_infected_count
        self.prev_inf = self.initial_infected_count
        self.day = 0
        return self.current_inf, self.prev_inf

    def step(self, action):
        # Base Economic Reward for Mobility
        if action == 2:
            l, base_reward = self.l_open, 1.0
        elif action == 1:
            l, base_reward = self.l_partial, 0.7
        else:
            l, base_reward = self.l_lockdown, 0.4
            
        # Movement: Only S, I, and R move. Dead (3) are removed from mobility.
        angles = np.random.uniform(0, 2*np.pi, NUM_AGENTS)
        alive = np.where(self.status != 3)[0]
        
        self.positions[alive, 0] += l * np.cos(angles[alive])
        self.positions[alive, 1] += l * np.sin(angles[alive])
        self.positions = np.abs(self.positions)
        self.positions = np.where(self.positions > self.grid_size, 2*self.grid_size - self.positions, self.positions)
        
        # Infection Update
        sus = np.where(self.status == 0)[0]
        inf = np.where(self.status == 1)[0]
        if len(sus) > 0 and len(inf) > 0:
            d = cdist(self.positions[sus], self.positions[inf])
            new_inf = sus[np.any(d <= self.inf_radius, axis=1)]
            self.status[new_inf] = 1
            
        # Capacity check for Dynamic Mortality
        self.current_inf = len(np.where(self.status == 1)[0])
        is_breached = self.current_inf > self.capacity
        current_cfr = self.cfr_breach if is_breached else self.cfr_base
            
        # Viral Resolution (Day 21) & Mortality Roll
        curr_inf_idx = np.where(self.status == 1)[0]
        self.timer[curr_inf_idx] += 1
        resolving = np.where((self.status == 1) & (self.timer > 21))[0]
        
        new_deaths = 0 # Track deaths on this specific day
        if len(resolving) > 0:
            death_rolls = np.random.rand(len(resolving))
            
            died = resolving[death_rolls < current_cfr]
            recovered = resolving[death_rolls >= current_cfr]
            
            self.status[died] = 3
            self.status[recovered] = 2
            
            new_deaths = len(died)
        
        self.prev_inf = self.current_inf
        self.current_inf = len(np.where(self.status == 1)[0])
        self.day += 1
        
        # The Pareto Math: Death heavily penalizes the economic score
        cost = 1.0 if is_breached else 0.0
        final_reward = base_reward - (10.0 * new_deaths)
        done = self.day >= DAYS
        
        # Return new_deaths as the 5th variable for logging
        return (self.current_inf, self.prev_inf), final_reward, cost, done, new_deaths