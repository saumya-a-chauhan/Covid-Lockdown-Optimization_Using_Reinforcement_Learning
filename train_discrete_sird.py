import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from scipy.spatial.distance import cdist

# --- 1. RESEARCH-BACKED CONSTANTS ---
NUM_EPISODES = 5000
DAYS = 200
NUM_AGENTS = 2500
OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Epidemic Parameters
BASE_FATALITY_RATE = 0.015  
BREACH_FATALITY_MULT = 3.0  
VIRAL_CYCLE_DAYS = 21

# The 3 Discrete Step Sizes (Calibrated to WHO Data)
L_MIN = 0.016 # Strict Lockdown
L_MID = 0.028 # Partial Open
L_MAX = 0.040 # Full Open

# --- 2. DISCRETE RCPO AGENT ---
class DiscreteRCPOAgent:
    def __init__(self, state_dim=4): 
        self.gamma = 0.99
        self.lr_actor = 0.0005 
        self.lr_critic = 0.001
        self.lr_lambda = 0.005
        
        self.lambda_penalty = 0.0 
        self.lambda_max = 5.0 
        
        # ACTOR: Outputs 3 distinct probabilities using Softmax
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Softmax(dim=-1) 
        )
        
        # CRITIC: Outputs 1 continuous Value estimate
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor)
        
        # Categorical distribution for weighted random sampling
        dist = torch.distributions.Categorical(probs)
        
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=1)
            
        return action.item(), dist.log_prob(action)

    def update_penalty_knob(self, is_breached):
        # Target-Based Asymmetric Gradient
        target_breach_rate = 0.10 
        penalty_gradient = is_breached - target_breach_rate 
        self.lambda_penalty += (self.lr_lambda * 0.2) * penalty_gradient
        self.lambda_penalty = np.clip(self.lambda_penalty, 0.0, self.lambda_max)

# --- 3. DISCRETE SIRD ENVIRONMENT ---
class CovidEnvSIRD_Discrete:
    def __init__(self, capacity_limit):
        self.grid_size = 1.0
        self.r_avg = 1.0 / np.sqrt(NUM_AGENTS)
        self.inf_radius = self.r_avg / 5.0
        self.capacity_limit = capacity_limit

    def reset(self):
        self.positions = np.random.rand(NUM_AGENTS, 2)
        self.status = np.zeros(NUM_AGENTS)  
        self.infection_timer = np.zeros(NUM_AGENTS) 
        
        # Patient Zero
        self.status[np.argmin(cdist(self.positions, [[0.5, 0.5]]))] = 1 
        
        self.current_inf = 1
        self.previous_inf = 1
        self.day = 0
        return self._get_state()

    def _get_state(self):
        inf_idx = np.where(self.status == 1)[0]
        if len(inf_idx) > 1:
            inf_pos = self.positions[inf_idx]
            spatial_spread = np.std(inf_pos[:, 0]) + np.std(inf_pos[:, 1])
        else:
            spatial_spread = 0.0

        return np.array([
            self.current_inf / NUM_AGENTS, 
            self.previous_inf / NUM_AGENTS,
            self.capacity_limit / NUM_AGENTS,
            spatial_spread 
        ])

    def step(self, action):
        # MAP INTEGER ACTION TO PHYSICAL STEP SIZE
        if action == 2:
            l = L_MAX
        elif action == 1:
            l = L_MID
        else:
            l = L_MIN
        
        # PHYSICS
        angles = np.random.uniform(0, 2*np.pi, NUM_AGENTS)
        self.positions[:, 0] += l * np.cos(angles)
        self.positions[:, 1] += l * np.sin(angles)
        self.positions = np.abs(self.positions)
        self.positions = np.where(self.positions > self.grid_size, 2*self.grid_size - self.positions, self.positions)
        
        # TRANSMISSION
        sus = np.where(self.status == 0)[0]
        inf = np.where(self.status == 1)[0]
        if len(sus) > 0 and len(inf) > 0:
            dists = cdist(self.positions[sus], self.positions[inf])
            newly_infected = sus[np.any(dists <= self.inf_radius, axis=1)]
            self.status[newly_infected] = 1
            
        # TIMER UPDATE
        current_inf_idx = np.where(self.status == 1)[0]
        self.infection_timer[current_inf_idx] += 1
        
        self.previous_inf = self.current_inf
        self.current_inf = len(np.where(self.status == 1)[0])
        capacity_breached = self.current_inf > self.capacity_limit
        
        # RESOLUTION (DAY 21) & DYNAMIC MORTALITY
        resolving = np.where((self.status == 1) & (self.infection_timer >= VIRAL_CYCLE_DAYS))[0]
        for idx in resolving:
            fatality_chance = BASE_FATALITY_RATE * BREACH_FATALITY_MULT if capacity_breached else BASE_FATALITY_RATE
            if np.random.rand() < fatality_chance:
                self.status[idx] = 3 # Dead
            else:
                self.status[idx] = 2 # Recovered

        self.day += 1

        # REWARD (Continuous Math applied to Discrete Choice)
        reward = (l - L_MIN) / (L_MAX - L_MIN) * 0.6 + 0.4 
        
        # COST (Action-Scaled Penalty)
        action_penalty_multiplier = (l / L_MAX)
        cost = action_penalty_multiplier if capacity_breached else 0.0
        
        is_breached_binary = 1.0 if capacity_breached else 0.0
        done = self.day >= DAYS
        
        info = {"deaths": len(np.where(self.status == 3)[0]), "step_size": l, "is_breached": is_breached_binary}
        return self._get_state(), reward, cost, done, info

# --- 4. TRAINING LOOP ---
def train_discrete_capacities():
    capacities_to_test = [500, 100, 50, 9] 
    
    for cap in capacities_to_test:
        print(f"\n{'='*50}\nSTARTING DISCRETE TRAINING FOR CAPACITY: {cap}\n{'='*50}")
        
        agent = DiscreteRCPOAgent(state_dim=4)
        training_history = []
        
        for episode in range(1, NUM_EPISODES + 1):
            env = CovidEnvSIRD_Discrete(capacity_limit=cap)
            state = env.reset()
            ep_reward, ep_cost = 0, 0
            
            for day in range(DAYS):
                action, log_prob = agent.select_action(state, explore=True)
                next_state, reward, cost, done, info = env.step(action)
                
                # Advantage Math
                final_score = reward - (agent.lambda_penalty * cost)
                
                state_tensor = torch.FloatTensor(state)
                next_state_tensor = torch.FloatTensor(next_state)
                
                current_value = agent.critic(state_tensor)
                next_value = agent.critic(next_state_tensor)
                target = final_score + (agent.gamma * next_value * (1 - int(done)))
                advantage = target - current_value
                
                # Discrete Actor Loss uses log_prob
                actor_loss = -log_prob.squeeze() * advantage.detach()
                critic_loss = advantage.pow(2)
                
                # Backprop
                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
                
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()
                
                # Update Penalty Knob using strict binary breach indicator
                agent.update_penalty_knob(info['is_breached'])
                
                ep_reward += reward
                ep_cost += info['is_breached'] 
                state = next_state

            # Logging
            if episode % 1 == 0:
                print(f"Cap: {cap:3d} | Ep: {episode:4d} | Econ: {ep_reward:5.1f} | Breaches: {ep_cost:3.0f} | Deaths: {info['deaths']:3d} | Lambda: {agent.lambda_penalty:.3f}")
                
            training_history.append({
                "Capacity": cap, "Episode": episode, "Econ_Reward": ep_reward, 
                "Days_Failed": ep_cost, "Deaths": info['deaths'], "Lambda": agent.lambda_penalty
            })

        # Save Model and Logs
        model_filename = os.path.join(OUTPUT_DIR, f"rcpo_discrete_actor_cap_{cap}.pth")
        log_filename = os.path.join(OUTPUT_DIR, f"train_log_discrete_cap_{cap}.csv")
        torch.save(agent.actor.state_dict(), model_filename)
        pd.DataFrame(training_history).to_csv(log_filename, index=False)
        print(f">>> Saved Discrete Agent for Capacity {cap} to {model_filename}")

if __name__ == "__main__":
    train_discrete_capacities()