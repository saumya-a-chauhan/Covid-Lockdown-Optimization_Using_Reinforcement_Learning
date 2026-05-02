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

# Epidemic Parameters (Verity et al. 2020)
BASE_FATALITY_RATE = 0.015  
BREACH_FATALITY_MULT = 3.0  
VIRAL_CYCLE_DAYS = 21

# Continuous Boundaries (Triambak & Mahapatra 2021)
L_MIN = 0.016 # Strict Lockdown
L_MAX = 0.040 # Full Open

# --- 2. CONTINUOUS RCPO AGENT ---
class ContinuousRCPOAgent:
    def __init__(self, state_dim=3):
        self.gamma = 0.99
        self.lr_actor = 0.0005 
        self.lr_critic = 0.001
        self.lr_lambda = 0.005
        
        self.lambda_penalty = 0.0 
        self.lambda_max = 5.0 
        
        # Actor outputs a single value squashed between 0 and 1 via Sigmoid
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Base network outputs [0, 1]
        raw_output = self.actor(state_tensor)
        
        if explore:
            # Add Gaussian noise for exploration, scale down as training progresses
            noise = torch.normal(0, 0.05, size=raw_output.size())
            noisy_output = torch.clamp(raw_output + noise, 0.0, 1.0)
        else:
            noisy_output = raw_output
            
        # Scale the [0, 1] output to the actual mechanical step size [0.016, 0.040]
        step_size = L_MIN + (noisy_output.item() * (L_MAX - L_MIN))
        
        # We return the step_size for the environment, and the raw tensor for backprop
        return step_size, raw_output

    def update_penalty_knob(self, cost):
        penalty_gradient = cost - 0.01 
        self.lambda_penalty += self.lr_lambda * penalty_gradient
        self.lambda_penalty = np.clip(self.lambda_penalty, 0.0, self.lambda_max)


# --- 3. THE CONTINUOUS SIRD ENVIRONMENT ---
class CovidEnvSIRD_Continuous:
    def __init__(self, capacity_limit):
        self.grid_size = 1.0
        self.r_avg = 1.0 / np.sqrt(NUM_AGENTS)
        self.inf_radius = self.r_avg / 5.0
        self.capacity_limit = capacity_limit

    def reset(self):
        self.positions = np.random.rand(NUM_AGENTS, 2)
        # 0: Susceptible, 1: Infected, 2: Recovered, 3: Dead
        self.status = np.zeros(NUM_AGENTS)  
        self.infection_timer = np.zeros(NUM_AGENTS) 
        
        # Patient Zero
        self.status[np.argmin(cdist(self.positions, [[0.5, 0.5]]))] = 1 
        
        self.current_inf = 1
        self.previous_inf = 1
        self.day = 0
        
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.current_inf / NUM_AGENTS, 
            self.previous_inf / NUM_AGENTS,
            self.capacity_limit / NUM_AGENTS
        ])

    def step(self, step_size):
        # Directly use the mechanical step size chosen by the AI
        l = step_size
        
        # Physics Update
        angles = np.random.uniform(0, 2*np.pi, NUM_AGENTS)
        self.positions[:, 0] += l * np.cos(angles)
        self.positions[:, 1] += l * np.sin(angles)
        self.positions = np.abs(self.positions)
        self.positions = np.where(self.positions > self.grid_size, 2*self.grid_size - self.positions, self.positions)
        
        # Transmission
        sus = np.where(self.status == 0)[0]
        inf = np.where(self.status == 1)[0]
        if len(sus) > 0 and len(inf) > 0:
            dists = cdist(self.positions[sus], self.positions[inf])
            newly_infected = sus[np.any(dists <= self.inf_radius, axis=1)]
            self.status[newly_infected] = 1
            
        # Timer Update
        current_inf_idx = np.where(self.status == 1)[0]
        self.infection_timer[current_inf_idx] += 1
        
        self.previous_inf = self.current_inf
        self.current_inf = len(np.where(self.status == 1)[0])
        
        capacity_breached = self.current_inf > self.capacity_limit
        
        # SIRD Resolution (Day 21)
        resolving = np.where((self.status == 1) & (self.infection_timer >= VIRAL_CYCLE_DAYS))[0]
        for idx in resolving:
            fatality_chance = BASE_FATALITY_RATE * BREACH_FATALITY_MULT if capacity_breached else BASE_FATALITY_RATE
            if np.random.rand() < fatality_chance:
                self.status[idx] = 3 # Dead
            else:
                self.status[idx] = 2 # Recovered

        self.day += 1

        # Reward (Economic) is directly proportional to how close the step size is to L_MAX (0.040)
        # If L = 0.040, reward = 1.0. If L = 0.016, reward ~ 0.4.
        reward = (l - L_MIN) / (L_MAX - L_MIN) * 0.6 + 0.4 
        
        # Cost (Safety)
        cost = 1.0 if capacity_breached else 0.0
        done = self.day >= DAYS
        
        info = {"deaths": len(np.where(self.status == 3)[0]), "step_size": l}
        return self._get_state(), reward, cost, done, info


# --- 4. CONTINUOUS TRAINING LOOP (SEPARATED CAPACITIES) ---
def train_separated_capacities():
    # We will train a completely separate AI for each of these capacities
    capacities_to_test = [500, 100, 50, 9] 
    
    for cap in capacities_to_test:
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR HOSPITAL CAPACITY: {cap}")
        print(f"{'='*50}")
        
        # 1. Initialize a FRESH agent for this specific capacity
        agent = ContinuousRCPOAgent()
        training_history = []
        
        for episode in range(1, NUM_EPISODES + 1):
            # The environment is now locked to the current loop's capacity
            env = CovidEnvSIRD_Continuous(capacity_limit=cap)
            state = env.reset()
            
            ep_reward, ep_cost = 0, 0
            
            for day in range(DAYS):
                step_size, raw_actor_output = agent.select_action(state, explore=True)
                next_state, reward, cost, done, info = env.step(step_size)
                
                final_score = reward - (agent.lambda_penalty * cost)
                
                state_tensor = torch.FloatTensor(state)
                next_state_tensor = torch.FloatTensor(next_state)
                
                current_value = agent.critic(state_tensor)
                next_value = agent.critic(next_state_tensor)
                
                target = final_score + (agent.gamma * next_value * (1 - int(done)))
                advantage = target - current_value
                
                actor_loss = -raw_actor_output.squeeze() * advantage.detach()
                critic_loss = advantage.pow(2)
                
                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
                
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()
                
                agent.update_penalty_knob(cost)
                
                ep_reward += reward
                ep_cost += cost
                state = next_state

            # Logging
            if episode % 1 == 0:
                print(f"Cap: {cap:3d} | Ep: {episode:4d} | Econ: {ep_reward:5.1f} | Breaches: {ep_cost:3.0f} | Deaths: {info['deaths']:3d} | Lambda: {agent.lambda_penalty:.3f}")
                
            training_history.append({
                "Capacity": cap, "Episode": episode, "Econ_Reward": ep_reward, 
                "Days_Failed": ep_cost, "Deaths": info['deaths'], "Lambda": agent.lambda_penalty
            })

        # 2. Save the distinct model and logs for THIS capacity before moving to the next
        model_filename = os.path.join(OUTPUT_DIR, f"rcpo_continuous_actor_cap_{cap}.pth")
        log_filename = os.path.join(OUTPUT_DIR, f"train_log_continuous_cap_{cap}.csv")
        
        torch.save(agent.actor.state_dict(), model_filename)
        pd.DataFrame(training_history).to_csv(log_filename, index=False)
        
        print(f">>> Saved Agent for Capacity {cap} to {model_filename}")

if __name__ == "__main__":
    train_separated_capacities()