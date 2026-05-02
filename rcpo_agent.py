import torch
import torch.nn as nn
import torch.optim as optim

class RCPOAgent:
    def __init__(self):
        self.gamma = 0.99
        self.lr_actor = 0.001
        self.lr_critic = 0.005
        self.lr_lambda = 0.01
        
        self.lambda_penalty = 0.0 
        self.lambda_max = 10.0 # Mathematical firewall prevents gradient shock
        
        # Tanh activation bounds signals to prevent exploding gradients
        self.actor = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Replaces raw .pow(2) to handle extreme advantage values gracefully
        self.critic_loss_fn = nn.SmoothL1Loss()

    def select_action(self, current_inf, prev_inf, total_pop=2500, deterministic=False):
        state = torch.FloatTensor([current_inf / total_pop, prev_inf / total_pop])
        probs = self.actor(state)
        
        if deterministic:
            action = torch.argmax(probs).item()
            return action, None, None 
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), dist.entropy()

    def update_penalty(self, cost):
        penalty_grad = cost - 0.05
        # Cap lambda between 0 and lambda_max
        self.lambda_penalty = max(0.0, min(self.lambda_penalty + (self.lr_lambda * penalty_grad), self.lambda_max))