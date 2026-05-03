import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

from train_continuous_sird import CovidEnvSIRD_Continuous, ContinuousRCPOAgent, L_MIN, L_MAX

OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVAL_LOG = os.path.join(OUTPUT_DIR, "master_evaluation_results.csv")

DAYS = 200
# Evaluate over 5 different seeds to prevent spatial anomalies
NUM_EVAL_SEEDS = 5 

def get_historical_step_size(day):
    if day < 54: return L_MAX 
    elif day < 122: return L_MIN 
    else: return 0.028 

def run_evaluation_simulation(policy_name, env, agent=None):
    state = env.reset()
    history_inf = []
    history_mob = []
    
    for day in range(DAYS):
        if policy_name == "RCPO_AI":
            step_size, _ = agent.select_action(state, explore=False)
        elif policy_name == "India_Historical":
            step_size = get_historical_step_size(day)
        elif policy_name == "Always_Open":
            step_size = L_MAX
        elif policy_name == "Always_Lockdown":
            step_size = L_MIN
        elif policy_name == "Always_Partial":
            step_size = 0.028
        elif policy_name == "Manual_Reactive":
            step_size = L_MIN if env.current_inf > (env.capacity_limit * 0.8) else L_MAX
            
        next_state, reward, cost, done, info = env.step(step_size)
        
        history_inf.append(env.current_inf)
        history_mob.append(step_size)
        state = next_state
        
    return np.array(history_inf), np.array(history_mob), info['deaths']

def evaluate_all():
    capacities_to_test = [500, 100, 50, 9]
    policies = ["RCPO_AI", "India_Historical", "Always_Open", "Always_Lockdown", "Always_Partial", "Manual_Reactive"]
    all_results = []
    
    for cap in capacities_to_test:
        print(f"\n{'='*50}\nEVALUATING HOSPITAL CAPACITY: {cap}\n{'='*50}")
        env = CovidEnvSIRD_Continuous(capacity_limit=cap)
        agent = ContinuousRCPOAgent(state_dim=4)
        model_path = os.path.join("models", f"rcpo_continuous_actor_cap_{cap}.pth")
        
        if os.path.exists(model_path):
            agent.actor.load_state_dict(torch.load(model_path))
        else:
            print(f"WARNING: Model {model_path} not found.")

        curves = {}

        for p in policies:
            total_mob, total_deaths, total_breaches = 0, 0, 0
            avg_inf_curve = np.zeros(DAYS)
            
            # Loop through multiple seeds to get the true average performance
            for seed_offset in range(NUM_EVAL_SEEDS):
                np.random.seed(42 + seed_offset) 
                
                inf, mob, run_deaths = run_evaluation_simulation(p, env, agent)
                avg_inf_curve += inf
                total_mob += np.sum(mob)
                total_deaths += run_deaths
                total_breaches += int(np.sum(inf > cap))
                
            # Compute averages
            avg_inf_curve = avg_inf_curve / NUM_EVAL_SEEDS
            curves[p] = avg_inf_curve
            
            avg_mob = round(total_mob / NUM_EVAL_SEEDS, 2)
            avg_breach = int(total_breaches / NUM_EVAL_SEEDS)
            avg_d = int(total_deaths / NUM_EVAL_SEEDS)
            
            print(f"[{p:18s}] Avg Mobility: {avg_mob:5.2f} | Avg Breaches: {avg_breach:3d} | Avg Deaths: {avg_d:3d}")
            
            all_results.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Capacity": cap, "Policy": p,
                "Total_Mobility": avg_mob, "Days_Breached": avg_breach, "Total_Deaths": avg_d
            })
            
        # GRAPH GENERATION
        plt.figure(figsize=(12, 6))
        plt.title(f"Avg Infection Curves vs Hospital Capacity ({cap} Beds) - Over {NUM_EVAL_SEEDS} Seeds")
        colors = {"RCPO_AI": "purple", "India_Historical": "orange", "Always_Open": "red", "Always_Lockdown": "blue", "Always_Partial": "cyan", "Manual_Reactive": "green"}
        
        for name, inf_data in curves.items():
            lw = 2.5 if name == "RCPO_AI" else 1.0
            alpha = 1.0 if name == "RCPO_AI" else 0.6
            plt.plot(inf_data, label=name, color=colors[name], linewidth=lw, alpha=alpha)

        plt.axhline(y=cap, color="black", linestyle="--", linewidth=2, label=f"Capacity Limit ({cap})")
        plt.xlabel("Days")
        plt.ylabel("Active Infections")
        plt.ylim(0, min(2500, max(np.max(list(curves.values())) * 1.1, cap * 2))) 
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_filename = os.path.join(OUTPUT_DIR, f"eval_graph_cap_{cap}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(all_results)
    if not os.path.isfile(EVAL_LOG): df.to_csv(EVAL_LOG, index=False)
    else: df.to_csv(EVAL_LOG, mode='a', header=False, index=False)

if __name__ == "__main__":
    evaluate_all()