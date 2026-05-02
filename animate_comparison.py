import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Import your environment and agent from your training file
from train_continuous_sird import CovidEnvSIRD_Continuous, ContinuousRCPOAgent, NUM_AGENTS, DAYS, L_MIN, L_MAX

OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select the capacity to showcase (9 or 50 are usually the most visually impressive for the AI)
CAPACITY_TO_ANIMATE = 9

def generate_comparative_animation():
    print(f"Generating Side-by-Side Simulation for Capacity: {CAPACITY_TO_ANIMATE}...")
    
    # 1. Initialize THREE distinct environments
    env_ai = CovidEnvSIRD_Continuous(capacity_limit=CAPACITY_TO_ANIMATE)
    env_open = CovidEnvSIRD_Continuous(capacity_limit=CAPACITY_TO_ANIMATE)
    env_lock = CovidEnvSIRD_Continuous(capacity_limit=CAPACITY_TO_ANIMATE)
    
    # 2. Load the trained AI
    agent = ContinuousRCPOAgent(state_dim=4)
    model_path = os.path.join(OUTPUT_DIR, f"rcpo_continuous_actor_cap_{CAPACITY_TO_ANIMATE}.pth")
    if os.path.exists(model_path):
        agent.actor.load_state_dict(torch.load(model_path))
    else:
        print(f"WARNING: Model {model_path} not found. Train it first!")
        return

    # 3. Synchronize Seeds (CRITICAL for fair comparison)
    seed = 42
    np.random.seed(seed)
    state_ai = env_ai.reset()
    np.random.seed(seed)
    state_open = env_open.reset()
    np.random.seed(seed)
    state_lock = env_lock.reset()
    
    history = {'ai': [], 'open': [], 'lock': []}
    
    print("Simulating 200 days...")
    for day in range(DAYS):
        # AI Actions
        step_ai, _ = agent.select_action(state_ai, explore=False)
        next_state_ai, _, _, _, info_ai = env_ai.step(step_ai)
        
        # Baseline Actions
        _, _, _, _, info_open = env_open.step(L_MAX) # Always Open
        _, _, _, _, info_lock = env_lock.step(L_MIN) # Always Lockdown
        
        # Save Frames
        history['ai'].append((env_ai.positions.copy(), env_ai.status.copy(), step_ai, info_ai['deaths'], env_ai.current_inf))
        history['open'].append((env_open.positions.copy(), env_open.status.copy(), L_MAX, info_open['deaths'], env_open.current_inf))
        history['lock'].append((env_lock.positions.copy(), env_lock.status.copy(), L_MIN, info_lock['deaths'], env_lock.current_inf))
        
        state_ai = next_state_ai

    print("Rendering 3-Panel Animation (This will take a few minutes)...")

    # 4. Setup Matplotlib Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Pandemic Mitigation Strategies (Capacity: {CAPACITY_TO_ANIMATE} Beds)", fontsize=18, fontweight='bold')
    
    titles = ["Always Lockdown", "RCPO AI (Dynamic)", "Always Open"]
    keys = ['lock', 'ai', 'open']
    scatters = []
    texts = []
    
    cmap = ListedColormap(['#3498db', '#e74c3c', '#2ecc71', '#2c3e50']) # Sus, Inf, Rec, Dead

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize=14)
        
        # Initial scatter
        pos, stat, _, _, _ = history[keys[i]][0]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=stat, cmap=cmap, s=8, alpha=0.8, vmin=0, vmax=3)
        scatters.append(sc)
        
        # Text overlays
        txt_day = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11, fontweight='bold')
        txt_inf = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=11, color='red')
        txt_mob = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=11, color='purple')
        txt_dth = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=11)
        texts.append((txt_day, txt_inf, txt_mob, txt_dth))

    # 5. Animation Loop
    def update(frame):
        updated_artists = []
        for i, key in enumerate(keys):
            pos, stat, step, deaths, inf = history[key][frame]
            
            scatters[i].set_offsets(pos)
            scatters[i].set_array(stat)
            
            openness = ((step - L_MIN) / (L_MAX - L_MIN)) * 100
            
            texts[i][0].set_text(f"Day: {frame}")
            texts[i][1].set_text(f"Active Infections: {inf}")
            texts[i][2].set_text(f"Economy Openness: {openness:.1f}%")
            texts[i][3].set_text(f"Total Deaths: {deaths}")
            
            updated_artists.extend([scatters[i], texts[i][0], texts[i][1], texts[i][2], texts[i][3]])
            
        return updated_artists

    ani = animation.FuncAnimation(fig, update, frames=DAYS, interval=100, blit=True)
    
    # Save as GIF
    output_path = os.path.join(OUTPUT_DIR, f"presentation_comparison_cap_{CAPACITY_TO_ANIMATE}.gif")
    ani.save(output_path, writer='pillow', fps=10)
    
    print(f">>> Success! Presentation GIF saved to: {output_path}")

if __name__ == "__main__":
    generate_comparative_animation()