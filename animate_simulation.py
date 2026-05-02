import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Import your environment and agent from your training file
from train_continuous_sird import CovidEnvSIRD_Continuous, ContinuousRCPOAgent, NUM_AGENTS, DAYS

OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_animation(capacity_to_animate):
    print(f"\n{'='*50}")
    print(f"Generating Simulation Data for Capacity: {capacity_to_animate}...")
    print(f"{'='*50}")
    
    # 1. Load Environment and Trained Agent
    env = CovidEnvSIRD_Continuous(capacity_limit=capacity_to_animate)
    agent = ContinuousRCPOAgent(state_dim=4)
    model_path = os.path.join(OUTPUT_DIR, f"rcpo_continuous_actor_cap_{capacity_to_animate}.pth")
    
    if os.path.exists(model_path):
        agent.actor.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded AI brain for capacity {capacity_to_animate}.")
    else:
        print(f"WARNING: Model {model_path} not found. Agent will act randomly.")

    # 2. Run the simulation and store the history of every frame
    state = env.reset()
    
    history_positions = []
    history_status = []
    history_metrics = [] # To display text on the video
    
    # Fix seed so we get a consistent, visible outbreak across all videos
    np.random.seed(42) 
    
    for day in range(DAYS):
        # Save current frame data
        history_positions.append(env.positions.copy())
        history_status.append(env.status.copy())
        
        # AI decides the lockdown strictness
        step_size, _ = agent.select_action(state, explore=False)
        next_state, reward, cost, done, info = env.step(step_size)
        
        # Save metrics for the video overlay
        history_metrics.append({
            'day': day,
            'active_inf': env.current_inf,
            'step_size': step_size,
            'deaths': info['deaths']
        })
        
        state = next_state

    print(f"Data generated for Capacity {capacity_to_animate}! Rendering animation (this may take a minute)...")

    # 3. Setup the Matplotlib Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_xticks([]) # Hide axes for a cleaner look
    ax.set_yticks([])
    ax.set_title(f"RCPO AI Policy Simulation (Capacity: {capacity_to_animate})", fontsize=14)

    # Colors: 0=Blue (Sus), 1=Red (Inf), 2=Green (Rec), 3=Black (Dead)
    cmap = ListedColormap(['#3498db', '#e74c3c', '#2ecc71', '#2c3e50'])
    
    # Initialize scatter plot
    scatter = ax.scatter(history_positions[0][:, 0], history_positions[0][:, 1], 
                         c=history_status[0], cmap=cmap, s=10, alpha=0.8, vmin=0, vmax=3)

    # Initialize Text overlays
    day_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, fontweight='bold')
    inf_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, color='red')
    mob_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=12, color='purple')
    death_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=12)
    
    # Add a horizontal line representing the "Hospital Capacity" in text
    ax.text(0.65, 0.95, f"Hospital Beds: {capacity_to_animate}", transform=ax.transAxes, fontsize=12, fontweight='bold')

    # 4. Animation Update Function
    def update(frame):
        # Update agent positions and colors
        scatter.set_offsets(history_positions[frame])
        scatter.set_array(history_status[frame])
        
        # Update text overlay
        metrics = history_metrics[frame]
        day_text.set_text(f"Day: {metrics['day']}")
        inf_text.set_text(f"Active Infections: {metrics['active_inf']}")
        
        # Convert step size to a visually understandable 0-100% "Openness" metric
        openness = ((metrics['step_size'] - 0.016) / (0.040 - 0.016)) * 100
        mob_text.set_text(f"Economy Openness: {openness:.1f}%")
        
        death_text.set_text(f"Total Deaths: {metrics['deaths']}")
        
        return scatter, day_text, inf_text, mob_text, death_text

    # 5. Create and Save the Animation
    ani = animation.FuncAnimation(fig, update, frames=DAYS, interval=100, blit=True)
    
    # Save as GIF (Requires Pillow, which is usually pre-installed with Matplotlib)
    output_path = os.path.join(OUTPUT_DIR, f"simulation_cap_{capacity_to_animate}.gif")
    ani.save(output_path, writer='pillow', fps=10)
    
    print(f">>> Success! Animation saved to: {output_path}")
    
    # Close the figure to free up memory before the next loop
    plt.close(fig)

if __name__ == "__main__":
    # Loop through all capacities automatically!
    capacities_to_test = [500, 100, 50, 9]
    
    for cap in capacities_to_test:
        generate_animation(cap)
        
    print("\nAll animations generated successfully!")