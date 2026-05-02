
```markdown
# RL COVID Spread Mitigation: Constrained Policy Optimization

This project applies a Continuous Reward Constrained Policy Optimization (RCPO) algorithm to a spatial SIRD epidemic model. The agent's goal is to dynamically balance economic mobility against strict hospital capacity constraints, seeking a Pareto-optimal policy that outperforms static human baselines.

## Features & Contributions
- **Custom Spatial SIRD Environment:** A continuous 2D environment tracking 2,500 agents.
- **Constrained RL (RCPO):** Implements an Advantage Actor-Critic architecture with an asymmetric, action-scaled Lagrangian multiplier.
- **Robust Evaluation:** Multi-seed evaluation pipeline to mathematically eliminate spatial variance.
- **Media Generation:** Automated visualization of agent policies dynamically altering movement speeds.

## Project Structure
- `train_continuous_sird.py`: Trains the RCPO agent across 4 hospital capacities (500, 100, 50, 9).
- `evaluate_baselines.py`: Evaluates the trained AI against 5 baselines across 5 random seeds.
- `animate_simulation.py` & `animate_comparison.py`: Generates GIF visualizations of the physical agent movements.
- `run.sh`: Automated bash script to execute the entire pipeline.

## How to Run (Evaluation Setup)
This repository is configured to run automatically in a fresh `ubuntu:22.04` environment. No manual dependency installation is required.

1. Clone the repository:
   ```bash
   git clone <your-github-repo-url>
   cd RL_COVID_SPREAD_PROJECT
   ```

2. Execute the bash script:
   
```bash
   bash run.sh
   ```

### What the script does automatically:
1. Installs `python3`, `python3-pip`, and `python3-venv` via `apt-get`.
2. Creates and activates a virtual environment (`venv`).
3. Installs dependencies from `requirements.txt`.
4. Executes the full pipeline (Training -> Evaluation -> Media Generation).
5. Saves all models, `.csv` logs, `.png` graphs, and `.gif` animations into a newly created `project_outputs/` directory using strictly relative paths.
```
