#!/bin/bash

# Exit immediately if any command fails
set -e 

echo "=================================================="
echo "1. System Setup & OS Dependencies"
echo "=================================================="
# In a raw ubuntu:22.04 docker, we need to ensure python3, pip, and venv exist.
# We use '|| sudo ...' just in case the evaluator runs it locally instead of in Docker.
apt-get update -y || sudo apt-get update -y
apt-get install -y python3 python3-pip python3-venv || sudo apt-get install -y python3 python3-pip python3-venv

echo "=================================================="
echo "2. Creating and Activating Virtual Environment"
echo "=================================================="
# Create the virtual environment in the current relative directory
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

echo "=================================================="
echo "3. Installing Python Dependencies"
echo "=================================================="
pip install --upgrade pip
pip install -r requirements.txt

echo "=================================================="
echo "4. Executing Pipeline: Phase 1 (Training)"
echo "=================================================="
# This will train the models and save them to ./project_outputs/
python3 train_continuous_sird.py

echo "=================================================="
echo "5. Executing Pipeline: Phase 2 (Evaluation)"
echo "=================================================="
# This runs the multi-seed evaluation against baselines and saves the CSV and PNG graphs
python3 evaluate_baselines.py

echo "=================================================="
echo "6. Executing Pipeline: Phase 3 (Media Generation)"
echo "=================================================="
# Generates the simulation GIFs
python3 animate_simulation.py
# Generates the side-by-side presentation GIF
python3 animate_comparison.py

echo "=================================================="
echo "SUCCESS! Pipeline executed completely without errors."
echo "All artifacts are saved in the 'project_outputs' directory."
echo "=================================================="