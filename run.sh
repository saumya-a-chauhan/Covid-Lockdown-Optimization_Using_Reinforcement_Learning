#!/bin/bash

# Exit immediately if any command fails
set -e 

# Prevents apt-get from freezing the script to ask for timezone/keyboard input
export DEBIAN_FRONTEND=noninteractive

echo "=================================================="
echo "1. System Setup & OS Dependencies"
echo "=================================================="

# Safety check: Use sudo only if it is installed (local machine). 
# If it's missing (Docker), run commands directly.
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

echo "Updating system..."
$SUDO apt-get update -y -o Acquire::Retries=3 -o Acquire::ForceIPv4=true
$SUDO apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential

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
pip install --no-cache-dir -r requirements.txt

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
