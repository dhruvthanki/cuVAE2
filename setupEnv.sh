#!/bin/bash

# Define the name of your virtual environment
VENV_DIR=".venv"

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install the dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please make sure it exists in the current directory."
    deactivate
    exit 1
fi

# Deactivate the virtual environment
deactivate

echo "Setup complete. To activate the virtual environment, run 'source $VENV_DIR/bin/activate'."
