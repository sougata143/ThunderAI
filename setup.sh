#!/bin/bash

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is required but not installed. Installing..."
    brew install python@3.10
fi

# Remove existing virtual environment if it exists
if [ -d "thunderai-env" ]; then
    echo "Removing existing virtual environment..."
    rm -rf thunderai-env
fi

# Create virtual environment with Python 3.10
echo "Creating new virtual environment..."
python3.10 -m venv thunderai-env

# Activate virtual environment
echo "Activating virtual environment..."
source thunderai-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Verify installations
echo "Verifying installations..."
python -c "import uvicorn; import fastapi; print('FastAPI and Uvicorn installed successfully!')"

echo "ThunderAI setup completed successfully!" 