#!/bin/bash

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is required but not installed. Installing..."
    brew install python@3.11
fi

# Create virtual environment with Python 3.11
python3.11 -m venv thunderai-env

# Activate virtual environment
source thunderai-env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install the package in development mode
pip install -e .

echo "ThunderAI setup completed successfully!" 