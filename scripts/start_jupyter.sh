#!/bin/bash

# move to script's directory
cd "$(dirname "$0")/.."

# check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not on your PATH."
    echo "Install Miniconda or Anaconda to continue: https://docs.conda.io/"
    exit 1
fi

# detect shell and initialize conda
shell_name=$(basename "$SHELL")
if eval "$(conda shell.${shell_name} hook)" 2>/dev/null; then
    echo "Conda initialized for $shell_name"
elif eval "$(conda shell.posix hook)" 2>/dev/null; then
    echo "Conda initialized using POSIX fallback"
else
    echo "Failed to initialize conda"
    echo "Please run 'conda init' and restart your terminal"
    exit 1
fi

ENV_NAME="roi-analysis"
ENV_FILE="environment.yml"

# check if the environment exists
if ! conda info --envs | awk '{print $1}' | grep -q "^$ENV_NAME$"; then
    echo "Environment '$ENV_NAME' not found. Creating it from $ENV_FILE..."
    if [ -f "$ENV_FILE" ]; then
        conda env create -f "$ENV_FILE"
    else
        echo "Error: $ENV_FILE not found in current directory."
        exit 1
    fi
else
    echo "Environment '$ENV_NAME' already exists."
fi

# activate the environment
conda activate "$ENV_NAME"

# start Jupyter Notebook
jupyter notebook
