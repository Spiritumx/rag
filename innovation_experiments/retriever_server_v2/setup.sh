#!/bin/bash

# Setup script for retriever_server_v2
# Creates symlink to baseline retriever models

echo "=================================================="
echo "Setting up Innovation V2 Retriever Server"
echo "=================================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create symlink to baseline models
if [ -L "model" ]; then
    echo "[INFO] Model symlink already exists"
elif [ -d "model" ]; then
    echo "[WARNING] 'model' directory exists but is not a symlink. Skipping..."
else
    echo "[INFO] Creating symlink to baseline models..."
    ln -s ../../retriever_server/model model
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Model symlink created successfully"
    else
        echo "[ERROR] Failed to create model symlink"
        exit 1
    fi
fi

# Verify baseline models exist
BASELINE_MODELS="../../retriever_server/model"
if [ ! -d "$BASELINE_MODELS" ]; then
    echo "[WARNING] Baseline models not found at $BASELINE_MODELS"
    echo "          Please ensure baseline retriever models are downloaded first"
fi

# Install additional dependencies for QueryAnalyzer
echo ""
echo "[INFO] Checking Python dependencies..."
python -c "import spacy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing spaCy..."
    pip install spacy
fi

python -m spacy info en_core_web_sm 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Downloading spaCy English model (en_core_web_sm)..."
    python -m spacy download en_core_web_sm
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To start the V2 retriever server:"
echo "  python serve_v2.py"
echo ""
echo "Server will run on port 8002 by default"
echo "  (set RETRIEVER_PORT environment variable to change)"
echo ""
