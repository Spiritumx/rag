#!/bin/bash

# Setup script for evaluate_v2 utils
# Creates symlinks to baseline utilities that can be shared

echo "=================================================="
echo "Setting up Evaluate V2 Utility Symlinks"
echo "=================================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# List of utilities to symlink from baseline
SHARED_UTILS=(
    "classifier_loader.py"
    "config_loader.py"
    "data_loader.py"
)

echo ""
echo "[INFO] Creating symlinks to baseline utilities..."
echo ""

for util in "${SHARED_UTILS[@]}"; do
    if [ -L "$util" ]; then
        echo "  ✓ $util (already exists)"
    elif [ -f "$util" ]; then
        echo "  ! $util (file exists, not a symlink - skipping)"
    else
        ln -s "../../../evaluate/utils/$util" "$util"
        if [ $? -eq 0 ]; then
            echo "  ✓ $util (symlink created)"
        else
            echo "  ✗ $util (failed to create symlink)"
        fi
    fi
done

echo ""
echo "=================================================="
echo "Symlink Setup Complete"
echo "=================================================="
echo ""
echo "Symlinked utilities (shared with baseline):"
echo "  - classifier_loader.py: Qwen classifier inference"
echo "  - config_loader.py: YAML config loading"
echo "  - data_loader.py: Test dataset loading"
echo ""
echo "V2-specific utilities (not symlinked):"
echo "  - confidence_verifier.py: Answer confidence scoring"
echo "  - routing_logger.py: Cascade decision tracking"
echo "  - result_manager.py: Uses outputs_v2/ directories"
echo "  - service_checker.py: Checks port 8002"
echo ""
