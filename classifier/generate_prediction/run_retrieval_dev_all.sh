#!/usr/bin/env bash

# This script iterates over all supported datasets and invokes
# run_retrieval_dev.sh for each one to generate predictions.

set -euo pipefail

systems=("ircot_qa" "oner_qa" "nor_qa")
valid_models=("flan-t5-xxl" "flan-t5-xl" "gpt" "none")
datasets=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

usage() {
    echo "Usage: $0 MODEL LLM_PORT_NUM"
    echo "  MODEL        : ${valid_models[*]}"
    echo "  LLM_PORT_NUM : Port of the LLM service (e.g., 8010)"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

MODEL="$1"
LLM_PORT="$2"

# Validation helpers (reuse logic from run_retrieval_dev.sh expectations)
check_in_list() {
    local value="$1"
    shift
    local list=("$@")
    for item in "${list[@]}"; do
        if [[ "$item" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

if ! check_in_list "$MODEL" "${valid_models[@]}"; then
    echo "Invalid MODEL: $MODEL. Expected one of: ${valid_models[*]}"
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for SYSTEM in "${systems[@]}"; do
    if [[ "$MODEL" == "none" && "$SYSTEM" != "oner" ]]; then
        echo "Skipping system $SYSTEM because MODEL 'none' is only valid with 'oner'."
        continue
    fi

    for dataset in "${datasets[@]}"; do
        echo
        echo "============================================================"
        echo ">> Running dev pipeline for system: $SYSTEM | dataset: $dataset"
        echo "============================================================"
        "${SCRIPT_DIR}/run_retrieval_dev.sh" "$SYSTEM" "$MODEL" "$dataset" "$LLM_PORT"
    done
done

echo
echo "All datasets processed successfully."

