#!/usr/bin/env bash

# Run retrieval + prediction + evaluation on the TEST split for multiple systems/datasets.

set -euo pipefail

valid_systems=("ircot" "ircot_qa" "oner" "oner_qa" "nor_qa")
valid_models=("flan-t5-xxl" "flan-t5-xl" "gpt" "none")
datasets=("hotpotqa" "2wikimultihopqa" "musique" "nq" "trivia" "squad")

usage() {
    echo "Usage: $0 SYSTEM_LIST MODEL LLM_PORT_NUM"
    echo "  SYSTEM_LIST  : comma-separated list of systems (e.g., ircot_qa,oner_qa,nor_qa)"
    echo "  MODEL        : ${valid_models[*]}"
    echo "  LLM_PORT_NUM : Port of the LLM service (e.g., 8010)"
    exit 1
}

if [[ $# -ne 3 ]]; then
    usage
fi

IFS=',' read -r -a SYSTEMS <<< "$1"
MODEL="$2"
LLM_PORT="$3"

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

for system in "${SYSTEMS[@]}"; do
    if ! check_in_list "$system" "${valid_systems[@]}"; then
        echo "Invalid SYSTEM: $system. Expected one of: ${valid_systems[*]}"
        usage
    fi
done

if ! check_in_list "$MODEL" "${valid_models[@]}"; then
    echo "Invalid MODEL: $MODEL. Expected one of: ${valid_models[*]}"
    usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for SYSTEM in "${SYSTEMS[@]}"; do
    if [[ "$MODEL" == "none" && "$SYSTEM" != "oner" ]]; then
        echo "Skipping system $SYSTEM because MODEL 'none' is only valid with 'oner'."
        continue
    fi

    for dataset in "${datasets[@]}"; do
        echo
        echo "============================================================"
        echo ">> Running TEST pipeline for system: $SYSTEM | dataset: $dataset"
        echo "============================================================"
        "${SCRIPT_DIR}/run_retrieval_test.sh" "$SYSTEM" "$MODEL" "$dataset" "$LLM_PORT"
    done
done

echo
echo "All TEST datasets processed successfully."

