#!/bin/bash
#
# Build all dataset indexes with BM25 + Dense + SPLADE
#
# Usage:
#   ./build_all_indexes.sh [--bm25-only|--with-dense|--with-splade|--all]
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default: use all indexes
INDEX_MODE="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bm25-only)
      INDEX_MODE="bm25"
      shift
      ;;
    --with-dense)
      INDEX_MODE="dense"
      shift
      ;;
    --with-splade)
      INDEX_MODE="splade"
      shift
      ;;
    --all)
      INDEX_MODE="all"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo
      echo "Options:"
      echo "  --bm25-only     Build indexes with BM25 only (fastest)"
      echo "  --with-dense    Build indexes with BM25 + Dense embeddings"
      echo "  --with-splade   Build indexes with BM25 + SPLADE"
      echo "  --all           Build indexes with BM25 + Dense + SPLADE (default)"
      echo "  --help, -h      Show this help message"
      echo
      echo "Examples:"
      echo "  $0                    # Build all indexes with all features"
      echo "  $0 --bm25-only        # Only BM25 (fast)"
      echo "  $0 --with-splade      # BM25 + SPLADE"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Build All Dataset Indexes${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Datasets to build
DATASETS=(
    "hotpotqa"
    "iirc"
    "2wikimultihopqa"
    "musique"
    "wiki"
)

# Determine index arguments
INDEX_ARGS=""
case $INDEX_MODE in
    bm25)
        echo -e "${YELLOW}Mode: BM25 only${NC}"
        INDEX_ARGS=""
        ;;
    dense)
        echo -e "${YELLOW}Mode: BM25 + Dense embeddings${NC}"
        INDEX_ARGS="--use-dense"
        ;;
    splade)
        echo -e "${YELLOW}Mode: BM25 + SPLADE${NC}"
        INDEX_ARGS="--use-splade"
        ;;
    all)
        echo -e "${YELLOW}Mode: BM25 + Dense + SPLADE${NC}"
        INDEX_ARGS="--use-dense --use-splade"
        ;;
esac

echo -e "${BLUE}Datasets to build: ${#DATASETS[@]}${NC}"
echo

# Check if Elasticsearch is running
echo -e "${BLUE}Checking Elasticsearch...${NC}"
cd "$SCRIPT_DIR"
ES_STATUS=$(./es.sh status || echo "not running")
if [[ "$ES_STATUS" == *"not running"* ]]; then
    echo -e "${RED}Elasticsearch is not running!${NC}"
    echo -e "${YELLOW}Starting Elasticsearch...${NC}"
    ./es.sh start
    echo
fi

cd "$PROJECT_ROOT"

# Track success/failure
SUCCESSFUL=()
FAILED=()
TOTAL=${#DATASETS[@]}
CURRENT=0

# Start time
START_TIME=$(date +%s)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Index Building${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Build each dataset
for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$CURRENT/$TOTAL] Building index: ${GREEN}$dataset${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    
    # Build the index
    DATASET_START=$(date +%s)
    
    if python retriever_server/build_index.py "$dataset" $INDEX_ARGS --force; then
        DATASET_END=$(date +%s)
        DATASET_TIME=$((DATASET_END - DATASET_START))
        
        echo
        echo -e "${GREEN}✓ Successfully built index: $dataset${NC}"
        echo -e "${GREEN}  Time taken: ${DATASET_TIME}s${NC}"
        SUCCESSFUL+=("$dataset")
    else
        DATASET_END=$(date +%s)
        DATASET_TIME=$((DATASET_END - DATASET_START))
        
        echo
        echo -e "${RED}✗ Failed to build index: $dataset${NC}"
        echo -e "${RED}  Time taken: ${DATASET_TIME}s${NC}"
        FAILED+=("$dataset")
        
        # Ask if user wants to continue
        echo
        read -p "Continue with remaining datasets? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Aborted by user${NC}"
            break
        fi
    fi
done

# End time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Build Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "Total time: ${YELLOW}${TOTAL_MINUTES}m ${TOTAL_SECONDS}s${NC}"
echo
echo -e "Successful: ${GREEN}${#SUCCESSFUL[@]}${NC}"
for dataset in "${SUCCESSFUL[@]}"; do
    echo -e "  ${GREEN}✓${NC} $dataset"
done
echo
echo -e "Failed: ${RED}${#FAILED[@]}${NC}"
for dataset in "${FAILED[@]}"; do
    echo -e "  ${RED}✗${NC} $dataset"
done
echo

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${RED}Some indexes failed to build${NC}"
    exit 1
else
    echo -e "${GREEN}All indexes built successfully!${NC}"
    exit 0
fi

