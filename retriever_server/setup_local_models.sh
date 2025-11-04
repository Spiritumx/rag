#!/bin/bash
#
# Quick setup script for downloading and using local models
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Local Models Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Default model directory
MODEL_DIR="${MODEL_DIR:-./models}"

# Parse command line arguments
DOWNLOAD_DENSE=false
DOWNLOAD_SPLADE=false
DOWNLOAD_ALL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --dense)
      DOWNLOAD_DENSE=true
      shift
      ;;
    --splade)
      DOWNLOAD_SPLADE=true
      shift
      ;;
    --all)
      DOWNLOAD_ALL=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo
      echo "Options:"
      echo "  --model-dir DIR   Directory to save models (default: ./models)"
      echo "  --dense           Download dense embedding model only"
      echo "  --splade          Download SPLADE model only"
      echo "  --all             Download all models (default if no option specified)"
      echo "  --help, -h        Show this help message"
      echo
      echo "Examples:"
      echo "  $0 --all"
      echo "  $0 --dense --model-dir /root/autodl-tmp/models"
      echo "  $0 --all --model-dir /root/autodl-tmp/models"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# If no specific download option is set, download all
if [ "$DOWNLOAD_DENSE" = false ] && [ "$DOWNLOAD_SPLADE" = false ] && [ "$DOWNLOAD_ALL" = false ]; then
  DOWNLOAD_ALL=true
fi

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Model directory: ${GREEN}$MODEL_DIR${NC}"
echo -e "  Download dense: ${YELLOW}$DOWNLOAD_DENSE${NC}"
echo -e "  Download SPLADE: ${YELLOW}$DOWNLOAD_SPLADE${NC}"
echo -e "  Download all: ${YELLOW}$DOWNLOAD_ALL${NC}"
echo

# Check Python and required packages
echo -e "${BLUE}Checking dependencies...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found${NC}"

# Check if required packages are installed
python -c "import transformers" 2>/dev/null || {
    echo -e "${YELLOW}Warning: transformers not found. Installing...${NC}"
    pip install transformers
}

python -c "import sentence_transformers" 2>/dev/null || {
    echo -e "${YELLOW}Warning: sentence-transformers not found. Installing...${NC}"
    pip install sentence-transformers
}

echo -e "${GREEN}✓ All dependencies ready${NC}"
echo

# Download models
cd "$PROJECT_ROOT"

DOWNLOAD_CMD="python retriever_server/download_model.py --model-dir $MODEL_DIR"

if [ "$DOWNLOAD_ALL" = true ]; then
  DOWNLOAD_CMD="$DOWNLOAD_CMD --all"
elif [ "$DOWNLOAD_DENSE" = true ] && [ "$DOWNLOAD_SPLADE" = true ]; then
  DOWNLOAD_CMD="$DOWNLOAD_CMD --download-dense --download-splade"
elif [ "$DOWNLOAD_DENSE" = true ]; then
  DOWNLOAD_CMD="$DOWNLOAD_CMD --download-dense"
elif [ "$DOWNLOAD_SPLADE" = true ]; then
  DOWNLOAD_CMD="$DOWNLOAD_CMD --download-splade"
fi

echo -e "${BLUE}Running download command:${NC}"
echo -e "${YELLOW}$DOWNLOAD_CMD${NC}"
echo

eval "$DOWNLOAD_CMD"

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Start Elasticsearch:"
echo -e "   ${YELLOW}cd retriever_server && ./es.sh start${NC}"
echo
echo -e "2. Build index with local models:"
echo -e "   ${YELLOW}python retriever_server/build_index.py <dataset> \\${NC}"

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DENSE" = true ]; then
  echo -e "   ${YELLOW}  --use-dense --dense-model-path $MODEL_DIR/dense/all-MiniLM-L6-v2 \\${NC}"
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPLADE" = true ]; then
  echo -e "   ${YELLOW}  --use-splade --splade-model-path $MODEL_DIR/splade/splade-cocondenser-ensembledistil${NC}"
fi

echo
echo -e "${BLUE}Example:${NC}"
echo -e "   ${YELLOW}python retriever_server/build_index.py wiki \\${NC}"

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_DENSE" = true ]; then
  echo -e "   ${YELLOW}  --use-dense --dense-model-path $MODEL_DIR/dense/all-MiniLM-L6-v2 \\${NC}"
fi

if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_SPLADE" = true ]; then
  echo -e "   ${YELLOW}  --use-splade --splade-model-path $MODEL_DIR/splade/splade-cocondenser-ensembledistil${NC}"
fi

echo

