#!/bin/bash

# EM-Caddie  Installation Script
# This script sets up the conda environment for the project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="em-caddie"
ENV_FILE="environment.yml"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   EM-Caddie Installation Script       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Error: conda is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Anaconda or Miniconda from:${NC}"
    echo -e "  https://www.anaconda.com/download"
    echo -e "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ conda found: $(conda --version)${NC}"
echo ""

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}✗ Error: $ENV_FILE not found${NC}"
    echo -e "  Please ensure environment.yml is in the current directory"
    exit 1
fi

echo -e "${GREEN}✓ Found $ENV_FILE${NC}"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}⚠ Environment '$ENV_NAME' already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n $ENV_NAME -y
        echo -e "${GREEN}✓ Removed existing environment${NC}"
    else
        echo -e "${BLUE}Skipping environment creation${NC}"
        echo -e "${YELLOW}To use the existing environment, run:${NC}"
        echo -e "  conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create the environment
echo -e "${BLUE}Creating conda environment '$ENV_NAME'...${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo ""

if conda env create -f $ENV_FILE; then
    echo ""
    echo -e "${GREEN}✓ Environment created successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Environment creation failed${NC}"
    echo -e "${YELLOW}Try running manually:${NC}"
    echo -e "  conda env create -f $ENV_FILE"
    exit 1
fi

# Verify installation
echo ""
echo -e "${BLUE}Verifying installation...${NC}"

# Get the conda executable path
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Test key imports
python -c "
import sys
packages = {
    'torch': 'PyTorch',
    'streamlit': 'Streamlit',
    'transformers': 'Transformers',
    'numpy': 'NumPy',
    'pandas': 'Pandas'
}

failed = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f'✓ {name}')
    except ImportError:
        print(f'✗ {name}')
        failed.append(name)
        
if failed:
    print(f'\n⚠ Warning: Failed to import: {', '.join(failed)}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All key packages verified!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Some packages failed to import${NC}"
    echo -e "${YELLOW}You may need to troubleshoot the installation${NC}"
fi

# Check for GPU support
echo ""
echo -e "${BLUE}Checking GPU support...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('ℹ No GPU detected - will use CPU')
"

conda deactivate

# Final instructions
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       Installation Complete! 🎉        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}To get started:${NC}"
echo ""
echo -e "  1. Activate the environment:"
echo -e "     ${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo -e "  2. Run your application:"
echo -e "     ${YELLOW}streamlit run app.py${NC}"
echo ""
echo -e "  3. When finished, deactivate:"
echo -e "     ${YELLOW}conda deactivate${NC}"
echo ""
echo -e "${BLUE}For help, see README.md${NC}"
echo ""