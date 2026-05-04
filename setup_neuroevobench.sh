# ============================================================================
# FILENAME: setup_neuroevobench.sh
# ============================================================================

#!/bin/bash

# NeuroEvoBench Setup Script
#
# This script sets up the environment for NeuroEvoBench data loading and
# visualization. It creates a conda environment, installs dependencies,
# downloads the dataset, and runs basic validation tests.
#
# Requirements:
# - macOS with M4 chip
# - conda or miniconda installed
# - Internet connection for downloads
#
# Usage:
#   chmod +x setup_neuroevobench.sh
#   ./setup_neuroevobench.sh

set -e  # Exit on any error

# Configuration
ENV_NAME="neuroevo-viz"
PYTHON_VERSION="3.11"
DATA_URL="https://zenodo.org/records/10008966/files/NeuroEvoBench.zip?download=1"
DATA_DIR="./neuroevobench_data"
REQUIREMENTS_FILE="requirements_neuroevobench.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check if on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        log_error "This script is designed for macOS. Please adapt for other systems."
        exit 1
    fi

    # Check for conda
    if ! command -v conda &> /dev/null; then
        log_error "conda not found. Please install miniconda or anaconda."
        exit 1
    fi

    # Check for curl or wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        log_error "Neither curl nor wget found. Please install one."
        exit 1
    fi

    log_info "System requirements check passed."
}

# Create conda environment
create_environment() {
    log_info "Creating conda environment: $ENV_NAME"

    if conda env list | grep -q "^$ENV_NAME "; then
        log_warn "Environment $ENV_NAME already exists. Removing..."
        conda env remove -n "$ENV_NAME" -y
    fi

    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    log_info "Environment created successfully."
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install core dependencies
    pip install --upgrade pip

    # Install from requirements file if it exists, otherwise install manually
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
    else
        log_warn "$REQUIREMENTS_FILE not found, installing core dependencies manually..."

        # Core scientific computing
        pip install numpy scipy pandas matplotlib

        # Machine learning and visualization
        pip install scikit-learn umap-learn plotly

        # Data handling
        pip install h5py tqdm

        # JAX for M4 Metal backend
        pip install jax jax-metal

        # Optional: additional viz libraries
        pip install seaborn

        # Create requirements file for future reference
        pip freeze > "$REQUIREMENTS_FILE"
        log_info "Created $REQUIREMENTS_FILE"
    fi

    log_info "Dependencies installed successfully."
}

# Download NeuroEvoBench dataset
download_dataset() {
    log_info "Downloading NeuroEvoBench dataset..."

    mkdir -p "$DATA_DIR"

    # Download using curl or wget
    if command -v curl &> /dev/null; then
        curl -L "$DATA_URL" -o "${DATA_DIR}/NeuroEvoBench.zip"
    else
        wget "$DATA_URL" -O "${DATA_DIR}/NeuroEvoBench.zip"
    fi

    # Unzip
    log_info "Extracting dataset..."
    cd "$DATA_DIR"
    unzip NeuroEvoBench.zip
    cd ..

    log_info "Dataset downloaded and extracted to $DATA_DIR"
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Test basic imports
    python -c "
import numpy as np
import h5py
import umap
from pathlib import Path
print('✓ Core imports successful')

# Test JAX Metal backend
try:
    import jax
    import jax.numpy as jnp
    print('✓ JAX available')
    if jax.default_backend() == 'METAL':
        print('✓ JAX Metal backend active')
    else:
        print('⚠ JAX not using Metal backend')
except ImportError:
    print('⚠ JAX not available')
"

    # Test data loading
    if [ -d "$DATA_DIR" ]; then
        python -c "
from neuroevobench_loader import NeuroEvoBenchLoader
import os

data_path = '$DATA_DIR'
if os.path.exists(data_path):
    try:
        loader = NeuroEvoBenchLoader(data_path)
        available = loader.list_available()
        print(f'✓ Found {len(available)} HDF5 files')
        if len(available) > 0:
            print(f'✓ Sample file: {available.iloc[0][\"filename\"]}')
    except Exception as e:
        print(f'⚠ Loader test failed: {e}')
else:
    print('⚠ Data directory not found')
"
    fi

    log_info "Validation completed."
}

# Print usage instructions
print_instructions() {
    log_info "Setup completed successfully!"
    echo ""
    echo "To use the environment:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To run the pipeline:"
    echo "  python run_pipeline.py --data-dir $DATA_DIR --task brax_ant --algorithm openes --seed 0"
    echo ""
    echo "To run example notebook:"
    echo "  jupyter notebook example_usage.ipynb"
    echo ""
    echo "Environment details:"
    echo "  Name: $ENV_NAME"
    echo "  Python: $PYTHON_VERSION"
    echo "  Data: $DATA_DIR"
}

# Main execution
main() {
    log_info "Starting NeuroEvoBench setup..."

    check_requirements
    create_environment
    install_dependencies
    download_dataset
    validate_installation
    print_instructions

    log_info "Setup script completed successfully!"
}

# Run main function
main "$@"