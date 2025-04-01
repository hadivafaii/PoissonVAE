#!/bin/bash


# Change to the parent directory of scripts/
cd "$(dirname "$0")/.." || { echo "Error: Failed to navigate to parent directory."; exit 1; }

# Infer project name if not provided
project_name="${1:-$(basename "$PWD")}"

# Define the FITS directory path
project_dir="$PWD"

# Change to the FITS directory
cd "${project_dir}" || { echo "Error: Directory ${project_dir} not found."; exit 1; }

# Run the Python script
fit="python3 -m utils.copyfits '${project_name}'"
eval "${fit}"
