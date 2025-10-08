#!/bin/bash

# Clean Python binaries and cache files

echo "Cleaning Python binaries and cache files..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove .egg-info directories
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove .pytest_cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove .coverage files
find . -type f -name ".coverage" -delete 2>/dev/null

# Remove build artifacts
rm -rf build/ dist/ 2>/dev/null

echo "Cleanup complete!"
