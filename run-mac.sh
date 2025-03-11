#!/bin/bash

# Check if XQuartz is installed
if ! command -v xquartz &> /dev/null; then
    echo "XQuartz is not installed. Please install it using:"
    echo "brew install --cask xquartz"
    exit 1
fi

# Start XQuartz if not running
if ! ps -e | grep -q XQuartz; then
    echo "Starting XQuartz..."
    open -a XQuartz
    sleep 3
fi

# Configure XQuartz to allow connections from network clients
defaults write org.xquartz.X11 nolisten_tcp 0

# Allow connections from Docker
xhost + localhost

# Set environment variables for docker-compose
export X11_SOCKET_PATH="/tmp/.X11-unix"
export DISPLAY_ENV="host.docker.internal:0"
export CAMERA_DEVICE="/dev/video0"

# Build and run the Docker container
echo "Building and starting the application..."
docker-compose up --build

# Clean up
xhost - localhost
echo "Application closed." 