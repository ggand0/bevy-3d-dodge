#!/bin/bash
# Start multiple headless Bevy game servers for parallel RL training.
#
# Usage:
#   ./start_parallel_servers.sh [NUM_ENVS] [START_PORT]
#
# Examples:
#   ./start_parallel_servers.sh 4        # Start 4 servers on ports 8000-8003
#   ./start_parallel_servers.sh 8 9000   # Start 8 servers on ports 9000-9007
#
# Stop all servers:
#   pkill -f 'bevy_3d_dodge.*--headless'

NUM_ENVS=${1:-4}
START_PORT=${2:-8000}

echo "Starting $NUM_ENVS headless Bevy servers..."

# Build release first (only once)
echo "Building release binary..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Start each server
for i in $(seq 0 $((NUM_ENVS-1))); do
    PORT=$((START_PORT + i))
    echo "  Starting server on port $PORT..."

    # AMD GPU Vulkan settings (remove if not on AMD)
    VK_LOADER_DEBUG=error \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    ./target/release/bevy_3d_dodge --headless --port $PORT &
done

echo ""
echo "Started $NUM_ENVS servers on ports $START_PORT-$((START_PORT + NUM_ENVS - 1))"
echo ""
echo "To stop all servers: pkill -f 'bevy_3d_dodge.*--headless'"
echo ""

# Wait for all background processes
wait
