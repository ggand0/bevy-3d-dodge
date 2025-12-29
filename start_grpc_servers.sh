#!/bin/bash
# Start multiple headless Bevy game servers with gRPC for parallel RL training.
#
# Usage:
#   ./start_grpc_servers.sh [NUM_ENVS]
#
# Examples:
#   ./start_grpc_servers.sh       # Start 4 gRPC servers (default)
#   ./start_grpc_servers.sh 8     # Start 8 gRPC servers
#
# Stop all servers:
#   pkill -f 'bevy_3d_dodge.*--headless'

NUM_ENVS=${1:-4}

echo "Starting $NUM_ENVS headless gRPC Bevy servers..."

# Build release first (only once)
echo "Building release binary..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Clean up old sockets
rm -f /tmp/bevy_rl_*.sock

# Start each server
for i in $(seq 0 $((NUM_ENVS-1))); do
    SOCKET="/tmp/bevy_rl_${i}.sock"
    echo "  Starting server on $SOCKET..."
    VK_LOADER_DEBUG=error \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    ./target/release/bevy_3d_dodge --headless --fps 240 --socket-path $SOCKET &
done

echo ""
echo "Started $NUM_ENVS gRPC servers:"
for i in $(seq 0 $((NUM_ENVS-1))); do
    echo "  /tmp/bevy_rl_${i}.sock"
done
echo ""
echo "To stop all servers: pkill -f 'bevy_3d_dodge.*--headless'"
echo ""

# Wait for all background processes
wait
