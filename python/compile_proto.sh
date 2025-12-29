#!/bin/bash
# Compile protobuf definitions for Python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../proto"
OUT_DIR="$SCRIPT_DIR/bevy_dodge_env"

echo "Compiling proto files..."
echo "  Proto dir: $PROTO_DIR"
echo "  Output dir: $OUT_DIR"

python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/rl_env.proto"

# Fix imports to be relative (required for package imports)
echo "Fixing imports to be relative..."
sed -i 's/import rl_env_pb2 as/from . import rl_env_pb2 as/' "$OUT_DIR/rl_env_pb2_grpc.py"

echo "Done! Generated files:"
ls -la "$OUT_DIR"/rl_env_pb2*.py
