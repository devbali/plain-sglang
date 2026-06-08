#!/bin/bash
# Start SGLang server with static partition policy

set -e

# Configuration
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=30000
NUM_USERS=4
LOG_FILE="$HOME/sglang_static_partition.log"

echo "Starting SGLang server with static partition policy..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Users: $NUM_USERS"
echo "Log: $LOG_FILE"

# Activate venv
source ~/sglang_venv/bin/activate

# Start server in background
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --scheduling-policy-path sglang.srt.cache_hooks.static_partition_policy.StaticPartitionSchedulingPolicy \
    --scheduling-policy-init-kwargs '{"n": '$NUM_USERS'}' \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "Waiting for server to be ready..."

# Wait for server to be ready
sleep 5
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "✗ Server failed to start. Check logs:"
tail -50 "$LOG_FILE"
exit 1
