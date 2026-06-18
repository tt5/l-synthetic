#!/bin/bash
# run_pipeline.sh

cd /home/n/data/l/synthetic

PIDS=""

cleanup() {
    echo "Stopping..."
    for pid in $PIDS; do
        kill $pid 2>/dev/null
    done
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

./nats-server &
PIDS="$PIDS $!"
sleep 2

uv run listener.py &
PIDS="$PIDS $!"
sleep 1

uv run classifier.py &
PIDS="$PIDS $!"
sleep 1

uv run publisher.py &
PIDS="$PIDS $!"

echo "Pipeline started. Press Ctrl+C to stop."
wait
