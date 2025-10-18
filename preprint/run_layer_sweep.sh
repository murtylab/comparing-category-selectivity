#!/bin/bash

LOG_DIR="logs"
LOGFILE="$LOG_DIR/layersweep_restarts.log"
OUTPUT_LOGFILE="$LOG_DIR/layersweep_output.log"

MAX_RETRIES=100
RETRY_COUNT=0

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Run Python and capture both stdout and stderr
    python main_analysis.py --layersweep "$@" 2>&1 | tee "$OUTPUT_LOGFILE"

    # Check for CUDA or MPS or Metal-related OOM messages
    if grep -E -q "CUDA out of memory|MPS error: out of memory|Metal device failed allocation|resource_tracker: There appear to be" "$OUTPUT_LOGFILE"; then
        echo "GPU/MPS OOM or resource warning detected. Restarting script... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Restarting due to GPU/MPS OOM or resource warning (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)" >> "$LOGFILE"

        RETRY_COUNT=$((RETRY_COUNT+1))
    else
        # Exit if no OOM/resource warning found
        break
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "Reached maximum retry limit ($MAX_RETRIES). Exiting."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Reached maximum retry limit ($MAX_RETRIES). Exiting." >> "$LOGFILE"
fi
