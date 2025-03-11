#!/bin/bash

LOG_DIR="logs"
LOGFILE="$LOG_DIR/layersweep_restarts.log"
OUTPUT_LOGFILE="$LOG_DIR/layersweep_output.log"

MAX_RETRIES=100
RETRY_COUNT=0

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    python main_analysis.py --layersweep "$@" 2>&1 | tee "$OUTPUT_LOGFILE"
    if grep -q "CUDA out of memory" "$OUTPUT_LOGFILE"; then
        echo "CUDA OOM detected. Restarting script... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Restarting due to CUDA OOM (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)" >> "$LOGFILE"
        RETRY_COUNT=$((RETRY_COUNT+1))
        # sleep 5  # Optional delay before retrying
    else
        break  # Exit if no OOM error
    fi
done

if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "Reached maximum retry limit ($MAX_RETRIES). Exiting."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Reached maximum retry limit ($MAX_RETRIES). Exiting." >> "$LOGFILE"
fi
