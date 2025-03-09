#!/bin/bash
LOGFILE="layersweep_restarts.log"
OUTPUT_LOGFILE="layersweep_output.log"

MAX_RETRIES=100
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    python main_analysis.py --layersweep "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
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
