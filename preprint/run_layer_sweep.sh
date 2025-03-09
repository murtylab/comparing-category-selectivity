#!/bin/bash

LOGFILE="layersweep_restart_log.log"

while true; do
    python main_analysis.py --layersweep "$@" 2>&1 | tee layersweep_output.log
    if grep -q "CUDA out of memory" output.log; then
        echo "CUDA OOM detected. Restarting script..."
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Restarting due to CUDA OOM" >> "$LOGFILE"
        # sleep 5  # Optional delay before retrying
    else
        break  # Exit if no OOM error
    fi
done
