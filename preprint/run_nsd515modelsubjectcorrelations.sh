#!/bin/bash

LOG_DIR="logs"
OUTPUT_LOGFILE="$LOG_DIR/modelsubjectcorrelations_output.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

python main_analysis.py --nsd515modelsubjectcorrelations "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
