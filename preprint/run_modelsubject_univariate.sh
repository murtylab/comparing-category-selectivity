#!/bin/bash

LOG_DIR="logs"
OUTPUT_LOGFILE="$LOG_DIR/modelsubjectunivariate_output.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

python main_analysis.py --modelsubjectunivariate "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
