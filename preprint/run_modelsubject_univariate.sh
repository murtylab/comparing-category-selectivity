#!/bin/bash

OUTPUT_LOGFILE="modelsubjectunivariate_output.log"

python main_analysis.py --modelsubjectunivariate "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
