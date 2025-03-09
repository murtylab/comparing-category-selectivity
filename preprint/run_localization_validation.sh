#!/bin/bash

OUTPUT_LOGFILE="localizationvalidation_output.log"

python main_analysis.py --localizationvalidation "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
