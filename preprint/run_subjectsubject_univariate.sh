#!/bin/bash

OUTPUT_LOGFILE="subjectsubjectunivariate_output.log"

python main_analysis.py --subjectsubjectunivariate "$@" 2>&1 | tee -a "$OUTPUT_LOGFILE"
