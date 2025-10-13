#!/usr/bin/env bash
set -e
cmd="$*"
ts="$(date '+%Y%m%d_%H%M%S')"
mkdir -p logs
log="logs/job_${ts}_$$.log"
echo "[START] $(date) :: $cmd" | tee -a "$log"
eval "$cmd" 2>&1 | tee -a "$log"
echo "[DONE ] $(date) :: exit=$?" | tee -a "$log"
