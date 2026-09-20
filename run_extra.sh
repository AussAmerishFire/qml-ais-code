#!/usr/bin/env bash
# Runs the two follow-up experiments back to back. Status in run_extra_status.txt.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> run_extra_status.txt; }
log "=== started ==="
for s in 26_quantum_features_rf 25_geometric_difference; do
  log "starting $s.py"
  t=$(date +%s)
  $PY -u $s.py > $s.log 2>&1
  log "$s.py exit $? ($(( ($(date +%s)-t)/60 )) min)"
done
log "=== done ==="
