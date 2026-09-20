#!/usr/bin/env bash
# Runs the two follow-up experiments back to back. Status in logs/run_extra_status.txt.
cd "$(dirname "$0")/.."          # run from the repo root
PY=./.venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M:%S')  $*" >> logs/run_extra_status.txt; }
log "=== started ==="
for s in 26_quantum_features_rf 25_geometric_difference; do
  log "starting scripts/$s.py"
  t=$(date +%s)
  $PY -u scripts/$s.py > logs/$s.log 2>&1
  log "scripts/$s.py exit $? ($(( ($(date +%s)-t)/60 )) min)"
done
log "=== done ==="
