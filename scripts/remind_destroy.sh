#!/usr/bin/env bash
# Nag until the VAST instance is gone. Exits by itself the moment the host stops
# answering -- i.e. the moment you destroy it -- so it cannot outlive its point.
set -uo pipefail
HOST="${1:?uso: $0 user@host porta [...]}"; PORT="${2:?porta}"; EVERY="${3:-600}"
for i in $(seq 1 200); do
  if ! ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" true 2>/dev/null; then
    echo "[$(date -u +%H:%M:%S)] host not answering -- assuming destroyed, stopping"
    osascript -e 'display notification "instance gone, reminder off" with title "BjjVision"' 2>/dev/null
    exit 0
  fi
  hrs=$(echo "scale=2; $i * $EVERY / 3600" | bc -l)
  cost=$(echo "scale=2; $hrs * 0.583" | bc -l)
  echo "[$(date -u +%H:%M:%S)] still up; ~\$$cost since the reminder started"
  osascript -e "display notification \"VAST instance still billing (~\\\$$cost since reminder). cloud.vast.ai/instances -> Destroy\" with title \"BjjVision\"" 2>/dev/null
  sleep "$EVERY"
done
