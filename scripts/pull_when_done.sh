#!/usr/bin/env bash
# Wait for the remote run to finish, then pull the results down. Detached and
# read-only on the remote side; safe to leave running unattended.
#
# Deliberately does NOT write into data/out/<slug>/ -- that directory holds the
# validated Metal baseline, and `bjj sync-down` would go straight over it.
#
#   nohup scripts/pull_when_done.sh > data/out/_autopull.log 2>&1 &
set -uo pipefail
HOST="${1:?uso: $0 user@host porta [...]}"
PORT="${2:?porta}"
SLUG="${3:-galvao-xande}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/out/${SLUG}_full"
SSH=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "waiting on $HOST:$PORT for $SLUG"

fails=0
for i in $(seq 1 720); do            # 720 x 30s = 6h ceiling
  state=$("${SSH[@]}" "$HOST" '
    cd "$HOME/BjjVision" 2>/dev/null || { echo UNREACHABLE; exit 0; }
    # Marker first, and never trust pgrep alone here: this very check is passed
    # to a remote shell whose own command line contains "bjjvision.cli run", so
    # a plain pgrep -f matches itself and reports RUNNING forever. The bracket
    # in bjjvision[.]cli stops the pattern from matching its own literal text.
    if grep -q "=== done" run.log 2>/dev/null; then echo DONE
    elif pgrep -f "bjjvision[.]cli run" >/dev/null; then echo RUNNING
    else echo STOPPED; fi' 2>/dev/null | grep -Eo "RUNNING|DONE|STOPPED|UNREACHABLE" | head -1)

  case "${state:-}" in
    RUNNING) fails=0 ;;
    DONE|STOPPED)
      log "remote state: $state -- pulling"
      mkdir -p "$DEST"
      # No --delete: never let a partial remote state erase a good local copy.
      rsync -az --partial -e "ssh -p $PORT" "$HOST:BjjVision/data/out/$SLUG/" "$DEST/" && {
        log "pulled to $DEST"; ls -lh "$DEST" | tail -n +2 | sed "s/^/    /"
      }
      "${SSH[@]}" "$HOST" 'tr "\r" "\n" < ~/BjjVision/run.log | grep -v "propagate in video\|frame loading" | tail -60' \
        2>/dev/null > "$DEST/run_tail.log" && log "log tail -> $DEST/run_tail.log"
      [ "$state" = STOPPED ] && log "WARNING: process gone but no '=== done' marker; the run may have died. Read run_tail.log."
      osascript -e "display notification \"$SLUG pulled to data/out/${SLUG}_full\" with title \"BjjVision\"" 2>/dev/null
      log "DONE. The VAST instance is still running and still billing -- destroy it at cloud.vast.ai/instances"
      exit 0 ;;
    *)
      fails=$((fails + 1))
      log "unreachable ($fails)"
      [ "$fails" -ge 20 ] && { log "giving up after 20 consecutive failures"; exit 1; } ;;
  esac
  sleep 30
done
log "ceiling reached without the run finishing"
