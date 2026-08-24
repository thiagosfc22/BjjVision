#!/usr/bin/env bash
# Watch a BjjVision run on the GPU host: phase, progress, RAM, disk, and the
# output files as they appear. Ctrl-C to stop; it only reads, never writes.
#
#   scripts/watch_remote.sh [user@host] [port] [seconds] [max_ticks]
set -uo pipefail
HOST="${1:-root@<host>}"
PORT="${2:-11436}"
EVERY="${3:-5}"
MAX="${4:-0}"                       # 0 = forever

# One multiplexed connection: the first tick pays the handshake, the rest are
# ~50ms. Without this a 5-second refresh spends most of its time in SSH setup.
CTL="/tmp/bjjwatch-$$.sock"
SSH=(ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15
     -o ControlMaster=auto -o ControlPath="$CTL" -o ControlPersist=30)
cleanup() { "${SSH[@]}" -O exit "$HOST" >/dev/null 2>&1; printf "\033[?25h\n"; }
trap cleanup EXIT INT TERM

printf "\033[?25l"                  # hide cursor; the trap puts it back
tick=0
while :; do
  tick=$((tick + 1))
  out=$("${SSH[@]}" "$HOST" '
    cd "$HOME/BjjVision" 2>/dev/null || { echo "no ~/BjjVision on this host"; exit 0; }
    pid=$(pgrep -f "bjjvision.cli run" | head -1)
    if [ -n "$pid" ]; then
      set -- $(ps -o rss=,etime= -p "$pid")
      printf "state    running  pid %s  elapsed %s  rss %.1f GB\n" "$pid" "$2" "$(echo "$1/1048576" | bc -l)"
    elif grep -q "=== done" run.log 2>/dev/null; then
      echo "state    finished"
    else
      echo "state    NOT RUNNING (check the tail below)"
    fi
    printf "host     %s free on /   %s\n" \
      "$(df -Ph / | awk "NR==2{print \$4}")" \
      "$(free -g | awk "NR==2{printf \"ram %s/%s GB used\", \$3, \$2}")"
    printf "gpu      %s\n" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    echo
    echo "phase"
    tr "\r" "\n" < run.log 2>/dev/null | grep -v "^[[:space:]]*$" | tail -1 | cut -c1-100
    last=$(tr "\r" "\n" < run.log 2>/dev/null | grep -E "frames  conf=" | tail -1)
    [ -n "$last" ] && echo "$last"
    echo
    echo "files in data/out/galvao-xande"
    if [ -d data/out/galvao-xande ]; then
      ls -lh --time-style=+%H:%M:%S data/out/galvao-xande | tail -n +2 | awk "{printf \"  %-32s %8s  %s\n\", \$7, \$5, \$6}"
    else
      echo "  (nothing yet)"
    fi
    echo
    echo "frames on disk"
    printf "  %s jpg in %s_frames\n" "$(ls data/interim/galvao-xande_frames 2>/dev/null | wc -l)" "galvao-xande"
  ' 2>&1 | grep -v "^Welcome to vast\|^Have fun\|^AI agents:")

  printf "\033[H\033[2J"            # home, then clear: no flicker
  echo "BjjVision  $HOST:$PORT   $(date -u +%H:%M:%S) UTC   tick $tick"
  echo "------------------------------------------------------------------"
  echo "$out"
  echo "------------------------------------------------------------------"
  echo "Ctrl-C to stop. Read-only: this never touches the run."

  [ "$MAX" -gt 0 ] && [ "$tick" -ge "$MAX" ] && break
  sleep "$EVERY"
done
