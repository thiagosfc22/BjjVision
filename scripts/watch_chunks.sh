#!/usr/bin/env bash
# Progress of a chunked full-match run. Read-only on the remote side.
#
#   scripts/watch_chunks.sh user@host 23771       # one shot
#   scripts/watch_chunks.sh user@host 23771 30    # refresh every 30s until Ctrl-C
#
# Why file sizes and not the run's own progress lines: the orchestrator pipes
# each chunk through `tail`, which buffers, so the log says nothing for the ~18
# minutes a chunk takes and then says everything at once. A quiet log is not a
# stalled run. `masks.bin` is written streaming, one record per frame, so its
# size is the honest liveness signal.
#
# Frames in flight is an ESTIMATE and says so: mask records are individually
# deflated, and two athletes standing apart compress far better than two
# interlocked, so bytes-per-frame is not constant. It self-calibrates from
# finished chunks -- measured at 3448 B/frame on this match against the 4800 a
# guess from one grappling clip gave -- so treat it as a guess until chunk 1
# lands, and as roughly right after.
set -uo pipefail
HOST="${1:?uso: $0 user@host porta [...]}"
PORT="${2:?porta}"
EVERY="${3:-0}"                      # 0 = one shot
REMOTE_DIR="${REMOTE_DIR:-BjjVision}"

payload() {
ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
  "cd ~/$REMOTE_DIR 2>/dev/null || { echo 'sem ~/$REMOTE_DIR nesse host'; exit 3; }
/venv/main/bin/python - <<'PY'
import json, os, glob, time
OUT = 'data/out'
B  = [0, 4000, 8000, 12000, 16000, 20000, 23306]
CH = [(B[i], B[i+1]) for i in range(6)]

done, done_frames, bpf = [], 0, None
for a, b in CH:
    idx = f'{OUT}/chunk_{a}_{b}/masks.idx.json'
    if not os.path.exists(idx):
        continue
    m  = json.load(open(idx)); n = len(m['frames'])
    sz = os.path.getsize(f'{OUT}/chunk_{a}_{b}/masks.bin')
    lo, hi = m['frames'][0], m['frames'][-1]
    # a chunk holding frames outside its own window is stale output from an
    # earlier run that the orchestrator moved without noticing the failure
    done.append((a, b, n, lo, hi, lo >= a - 1 and hi <= b + 1))
    done_frames += n; bpf = sz / max(n, 1)

print(f'chunks concluidos: {len(done)}/6')
for a, b, n, lo, hi, ok in done:
    print(f'   {a:>5}-{b:<5} {n:>5} frames  (masks {lo}-{hi})  {\"ok\" if ok else \"<<< FAIXA ERRADA\"}')

cur = next(((a, b) for a, b in CH if not os.path.exists(f'{OUT}/chunk_{a}_{b}')), None)
cur_frames = 0
if cur is None:
    print('\nTODOS OS CHUNKS TERMINARAM')
else:
    live = f'{OUT}/galvao-xande/masks.bin'
    print(f'\nchunk em andamento: {cur[0]}-{cur[1]}')
    if os.path.exists(live):
        sz = os.path.getsize(live); k = bpf or 3448.0
        cur_frames = min(int(sz / k), cur[1] - cur[0])
        src = f'calibrado em {k:.0f} B/frame' if bpf else 'chute, nenhum chunk fechou'
        print(f'   masks.bin {sz/1048576:6.1f} MB  ->  ~{cur_frames} de {cur[1]-cur[0]} '
              f'({100.0*cur_frames/(cur[1]-cur[0]):.0f}%)  [{src}]')
        raw = glob.glob(f'{OUT}/galvao-xande/*.raw.mp4')
        if raw:
            print(f'   raw.mp4   {os.path.getsize(raw[0])/1048576:6.1f} MB, '
                  f'escrito ha {time.time()-os.path.getmtime(raw[0]):.0f}s')
    else:
        print('   iniciando (extraindo frames ou carregando modelos)')

tot = done_frames + cur_frames
print(f'\nprogresso geral: {tot}/{B[-1]} frames  ({100.0*tot/B[-1]:.1f}%)')
if len(done) >= 1:
    per = 1062                                  # medido: chunk 1 levou 17.7 min
    print(f'faltam ~{(6-len(done))*per//60} min ({6-len(done)} chunks x ~18 min)')
PY
echo
echo -n 'GPU ............ '; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo -n 'pipeline ....... '; pgrep -c -f 'bjjvision.cli run' 2>/dev/null || echo 0
echo -n 'disco .......... '; df -Ph / | awk 'NR==2{print \$4\" livres (\"\$5\" usado)\"}'
echo -n 'hora UTC ....... '; date -u +%H:%M:%S" 2>&1 | grep -viE 'welcome to vast|have fun|AI agents'
}

if [ "$EVERY" -eq 0 ]; then
  payload
else
  while :; do
    payload
    echo; echo "--- atualiza em ${EVERY}s (Ctrl-C para sair) ---"
    sleep "$EVERY"
  done
fi
