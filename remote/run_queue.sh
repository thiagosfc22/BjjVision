#!/usr/bin/env bash
# Teacher queue for one rented box: every fight, chunked, disk- and RAM-safe.
#
#   bash remote/run_queue.sh [slug ...]        # default: the campaign queue
#
# SAM2's init_state materialises EVERY frame of the frame directory as a CPU
# tensor at 12.6 MB/frame -- a full match is 180+ GB and the first version of
# this script killed five smoke tests in five minutes learning that. So the
# unit of work is a CHUNK: extract ~1200 frames (15 GB on a 31 GB box), run,
# move the outputs into data/out/<slug>/chunk_A_B/ (the layout studentdata
# already reads), delete the JPEGs, next. Chunks that exist are skipped, so a
# dead run resumes where it stopped instead of re-paying for what finished.
#
# Order: held-out fights first -- their labels unlock the only honest
# generalisation number -- then easiest-first so early results flow back.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${BJJ_PYTHON:-/venv/main/bin/python}"

# Chunks held to what THIS box's RAM proves it can hold, not what a previous
# box could. Override per launch: CHUNK=2000 bash remote/run_queue.sh ...
CHUNK="${CHUNK:-1200}"
CFG="${CFG:-config/vast_teacher.yaml}"

QUEUE=("$@")
[ ${#QUEUE[@]} -eq 0 ] && QUEUE=(paulista23-master2 wardzinski-ferreira
                                 wardzinski-lo paulista22-roxa gracie-bastos)

for slug in "${QUEUE[@]}"; do
  echo "######## $slug $(date -u +%H:%M) ########"
  if [ -f "data/out/$slug/DONE" ]; then
    echo "$slug: DONE ja existe, pulando"; continue
  fi
  if [ ! -f "data/interim/${slug}_norm.mp4" ]; then
    echo "$slug: sem norm.mp4 -- sync-up faltou"; continue
  fi
  N=$("$PY" -c "import json; print(json.load(open('data/interim/${slug}.json'))['n_frames'])")
  # Calibrate ONCE per fight, not once per chunk: gi colour is invariant
  # across the match, and a chunk covering a restart or the neighbouring mat
  # has no frame to calibrate from -- three paulista23 chunks died exactly
  # that way. The middle of the longest mat shot is where the fight lives;
  # `bjj run` picks the _calib directory up automatically.
  # NO_CALIB="slug1 slug2" forces per-chunk self-calibration for those fights.
  # Measured on this campaign: the shared calib dir works when the mat is
  # constant across shots (roxa, single camera) and BREAKS the matfield path
  # when it is not (ferreira, paulista23) -- the per-shot mat refit and a
  # cross-shot calibration directory disagree about what the mat looks like.
  CAL="data/interim/${slug}_calib"
  if [[ " ${NO_CALIB:-} " == *" $slug "* ]]; then
    rm -rf "$CAL"
    echo "  $slug: auto-calibracao por chunk (NO_CALIB)"
  elif ! ls "$CAL"/*.jpg >/dev/null 2>&1; then
    read -r ca cb <<< "$("$PY" -c "
import json
d = json.load(open('data/interim/${slug}_shots.json'))
sh = [s for s in d['shots'] if s['kind'] in ('match', 'mat')]
s = max(sh, key=lambda s: s['end'] - s['start'])
m = (s['start'] + s['end']) // 2
print(max(s['start'], m - 75), min(s['end'], m + 75))")"
    echo "  calibracao da luta: frames $ca-$cb (meio do shot mais longo)"
    ./bjj frames "$slug" --frames "$ca:$cb" --calib-frames "$ca:$cb" \
      || echo "  extracao de calibracao falhou; chunks calibram sozinhos"
  fi
  fail=0
  for ((a = 0; a < N; a += CHUNK)); do
    b=$((a + CHUNK)); [ "$b" -gt "$N" ] && b="$N"
    ck="data/out/$slug/chunk_${a}_${b}"
    if [ -f "$ck/masks.idx.json" ]; then
      echo "  chunk $a-$b ja existe, pulando"; continue
    fi
    echo "  == chunk $a-$b ($(date -u +%H:%M)) =="
    ./bjj frames "$slug" --frames "$a:$b" || { echo "  frames FALHOU"; fail=1; break; }
    ./bjj run "$slug" --config "$CFG" --no-llm --frames "$a:$b" \
      || { echo "  chunk $a-$b FALHOU -- seguindo para o proximo"; fail=1; continue; }
    mkdir -p "$ck"
    mv "data/out/$slug/masks.bin" "data/out/$slug/masks.idx.json" "$ck/" \
      || { echo "  saida do chunk sumiu"; fail=1; continue; }
    mv "data/out/$slug/features.parquet" "$ck/" 2>/dev/null || true
    cp "data/out/$slug/report.json" "$ck/" 2>/dev/null || true
    rm -f "data/out/$slug/${slug}_analysis"*.mp4 2>/dev/null || true
  done
  rm -rf "data/interim/${slug}_frames"
  if [ "$fail" -eq 0 ]; then
    touch "data/out/$slug/DONE"
    echo "$slug: COMPLETO"
  else
    echo "$slug: terminou com falhas -- chunks presentes sao validos, DONE nao marcado"
  fi
done
echo "######## FILA COMPLETA $(date -u +%H:%M) -- sync-down e destruir a instancia ########"
