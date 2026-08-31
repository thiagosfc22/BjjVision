#!/usr/bin/env bash
# Teacher queue for one rented box: every fight, sequentially, disk-safe.
#
#   bash remote/run_queue.sh [slug ...]        # default: the campaign queue
#
# The expensive lesson this script encodes: a rental burns money while a human
# decides what to run next. So each fight goes smoke-test -> full run -> frame
# cleanup with no pauses, failures move to the next fight instead of idling,
# and the JPEG directory (the disk hog: ~3.4 GB per 10 min of match) is
# deleted the moment masks.bin exists. Smoke first, always -- 900 frames tell
# you calibration locked onto the right two people before you pay for the
# other 20,000 (see remote/README.md).
#
# Order: held-out fights first -- their labels unlock the only honest
# generalisation number -- then easiest-first so early results flow back while
# the hard 2009 footage still runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

QUEUE=("$@")
[ ${#QUEUE[@]} -eq 0 ] && QUEUE=(paulista23-master2 wardzinski-ferreira
                                 wardzinski-lo adgs26-artsygov paulista22-roxa
                                 paulista22-branca gracie-calasans gracie-bastos)
CFG=config/vast_teacher.yaml

for slug in "${QUEUE[@]}"; do
  echo "######## $slug $(date -u +%H:%M) ########"
  if [ -f "data/out/$slug/masks.idx.json" ]; then
    echo "$slug: masks.bin ja existe, pulando"; continue
  fi
  if [ ! -f "data/interim/${slug}_norm.mp4" ]; then
    echo "$slug: sem norm.mp4 -- sync-up faltou"; continue
  fi
  ./bjj frames "$slug" || { echo "$slug: frames FALHOU"; continue; }
  ./bjj run "$slug" --config "$CFG" --no-llm --max-frames 900 \
    || { echo "$slug: SMOKE FALHOU -- pulando o run completo"; continue; }
  ./bjj run "$slug" --config "$CFG" --no-llm \
    || { echo "$slug: run completo FALHOU"; continue; }
  rm -rf "data/interim/${slug}_frames"
  echo "$slug: completo, frames limpos"
done
echo "######## FILA COMPLETA $(date -u +%H:%M) -- rode sync-down e DESTRUA a instancia ########"
