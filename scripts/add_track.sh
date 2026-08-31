#!/usr/bin/env bash
# Mux a music track onto the finished Stories cut.
#
#   scripts/add_track.sh <audio-file> [start_seconds] [out.mp4]
#
# Default start is 28 s, matching the cut Thiago asked for. Video is copied,
# not re-encoded, so this costs seconds and loses no quality.
set -euo pipefail

AUDIO="${1:?usage: add_track.sh <audio-file> [start_seconds] [out.mp4]}"
START="${2:-28}"
VIDEO="data/out/galvao-xande_scale/story_final_sem_trilha.mp4"
OUT="${3:-data/out/galvao-xande_scale/story_final.mp4}"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
FADEOUT=$(python3 -c "print(max(float('$DUR')-2.5, 0))")

ffmpeg -y -hide_banner -loglevel error \
  -i "$VIDEO" -ss "$START" -i "$AUDIO" \
  -filter_complex "[1:a]atrim=duration=${DUR},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.8,afade=t=out:st=${FADEOUT}:d=2.5,aresample=48000[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 256k -movflags +faststart "$OUT"

echo "wrote $OUT"
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name -of default=noprint_wrappers=1 "$OUT" | head -4
