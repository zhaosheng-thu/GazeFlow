#!/usr/bin/env bash
# Download EGTEA Gaze+ clips (~22 GB) and gaze data into data/egtea/ (or the folder given as $1).
set -euo pipefail

ROOT="${1:-data/egtea}"
mkdir -p "$ROOT"
cd "$ROOT"

fetch() {
  if [ -f "$2" ]; then echo "[skip] $2 already downloaded"; return; fi
  echo "[download] $2"
  wget -c -O "$2.part" "$1" && mv "$2.part" "$2"
}

fetch "https://www.dropbox.com/s/2aryvztw044w9ih/gaze_data.zip?dl=1" gaze_data.zip
fetch "https://www.dropbox.com/s/udynz2u62wpdva6/video_clips.tar?dl=1" video_clips.tar

echo "[extract] gaze_data/"
unzip -q -o gaze_data.zip
if [ -d gaze_data/gaze_data ]; then  # the official zip nests the .txt files one level deeper
  mv gaze_data/gaze_data/*.txt gaze_data/
  rmdir gaze_data/gaze_data
fi
echo "[extract] cropped_clips/ (this takes a few minutes)"
tar -xf video_clips.tar

n_clips=$(find cropped_clips -name "*.mp4" | wc -l)
n_gaze=$(find gaze_data -maxdepth 1 -name "*.txt" | wc -l)
echo "[done] $n_clips clips in $ROOT/cropped_clips, $n_gaze gaze files in $ROOT/gaze_data"
echo "You can delete $ROOT/video_clips.tar and $ROOT/gaze_data.zip to save space."
