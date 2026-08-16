#!/bin/bash
# Full pipeline for a list of songs, one after another.
#
#   ./custom_tools/batch_build.sh "Song A" "Song B" ...
#
# Each song: analysis (demucs stems) -> cue detection -> generate -> render ->
# verify. Harmony/lyrics run inside generate and cache themselves. Everything
# is idempotent: analysis and cue detection are skipped when their cached
# output already exists, so a re-run after a failure resumes rather than
# repeating the expensive parts.
set -u
cd /Users/kristsauders/xlights-mcp-server || exit 1
PY=.venv/bin/python3
SHOW="/Volumes/XLIGHTS/XLights 2026"
XL=/Applications/xLights.app/Contents/MacOS/xLights

# Next free version number for a song, so a rebuild never silently overwrites
# the version Krists is currently reviewing in xLights. Songs joined the set at
# different times (Gooey is on v20, the newest songs on v1), so a single
# hardcoded version was wrong for everyone; this gives each song its own
# next-up number. Override for the whole batch with VER=<n>.
next_version() {
  local song="$1" n=0 f
  for f in "$SHOW/$song (v"*").xsq"; do
    [ -e "$f" ] || continue
    local v="${f##*\(v}"; v="${v%%\)*}"
    case "$v" in (*[!0-9]*|"") continue;; esac
    [ "$v" -gt "$n" ] && n="$v"
  done
  echo $((n + 1))
}

# Layout and channel map are derived from the show file and go stale silently
# whenever props move, are added, or are disabled -- and a stale channel map
# means every per-prop measurement reads the wrong pixels. Measured after one
# round of Krists's changes: 168 of 227 models had shifted channels. Refreshed
# here so it cannot be forgotten.
echo "[0/5] refreshing layout + channel map ..."
$PY custom_tools/extract_layout.py >/dev/null 2>&1
$PY -c "import sys; sys.path.insert(0,'custom_tools'); import channel_map; channel_map.build(force=True)" >/dev/null 2>&1

for SONG in "$@"; do
  V="${VER:-$(next_version "$SONG")}"
  echo "================ $SONG (v$V) ================"
  GEN="$SHOW/.generation/$SONG"

  if [ ! -f "$GEN/analysis.json" ]; then
    echo "[1/5] analysis (stem separation) ..."
    $PY custom_tools/run_analysis.py "$SONG" 2>&1 | tail -2
  else
    echo "[1/5] analysis cached"
  fi
  [ -f "$GEN/analysis.json" ] || { echo "!! analysis failed, skipping $SONG"; continue; }

  if [ ! -f "$GEN/cues.json" ]; then
    echo "[2/5] cue detection ..."
    $PY custom_tools/detect_cues.py "$SONG" 2>&1 | tail -2
  else
    echo "[2/5] cues cached"
  fi

  echo "[3/5] generate ..."
  $PY custom_tools/generate_song.py "$SONG" --out "$SONG (v$V).xsq" 2>&1 \
    | grep -E "Sections:|Harmony:|Vocals:|Instrument voices|Beat accent|Beat-locked|Run-up|Drum fills|Unison|Whole-house|placements across" \
    | sed 's/^/      /'

  if [ ! -f "$SHOW/$SONG (v$V).xsq" ]; then echo "!! generate failed"; continue; fi

  echo "[4/5] render ..."
  "$XL" -s "$SHOW" -r "$SHOW/$SONG (v$V).xsq" >/dev/null 2>&1

  echo "[5/5] verify ..."
  $PY custom_tools/verify_render.py "$SONG (v$V).xsq" --no-render 2>&1 | tail -4 | sed 's/^/      /'
  $PY custom_tools/verify_sequence.py "$SONG (v$V).xsq" 2>&1 \
    | grep -A2 "STALE" | sed 's/^/      /'
  # Rhythm check: is the render actually ACCENTING the beat and breathing
  # before section changes? The two verifiers above cannot see either -- one
  # checks what we asked for, the other checks aggregate colour -- and a
  # sequence passing both while having no relationship to the beat is exactly
  # the state this pipeline shipped in for several rounds.
  $PY custom_tools/verify_beat_lock.py "$SONG (v$V)" "$SONG" 2>&1 \
    | grep -E "whole display|NOT participating|median|dipping" | sed 's/^/      /'
done
echo "================ BATCH COMPLETE ================"
