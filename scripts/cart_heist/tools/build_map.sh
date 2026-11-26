#!/usr/bin/env bash
# Author the store scene, package it as black_friday_store.map (King Crab gold skin as the
# default robot), and run jumper-design's checks. DESIGN = a jumper-design checkout with its venv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DESIGN="${DESIGN:?set DESIGN}"
SKIN="${SKIN:-$HERE/../../skins/king-crab-gold-1.skin}"
BUILD="${BUILD:-$HERE/build}"
source "$DESIGN/.venv/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python "$HERE/scene/author_scene.py" "$BUILD/scene"
MAP="$HERE/../../maps/black_friday_store.map"
rm -f "$MAP"
python "$DESIGN/scripts/shellflow.py" export-map "$BUILD/scene/scene.json" --default-skin "$SKIN" \
       --output "$MAP" > "$BUILD/export_map.json"
python "$DESIGN/scripts/shellflow.py" verify-package "$MAP" \
       --profile "$DESIGN/robots/jumper/profile.json" --capability rigid --capability jumper --mujoco \
       > "$BUILD/verify_map.json"
python - "$BUILD/export_map.json" "$BUILD/verify_map.json" <<'PY'
import json, sys
e, v = (json.load(open(p)) for p in sys.argv[1:])
assert e["ok"] and v["ok"], (e, v)
c = v["native_mujoco"]["composition"]
print("export ok; verify ok; mujoco", v["native_mujoco"]["passed"], "; composition", c["passed"],
      "; spawn", c["spawn"]["position"], "yaw", c["spawn"]["yaw"], "; spawn support", c["spawn_support"]["passed"],
      "; severe spawn contacts", c["severe_spawn_contacts"])
PY
cp "$BUILD/scene/scene.json" "$HERE/scene/black_friday_store.scene.json"
