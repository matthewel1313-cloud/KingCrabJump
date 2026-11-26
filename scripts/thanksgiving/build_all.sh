#!/usr/bin/env bash
# Rebuild King Crab end to end: packages -> simulation -> frames -> final mp4.
#
#   JUMPER=/path/to/jumper DESIGN=/path/to/jumper-design ./build_all.sh
#
# JUMPER  : a checkout of github.com/KingKongRobotics/jumper with its venv at .venv
#           (pip install -e ".[dev]")
# DESIGN  : a checkout of github.com/KingKongRobotics/jumper-design with its own venv
#           at .venv (pip install -e ".[sim,dev]" pillow) and the robot meshes pulled
#           from Git LFS (robots/jumper/**)
# Headless rendering needs EGL (Mesa is fine): MUJOCO_GL=egl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JUMPER="${JUMPER:?set JUMPER}"
DESIGN="${DESIGN:?set DESIGN}"
BUILD="${BUILD:-$HERE/build}"
OUT="${OUT:-$HERE/../out_final}"
mkdir -p "$BUILD" "$HERE/packages" "$OUT"
export MUJOCO_GL=egl

# 1. .skin (jumper-design: assemble -> export-skin -> verify-package --mujoco)
( source "$DESIGN/.venv/bin/activate"
  python "$HERE/tools/build_skin.py" --design "$DESIGN" --work "$BUILD/skin" \
         --out "$HERE/packages/king-crab.skin"
# 2. .map (scene.json -> export-map with the King Crab skin as its default robot)
  python "$HERE/scene/author_scene.py" "$BUILD/scene"
  python "$DESIGN/scripts/shellflow.py" export-map "$BUILD/scene/scene.json" \
         --default-skin "$HERE/packages/king-crab.skin" --output "$HERE/packages/king-crab-table.map"
  python "$DESIGN/scripts/shellflow.py" verify-package "$HERE/packages/king-crab-table.map" \
         --profile "$DESIGN/robots/jumper/profile.json" --capability rigid --capability jumper --mujoco
  cp "$BUILD/scene/scene.json" "$HERE/scene/king-crab-table.scene.json" )

source "$JUMPER/.venv/bin/activate"
# 3. .app (shipped bundle trimmed to the four modes used; reference check)
python "$HERE/tools/build_app.py" --jumper "$JUMPER" --out "$HERE/packages/king-crab.app"
# 4. the training robot model the policies were trained on
( cd "$JUMPER" && python "$HERE/tools/dump_robot_model.py" "$BUILD/robot_dump.xml" )
# 5. simulate the story (policies + scripted operator), record every 5 ms tick
python "$HERE/sequencer.py" --out "$BUILD/run" --jumper "$JUMPER" --robot-dump "$BUILD/robot_dump.xml"
# 6. frames through the edit decision list
python "$HERE/render.py" --run "$BUILD/run" --out "$BUILD/frames" --jumper "$JUMPER" \
       --robot-dump "$BUILD/robot_dump.xml" --tmp "$BUILD/render_tmp" --workers "${WORKERS:-3}"
# 7. captions, soundtrack, mux
python "$HERE/edit/compose.py" --run "$BUILD/run" --frames "$BUILD/frames" --out "$OUT/king_crab_purpose.mp4"
