#!/usr/bin/env bash
# Rebuild Cart Heist end to end: map -> app -> training robot model -> simulation -> frames -> final mp4.
#
#   JUMPER=/path/to/jumper DESIGN=/path/to/jumper-design ./build_all.sh
#
# JUMPER : github.com/KingKongRobotics/jumper with its venv at .venv (pip install -e ".[dev]", torch)
# DESIGN : github.com/KingKongRobotics/jumper-design with its venv at .venv
#          (pip install -e ".[sim,dev]" pillow) and the robot meshes pulled from Git LFS
# Headless rendering needs EGL (Mesa is fine): MUJOCO_GL=egl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JUMPER="$(cd "${JUMPER:?set JUMPER}" && pwd)"
DESIGN="$(cd "${DESIGN:?set DESIGN}" && pwd)"
export BUILD="${BUILD:-$HERE/build}" MUJOCO_GL=egl
mkdir -p "$BUILD" "$REPO/renders"

# 1. the map: scene.json -> export-map (King Crab gold-crown skin as default robot) -> verify-package
DESIGN="$DESIGN" "$HERE/tools/build_map.sh"

source "$JUMPER/.venv/bin/activate"
# 2. the app: the shipped bundle trimmed to the six modes used; deploy.py --check-reference
python "$HERE/tools/build_app.py" --jumper "$JUMPER" --out "$REPO/apps/cart_heist.app"
# 3. the training robot model the policies were trained on (jumper.posture, compiled by mjlab)
( cd "$JUMPER" && python "$HERE/tools/dump_robot_model.py" "$BUILD/robot_dump.xml" )
# 4. simulate the story (shipped policies + scripted pad operator), record every 5 ms tick
python "$HERE/sequencer.py" --out "$BUILD/run" --jumper "$JUMPER" --robot-dump "$BUILD/robot_dump.xml" \
       | tee "$BUILD/run_log.txt"
# 5. legs never pass through the basket or the cart rack (exact signed distances)
python "$HERE/tools/check_clearance.py" "$BUILD/run" "$BUILD/robot_dump.xml" "$JUMPER"
# 6. frames through the edit decision list
python "$HERE/render.py" --run "$BUILD/run" --out "$BUILD/frames" --jumper "$JUMPER" \
       --robot-dump "$BUILD/robot_dump.xml" --tmp "$BUILD/render_tmp" --workers "${WORKERS:-4}"
# 7. captions, watermark, end card, original soundtrack, mux
python "$HERE/edit/compose.py" --run "$BUILD/run" --frames "$BUILD/frames" --work "$BUILD/edit" \
       --out "$REPO/renders/cart_heist.mp4"
