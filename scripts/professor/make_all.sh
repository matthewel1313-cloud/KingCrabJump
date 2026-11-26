#!/usr/bin/env bash
# King Crab Grades Your Midterm -- rebuild every deliverable from source, in order.
#
# Expects sibling checkouts with their own environments (see README "Reproduce"):
#   $JUMPER  = KingKongRobotics/jumper        (Train; .venv with `pip install -e .`)
#   $DESIGN  = KingKongRobotics/jumper-design (Design; .venv with `pip install -e ".[sim,dev]"`, LFS assets)
# and MUJOCO_GL=egl (or osmesa) for headless rendering.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JUMPER="${JUMPER:-$HERE/../kingkongrobotics/jumper}"
DESIGN="${DESIGN:-$HERE/../kingkongrobotics/jumper-design}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
OUT="$HERE/output/main"     # the main build; the F build is assembled in step 7
ALT="$HERE/output/alt_F"
mkdir -p "$OUT" "$ALT"

design() { (cd "$DESIGN" && . .venv/bin/activate && "$@"); }
train()  { (cd "$JUMPER" && . .venv/bin/activate && "$@"); }

echo "== 1. skin: shell + crown + glasses + tweed paint, assembled on the trusted jumper profile"
design python - <<'EOF'
import json, numpy as np, trimesh
p = json.load(open("robots/jumper/profile.json")); t = p["cad_mm_to_parent_m"]
m = trimesh.load_mesh("robots/jumper/meshes/visual/upper_shell_link.stl", process=False)
m.vertices = ((m.vertices - np.asarray(t["translation_m"])) @ np.asarray(t["rotation_matrix"])) / t["scale"]
m.export("/tmp/kc_upper_cad_mm.stl")
EOF
design python "$HERE/skin/build_shell.py" --design "$DESIGN" --stock /tmp/kc_upper_cad_mm.stl --out "$HERE/skin/work"
rm -rf "$HERE/skin/work/simulation"
design python scripts/export_simulation.py --shell "$HERE/skin/work/shell.stl" --ams3mf "$HERE/skin/work/ams.3mf" \
  --visual-overrides "$HERE/skin/work/visual-overrides.json" --output "$HERE/skin/work/simulation"
design python "$HERE/render/views.py" "$HERE/skin/work/simulation/scene.xml" \
  /tmp/kc_skin_views --pose "$HERE/skin/work/simulation/preview-pose.json" --size 1024 --views three_quarter
cp /tmp/kc_skin_views/three_quarter.png "$HERE/skin/work/preview.png"
rm -f "$OUT/king_crab_professor.skin"
design python scripts/shellflow.py export-skin --simulation "$HERE/skin/work/simulation" --profile robots/jumper/profile.json \
  --shell "$HERE/skin/work/shell.stl" --id king-crab-professor --title "King Crab Professor" --version 1.0.0 \
  --author "matthewel1313-cloud" --license "Apache-2.0" --preview "$HERE/skin/work/preview.png" \
  --output "$OUT/king_crab_professor.skin"
design python scripts/shellflow.py verify-package "$OUT/king_crab_professor.skin" --profile robots/jumper/profile.json --mujoco

echo "== 2. map: textures, meshes, scene spec, robot-free preview, export, verify"
design python "$HERE/scene/make_assets.py" "$HERE/scene/assets"
design python "$HERE/scene/build_scene.py"
rm -rf /tmp/kc_scene && design python -m mjscene build "$HERE/scene/king_crab_classroom.scene.json" --out /tmp/kc_scene
rm -rf "$DESIGN/workspace" "$HERE/workspace"
design python "$HERE/scene/render_preview.py" /tmp/kc_scene/king-crab-classroom/scene.xml "$HERE/scene/work/preview.png"
rm -f "$OUT/king_crab_classroom.map"
design python scripts/shellflow.py export-map "$HERE/scene/king_crab_classroom.scene.json" --id king-crab-classroom \
  --title "King Crab's Class" --default-skin "$OUT/king_crab_professor.skin" --profile robots/jumper/profile.json \
  --preview "$HERE/scene/work/preview.png" --output "$OUT/king_crab_classroom.map"
design python scripts/shellflow.py verify-package "$OUT/king_crab_classroom.map" --profile robots/jumper/profile.json \
  --capability rigid --capability jumper --mujoco

echo "== 3. apps: the shipped exports, bundled with the repo's own deploy script (one per build)"
rm -rf "$HERE/motion/bundle" "$HERE/motion/bundle_alt_F"
train python scripts/deploy.py --manifests "$HERE/motion/manifests.json" --manifest king_crab_grader \
  --allow-incomplete --bundle "$HERE/motion/bundle"
train python scripts/deploy.py --manifests "$HERE/motion/manifests.json" --manifest king_crab_grader_alt_F \
  --allow-incomplete --bundle "$HERE/motion/bundle_alt_F"
cp "$HERE/motion/bundle/king_crab_grader.app" "$OUT/king_crab_grader.app"
cp "$HERE/motion/bundle_alt_F/king_crab_grader_alt_F.app" "$ALT/king_crab_grader_alt_F.app"

echo "== 4. the run: the app's controller, a scripted pad, the map, recorded"
train python "$HERE/motion/run_sequence.py" --app "$OUT/king_crab_grader.app" --timeline "$HERE/motion/timeline.py" \
  --scene "$OUT/king_crab_classroom.map" --seconds 30 --out "$HERE/motion/sequence.npz"
# the F build's own app over its 22 s; must match the main run step for step (it did: 0.0 difference)
train python "$HERE/motion/run_sequence.py" --app "$ALT/king_crab_grader_alt_F.app" --timeline "$HERE/motion/timeline.py" \
  --scene "$OUT/king_crab_classroom.map" --seconds 21.995 --out "$HERE/motion/sequence_alt_F.npz"

echo "== 5. render: the run replayed on the skin in the map, 1080x1920"
design python "$HERE/render/render_shots.py" --sequence "$HERE/motion/sequence.npz" --skin "$OUT/king_crab_professor.skin" \
  --map "$OUT/king_crab_classroom.map" --out "$HERE/render/out/raw_main.mp4"

echo "== 6. post: overlays, captions, audio; both cuts"
(cd "$HERE/post" && design python "$HERE/post/post.py" --raw "$HERE/render/out/raw_main.mp4" --out "$OUT/king_crab_main.mp4" --cut main ${MUSIC:+--music "$MUSIC"})
(cd "$HERE/post" && design python "$HERE/post/post.py" --raw "$HERE/render/out/raw_main.mp4" --out "$ALT/king_crab_alt_F.mp4" --cut alt_f ${MUSIC:+--music "$MUSIC"})

echo "== 7. the F build's packages: the same skin and map (byte-identical), its own app; verify all"
cp "$OUT/king_crab_professor.skin" "$OUT/king_crab_classroom.map" "$ALT/"
for b in "$OUT" "$ALT"; do
  design python scripts/shellflow.py verify-package "$b/king_crab_professor.skin" --profile robots/jumper/profile.json --mujoco
  design python scripts/shellflow.py verify-package "$b/king_crab_classroom.map" --profile robots/jumper/profile.json \
    --capability rigid --capability jumper --mujoco
done
echo "done: $OUT $ALT"
