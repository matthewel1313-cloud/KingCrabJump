# King Crab, Surgical Tech 👑🦀🩺: notes

A 23.6 s vertical (1080×1920, 30 fps) MuJoCo video of the open-source
[Jumper](https://github.com/KingKongRobotics/jumper) crab robot working as a scrub tech:
it fetches instruments from a Mayo stand and passes them to a surgeon's gloved hand.

Output: `out_final/king_crab_surgeon.mp4` (H.264 + AAC).

## Folder

| path | what |
|---|---|
| `out_final/king_crab_surgeon.mp4` | the video |
| `scene/operating_room.map` | the scene, a `kk-scene-package/1` that Jumper's own importer loads (`scripts/play.py --scene`) |
| `scene/scene.xml`, `scene/build_map.py` | its source and builder |
| `skin/king_crab_surgeon.skin`, `skin/build_skin.py` | the appearance overlay (see "the .skin" below) |
| `app/jumper_example.app` | the policy bundle that drives the robot: Jumper's shipped example `.app`, unmodified |
| `app/choreography.json` | the show as the operator plays it: modes, pad inputs and measured beat times |
| `sim/harness.py` | builds the world (Jumper env + map + skin + welds) and the scripted gamepad |
| `sim/choreography.py` | **the control script**: the scripted operator, the hybrid grasp, the glove, the beats |
| `sim/render.py`, `sim/camera.py` | replays recorded takes and draws them from the three cameras |
| `edit/assemble.py`, `edit/captions.py`, `edit/sound.py` | edit decision list, captions, synthesized audio, mux |
| `CREDITS.md` | sources and licenses |

## How to re-run

```bash
# 1. Jumper itself (CPU is fine)
git clone https://github.com/KingKongRobotics/jumper ~/jumper && cd ~/jumper
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
pip uninstall -y triton          # the CUDA torch wheel's triton segfaults on a GPU-less box
sudo apt-get install -y libosmesa6 libegl1 fonts-noto-color-emoji ffmpeg
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa     # or egl on a machine with a GPU

# 2. this folder (paths: set JUMPER_ROOT to your jumper checkout; APP defaults to its out/bundle_example/jumper.app)
K=<this repo>/king_crab
python $K/scene/build_map.py                 # -> scene/operating_room.map
python $K/skin/build_skin.py                 # -> skin/king_crab_surgeon.skin
python $K/sim/choreography.py main 22.5      # simulate the show -> out/sim_main.{npz,json}
python $K/sim/choreography.py blooper        # the outtake      -> out/sim_blooper.{npz,json}
python $K/sim/render.py preview main         # optional: low-res 3-camera check
KC_WORKERS=4 python $K/edit/assemble.py      # render 1080x1920, caption, score, mux
```

Every run is deterministic: the spawn pose is pinned and the task's reset/startup
randomization (pose, friction, encoder bias, COM) is switched off in `harness.make_world`.
On this 4-core cloud box, simulation runs about 3× slower than real time (~75 s for the
main take) and the full-resolution software render takes roughly half an hour.

## What drives what: policy vs. scripted

| | how |
|---|---|
| every robot joint, every step | **policy**: Jumper's shipped ONNX policies through the bundle's own compiled controller (`controller.so`), via `mjrl.app_play.AppPlayer`, exactly as `play.py --app` runs it |
| walking, sideways scuttling, body twist toward the surgeon, nose-up "look up" / "offer it up" | **policy** (`claw_left` mode = `jumper.five_foot`; the sticks are its velocity and pose commands) |
| carrying the claw, opening and closing it | **policy** + the task's deploy hook (LT closes the claw) |
| victory wiggle, claws up | **policy** (`dance_dream_wings`, entered with menu + d-pad right) |
| *which* buttons and sticks, and when | **scripted operator** (`sim/choreography.py`): a virtual gamepad that closes the loop like a person would, pushing the sticks toward the next handle from the measured claw position |
| holding a tool | **scripted physics glue**: a MuJoCo weld (tool → `LF_palm_link`) switched on when the claw has closed. It then slides the tool ~3 cm into the jaws over 0.12–0.16 s (`seat`), because the carried claw lies *across* the crab's front and the handle overhang is not between the grip inserts when the jaws close. A held tool does not collide |
| putting the gauze back | **scripted**: the weld target is steered back to where the gauze was picked up, then released; it lands under physics |
| the glove | **scripted** kinematic mocap body on an eased path; at arrival a weld (tool → glove) switches on and the claw weld off; the claw then opens (policy) |
| used tools | moved off-set to a hidden back table once the glove has left frame |
| tools resting, the gauze landing, the blooper | **physics**: free rigid bodies with real masses and friction |
| the blooper | **physics only**: a separate take built with `weld_grasp=False`, so the jaws really collide with the tool and no weld exists. Squeezed by friction alone, the scalpel is shoved ~2.6 cm along the table instead of lifted |

The grasp is therefore a hybrid: approach, alignment and closing are the policy driven
by a closed-loop scripted operator; the hold is a weld. That is the honest trade: the
shipped `jumper.five_foot` claw policy was trained to carry and close the claw, not to
pick up a 15 g, 5 mm thick instrument, and the blooper shows what friction alone does.

## What Jumper supports out of the box, and what had to be added

Out of the box (verified here):
- `assets/jumper/jumper.xml`: the robot (22 DoF, 2.5 kg; body ~18 cm long, ~45 cm leg span, 14 cm claws)
- `out/bundle_example/jumper.app`: one bundle with 12 trained modes: `locomotion` (posture),
  `claw_left`/`claw_right` (five-foot walking with a carried claw, LT/RT closes it),
  4 dances, 4 gestures, `jump`. The robot is driven through the bundle's compiled controller
- `scripts/play.py --app <app> --headless --video clip.mp4`: the one MP4 recorder. Verified with a 5 s clip
- `.map` (`kk-scene-package/1|2`) import for replay (`scenes/package.py`)
- the gamepad and keyboard vocabulary (`controller/vocabulary.json`, the bundle's `manual.en.json`)

Added for this video:
- a scripted gamepad (`ScriptedPad`) handed to `AppPlayer` in place of a real pad
- the OR scene (`.map`), welds and contact excludes injected through the scene's `spec_fn` hook
- a `.skin` loader (Train has none; see below)
- an offline replay renderer with three cameras, captions, sound and the edit

## Things that worked differently than the brief assumed (and what was done)

- **Robot scale.** The brief says a "13 cm sim robot". Jumper's body is ~18 cm long with a
  ~45 cm leg span and 14 cm claws, so the instruments are ~8.5 cm rather than 4–6 cm. At
  4–6 cm they did not read on screen and were smaller than the claw's grip pads.
- **There is no separate "walk / turn / grasp / dance" policy set.** The example `.app` is
  one controller with modes, entered by pad buttons. Grasping is `claw_left` (walk on five
  legs, carry the left claw, LT closes it). "Turn" is the right stick: its first half
  twists the body in place, beyond 0.5 it walks a turn. Sticks have a 0.15 deadband.
- **The arm presets (d-pad) point this claw forward-down**, which would aim a held tool at
  the table, so "presenting" uses the policy's nose-up pitch instead.
- **No `.skin` loader in Train, and `.skin/3` forbids added geometry.** jumper-design's
  `kk-skin-package/3` is a whole assembled robot from its print/assembly pipeline, and its
  validator allows only recolours of baseline visual geoms. A crown, cap and mask are added
  geometry, so `king_crab_surgeon.skin` is an overlay with its own schema id,
  `kingcrab-skin-overlay/1`: `/3`'s `visual_overrides` vocabulary plus `accessories`
  (primitive geoms, contype=0, conaffinity=0, density=0, so no collision and no mass).
  It is **not** a validator-passing `/3` package.
- **The `.app` was not rebuilt.** Building a bundle (`scripts/deploy.py`) cross-compiles the
  Rust controller for three hosts and converts to RKNN; none of that changes what runs in
  sim. The show uses the shipped bundle unmodified; the show itself is the control script
  plus `app/choreography.json`.
- **Headless GL.** There is no GPU here: EGL has no device, so rendering uses OSMesa. One
  process can hold only one working OSMesa context (later renderers produced garbage), so
  the final render uses one process per worker.
- **Native backend quirks** (`mujoco.rollout`): `eq_active` is re-initialized from the
  model's `eq_active0` every step, so a weld must be switched in both; the mocap buffer
  starts at zero (the glove would sit inside the robot), so the glove is parked explicitly.
- **Weld `relpose`** is body2's pose in body1's frame (not the reverse); thin tools needed
  a slightly padded roll inertia, or the stiff weld buzzed.
- **A `fell_over` reset** happens if the robot walks off the tray edge; the choreography stays on it.

## Known limits

- The grasp is a weld (see above); the tool slides ~3 cm into the jaws in the frames after
  closing. The jaw links and the held tool do not collide, so a close look shows the
  handle passing into the jaw meshes.
- The glove is kinematic and has no fingers that move; it is a prop on a path.
- "Faster" is faster because the operator pushes harder and waits less, not because a
  policy learned anything. The forceps beat is shorter than the scalpel beat for the same
  reason (the caption says "shows learning"; in sim it is the operator getting slicker).
- Beat timings come from the simulation, not the brief: hook 0–2.1 s, scalpel 2.1–5.7,
  forceps 5.7–9.0, scissors (with the gauze mistake) 9.0–15.6, "Faster!" 15.6–18.6,
  "perfect" and the dance 18.6–21.7, blooper 21.7–23.65.
- Software rendering: no anti-aliasing beyond MuJoCo's default, shadows from one light.
- Sim only. Nothing here was run on a physical Jumper.

## How it's useful

A scrub tech spends a surgery anticipating and passing instruments, and every handoff is
a chance for the wrong tool, a fumble or a reach across the sterile field. A small,
table-top instrument handler that fetches the called tool and presents it handle-first in
a fixed spot shortens the surgeon's reach, keeps eyes on the field, and makes each pass
countable and checkable, which is the same bookkeeping that prevents retained
instruments. This video shows the loop in simulation with an open-source robot and its
shipped policies: called tool → locate → grasp → present → confirm. The parts that are
still scripted (the hold, the glove) are listed above, and they are exactly where the
next work is: a grasp policy trained on instrument geometry, and perception that reads
the tray instead of a lookup table.
