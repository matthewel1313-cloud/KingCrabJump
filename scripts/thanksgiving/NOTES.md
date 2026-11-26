# King Crab — "What is my purpose?" — build notes

A 27.8 s vertical (1080×1920, 30 fps) MuJoCo short for the KingKong Robotics Crab Robot
Challenge, made with the open-source **Jumper** crab robot. A crowned Jumper sits in the
middle of a Thanksgiving table. The opening joke is that it only passes butter. It ends up
passing whatever brings the family together. The concept is an original parody nod: no
characters, designs, quotes or audio from any show are used.

Final video: `out_final/king_crab_purpose.mp4` (H.264 High / AAC, 1080×1920, 30 fps).

## Step 0 — what the repos actually contain

Sources (cloned 2026-11-26):

| repo | commit | what we used |
|---|---|---|
| [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper) | `198686eac0d1` | robot MJCF, the shipped `.app` bundle (controller + ONNX policies), `play.py`, `deploy.py` |
| [KingKongRobotics/jumper-design](https://github.com/KingKongRobotics/jumper-design) | `918d192857fc` | `.skin` / `.map` protocol, `shellflow` CLI (`assemble`, `export-skin`, `export-map`, `verify-package`), robot profile |

Findings, and where the repos differ from the brief:

* **Robot MJCF:** `jumper/assets/jumper/jumper.xml` has 22 actuated joints (`LF_J0..J4`,
  `RF_J0..J4` and three per mid/rear leg) and a free base. The body is about 20 cm long and
  stands about 13 cm tall to the top of the shell (`STAND_Z` = 0.106 m base height). Its actuators are
  *not* in the XML. mjlab adds them as a PD "servo curve" actuator
  (`tasks/jumper/common/actuator.py`: kp 10, kd 0.5, 1.746 N·m plateau, measured torque–speed
  curve).
* **`.skin` / `.map` live in jumper-design, not in jumper.** Both are ZIP packages with a
  hash-inventoried manifest (`kk-skin-package/3`, `kk-scene-package/2`,
  `docs/content-packages.md`). A `.skin/3` is display-only. It may only recolour non-collision
  visual geoms (`visual_overrides`) and replace the upper-shell visual. The replacement comes
  from a shell STL in CAD millimetres plus an optional Bambu AMS 3MF that carries the colours.
  A `.map/2` is an mjscene-compiled `scene.xml` plus a verified default `.skin`. Both are
  checked by `shellflow verify-package`.
* **`.app` is the deployment bundle** (`kk-policy-bundle/1`, `jumper/deploy/BUNDLE.md`). It
  holds `controller.toml` (the FSM cascade and buttons), one contract and policy per mode
  (ONNX + RKNN), recorded reference frames, and the compiled controller for every host: the
  board (aarch64), the web (wasm) and mjlab (`controller.so`, which Python imports as
  `mjrl_fsm`). The repo ships one: `jumper/out/bundle_example/jumper.app`.
* **Shipped policies (in that bundle):** `locomotion` (walk / scuttle / turn / posture, from
  `jumper.posture`, 200 Hz), `claw_left` / `claw_right` (the "grasp" policy, `jumper.five_foot`:
  it walks on five legs and carries the sixth as a claw, 50 Hz; a d-pad preset holds the arm
  out and the trigger closes the jaw), `dance_crab` / `dance_brazilian` / `dance_maze` /
  `dance_dream_wings`, `gesture_*` and `jump`. There is no separate "turn" policy: turning is
  `locomotion`'s yaw-rate command.
* **Headless rendering:** `MUJOCO_GL=egl`. This container had no libEGL; installing Mesa's
  `libegl1 libegl-mesa0 libgl1-mesa-dri` (software rendering) was enough. The test clip
  `step0/test_play_app.mp4` (5 s, 151 frames) was made with the repo's own recorder:
  `python scripts/play.py --app out/bundle_example/jumper.app --headless --steps 1000 --video …`.
* Git LFS objects for jumper-design (robot meshes, library packages) aren't served by this
  environment's git proxy. They were fetched from `media.githubusercontent.com`, and each
  file's SHA-256 was checked against its LFS pointer before use.

## Step 1 — the scene (`packages/king-crab-table.map`)

`scene/author_scene.py` writes an mjscene `scene.json` (kept as
`scene/king-crab-table.scene.json`). `shellflow export-map … --default-skin king-crab.skin`
then compiles and packages it. `verify-package --capability rigid --capability jumper --mujoco`
passes. Spawn-support rays hit the table top at z = 0.76 m.

* **Room:** a long walnut table (2.3 × 1.04 m, top at 0.76 m), a deep-green runner with gold
  diamonds, five place settings (plates, cider glasses, cutlery), a roast turkey, pumpkins and
  five candles with emissive flames.
* **Lighting:** a low warm "golden-hour" spot through the window side, a neutral fill, a warm
  rim, and four point lights for the candle glow. The walls carry a procedurally painted dining
  room (window with a low sun, curtains, family pictures, sideboard lamps, bokeh) that is
  pre-blurred, so the background reads as shallow depth of field. The renderer also adds a
  depth-of-field pass from the depth buffer.
* **Props** (dynamic rigid bodies, scaled to the robot, 12–50 g): butter dish, bread-roll
  basket, gravy boat, cranberry dish, smartphone, a kid's folded drawing (two thin card panels
  as a tent, with a crayon hand-turkey on them), and a folded napkin (a soft contact,
  `solref` 0.05). Props live on their own collision bit (2048). They rest on the table but
  never tangle with the robot's per-leg collision channels.
* **Family:** dad (red flannel + watch), mom (mustard cable-knit), aunt (olive knit), kid (teal
  hoodie, a smaller hand) and grandma (lavender cardigan + pearl bracelet). Each is one
  `hand_*` body built from primitives: sleeve, cuff, wrist, palm, four cupped fingers and a
  thumb. They are authored as `static` and parked under the table edge. The sequencer turns
  them into **mocap (kinematic) bodies** and flies them in on eased paths that come up over the
  table edge.
* All textures are generated in code, so the scene contains no third-party assets.

## Step 2 — the look (`packages/king-crab.skin`)

`tools/build_skin.py` runs jumper-design's own pipeline. The design was chosen in the brief
(warm orange-red Jumper, tiny gold crown, napkin bib), so no concept round was needed.

1. Invert the profile's `cad_mm_to_parent_m` to bring the stock upper-shell mesh back to CAD
   millimetres exactly, so the seam-registration check passes untouched.
2. Add a gold crown (band, five points with balls, ruby gems, velvet cap, tilted 9°) on top,
   and a cream napkin bib with a red stripe tucked under the display, in the same frame.
3. Write a Bambu-style AMS 3MF whose triangles match the STL one-for-one, painted with four
   palette slots: `#E2461E` shell, `#E9B43A` gold, `#F3ECDA` napkin, `#A3122A` ruby.
4. Write `visual_overrides`: limbs in orange-red, the formerly grey anodised parts in peach
   `#F2BD95`, and the lower body in a deep roasted red. These are visual geoms only.
5. Run `export_simulation.py` (the `assemble` exporter), then `export-skin`, then
   `verify-package --profile robots/jumper/profile.json --mujoco`. The colour report reads
   `palette_from_exact_matching_ams`; a monochrome fallback would have stopped the build.

The video's physics robot is the *training* model (below), so the skin's display layer is
applied to it at load (`sim/world.py::_skin_visuals`): every visual colour, plus the skin's
shell meshes in place of the stock shell visual. Collision, inertia and joints stay the
training model's.

Not in the skin: the glowing **eyes** on the robot's display module. They are what the screen
shows (a runtime layer in `sim/world.py::_eyes`), with expressions driven by the sequencer.

## Step 3 — behaviour (`packages/king-crab.app` + `sequencer.py`)

### The `.app`

`tools/build_app.py` makes `king-crab.app` from the shipped bundle, trimmed to the four modes
the story uses: `locomotion`, `claw_left`, `claw_right` and `dance_crab`. **No policy was
retrained and no controller rebuilt.** The kept files are byte-identical to the source bundle:
controller builds, ONNX/RKNN, contracts, recordings and reference frames. `controller.toml`
loses the dropped modes' states, rules and buttons. `bundle.json` lists only what is shipped,
with fresh hashes. `manual.en.json` is regenerated by `deploy.py::write_manual` from the
controller's own `pad_guide()`. `deploy.py --check-reference` passes on the result:

```
24 frames, 18 inferences; observation worst 0.000e+00; targets worst 2.235e-08 — this host matches the reference
```

`play.py --app king-crab.app` loads it, using the app's own controller build.

### The runtime

`sim/crab_runtime.py` is `play --app` without mjlab:

* **Robot model:** `jumper.posture`'s *compiled training model*, dumped by
  `tools/dump_robot_model.py`. It keeps the same collision bits, contact parameters and joint
  armature/damping/friction. Physics runs at 1 kHz, implicitfast, 10 solver iterations.
* **Controller:** the app's `controller.so`, loaded the way `mjrl/app_play.py` loads it. Every 5 ms tick
  it gets joint state, torques, the IMU quaternion and the body-frame gyro, plus a gamepad
  frame. It returns joint targets and per-joint gains; when a mode is due, it builds that
  mode's observation and runs its **ONNX policy**.
* **Servos:** PD torque at 1 kHz with the controller's kp/kd, clipped to the measured servo
  torque–speed curve. The thermal derate is not modelled; the clip is short.

### What is policy and what is scripted

| | how |
|---|---|
| Standing, scuttling sideways, walking, turning | **policy** — `locomotion`, then `claw_left` (five legs + carried claw) |
| Looking around (body twist), the head droop (nose-down pitch), aiming the claw at the table | **policy** — the `body_pose` command of `claw_left`, set from the pad |
| Arm held out to present an item; jaw closing / opening | **policy + controller hook** — d-pad preset and trigger, `jumper.five_foot`'s deploy hook |
| Victory wiggle | **policy** — `dance_crab` (reference-tracking dance), entered with menu + d-pad up |
| The "operator" | **scripted** — a closed-loop go-to-pose turns position and heading error into stick deflection every tick; buttons and triggers on a timeline |
| Grasp | **scripted weld on contact.** When the trigger has closed the jaw (> 0.6) at the item, a weld attaches the item to the claw at its current relative pose (no snap). The carried item is then kept upright under the grip point (the weld's relative pose is re-aimed each tick, like a gimballed tray). |
| Hand-off | **scripted** — the family hand (mocap) moves under the item; the weld moves from the claw to the hand; then the trigger opens the jaw |
| Family hands, set-down of items | **scripted** kinematic paths |
| Eyes, crown sparkles, cameras, captions | **scripted** presentation |

Measured jaw-to-grip gap when each weld engaged (from `build/run/run.json`):
butter 9 mm, rolls 11 mm, gravy 30 mm, cranberry 26 mm, phone 23 mm, drawing 3 mm,
napkin 20 mm. The remainder is mostly lateral: the go-to-pose times out before fully
converging. Any gap over 80 mm aborts the run rather than welding across the table.
`JAW_OFFSET` (where the jaws meet in the nose-down closed pose, 0.156 m ahead and 0.044 m left
of the base) was measured with the shipped `claw_left` policy, then corrected once from the
first run's gaps.

Controller transitions in the final run (the controller's own log):
`locomotion → claw_left` (button) at 0.50 s, `claw_left → locomotion` (released) at 46.93 s,
`locomotion → dance_crab` (button) at 47.57 s. Policy inferences: claw_left 2276,
dance_crab 167, locomotion 100. There were no falls into `safe`.

### The story (sim seconds → video)

The simulation runs 51.7 s of continuous physics. The edit (`render.py::edl`) maps it to
27.8 s of video. Shots can play faster than real time: the rapid-request section runs at
about 2.2×, 2.4× and 2.6× ("faster each time"). Cuts skip walking time in the slow section.

| video | beat |
|---|---|
| 0–3 s | close-up: wakes, claw up, looks around — "what is my purpose?" / "you pass butter." |
| 3–6.4 s | grabs the butter, turns to dad (its left), passes it — "oh." — holds, then droops (sad eyes, nose down) |
| 6.4–12 s | "rolls?" → mom, "gravy?" → aunt, "cranberry?" → kid, each faster; ends on tired eyes |
| 12–20.5 s | slower, warm piano: phone → grandma at the far end; the kid's hand sets a drawing down and the crab carries it to dad; a napkin to the kid (who had the cranberry) |
| 20.5–25 s | wide overhead: all five hands hold something in toward the crab; slow push-in — "my purpose… is the table. 🫶" |
| 25–27.8 s | `dance_crab` victory wiggle, crown sparkles, end card |

## Step 4 — cameras

Defined per shot in `render.py::edl` (MuJoCo free camera, smoothed within each shot, cut
between shots):

* **close-up** at the crab's eye level, with a slow drift;
* **medium tracking** at 30° down, following the mean of the crab and whatever it carries or
  whoever receives it. The azimuth puts the camera in front of the crab's facing direction
  for each hand-off.
* **wide overhead** from the head of the table at 62° down and 58° FOV, pushing in to the crab
  for the final line.

## Step 5 — edit

`edit/compose.py`:
* **Captions:** bold white Noto Sans with a black outline, in the upper third (≈21% from the
  top), with a pop-in. Emoji are Noto Color Emoji. Caption and SFX timing comes from the
  sequencer's markers mapped through the EDL; nothing is hand-timed.
* **Bottom line:** "real physics sim · open-source Jumper robot".
* **End card:** "Happy Thanksgiving / from King Crab 👑🦀 / #Jumper".
* **Audio:** `edit/audio.py`, all original and synthesised in code (see `CREDITS.md`): a plucky
  pizzicato polka for 0–12 s that crossfades into a warm felt piano, a soft wooden tap on
  each of the seven hand-offs, and glockenspiel twinkles under the sparkles.

## Reproduce

```bash
JUMPER=/path/to/jumper DESIGN=/path/to/jumper-design ./build_all.sh
```

| file | role |
|---|---|
| `sim/crab_runtime.py` | controller + ONNX + servo PD in plain MuJoCo |
| `sim/world.py` | training robot + `.map` world (attached) + `.skin` look + mocap hands, welds, eyes, sparkles |
| `sequencer.py` | the beat sheet: scripted operator, grasp/hand-off welds, hands; records every tick |
| `render.py` | EDL, cameras, eyes, sparkles, depth of field; parallel frame rendering |
| `edit/compose.py`, `edit/audio.py` | captions, end card, soundtrack, mux |
| `tools/build_skin.py`, `scene/author_scene.py`, `tools/build_app.py`, `tools/dump_robot_model.py` | the three packages and the training model |
| `packages/` | `king-crab.skin`, `king-crab-table.map`, `king-crab.app` |

## Limits, stated plainly

* The grasp is a weld, not friction: the policy brings the jaw to the item and closes it,
  and the weld makes the pickup reliable. Items do not collide with the robot; they do
  collide with the table.
* The family's hands are kinematic and do not push anything.
* The robot model has no domain randomisation and no servo thermal model. The policies were
  trained with both, so this is the nominal robot.
* `.skin` / `.map` checks prove package integrity and MuJoCo loading. They don't prove
  physical fit; `physical_fit_tested` stays false, as the protocol requires.
