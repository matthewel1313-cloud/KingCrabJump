# King Crab — "Cart Heist" (Black Friday) — build notes

A 28.0 s vertical (1080×1920, 30 fps) MuJoCo short for the KingKong Robotics Crab Robot
Challenge, made with the open-source **Jumper** crab robot. Two shoppers fight over Cart A.
Under the cart, on its bottom rack, sits the last deal: a phone box with a "70% OFF" tag. King
Crab crab-walks in, pinches it, crab-walks away past the shoppers, and sets it in the empty
basket of a kid at the checkout, then bows. The shoppers finally look down at an empty rack.
King Crab sambas and salutes.

Final video: [`../../renders/cart_heist.mp4`](../../renders/cart_heist.mp4) (H.264 High / AAC,
1080×1920, 30 fps, 840 frames, −15.5 LUFS integrated, true peak −0.7 dBFS).

Everything on screen is a physics simulation. Nothing here implies a physical robot.

## Sources

| repo | commit | used for |
|---|---|---|
| [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper) | `198686e` | the shipped `.app` bundle (controller + ONNX policies), `play.py`, `deploy.py`, the `jumper.posture` training model |
| [KingKongRobotics/jumper-design](https://github.com/KingKongRobotics/jumper-design) | `918d192` | mjscene, `shellflow export-map` / `verify-package`, the robot profile |

The robot meshes in jumper-design are Git LFS objects that this environment's git proxy would
not serve; they were fetched from `media.githubusercontent.com` at that commit and each file's
SHA-256 was checked against its LFS pointer (41/41 matched).

## The three packages

### `maps/black_friday_store.map` (`kk-scene-package/2`)

`scene/author_scene.py` writes the mjscene `scene.json` (kept as
`scene/black_friday_store.scene.json`); `tools/build_map.sh` runs `shellflow export-map` with
King Crab's gold-crown skin as the default robot, then `verify-package --capability rigid
--capability jumper --mujoco`. Result: export ok, verify ok, native MuJoCo composition passed,
spawn (1.60, −0.55) facing −X, spawn-support rays hit the floor, 0 severe spawn contacts.
`play.py --app ../KingCrabJump/apps/cart_heist.app --scene ../KingCrabJump/maps/black_friday_store.map`
loads both and the crab stands at the spawn.

* **Store aisle at full human scale.** Gondola shelves both sides with "BLACK FRIDAY DEALS"
  headers, three hanging banners (BLACK FRIDAY / DOORBUSTERS), BLACK FRIDAY posters on both end
  walls, a 70% OFF endcap, warm ceiling panels, a glossy tile floor, and a checkout lane
  (counter, belt, register, lit "3 CHECKOUT" sign). Two warm key spots cast shadows.
* **Cart A** at the origin. Its bottom rack's top is **6.0 cm** off the floor. **The deal** is
  a free rigid body on that rack: a phone box 14 × 8 × 3.2 cm, **0.15 kg**, with a 70% OFF
  tag. The claw policy trained with payloads of 0–0.6 kg (`five_foot/claw.py::PAYLOAD_RANGE`).
* **Cart B** (empty) dresses the far end.
* **Figures:** two adult shoppers at either end of Cart A, facing each other (a tug-of-war
  pose), and a smaller kid at the checkout with an empty red basket on the floor (rim 5.5 cm).
  Each is two static bodies (body + head) built from primitives. They are mannequins, not
  simulated humans.
* **Collision bits:** the box is on bit 2048, as are the rack plate, the basket and an invisible
  floor slab, so the box rests on the rack and lands in the basket under physics. The robot's
  bits never include 2048, so it can't snag on the cart or the basket. `tools/check_clearance.py`
  shows the choreography keeps them apart anyway (below).
* Every texture is generated in code. Signs are thin panels turned to face the viewer, because
  MuJoCo maps a 2D texture onto a box's local ±Z faces only.

### The skin: `skins/king-crab-gold-1.skin`, reused unchanged

King Crab's gold crown and #1 jersey from the volleyball film. It passes `verify-package
--profile robots/jumper/profile.json --mujoco` and is bundled inside the map as its default
robot. The renderer applies its display layer (colours, and the shell meshes with the crown) to
the physics robot. Visual geoms only; collision, inertia and joints stay the training model's.
The glowing eyes are what the robot's display shows (`sim/world.py::_eyes`), a presentation layer.

### `apps/cart_heist.app`

`tools/build_app.py` trims Jumper's shipped bundle (`out/bundle_example/jumper.app`) to the six
modes this film is allowed: `locomotion` (walk), `claw_left` (the claw grasp policy,
`jumper.five_foot`), `gesture_bow`, `gesture_hello`, `gesture_salute`, `dance_brazilian`. **No
policy was trained or retrained and no controller rebuilt.** Checks:

* `deploy.py --check-reference`: 24 frames, 18 inferences, observation and target worst error
  0.0, "this host matches the reference".
* All 29 kept files (controller builds, ONNX/RKNN, contracts, recordings, reference) are
  byte-identical to the shipped bundle.
* The whole film simulated with `cart_heist.app` and with the full shipped bundle gives a
  **0.0** difference in every `qpos` over all 11,679 control ticks.
* `gesture_hello` is in the app, as the brief lists it, but the final cut doesn't use it.

## The runtime

`sim/crab_runtime.py` is `play --app` without mjlab, as in the Thanksgiving film: the
`jumper.posture` *compiled training model* (dumped by `tools/dump_robot_model.py`: same collision
bits, contact parameters, joint armature/damping/friction), physics at 1 kHz implicitfast, the
app's own compiled controller (`controller.so`), each mode's ONNX policy on onnxruntime, and PD
servos at the controller's gains clipped to the measured servo torque–speed curve.

## Measured first: where the claw can reach (`tools/probe_claw.py`)

With the jaw shut, the stowed arm's jaw sits 1.7 cm (full nose-down stick) to 24 cm (full
nose-up) off the floor, about 0.21 m ahead and 0.045 m left of the base; at a level stick,
8.6 cm. Held straight out on the claw hook's d-pad preset it reaches about 0.28 m ahead. Claw
mode walks about 0.45 m/s at full stick. Those numbers set the rack height (6 cm) and the box
(top 9.2 cm).

One more measurement shaped the choreography. In `claw_left` the *right* claw walks on its
finger tip about 0.20 m ahead of the body, level with the stowed jaw and 10 cm to its right.
With the arm stowed, that foot went up through the rack plate and into the basket wall. So the
pinch and the place both use the d-pad preset (arm straight out), which puts the jaw 8 cm ahead
of that foot.

## What is policy and what is scripted

| | how |
|---|---|
| The sneak (sideways crab-walk), turning, walking with the box | **policy**: `locomotion` (`jumper.posture`), `claw_left` (`jumper.five_foot`) |
| Claw up; "snip snip"; arm held out; jaw height; the pinch; the lift; lowering over the basket; the jaw opening | **policy + the claw hook**: right-stick pitch (body pose, and the stowed shoulder follows it), the d-pad preset, the trigger closing/opening the jaw |
| The bow, the samba, the salute | **policy**: `gesture_bow`, `dance_brazilian`, `gesture_salute`, entered with the pad's own bindings (d-pad down; Menu + d-pad down; d-pad right). The samba is left early with Menu. |
| The "operator" | **scripted**: `sequencer.py` computes a gamepad frame every 5 ms tick. A closed-loop go-to-pose turns position/heading error into stick deflection (never below the policy's gait gate). A jaw servo puts the closed-jaw point on the box edge, using the pitch stick for height. Buttons and triggers follow the beat sheet. |
| Holding the box | **scripted weld while held**, as the brief allows. When the trigger has shut the jaw at the box (gap check, abort above 30 mm), a weld attaches the box where it is. While carried, the box is kept level under the grip point. Measured jaw-to-grip gap at the pinch: **3.9 mm**. |
| The place | **physics**: over the basket, the operator opens the trigger; the weld comes off as the jaw opens; the box drops 58 mm and lands in the basket (it settled 0.19 s after release, 9 mm from the target point). |
| Shoppers and kid | **scripted**: mocap figures with a sway, head bobs, a "notice" turn and a double-take; the kid looks up and hops |
| Eyes, crown sparkles, cameras, captions, music | **scripted** presentation |

Controller transitions in the final run (the controller's own log): `locomotion → claw_left`
(button) 9.02 s, `claw_left → locomotion` 34.62 s, `→ gesture_bow` 35.91 s, `→ locomotion`
44.94 s (the recording ended), `→ dance_brazilian` 45.61 s, `→ locomotion` 50.33 s (Menu),
`→ gesture_salute` 50.94 s, `→ locomotion` 57.89 s. No falls into `safe`. Inferences:
`locomotion` 2102, `claw_left` 1235, `gesture_bow` 428, `dance_brazilian` 204,
`gesture_salute` 301.

**Clearance** (`tools/check_clearance.py`: on a copy of the model the basket and the rack plate
are made collidable with the robot, and MuJoCo's own narrowphase is run on the recorded state
every 50 ms, for every robot collision geom except the carried claw arm): the legs come no
closer than **41.3 mm** to the basket and **38.6 mm** to the cart rack. Nothing interpenetrates.

## The edit (`render.py::edl`)

58.4 s of continuous physics → 28.0 s of video. No physics runs at render time; each frame
restores the recorded state.

| video | shot | sim (s) | speed |
|---|---|---|---|
| 0.0–2.0 s | hook: the tug-of-war over Cart A (the kid waits at the checkout behind) | 0.00–2.00 | 1.00× |
| 2.0–6.0 s | sneak: low angle from the floor by Cart A | 3.10–8.91 | 1.45× |
| 6.0–7.4 s | claw up, snip snip | 8.70–11.18 | 1.77× |
| 7.4–11.0 s | close-up: reach, pinch, lift | 14.44–21.45 | 1.95× |
| 11.0–16.0 s | escape, camera leading the crab past the shoppers | 21.65–27.30 | 1.13× |
| 16.0–18.6 s | the place, in profile: kid, basket, crab | 29.61–32.64 | 1.16× |
| 18.6–21.0 s | the bow | 36.49–40.49 | 1.67× |
| 21.0–24.0 s | meanwhile at Cart A: they notice ("wait…") | 36.49–39.49 | 1.00× |
| 24.0–26.3 s | samba | 46.25–48.75 | 1.09× |
| 26.3–28.0 s | salute + end card | 51.62–53.62 | 1.18× |

The bow and the notice shot show the same sim seconds: it is cross-cut "meanwhile" action.
Some shots play faster than real time (up to 1.95×) and cuts skip walking and servoing time.

Overlays (`edit/compose.py`, all timed from the sequencer's markers through the EDL): "Black
Friday be like…", "they didn't notice 👀", "Meanwhile, I'm a gentleman 👑", "wait…", the end card
"Happy Black Friday / from King Crab 👑🦀 / #Jumper", and the always-on line "real physics sim ·
open-source Jumper robot".

Audio (`edit/audio.py`, all synthesised in code): store ambience, a bickering hook, a sneaky
pizzicato for the heist, claw clicks, a bell "ding" when the box lands, a warm swell under the
bow, a comic two-note drop for "wait…", and a samba with a ta-da hit on the end card. See
`CREDITS.md`.

## Reproduce

```bash
JUMPER=/path/to/jumper DESIGN=/path/to/jumper-design ./build_all.sh
```

| file | role |
|---|---|
| `scene/author_scene.py`, `tools/build_map.sh` | the store scene and the `.map` (+ checks) |
| `tools/build_app.py` | the trimmed `.app` (+ `--check-reference`) |
| `tools/dump_robot_model.py` | the `jumper.posture` training model |
| `sim/crab_runtime.py`, `sim/world.py` | controller + ONNX + servos in plain MuJoCo; world = robot + map + skin + weld + mocap figures |
| `sequencer.py` | the beat sheet: scripted pad operator, grasp weld, release, figures; records every tick |
| `tools/probe_claw.py`, `tools/check_clearance.py` | the reach measurements and the clearance check |
| `render.py` | EDL, cameras, eyes, sparkles, depth of field |
| `edit/compose.py`, `edit/audio.py` | overlays, end card, soundtrack, mux |
| `tools/sheet.py` | contact sheets for previews |

Previews: `render.py --stills 1.0,4.5,9.0 --scale 0.4` writes stills at video times; `--scale
0.3` renders the whole cut in about 4 minutes on 4 cores.

## Limits, stated plainly

* The hold is a weld, not friction: the policy brings the jaw to the box and shuts it, and the
  weld makes the carry reliable. The box doesn't collide with the robot. It does collide with
  the rack, the basket and the floor, and the place is a real drop.
* The shoppers and the kid are kinematic mannequins; they push nothing.
* The robot model is the nominal training robot: no domain randomisation, no servo thermal model.
* The `.map` and `.skin` checks prove package integrity and MuJoCo loading, not physical fit.
  `load_skin_map.py` composes `king-crab-gold-1` onto the map (validate, attach, compile, forward).
* Rebuilding the map and the app with `build_all.sh` reproduced both byte for byte
  (map `f4faeb0e5f73cd01…`, app `399f19d74b70c62b…`).
* Board builds in the `.app` are the shipped ones, byte-identical, and untested on hardware here.
