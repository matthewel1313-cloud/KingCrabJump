# Credits and sources

## Robot, policies, simulator
- **Jumper** robot model, trained policies, controller and the example bundle
  (`app/jumper_example.app`, copied unmodified from `out/bundle_example/jumper.app`):
  KingKong Robotics, https://github.com/KingKongRobotics/jumper, commit
  `198686eac0d188815d0943e9fee50de415edc80a`, Apache-2.0. Its own NOTICE lists its third-party parts.
- **jumper-design** (read for the `.skin` / `.map` package formats only; no files copied):
  https://github.com/KingKongRobotics/jumper-design, commit `918d192857fcea1678c2a40b339501004a94d838`.
- **MuJoCo** (Apache-2.0), **mjlab** and **rsl_rl** (vendored in Jumper), **onnxruntime** (MIT).

## Made for this video (Apache-2.0, like the rest of this folder)
- Operating-room scene, instruments, glove: `scene/scene.xml` (primitive geoms, procedural textures).
- Crown, surgical cap, mask and colours: `skin/build_skin.py` (primitive geoms).
- Control script, renderer, edit: `sim/`, `edit/`.

## Audio: all self-made, no samples
Every sound is synthesized with numpy in `edit/sound.py`:
- heart-monitor beep (1 kHz sine, 76 bpm, 104 bpm during "Faster!")
- instrument "clink" on each handoff (inharmonic metal partials + noise transient)
- music bed: an original 4-bar arpeggio (Cmaj7 – Am7 – Fmaj7 – G6) with bass, mixed at −27 dB
- "perfect" chime, the "oops" wah, the blooper clatter

No third-party or CC0 recordings were used.

## Fonts (captions)
- **Inter** Bold, by Rasmus Andersson, SIL Open Font License 1.1 (Ubuntu `fonts-inter`).
- **Noto Color Emoji**, Google, SIL Open Font License 1.1 (Ubuntu `fonts-noto-color-emoji`).

## Tools
- ffmpeg / libx264 / AAC for encoding; Pillow for caption compositing; OSMesa for headless rendering.
