# Credits

## Robot, policies, controller

* **Jumper** crab robot by KingKong Robotics: robot model, meshes, the shipped policy bundle
  (`locomotion`, `claw_left`/`claw_right` from `jumper.five_foot`, `dance_crab`) and its compiled
  controller. From <https://github.com/KingKongRobotics/jumper> (commit `198686eac0d1`),
  Apache-2.0 (see that repository's LICENSE and NOTICE). `packages/king-crab.app` redistributes
  a trimmed copy of the shipped bundle. Its kept files are unmodified.
* **jumper-design** by KingKong Robotics: the `.skin` / `.map` package protocol, the `shellflow`
  / `mjscene` tools that built and verified our packages, the Jumper robot profile and display
  meshes. From <https://github.com/KingKongRobotics/jumper-design> (commit `918d192857fc`),
  Apache-2.0. Both packages embed the baseline robot meshes under their original terms.
* Simulation: [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0). Policy
  inference: ONNX Runtime (MIT).

## Audio — all original, CC0

Every sound in the video is synthesised by `edit/audio.py`. There are no samples, loops or
recordings, and nothing from any show or existing song. It is dedicated to the public domain
under **CC0-1.0**.

| cue | time | what it is |
|---|---|---|
| "Pass the Butter Polka" | 0 – ~12 s | Karplus-Strong pizzicato strings, oom-pah plucked bass and woodblock at 132 bpm in C major; the melody doubles up during the rapid requests |
| "The Table" | ~12 s – end (crossfade ~11.4–13.2 s) | an additive felt-piano voice (inharmonic partials, detuned unisons, hammer thump) playing I–V6–vi–IV in F major at 72 bpm, ending on a rolled F major chord |
| tap | each of the 7 hand-offs | band-passed noise knock + 190 Hz body, ~0.1 s |
| twinkles | under the crown sparkles | glockenspiel-like sine partials on the F major pentatonic |

## Visuals — all original

* Every texture (dining-room backdrop, table and floor wood, runner, sleeves, wicker, the kid's
  hand-turkey drawing, the phone's lock screen) is generated procedurally by
  `scene/author_scene.py`.
* The King Crab appearance (crown, napkin bib, palette) and every scene object are built from
  code in this folder.
* **Fonts:** Noto Sans Bold and Noto Color Emoji (SIL Open Font License 1.1).

## Concept

The "what is my purpose?" setup is a general parody nod. The video uses no characters,
designs, dialogue, music or sound from any show. All captions are written for this video.

## Licence of this folder

The code, scene and appearance in this folder are offered under CC0-1.0 unless noted above.
The `.skin` and `.map` manifests record license `CC0-1.0` and author "King Crab (poly-videos)
contributors". Change those if you want a different name or licence: re-run `build_all.sh`
with `build_skin.py --author/--license`. Embedded Jumper assets keep their own terms
(Apache-2.0).
