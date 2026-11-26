# Credits

## Music and sound effects: all original

Every sound in both videos was synthesized from scratch in code for this project. There are no
samples, loops, stock libraries or third-party recordings, so there is nothing to license and no
CC0 material was needed.

| Video | Score | Sound effects | Source |
|---|---|---|---|
| Volleyball (both cuts) | An original 120 bpm tropical groove: a steel-pan melody over a I–V–vi–IV progression, marimba, bass and synthesized drums | A riser, whoosh, boing, ball "thwack", sand thud, small-crowd cheer and ding | [`scripts/volleyball/music.py`](scripts/volleyball/music.py) (numpy only) |
| Professor grades midterm (both cuts) | An original pentatonic pluck-and-bass groove | A whoosh, claw click, scan sweep, pen pops, sad trombone for the F, sparkle for the A+ | [`scripts/professor/post/audio.py`](scripts/professor/post/audio.py) (numpy only) |
| "What is my purpose?" (Thanksgiving) | "Pass the Butter Polka" (Karplus-Strong pizzicato, oom-pah bass, woodblock, 132 bpm), crossfading into "The Table" (an additive felt-piano voice, I–V6–vi–IV in F, 72 bpm) | A wooden tap on each of the seven hand-offs, glockenspiel twinkles under the sparkles | [`scripts/thanksgiving/edit/audio.py`](scripts/thanksgiving/edit/audio.py); dedicated to the public domain under CC0-1.0 |
| King Crab, Surgical Tech | An original 4-bar arpeggio bed (Cmaj7–Am7–Fmaj7–G6) with bass | A heart-monitor beep (faster during "Faster!"), instrument clinks, a "perfect" chime, an "oops" wah, the blooper clatter | [`scripts/surgeon/edit/sound.py`](scripts/surgeon/edit/sound.py) (numpy only) |

None of the films use third-party recordings. The Thanksgiving concept is a general parody nod and
uses no characters, dialogue, music or sound from any show.

## Robot, motions and tools

- **Jumper robot, meshes, trained policies, deployment controller, `shellflow` / `mjscene` tooling:**
  [KingKong Robotics](https://github.com/KingKongRobotics/jumper), Apache-2.0. They are used
  unchanged; see [NOTICE](NOTICE).
- **Physics and rendering:** [MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0),
  through Jumper's vendored [mjlab](https://github.com/mujocolab/mjlab) and
  [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
- **Video encoding:** FFmpeg.

## Fonts

- Inter Display / Inter Bold (SIL OFL 1.1): volleyball and Surgical Tech captions.
- Noto Sans Bold (SIL OFL 1.1): Thanksgiving captions.
- DejaVu Sans (Bitstream Vera licence): professor captions.
- Noto Color Emoji (SIL OFL 1.1): emoji in both.

## Everything else

The skin designs, maps, textures (chalkboard, midterm pages, desk, book covers, the Thanksgiving
dining room and the kid's hand-turkey drawing, the operating-room drape), scripts, camera work and
edits are original work for this project. Each film's own credits are in `scripts/*/CREDITS.md`.
