# King Crab — "What is my purpose?" 👑🦀

A 27.8 s vertical MuJoCo short for the KingKong Robotics **Crab Robot Challenge**, made with
the open-source [Jumper](https://github.com/KingKongRobotics/jumper) crab robot. A crowned
Jumper passes things around a Thanksgiving table. It starts out only passing butter and ends
up passing what brings the family together.

**Video:** [`../out_final/king_crab_purpose.mp4`](../out_final/king_crab_purpose.mp4)
(1080×1920, 30 fps, H.264/AAC)

| deliverable | file |
|---|---|
| appearance (`.skin/3`, verified) | [`packages/king-crab.skin`](packages/king-crab.skin) |
| scene (`.map/2`, verified, King Crab as default robot) | [`packages/king-crab-table.map`](packages/king-crab-table.map) |
| behaviour bundle (`.app`, shipped policies trimmed to the 4 modes used; reference check passes) | [`packages/king-crab.app`](packages/king-crab.app) |
| sequencer (beat sheet → policies + welds + hands) | [`sequencer.py`](sequencer.py) |
| runtime / world / render / edit | [`sim/`](sim), [`render.py`](render.py), [`edit/`](edit) |
| how it was built, what is policy vs scripted | [`NOTES.md`](NOTES.md) |
| music & asset credits (all original / CC0) | [`CREDITS.md`](CREDITS.md) |
| "useful" blurb | [`BLURB.md`](BLURB.md) |
| Step 0 headless test clip | [`step0/test_play_app.mp4`](step0/test_play_app.mp4) |

Rebuild everything: `JUMPER=… DESIGN=… ./build_all.sh` (see NOTES.md → Reproduce).
