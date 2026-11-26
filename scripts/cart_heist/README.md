# King Crab — "Cart Heist" 👑🦀🛒

A 28 s vertical MuJoCo short for the KingKong Robotics **Crab Robot Challenge**, made with the
open-source [Jumper](https://github.com/KingKongRobotics/jumper) crab robot. On Black Friday two
shoppers fight over the last deal. King Crab quietly pinches it from under the cart and gives it
to a kid who couldn't get one.

**Video:** [`../../renders/cart_heist.mp4`](../../renders/cart_heist.mp4) (1080×1920, 30 fps, H.264/AAC)

| deliverable | file |
|---|---|
| scene (`.map/2`, verified, King Crab gold-crown skin as default robot) | [`../../maps/black_friday_store.map`](../../maps/black_friday_store.map) |
| behaviour (`.app`: walk, claw grasp, bow, hello, salute, Brazilian samba; shipped policies, reference check passes) | [`../../apps/cart_heist.app`](../../apps/cart_heist.app) |
| skin (reused, verified) | [`../../skins/king-crab-gold-1.skin`](../../skins/king-crab-gold-1.skin) |
| scene + props source | [`scene/author_scene.py`](scene/author_scene.py), [`scene/black_friday_store.scene.json`](scene/black_friday_store.scene.json) |
| sequencer (beat sheet → shipped policies + grasp weld + figures) | [`sequencer.py`](sequencer.py) |
| runtime / render / edit | [`sim/`](sim), [`render.py`](render.py), [`edit/`](edit) |
| how it was built, what is policy vs scripted, measurements | [`NOTES.md`](NOTES.md) |
| credits (all music/SFX original, generated in code) | [`CREDITS.md`](CREDITS.md) |

Rebuild everything: `JUMPER=… DESIGN=… ./build_all.sh` (see NOTES.md → Reproduce).
