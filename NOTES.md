# Notes

Build notes, checks, and known limitations for the King Crab packages.

## Package hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `skins/king-crab-gold-1.skin` | `8b22f32989707a565cb21811a932c3f832ac3f7ab1e2fd1cf90369b37cb529ee` |
| `skins/king_crab_professor.skin` | `fe46556d91c50fd696984f381050b6cd840b68bd07836e9abd725fe0a03d083d` |
| `maps/beach-volleyball-court.map` | `f534f7a90b61603894bb9c8c53faae54f2915e9c4d58d926373c222f529c49b2` |
| `maps/beach-volleyball-real-ball.map` | `6e7b6c2435f3a7eff650a35b0ce32e97eb62e9006501a303543cd2fd7bec5365` |
| `maps/king_crab_classroom.map` | `fb93f18ced9781c0ede33dbc973a1bce4ed7e72a4917ace784517f150cb89c3c` |
| `apps/king_crab_volleyball.app` | `6849223511a6e7982c390a7261dcdd98e5ded7b201e7754ff1c9b5fa79e646db` |
| `apps/king_crab_grader.app` | `ffd5a4e9cb8970d63c7d7fa695340e566ffe2207eb6be9046f04d9321d76923f` |
| `apps/king_crab_grader_alt_F.app` | `196e8e2ff8ab25b55bcad73557d3db0992158756a6b473760517b8bb6d41b906` |
| `skins/king-crab-thanksgiving.skin` | `6d6413e9cfdd0a4ee0a4c71d4f613bdfd01e0407ab7f4e1b6fbfbdb655012b28` |
| `skins/king-crab-surgeon.skin` | `4ebaae145a1cdf26bb493b7799070a4e697878740606b85e4910fd1a587e43ef` |
| `maps/king-crab-thanksgiving-table.map` | `0742da0356817e402e9935b76f12f27826e144a54aa155bcb6ad96b00d99e208` |
| `maps/king-crab-operating-room.map` | `a6e1656b75dfdc73f69caf9daa7d9354e3b42532c7cefc02fddd2066bccd1cac` |
| `apps/king-crab-thanksgiving.app` | `5553348f5a33a79b445792a937e3af174bead56a48a10819bfbea4887ac62033` |
| `apps/king-crab-surgeon.app` | `e6369813a3e32b12fe1bb9d0bdfefed09f4f36783e804b52c30a571b4609933c` |

Each video's build uses one skin, one map and one app:

| Build | Skin | Map | App |
|---|---|---|---|
| Volleyball (animated ball) | `king-crab-gold-1` | `beach-volleyball-court` | `king_crab_volleyball` |
| Volleyball (real physics) | `king-crab-gold-1` | `beach-volleyball-real-ball` | `king_crab_volleyball` |
| Professor, main cut | `king_crab_professor` | `king_crab_classroom` | `king_crab_grader` |
| Professor, alt-F cut | `king_crab_professor` | `king_crab_classroom` | `king_crab_grader_alt_F` |
| Thanksgiving, "What is my purpose?" | `king-crab-thanksgiving` | `king-crab-thanksgiving-table` | `king-crab-thanksgiving` |
| Surgical Tech | `king-crab-surgeon` | `king-crab-operating-room` | `king-crab-surgeon` |

The professor's main and alt-F builds share byte-identical skin and map packages, so each is stored once.
The Thanksgiving packages are that film's own, renamed from `king-crab.skin`,
`king-crab-table.map` and `king-crab.app`. A package's identity is in its manifest, not its
filename, and the bytes are unchanged.

## Checks run before publishing

- **Skins:** `shellflow verify-package --profile robots/jumper/profile.json --mujoco` passes for both,
  against the trusted Jumper profile: complete robot with 41 links and 22 joints; URDF, MJCF and
  scene compile in MuJoCo.
- **Maps:** `verify-package --capability rigid --capability jumper --mujoco` passes for all three.
  `compose_default` passes: the King Crab skin is attached as the default robot and spawns on
  supported ground. `examples/load_skin_map.py` attaches, compiles and steps each skin on its map.
- **Apps:**
  - `scripts/deploy.py --check-reference` gives a 0.0 difference in observations and joint
    targets for all three.
  - `scripts/play.py --app` loads and runs each one in its own map.
  - The volleyball app carries English and Chinese manuals; the professor apps carry English.
- **Volleyball real ball:** in simulation the jump's claws strike the ball at 0.42 s. It peaks
  0.37 m up at x = 0.47 m, which is where the net is placed (top 0.27 m, 5 cm of clearance), and
  lands at x ≈ 0.75 m. Three repeat attempts land within 2 cm of each other. The toss is from a
  51-trial sweep (`scripts/volleyball/ball_sweep.py`).
- **Thanksgiving packages:** the skin and map pass `verify-package` (trusted profile, MuJoCo
  loading), and `load_skin_map.py` composes them. The app passes `--check-reference` and runs on
  its map in `play --app`.
- **Surgical Tech packages, rebuilt for publishing:** the film itself was made with a prototype
  appearance overlay (`kingcrab-skin-overlay/1`, which `verify-package` rejects), a
  `kk-scene-package/1` operating room without a default robot, and Jumper's shipped example
  `.app`, which isn't custom work. For this repo they were rebuilt with
  `scripts/surgeon/package/`:
  - **`king-crab-surgeon.skin`:** the overlay's crown, cap and mask, at the overlay's exact
    positions, built into the shell mesh through jumper-design's pipeline (`kk-skin-package/3`,
    verified). The mask's pleat seams use the cap blue, since the palette has 4 slots.
  - **`king-crab-operating-room.map`:** the film's own `scene.xml` as a `kk-scene-package/2`,
    with the film's spawn on the tray and the skin as default robot. It is verified, and
    `compose_default`'s spawn-support check passes.
  - **`king-crab-surgeon.app`:** the shipped bundle trimmed to the film's three modes
    (`trim_app.py`). Its policies are byte-identical to the example app's. The film's
    choreography run for 6 s with each app gave a 0.0 difference in every joint over 1,201 steps.
  The film's renderer still loads its prototype files (`scripts/surgeon/skin/`,
  `scripts/surgeon/scene/`), so the video matches the rebuilt skin closely but not pixel-for-pixel.
- **Professor sequence:** the alt-F app, re-run over that cut's 22 s, matches the main run exactly
  over 4,400 control steps (`scripts/professor/motion/alt_F_reproduction.json`).

## Known limitations

- **Board builds are untested on hardware.** `king-crab-thanksgiving.app` and `king-crab-surgeon.app`
  keep KingKong's shipped robot-board controller and `.rknn` models, byte-identical, but nothing here
  has run on a physical Jumper.
- **The other three apps are not robot-ready.** They were built with `--allow-incomplete`: they carry the
  browser (wasm) and Linux `play --app` controllers, but not the robot-board binary, the `.rknn`
  NPU models, or the Windows and macOS `play` runtimes. Each app's `bundle.json` notes list
  what's missing. Rebuild them on a machine with Docker:
  `python scripts/deploy.py --manifests <this repo>/scripts/volleyball/app/manifests.json --manifest king_crab_volleyball`
  (and likewise `scripts/professor/motion/manifests.json` for `king_crab_grader` and
  `king_crab_grader_alt_F`).
- **The apps were rebuilt for publishing.** They were rebuilt from a neutral checkout path so that
  no build-machine paths ship inside them. The contents match the originals except
  `bundle.json`, the two compiled controllers, and the alt-F app's regenerated `reference.json`.
  All three pass the reference check again.
- **Skins are display-only.** `.skin/3` packages are not printable shells, and their physical fit
  is untested (`physical_fit_tested=false`, the same as every library skin). The crown, glasses,
  piping and number have no mass or collision. `king-crab-gold-1` leaves its `author` and
  `license` fields empty.
- **Map thumbnails are native MuJoCo renders.** jumper-design's release rules ask for thumbnails
  rendered with the BE website's Web importer, but that importer is built from a private source
  tree. Each map still carries a validated be-web/1 `visual-appearance.json`.
- **The professor render shows the simulation on the skin by kinematic replay.** Jumper's Train
  side has no `.skin` loader, so the physics ran on Train's own robot model and was replayed on
  the skin robot, which has identical joints. Finger-tip positions agree to the millimetre.
- **The Thanksgiving and Surgical Tech grips are welds,** and their hands and glove are kinematic.
  Each film's own NOTES (`scripts/thanksgiving/NOTES.md`, `scripts/surgeon/NOTES.md`) states its
  limits in full.
- **What's scripted or post-produced** is listed in the README's "Policy vs. scripted" table.
