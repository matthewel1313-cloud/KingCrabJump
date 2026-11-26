# app/

- `jumper_example.app`: the policy bundle that drives every joint in the video. It is
  Jumper's shipped example bundle (`out/bundle_example/jumper.app`, commit `198686ea`,
  Apache-2.0, KingKong Robotics), **copied unmodified**: its compiled controller and ONNX
  policies run exactly as `scripts/play.py --app` runs them.
- `choreography.json`: the show as an operator plays it on that bundle: the modes, the pad
  inputs per beat, and the beat/grasp/handoff times measured in the recorded take.

The show was not packaged as a new `.app`: `scripts/deploy.py` builds bundles by
cross-compiling the Rust controller for the board, browser and desktop and converting
policies to RKNN, and nothing in this video changes a policy or the controller. The
sequencing lives in `../sim/choreography.py`.

Play the bundle by hand: `python scripts/play.py --app king_crab/app/jumper_example.app --scene king_crab/scene/operating_room.map`
