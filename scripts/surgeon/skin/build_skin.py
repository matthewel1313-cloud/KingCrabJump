"""Build `king_crab_surgeon.skin`: a display-only appearance overlay for Jumper.

**What this is, and what it is not.** jumper-design's `kk-skin-package/3` is a
whole assembled robot (MJCF + URDF + meshes from its print/assembly pipeline)
whose validator allows only `visual_overrides` -- recolours of baseline visual
geoms -- and rejects any added geometry. A crown, a cap and a mask are added
geometry, so this cannot be an honest `/3` package. It is a small overlay with
its own schema id, `kingcrab-skin-overlay/1`, that reuses `/3`'s
`visual_overrides` vocabulary (`{link, geom, rgba}`) and adds `accessories`:
primitive geoms attached to robot links, always contype=0, conaffinity=0,
density=0 (no collision, no mass, no inertia change). `sim/harness.py`'s
`apply_skin` loads it into the robot spec. Jumper's Train repo has no `.skin`
loader of its own (docs/WORKFLOWS.md), so a loader had to be written either way.

    python king_crab/skin/build_skin.py      # writes king_crab/skin/king_crab_surgeon.skin
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

WHITE = [0.90, 0.91, 0.93, 1.0]
LIGHT_GRAY = [0.74, 0.76, 0.80, 1.0]
MID_GRAY = [0.62, 0.65, 0.69, 1.0]
CLAW_RED = [0.85, 0.08, 0.05, 1.0]   # the baseline red, kept on purpose
GOLD = [0.96, 0.74, 0.16, 1.0]
GEM = [0.80, 0.05, 0.25, 1.0]
CAP_BLUE = [0.36, 0.62, 0.92, 1.0]
CAP_BAND = [0.28, 0.52, 0.85, 1.0]
MASK = [0.97, 0.98, 1.0, 1.0]
MASK_PLEAT = [0.86, 0.90, 0.95, 1.0]
STRAP = [0.90, 0.92, 0.95, 1.0]


def overrides() -> list[dict]:
    out = [
        {"link": "upper_shell_link", "geom": "upper_shell_link_visual", "rgba": WHITE},
        {"link": "base_link", "geom": "base_link_visual", "rgba": MID_GRAY},
    ]
    for side in ("LF", "RF"):
        out += [
            {"link": f"{side}_shoulder_link", "geom": f"{side}_shoulder_link_visual", "rgba": WHITE},
            {"link": f"{side}_upper_arm_link", "geom": f"{side}_upper_arm_link_visual", "rgba": LIGHT_GRAY},
            # palm + finger stay red: the claws must read on screen
        ]
    for leg in ("LM", "RM", "LR", "RR"):
        out += [
            {"link": f"{leg}_hip_link", "geom": f"{leg}_hip_link_visual", "rgba": WHITE},
            {"link": f"{leg}_thigh_link", "geom": f"{leg}_thigh_link_visual", "rgba": LIGHT_GRAY},
            {"link": f"{leg}_calf_link", "geom": f"{leg}_calf_link_visual", "rgba": WHITE},
        ]
    return out


def accessories() -> list[dict]:
    """Primitive visual geoms in `base_link`'s frame (x forward, z up).

    Shell: x -0.112..0.094, y +-0.091, top z 0.072. Display: front-top,
    x 0.066..0.091. ToF/camera "mouth": x ~0.09, z 0.008..0.026.
    """
    acc = []
    # --- blue bouffant surgical cap over the rear/top of the shell
    acc.append(dict(name="cap_dome", link="base_link", type="ellipsoid",
                    size=[0.080, 0.078, 0.026], pos=[-0.030, 0.0, 0.066], rgba=CAP_BLUE))
    acc.append(dict(name="cap_band", link="base_link", type="ellipsoid",
                    size=[0.084, 0.082, 0.010], pos=[-0.030, 0.0, 0.060], rgba=CAP_BAND))
    # --- tiny gold crown, perched jauntily on the cap
    cx, cy, cz = -0.028, 0.0, 0.095
    acc.append(dict(name="crown_band", link="base_link", type="cylinder",
                    size=[0.024, 0.007], pos=[cx, cy, cz], euler=[0, -8, 0], rgba=GOLD))
    import math
    for k in range(5):
        a = 2 * math.pi * k / 5
        px, py = cx + 0.021 * math.cos(a), cy + 0.021 * math.sin(a)
        acc.append(dict(name=f"crown_spike{k}", link="base_link", type="capsule",
                        size=[0.0042],
                        fromto=[px, py, cz + 0.004, cx + 0.023 * math.cos(a),
                                cy + 0.023 * math.sin(a), cz + 0.021],
                        rgba=GOLD))
        acc.append(dict(name=f"crown_gem{k}", link="base_link", type="sphere",
                        size=[0.0045], pos=[cx + 0.023 * math.cos(a), cy + 0.023 * math.sin(a),
                                            cz + 0.023], rgba=GEM if k % 2 == 0 else GOLD))
    # --- white surgical mask over the front "mouth" (ToF + camera)
    acc.append(dict(name="mask", link="base_link", type="box",
                    size=[0.004, 0.040, 0.014], pos=[0.097, 0.007, 0.017], rgba=MASK))
    for k, z in enumerate((0.010, 0.017, 0.024)):
        acc.append(dict(name=f"mask_pleat{k}", link="base_link", type="box",
                        size=[0.0042, 0.039, 0.0009], pos=[0.0972, 0.007, z], rgba=MASK_PLEAT))
    for side, y in (("l", 0.047), ("r", -0.033)):
        acc.append(dict(name=f"mask_strap_{side}", link="base_link", type="capsule",
                        size=[0.0016], fromto=[0.096, y, 0.017, 0.050, y + (0.04 if y > 0 else -0.05), 0.040],
                        rgba=STRAP))
    return acc


def build(out: Path = HERE / "king_crab_surgeon.skin", preview: Path | None = None) -> Path:
    manifest = {
        "schema": "kingcrab-skin-overlay/1",
        "id": "king-crab-surgeon",
        "title": "King Crab, Surgical Tech",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "platform": "jumper (assets/jumper/jumper.xml)",
        "display_only": True,
        "notes": ("Recolours (kk-skin-package/3 `visual_overrides` vocabulary) plus visual-only "
                  "accessories. Accessories are contype=0, conaffinity=0, density=0: no collision, "
                  "no mass. Not a kk-skin-package/3: /3 forbids added geometry."),
        "visual_overrides": overrides(),
        "accessories": accessories(),
        "files": {},
    }
    blobs = {}
    if preview is not None and preview.exists():
        data = preview.read_bytes()
        blobs["preview/preview.png"] = data
        manifest["files"]["preview/preview.png"] = {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("skin-package.json", json.dumps(manifest, indent=2))
        for name, data in blobs.items():
            z.writestr(name, data)
    return out


if __name__ == "__main__":
    import sys

    prev = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "preview.png"
    p = build(preview=prev)
    print(f"wrote {p} ({p.stat().st_size} bytes)")
