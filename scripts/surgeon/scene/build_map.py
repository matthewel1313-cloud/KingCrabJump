"""Build `operating_room.map` -- a kk-scene-package/1 -- from `scene.xml`.

The package is what Jumper's own importer (`scenes/package.py`) reads, the same
one `scripts/play.py --scene <file>.map` uses:

    python king_crab/scene/build_map.py            # writes king_crab/scene/operating_room.map
    python scripts/play.py --task jumper.posture --scene <repo>/king_crab/scene/operating_room.map

Layout of the archive:
    scene-package.json   manifest: schema, id, world file, ground geom, file hashes
    scene.xml            the world (no <option>, no <include>, as the importer requires)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(out: Path = HERE / "operating_room.map") -> Path:
    world = (HERE / "scene.xml").read_bytes()
    manifest = {
        "schema": "kk-scene-package/1",
        "id": "king-crab-operating-room",
        "title": "King Crab: OR instrument tray",
        "version": "1.0.0",
        "license": "Apache-2.0",
        "use": "replay",
        "requiredCapabilities": ["rigid"],
        "world": {
            "file": "scene.xml",
            "ground": {"geom": "tray_top"},
            "attach": {"prefix": "or_"},
        },
        "physics": {"timestep": 0.005, "note": "advisory; the host (Jumper) keeps its own"},
        "files": {
            "scene.xml": {"bytes": len(world), "sha256": hashlib.sha256(world).hexdigest()},
        },
        "props": [],
        "notes": (
            "Stainless instrument tray (the ground), blue sterile drape, four free-body "
            "instruments (scalpel, forceps, scissors, gauze) and a kinematic mocap "
            "glove_hand. The hand has no collision; tools are handed to it with weld "
            "constraints added by the control script."
        ),
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene-package.json", json.dumps(manifest, indent=2))
        z.writestr("scene.xml", world)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({p.stat().st_size} bytes)")
