"""Write the King Crab's Class desk diorama as an mjscene spec (scene/king_crab_classroom.scene.json).

    python scene/build_scene.py            # then: shellflow export-map scene/king_crab_classroom.scene.json ...

World frame (the .map's): metres, Z up. The spawn is the origin with heading +X, which is also
how Train's importer places it, so Train's frame and the map's are the same frame. The camera
side is +X; the chalkboard stands behind the robot at -X.

Layout is measured, not guessed: the paper stack sits where the right claw's jaw closes in the
dry run of motion/timeline.py (pad at x 0.227, y -0.488, z 0.077 at the grasp), and the top
sheet overhangs the books toward the robot so the lower jaw can get under it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The stack: two textbooks raise five sheets so the top one is at jaw height.
STACK_Y = -0.488
BOOK_EDGE_X = 0.25           # near edge of both books (towards the robot)
SHEET = (0.06, 0.045, 0.001)  # half sizes: 12 x 9 cm, 2 mm thick
BOOK_TOP = 0.0665
TOP_SHEET_NEAR_EDGE = 0.205  # 4.5 cm overhang


def quat(euler_deg):
    """XYZ (intrinsic, MuJoCo's eulerseq="xyz") degrees -> wxyz quaternion.

    Orientations are written as quaternions, never `euler`: a map's world is attached into a
    host spec (Train's environment, a renderer's robot), and an attached child's Euler angles
    are read with the *host's* compiler angle unit, which turns degrees into radians."""
    import mujoco
    import numpy as np
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians(euler_deg), "xyz")
    return [round(float(v), 7) for v in q]


def aim(origin, target):
    """Quaternion turning local -Z (a shade's opening) toward `target`."""
    import mujoco
    import numpy as np
    d = np.asarray(target, float) - np.asarray(origin, float)
    d /= np.linalg.norm(d)
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, -d)   # local +Z -> -d, so local -Z -> d
    return [round(float(v), 7) for v in q]


def along(origin, target, dist):
    import numpy as np
    d = np.asarray(target, float) - np.asarray(origin, float)
    return [round(float(v), 5) for v in np.asarray(origin) + d / np.linalg.norm(d) * dist]


def fix_orientations(obj):
    if "euler" in obj:
        obj["quat"] = quat(obj.pop("euler"))
    for child in obj.get("children", []):
        fix_orientations(child)
    return obj


def material(name, rgba, **kw):
    return {"name": name, "rgba": rgba, **kw}


def paper(k: int, x: float, y: float, z: float, yaw: float) -> dict:
    """One test paper: a thin solid box (the contact body) and a textured face (visual shell)."""
    return {
        "name": f"paper_{k}", "kind": "dynamic", "shape": "box", "size": list(SHEET),
        "pos": [round(x, 4), round(y, 4), round(z, 5)], "euler": [0, 0, yaw],
        "mass": 0.006, "material": "paper_edge", "friction": [0.7, 0.005, 0.0001], "condim": 4,
        "children": [{
            "name": f"paper_{k}_face", "shape": "mesh", "mesh": "paper_face",
            "pos": [0, 0, SHEET[2] + 0.00012], "material": "midterm", "collision": False, "mass": 1e-6,
        }, {
            "name": f"paper_{k}_back", "shape": "mesh", "mesh": "paper_back_face",
            "pos": [0, 0, -(SHEET[2] + 0.00012)], "material": "midterm_back", "collision": False, "mass": 1e-6,
        }],
    }


def spec() -> dict:
    papers = []
    jitter = [(-1.5, 0.004), (2.0, -0.003), (-0.8, 0.002), (1.2, -0.002)]
    for k in range(4):
        yaw, dy = jitter[k]
        papers.append(paper(k + 1, BOOK_EDGE_X + SHEET[0] + 0.002, STACK_Y + dy,
                            BOOK_TOP + SHEET[2] + 2 * SHEET[2] * k, yaw))
    papers.append(paper(5, TOP_SHEET_NEAR_EDGE + SHEET[0], STACK_Y,
                        BOOK_TOP + SHEET[2] + 2 * SHEET[2] * 4, 0))

    lamp_base = (0.02, -1.0)
    shade = (0.16, -0.80, 0.215)
    target = (0.30, -0.52, 0.07)

    objects = [
        # back wall and chalkboard (an easel board standing on the desk)
        {"name": "wall", "kind": "static", "shape": "box", "size": [0.01, 3.0, 0.6],
         "pos": [-0.70, -0.35, 0.6], "material": "wall_cream", "collision": False},
        {"name": "board_slab", "kind": "static", "shape": "box", "size": [0.012, 0.30, 0.155],
         "pos": [-0.43, -0.30, 0.155], "material": "board_frame"},
        {"name": "board_face", "kind": "static", "shape": "mesh", "mesh": "board_face",
         "pos": [-0.4175, -0.30, 0.162], "material": "chalkboard", "collision": False},
        {"name": "board_tray", "kind": "static", "shape": "box", "size": [0.02, 0.28, 0.004],
         "pos": [-0.40, -0.30, 0.022], "material": "board_frame"},
        {"name": "chalk_1", "kind": "static", "shape": "capsule", "size": [0.003, 0.012],
         "pos": [-0.395, -0.12, 0.029], "euler": [90, 0, 10], "rgba": [0.95, 0.94, 0.9, 1],
         "collision": False},
        {"name": "chalk_2", "kind": "static", "shape": "capsule", "size": [0.003, 0.009],
         "pos": [-0.392, -0.50, 0.029], "euler": [90, 0, -20], "rgba": [0.98, 0.85, 0.45, 1],
         "collision": False},
        # the textbooks the stack sits on
        {"name": "book_tides", "kind": "static", "shape": "box", "size": [0.085, 0.066, 0.017],
         "pos": [BOOK_EDGE_X + 0.085, STACK_Y, 0.017], "material": "book_red",
         "friction": [0.8, 0.005, 0.0001]},
        {"name": "book_shells", "kind": "static", "shape": "box", "size": [0.08, 0.06, 0.01625],
         "pos": [BOOK_EDGE_X + 0.08, STACK_Y, 0.034 + 0.01625], "euler": [0, 0, 2.5],
         "material": "book_navy", "friction": [0.8, 0.005, 0.0001]},
        *papers,
        # coffee mug (static: scenery, but solid)
        {"name": "mug", "kind": "static", "shape": "cylinder", "size": [0.033, 0.04],
         "pos": [0.42, -0.84, 0.04], "material": "ceramic",
         "children": [
             {"name": "mug_coffee", "shape": "cylinder", "size": [0.0295, 0.0008], "pos": [0, 0, 0.0402],
              "material": "coffee", "collision": False, "mass": 1e-6},
             {"name": "mug_band", "shape": "cylinder", "size": [0.0334, 0.004], "pos": [0, 0, 0.018],
              "material": "mug_red", "collision": False, "mass": 1e-6},
             {"name": "mug_handle", "shape": "mesh", "mesh": "mug_handle", "pos": [0.0, 0.03, 0.0],
              "euler": [0, 0, 90], "material": "ceramic", "collision": False, "mass": 1e-6},
         ]},
        # desk lamp, its shade aimed at the stack
        {"name": "lamp_base", "kind": "static", "shape": "cylinder", "size": [0.045, 0.006],
         "pos": [lamp_base[0], lamp_base[1], 0.006], "material": "lamp_green"},
        {"name": "lamp_arm_1", "kind": "static", "shape": "capsule", "size": [0.006],
         "fromto": [lamp_base[0], lamp_base[1], 0.012, lamp_base[0] + 0.04, lamp_base[1] + 0.05, 0.175],
         "material": "brass", "collision": False},
        {"name": "lamp_arm_2", "kind": "static", "shape": "capsule", "size": [0.005],
         "fromto": [lamp_base[0] + 0.04, lamp_base[1] + 0.05, 0.175, shade[0] - 0.01, shade[1] - 0.02, shade[2] + 0.04],
         "material": "brass", "collision": False},
        {"name": "lamp_joint", "kind": "static", "shape": "sphere", "size": [0.009],
         "pos": [lamp_base[0] + 0.04, lamp_base[1] + 0.05, 0.175], "material": "brass", "collision": False},
        {"name": "lamp_shade", "kind": "static", "shape": "mesh", "mesh": "lamp_shade",
         "pos": list(shade), "quat": aim(shade, target), "material": "lamp_green", "collision": False},
        {"name": "lamp_bulb", "kind": "static", "shape": "sphere", "size": [0.014],
         "pos": along(shade, target, 0.012), "material": "bulb", "collision": False},
        # an apple for the teacher, by the mug
        {"name": "apple", "kind": "static", "shape": "sphere", "size": [0.03],
         "pos": [0.52, -0.66, 0.03], "material": "apple_red",
         "children": [{"name": "apple_stem", "shape": "capsule", "size": [0.0025, 0.008],
                       "pos": [0, 0, 0.033], "euler": [15, 0, 0], "material": "stem", "collision": False,
                       "mass": 1e-6}]},
        # a pencil lying on the desk near the start
        {"name": "pencil", "kind": "static", "shape": "capsule", "size": [0.004, 0.07],
         "pos": [0.30, 0.10, 0.004], "euler": [0, 90, 25], "material": "pencil_yellow"},
    ]

    return {
        "name": "king-crab-classroom",
        "title": "King Crab's Class",
        "description": ("A warm classroom desk diorama at Jumper scale: a chalkboard reading King Crab's "
                        "Class, a stack of five midterm papers on two textbooks at claw height, a coffee "
                        "mug, a desk lamp and an apple. The papers are free rigid bodies with contact; "
                        "everything else is static."),
        "tags": ["indoor", "desk", "diorama", "manipulation"],
        "sky": {"preset": "indoor", "fog": False},
        "lighting": {
            "ambient": [0.30, 0.26, 0.22], "headlight": False, "shadows": True, "shadow_quality": "high",
            "lights": [
                {"name": "key", "type": "directional", "pos": [1.4, 0.4, 1.6], "target": [0.0, -0.4, 0.0],
                 "diffuse": [0.85, 0.70, 0.52], "specular": [0.2, 0.18, 0.15], "castshadow": True},
                {"name": "lamp", "type": "spot", "pos": [shade[0], shade[1], shade[2] - 0.02],
                 "target": list(target), "cutoff": 50, "exponent": 6, "diffuse": [0.9, 0.62, 0.32],
                 "specular": [0.3, 0.2, 0.1], "attenuation": [1, 0, 0], "castshadow": True},
                {"name": "fill", "type": "directional", "pos": [-0.5, 1.2, 1.0], "target": [0.0, -0.4, 0.0],
                 "diffuse": [0.22, 0.20, 0.19], "specular": [0, 0, 0], "castshadow": False},
            ],
        },
        "ground": {"type": "plane", "size": [3.0, 3.0], "material": "desk_wood", "friction": "wood"},
        "materials": [
            material("desk_wood", [1, 1, 1, 1], texture={"file": "assets/desk_wood.png"}, texrepeat=[2.5, 2.5],
                     texuniform=True, roughness=0.6, specular=0.25, shininess=0.4),
            material("wall_cream", [0.93, 0.86, 0.72, 1], roughness=0.95, specular=0.05),
            material("chalkboard", [1, 1, 1, 1], texture={"file": "assets/chalkboard.png"}, roughness=0.95,
                     specular=0.03),
            material("board_frame", [0.42, 0.26, 0.14, 1], roughness=0.7),
            material("book_red", [1, 1, 1, 1], texture={"file": "assets/book_red.png"}, roughness=0.8),
            material("book_navy", [1, 1, 1, 1], texture={"file": "assets/book_navy.png"}, roughness=0.8),
            material("midterm", [1, 1, 1, 1], texture={"file": "assets/midterm.png"}, roughness=0.9, specular=0.05),
            material("midterm_back", [1, 1, 1, 1], texture={"file": "assets/midterm_back.png"}, roughness=0.9,
                     specular=0.05),
            material("paper_edge", [0.97, 0.96, 0.92, 1], roughness=0.9),
            material("ceramic", [0.95, 0.92, 0.86, 1], roughness=0.3, specular=0.5, shininess=0.7),
            material("mug_red", [0.72, 0.12, 0.12, 1], roughness=0.3),
            material("coffee", [0.20, 0.11, 0.05, 1], roughness=0.2, specular=0.6),
            material("lamp_green", [0.10, 0.32, 0.22, 1], roughness=0.35, metallic=0.4, specular=0.5),
            material("brass", [0.78, 0.62, 0.30, 1], roughness=0.3, metallic=0.8, specular=0.7),
            material("bulb", [1.0, 0.92, 0.70, 1], emission=0.9),
            material("apple_red", [0.70, 0.06, 0.06, 1], roughness=0.35, specular=0.5),
            material("stem", [0.30, 0.20, 0.10, 1]),
            material("pencil_yellow", [0.95, 0.75, 0.15, 1], roughness=0.5),
        ],
        "meshes": [
            {"name": "board_face", "file": "assets/board_face.obj"},
            {"name": "paper_face", "file": "assets/paper_face.obj"},
            {"name": "paper_back_face", "file": "assets/paper_back_face.obj"},
            {"name": "mug_handle", "file": "assets/mug_handle.obj"},
            {"name": "lamp_shade", "file": "assets/lamp_shade.obj"},
        ],
        "objects": [fix_orientations(o) for o in objects],
        "cameras": [
            {"name": "vertical_main", "pos": [1.05, -0.40, 0.30], "target": [0.05, -0.40, 0.09], "fovy": 34},
            {"name": "hook_closeup", "pos": [0.42, 0.06, 0.17], "target": [0.0, 0.0, 0.13], "fovy": 30},
        ],
        "spawn": {"position": [0, 0, 0], "yaw": 0, "clearance": 0.2,
                  "bounds": [-0.35, -1.0, 0.0, 0.6, 0.3, 0.5]},
        "viewer": {"camera_pos": [1.05, -0.40, 0.30], "camera_target": [0.05, -0.40, 0.09]},
    }


def main():
    out = HERE / "king_crab_classroom.scene.json"
    out.write_text(json.dumps(spec(), indent=1) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
