"""Cut, caption, score and mux the final video.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python king_crab/edit/assemble.py

1. builds the edit decision list from the marks the simulation recorded
   (`out/sim_main.json`, `out/sim_blooper.json`): which take, which source
   time, which camera, for every output frame (30 fps, 1:1 with sim time --
   nothing is sped up or slowed down);
2. computes the camera for each frame (smoothed tracking, reset on each cut);
3. renders the frames at 1080x1920 in parallel worker processes (OSMesa gives
   one usable GL context per process);
4. composites captions (upper third) and the footer, mixes the synthesised
   audio, and encodes H.264 + AAC to out_final/king_crab_surgeon.mp4.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out"
FINAL = ROOT / "out_final" / "king_crab_surgeon.mp4"
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(HERE))

FPS = 30
W, H = 1080, 1920
WORKERS = int(os.environ.get("KC_WORKERS", "4"))


# --------------------------------------------------------------------------
# 1. the edit
# --------------------------------------------------------------------------
def marks(take):
    return json.loads((OUT / f"sim_{take}.json").read_text())["marks"]


def mk(ms, name, n=0, **kw):
    hits = [m for m in ms if m["name"] == name and all(m.get(k) == v for k, v in kw.items())]
    return hits[n]["t"]


def build_edl():
    m = marks("main")
    b = marks("blooper")
    T = {k: mk(m, "beat", beat=k) for k in ("scalpel", "forceps", "scissors", "faster", "perfect")}
    G = lambda tool, n=0: mk(m, "grasp", n, tool=tool)  # noqa: E731
    P = lambda tool, n=0: mk(m, "present", n, tool=tool)  # noqa: E731
    HO = lambda tool, n=0: mk(m, "handoff", n, tool=tool)  # noqa: E731
    C = lambda tool, n=0: mk(m, "check", n, tool=tool)  # noqa: E731
    main_end = round(mk(m, "dance") + 2.5, 2)

    cuts = [  # (start, shot) on the main take; each runs to the next start
        (0.0, "hook"),
        (T["scalpel"], "wide"), (G("scalpel") - 0.45, "close"), (P("scalpel") - 0.15, "pov"),
        (T["forceps"], "wide"), (G("forceps") - 0.35, "close"), (P("forceps") - 0.1, "pov"),
        (T["scissors"], "wide"), (G("gauze") - 0.35, "close"),
        (mk(m, "putdown", tool="gauze") + 0.35, "wide"), (G("scissors") - 0.35, "close"),
        (P("scissors") - 0.1, "pov"),
        (T["faster"], "wide"), (P("gauze") - 0.25, "pov"),
        (T["perfect"] + 0.1, "finale"),
    ]
    cuts.sort()
    shots = []
    for i, (t0, shot) in enumerate(cuts):
        t1 = cuts[i + 1][0] if i + 1 < len(cuts) else main_end
        if t1 - t0 > 0.05:
            shots.append({"take": "main", "src0": t0, "src1": t1, "shot": shot})
    # the outtake: from just before the squeeze until it has looked at camera
    g = mk(b, "grasp_physics")
    shots.append({"take": "blooper", "src0": g - 0.45, "src1": g + 1.5, "shot": "blooper"})

    # output timeline
    t = 0.0
    for s in shots:
        s["out0"] = t
        t += s["src1"] - s["src0"]
        s["out1"] = t
    total = t
    bl = shots[-1]["out0"]

    caps = [  # (out0, out1, text, style)
        (0.12, T["scalpel"], "POV: your surgical tech\nis a crab 👑🦀", "big"),
        (T["scalpel"], HO("scalpel"), "Scalpel.", "big"),
        (HO("scalpel"), HO("scalpel") + 0.85, "✅", "check"),
        (T["forceps"], HO("forceps"), "Forceps.", "big"),
        (HO("forceps"), HO("forceps") + 0.8, "✅", "check"),
        (T["scissors"], mk(m, "oops", tool="gauze"), "Scissors.", "big"),
        (mk(m, "oops", tool="gauze"), mk(m, "putdown", tool="gauze") + 0.6, "…close enough 😅", "big"),
        (HO("scissors"), HO("scissors") + 0.8, "✅", "check"),
        (T["faster"], HO("gauze"), "Faster!", "big"),
        (T["perfect"] + 0.05, main_end - 1.45, "perfect. 👏", "big"),
        (main_end - 1.45, main_end, "King Crab,\nSurgical Tech 🩺", "card"),
        (bl, total, "blooper 🙃", "big"),
        (bl + 0.2, total, "friction-only grip, no weld.\nit slipped.", "small"),
    ]
    audio = {
        "clinks": [HO(t) for t in ("scalpel", "forceps", "scissors", "gauze")],
        "oops": mk(m, "oops", tool="gauze") + 0.05,
        "fast": (T["faster"], T["perfect"]),
        "chime": T["perfect"] + 0.05,
        "music_end": main_end,
        "clatter": bl + (g - shots[-1]["src0"]) + slip_delay(),
    }
    return {"shots": shots, "captions": caps, "audio": audio, "total": total,
            "main_end": main_end}


def slip_delay():
    """When the blooper's scalpel actually moves: first time after the
    squeeze its speed exceeds 5 cm/s, read from the recorded trajectory."""
    import mujoco

    z = np.load(OUT / "sim_blooper.npz")
    model = mujoco.MjModel.from_binary_path(str(OUT / "model_blooper.mjb"))
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "or_scalpel")
    qa = model.jnt_qposadr[model.body_jntadr[b]]
    p = z["qpos"][:, qa:qa + 3]
    t = z["t"]
    g = mk(marks("blooper"), "grasp_physics")
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    idx = np.nonzero((t[1:] > g) & (v > 0.05))[0]
    return float(t[idx[0] + 1] - g) if len(idx) else 0.4


# --------------------------------------------------------------------------
# 2. cameras
# --------------------------------------------------------------------------
def frame_jobs(edl):
    import mujoco
    from render import Shots, Take

    model = mujoco.MjModel.from_binary_path(str(OUT / "model_main.mjb"))
    takes = {k: Take(k, model) for k in ("main", "blooper")}
    sh = Shots(model)
    jobs = []
    n = int(round(edl["total"] * FPS))
    cur = None
    for i in range(n):
        to = i / FPS
        s = next((s for s in edl["shots"] if s["out0"] <= to < s["out1"]), edl["shots"][-1])
        if s is not cur:
            sh.reset()
            cur = s
        ts = s["src0"] + (to - s["out0"])
        d = takes[s["take"]].set(ts)
        u = (to - s["out0"]) / max(1e-6, s["out1"] - s["out0"])
        cam = getattr(sh, s["shot"])(d, ts, u)
        jobs.append({"i": i, "take": s["take"], "t": ts,
                     "cam": [list(map(float, cam.lookat)), cam.distance, cam.azimuth,
                             cam.elevation, cam.fovy]})
    return jobs


# --------------------------------------------------------------------------
# 3. render (worker processes)
# --------------------------------------------------------------------------
def worker(job_file: str, k: int, n: int):
    import mujoco
    from camera import Cam, Renderer
    from PIL import Image
    from render import Take

    jobs = json.loads(Path(job_file).read_text())[k::n]
    model = mujoco.MjModel.from_binary_path(str(OUT / "model_main.mjb"))
    takes = {t: Take(t, model) for t in ("main", "blooper")}
    r = Renderer(model, W, H)
    fdir = OUT / "frames"
    for j in jobs:
        d = takes[j["take"]].set(j["t"])
        la, dist, az, el, fov = j["cam"]
        img = r.render(d, Cam(la, dist, az, el, fov))
        Image.fromarray(img).save(fdir / f"{j['i']:05d}.png", compress_level=1)


# --------------------------------------------------------------------------
# 4. finish
# --------------------------------------------------------------------------
def finish(edl):
    from captions import caption, place
    from PIL import Image
    from sound import mix, write_wav

    total = edl["total"]
    wav = OUT / "mix.wav"
    write_wav(wav, mix(total, edl["audio"]))

    styles = {"big": dict(size=92, stroke=9), "check": dict(size=150, stroke=10),
              "small": dict(size=50, stroke=6), "card": dict(size=104, stroke=10)}
    cache = {}
    footer = caption("real physics sim · open-source Jumper robot", size=36, stroke=5)
    tag = caption("#Jumper", size=64, stroke=7)

    FINAL.parent.mkdir(exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav),
         "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
         "-profile:v", "high", "-movflags", "+faststart",
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",   # social-platform loudness
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", str(FINAL)],
        stdin=subprocess.PIPE)
    n = int(round(total * FPS))
    for i in range(n):
        t = i / FPS
        fr = Image.open(OUT / "frames" / f"{i:05d}.png").convert("RGBA")
        for (a, b, text, style) in edl["captions"]:
            if not (a <= t < b):
                continue
            key = (text, style)
            if key not in cache:
                cache[key] = caption(text, **styles[style])
            age = t - a
            pop = min(1.0, age / 0.12)
            scale = 0.82 + 0.18 * (1 - (1 - pop) ** 3) + (0.06 * np.sin(np.pi * pop) if style == "check" else 0)
            fade = min(1.0, (b - t) / 0.08)
            if style == "card":
                dim = Image.new("RGBA", fr.size, (8, 40, 38, int(150 * min(1, age / 0.25))))
                fr.alpha_composite(dim)
                place(fr, cache[key], 860, scale, fade)
                place(fr, tag, 1060, 1.0, min(1.0, age / 0.3))
            elif style == "small":
                place(fr, cache[key], 600, 1.0, fade)
            elif style == "check":
                place(fr, cache[key], 470, scale, fade)
            else:
                place(fr, cache[key], 400, scale, fade)
        place(fr, footer, 1600, 1.0, 0.92)
        ff.stdin.write(fr.convert("RGB").tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", FINAL, f"{FINAL.stat().st_size / 1e6:.1f} MB")


def main():
    edl = build_edl()
    (OUT / "edl.json").write_text(json.dumps(edl, indent=1, ensure_ascii=False))
    print(f"total {edl['total']:.2f}s, {len(edl['shots'])} shots")
    for s in edl["shots"]:
        print(f"  {s['out0']:6.2f}-{s['out1']:6.2f}  {s['shot']:8s} {s['take']} src {s['src0']:.2f}")
    if "--edl-only" in sys.argv:
        return
    if "--no-render" not in sys.argv:
        jobs = frame_jobs(edl)
        jf = OUT / "frame_jobs.json"
        jf.write_text(json.dumps(jobs))
        fdir = OUT / "frames"
        fdir.mkdir(exist_ok=True)
        for f in fdir.glob("*.png"):
            f.unlink()
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(jf), str(k),
                                   str(WORKERS)]) for k in range(WORKERS)]
        for p in procs:
            if p.wait() != 0:
                raise SystemExit("a render worker failed")
    finish(edl)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
