"""Shot lists for each contest video. Times are worked out from the captures
(see README): jump apex/contact at sim t=0.555 s, dance claws-up 19-24.5 s, etc.
"""

from __future__ import annotations

from edit import shot_duration, sim_to_out

BEAT = 0.5  # 120 bpm

VB_CTX = dict(hit_t=0.555, hit_p=[0.235, 0.0, 0.205], net_x=0.9,
              set_from=[0.15, 0.75, 0.1], set_time=1.0, shot_v=[2.6, -0.1, 1.8])
COURT_BEHIND = dict(net_x=-0.75)
T_LAND = 0.555 + 0.443  # ball lands in the sand (from the shot velocity above)


def _place(shots):
    starts, acc = [], 0.0
    for s in shots:
        starts.append(acc)
        acc += shot_duration(s)
    return starts, acc


SKIN = "gold_team"  # yellow #1 jersey + gold crown (render.SKINS)


#: Measured from captures/jump_realball (tools/ball_sweep.py best toss, BALL=
#: "0.19,0.15,0.16,0,-0.3846,1.913"): claws strike the ball at 0.42 s, it peaks
#: 0.37 m up at x=0.47 (net placed there, top 0.27 m: real 5 cm clearance) and
#: lands at x=0.75 at 0.935 s. Three attempts: peak 0.35-0.37 m, land 0.74-0.75 m.
REAL_CTX = dict(hit_t=0.42, hit_p=[0.215, -0.012, 0.1], land_t=0.935, land_p=[0.754, -0.1],
                net_x=0.47, net_top=0.27)


def volleyball_real(aspect="vertical"):
    return volleyball(aspect, real=True)


def volleyball(aspect="vertical", real=False):
    vert = aspect == "vertical"
    size = (1080, 1920) if vert else (1920, 1080)
    fov = 46 if vert else 34
    d = 1.0 if vert else 0.95

    cap_name, prop, ctx = (("jump_realball", "realball", REAL_CTX) if real
                           else ("jump_beach", "volleyball", VB_CTX))
    th = REAL_CTX["hit_t"] if real else 0.555  # sim time of the hit
    dt_ = th - 0.555  # shift every scripted time by the difference
    hook = dict(
        capture=cap_name, props=prop, ctx=ctx, fovy=fov,
        timeline=[[max(0.0, 0.05 + dt_), 0.45 + dt_, 0.75], [0.45 + dt_, 0.68 + dt_, 0.16],
                  [0.68 + dt_, 1.02 + dt_, 0.55]],
        cam=([dict(u=0, lookat=[0.07, 0, 0.17], dist=1.15 * d, az=100, el=-4),
              dict(u=0.45, lookat=[0.09, 0, 0.21], dist=0.95 * d, az=104, el=0),
              dict(u=1, lookat=[0.15, 0, 0.22], dist=1.0 * d, az=100, el=2)] if real else
             [dict(u=0, lookat=[0.1, 0, 0.17], dist=1.2 * d, az=128, el=-4),
              dict(u=0.45, lookat=[0.12, 0, 0.21], dist=0.95 * d, az=130, el=0),
              dict(u=1, lookat=[0.12, 0, 0.21], dist=1.0 * d, az=124, el=2)]),
    )
    side = dict(
        capture=cap_name, props=prop, ctx=ctx, fovy=fov - 6, follow=False,
        timeline=[[0.5 + dt_, 1.45 + dt_ - (0.3 if real else 0), 0.6]],
        cam=[dict(u=0, lookat=[0.15, 0, 0.2], dist=1.45 * d, az=272, el=-3),
             dict(u=0.3, lookat=[0.55, 0, 0.3], dist=1.4 * d, az=270, el=-2),
             dict(u=0.6, lookat=[1.0, 0, 0.24], dist=1.4 * d, az=268, el=-3),
             dict(u=1, lookat=[1.45, 0, 0.12], dist=1.3 * d, az=264, el=-5)],
    )
    pov = dict(
        capture=cap_name, props=prop, ctx=ctx, fovy=fov + 4, follow=False,
        timeline=[[0.3 + dt_, 0.5 + dt_, 0.6], [0.5 + dt_, 0.75 + dt_, 0.3],
                  [0.75 + dt_, 1.15 + dt_, 0.6]],
        cam=[dict(u=0, lookat=[0.1, 0, 0.14], dist=1.5, az=182, el=-9),
             dict(u=1, lookat=[0.25, 0, 0.16], dist=1.4, az=179, el=-7)],
    )
    samba = dict(
        capture="dance_brazilian_beach", props="court", ctx=COURT_BEHIND, fovy=fov,
        timeline=[[0.6, 4.1, 1.0]],
        cam=[dict(u=0, lookat=[0, 0, 0.2], dist=1.45 * d, az=208, el=-6),
             dict(u=1, lookat=[0, 0, 0.19], dist=1.2 * d, az=166, el=-2)],
    )
    raise_roof = dict(
        capture="dance_beach", props="court", ctx=COURT_BEHIND, fovy=fov,
        timeline=[[19.0, 23.0, 1.0]],
        cam=[dict(u=0, lookat=[-0.05, 0, 0.22], dist=1.55 * d, az=140, el=-10),
             dict(u=1, lookat=[-0.05, 0, 0.21], dist=1.2 * d, az=215, el=-1)],
    )
    high_five = dict(
        capture="gesture_paw_beach", props="court", ctx=COURT_BEHIND, fovy=fov,
        timeline=[[1.3, 4.8, 1.0]],
        cam=[dict(u=0, lookat=[0.02, 0, 0.17], dist=1.25 * d, az=184, el=0),
             dict(u=1, lookat=[0.05, 0, 0.16], dist=1.05 * d, az=178, el=3)],
    )
    shots = [hook, side, pov, samba, raise_roof, high_five]
    for sh in shots:
        sh["skin"] = SKIN
    starts, total = _place(shots)
    hit = starts[0] + sim_to_out(hook, th)
    takeoff = starts[0] + sim_to_out(hook, 0.38 + dt_)
    net_cross = starts[1] + sim_to_out(side, 0.68 if real else 0.555 + 0.2)
    land = starts[1] + sim_to_out(side, REAL_CTX["land_t"] if real else T_LAND)

    y_top = 0.13 if vert else 0.14
    y_low = 0.80 if vert else 0.84
    cap = [
        dict(t=0.0, dur=starts[1] - 0.02, text="KING CRAB ROBOT 👑", size=96, y=y_top,
             anim="none"),
        dict(t=0.0, dur=starts[1] - 0.02,
             text="REAL-PHYSICS VOLLEYBALL 🏐" if real else "vs BEACH VOLLEYBALL 🏐",
             size=60 if real else 78, y=y_top + (0.055 if vert else 0.095),
             color="y", anim="none"),
        dict(t=hit, dur=starts[1] - hit, text="BUMP!", size=150, y=y_low, color="y"),
        dict(t=net_cross - 0.1, dur=land - net_cross + 0.1, text="over the net... 👀", size=82,
             y=y_top),
        dict(t=land, dur=starts[2] - land, text="POINT, CRAB! 🦀🔥", size=104, y=y_top,
             color="y"),
        dict(t=starts[2], dur=shot_duration(pov), text="POV:\nyou're on the other team 😳",
             size=72, y=y_top),
        dict(t=starts[3], dur=shot_duration(samba), text="victory samba 💃", size=92, y=y_top),
        dict(t=starts[4], dur=shot_duration(raise_roof), text="claws UP 🙌", size=110, y=y_top,
             color="y"),
        dict(t=starts[5], dur=total - starts[5], text="high five? ✋", size=100, y=y_top),
        dict(t=starts[5] + 0.6, dur=total - starts[5] - 0.6,
             text="open-source crab robot · #Jumper", size=48, y=0.93 if vert else 0.92,
             anim="fade", box=(0, 0, 0, 140)),
    ]
    if real:
        cap.append(dict(t=starts[1], dur=starts[3] - starts[1],
                        text="no animation: the simulator decides 🔬", size=46,
                        y=0.93 if vert else 0.92, anim="fade", box=(0, 0, 0, 150)))
    sfx = [
        dict(t=max(hit - 1.1, 0), fx="whoosh", args=dict(dur=1.0), gain=0.6, pan=-0.3),
        dict(t=takeoff, fx="boing", gain=0.9),
        dict(t=hit, fx="thwack", gain=1.0),
        dict(t=starts[1] + 0.05, fx="whoosh", args=dict(dur=0.5, rise=False), gain=0.5, pan=0.3),
        dict(t=land, fx="sand_thud", gain=1.0),
        dict(t=land + 0.05, fx="crowd_cheer", args=dict(dur=2.8), gain=0.9),
        dict(t=starts[2] + sim_to_out(pov, 0.38 + dt_), fx="boing", gain=0.6),
        dict(t=starts[2] + sim_to_out(pov, th), fx="thwack", gain=0.9),
        dict(t=starts[2] + sim_to_out(pov, th + 0.065), fx="whoosh",
             args=dict(dur=0.4, rise=False),
             gain=0.7),
        dict(t=starts[5] + 1.0, fx="ding", gain=0.8),
    ]
    flashes = [dict(t=hit, dur=0.12, amount=0.4)]
    return dict(size=size, shots=shots, captions=cap, sfx=sfx, flashes=flashes,
                music=dict(drop_at=hit, outro_at=None))


VIDEOS = {"volleyball": volleyball, "volleyball_real": volleyball_real}


def get(name, aspect="vertical"):
    return VIDEOS[name](aspect)
