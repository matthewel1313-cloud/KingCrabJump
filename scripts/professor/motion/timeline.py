"""King Crab Grades Your Midterm: the scripted gamepad, in video seconds.

Read by run_sequence.py. Conventions are the robot pad service's (controller/xbox.py):
Lx positive right, Ly positive *down*, Rx positive right, Ry positive *towards the operator*,
triggers 0..1, hat_y positive *down*. What each control does is the bundle's: see
manifests.json (switches) and each task's controls.yaml (sticks, triggers, d-pad).

  locomotion (jumper.posture): Lx -> lin_vel_y (right is -y), Rx first half -> body twist,
                               Ry -> pitch (back is nose up)
  claw_right (jumper.five_foot, mirrored): RT closes the claw, dpad_up holds the arm out,
                               Ry -> pitch, Lx -> lin_vel_y
  dance_crab (jumper.dance):   menu + dpad_up enters; menu alone, released, leaves
"""
import math

PRE_ROLL = 3.0          # warm start (2 s) + settle, discarded

# Beats (video seconds). Kept here so the edit can read the same numbers.
TURN = (0.9, 1.7)       # twist back to face the camera
WALK = (2.0, 5.3)       # crab-walk to the stack
CLAW_IN = 5.45          # RB: into claw_right (1 s ramp)
ARM_OUT = (6.6, 9.0)    # dpad_up: arm out over the stack
CLOSE = (7.5, 8.0)      # RT ramps to closed
GRASP = 8.0             # attach from here
LIFT = 9.0              # dpad released, RT held: the arm stows, paper up to the eyes
READ = (10.0, 13.6)     # pitch nose-up, reading
SHUFFLE = (13.7, 14.6)  # sidestep away from the stack in claw mode
PLACE_ARM = (14.7, 16.4)
LOWER = (15.0, 16.3)    # nose down while the arm is out: the paper goes down to the desk
RELEASE = 15.95         # RT let go: the paper is put down
CLAW_OUT = 16.6         # RB: back to locomotion
SHAKE = (18.0, 21.8)    # slow head shake (twist)
DANCE_IN = 22.0         # menu + up
DANCE_OUT = 27.3        # menu alone, released
END = 30.0


def ramp(t, a, b, edge=0.25):
    """1 inside [a, b], easing in and out over `edge` seconds."""
    if t <= a or t >= b:
        return 0.0
    k = min((t - a) / edge, (b - t) / edge, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * k)


def press(t, at, width=0.15):
    return at <= t < at + width


def pad(t):
    p = {"buttons": set()}
    # Hook: hold the body twisted away (ccw), then twist back to face the camera.
    if t < TURN[0]:
        p["rx"] = -0.45
    elif t < TURN[1]:
        p["rx"] = -0.45 * (1 - (t - TURN[0]) / (TURN[1] - TURN[0]))
    # Crab-walk sideways to the right (-y).
    p["lx"] = 0.19 * ramp(t, *WALK, edge=0.4)
    if press(t, CLAW_IN) or press(t, CLAW_OUT):
        p["buttons"].add("RB")
    if ARM_OUT[0] <= t < ARM_OUT[1] or PLACE_ARM[0] <= t < PLACE_ARM[1]:
        p["hat_y"] = -1
    if CLOSE[0] <= t < RELEASE:
        p["rt"] = min(1.0, (t - CLOSE[0]) / (CLOSE[1] - CLOSE[0]))
    p["ry"] = 0.8 * ramp(t, *READ, edge=0.6) - 0.5 * ramp(t, *LOWER, edge=0.5)
    p["lx"] += 0.4 * ramp(t, *SHUFFLE, edge=0.2)
    if SHAKE[0] <= t < SHAKE[1]:
        p["rx"] = -0.36 * math.sin(2 * math.pi * 0.55 * (t - SHAKE[0]))
    if press(t, DANCE_IN):
        p["buttons"].add("menu")
        p["hat_y"] = -1
    if press(t, DANCE_OUT):
        p["buttons"].add("menu")
    return p


def attach(t):
    """The top paper is held in the claw from the grasp to the release (README: kinematic attach)."""
    return "paper_5" if GRASP <= t < RELEASE else None
