"""Experiment: where does the right claw reach with each arm preset?"""
PRE_ROLL = 3.0
def pad(t):
    p = {}
    if 1.0 <= t < 1.2: p["buttons"] = {"RB"}                 # enter claw_right
    if 5 <= t < 8: p["hat_y"] = -1                           # dpad_up: arm out, thumb up
    if 9 <= t < 12: p["hat_y"] = 1                           # dpad_down: arm out, thumb down
    if 13 <= t < 16: p["hat_x"] = -1                         # dpad_left: arm out, web up
    if 17 <= t < 21: p["hat_y"] = -1                         # arm out again...
    if 18 <= t < 21: p["rt"] = 1.0                           # ...and close the claw
    if 21 <= t < 24: p["ry"] = 1.0                           # pull back: nose up (pitch)
    return p
