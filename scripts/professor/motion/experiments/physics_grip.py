"""Experiment: the same timeline with no kinematic attach -- does the claw hold the paper by contact alone?"""
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("timeline", pathlib.Path(__file__).parents[1] / "timeline.py")
_t = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_t)
PRE_ROLL, pad = _t.PRE_ROLL, _t.pad
