"""
wtb_yolo_modules.py
====================
Thin adapter layer that exposes WTB-DefectNet's backbone pieces (DSPS,
Stage1..Stage4 -- i.e. the stem + TSDB/ASA/DRFB/WGFR/MSCA stack) as
Ultralytics-parseable modules, so they can be dropped into a YOLO model
YAML exactly like Conv/C3k2/SPPF are.

Ultralytics' parse_model() builds each layer as:
    m(*(c1, *args))
where c1 is auto-inferred from the previous layer's output channels and
`args` is whatever list you wrote after the module name in the YAML. Each
class below just forwards (c1, c2) into the real block from wtb/model.py,
unchanged -- no architecture logic is duplicated or modified here.

LTCP (the classification head) is intentionally NOT wrapped here. Object
detection needs per-anchor/per-cell box+cls+objectness outputs, which is
what Ultralytics' own Detect head provides in the YAML's `head:` section.
Only the backbone (stem + 4 stages) transfers to detection.
"""

import torch.nn as nn
from wtb.model import DSPS, Stage


class WTBStem(nn.Module):
    """DSPS stem. c1=3 (RGB input), c2=64. Output stride: /4."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.m = DSPS(c1, c2)

    def forward(self, x):
        return self.m(x)


class WTBStage1(nn.Module):
    """TSDB+ASA, no downsample, no extra block. Stays at /4."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        assert c1 == c2, f"WTBStage1 doesn't downsample: expected c1==c2, got {c1}!={c2}"
        self.m = Stage(c1, c2, downsample=False, extra=None)

    def forward(self, x):
        return self.m(x)


class WTBStage2(nn.Module):
    """Transition + TSDB+ASA+DRFB. Downsamples /4 -> /8. This is P3."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.m = Stage(c1, c2, downsample=True, extra="drfb")

    def forward(self, x):
        return self.m(x)


class WTBStage3(nn.Module):
    """Transition + TSDB+ASA+WGFR. Downsamples /8 -> /16. This is P4."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.m = Stage(c1, c2, downsample=True, extra="wgfr")

    def forward(self, x):
        return self.m(x)


class WTBStage4(nn.Module):
    """Transition + TSDB+ASA+MSCA. Downsamples /16 -> /32. This is P5."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.m = Stage(c1, c2, downsample=True, extra="msca")

    def forward(self, x):
        return self.m(x)


def register_wtb_modules():
    """
    Makes WTBStem/WTBStage1-4 usable inside a YOLO model YAML, exactly
    like Conv/C3k2/SPPF are. Two things have to happen:

    1. The class names must resolve when parse_model() does
       `globals()[m]` for a YAML module-name string -- done by injecting
       them into ultralytics.nn.tasks' module namespace.

    2. parse_model() must know these are "base modules" so it performs
       its standard c1,c2 channel auto-inference (c1 = previous layer's
       output channels, c2 = the width you wrote in the YAML) before
       calling the constructor -- otherwise it defaults to treating
       unrecognized modules as channel-preserving (c2 = c1), which is
       wrong for a stem that goes 3ch -> 64ch. Ultralytics keeps that
       "which modules get c1,c2 injected" list as a frozenset literal
       local to parse_model() itself (not something exposed for
       extension), so we patch a copy of parse_model with our 5 classes
       added to that set and swap it in. The rest of the function is
       untouched -- this only affects module resolution, not training,
       loss, or the neck/head parsing logic.

    MUST be called before `YOLO("yolo11-wtbdefectnet.yaml")` is
    constructed. Safe to call more than once. If a future Ultralytics
    release changes parse_model()'s source formatting, the assertion
    below will fail loudly (rather than silently mis-building the model)
    -- in that case, register_wtb_modules() needs its `old` string
    updated to match the new source.
    """
    import ultralytics.nn.tasks as tasks
    import inspect

    tasks.WTBStem = WTBStem
    tasks.WTBStage1 = WTBStage1
    tasks.WTBStage2 = WTBStage2
    tasks.WTBStage3 = WTBStage3
    tasks.WTBStage4 = WTBStage4

    if getattr(tasks.parse_model, "_wtb_patched", False):
        return  # already patched (e.g. called twice)

    src = inspect.getsource(tasks.parse_model)
    old = "base_modules = frozenset(\n        {\n            Classify,\n            Conv,"
    new = (
        "base_modules = frozenset(\n        {\n            Classify,\n            Conv,"
        "\n            WTBStem,\n            WTBStage1,\n            WTBStage2,"
        "\n            WTBStage3,\n            WTBStage4,"
    )
    assert old in src, (
        "register_wtb_modules(): couldn't find the expected parse_model() "
        "source pattern -- your installed `ultralytics` version likely "
        "changed parse_model()'s internals. Pin `ultralytics==8.3.*` "
        "(the version this was tested against) or update the patch string."
    )
    patched_src = src.replace(old, new, 1)

    # Execute the patched function body in parse_model's own module
    # namespace so every name it references (Conv, C3k2, SPPF, LOGGER,
    # make_divisible, etc.) still resolves correctly.
    namespace = tasks.__dict__
    exec(compile(patched_src, "<wtb_patched_parse_model>", "exec"), namespace)
    patched_fn = namespace["parse_model"]
    patched_fn._wtb_patched = True
    tasks.parse_model = patched_fn
