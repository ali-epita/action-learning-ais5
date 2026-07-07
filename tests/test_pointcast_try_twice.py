"""CPU-only tests for the Try-Twice harness (scripted fake model, no real VLM)."""

from __future__ import annotations

from PIL import Image

from ais5.models.base import GUIModel, ModelOutput
from ais5.prompt.action import ParsedAction
from ais5.tile.try_twice import TryTwiceConfig, try_twice


class ScriptedModel(GUIModel):
    """Returns a predetermined point per call. Crop points are crop-local."""

    name = "scripted"
    param_count_b = 0.0

    def __init__(self, points):
        self.points = list(points)
        self.calls = 0

    def predict(self, image, instruction, **_):
        pt = self.points[self.calls]
        self.calls += 1
        text = f"<click>{pt[0]}, {pt[1]}</click>" if pt else "no idea"
        return ModelOutput(text=text, parsed=ParsedAction(point=pt, raw=text, parser="scripted"))


IMG = Image.new("RGB", (1000, 1000), "white")
CFG = TryTwiceConfig(crop_sizes=(768, 512), displacement_frac=0.06, min_crop=384)


def _close(a, b, tol=1.0):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def test_accept_at_768():
    # coarse (500,500); 768-crop reclick agrees -> locked at first window
    m = ScriptedModel([(500, 500), (388, 388)])
    r = try_twice(m, IMG, "x", CFG)
    assert r.accepted and r.badge == "locked (768)"
    assert _close(r.point, (504, 504))
    assert m.calls == 2  # 2 model calls
    assert r.metadata["calls"] == 2


def test_escalate_to_512():
    # 768 disagrees (jumps far), 512 reclick agrees -> locked at tighter window
    m = ScriptedModel([(500, 500), (700, 700), (330, 330)])
    r = try_twice(m, IMG, "x", CFG)
    assert r.accepted and r.badge == "locked (512)"
    assert _close(r.point, (818, 818))
    assert m.calls == 3


def test_uncertain_when_never_agrees():
    # both crops disagree -> unaccepted, refuse-and-ask
    m = ScriptedModel([(500, 500), (700, 700), (500, 500)])
    r = try_twice(m, IMG, "x", CFG)
    assert not r.accepted and r.badge == "uncertain"
    assert _close(r.point, (988, 988))  # returns the most-zoomed best estimate
    assert len(r.candidates) >= 2  # multiple distinct points -> disambiguation fuel


def test_stage1_parse_fail():
    m = ScriptedModel([None])
    r = try_twice(m, IMG, "x", CFG)
    assert r.point is None and not r.accepted and r.badge == "no match"
    assert m.calls == 1  # no wasted crop calls


class SizeAwareModel(ScriptedModel):
    """Also records the size of every image it is asked to ground."""

    def __init__(self, points):
        super().__init__(points)
        self.sizes = []

    def predict(self, image, instruction, **kw):
        self.sizes.append(image.size)
        return super().predict(image, instruction, **kw)


def test_hires_reclick_crops_native_resolution_and_maps_back():
    # Base (grounding) image is a 0.5x downscale of a 2000x2000 native capture.
    full = Image.new("RGB", (2000, 2000), "white")
    # coarse at (500,500) base; hires crop 768 native centered at (1000,1000)
    # full -> box (616,616,1384,1384); crop-local (388,388) -> full (1004,1004)
    # -> base (502,502): displacement 2.8 <= 0.06 * (768*0.5) = 23 -> locked.
    m = SizeAwareModel([(500, 500), (388, 388)])
    r = try_twice(m, IMG, "x", CFG, hires=(full, 0.5))
    assert r.accepted and r.badge == "locked (768)"
    assert _close(r.point, (502, 502))
    assert m.sizes[0] == (1000, 1000)  # coarse still sees the base image
    assert m.sizes[1] == (768, 768)  # re-click sees a true native-res crop


def test_hires_gate_threshold_uses_base_window():
    # Same geometry, but the re-click lands 40 base-px away: over the 23-px
    # threshold (0.06 * 768 * 0.5), so the ladder must NOT accept at 768.
    full = Image.new("RGB", (2000, 2000), "white")
    m = SizeAwareModel([(500, 500), (468, 388), (256, 256)])
    r = try_twice(m, IMG, "x", TryTwiceConfig(crop_sizes=(768,), displacement_frac=0.06, min_crop=384),
                  hires=(full, 0.5))
    assert not r.accepted  # 768 disagreed and the ladder is exhausted


def test_hires_ignored_when_scale_is_unity():
    # full_scale >= 1.0 means the base image IS the capture: behave exactly
    # like the non-hires path.
    m = ScriptedModel([(500, 500), (388, 388)])
    r = try_twice(m, IMG, "x", CFG, hires=(IMG, 1.0))
    assert r.accepted and _close(r.point, (504, 504))


def test_min_crop_floor():
    # crop sizes below min_crop are clamped up (avoid over-zoom)
    cfg = TryTwiceConfig(crop_sizes=(200,), displacement_frac=0.06, min_crop=384)
    m = ScriptedModel([(500, 500), (192, 192)])  # 384-crop centered at 500 -> box (308,308,692,692)
    r = try_twice(m, IMG, "x", cfg)
    # box x1 = 500-192 = 308; refined 192 local -> 500 full -> agrees with coarse
    assert _close(r.point, (500, 500)) and r.accepted


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} try-twice tests passed")
