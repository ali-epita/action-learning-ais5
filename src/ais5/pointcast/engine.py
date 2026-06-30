"""Grounding engine — turns a screenshot + instruction into a click point.

In M1 this is a single deterministic call (benchmark-parity). The Try-Twice
ladder + stability gate + disambiguation candidates slot into
``_ground_try_twice`` in M2/M3 without changing the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from PIL import Image

from ..utils.logging import get_logger
from .backends.base import GroundingBackend
from .config import PointCastConfig

Point = tuple[float, float]
_log = get_logger("pointcast.engine")


@dataclass
class GroundResult:
    point: Point | None  # in grounding-image pixel coordinates
    text: str = ""
    parser: str = "none"
    badge: str = "locked"  # short status shown in the HUD
    accepted: bool = True  # did the stability gate lock it? (False → refuse-and-ask)
    candidates: list[Point] = field(default_factory=list)  # for disambiguation (M3)
    metadata: dict[str, Any] = field(default_factory=dict)


class GroundingEngine:
    def __init__(self, backend: GroundingBackend, config: PointCastConfig):
        self.backend = backend
        self.cfg = config

    def warmup(self) -> None:
        """First MLX call compiles Metal kernels (~60s); do it once at startup
        on a dummy image so the user's first real interaction is fast."""
        try:
            self.backend.load()
            self.backend.predict(Image.new("RGB", (96, 96), "white"), "warmup")
        except Exception:
            pass

    def ground(self, image: Image.Image, instruction: str) -> GroundResult:
        if self.cfg.use_try_twice:
            return self._ground_try_twice(image, instruction)
        _log.info("grounding (single call) for %r", instruction)
        out = self.backend.predict(image, instruction)
        pt = out.parsed.point
        _log.info("    point %s (parser=%s)", pt, out.parsed.parser)
        return GroundResult(
            point=pt,
            text=out.text,
            parser=out.parsed.parser,
            badge="locked" if pt else "no match",
            accepted=pt is not None,
            candidates=[pt] if pt else [],
            metadata=out.metadata,
        )

    def _ground_try_twice(self, image: Image.Image, instruction: str) -> GroundResult:
        """Full Try-Twice harness: coarse → 768 → 512 ladder + stability gate."""
        from ..tile.try_twice import TryTwiceConfig, try_twice

        cfg = TryTwiceConfig(
            crop_sizes=self.cfg.crop_sizes,
            displacement_frac=self.cfg.gate_displacement_frac,
        )
        _log.info("grounding (Try-Twice ladder %s) for %r", list(self.cfg.crop_sizes), instruction)
        r = try_twice(self.backend, image, instruction, cfg)
        for st in r.stages:
            if st.displacement is None:
                _log.info("    %-8s point=%s", st.name, st.point)
            else:
                verdict = "AGREE" if st.accepted else "disagree"
                _log.info(
                    "    %-8s point=%s  disp=%.0f <= %.0f? %s",
                    st.name, st.point, st.displacement, st.threshold, verdict,
                )
        _log.info("    result: %s  point=%s", r.badge.upper(), r.point)
        candidates = list(r.candidates)
        # Hybrid disambiguation fuel: when the gate is uncertain but the ladder
        # only produced one cluster, draw a few stochastic samples to expose
        # genuinely different candidate locations.
        if not r.accepted and len(candidates) < 2:
            candidates = self._disambiguation_candidates(image, instruction, candidates)
        return GroundResult(
            point=r.point,
            text="",
            parser="try-twice",
            badge=r.badge,
            accepted=r.accepted,
            candidates=candidates,
            metadata=r.metadata,
        )

    # ── disambiguation candidate generation (M3) ─────────────────────────────
    def _disambiguation_candidates(
        self, image: Image.Image, instruction: str, seeds: list[Point]
    ) -> list[Point]:
        """Hybrid: ladder ``seeds`` first; if too few distinct, add stochastic
        samples and cluster into distinct candidate locations."""
        _log.info(
            "    uncertain with %d seed candidate(s); drawing %d stochastic samples",
            len(seeds), self.cfg.disambig_samples,
        )
        pts = list(seeds)
        for _ in range(self.cfg.disambig_samples):
            try:
                out = self.backend.predict(
                    image, instruction, sample=True, temperature=self.cfg.disambig_temperature
                )
            except Exception:
                break
            if out.parsed.point is not None:
                pts.append(out.parsed.point)
        clusters = _cluster_points(pts, self.cfg.disambig_cluster_tol)
        _log.info("    clustered into %d distinct candidate(s)", len(clusters))
        return clusters[: self.cfg.max_candidates]


def _cluster_points(points: list[Point], tol: float) -> list[Point]:
    """Greedy spatial clustering; returns cluster centroids ordered by support."""
    clusters: list[dict[str, Any]] = []
    for p in points:
        if p is None:
            continue
        placed = False
        for c in clusters:
            cx, cy = c["centroid"]
            if hypot(p[0] - cx, p[1] - cy) <= tol:
                c["members"].append(p)
                n = len(c["members"])
                c["centroid"] = (
                    sum(m[0] for m in c["members"]) / n,
                    sum(m[1] for m in c["members"]) / n,
                )
                placed = True
                break
        if not placed:
            clusters.append({"centroid": (float(p[0]), float(p[1])), "members": [p]})
    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return [(round(c["centroid"][0], 1), round(c["centroid"][1], 1)) for c in clusters]
