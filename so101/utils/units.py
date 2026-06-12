"""units.py — Joint unit conversions shared across SO-101 sim and real pipelines.

A single source of truth for the five units used across :mod:`so101_real` and
:mod:`so101_rl`:

* ``"rad"``  — *canonical* radians, i.e. the URDF / training frame
  (zero-at-home, training conventions).  All policy-facing joint values
  and ``robot.yaml::joint_limits`` are in this unit.
* ``"deg"``  — canonical degrees.  Same frame as ``rad``, just scaled.
* ``"norm"`` — per-joint fraction of the physical range in
  :math:`[-1, 1]`, mapped through ``joint_limits`` so that
  :math:`-1` is the lower bound, :math:`+1` is the upper bound and
  :math:`0` is the midpoint.  This is the action / observation space the
  trained policy operates in.
* ``"lrad"`` — *LeRobot* radians, i.e. the motor frame returned by the
  LeRobot follower.  Related to canonical radians by the per-joint linear
  calibration ``q_rad = scale * q_lrad + offset_rad``.
* ``"ldeg"`` — LeRobot degrees.  Same frame as ``lrad``, just scaled.

All conversions go through canonical radians, so the matrix of (src, dst)
pairs is implemented as ``src → rad → dst`` rather than a hand-coded
N×N table.  This keeps the per-joint formulas (``norm`` requires
``joint_limits``; ``lrad`` / ``ldeg`` require the per-joint calibration)
in exactly one place.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Union

import torch

# All supported unit identifiers.  Kept as a tuple so it doubles as the
# argparse ``choices`` list (and so callers can do ``unit in UNITS``).
UNITS: tuple[str, ...] = ("rad", "deg", "norm", "lrad", "ldeg")

ScalarLike = Union[float, int]
VectorLike = Union[torch.Tensor, Sequence[float]]


def _as_tensor(values: VectorLike, *, dtype=torch.float32) -> torch.Tensor:
    """Coerce ``values`` to a ``dtype`` tensor without copying when already one."""
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype) if values.dtype != dtype else values
    return torch.tensor(list(values), dtype=dtype)


class JointUnitConverter:
    """Per-joint conversions between ``rad`` / ``deg`` / ``norm`` / ``lrad`` / ``ldeg``.

    Constructed once per (joint list, robot.yaml) pair.  Conversions that
    require additional inputs are guarded:

    * ``norm`` requires ``lower_rad`` / ``upper_rad`` (``joint_limits``).
    * ``lrad`` / ``ldeg`` require ``lero_scale`` / ``lero_offset_rad``
      (``joint_calibration``).

    Parameters
    ----------
    joint_names:
        Names in the order they appear in every vector input/output of the
        converter.  Length defines ``n_joints``.
    lower_rad / upper_rad:
        Per-joint canonical-radian bounds from ``robot.yaml::joint_limits``.
        Required to use ``unit="norm"``.  Each entry must satisfy
        ``upper > lower``.
    lero_scale / lero_offset_rad:
        Per-joint linear map ``q_rad = scale * q_lrad + offset_rad`` from
        ``robot.yaml::joint_calibration``.  Required to use ``unit="lrad"``
        or ``unit="ldeg"``.  Unlisted joints use identity ``(1.0, 0.0)``;
        passing ``None`` omits LeRobot support entirely.  Each entry must
        satisfy ``scale != 0``.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        *,
        lower_rad: Optional[Sequence[float]] = None,
        upper_rad: Optional[Sequence[float]] = None,
        lero_scale: Optional[Sequence[float]] = None,
        lero_offset_rad: Optional[Sequence[float]] = None,
    ) -> None:
        names = list(joint_names)
        if len(names) == 0:
            raise ValueError("JointUnitConverter requires at least one joint name.")
        self._joint_names: list[str] = names
        self._n: int = len(names)

        self._lower: Optional[torch.Tensor] = None
        self._upper: Optional[torch.Tensor] = None
        if (lower_rad is None) != (upper_rad is None):
            raise ValueError(
                "lower_rad and upper_rad must be provided together (or both None)."
            )
        if lower_rad is not None and upper_rad is not None:
            lo = _as_tensor(lower_rad)
            hi = _as_tensor(upper_rad)
            if lo.shape != (self._n,) or hi.shape != (self._n,):
                raise ValueError(
                    f"lower_rad/upper_rad must each have shape ({self._n},); "
                    f"got {tuple(lo.shape)} and {tuple(hi.shape)}."
                )
            span = hi - lo
            if not torch.all(span > 0):
                raise ValueError(
                    "Every joint must have upper_rad > lower_rad; got "
                    f"lower={lo.tolist()}, upper={hi.tolist()}."
                )
            self._lower = lo
            self._upper = hi

        self._scale: Optional[torch.Tensor] = None
        self._offset: Optional[torch.Tensor] = None
        if (lero_scale is None) != (lero_offset_rad is None):
            raise ValueError(
                "lero_scale and lero_offset_rad must be provided together "
                "(or both None)."
            )
        if lero_scale is not None and lero_offset_rad is not None:
            s = _as_tensor(lero_scale)
            o = _as_tensor(lero_offset_rad)
            if s.shape != (self._n,) or o.shape != (self._n,):
                raise ValueError(
                    f"lero_scale/lero_offset_rad must each have shape ({self._n},); "
                    f"got {tuple(s.shape)} and {tuple(o.shape)}."
                )
            if torch.any(s == 0):
                raise ValueError(
                    f"lero_scale must be non-zero for every joint; got {s.tolist()}."
                )
            self._scale = s
            self._offset = o

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def lower_rad(self) -> Optional[torch.Tensor]:
        return None if self._lower is None else self._lower.clone()

    @property
    def upper_rad(self) -> Optional[torch.Tensor]:
        return None if self._upper is None else self._upper.clone()

    @property
    def lero_scale(self) -> Optional[torch.Tensor]:
        return None if self._scale is None else self._scale.clone()

    @property
    def lero_offset_rad(self) -> Optional[torch.Tensor]:
        return None if self._offset is None else self._offset.clone()

    @property
    def has_joint_limits(self) -> bool:
        return self._lower is not None

    @property
    def has_lero_calibration(self) -> bool:
        return self._scale is not None

    # ── Vector helpers (preserve device + dtype of input) ────────────────────

    def normalized_to_canonical(self, action: torch.Tensor) -> torch.Tensor:
        """``norm → rad`` for a batched action tensor (``..., n_joints``).

        Equivalent to ``lower + 0.5 * (action + 1) * (upper - lower)``.
        Does **not** clamp; callers that require a strict ``[-1, 1]`` input
        should use :meth:`to_canonical_rad` (which validates) or clamp first.
        """
        self._require_limits("norm")
        lo = self._lower.to(device=action.device, dtype=action.dtype)
        hi = self._upper.to(device=action.device, dtype=action.dtype)
        return lo + 0.5 * (action + 1.0) * (hi - lo)

    def canonical_to_normalized(self, q_rad: torch.Tensor) -> torch.Tensor:
        """``rad → norm`` for a batched canonical tensor (``..., n_joints``)."""
        self._require_limits("norm")
        lo = self._lower.to(device=q_rad.device, dtype=q_rad.dtype)
        hi = self._upper.to(device=q_rad.device, dtype=q_rad.dtype)
        return 2.0 * (q_rad - lo) / (hi - lo) - 1.0

    def lero_rad_to_canonical(self, q_lrad: torch.Tensor) -> torch.Tensor:
        """``lrad → rad`` for a batched tensor (``..., n_joints``)."""
        self._require_lero("lrad")
        s = self._scale.to(device=q_lrad.device, dtype=q_lrad.dtype)
        o = self._offset.to(device=q_lrad.device, dtype=q_lrad.dtype)
        return s * q_lrad + o

    def canonical_to_lero_rad(self, q_rad: torch.Tensor) -> torch.Tensor:
        """``rad → lrad`` for a batched tensor (``..., n_joints``)."""
        self._require_lero("lrad")
        s = self._scale.to(device=q_rad.device, dtype=q_rad.dtype)
        o = self._offset.to(device=q_rad.device, dtype=q_rad.dtype)
        return (q_rad - o) / s

    # ── Generic API ──────────────────────────────────────────────────────────

    def to_canonical_rad(
        self,
        value,
        unit: str,
        *,
        joint_index: Optional[int] = None,
        validate_norm: bool = True,
    ):
        """Convert ``value`` (in ``unit``) into canonical radians.

        * If ``joint_index`` is ``None``, ``value`` is treated as a vector
          of length ``n_joints`` (Tensor or ``Sequence[float]``) and the
          return value is a ``torch.Tensor``.
        * Otherwise ``value`` is a scalar for joint ``joint_index`` and the
          return value is a Python ``float``.

        When ``unit == "norm"`` and ``validate_norm`` is True (default),
        each input value must lie in ``[-1, 1]``; this guards command
        paths (e.g. ``run-static --unit norm``) against unsafe inputs.
        Set ``validate_norm=False`` when converting *measured* state
        (which may legitimately drift outside the limits).
        """
        self._check_unit(unit)
        if joint_index is None:
            t = _as_tensor(value)
            if t.shape != (self._n,):
                raise ValueError(
                    f"Expected a length-{self._n} vector for unit={unit!r}; "
                    f"got shape {tuple(t.shape)}."
                )
            return self._vec_to_canonical(t, unit, validate_norm=validate_norm)
        self._check_joint_index(joint_index)
        x = float(value)
        return self._scalar_to_canonical(x, unit, joint_index, validate_norm)

    def from_canonical_rad(
        self,
        q_rad,
        unit: str,
        *,
        joint_index: Optional[int] = None,
    ):
        """Inverse of :meth:`to_canonical_rad`.  Same shape contract."""
        self._check_unit(unit)
        if joint_index is None:
            t = _as_tensor(q_rad)
            if t.shape != (self._n,):
                raise ValueError(
                    f"Expected a length-{self._n} vector; got shape {tuple(t.shape)}."
                )
            return self._vec_from_canonical(t, unit)
        self._check_joint_index(joint_index)
        x = float(q_rad)
        return self._scalar_from_canonical(x, unit, joint_index)

    def convert(
        self,
        value,
        src: str,
        dst: str,
        *,
        joint_index: Optional[int] = None,
        validate_norm: bool = True,
    ):
        """Convert ``value`` from ``src`` units to ``dst`` units via canonical rad."""
        q = self.to_canonical_rad(
            value, src, joint_index=joint_index, validate_norm=validate_norm
        )
        return self.from_canonical_rad(q, dst, joint_index=joint_index)

    # ── Internals ────────────────────────────────────────────────────────────

    def _check_unit(self, unit: str) -> None:
        if unit not in UNITS:
            raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}.")

    def _check_joint_index(self, j: int) -> None:
        if not (0 <= j < self._n):
            raise IndexError(f"joint_index={j} out of range [0, {self._n}).")

    def _require_limits(self, unit: str) -> None:
        if self._lower is None or self._upper is None:
            raise ValueError(
                f"Unit {unit!r} requires joint_limits (lower_rad / upper_rad); "
                "construct JointUnitConverter with those bounds."
            )

    def _require_lero(self, unit: str) -> None:
        if self._scale is None or self._offset is None:
            raise ValueError(
                f"Unit {unit!r} requires LeRobot calibration "
                "(lero_scale / lero_offset_rad); construct JointUnitConverter "
                "with joint_calibration."
            )

    def _vec_to_canonical(
        self, t: torch.Tensor, unit: str, *, validate_norm: bool
    ) -> torch.Tensor:
        if unit == "rad":
            return t.clone()
        if unit == "deg":
            return t * (math.pi / 180.0)
        if unit == "norm":
            if validate_norm and (torch.any(t < -1.0) or torch.any(t > 1.0)):
                bad = [
                    (self._joint_names[i], float(v))
                    for i, v in enumerate(t.tolist())
                    if v < -1.0 or v > 1.0
                ]
                raise ValueError(
                    "Normalized inputs must lie in [-1, 1]; offending "
                    f"(joint, value) pairs: {bad}."
                )
            return self.normalized_to_canonical(t)
        if unit == "lrad":
            return self.lero_rad_to_canonical(t)
        if unit == "ldeg":
            return self.lero_rad_to_canonical(t * (math.pi / 180.0))
        raise AssertionError("unreachable")  # pragma: no cover

    def _vec_from_canonical(self, q_rad: torch.Tensor, unit: str) -> torch.Tensor:
        if unit == "rad":
            return q_rad.clone()
        if unit == "deg":
            return q_rad * (180.0 / math.pi)
        if unit == "norm":
            return self.canonical_to_normalized(q_rad)
        if unit == "lrad":
            return self.canonical_to_lero_rad(q_rad)
        if unit == "ldeg":
            return self.canonical_to_lero_rad(q_rad) * (180.0 / math.pi)
        raise AssertionError("unreachable")  # pragma: no cover

    def _scalar_to_canonical(
        self, v: float, unit: str, j: int, validate_norm: bool
    ) -> float:
        if unit == "rad":
            return v
        if unit == "deg":
            return math.radians(v)
        if unit == "norm":
            self._require_limits("norm")
            if validate_norm and (v < -1.0 or v > 1.0):
                raise ValueError(
                    f"Normalized input for joint {self._joint_names[j]!r} must "
                    f"lie in [-1, 1]; got {v}."
                )
            lo = float(self._lower[j])
            hi = float(self._upper[j])
            return lo + 0.5 * (v + 1.0) * (hi - lo)
        if unit == "lrad":
            self._require_lero("lrad")
            return float(self._scale[j]) * v + float(self._offset[j])
        if unit == "ldeg":
            self._require_lero("ldeg")
            return float(self._scale[j]) * math.radians(v) + float(self._offset[j])
        raise AssertionError("unreachable")  # pragma: no cover

    def _scalar_from_canonical(self, q_rad: float, unit: str, j: int) -> float:
        if unit == "rad":
            return q_rad
        if unit == "deg":
            return math.degrees(q_rad)
        if unit == "norm":
            self._require_limits("norm")
            lo = float(self._lower[j])
            hi = float(self._upper[j])
            return 2.0 * (q_rad - lo) / (hi - lo) - 1.0
        if unit == "lrad":
            self._require_lero("lrad")
            return (q_rad - float(self._offset[j])) / float(self._scale[j])
        if unit == "ldeg":
            self._require_lero("ldeg")
            q_lrad = (q_rad - float(self._offset[j])) / float(self._scale[j])
            return math.degrees(q_lrad)
        raise AssertionError("unreachable")  # pragma: no cover


# ── JointParser ─────────────────────────────────────────────────────────────

_JOINT_LIMIT_TOL: float = 1e-5
"""Float-rounding tolerance (≈ 0.001 °) used by :class:`JointParser` when
validating user-supplied targets.  Prevents spurious rejection of boundary
values that drift by a single ULP after unit conversion (e.g. ``norm → rad``)."""


class JointParser:
    """Converts user-supplied joint targets to validated, clamped canonical radians.

    Wraps a :class:`JointUnitConverter` so that per-joint unit conversion,
    joint-limit validation, and float-rounding tolerance are encapsulated.
    Call sites reduce to ``jp.parse(value, unit, joint_index=i)``.

    Parameters
    ----------
    converter:
        A :class:`JointUnitConverter` configured with joint limits.
    tol:
        Float-rounding tolerance in radians (default :data:`_JOINT_LIMIT_TOL`).
        Values that land within *tol* of a bound after conversion are silently
        clamped; values farther outside raise :class:`ValueError`.
    """

    def __init__(
        self, converter: JointUnitConverter, tol: float = _JOINT_LIMIT_TOL
    ) -> None:
        if not converter.has_joint_limits:
            raise ValueError(
                "JointParser requires a JointUnitConverter with joint limits "
                "(lower_rad / upper_rad)."
            )
        self._converter = converter
        self._tol = tol
        self._lower: list[float] = converter.lower_rad.tolist()  # type: ignore[union-attr]
        self._upper: list[float] = converter.upper_rad.tolist()  # type: ignore[union-attr]

    def parse(self, value: float, unit: str, *, joint_index: int) -> float:
        """Convert *value* (in *unit*) to canonical radians, validate, and clamp.

        Raises :class:`ValueError` if the converted value exceeds the joint
        limit by more than *tol* radians.
        """
        joint_name = self._converter.joint_names[joint_index]
        lower = self._lower[joint_index]
        upper = self._upper[joint_index]
        try:
            q = self._converter.to_canonical_rad(value, unit, joint_index=joint_index)
        except ValueError as exc:
            raise ValueError(
                f"joint[{joint_index}] ({joint_name}) = {value} ({unit}): {exc}"
            ) from exc
        q = float(q)
        q_clamped = max(lower, min(upper, q))
        if abs(q - q_clamped) > self._tol:
            raise ValueError(
                f"joint[{joint_index}] ({joint_name}) = {value} ({unit}) → {q:.4f} rad "
                f"is outside joint_limits [{lower:.4f}, {upper:.4f}]."
            )
        return q_clamped


def from_robot_config(
    joint_names: Sequence[str],
    *,
    lower_rad: Optional[Sequence[float]] = None,
    upper_rad: Optional[Sequence[float]] = None,
    joint_calibration: Optional[dict] = None,
    lero_scale: Optional[Sequence[float]] = None,
    lero_offset_rad: Optional[Sequence[float]] = None,
) -> JointUnitConverter:
    """Build a :class:`JointUnitConverter` from ``robot.yaml`` data.

    Two calling conventions are supported:

    **New (unified schema)** — pass pre-computed ``lero_scale`` and
    ``lero_offset_rad`` lists (derived from ``JointLimitEntry.lero_scale`` /
    ``lero_offset_rad`` properties)::

        from_robot_config(joint_names, lero_scale=[...], lero_offset_rad=[...])

    **Legacy** — pass the old ``joint_calibration`` dict keyed by joint name
    with ``.scale`` / ``.offset_rad`` attributes (duck-typed)::

        from_robot_config(joint_names, joint_calibration=cfg.joint_calibration)

    ``lower_rad`` / ``upper_rad`` are always optional; supply them to enable
    ``unit='norm'`` conversions.  ``lero_scale`` / ``lero_offset_rad`` take
    precedence over ``joint_calibration`` if both are supplied.
    """
    _lero_scale: Optional[list[float]] = None
    _lero_offset: Optional[list[float]] = None

    if lero_scale is not None and lero_offset_rad is not None:
        # New calling convention — pre-computed lists, one entry per joint.
        _lero_scale = [float(s) for s in lero_scale]
        _lero_offset = [float(o) for o in lero_offset_rad]
    elif joint_calibration is not None:
        # Legacy calling convention — dict keyed by joint name.
        _lero_scale = []
        _lero_offset = []
        for name in joint_names:
            entry = joint_calibration.get(name)
            if entry is None:
                _lero_scale.append(1.0)
                _lero_offset.append(0.0)
            else:
                # Support both attribute access (dataclass) and dict access.
                if hasattr(entry, "lero_scale"):
                    _lero_scale.append(float(entry.lero_scale))
                    _lero_offset.append(float(entry.lero_offset_rad))
                elif hasattr(entry, "scale"):
                    _lero_scale.append(float(entry.scale))
                    _lero_offset.append(float(entry.offset_rad))
                else:
                    _lero_scale.append(float(entry["scale"]))
                    _lero_offset.append(float(entry["offset_rad"]))

    return JointUnitConverter(
        joint_names=joint_names,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
        lero_scale=_lero_scale,
        lero_offset_rad=_lero_offset,
    )
