from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# ── Calibration table  (a, b) vs dt ──────────────────────────────────────────
# Sources: Quoilin 2016 / Ayala-Gilardón 2018 / Riesen 2017
_ANCHOR = np.array(
    [
        [8760.0, 0.72, 0.85],
        [720.0, 0.78, 0.87],
        [24.0, 0.86, 0.89],
        [1.0, 0.97, 0.91],
        [0.25, 1.10, 0.93],
        [0.0833, 1.22, 0.95],
        [0.0167, 1.45, 0.97],
    ]
)
_LOG_DT = np.log(_ANCHOR[::-1, 0])
_A_TAB = _ANCHOR[::-1, 1]
_B_TAB = _ANCHOR[::-1, 2]


def _params_from_dt(dt_hours: float) -> tuple[float, float]:
    x = np.log(np.clip(dt_hours, _ANCHOR[-1, 0], _ANCHOR[0, 0]))
    return float(np.interp(x, _LOG_DT, _A_TAB)), float(np.interp(x, _LOG_DT, _B_TAB))


@dataclass
class ProfileParams:
    a: float = 0.72
    b: float = 0.85

    @classmethod
    def from_dt(cls, dt_hours: float) -> "ProfileParams":
        a, b = _params_from_dt(dt_hours)
        return cls(a=a, b=b)


class IneqRow(NamedTuple):
    """Solver-agnostic LP row (upper-bound form).

        sum( coefficients[i] * x[ indices[i] ] )  <=  rhs

    HiGHS
    -----
    h.addRow(-h.kHighsInf, row.rhs, row.indices, row.coefficients)

    scipy.linprog
    -------------
    A_ub[k, row.indices] = row.coefficients
    b_ub[k] = row.rhs
    """

    indices: list[int]
    coefficients: list[float]
    rhs: float


@dataclass
class LinearizedSCConstraint:
    """First-order Taylor linearisation of  e_self(e_pv, e_load)  around the operating point  (pv_0, load_0).

        e_self  <=  c_const  +  c_pv * e_pv  +  c_load * e_load

    Derivation
    ----------
    e_self = load * SS(r),   r = pv / load,   SS(r) = 1 - exp(-a * r^b)

        d e_self / d e_pv   = a*b*r^(b-1) * exp(-a*r^b)           = c_pv
        d e_self / d e_load = SS(r) - a*b*r^b * exp(-a*r^b)       = c_load
        c_const             = e_self_0 - c_pv*pv_0 - c_load*load_0

    Guarantee
    ---------
    SS(r) is concave  =>  linearisation >= exact everywhere.
    The SC row is therefore a valid LP upper bound when maximising e_self.

    Attributes:
    ----------
    c_const  : float   affine offset
    c_pv     : float   sensitivity w.r.t. PV energy   [Wh/Wh]
    c_load   : float   sensitivity w.r.t. load energy [Wh/Wh]
    pv_0     : float   operating-point PV   [Wh]
    load_0   : float   operating-point load [Wh]
    sc_0     : float   SC ratio at operating point
    e_self_0 : float   self-consumed energy at operating point [Wh]
    """

    c_const: float
    c_pv: float
    c_load: float
    pv_0: float
    load_0: float
    sc_0: float
    e_self_0: float

    def evaluate(self, pv: float, load: float | None = None) -> float:
        """Linear upper bound on e_self at the given (pv, load)."""
        l = load if load is not None else self.load_0
        return self.c_const + self.c_pv * pv + self.c_load * l

    def rhs_fixed_load(self, load: float | None = None) -> float:
        """RHS when load is a fixed parameter (not a solver variable).

        e_self - c_pv * e_pv  <=  rhs_fixed_load
        """
        l = load if load is not None else self.load_0
        return self.c_const + self.c_load * l

    def approx_error(self, pv: float, exact_e_self: float, load: float | None = None) -> float:
        """Signed linearisation error = linear_UB - exact.

        Always >= 0 due to concavity (positive = conservative / safe for LP).
        """
        return self.evaluate(pv, load) - exact_e_self

    def sc_row(
        self,
        idx_self: int,
        idx_pv: int,
        idx_load: int | None = None,
        load: float | None = None,
    ) -> IneqRow:
        """SC linearisation row.

        idx_load=None  ->  load is a fixed parameter (folded into RHS)
        idx_load=int   ->  load is a solver variable
        """
        if idx_load is None:
            return IneqRow(
                indices=[idx_self, idx_pv],
                coefficients=[1.0, -self.c_pv],
                rhs=self.rhs_fixed_load(load),
            )
        return IneqRow(
            indices=[idx_self, idx_pv, idx_load],
            coefficients=[1.0, -self.c_pv, -self.c_load],
            rhs=self.c_const,
        )

    def pv_bound_row(self, idx_self: int, idx_pv: int) -> IneqRow:
        """Physical bound:  e_self <= e_pv."""
        return IneqRow(
            indices=[idx_self, idx_pv],
            coefficients=[1.0, -1.0],
            rhs=0.0,
        )

    def load_bound_row(
        self,
        idx_self: int,
        idx_load: int | None = None,
        load: float | None = None,
    ) -> IneqRow:
        """Physical bound:  e_self <= e_load."""
        if idx_load is None:
            l = load if load is not None else self.load_0
            return IneqRow(indices=[idx_self], coefficients=[1.0], rhs=l)
        return IneqRow(
            indices=[idx_self, idx_load],
            coefficients=[1.0, -1.0],
            rhs=0.0,
        )

    def all_rows(
        self,
        idx_self: int,
        idx_pv: int,
        idx_load: int | None = None,
        load: float | None = None,
    ) -> list[IneqRow]:
        """Returns [sc_row, pv_bound_row, load_bound_row] – ready to loop over.

        Examples:
        --------
        # HiGHS
        for row in lc.all_rows(idx_self=2, idx_pv=0):
            h.addRow(-h.kHighsInf, row.rhs, row.indices, row.coefficients)

        # scipy.linprog
        for i, row in enumerate(lc.all_rows(idx_self=2, idx_pv=0)):
            A_ub[i, row.indices] = row.coefficients
            b_ub[i] = row.rhs
        """
        return [
            self.sc_row(idx_self, idx_pv, idx_load, load),
            self.pv_bound_row(idx_self, idx_pv),
            self.load_bound_row(idx_self, idx_load, load),
        ]


class FraunhoferSCModel:
    """Analytical PV self-consumption model -- Fraunhofer / Quoilin methodology.

    Parameters
    ----------
    baseload_wh : float
        Nominal load energy per time step [Wh].  Used as default load.
    dt : float
        Time-step width in hours  (0.25 = 15 min,  1.0 = 1 h,  8760 = 1 year).
        Calibration parameters a and b are selected automatically.
    params : ProfileParams, optional
        Override auto-calibrated parameters manually.

    Core formula  (Quoilin et al. 2016)
    ------------------------------------
    SC(r) = (1 - exp(-a*r^b)) / r      self-consumption ratio  [0, 1]
    SS(r) =  1 - exp(-a*r^b)           self-sufficiency ratio  [0, 1]
    r     = E_PV / E_load

    Calibration table  (log-linear interpolation)
    -----------------------------------------------
    dt = 8760 h  ->  a=0.72, b=0.85   annual    (Quoilin 2016)
    dt =    1 h  ->  a=0.97, b=0.91   hourly    (Ayala-Gilardón 2018)
    dt = 0.25 h  ->  a=1.10, b=0.93   15-min    (Riesen 2017)
    """

    def __init__(
        self,
        baseload_wh: float,
        dt: float = 1.0,
        params: ProfileParams | None = None,
    ):
        if baseload_wh <= 0:
            raise ValueError("baseload_wh must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        self.baseload_wh = float(baseload_wh)
        self.dt = float(dt)
        self.params = params or ProfileParams.from_dt(dt)

    # ── internal ─────────────────────────────────────────────────────────────

    def _r(self, pv, load):
        if load is None:
            load = self.baseload_wh
        pv = np.asarray(pv, float)
        load = np.asarray(load, float)
        r = np.full(np.broadcast_shapes(pv.shape, load.shape), np.inf, dtype=float)
        np.divide(pv, load, out=r, where=load > 1e-9)
        return r

    def _sc(self, r):
        r = np.asarray(r, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = np.where(r > 0, r, 0.0)
            ss = 1.0 - np.exp(-self.params.a * s**self.params.b)
            return np.clip(np.where(r > 1e-9, ss / r, 1.0), 0.0, 1.0)

    def _ss(self, r):
        r = np.asarray(r, float)
        with np.errstate(over="ignore"):
            s = np.where(r > 0, r, 0.0)
            return np.clip(1.0 - np.exp(-self.params.a * s**self.params.b), 0.0, 1.0)

    # ── public energy API ─────────────────────────────────────────────────────

    def sc_ratio(self, pv_wh, load_wh=None):
        """Self-consumption ratio  SC = E_self / E_PV  in [0, 1]."""
        out = self._sc(self._r(pv_wh, load_wh))
        return float(out) if np.ndim(out) == 0 else out

    def ss_ratio(self, pv_wh, load_wh=None):
        """Self-sufficiency ratio  SS = E_self / E_load  in [0, 1]."""
        out = self._ss(self._r(pv_wh, load_wh))
        return float(out) if np.ndim(out) == 0 else out

    def self_consumed_wh(self, pv_wh, load_wh=None):
        """Absolute self-consumed energy  E_self = SC * E_PV  [Wh]."""
        pv = np.asarray(pv_wh, float)
        out = self._sc(self._r(pv, load_wh)) * pv
        return float(out) if np.ndim(out) == 0 else out

    def grid_feed_in_wh(self, pv_wh, load_wh=None):
        """Grid feed-in  E_grid = (1 - SC) * E_PV  [Wh]."""
        pv = np.asarray(pv_wh, float)
        out = (1.0 - self._sc(self._r(pv, load_wh))) * pv
        return float(out) if np.ndim(out) == 0 else out

    # ── linearisation ─────────────────────────────────────────────────────────

    def linearize(
        self,
        pv_0: float,
        load_0: float | None = None,
    ) -> LinearizedSCConstraint:
        """First-order Taylor linearisation of e_self(e_pv, e_load) around the operating point (pv_0, load_0).

        Returns a LinearizedSCConstraint whose .all_rows() produces
        solver-agnostic IneqRow objects for HiGHS, scipy, PuLP, etc.

        The constraint is always a valid upper bound (concavity of SS(r)
        guarantees  linear >= exact  everywhere).

        Example:
        -------
        lc = model.linearize(pv_0=60.0, load_0=50.0)

        for row in lc.all_rows(idx_self=2, idx_pv=0):
            h.addRow(-h.kHighsInf, row.rhs, row.indices, row.coefficients)
        """
        if load_0 is None:
            load_0 = self.baseload_wh
        a, b = self.params.a, self.params.b
        if load_0 <= 1e-9:
            r0 = 1e9
        else:
            # For b < 1, dSS/dr contains r^(b-1) and can diverge at r=0.
            # Keep a tiny positive floor so linearization remains numerically stable.
            r0 = max(float(pv_0) / float(load_0), 1e-6)
        exp0 = float(np.exp(-a * r0**b))
        ss0 = 1.0 - exp0
        c_pv = a * b * r0 ** (b - 1) * exp0
        c_load = ss0 - a * b * r0**b * exp0
        e0 = ss0 * load_0
        return LinearizedSCConstraint(
            c_const=float(e0 - c_pv * pv_0 - c_load * load_0),
            c_pv=float(c_pv),
            c_load=float(c_load),
            pv_0=float(pv_0),
            load_0=float(load_0),
            sc_0=float(ss0 / r0) if r0 > 1e-9 else 1.0,
            e_self_0=float(e0),
        )

    def linearize_batch(
        self,
        pv_0: np.ndarray,
        load_0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized linearization of e_self(e_pv, e_load) for multiple time steps.

        Computes first-order Taylor coefficients for the constraint:
            e_self <= c_const + c_pv * e_pv + c_load * e_load

        This is a valid upper bound due to concavity of SS(r).

        Args:
            pv_0: Array of PV operating points [Wh] - shape (T,)
            load_0: Array of load operating points [Wh] - shape (T,), or None for baseload

        Returns:
            (c_const, c_pv, c_load): Tuple of coefficient arrays, each shape (T,)
        """
        pv_0 = np.asarray(pv_0, dtype=np.float64)
        if load_0 is None:
            load_0 = np.full_like(pv_0, self.baseload_wh, dtype=np.float64)
        else:
            load_0 = np.asarray(load_0, dtype=np.float64)

        a, b = self.params.a, self.params.b

        # Compute r = pv / load without invalid-divide warnings on zero load.
        r0 = np.full_like(pv_0, 1e9, dtype=np.float64)
        np.divide(pv_0, load_0, out=r0, where=load_0 > 1e-9)
        r0 = np.maximum(r0, 1e-6)  # Floor to avoid instability when b < 1

        # Compute derivatives
        with np.errstate(over="ignore"):
            exp0 = np.exp(-a * r0**b)
        ss0 = 1.0 - exp0

        c_pv = a * b * r0 ** (b - 1) * exp0
        c_load = ss0 - a * b * r0**b * exp0
        e0 = ss0 * load_0
        c_const = e0 - c_pv * pv_0 - c_load * load_0

        return c_const, c_pv, c_load

    # ── grid helpers ──────────────────────────────────────────────────────────

    def grid(
        self,
        pv_range: tuple[float, float | None] = (0.0, None),
        load_range: tuple[float, float | None] = (0.0, None),
        resolution: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute 2-D SC-ratio surface. Returns (pv_axis, load_axis, sc_grid)."""
        pm = pv_range[1] or 3.0 * self.baseload_wh
        lm = load_range[1] or 3.0 * self.baseload_wh
        pa = np.linspace(pv_range[0], pm, resolution)
        la = np.linspace(load_range[0], lm, resolution)
        PV, LOAD = np.meshgrid(pa, la)
        return pa, la, self._sc(self._r(PV, LOAD))

    def query_grid(
        self,
        pv_wh: float,
        load_wh: float,
        pv_ax: np.ndarray,
        load_ax: np.ndarray,
        sc_grid: np.ndarray,
    ) -> float:
        """Bilinear lookup on a precomputed grid (scipy-free)."""
        pi = np.interp(pv_wh, pv_ax, np.arange(len(pv_ax)))
        li = np.interp(load_wh, load_ax, np.arange(len(load_ax)))
        i0 = int(np.clip(np.floor(li), 0, sc_grid.shape[0] - 2))
        j0 = int(np.clip(np.floor(pi), 0, sc_grid.shape[1] - 2))
        di, dj = li - i0, pi - j0
        v = (
            sc_grid[i0, j0] * (1 - di) * (1 - dj)
            + sc_grid[i0, j0 + 1] * (1 - di) * dj
            + sc_grid[i0 + 1, j0] * di * (1 - dj)
            + sc_grid[i0 + 1, j0 + 1] * di * dj
        )
        return float(np.clip(v, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"FraunhoferSCModel(baseload_wh={self.baseload_wh:.1f}, "
            f"dt={self.dt}h, a={self.params.a:.3f}, b={self.params.b:.3f})"
        )
