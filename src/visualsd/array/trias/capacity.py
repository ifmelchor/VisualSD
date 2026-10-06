#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CapacityRange:
    lwin_range: np.ndarray
    slowmax_range: np.ndarray
    lambda_range: np.ndarray
    # bordes por lambda_min
    by_lambda: pd.DataFrame
 
    def __post_init__(self):
        self.lwin_range    = np.asarray(self.lwin_range, dtype=int)
        self.slowmax_range = np.asarray(self.slowmax_range, dtype=float)
        self.lambda_range  = np.asarray(self.lambda_range, dtype=float)
 
    def __repr__(self):
        w = self.lwin_range
        s = self.slowmax_range
        l = self.lambda_range
        return (f"<CapacityRange lwin {w.min()}..{w.max()} ({len(w)}), slowmax {s.min():.3g}..{s.max():.3g} ({len(s)}), lambda_min {l.min():g}..{l.max():g} ({len(l)})>")


def n_capable(qhist):
    """n_capable[q] = tríadas con >= q compañeros de cuarteto (q = 0..N-3)."""
    qhist = np.clip(np.asarray(qhist), 0, None)
    return np.cumsum(qhist[..., ::-1], axis=-1)[..., ::-1]


class TRIASCapacity:
    """
    Barrido de capacidad sobre lwin x lambda_min x slowmax.
    """

    def __init__(self, trias, rng, qhist):

        self.trias = trias
        self.range = rng
        self.qhist = np.asarray(qhist, dtype=np.int32)

        expected = (len(rng.lwin_range), len(rng.lambda_range), len(rng.slowmax_range), trias.array.nsta - 2)

        if self.qhist.shape != expected:
            raise ValueError(f"qhist {self.qhist.shape} no coincide con la malla y las estaciones {expected}")

    @property
    def lwin(self):
        return self.range.lwin_range
 
    @property
    def lam(self):
        return self.range.lambda_range
 
    @property
    def smax(self):
        return self.range.slowmax_range

    @property
    def fmin_eff(self):
        return np.array([self.trias.fmin_eff(w) for w in self.lwin])

    def __repr__(self):
        return (f"<TRIASCapacity {len(self.trias.ids)} est. ntriads={self.ntriads}\n lwin={self.lwin.min()}..{self.lwin.max()} ({len(self.lwin)})\n lambda_min={len(self.lam)} slowmax={self.smax.min():g}..{self.smax.max():g} ({len(self.smax)})>")

    @property
    def valid_lwin(self):
        return self.fmin_eff < self.trias.fmax

    @property
    def qcap(self):
        return self.trias.qcap

    @property
    def ntriads(self):
        return self.trias.ntriads

    @property
    def n_capable(self):
        """(nlwin, nlam, nsmax, N-2): tríadas con >= q compañeros; 0 en lwin inválidas."""
        return n_capable(self.qhist)

    def at(self, lwin, lambda_min, slowmax):
        """qhist de una configuración del barrido."""
        def idx(arr, v, name):
            i = np.flatnonzero(np.isclose(arr, v))
            if not len(i):
                raise KeyError(f"{name}={v} no está en el barrido: {list(arr)}")
            return i[0]
        return self.qhist[idx(self.lwin, lwin, "lwin"), idx(self.lam, lambda_min, "lambda_min"), idx(self.smax, slowmax, "slowmax")]

    def table(self, q_min):
        if not 0 <= q_min <= self.qcap:
            raise ValueError(f"q_min={q_min} fuera de 0..{self.qcap}")

        nc = self.n_capable
        a, b, c = (g.ravel() for g in np.meshgrid(*(np.arange(n) for n in nc.shape[:3]), indexing="ij"))
        df = pd.DataFrame(dict(lwin=self.lwin[a], lwin_s=self.lwin[a] / self.trias.fs, lambda_min=self.lam[b], slowmax=self.smax[c], fmin_eff=self.fmin_eff[a], n_admissible=nc[a, b, c, 0], n_capable=nc[a, b, c, q_min]))
        df = df[self.valid_lwin[a]]

        return df.sort_values(["n_capable", "lwin"], ascending=[False, True]).reset_index(drop=True)

    def _qh(self):
        return np.clip(self.qhist, 0, None).astype(float)       # lwin inválidas -> 0 disponibles

    def _p_cond(self, q):
        """P(q_tau >= q | tríada disponible), por celda."""
        QH = self._qh()
        nav, n = QH.sum(-1), QH[..., q:].sum(-1)
        return np.divide(n, nav, out=np.zeros_like(n), where=nav > 0)

    def _pav(self):
        """% de C(N,3) disponibles (forma + los tres pares con B > 0), por celda."""
        return 100.0 * self._qh().sum(-1) / self.trias.ntriads

    def _smax_limit(self, p, level):
        """Mayor slowmax que aún cumple p >= level"""
        ok = p >= level
        idx = np.where(ok.any(-1), ok.shape[-1] - 1 - np.argmax(ok[..., ::-1], axis=-1), -1)
        return np.where(idx >= 0, self.smax[np.clip(idx, 0, None)], np.nan)

    def _check_idx(self, lwin_ref, lam_ref):
        """lwin_ref, lam_ref: índices en lwin_range y lambda_range."""
        if not -len(self.lwin) <= lwin_ref < len(self.lwin):
            raise IndexError(f"lwin_ref={lwin_ref} fuera de 0..{len(self.lwin) - 1}")

        if not -len(self.lam) <= lam_ref < len(self.lam):
            raise IndexError(f"lam_ref={lam_ref} fuera de 0..{len(self.lam) - 1}")

        if not self.valid_lwin[lwin_ref]:
            raise ValueError(f"lwin_range[{lwin_ref}] = {self.lwin[lwin_ref]} no resuelve la banda")

        return int(lwin_ref) % len(self.lwin), int(lam_ref) % len(self.lam)

    def plot(self, lwin_ref=0, lam_ref=0, q_list=None, level=0.85, n_list=(25, 50, 75), fig=None, show=True):

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.gridspec import GridSpec

        q_list = list(range(1, self.qcap + 1)) if q_list is None else [q for q in q_list if 1 <= q <= self.qcap]

        iL, iLam = self._check_idx(lwin_ref, lam_ref)
        twin = self.lwin / self.trias.fs
        v    = self.valid_lwin
        pav  = self._pav()
        s_lo, s_hi = float(self.smax.min()), float(self.smax.max())
        smax_top = s_hi - 1e-9

        qcol = dict(zip(q_list, plt.get_cmap("Oranges")(np.linspace(0.45, 0.95, max(len(q_list), 1)))))

        ncol = dict(zip(n_list, plt.get_cmap("Blues")(np.linspace(0.45, 0.9, max(len(n_list), 1)))))

        def draw(ax, xs, curves, ncurves, xlab):
            drawn = at_top = 0
            for n, y in ncurves:                   # cuántas tríadas quedan disponibles (fondo)
                y = np.where(y >= smax_top, np.nan, y)
                ax.plot(xs, y, color=ncol[n], lw=0.9, ls=(0, (5, 2)), marker="o", ms=2, zorder=2)
            for q, y in curves:
                # si el límite toca el techo de la malla solo dice ">= slowmax máximo": no se dibuja
                at_top += int(np.sum(y >= smax_top))
                y = np.where(y >= smax_top, np.nan, y)
                drawn += int(np.isfinite(y).sum())
                ax.plot(xs, y, color=qcol[q], lw=1.2, marker="o", ms=2.5, zorder=3)
            if drawn == 0:
                msg = (f"límite >= slowmax máx. ({s_hi:g})\nampliar el rango de slowmax" if at_top
                       else f"ningún punto con P >= {level:g}")
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_xlim(xs.min(), xs.max())
            ax.set_ylim(s_lo, s_hi)
            ax.set_xlabel(xlab)
            ax.set_ylabel(r"$s_\mathrm{max}$  [s/km]")
            ax.grid(True, color="k", lw=0.4, alpha=0.3)
            ax.set_axisbelow(True)
 
        def empty(ax, msg, xlab):
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", color="0.45")
            ax.set_xlabel(xlab); ax.set_ylim(s_lo, s_hi); ax.set_xticks([])
 
        if len(self.smax) < 10 or v.sum() < 4:
            print(f"  >> malla gruesa (lwin válidas={int(v.sum())}, slowmax={len(self.smax)})")

        fig = fig or plt.figure(figsize=(7.2, 4.6))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.5], height_ratios=[1, 1], hspace=0.55, wspace=0.28, left=0.07, right=0.84, top=0.92, bottom=0.10)

        # (a) geometría
        ax = fig.add_subplot(gs[:, 0])
        x, y = np.asarray(self.trias.x), np.asarray(self.trias.y)
        ax.scatter(x - x.mean(), y - y.mean(), s=30, facecolor="white", edgecolor="k", lw=0.9, zorder=3)
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_aspect("equal")
        ax.grid(True, color="k", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(f"{len(x)} est., {self.ntriads} tríadas, qcap = {self.qcap}")

        # (b) plano (L/fs, slowmax) a lambda_min fijo
        ax = fig.add_subplot(gs[0, 1])
        if v.sum() > 1:
            xs = twin[v]
            cur = [(q, self._smax_limit(self._p_cond(q)[v, iLam, :], level)) for q in q_list]
            ncur = [(n, self._smax_limit(pav[v, iLam, :], n)) for n in n_list]
            draw(ax, xs, cur, ncur, r"$L/f_s$  [s]")
        else:
            empty(ax, "una sola lwin válida en el rango", r"$L/f_s$  [s]")
        ax.set_title(rf"$\lambda_\mathrm{{min}}={self.lam[iLam]:.2f}$")

        # (c) plano (lambda_min, slowmax) a L fijo
        ax = fig.add_subplot(gs[1, 1])
        if len(self.lam) > 1:
            cur = [(q, self._smax_limit(self._p_cond(q)[iL, :, :], level)) for q in q_list]
            ncur = [(n, self._smax_limit(pav[iL, :, :], n)) for n in n_list]
            draw(ax, self.lam, cur, ncur, r"$\lambda_\mathrm{min}$")
        else:
            empty(ax, "un solo lambda_min en el rango", r"$\lambda_\mathrm{min}$")
        ax.set_title(rf"$L/f_s={twin[iL]:g}$ s")

        # leyenda común a (b) y (c)
        hq = [Line2D([], [], color=qcol[q], lw=1.2, label=rf"$q_{{min}}={q}$") for q in q_list]
        hn = [Line2D([], [], color=ncol[n], lw=0.9, ls=(0, (5, 2)), label=f"{n}% disp.") for n in n_list]
        fig.legend(handles=hq + hn, loc="center left", title=f"P >= {level:g}", title_fontsize=6.5, bbox_to_anchor=(0.85, 0.5), frameon=False)

        if show:
            plt.show()
        
        return fig