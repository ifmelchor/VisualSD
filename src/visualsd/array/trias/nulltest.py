#!/usr/bin/env python3
# coding=utf-8

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .capacity import n_capable
from .utils import _evaluable_chi2, _parse_trios


def synthetic_noise(rng, npts, nsta, fs, fmin, fmax, npad=2000):
    sos = butter(4, [fmin, fmax], btype="band", fs=fs, output="sos")
    x   = sosfiltfilt(sos, rng.standard_normal((npts + 2 * npad, nsta)), axis=0)[npad:npad + npts]
    return x / x.std(axis=0, keepdims=True)


def surrogate_noise(rng, noise, npts, min_shift):
    """Tramo (npts, nsta) de ruido real con un inicio distinto por estación,
    separados al menos `min_shift` muestras entre sí (sin coherencia entre estaciones)."""

    L, nsta = noise.shape
    slack = (L - npts) - min_shift * (nsta - 1)
    if slack < 0:
        raise ValueError(f"registro de ruido corto: hacen falta >= {npts + min_shift * (nsta - 1)} muestras")
    starts = rng.permutation(np.sort(rng.integers(0, slack + 1, nsta)) + min_shift * np.arange(nsta))
    seg = np.column_stack([noise[s:s + npts, c] for c, s in enumerate(starts)])
    return seg / seg.std(axis=0, keepdims=True)

# estadística del null

def _window_thresholds(trios, nsta, nwin, min_vtriads, chi2_max):
    """
    (nwin, qcap): para cada ventana, el chi2 que hace falta para detectar con q_min = 1..qcap
    (inf: no detecta). cens: detectaría pero con chi2 >= chi2_max (fuera de lo que el null ve).
    """
    qcap = nsta - 3
    W = np.full((nwin, qcap), np.inf)
    cens = np.zeros((nwin, qcap), dtype=bool)
    if len(trios) == 0:
        return W, cens

    # por tríada: chi2 de sus cuartetos ordenados; con q_min = q valida si el q-ésimo < chi2_th
    T = np.sort(_evaluable_chi2(trios, nsta), axis=1)[:, :qcap]
    df = pd.DataFrame(T)
    df["w"] = trios["window_id"].to_numpy()

    # por ventana: hacen falta min_vtriads tríadas válidas -> el min_vtriads-ésimo menor
    m = min_vtriads
    if m == 1:
        g = df.groupby("w").min()
    else:
        g = df.groupby("w").agg(lambda s: np.sort(s.to_numpy())[m - 1] if len(s) >= m else np.inf)

    Wf = g.to_numpy()
    idx = g.index.to_numpy()
    cens[idx] = np.isfinite(Wf) & (Wf >= chi2_max)
    W[idx] = np.where(Wf >= chi2_max, np.inf, Wf)
    return W, cens


def _unique_quartet_chi2(trios, nsta, chi2_max):
    """Extrae los valores válidos de chi2"""

    if len(trios) == 0:
        return np.empty(0)
    
    C = _evaluable_chi2(trios, nsta)
    r, l = np.nonzero(np.isfinite(C))

    q = np.sort(np.column_stack([trios["i"].to_numpy()[r] - 1, trios["j"].to_numpy()[r] - 1, trios["k"].to_numpy()[r] - 1, l]), axis=1)

    key = trios["window_id"].to_numpy()[r].astype(np.int64)
    for c in range(4):
        key = key * nsta + q[:, c]

    _, idx = np.unique(key, return_index=True)
    vals = C[r, l][idx]
    
    return vals[vals < chi2_max]


def _threshold_at(values, n_total, target, chi2_max):
    """calcula un umbral de corte de chi2 FA <= target"""
    v = np.sort(values[np.isfinite(values)])
    n_allow = int(np.floor(target * n_total))
    
    if n_allow >= len(v):
        return chi2_max, len(v) / n_total, True
   
    th = float(v[n_allow])
    return th, np.count_nonzero(v < th) / n_total, False


def _hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _check_same(obj, codes, x, y, config, tol_km=0.001):
    """Mismas estaciones, geometría y (fmin, fmax, fs) que `obj` (null test o calibración)."""
    
    codes = list(codes)
    if codes != obj.station_codes:
        raise ValueError(f"estaciones {codes} distintas de {obj.station_codes} ({obj.id})")
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = np.asarray(obj.x), np.asarray(obj.y)
    err = np.hypot((x - x[0]) - (cx - cx[0]), (y - y[0]) - (cy - cy[0])).max()
    if err > tol_km:
        raise ValueError(f"la geometría cambió {err * 1000:.1f} m respecto de {obj.id}")
   
    p = obj.params
    fmin, fmax, fs = config
    same = ((fmin is None) == (p["fmin"] is None) and (fmin is None or abs(fmin - p["fmin"]) < 1e-9) and abs(fmax - p["fmax"]) < 1e-9 and abs(fs - p["fs"]) < 1e-9)
    
    if not same:
        raise ValueError(f"{obj.id} con (fmin, fmax, fs) = {(p['fmin'], p['fmax'], p['fs'])}, este TRIAS es {tuple(config)}")


def _config_id(meta):
    return _hash({k: v for k, v in meta.items() if k not in ("id", "created")})


class TRIASNullTest:
    """
    Estadística del null para una configuración (estaciones, banda, fs, params).
    """

    def __init__(self, meta, W, cens, chi2_edges, chi2_counts):
        # Hacemos una copia para no mutar el diccionario externo
        self.meta = dict(meta)
        
        # Assegurar arrays contiguos antes de tobytes
        self.W = np.ascontiguousarray(W, dtype=float)
        self.cens = np.ascontiguousarray(cens, dtype=bool)
        self.chi2_edges = np.asarray(chi2_edges, dtype=float)
        self.chi2_counts = np.asarray(chi2_counts, dtype=int)

    def __getattr__(self, item):
        meta = self.__dict__.get("meta", {})
        if item in meta:
            return meta[item]
        raise AttributeError(f"'{type(self).__name__}' no tiene '{item}'")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.__dict__.get("meta", {})))

    @property
    def n_mc(self):
        return self.W.shape[0]

    @property
    def nwin(self):
        return self.W.shape[1]

    @property
    def qcap(self):
        return self.W.shape[2]

    @property
    def n_total(self):
        return self.n_mc * self.nwin

    @property
    def n_capable(self):
        return n_capable(self.structure["qhist"])

    def __repr__(self):
        p = self.params
        return (f"<TRIASNullTest {self.id} {len(self.station_codes)} est. {p['fmin']}-{p['fmax']} Hz @ {p['fs']:g} sps\n lwin={p['lwin']} lambda_min={p['lambda_min']:.3g} slowmax={p['slowmax']:.3g}\n {self.noise['source']} n_mc={self.n_mc} ({self.n_total} ventanas)>")

    def check(self, codes, x, y, config):
        _check_same(self, codes, x, y, config)

    # lo que no depende de target_fa
    def fa(self, q_min, chi2_th):
        """FA del null con (q_min, chi2_th): fracción de ventanas que detectarían."""
        return np.count_nonzero(self.W[:, :, q_min - 1] < chi2_th) / self.n_total

    def fa_curve(self, q_min, th_grid=None):
        """(th_grid, FA(th)) para ese q_min."""
        chi2_max = self.params["chi2_max"]
        th = np.logspace(-2, np.log10(chi2_max), 200) if th_grid is None else np.asarray(th_grid)
        v = np.sort(self.W[:, :, q_min - 1].ravel())
        v = v[np.isfinite(v)]
        return th, np.searchsorted(v, th, side="left") / self.n_total

    # la decisión: target_fa -> chi2_th
    def threshold(self, q_min, target_fa):
        """chi2_th para (q_min, target_fa): FA = target_fa, o menor si hay techo (chi2_max)."""
        if not 1 <= q_min <= self.qcap:
            raise ValueError(f"q_min={q_min} fuera de 1..{self.qcap}")
        if self.n_capable[q_min] == 0:
            raise ValueError(f"q_min={q_min}: ninguna tríada lo alcanza con esta configuración")
        th, _, _ = _threshold_at(self.W[:, :, q_min - 1].ravel(), self.n_total, target_fa, self.params["chi2_max"])
        return float(th)

    def trias_kw(self, q_min, target_fa):
        """kwargs de trias() para producción con (q_min, target_fa)."""
        p = self.params
        return dict(min_rho=p["min_rho"], max_pval=p["max_pval"], lambda_min=p["lambda_min"],
                    upsample=int(p["upsample"]), min_vtriads=int(p["min_vtriads"]), min_quartets=int(q_min),
                    chi2_th=self.threshold(q_min, target_fa), **self.extra_kw)

    def family(self, target_fa, n_boot=200, seed=0):
        """Un chi2_th por q_min con FA <= target_fa, IC bootstrap sobre realizaciones, techo y capacidad."""
        chi2_max = self.params["chi2_max"]
        rng = np.random.default_rng(seed)
        ncap = self.n_capable
        out = []
        for q in range(1, self.qcap + 1):
            per = self.W[:, :, q - 1]
            th, fa, ceil = _threshold_at(per.ravel(), self.n_total, target_fa, chi2_max)
            boots = [_threshold_at(per[rng.integers(0, self.n_mc, self.n_mc)].ravel(), self.n_total, target_fa, chi2_max)[0] for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
            fa_cens = self.cens[:, :, q - 1].sum() / self.n_total       # cota inferior de lo que no se ve
            out.append(dict(q=q, chi2_th=float(th), fa=float(fa), ceiling=bool(ceil), ceiling_by_chi2_max=bool(ceil and fa + fa_cens >= target_fa), fa_above_chi2_max=float(fa_cens), ci_lo=float(lo), ci_hi=float(hi), n_capable=int(ncap[q])))
        return out

    def plot(self, target_fa=None, q_min=None, axes=None):
        """
        (a) densidad + acumulada del chi2 de cuartetos bajo H0.
        (b) FA vs chi2_th para cada q_min. Con target_fa: la línea del objetivo y el punto de cada
            curva (la familia iso-FA); con q_min: ese punto destacado.
        """
        import matplotlib.pyplot as plt
 
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        axA, axB = axes
 
        e, c = self.chi2_edges, self.chi2_counts.astype(float)
        mid = np.sqrt(e[:-1] * e[1:])
        dens = c / np.diff(np.log10(e))
        axA.plot(mid, dens / max(dens.max(), 1), color="0.2", lw=1.2, label="density")
        axA.plot(e[1:], np.cumsum(c) / max(c.sum(), 1), color="0.2", lw=1.2, ls="--", label="cumulative")
        axA.set(xscale="log", ylim=(0, 1.05), xlabel=r"quartet $\chi^2$", ylabel="normalized density")
        axA.legend(frameon=False)
 
        fam = {f["q"]: f for f in self.family(target_fa, n_boot=0)} if target_fa is not None else {}
        cmap = plt.get_cmap("viridis")
        for q in range(1, self.qcap + 1):
            col = cmap((q - 1) / max(self.qcap - 1, 1))
            th, fa = self.fa_curve(q)
            axB.plot(th, fa, color=col, lw=1.2, label=f"$q_{{min}}$={q}")
            if q in fam and not fam[q]["ceiling"]:
                axB.plot(fam[q]["chi2_th"], fam[q]["fa"], "o", ms=5, color=col, mec="white", mew=1)
        if target_fa is not None:
            axB.axhline(target_fa, color="0.5", lw=0.8, ls=":")
            if q_min is not None and q_min in fam:
                axA.axvline(fam[q_min]["chi2_th"], color="0.5", lw=0.8, ls=":")
                axB.plot(fam[q_min]["chi2_th"], fam[q_min]["fa"], "o", ms=10, mfc="none", mec="k", mew=1.2)
        axB.set(xscale="log", xlabel=r"$\chi^2_{th}$", ylabel="false alarm rate", ylim=(0, None))
        axB.legend(frameon=False, fontsize=8)
        return axes

    def save(self, path):
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), W=self.W, cens=self.cens, chi2_edges=self.chi2_edges, chi2_counts=self.chi2_counts)
        return Path(path)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(json.loads(str(f["meta"])), f["W"], f["cens"], f["chi2_edges"], f["chi2_counts"])


def _null_config(tr, params, *, nadv, noise=None, noise_window=None, stream_kw=None, chi2_max=1000.0, n_mc=350, npts=None, seed=0, min_rho=0.5, max_pval=0.5, upsample=20, min_vtriads=1, extra_kw=None, min_shift_s=None, verbose=True):
    """
    Configuración completa del null test
    """
    
    missing = {"lwin", "lambda_min", "slowmax"} - set(params)
    if missing:
        raise ValueError(f"faltan en params: {sorted(missing)} (usar tr.capacity_params(...))")
    
    lwin, lambda_min, slowmax = int(params["lwin"]), float(params["lambda_min"]), float(params["slowmax"])
    nsta, qcap = tr.array.nsta, tr.qcap
    
    if nsta < 4:
        raise ValueError("hacen falta >= 4 estaciones para validar por cuartetos")
 
    # capacidad de esta configuración
    qhist = tr._admissible(lwin, lambda_min, slowmax)
    ncap = n_capable(qhist)
    if ncap[1] == 0:
        raise ValueError(f"lwin={lwin}, lambda_min={lambda_min}, slowmax={slowmax}: ninguna tríada con compañeros de cuarteto (revisar con capacity / capacity_params)")
    if verbose:
        print(f"  capacidad: {ncap[0]}/{tr.ntriads} tríadas disponibles; con >= q compañeros: "
              + ", ".join(f"q={q}: {ncap[q]}" for q in range(1, qcap + 1)))
 
    if noise is None:
        source, sha = "synthetic", None
    
    else:
        noise = np.asarray(noise, dtype=float)
        if noise.ndim != 2 or noise.shape[1] != nsta:
            raise ValueError("noise debe ser (muestras, nsta) en el orden de tr.ids")
        
        source, sha = "surrogate", _hash(np.ascontiguousarray(noise).tobytes())
        min_shift_s = float(min_shift_s or 10 * lwin / tr.fs)
 
    B = tr.bmatrix(dict(lwin=lwin, lambda_min=lambda_min, slowmax=slowmax))
    meta = dict(
        station_codes=list(tr.ids),
        x=[float(v) for v in tr.x],
        y=[float(v) for v in tr.y],
        params=dict(fs=tr.fs, lwin=lwin, nadv=float(nadv), fmin=tr.fmin, fmax=tr.fmax, slowmax=slowmax, lambda_min=lambda_min, min_rho=float(min_rho), max_pval=float(max_pval), upsample=int(upsample), min_vtriads=int(min_vtriads), chi2_max=float(chi2_max), g_min=float(tr.g_min)),
        extra_kw=dict(extra_kw or {}), stream_kw=dict(stream_kw or {}),
        noise=dict(source=source, n_mc=int(n_mc), npts=int(npts or 20 * lwin), seed=int(seed), min_shift_s=min_shift_s, sha=sha, window=None if noise_window is None else [str(t) for t in noise_window]),
        structure=dict(triads=np.asarray(tr.triads).tolist(), B=np.asarray(B).tolist(), qhist=[int(v) for v in qhist]),
        )
    meta["id"] = _config_id(meta)
    return meta

def _null_test(tr, meta, path, *, noise=None, verbose=True):
    """
    Monte Carlo del null para la configuración `meta` (de null_config) y lo guarda en `path`.
    Quien llama (TRIAS.null_test) ya verificó que `path` no existe.
    trias corre con chi2_th = chi2_max y min_quartets = 1: los umbrales se eligen después.
    """
    p, nz, extra_kw = meta["params"], meta["noise"], meta["extra_kw"]
    lwin, nadv, slowmax, chi2_max = p["lwin"], p["nadv"], p["slowmax"], p["chi2_max"]
    n_mc, npts, min_vtriads = nz["n_mc"], nz["npts"], p["min_vtriads"]
    nsta, qcap = len(meta["station_codes"]), len(meta["station_codes"]) - 3
    fs, fmax, fmin_e, jt = tr.fs, tr.fmax, tr.fmin_eff(lwin), tr._jl
    
    if nz["source"] == "surrogate":
        if noise is None or _hash(np.ascontiguousarray(np.asarray(noise, dtype=float)).tobytes()) != nz["sha"]:
            raise ValueError("el ruido no coincide con el de la configuración (meta['noise']['sha'])")
        noise, shift = np.asarray(noise, dtype=float), int(round(nz["min_shift_s"] * fs))
 
    rng = np.random.default_rng(nz["seed"])
    kw_null = dict(min_rho=p["min_rho"], max_pval=p["max_pval"], lambda_min=p["lambda_min"],
                   upsample=p["upsample"], chi2_th=chi2_max, min_quartets=1, min_vtriads=1, **extra_kw)
    
    nwin = jt.nwin_total(npts, lwin, nadv)
    W = np.full((n_mc, nwin, qcap), np.inf)
    cens = np.zeros((n_mc, nwin, qcap), dtype=bool)
    
    chi2_all = []
    for r in range(n_mc):
        data = (synthetic_noise(rng, npts, nsta, fs, fmin_e, fmax) if nz["source"] == "synthetic"
                else surrogate_noise(rng, noise, npts, shift))
        _, result = jt.trias(data, tr.x, tr.y, fs=fs, lwin=lwin, nadv=nadv, fmin=fmin_e, fmax=fmax, slowmax=slowmax, **kw_null)
        trios = _parse_trios(result, nsta)
        W[r], cens[r] = _window_thresholds(trios, nsta, nwin, min_vtriads, chi2_max)
        chi2_all.append(_unique_quartet_chi2(trios, nsta, chi2_max))
        
        if verbose and (r + 1) % 50 == 0:
            print(f"  null {r + 1}/{n_mc}")
 
    chi2v = np.concatenate(chi2_all) if chi2_all else np.empty(0)
    lo_e = max(float(chi2v.min()) if len(chi2v) else 1e-3, 1e-6)
    edges = np.logspace(np.log10(lo_e), np.log10(chi2_max), 81)
    counts, _ = np.histogram(chi2v, bins=edges)
 
    meta = {**meta, "created": dt.datetime.now().isoformat(timespec="seconds")}
    return TRIASNullTest(meta, W, cens, edges, counts).save(path)


class NullTestStore:
    """Carpeta con los null tests (<id>.npz) de un array."""
 
    def __init__(self, path):
        self.path = Path(path)
 
    def table(self, codes=None):
        rows = []
        for p in sorted(self.path.glob("*.npz")):
            try:
                nt = TRIASNullTest.load(p)
            except Exception:
                continue
            
            if codes is not None and nt.station_codes != list(codes):
                continue
            
            q = nt.params
            rows.append(dict(path=str(p), id=nt.id, created=nt.meta.get("created"), noise=nt.noise["source"], n_mc=nt.n_mc, seed=nt.noise["seed"], fmin=q["fmin"], fmax=q["fmax"], fs=q["fs"], lwin=q["lwin"], nadv=q["nadv"], lambda_min=q["lambda_min"], slowmax=q["slowmax"], chi2_max=q["chi2_max"]))

        return pd.DataFrame(rows)