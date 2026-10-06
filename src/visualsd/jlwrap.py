#!/usr/bin/env python3
# coding=utf-8

import time
import numpy as np

_JL = None
_INSTANCES = {}


def _julia():
    """Arranca Julia"""
    global _JL
    if _JL is None:
        from juliacall import Main
        _JL = Main
    return _JL


class _JlSAB:
    def __init__(self):
        raise NotImplementedError("wrapper de ZLCC todavía no escrito")


class _JlTrias:
    def __init__(self, warmup=False):

        t0 = time.time()

        jl = _julia()
        jl.seval(f"import TRIAS")

        self._mat = jl.seval("x -> Matrix{Float64}(x)")
        self._vec = jl.seval("x -> Vector{Float64}(x)")
        self._trias = jl.seval(f"TRIAS.trias")
        self._capacity = jl.seval(f"TRIAS.array_capacity")
        self._windowing = jl.seval(f"TRIAS.windowing")
        self._init_triads = jl.seval(f"TRIAS.init_triads")
        self._bmatrix = jl.seval("TRIAS.Bmatrix")

        if warmup:
            self._warmup()
            print(f" >> TRIAS.jl precompiled ({time.time() - t0:.1f} s)")

    def _warmup(self):
        k = np.arange(6); x = 0.05 * np.cos(k * np.pi / 3); y = 0.05 * np.sin(k * np.pi / 3)
        t = np.arange(1024) / 100.0
        d = np.column_stack([np.sin(2 * np.pi * 4.7 * (t - (0.3 * a + 0.2 * b))) + np.sin(2 * np.pi * 7.3 * (t - (0.3 * a + 0.2 * b)) + 1.0) for a, b in zip(x, y)])
        
        self.trias(d, x, y, fs=100.0, lwin=256, nadv=0.5, fmin=1.0, fmax=10.0, slowmax=1.0, min_rho=0.5, max_pval=0.5, lambda_min=0.3, upsample=20, min_vtriads=1, min_quartets=1, chi2_th=13.0)

        self.capacity(x, y, lwin=256, fmin=1.0, fmax=10.0, fs=100.0, lambda_min=0.3, slowmax=1.0, g_min=0.1)

    def trias(self, data, x, y, *, fs, lwin, nadv, fmin, fmax, slowmax, **kw):
        kw = {k: (float(v) if isinstance(v, float) else v) for k, v in kw.items()}
        return self._trias(self._mat(np.asarray(data, dtype=float)), self._vec(x), self._vec(y), float(fs), int(lwin), float(nadv), float(fmin), float(fmax), float(slowmax), **kw)

    def triads(self, x, y, g_min=0.1):
        r = self._init_triads(self._vec(x), self._vec(y), shape_min=float(g_min))
        return np.array([[t.i, t.j, t.k] for t in r[3]], dtype=int).reshape(-1, 3)

    def capacity(self, x, y, *, lwin, fmin, fmax, fs, lambda_min, slowmax, g_min):
        cap = self._capacity(self._vec(x), self._vec(y), int(lwin), float(fmax), float(fmin), float(fs), float(lambda_min), float(slowmax), float(g_min))
        qhist = np.asarray(cap.qhist, dtype=np.int32)
        return qhist

    def nwin_total(self, npts, lwin, nadv):
        _, nwin = self._windowing(int(npts), int(lwin), float(nadv))
        return int(nwin)

    def bmatrix(self, mdist, *, lwin, fmin, fmax, fs, lambda_min, slowmax):
        B = self._bmatrix(self._mat(mdist), int(lwin), float(fmax), float(fmin), float(fs), float(lambda_min), float(slowmax))
        return np.asarray(B, dtype=int)


class _JlZLCC:
    def __init__(self, warmup=True):

        t0 = time.time()
        jl = _julia()
        jl.seval("import ZLCC")

        self._mat = jl.seval("x -> Matrix{Float64}(x)")
        self._vec = jl.seval("x -> Vector{Float64}(x)")
        self._zlcc = jl.seval("ZLCC.zlcc")

        if warmup:
            self._warmup()
            print(f" >> ZLCC.jl precompiled ({time.time() - t0:.1f} s)")

    def _warmup(self):
        k = np.arange(6); x = 0.05 * np.cos(k * np.pi / 3); y = 0.05 * np.sin(k * np.pi / 3)
        t = np.arange(1024) / 100.0
        tau = 0.3 * x + 0.2 * y
        d = np.column_stack([np.sin(2 * np.pi * 4.7 * (t - tk)) + np.sin(2 * np.pi * 7.3 * (t - tk) + 1.0) for tk in tau])
        self.zlcc(d, x, y, fs=100.0, lwin=256, nadv=0.5, fmin=1.0, fmax=10.0, slowmax=1.0, toff=0.2, slowint_c=0.1, slowint_f=0.01, ccerr=0.9, maac_th=0.5, slowfw=0.5, return_cmap=False)

    def zlcc(self, data, x, y, *, fs, lwin, nadv, fmin, fmax, slowmax, toff, slowint_c, slowint_f, ccerr, maac_th, slowfw, return_cmap):

        return self._zlcc(self._mat(data), self._vec(x), self._vec(y), float(fs), int(lwin), float(nadv), float(fmin), float(fmax), float(slowmax), float(toff), float(slowint_c), float(slowint_f), float(ccerr), float(maac_th), float(slowfw), bool(return_cmap))


_CLASSES = {"SeisArrayBase":_JlSAB, "TRIAS": _JlTrias, "ZLCC": _JlZLCC}


def jl_call(module):
    if module not in _CLASSES:
        raise ValueError(f"paquete Julia desconocido: {module!r} (disponibles: {list(_CLASSES)})")
    
    if module not in _INSTANCES:
        _INSTANCES[module] = _CLASSES[module]()
    
    return _INSTANCES[module]
