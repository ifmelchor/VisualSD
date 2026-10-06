#!/usr/bin/env python3
# coding=utf-8

import time
from pathlib import Path
import pandas as pd
import numpy as np
from ...jlwrap import jl_call

from .utils import TRIASResult, _effective_fmin
from .capacity import TRIASCapacity, n_capable, CapacityRange
from .nulltest import TRIASNullTest, NullTestStore, _null_test, _null_config

def _same_config(A, B, tol=1e-9):
    """(fmin, fmax, fs) iguales; fmin puede ser None."""
    (fmin_a, fmax_a, fs_a) = A
    (fmin_b, fmax_b, fs_b) = B
    same_min = (fmin_a is None and fmin_b is None) or \
               (fmin_a is not None and fmin_b is not None and abs(fmin_a - fmin_b) < tol)
    return same_min and abs(fmax_a - fmax_b) < tol and abs(fs_a - fs_b) < tol


class TRIAS:
    def __init__(self, array, *, fs, fmax, fmin=None, station_list=None, nulldir="nulltest", g_min=0.1):
        
        self.array = array.subarray(station_list) if station_list else array
        
        self.ids, self.x, self.y = self.array.positions_km()

        self.g_min = g_min
        self.fmin = None if fmin is None else float(fmin)
        self.fmax = float(fmax)
        self.fs   = float(fs)

        # check
        if self.fs <= 0 or self.fmax <= 0:
            raise ValueError(f"fmax={fmax} y fs={fs} tienen que ser > 0")

        if self.fmin is not None and self.fmin >= self.fmax:
            raise ValueError(f"fmin={self.fmin} >= fmax={self.fmax}")
        
        if self.fmax > self.fs / 2:
            raise ValueError(f"fmax={self.fmax} > Nyquist ({self.fs / 2:g} Hz con fs={self.fs:g})")

        self.null = None    # null test cargado: null_test() o load_null_test()
        self._jl = jl_call("TRIAS")
        self.triads = self._jl.triads(self.x, self.y, g_min=g_min)
        if self.ntriads == 0:
            raise ValueError(f"ninguna tríada con forma > {g_min:g}")

        self.store = NullTestStore(Path(nulldir).resolve())
        n_all = len(list(self.store.path.glob("*.npz")))
        mine = self.get_nulltests()
        print(f"  nulldir: {self.store.path} ({n_all} null tests, {len(mine)} de este TRIAS)")
        for _, r in mine.iterrows():
            print(f"    {Path(r['path']).name}  {r['noise']:9s} n_mc={r['n_mc']} seed={r['seed']}  lwin={r['lwin']} lambda_min={r['lambda_min']:.3g}  slowmax={r['slowmax']:.3g}  ({r['created']})")

    def __repr__(self):
        band = f"{self.fmin if self.fmin is not None else 'auto'}-{self.fmax:g} Hz @ {self.fs:g} sps"
        
        null = f"null={self.null.id}" if self.null else "sin null test"
        return f"<TRIAS {self.array.code}.{self.array.component} {self.array.nsta} est. {band}, {null}>"

    @property
    def config(self):
        """(fmin, fmax, fs): lo que fija el TRIAS además de las estaciones."""
        return self.fmin, self.fmax, self.fs

    @property
    def nsta(self):
        return len(self.ids)

    @property
    def qcap(self):
        """Núm máx de cuartetos por tríada: N - 3."""
        return self.nsta - 3

    @property
    def ntriads(self):
        """tríadas que pasan el filtro de forma"""
        return len(self.triads)

    def get_nulltests(self):
        """Null tests guardados de estas estaciones con esta banda y fs."""
        df = self.store.table(codes=self.ids)
        if len(df):
            keep = [_same_config(self.config, (None if fm != fm else fm, fx, f)) for fm, fx, f in zip(df["fmin"], df["fmax"], df["fs"])]
            df = df[keep].reset_index(drop=True)
        return df

    def _check_qf(self, q_min, frac):
        if not 1 <= q_min <= self.qcap:
            raise ValueError(f"q_min={q_min} fuera de 1..{self.qcap} (qcap = N - 3)")
        
        if not 0 < frac <= 1:
            raise ValueError("frac tiene que estar en (0, 1]")
        
        if self.ntriads == 0:
            raise ValueError(f"ninguna tríada con forma > {self.g_min:g}")

        return max(1, int(np.ceil(frac * self.ntriads)))

    def fmin_eff(self, lwin):
        """fmin de trias: max(fmin, 1.5 fs/lwin)."""
        return _effective_fmin(self.fmin, self.fs, lwin)

    def bmatrix(self, params):
        """
        Matriz B_ij (nsta, nsta) de la configuración `params` (lwin, lambda_min, slowmax
        """
        lwin, lambda_min, slowmax = int(params["lwin"]), float(params["lambda_min"]), float(params["slowmax"])
        
        fe = self.fmin_eff(lwin)
        if fe >= self.fmax:
            raise ValueError(f"lwin={lwin} no resuelve la banda: fmin efectivo {fe:.3g} Hz >= fmax={self.fmax:g} Hz")

        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        mdist = np.hypot(np.subtract.outer(x, x), np.subtract.outer(y, y))
        
        return self._jl.bmatrix(mdist, lwin=lwin, fmin=fe, fmax=self.fmax, fs=self.fs, lambda_min=lambda_min, slowmax=slowmax)

    ## ------------ CAPACIDAD ------------

    def _admissible(self, lwin, lambda_min, slowmax):
        """qhist (N-2,): tríadas admisibles con exactamente q compañeros, para una configuración."""
        
        # check fmin
        fe = self.fmin_eff(lwin)
        if fe >= self.fmax: 
            raise ValueError(f"lwin={lwin} ({lwin / self.fs:g} s) no resuelve la banda: fmin efectivo {fe:.3g} Hz >= fmax={self.fmax:g} Hz; hace falta lwin > {1.5 * self.fs / self.fmax:.0f}")

        return np.asarray(self._jl.capacity(self.x, self.y, lwin=int(lwin), fmin=self.fmin_eff(lwin), fmax=self.fmax, fs=self.fs, lambda_min=lambda_min, slowmax=slowmax, g_min=self.g_min))

    def _lwin_min(self, slowmax, lambda_min, frac=1.0, q_min=1, lwin_hi=2 ** 16):
        """
        Menor lwin con al menos `frac` de las ntriads utilizables a ese slowmax y lambda_min.
        """
        
        need = self._check_qf(q_min, frac)

        def ok(w):
            if self.fmin_eff(w) >= self.fmax:
                return False
            return n_capable(self._admissible(w, lambda_min, slowmax))[q_min] >= need

        # primer lwin con banda no vacía
        lo = int(np.floor(1.5 * self.fs / self.fmax)) + 1

        if ok(lo):
            return lo

        hi = lo
        while not ok(hi):
            lo, hi = hi, hi * 2
            if hi > lwin_hi:
                raise ValueError(f"ningún lwin <= {lwin_hi} da {frac:.0%} de tríadas con q >= {q_min} a slowmax={slowmax}")

        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)

        return hi

    def _slowmax_max(self, lwin, lambda_min, q_min=1, frac=1.0, tol=1e-3):
        """
        Mayor slowmax con al menos `frac` de las ntriads con >= q_min compañeros, a ese lwin.
        """
        need = self._check_qf(q_min, frac)

        ok = lambda s: n_capable(self._admissible(lwin, lambda_min, s))[q_min] >= need
        
        lo, hi = 1e-6, lwin / (self.fs * self.array.dmin)

        if not ok(lo):
            raise ValueError(f"lwin={lwin} ({lwin / self.fs:g} s): ni con slowmax -> 0 hay {frac:.0%} de tríadas con q >= {q_min}")

        while hi - lo > tol * hi:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        
        return lo

    def _lambda_max(self, lwin, slowmax, q_min=1, frac=1.0, tol=1e-3):
        """
        Mayor lambda_min con al menos `frac` de las ntriads con >= q_min.
        """
        need = self._check_qf(q_min, frac)
        ok = lambda lam: n_capable(self._admissible(lwin, lam, slowmax))[q_min] >= need
        lo, hi = 1e-3, 0.999
        
        if not ok(lo):
            raise ValueError(f"lwin={lwin}, slowmax={slowmax}: ni con lambda_min -> 0 hay {frac:.0%} de tríadas con q >= {q_min}")
        if ok(hi):
            return hi
        
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        
        return lo

    def capacity_range(self, lambda_range, *, lwin_max=500, smax_max=3.5, n_lwin=10, n_smax=10, verbose=True):
        """
        Rangos de lwin y slowmax que cubren toda la transición de capacidad, para cada
        lambda_min de lambda_range.
        """
        
        fs = self.fs
        q_full = self.qcap
        f_last = 1.0 / self.ntriads
        
        lams = np.unique(np.asarray(lambda_range, dtype=float))
        w_lo = self._lwin_min(1e-6, float(lams[-1]), q_min=q_full)

        rows = []
        for lam in lams:
            w_hi = int(lwin_max) if lwin_max is not None else self._lwin_min(smax_max, lam, q_min=q_full)
            w_hi = max(w_hi, w_lo + 1)
            s_lo = self._slowmax_max(w_lo, lam, q_min=q_full)
            s_hi = self._slowmax_max(w_hi, lam, q_min=1, frac=f_last)
            rows.append(dict(lambda_min=lam, lwin_lo=w_lo, lwin_hi=w_hi, smax_lo=s_lo, smax_hi=s_hi))

        tab = pd.DataFrame(rows) 
        lw_hi = int(tab["lwin_hi"].max())

        s_lo, s_hi = 0.8 * tab["smax_lo"].min(), 1.1 * tab["smax_hi"].max()
        if smax_max is not None:
            s_hi = min(s_hi, 1.1 * smax_max)

        if verbose:
            print(f"  tríadas: {self.ntriads} con forma > {self.g_min:g}, qcap = {q_full}, apertura = {self.array.aperture:.3g} km")
            print(tab.round(3).to_string(index=False))
            print(f"  -> lwin {w_lo}..{lw_hi} ({w_lo / fs:g}..{lw_hi / fs:g} s), slowmax {s_lo:.3g}..{s_hi:.3g} s/km")
        
        return CapacityRange(lwin_range=np.unique(np.geomspace(w_lo, lw_hi, n_lwin).round().astype(int)), slowmax_range=np.round(np.linspace(s_lo, s_hi, n_smax), 4), lambda_range=lams, by_lambda=tab)

    def capacity(self, rng, verbose=True):
        """Barrido lwin x lambda_min x slowmax sobre `rng` (CapacityRange)."""

        lw = rng.lwin_range
        la = rng.lambda_range
        sm = rng.slowmax_range
        
        valid = np.array([self.fmin_eff(w) < self.fmax for w in lw])
        qhist = np.full((len(lw), len(la), len(sm), self.array.nsta - 2), -1, dtype=np.int32)

        t0    = time.time()
        total = int(valid.sum()) * len(la) * len(sm)
        done  = 0

        for a, w in enumerate(lw):
            if not valid[a]:
                continue
            
            for b, lam in enumerate(la):
                for c, s in enumerate(sm):
                    qhist[a, b, c] = self._admissible(w, lam, s)
            
            done += len(la) * len(sm)
            
            if verbose:
                el = time.time() - t0
                print(f"  lwin={w:5d}  {el:6.1f} s  ETA {el / done * (total - done):6.1f} s", flush=True)

        return TRIASCapacity(self, rng, qhist)
    
    def capacity_params(self, q_min, frac=1.0, *, lwin=None, lambda_min=None, slowmax=None, verbose=True):
        """
        Parámetros para null_test / run a partir de un objetivo de capacidad: al menos `frac` de las
        ntriads con >= q_min compañeros. Se dan dos de (lwin, lambda_min, slowmax) y se resuelve el tercero:
          lwin + slowmax        -> lambda_min MÁXIMO
          lambda_min + slowmax  -> lwin MÍNIMO
          lwin + lambda_min     -> slowmax MÁXIMO
          tr.null_test(tr.capacity_params(q_min=2, lambda_min=0.3, slowmax=1.5), nadv=0.5)
        """

        given = {k for k, v in dict(lwin=lwin, lambda_min=lambda_min, slowmax=slowmax).items() if v is not None}
        
        if len(given) != 2:
            raise ValueError("dar exactamente dos de lwin, lambda_min, slowmax")

        if "lambda_min" not in given:
            lambda_min = self._lambda_max(int(lwin), slowmax, q_min=q_min, frac=frac)
        
        elif "lwin" not in given:
            lwin = self._lwin_min(slowmax, lambda_min, q_min=q_min, frac=frac)
        
        else:
            slowmax = self._slowmax_max(int(lwin), lambda_min, q_min=q_min, frac=frac)
        
        lwin = int(lwin)
 
        if verbose:
            n = n_capable(self._admissible(lwin, lambda_min, slowmax))[q_min]
            print(f"  lwin={lwin} ({lwin / self.fs:g} s), lambda_min={lambda_min:.3f}, slowmax={slowmax:.3f} s/km: {n}/{self.ntriads} tríadas con q >= {q_min} ({n / self.ntriads:.0%})")

        return dict(lwin=lwin, lambda_min=float(lambda_min), slowmax=float(slowmax))

    ## ------------ NULL TEST ------------

    def null_test(self, params, *, nadv, noise_window=None, stream_kw=None, gaps="reject", **kw):
        
        """
        Monte Carlo del null con la configuración.
          p  = tr.capacity_params(q_min=2, lambda_min=0.3, slowmax=1.5)
          nt = tr.null_test(p, nadv=0.5);  nt.plot()
        
        noise_window=(t0, t1): surrogate con ruido real del sitio o None: sintético.
        **kw: chi2_max, n_mc, npts, seed, min_rho, max_pval, upsample, min_vtriads, extra_kw, min_shift_s, verbose.
        """
        verbose = kw.get("verbose", True)
        stream_kw = dict(stream_kw or {})
        stream_kw["sample_rate"] = self.fs
        
        noise = self._noise(noise_window, gaps, **stream_kw) if noise_window else None
        
        meta = _null_config(self, params, nadv=nadv, noise=noise, noise_window=noise_window, stream_kw=stream_kw, **kw)
        path = self.store.path / f"{meta['id']}.npz"
        
        if path.is_file():
            print(f"  null test ya existe: {path}")
        else:
            self.store.path.mkdir(parents=True, exist_ok=True)
            _null_test(self, meta, path, noise=noise, verbose=verbose)
            print(f"  null test -> {path}")
        
        self.load_null_test(path)
        return path

    def load_null_test(self, nullpath):
        """
        Abre un null test (.npz) y verifica que sea apto.
        nullpath: ruta al .npz, o solo el nombre (con o sin .npz), que se busca en nulldir.
        """
        nullpath = Path(nullpath)
        if not nullpath.is_file():
            name = nullpath.name if nullpath.suffix == ".npz" else f"{nullpath.name}.npz"
            nullpath = self.store.path / name
        
        if not nullpath.is_file():
            raise FileNotFoundError(f"no existe {nullpath}")
        
        nt = TRIASNullTest.load(nullpath)
        nt.check(self.ids, self.x, self.y, self.config)
        self.null = nt
        return nt


    def run(self, starttime, endtime, q_min, *, target_fa=0.03, params=None, nadv=None, chi2_th=None, min_rho=0.5, max_pval=0.5, upsample=20, min_vtriads=1, extra_kw=None, gaps="reject", keep_llmap=False, **stream_kw):
        """
        Corre TRIAS. Dos modos, excluyentes:
          calibrado:  tr.run(t0, t1, q_min=2, target_fa=0.03)
                      con el null test cargado (tr.null): params, nadv y preprocesado salen de él,
                      chi2_th de (q_min, target_fa)
          a mano:     tr.run(t0, t1, q_min=2, params=p, nadv=0.5, chi2_th=50)
                      p = dict(lwin, lambda_min, slowmax), p. ej. de capacity_params; sin FA conocida
        gaps: 'reject' | 'interpolate'. **stream_kw va a get_stream (preprocesado).
        Devuelve un TRIASResult; en sus params van null_id y target_fa (None a mano).
        """

        if not 1 <= q_min <= self.qcap:
            raise ValueError(f"q_min={q_min} fuera de 1..{self.qcap}")
        
        nt = None
        manual = dict(params=params, nadv=nadv, chi2_th=chi2_th)
        
        if all(v is None for v in manual.values()):
            nt = self.null
            if nt is None:
                raise RuntimeError("sin null test cargado: tr.null_test(params, ...) o tr.load_null_test(path); o a mano con params, nadv y chi2_th")

            p = nt.params
            lwin, nadv, lambda_min, slowmax = int(p["lwin"]), p["nadv"], p["lambda_min"], p["slowmax"]
            trias_kw = nt.trias_kw(q_min, target_fa)
            stream_kw = {**nt.stream_kw, **stream_kw}
        
        else:
            missing = [k for k, v in manual.items() if v is None]
            if missing:
                raise ValueError(f"a mano hay que dar params, nadv y chi2_th (falta {missing})")
            lwin, lambda_min, slowmax = int(params["lwin"]), float(params["lambda_min"]), float(params["slowmax"])
            self._admissible(lwin, lambda_min, slowmax)
            trias_kw = dict(min_rho=min_rho, max_pval=max_pval, lambda_min=lambda_min, upsample=int(upsample), min_vtriads=int(min_vtriads), min_quartets=int(q_min), chi2_th=float(chi2_th), **(extra_kw or {}))

        # todo lo que recibe trias, en un solo dict: se usa para llamar y queda en el resultado
        run_kw = dict(fs=self.fs, lwin=lwin, nadv=nadv, fmin=self.fmin_eff(lwin), fmax=self.fmax, slowmax=slowmax, **trias_kw)

        # datos a la fs del TRIAS
        user_sr = stream_kw.pop("sample_rate", None)
        if user_sr is not None and abs(user_sr - self.fs) > 1e-9:
            print(f" >> sample_rate={user_sr:g} ignorado: se usa la fs del TRIAS ({self.fs:g} Hz)")

        stream_kw["sample_rate"] = self.fs
        ids, data, t0, fs = self.array.get_stream(starttime, endtime, 0.0, gaps=gaps, return_array=True, **stream_kw)

        if data is None:
            raise ValueError(f"[{self.array.code}] sin datos entre {starttime} y {endtime}")

        if abs(fs - self.fs) > 1e-6:
            raise ValueError(f"fs de los datos ({fs:g} Hz) distinta de la del TRIAS ({self.fs:g} Hz)")

        codes = [i.rsplit(".", 1)[0] for i in ids]
        missing = [c for c in self.ids if c not in codes]
        if missing:
            raise ValueError(f"sin datos utilizables para {missing}")
        
        data = data[:, [codes.index(c) for c in self.ids]]
 
        jlout = self._jl.trias(data, self.x, self.y, **run_kw)
        nwin = self._jl.nwin_total(data.shape[0], lwin, nadv)
        p = {**run_kw, "null_id": nt.id if nt else None, "target_fa": float(target_fa) if nt else None}
        
        return TRIASResult(jlout, self, p, q_min, starttime=t0, nwin_total=nwin, keep_llmap=keep_llmap, label=f"{self.array.code}.{self.array.component}")

    def _noise(self, window, gaps, **stream_kw):
        """Ruido real del sitio (para surrogates), columnas en el orden de self.ids."""
        ids, data, _, fs = self.array.get_stream(window[0], window[1], 0.0, gaps=gaps, return_array=True, **stream_kw)

        if data is None:
            raise ValueError(f"sin datos de ruido en {window}")
        
        have = [i.rsplit(".", 1)[0] for i in ids]
        missing = [c for c in self.ids if c not in have]
        if missing:
            raise ValueError(f"sin ruido utilizable en {window} para {missing} ")

        if abs(fs - self.fs) > 1e-6:
            raise ValueError(f"fs del ruido ({fs:g} Hz) distinta de la del TRIAS ({self.fs:g} Hz)")

        return data[:, [have.index(c) for c in self.ids]]