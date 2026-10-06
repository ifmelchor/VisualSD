#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import pandas as pd
from datetime import timedelta

TRIO_BASE_COLS = ["i", "j", "k", "dt_ij", "dt_jk", "dt_ki", "fq_ij", "fq_jk", "fq_ki", "rho", "sx", "sy", "weight", "sigma"]

class _MisfitKind:
    LIKELIHOOD = "likelihood"
    RAW = "raw_misfit"


def _effective_fmin(fmin, fs, lwin):
    fres = 1.5 * fs / lwin
    return fres if (fmin is None or fmin < fres) else float(fmin)


def add_seconds(t0, seconds):
    if t0 is None:
        return None
    try:
        return t0 + seconds
    except TypeError:
        return t0 + timedelta(seconds=seconds)


def _chi2_cols(nsta):
    return [f"chi2_l{l + 1}" for l in range(nsta)]


def _parse_trios(result, nsta):
    """Una fila por tríada válida (todas las ventanas), con window_id."""
    cols = ["window_id"] + TRIO_BASE_COLS + _chi2_cols(nsta)
    
    if result is None:
        return pd.DataFrame(columns=cols)
    
    blocks = []
    for w, mat in enumerate(result.trios):
        arr = np.asarray(mat, dtype=float)
        if arr.shape[0] == 0:
            continue
        blocks.append(np.column_stack([np.full(arr.shape[0], w), arr]))
    
    if not blocks:
        return pd.DataFrame(columns=cols)
    
    df = pd.DataFrame(np.vstack(blocks), columns=cols)
    for c in ("window_id", "i", "j", "k"):
        df[c] = df[c].astype(int)
    
    return df


def _parse_windows(result, lwin, fs, starttime=None):
    """Una fila por ventana detectada: tiempos, solución, se_s y tipo de mapa."""
    
    cols = ["window_id", "time_s", "t_start", "t_end", "n_trios", "sx", "sy", "se_s", "misfit_kind"]
    
    if result is None:
        return pd.DataFrame(columns=cols)
    
    t  = np.asarray(result.time_s, dtype=float)
    se = np.asarray(result.se_s, dtype=float)
    dur = lwin / fs
    
    return pd.DataFrame({
        "window_id": np.arange(len(t)),
        "time_s": t,
        "t_start": [add_seconds(starttime, float(x)) for x in t],
        "t_end": [add_seconds(starttime, float(x) + dur) for x in t],
        "n_trios": np.asarray(result.n_trios, dtype=int),
        "sx": np.asarray(result.sx, dtype=float),
        "sy": np.asarray(result.sy, dtype=float),
        "se_s": se,
        "misfit_kind": np.where(np.isnan(se), _MisfitKind.RAW, _MisfitKind.LIKELIHOOD),
    })


def _parse_jk(result, nsta):
    """Réplicas del jackknife (n_win, nsta), alineadas con _parse_windows."""
    if result is None:
        z = np.empty((0, nsta))
        return dict(sx=z, sy=z, ntri=z.astype(int), half=np.empty(0))
    
    return dict(sx=np.asarray(result.jk_sx, dtype=float).reshape(-1, nsta), sy=np.asarray(result.jk_sy, dtype=float).reshape(-1, nsta), ntri=np.asarray(result.jk_ntri, dtype=int).reshape(-1, nsta), half=np.asarray(result.jk_half, dtype=float))


def _evaluable_chi2(trios, nsta):
    """Limpia y estandariza los valores de chi2"""
    C = trios[_chi2_cols(nsta)].to_numpy(dtype=float)
    return np.where(np.isfinite(C) & (C < 1e10), C, np.inf)


class TRIASResult:
    """
    Salida de TRIAS.run.
      params: todo lo que recibió trias (fs, lwin, nadv, fmin efectivo, fmax, slowmax, lambda_min,
              chi2_th, min_quartets, ...) + null_id y target_fa (None si se corrió a mano).
      fa:     FA del null test para (q_min, chi2_th); None a mano.
    """
    def __init__(self, jlout, tr, params, q_min, starttime=None, nwin_total=None, keep_llmap=False, label=None):

        self.params = dict(params)
        self.station_codes = list(tr.ids)
        self.nsta = tr.nsta
        self.fs   = params["fs"]
        self.lwin = params["lwin"]
        self.q_min = int(q_min)
        self.chi2_th = float(params["chi2_th"])
        self.null_id = params.get("null_id")
        self.target_fa = params.get("target_fa")
        self.fa = tr.null.fa(self.q_min, self.chi2_th) if self.null_id else None
        self.starttime = starttime
        self.nwin_total = nwin_total
        self.label = label

        drate, result = jlout
        self.drate = float(drate)
        self.windows = _parse_windows(result, self.lwin, self.fs, starttime)
        self.trios = _parse_trios(result, self.nsta)
        self.jk    = _parse_jk(result, self.nsta)
        self.llmap = ([None if m is None else np.asarray(m) for m in result.llmap] if keep_llmap and result is not None else None)

    def __repr__(self):
        src = f"null={self.null_id} fa={self.fa:.3g}" if self.null_id else "a mano"
        return (f"<TRIASResult {self.label or ''} {src} q_min={self.q_min} "
                f"chi2_th={self.chi2_th:.3g} drate={self.drate:.3f} n_win={len(self.windows)}>")

    def station_performance(self, min_n=10):
        """
        Rendimiento relativo por estación, a partir de s_(-l) (estimación sin la estación l).
        Solo cuentan las ventanas donde la estación participa (sacarla quita alguna tríada).
     
          participation  fracción de ventanas en las que participa
          influence      mediana de n*|s_(-l) - s_bar|^2 / sum_k |...|^2: su peso en se_s
                         (alto = domina la incertidumbre: estación mala o clave por geometría)
          bias           |media de s_(-l) - s_bar| en s/km: corrimiento SISTEMÁTICO al sacarla
          bias_snr       bias / (dispersión / sqrt(n))
          censored       fracción de réplicas en el borde de la grilla (desplazamiento mayor al guardado)
     
        Mala: influence Y bias altos respecto del resto. Clave por geometría: influence alta, bias ~ 0.
        Los bias suman ~0 entre estaciones: una mala con b arrastra a las demás a ~ -b/(n-1),
        así que se leen comparando entre estaciones, no contra un umbral fijo.
        """
        jk = self.jk
        slowint_f = float(self.params.get("slowint_f", 0.01))
        ntr = self.windows["n_trios"].to_numpy()[:, None]
     
        part = jk["ntri"] < ntr
        valid = part & np.isfinite(jk["sx"])
        n = valid.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            jx, jy = np.where(valid, jk["sx"], np.nan), np.where(valid, jk["sy"], np.nan)
            cx = jx - np.nansum(jx, axis=1, keepdims=True) / n
            cy = jy - np.nansum(jy, axis=1, keepdims=True) / n
            d2 = cx ** 2 + cy ** 2
            infl = n * d2 / np.nansum(d2, axis=1, keepdims=True)
        dx = jx - self.windows["sx"].to_numpy()[:, None]
        dy = jy - self.windows["sy"].to_numpy()[:, None]
        edge = valid & (np.fmax(np.abs(dx), np.abs(dy)) >= jk["half"][:, None] - slowint_f / 2)
     
        rows = []
        for l, code in enumerate(self.station_codes):
            v = valid[:, l] & ~edge[:, l]                    # sin censuradas para el bias
            k = int(v.sum())
            bx, by = (cx[v, l].mean(), cy[v, l].mean()) if k else (np.nan, np.nan)
            sd = np.sqrt(cx[v, l].var(ddof=1) + cy[v, l].var(ddof=1)) if k > 1 else np.nan
            ok = valid[:, l].sum() >= min_n
            rows.append(dict(
                station=code,
                participation=part[:, l].mean() if len(part) else np.nan,
                influence=np.nanmedian(infl[valid[:, l], l]) if ok else np.nan,
                bias=np.hypot(bx, by) if k >= min_n else np.nan,
                bias_snr=np.hypot(bx, by) / (sd / np.sqrt(k)) if k >= min_n and sd > 0 else np.nan,
                censored=edge[part[:, l], l].mean() if part[:, l].any() else np.nan,
            ))
        return pd.DataFrame(rows)