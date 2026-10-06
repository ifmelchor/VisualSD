#!/usr/bin/python3

import numpy as np
from multitaper import MTSpec
from obspy import Stream, Trace, UTCDateTime

import scipy.signal as scis
from scipy.stats import gaussian_kde

GAP_POLICIES = ("reject", "interpolate", "nan")

def _process(tr, config):
    """Procesado de UNA traza continua: demean, detrend, taper, corrección, filtro, remuestreo."""

    st = Stream([tr.copy()])
    
    if config.get('demean', True):
        st.detrend('demean')
    
    if config.get('detrend', True):
        st.detrend('linear')
    
    rm_resp = config.get('rm_resp', False)

    if config.get('taper', True) or rm_resp:
        st.taper(max_percentage=config.get('taper_fraction', 0.05), max_length=config.get('taper_max_length', None))

    rm_sens = config.get('rm_sens', False) and not rm_resp
    if rm_resp or rm_sens:
        val = (config.get('rm_dict') or {}).get(tr.stats.channel)
        if val is None:
            raise ValueError(f"{tr.id}: sin {'respuesta' if rm_resp else 'sensibilidad'} en rm_dict")
        if rm_resp:
            st[0].remove_response(inventory=val, output=config.get('output', 'VEL'), pre_filt=config.get('resp_prefilt', (0.005, 0.01, 35, 40)), taper=False)
        else:
            st[0].data = st[0].data / val

    prefilt = config.get('filt', (0.01, 35))
    if prefilt:
        st.filter('bandpass', freqmin=prefilt[0], freqmax=prefilt[1], zerophase=True)
    
    sr = config.get('sample_rate')
    if sr and abs(tr.stats.sampling_rate - sr) > 1e-9:
        st.resample(sr)
    
    return st[0]


def _gaps(segs, t0, t1):
    """Lista de huecos (inicio, fin, duración en s), incluidos los faltantes en los bordes."""
    dt = segs[0].stats.delta
    out = []
    if t0 is not None and segs[0].stats.starttime > t0 + 0.5 * dt:
        out.append((t0, segs[0].stats.starttime, segs[0].stats.starttime - t0))
    
    for a, b in zip(segs[:-1], segs[1:]):
        out.append((a.stats.endtime, b.stats.starttime, b.stats.starttime - a.stats.endtime - dt))
    
    if t1 is not None and segs[-1].stats.endtime < t1 - 0.5 * dt:
        out.append((segs[-1].stats.endtime, t1, t1 - segs[-1].stats.endtime))

    return out


def _assemble_nan(segs, t0, t1, margin):
    """Segmentos ya procesados -> una traza en la grilla común, NaN en los huecos extendidos `margin` segundos (bordes deformados por taper y filtro)."""

    dt = segs[0].stats.delta
    start = t0 if t0 is not None else segs[0].stats.starttime
    end   = t1 if t1 is not None else segs[-1].stats.endtime
    n = int(round((end - start) / dt)) + 1
    
    out = np.full(n, np.nan)
    for tr in segs:
        i0 = int(round((tr.stats.starttime - start) / dt))
        a, b = max(i0, 0), min(i0 + tr.stats.npts, n)
        if b > a:
            out[a:b] = tr.data[a - i0:b - i0]
    
    m = int(round(margin / dt))
    bad = np.isnan(out)
    if m > 0 and bad.any():
        edges = np.diff(np.r_[False, bad, False].astype(np.int8))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out[max(s - m, 0):min(e + m, n)] = np.nan
    
    stats = segs[0].stats.copy()
    stats.starttime, stats.npts = start, n
    return Trace(out, header=stats)


def stream_preprocess(st_raw, config):
    """
    Preprocesado con política de huecos. st_raw puede traer varios segmentos por canal.

    config (dict):
      gaps ('reject'):
        'reject'      -> canal con cualquier hueco (o cobertura incompleta) se descarta.
        
        'interpolate' -> huecos internos <= max_gap (s) se rellenan linealmente ANTES de procesar; si hay uno mayor, o faltan datos en los bordes, se descarta.
        
        'nan'         -> cada segmento se procesa por separado y se arma en la grilla común con NaN en los huecos, extendidos gap_margin (s) a cada lado.

      max_gap (s, 1.0), gap_margin (s; por defecto 1/filt[0], o 10 s sin filtro)

    Devuelve un Stream con UNA traza por canal aceptado.
    """

    policy = config.get('gaps', 'reject')

    if policy not in GAP_POLICIES:
        raise ValueError(f"gaps='{policy}': usar uno de {GAP_POLICIES}")

    t0 = UTCDateTime(config['starttime']) if config.get('starttime') is not None else None
    t1 = UTCDateTime(config['endtime']) if config.get('endtime') is not None else None
    
    filt   = config.get('filt', (0.01, 35))
    margin = config.get('gap_margin', 1.0 / filt[0] if filt else 10.0)

    out = Stream()
    for tid in sorted({tr.id for tr in st_raw}):
        segs = sorted(st_raw.select(id=tid), key=lambda tr: tr.stats.starttime)
        gaps = _gaps(segs, t0, t1)

        if not gaps:
            out += _process(segs[0], config)
            continue

        worst = max(g[2] for g in gaps)
        if policy == 'reject':
            print(f" >> [{tid}] {len(gaps)} hueco(s), el mayor de {worst:.2f} s: descartado")
            continue

        if policy == 'interpolate':
            edge = (t0 is not None and gaps[0][0] == t0) or (t1 is not None and gaps[-1][1] == t1)
            
            max_gap = config.get('max_gap', 1.0)
            if edge or worst > max_gap:
                why = "faltan datos en los bordes" if edge else f"hueco de {worst:.2f} s > max_gap={max_gap} s"
                print(f" >> [{tid}] {why}: descartado")
                continue

            joined = Stream([s.copy() for s in segs]).merge(method=1, fill_value='interpolate')
            out += _process(joined[0], config)
            continue

        # policy nan...
        seg_cfg = {**config, 'taper_max_length': margin}
        done = [_process(s, seg_cfg) for s in segs
                if s.stats.npts * s.stats.delta > 2 * margin]
        
        if not done:
            print(f" >> [{tid}] todos los segmentos más cortos que 2*gap_margin: descartado")
            continue
        
        out += _assemble_nan(done, t0, t1, margin)
        print(f" >> [{tid}] {len(gaps)} hueco(s) rellenados con NaN (+{margin:.0f} s a cada lado)")

    return out


def nearest_pow_2(x):
    """
    Find power of two nearest to x
    """
    a = int(np.pow(2, np.ceil(np.log2(x))))
    b = int(np.pow(2, np.floor(np.log2(x))))

    if abs(a - x) < abs(b - x):
        return a

    else:
        return b


def spectrogram(data, fs, winl=50, olap=0.75, fq_band=(), date_list=None, **kwargs):
    """
    Code to compute the spectrogram.
    The code returns AxesImage and the upper and lower value of the norm.
    kwargs: rel_norm (False), v_max (None), v_min (None), interpolation ('gaussian')
    fq_logscale (False), axis_bar, axis_bar_label
    """

    npts = len(data)
    
    # defining NFFT
    nfft = int(nearest_pow_2(winl * fs))

    if nfft > npts:
        nfft = int(nearest_pow_2(npts / 8.0))

    # pad with zeros to smooth the spectrogram
    mult = 8*nfft

    # overlap
    nlap = int(nfft * olap)

    # compute the spectrogram
    freq, time, sxx = ss.spectrogram(data, fs=fs,
        nperseg=nfft, nfft=mult, noverlap=nlap, scaling='spectrum')

    if fq_band != ():
        fnptlo = np.argmin(np.abs(freq-fq_band[0]))
        fnpthi = np.argmin(np.abs(freq-fq_band[1]))
        freq = freq[fnptlo:fnpthi]
        sxx = sxx[fnptlo:fnpthi,:]

    return freq, sxx


def psd(data, fs, winl=50, olap=0.75, window="hann", scaling='density'):

    npts = len(data)

    # defining NFFT
    nfft = int(nearest_pow_2(winl * fs))
    
    if nfft > npts:
        nfft = int(nearest_pow_2(npts / 8.0))

    nlap = int(nfft * olap)

    f, pxx = welch(data, fs=fs, window=window, nperseg=nfft, noverlap=nlap, scaling=scaling)

    return f, pxx


def multitaper(data, fs, nw=4.0, kspec=None):

    dt = 1.0 / fs
    
    if kspec is None:
        kspec = int(2 * nw - 1)
        
    psd_obj = MTSpec(data, nw=nw, kspec=kspec, dt=dt)
    
    f   = psd_obj.freq
    pxx = psd_obj.spec
    
    return f, pxx


def pdf(x, y, bw_method=None, weights=None):
    """
    Calcula la Función de Densidad de Probabilidad (PDF) de una matriz de datos.
    
    Parámetros:
    :param x: Array 2D de datos (M, N), donde M son observaciones y N son pasos/trazas.
    :param y: Array 1D del espacio donde se evaluará la PDF
    :param bw_method: Método para calcular el ancho de banda
    :param weights: (opcional)
    
    Retorna:
    :return: Array 2D de forma (len(y), N) con las densidades evaluadas.
    """

    # Preparamos el espacio "y" como 1D plano
    y_eval = y.reshape(-1)
    
    # Matriz vacía para guardar los resultados
    pdf_matrix = np.empty((len(y), x.shape[1]))

    for i in range(x.shape[1]):
        col_data = x[:, i]
        
        # Filtrar NaNs
        valid_mask = np.isfinite(col_data)
        col_clean = col_data[valid_mask]
        
        if len(col_clean) <= 1:
            pdf_matrix[:, i] = np.nan
            continue
            
        col_weights = weights[valid_mask] if weights is not None else None

        try:
            kde = gaussian_kde(col_clean, bw_method=bw_method, weights=col_weights)
            pdf_matrix[:, i] = kde(y_eval)
        
        except np.linalg.LinAlgError:
            pdf_matrix[:, i] = np.nan

    return pdf_matrix