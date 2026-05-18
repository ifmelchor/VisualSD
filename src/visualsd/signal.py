#!/usr/bin/python3

import numpy as np
import scipy.signal as scis
from multitaper import MTSpec
from scipy.stats import gaussian_kde


def stream_preprocess(st_raw, config):
    # config es un dict
    # keys: "method"/"fill_value" para merge
    #       "demean" (true)
    #       "detreand" (true)
    #       "taper" (true) / "taper_fraction" (0.05)
    #       "rm_resp" (archivo de respuesta instrumental)
    #       "rm_sens" (recibe un diccionario de float para dividir)
    #       "filt" (opcional, por defecto, (0.01, 50))

    st = st_raw.copy()
    
    # Limpieza inicial (Merge)
    if 'method' in config or 'fill_value' in config:
        st.merge(method=config.get('method', 0), fill_value=config.get('fill_value'))

    # Centrar la señal en cero (Demean)
    if config.get('demean', True):
        st.detrend('demean')
        
    # Quitar tendencias lineales (Detrend)
    if config.get('detrend', True):
        st.detrend('linear')
        
    # Suavizar los bordes a cero (Taper)
    if config.get('taper', True):
        st.taper(max_percentage=config.get('taper_fraction', 0.05))

    if config.get('rm_resp'):
        resp_dict = config.get('rm_resp')
        output_units = config.get('output', 'VEL')
        pre_filt = config.get('resp_prefilt', (0.005, 0.01, 35, 40))
        
        for tr in st:
            ch_name = tr.stats.channel
            if ch_name in resp_dict and resp_dict[ch_name] is not None:
                try:
                    inv = resp_dict[ch_name]
                    tr.remove_response(inventory=inv, pre_filt=pre_filt, output=output_units)
                except Exception as e:
                    print(f" Error removiendo respuesta en {ch_name}: {e}")

    elif config.get('rm_sens'):
        sens_dict = config.get('rm_dict')
        for tr in st:
            ch_name = tr.stats.channel
            tr.data = tr.data / sens_dict[ch_name]

    # Prefiltro (prefilt)
    prefilt = config.get('filt', (0.01,35))
    if prefilt:
        st.filter('bandpass', freqmin=prefilt[0], freqmax=prefilt[1], zerophase=True)

     # Remuestreo
    target_sr = config.get('sample_rate', None)
    if target_sr:
        st.resample(target_sr)
        
    return st


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


def psd_multitaper(data, fs, nw=4.0, kspec=None):

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