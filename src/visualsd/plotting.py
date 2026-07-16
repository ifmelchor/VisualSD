#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as plg
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.colors as mcolor

def plot_utm(labels, x_coords, y_coords, padding=1.1, dem=None, levels=50):
    x_raw = np.array(x_coords) / 1000 # to km
    y_raw = np.array(y_coords) / 1000 # to km
    
    centro_x = (x_raw.max() + x_raw.min()) / 2
    centro_y = (y_raw.max() + y_raw.min()) / 2
    
    # 2. Trasladar los puntos: el centro ahora es (0,0)
    x_rel = x_raw - centro_x
    y_rel = y_raw - centro_y
    
    # 3. Calcular la apertura máxima para que el cuadro sea simétrico
    max_range = max(np.abs(x_rel).max(), np.abs(y_rel).max()) * padding
    
    # Configuración de la figura
    fig = plt.figure(figsize=(10, 10))
    gs = plg.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    if dem is not None:
        X_dem = dem["X"] / 1000 - centro_x
        Y_dem = dem["Y"] / 1000 - centro_y
        Z_dem = dem["Z"]

        # Recortar al extent del plot para no pintar de más
        mask_x = (X_dem[0, :] >= -max_range) & (X_dem[0, :] <= max_range)
        mask_y = (Y_dem[:, 0] >= -max_range) & (Y_dem[:, 0] <= max_range)
        X_crop = X_dem[np.ix_(mask_y, mask_x)]
        Y_crop = Y_dem[np.ix_(mask_y, mask_x)]
        Z_crop = Z_dem[np.ix_(mask_y, mask_x)]

        levels = np.arange(np.nanmin(Z_crop), np.nanmax(Z_crop), levels)
        cs = ax.contour(X_crop, Y_crop, Z_crop, levels=levels, colors="k", linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%d m")
    
    # 4. Dibujar puntos relativos
    ax.scatter(x_rel, y_rel, color='red', s=60, edgecolors='black', zorder=3)
    
    # Dibujar líneas de eje en el centro (cruz de referencia)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    
    # Etiquetas de los puntos
    for i, label in enumerate(labels):
        ax.annotate(label, (x_rel[i], y_rel[i]), xytext=(5, 5), 
                    textcoords="offset points", fontsize=9)
    
    # 5. Ajustar límites simétricos
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    # Mantener escala 1:1
    ax.set_aspect('equal')
    
    # Estética y títulos
    ax.set_xlabel('Oeste-Este [km]')
    ax.set_ylabel('Norte-Sur [km]')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()

    return fig, ax

def plotgram(y, array, x, axis=None, v_min=None, v_max=None, cmap='Spectral_r', interpolation='spline36', aspect='auto', axis_bar=None, orientation='vertical'):
    
    """
    Optimized for plotting spectrograms and polargrams using Matplotlib.
    
    Parameters:
    :param y: 1D array of Y-axis coordinates (e.g., frequencies).
    :param array: 2D numpy array of data (freq,time).
    :param x: 1D array of X-axis coordinates (dates or linear).
    :param axis: Matplotlib axis object.
    :param v_min, v_max: Min/Max for color normalization (uses min/max if None).
    :param cmap: Colormap name.
    :param interpolation: Interpolation method for imshow.
    :param aspect: Aspect ratio ('auto' or 'equal').
    :param axis_bar: Optional axis to draw the colorbar.
    :param orientation: Colorbar orientation ('vertical' or 'horizontal').
    :param kwargs: Additional arguments passed to imshow or axis configuration.
    """

    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError('array must be a 2D numpy ndarray (M, N)')

    if axis is None:
        fig = plt.figure(figsize=(10, 6))
        gs = plg.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.05, hspace=0.05)
        axis = fig.add_subplot(gs[0, 0])
        axis_bar = fig.add_subplot(gs[0, 1])

    is_time = isinstance(x[0], (dt.datetime, np.datetime64))

    if is_time:
        xt = mdates.date2num(x)
    else:
        xt = x

    extent = [xt[0], xt[-1], y[0], y[-1]]

    mask = np.isfinite(array)
    if not np.any(mask):
        return None

    if v_min is None:
        v_min = np.percentile(array[mask], 5)
    
    if v_max is None:
        v_max = np.percentile(array[mask], 95)

    norm = mcolor.Normalize(v_min, v_max)

    im = axis.imshow(np.flipud(array), cmap=cmap, norm=norm, interpolation=interpolation, extent=extent, aspect=aspect)

    axis.axis('tight')
    axis.grid(which="major", color="k", ls="-", alpha=0.2, zorder=4)

    if is_time:
        axis.xaxis_date()
        axis.get_figure().autofmt_xdate()

    if axis_bar:
        orientation = kwargs.get('orientation', 'vertical')
        cbar = plt.colorbar(im, cax=axis_bar, orientation=orientation)
        cbar.locator = mtick.MaxNLocator(nbins=4)
        cbar.update_ticks()
        return im, cbar

    return im


def slowmap(slomap, slowx, slowy, axis=None, axis_bar=None, v_min=0, v_max=1, ccerr=0.9, cmap='Spectral_r', interpolation='spline36', aspect='auto', orientation='vertical'):

    """
    Dibuja un mapa de lentitud (slowness map) en un eje de Matplotlib proporcionado.

    Parameters:
    :param slomap: 2D numpy array con los valores del mapa.
    :param slowx: 1D numpy array con las coordenadas X (lentitud en X).
    :param slowy: 1D numpy array con las coordenadas Y (lentitud en Y).
    :param axis: Matplotlib axis object donde se dibujará el mapa
    :param ccerr: Coeficiente de error para el contorno (default 0.9).
    :param axis_bar: Matplotlib axis object opcional para dibujar la barra de color.
    :param cmap: Mapa de colores.
    :param interpolation: Método de interpolación para imshow.
    :param kwargs: Argumentos adicionales (title, bar_label, orientation).
    """

    if axis is None:
        fig = plt.figure(figsize=(6, 6))
        gs = plg.GridSpec(1, 2, width_ratios=[1, 0.02], wspace=0.05, hspace=0.05)
        axis = fig.add_subplot(gs[0, 0])
        axis_bar = fig.add_subplot(gs[0, 1])
        axes = [axis, axis_bar]

    extent = [slowx.min(), slowx.max(), slowy.min(), slowy.max()]
    slomap = slomap.T

    im = axis.imshow(slomap, cmap=cmap, interpolation=interpolation, extent=extent, aspect=aspect, vmin=v_min, vmax=v_max, origin='lower', rasterized=True, zorder=1)
    
    # Dibujar contornos
    maacth = slomap.max() * ccerr
    axis.contour(slowx, slowy, slomap, levels=[maacth], colors="r")

    maxpos = np.where(slomap == slomap.max())
    idx_y, idx_x = maxpos[0][0], maxpos[1][0]
    slov0x = np.linspace(0, slowx[idx_x], 100)
    slov0y = np.linspace(0, slowy[idx_y], 100)

    axis.plot(slov0x, slov0y, ls="--", lw=0.8, color="k", alpha=0.7, zorder=2)
    axis.scatter(slowx[idx_x], slowy[idx_y], marker="o", color="r", ec="k", zorder=3)
    axis.scatter(0, 0, marker="o", color="k", ec="k", zorder=3)

    sloint = np.abs(slowx[1] - slowx[0])
    axis.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axis.xaxis.set_minor_locator(mtick.AutoMinorLocator(2))
    axis.xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(mtick.MaxNLocator(nbins=5))

    axis.grid(which="major", ls="-", lw=0.8, color="k", alpha=0.3, zorder=3)
    axis.grid(which="minor", ls=":", lw=0.5, color="k", alpha=0.3, zorder=3)

    if axis_bar:
        cbar = plt.colorbar(im, cax=axis_bar, orientation=orientation)
        cbar.locator = mtick.MaxNLocator(nbins=4)
        cbar.update_ticks()

        return fig, (im, cbar), axes

    return fig, im, axes


def plot_trace(trace, psd_data=None, spec_data=None, time_mode='datetime', ax_trac=None, ax_psd=None, ax_spec=None, ax_cbar=None, **pg_kwargs):

    if time_mode == 'datetime':
        t0 = np.datetime64(trace.stats.starttime.datetime)
        time_vector = t0 + (trace.times() * 1e6).astype('timedelta64[us]')
    else:
        time_vector = trace.times()

    has_spec = spec_data is not None
    has_psd  = psd_data is not None

    # Layout con GridSpec
    if ax_trac is None:
        fig_width = 13 if has_psd else 10
        fig = plt.figure(figsize=(fig_width, 6))
        
        if has_spec and has_psd:
            gs = plg.GridSpec(2, 3, height_ratios=[1, 1.5], width_ratios=[1, 0.02, 0.3], wspace=0.1, hspace=0.05)

            ax_trac = fig.add_subplot(gs[0, 0])
            ax_psd  = fig.add_subplot(gs[0, 2])
            ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_trac)
            ax_cbar = fig.add_subplot(gs[1, 1])
            
        elif has_spec and not has_psd:
            gs = plg.GridSpec(2, 2, height_ratios=[1, 1.5], width_ratios=[1, 0.02], wspace=0.05, hspace=0.05)
            ax_trac = fig.add_subplot(gs[0, 0])
            ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_trac)
            ax_cbar = fig.add_subplot(gs[1, 1])
            
        elif not has_spec and has_psd:
            gs = plg.GridSpec(1, 2, width_ratios=[1, 0.3], wspace=0.15)
            ax_trac = fig.add_subplot(gs[0])
            ax_psd  = fig.add_subplot(gs[1])
        else:
            ax_trac = fig.add_subplot(111)
    else:
        fig = ax_trac.get_figure()

     # Dibujar Traza
    ax_trac.plot(time_vector, trace.data, color='k', lw=0.5)
    last_ax = ax_trac
    axes_to_return = [ax_trac]

    # Dibujar Espectrograma (Opcional)
    if has_spec and (ax_spec is not None):
        freqs, spec_mat = spec_data
        ax_trac.xaxis.set_major_formatter(mtick.NullFormatter())
        pg_kwargs["axis"] = ax_spec
       
        if ax_cbar is not None:
            pg_kwargs["axis_bar"] = ax_cbar

        plotgram(freqs, spec_mat, time_vector, **pg_kwargs)
        ax_spec.set_ylabel("Freq [Hz]")
        last_ax = ax_spec
        axes_to_return.append(ax_spec)
    
    # Dibujar PSD (Opcional)
    if has_psd and (ax_psd is not None):
        f_psd, pxx_psd = psd_data
        ax_psd.plot(f_psd, pxx_psd, color='k', lw=1.2)
        
        ax_psd.set_xlabel("Freq [Hz]")
        ax_psd.set_ylabel("Power")
        # ax_psd.set_xscale('log')
        ax_psd.grid(True, which="both", ls=":", alpha=0.5, zorder=0)
        axes_to_return.append(ax_psd)

    if time_mode == 'datetime':
        last_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()

    return fig, axes_to_return


def dayplot(trace, red_periods, **kwargs):
    from obspy.imaging.waveform import WaveformPlotting
    from obspy import UTCDateTime

    plot_kwargs = {  
        "type": "dayplot",
        "show": False,
        "color": "k",
        "interval": 60
    }
    plot_kwargs.update(kwargs)
    plot_kwargs["stream"] = trace
    
    wp = WaveformPlotting(**plot_kwargs)
    wp.plot_waveform()

    ax = wp.fig.axes[0]
    t0 = wp.starttime
    interval_sec = wp.interval

    waveform_lines = [
        line for line in ax.lines if len(line.get_xdata()) > 10
    ]
    
    for i, line in enumerate(waveform_lines):
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        row_start_time = t0 + (i * interval_sec)
        row_end_time = row_start_time + interval_sec

        for r_start, r_end in red_periods:
            if r_end < row_start_time or r_start > row_end_time:
                continue
            
            r_start = UTCDateTime(r_start)
            r_end   = UTCDateTime(r_end)
            t_start = max(row_start_time, r_start)
            t_end = min(row_end_time, r_end)
            
            frac_start = (t_start - row_start_time) / interval_sec
            frac_end = (t_end - row_start_time) / interval_sec
            
            x_start = frac_start * wp.width
            x_end = frac_end * wp.width
              
            mask = (xdata >= x_start) & (xdata <= x_end)
            
            if np.any(mask):
                x_seg = xdata[mask]
                y_seg = ydata[mask]
                ax.plot(x_seg, y_seg, color='red', linewidth=1.2)
      
    return wp


def plot_stream(stream, time_mode='datetime', fig=None, **kwargs):
    
    n_traces = len(stream)

    if n_traces == 0:
        return None

    if fig is None:
        fig = plt.figure(figsize=(10, 2 * n_traces))
    
    fig.clf()

    gs = plg.GridSpec(n_traces, 1, hspace=0.1) 
    
    axes_list = []
    main_ax = None

    for i, trace in enumerate(stream):
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            main_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharex=main_ax)
        
        # Llamamos a plot_trace inyectando el eje
        plot_trace(trace, time_mode=time_mode, ax_trac=ax)

        # Optimizamos la visualización: ocultamos etiquetas X excepto en la última traza
        if i < n_traces - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")
        
        axes_list.append(ax)

    fig.tight_layout()

    return fig, axes_list




def plot_beams(aligned_data, beam_stacked, beam_product, fs=100.0):
    """
    Grafica las señales alineadas, el Linear Beam y el Product Beam.
    
    :param aligned_data: Array 2D (N, M) con las trazas ya desfasadas.
    :param beam_stacked: Array 1D de longitud M (Linear Beam).
    :param beam_product: Array 1D de longitud M (Product Beam).
    :param fs: Frecuencia de muestreo en Hz (para el eje temporal).
    :param figsize: Tamaño de la figura.
    """

    N, M = aligned_data.shape
    time_vector = np.arange(M) / fs
    
    # Crear figura con 3 subplots compartiendo el eje X
    figsize = (10,6)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.15})
    
    # ---------------------------------------------------------
    # SEÑALES DESPLAZADAS
    # ---------------------------------------------------------
    ax0 = axes[0]
    offset_step = 2.0  # Separación vertical entre trazas
    
    for i in range(N):
        trace = aligned_data[i, :]
        # Normalizar la traza
        max_val = np.max(np.abs(trace))
        if max_val == 0: 
            max_val = 1.0
        
        norm_trace = trace / max_val
        
        # Graficar con offset vertical
        ax0.plot(time_vector, norm_trace + (i * offset_step), color='k', lw=0.6, alpha=0.7)
        
    ax0.set_title("Shifted traces", loc='left', fontweight='bold', fontsize=11)
    ax0.set_yticks([])
    ax0.grid(True, axis='x', ls=':', alpha=0.6)

    # ---------------------------------------------------------
    # BEAM STACK
    # ---------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(time_vector, beam_stacked, color='k', lw=1.2)
    ax1.set_title("Stacked Beam", loc='left', fontweight='bold', fontsize=11)
    ax1.grid(True, ls=':', alpha=0.6)

    # ---------------------------------------------------------
    # 3. BEAM PRODUCT (Multiplicative)
    # ---------------------------------------------------------
    ax2 = axes[2]
    ax2.plot(time_vector, beam_product, color='red', lw=1.2)
    ax2.set_title("Product Beam", loc='left', fontweight='bold', fontsize=11)
    ax2.set_xlabel("Tiempo [s]", fontsize=10, fontweight='bold')
    ax2.grid(True, ls=':', alpha=0.6)

    ax2.set_xlim(time_vector[0], time_vector[-1])
    
    # plt.tight_layout() puede chocar con gridspec_kw
    return fig, axes