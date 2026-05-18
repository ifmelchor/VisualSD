#!/usr/bin/env python3
# coding=utf-8

import pyqtgraph as pg
from PyQt5.QtCore import QRectF
import numpy as np


def plot_trace_pg(trace, plot_item_trac, spec_data=None, plot_item_spec=None, psd_data=None, plot_item_psd=None, color='w', vmin=None, vmax=None, cmap="Spectral_r"):
    
    """
    Dibuja una traza de ObsPy y opcionalmente su espectrograma y PSD en PyQtGraph.
    
    :param trace: Objeto Trace de ObsPy.
    :param plot_item_trac: El PlotItem principal donde va la forma de onda.
    :param spec_data: Tupla (freqs, spec_mat) con los datos del espectrograma.
    :param plot_item_spec: El PlotItem secundario donde va el espectrograma.
    :param psd_data: Tupla (f_psd, pxx_psd) con los datos del PSD.
    :param plot_item_psd: El PlotItem donde va el PSD.
    """

    plot_item_trac.clear()
    if plot_item_spec: plot_item_spec.clear()
    if plot_item_psd: plot_item_psd.clear()

    time_vector = trace.times("timestamp")
    t_start = time_vector[0]
    t_span = time_vector[-1] - time_vector[0]
    
    # 1. Dibujar Traza
    curve = plot_item_trac.plot(time_vector, trace.data, pen=pg.mkPen(color=color, width=1), autoDownsample=True, clipToView=True)
    plot_item_trac.setLabel('left', trace.stats.channel)

    # Ocultar eje X superior solo si hay espectrograma
    if spec_data is not None and plot_item_spec is not None:
        plot_item_trac.hideAxis('bottom')

    img = None
    psd_curve = None

    # Dibujar Espectrograma (Opcional)
    if spec_data is not None and plot_item_spec is not None:
        freqs, spec_mat = spec_data
        
        # Sincroniza zoom horizontal
        plot_item_spec.setXLink(plot_item_trac)
        
        # Invierte el eje Y
        plot_item_spec.invertY(True)

        img = pg.ImageItem()
        
        # Filtrar infinitos y NaNs
        mask = np.isfinite(spec_mat)
        if np.any(mask):
            vmin = vmin if vmin is not None else np.percentile(spec_mat[mask], 5)
            vmax = vmax if vmax is not None else np.percentile(spec_mat[mask], 95)

            img.setImage(spec_mat.T, autoLevels=False, levels=[vmin, vmax])

            # Posicionar y escalar la imagen
            f_start = freqs[0]
            f_span = freqs[-1] - freqs[0]
            img.setRect(QRectF(t_start, f_start, t_span, f_span))

            # Aplicar el colormap
            colormap = pg.colormap.getFromMatplotlib(cmap)
            img.setLookupTable(colormap.getLookupTable())

            plot_item_spec.addItem(img)
            plot_item_spec.setLabel('left', 'Freq [Hz]')
            plot_item_spec.setYRange(f_start, freqs[-1])
            
    # Dibujar PSD
    if psd_data is not None and plot_item_psd is not None:
        f_psd, pxx_psd = psd_data
        
        # plot_item_psd.setLogMode(x=False, y=True)

        psd_curve = plot_item_psd.plot(f_psd, pxx_psd, pen=pg.mkPen(color=color, width=1.2))
        
        plot_item_psd.setLabel('bottom', "Freq [Hz]")
        plot_item_psd.setLabel('left', "Power [dB]")
        plot_item_psd.showGrid(x=True, y=True, alpha=0.5)

    return curve, img, psd_curve


def build_beams_widget(aligned_data, beam_stacked, beam_product, fs=100.0, offset_step=2.0, trace_color=(200, 200, 200, 150)):
    """
    Construye el widget de PyQtGraph con los 3 paneles sincronizados.
    :return: (gl_widget, axes_list)
    """

    N, M = aligned_data.shape
    time_vector = np.arange(M) / fs
    
    glw = pg.GraphicsLayoutWidget(title="Beam Plot")
    glw.setBackground('k')
    
    ax0 = glw.addPlot(row=0, col=0, title="Stacked traces")
    ax1 = glw.addPlot(row=1, col=0, title="Stacked Beam")
    ax2 = glw.addPlot(row=2, col=0, title="Product Beam")
    axes_list = [ax0, ax1, ax2]
    
    # Sincronizar el eje X
    ax1.setXLink(ax0)
    ax2.setXLink(ax0)

    # Vectorización completa: normalización + offset vertical
    max_vals = np.max(np.abs(aligned_data), axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    y_offsets = (aligned_data / max_vals) + np.arange(N)[:, np.newaxis] * offset_step

    x_plot = np.tile(np.append(time_vector, np.nan), N)[:-1]
    y_plot = np.append(y_offsets, np.nan, axis=1).ravel()[:-1]

    ax0.plot(x_plot, y_plot, pen=pg.mkPen(color=trace_color, width=1), connect='finite', autoDownsample=True)
    ax0.hideAxis('left')
    ax0.setLabel('bottom', "Tiempo", units='s')

    ax1.plot(time_vector, beam_stacked, pen=pg.mkPen('w', width=1.5))
    ax2.plot(time_vector, beam_product, pen=pg.mkPen('r', width=1.5))
    ax2.setLabel('bottom', "Tiempo", units='s')

    for ax in axes_list:
        ax.showGrid(x=True, y=False, alpha=0.3)
        ax.setMouseEnabled(y=False)  # Recomendado para beams sísmicos
        
    return glw, axes_list


def plot_stream_pg(stream, gl_widget=None, **kwargs):
    """
    Dibuja múltiples trazas apiladas usando PyQtGraph.
    
    :param stream: Objeto Stream de ObsPy.
    :param gl_widget: Instancia de pg.GraphicsLayoutWidget (tu canvas). Si es None, crea uno.
    :param kwargs: Argumentos extra (color, vmin, vmax, etc.) que se pasan a plot_trace_pg.
    """

    n_traces = len(stream)

    if n_traces == 0:
        return None, []

    # Si no nos pasan un lienzo, creamos uno nuevo (útil para pruebas sueltas)
    if gl_widget is None:
        gl_widget = pg.GraphicsLayoutWidget()

    # Limpiamos todo lo que hubiera dibujado antes (equivalente a fig.clf())
    gl_widget.clear()

    axes_list = []
    main_ax = None

    for i, trace in enumerate(stream):
        # 1. Crear un eje X especial
        date_axis = pg.DateAxisItem(orientation='bottom')
        
        # Añadir el PlotItem a la grilla
        ax = gl_widget.addPlot(row=i, col=0, axisItems={'bottom': date_axis})
        
        # Vincular los ejes
        if i == 0:
            main_ax = ax
        else:
            ax.setXLink(main_ax)
        
        # Inyectar el plot_item en nuestra función atómica
        plot_trace_pg(trace, plot_item_trac=ax, **kwargs)

        # Ocultar el eje X en todas las trazas excepto en la última
        if i < n_traces - 1:
            ax.hideAxis('bottom')
        
        # Mostrar una grilla tenue para guiar la vista
        ax.showGrid(x=True, y=False, alpha=0.3)
        
        axes_list.append(ax)

    # Ajustar el espacio entre gráficos
    gl_widget.layout.setSpacing(5)

    return gl_widget, axes_list