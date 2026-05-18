#!/usr/bin/env python3
# coding=utf-8

import os
import sys
import numpy as np
import pyqtgraph as pg
import pandas as pd
import datetime as dt
from PyQt5 import QtWidgets, QtCore
from .utils import NavigatePG

pg.setConfigOption('background', '#121212') 
pg.setConfigOption('foreground', '#D1D1D1')

class NavigateZLCC(NavigatePG):
    def on_mouse_clicked(self, evt):
        """Sobrescribe el clic para usar los botones laterales del ratón (Atrás/Adelante)."""
        for ax in self.axes:
            if ax.sceneBoundingRect().contains(evt.scenePos()):
                mousePoint = ax.vb.mapSceneToView(evt.scenePos())
                x_val = mousePoint.x()

                # Doble clic izquierdo -> Resetear el Zoom
                if evt.double() and evt.button() == QtCore.Qt.LeftButton:
                    for a in self.axes:
                        a.enableAutoRange(axis=pg.ViewBox.XAxis)
                    evt.accept()
                    break

                # === BOTONES LATERALES DEL RATÓN ===
                
                # Botón Lateral 1 (Atrás / BackButton) -> Tick Rojo (Izquierdo)
                if evt.button() in (QtCore.Qt.BackButton, QtCore.Qt.ExtraButton1):
                    self.draw_tick(x_val, 'left')
                    evt.accept()

                # Botón Lateral 2 (Adelante / ForwardButton) -> Tick Verde (Derecho)
                elif evt.button() in (QtCore.Qt.ForwardButton, QtCore.Qt.ExtraButton2):
                    self.draw_tick(x_val, 'right')
                    evt.accept()

                break

    def draw_tick(self, x_val, which):
        self.ticks[which] = x_val
        for line in self.tick_lines[which]:
            line.setPos(x_val)
            line.setVisible(True)


class ZLCCViewer(QtWidgets.QMainWindow):
    def __init__(self, zlcc_result, data, pad_sec, outpath):
        super().__init__()
        self.ans     = zlcc_result
        self.data    = data
        self.pad_sec = pad_sec
        self.outpath = outpath
        
        self.setWindowTitle("ZLCC Interactive Viewer")
        self.resize(1000, 800)
        
        # Layout
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)

        # LÍMITES HORIZONTALES
        self.t_min = self.ans.time_s.min()
        self.t_max = self.ans.time_s.max()
        t_pad_min = -self.pad_sec
        t_pad_max = (self.ans.lwin / self.ans.fs) + self.pad_sec
        
        # --- FILA 0: MAAC Y BEAMS ---
        self.p_mb = self.win.addPlot(row=0, col=0, colspan=2)
        self.p_mb.setMouseEnabled(x=True, y=False)
        self.p_mb.setLabel('left', "MAAC", color='#00E5FF')
        self.p_mb.setLabel('right', "Beam Power [dB]", color='#FFD740')

        # eje izq.
        self.scatter_maac = pg.ScatterPlotItem(x=self.ans.time_s, y=self.ans.maac, 
            pen=None, symbol='o', brush='#00E5FF', size=9)
        self.p_mb.addItem(self.scatter_maac)
        self.p_mb.setYRange(0, 1, padding=0.05)

        # eje der.
        self.vb_beam = pg.ViewBox()
        self.p_mb.scene().addItem(self.vb_beam)
        self.p_mb.getAxis('right').linkToView(self.vb_beam)
        self.vb_beam.setXLink(self.p_mb)

        self.scatter_bmax = pg.ScatterPlotItem(x=self.ans.time_s, y=self.ans.beam_max, 
                                               symbol='t', brush='#FF5252', size=9)
        self.scatter_bavg = pg.ScatterPlotItem(x=self.ans.time_s, y=self.ans.beam, 
                                               symbol='o', brush='#FFD740', size=9)
        # self.scatter_bmax = pg.ScatterPlotItem(x=self.ans.time_s, y=self.ans.beam_max, 
        #     symbol='t', brush='#5DADE2', size=9)
        # self.scatter_bavg = pg.ScatterPlotItem(x=self.ans.time_s, y=self.ans.beam, 
        #     symbol='o', brush='#DAA520', size=9)

        self.vb_beam.addItem(self.scatter_bmax)
        self.vb_beam.addItem(self.scatter_bavg)

        self.p_mb.setLimits(xMin=self.t_min, xMax=self.t_max, yMin=0, yMax=1)
        
        # --- FILA 1: SLOWNESS ---
        y_baz = self.ans.baz[:, 1]
        y_slow = self.ans.slow[:, 1]

        self.p_slow= self.win.addPlot(row=1, col=0, colspan=2)
        self.p_slow.setMouseEnabled(x=True, y=False)
        self.p_slow.setXLink(self.p_mb)

        # eje izq
        self.p_slow.setLabel('left', "BAZ [deg]", color='#BB86FC')
        self.scatter_baz = pg.ScatterPlotItem(x=self.ans.time_s, y=y_baz, 
                                              brush='#BB86FC', size=9)
        self.p_slow.addItem(self.scatter_baz)
        self.p_slow.setYRange(0, 360)
        
        # eje der.
        self.p_slow.setLabel('right', "Slowness [s/km]", color='#03DAC6')
        self.vb_slow = pg.ViewBox()
        self.p_slow.scene().addItem(self.vb_slow)
        self.p_slow.getAxis('right').linkToView(self.vb_slow)
        self.vb_slow.setXLink(self.p_slow)

        self.scatter_slow = pg.ScatterPlotItem(x=self.ans.time_s, y=y_slow, 
                                               symbol='s', brush='#03DAC6', size=9)
        self.vb_slow.addItem(self.scatter_slow)

        # === LÍNEAS DE RESALTADO (HIGHLIGHT) ===
        pen_hl = pg.mkPen('m', width=2, style=QtCore.Qt.SolidLine)
        self.hl_line0 = pg.InfiniteLine(angle=90, movable=False, pen=pen_hl)
        self.hl_line1 = pg.InfiniteLine(angle=90, movable=False, pen=pen_hl)
        
        self.p_mb.addItem(self.hl_line0, ignoreBounds=True)
        self.p_slow.addItem(self.hl_line1, ignoreBounds=True)

        # --- FILA 2: Trazas y Slowmap ---
        self.p_traces = self.win.addPlot(row=2, col=0)
        self.p_traces.setMouseEnabled(x=True, y=False)
        self.p_traces.setLabel('bottom', "Time [s]")
        self.p_traces.showGrid(x=True, y=True, alpha=0.2)
        self.p_traces.setLimits(xMin=t_pad_min, xMax=t_pad_max)
        self.p_traces.setXRange(t_pad_min, t_pad_max, padding=0)

        num_traces = self.data.shape[1] 
        self.trace_curves = []
        for _ in range(num_traces):
            c = self.p_traces.plot(pen=pg.mkPen(color='#00FFFF', width=1.5))
            self.trace_curves.append(c)

        self.p_slowmap = self.win.addPlot(row=2, col=1, title="Slowmap")
        self.p_slowmap.setAspectLocked(True)
        self.p_slowmap.showGrid(x=True, y=True, alpha=0.3)
        self.p_slowmap.setLabel('bottom', "Slow X [s/km]")
        self.p_slowmap.setLabel('left', "Slow Y [s/km]")

        self.img_slowmap = pg.ImageItem()
        self.p_slowmap.addItem(self.img_slowmap)
        self.img_slowmap.setColorMap(pg.colormap.get('CET-R3'))

        self.iso_slowmap = pg.IsocurveItem(pen=pg.mkPen('r', width=1.5))
        self.iso_slowmap.setParentItem(self.img_slowmap)

        self.line_slowmap = pg.PlotDataItem(pen=pg.mkPen('k', style=QtCore.Qt.DashLine, width=1))
        self.p_slowmap.addItem(self.line_slowmap)

        self.scatter_max = pg.ScatterPlotItem(symbol='o', brush='r', pen='k', size=8, zValue=10)
        self.scatter_origin = pg.ScatterPlotItem(x=[0], y=[0], symbol='o', brush='k', pen='k', size=8, zValue=10)
        self.p_slowmap.addItem(self.scatter_max)
        self.p_slowmap.addItem(self.scatter_origin)

        # Layout Stretch
        grid_layout = self.win.ci.layout
        grid_layout.setRowStretchFactor(0, 1)
        grid_layout.setRowStretchFactor(1, 1)
        grid_layout.setRowStretchFactor(2, 2)
        grid_layout.setColumnStretchFactor(0, 3)
        grid_layout.setColumnStretchFactor(1, 1)

        # Conectar eventos
        self.scatter_maac.sigClicked.connect(self.on_scatter_clicked)
        self.scatter_bmax.sigClicked.connect(self.on_scatter_clicked)
        self.scatter_bavg.sigClicked.connect(self.on_scatter_clicked)
        self.scatter_baz.sigClicked.connect(self.on_scatter_clicked)
        self.scatter_slow.sigClicked.connect(self.on_scatter_clicked)

        self.navigator1 = NavigateZLCC(self.win, [self.p_mb, self.p_slow])
        self.navigator2 = NavigateZLCC(self.win, [self.p_traces])

        self.p_mb.vb.sigResized.connect(self.updateViews)
        self.p_slow.vb.sigResized.connect(self.updateViews)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Cargar el índice 0 por defecto
        self.update_lower_plots(0)

    def updateViews(self):
        self.vb_beam.setGeometry(self.p_mb.vb.sceneBoundingRect())
        self.vb_slow.setGeometry(self.p_slow.vb.sceneBoundingRect())

    def keyPressEvent(self, event):
        """Detecta cuando se presiona la tecla W."""
        if event.key() == QtCore.Qt.Key_W:
            self.export_selection_to_csv()

        # Flecha Derecha -> Siguiente índice
        elif event.key() == QtCore.Qt.Key_Right:
            if self.current_index < len(self.ans.time_s) - 1:
                self.update_lower_plots(self.current_index + 1)
                
        # Flecha Izquierda -> Índice anterior
        elif event.key() == QtCore.Qt.Key_Left:
            if self.current_index > 0:
                self.update_lower_plots(self.current_index - 1)

        elif event.key() == QtCore.Qt.Key_1 or event.key() == QtCore.Qt.Key_2:
            for ax in self.navigator1.axes:
                if ax.sceneBoundingRect().contains(event.scenePos()):
                    mousePoint = ax.vb.mapSceneToView(event.scenePos())
                    x_val = mousePoint.x()

                    if event.key() == QtCore.Qt.Key_1:
                        self.navigator1.draw_tick(x_val, 'right')
                    else:
                        self.navigator1.draw_tick(x_val, 'left')

        super().keyPressEvent(event)

    def export_selection_to_csv(self):
        """Guarda automáticamente en ./zlcc_out/ sin preguntar."""
        t1 = self.navigator1.ticks['left']
        t2 = self.navigator1.ticks['right']

        if t1 is None or t2 is None:
            print("Error: Marque inicio y fin con los botones laterales del ratón.")
            return

        # Preparar datos
        t_start, t_end = sorted([t1, t2])
        mask = (self.ans.time_s >= t_start) & (self.ans.time_s <= t_end)
        rel_time = self.ans.time_s[mask]
        abs_time = self.ans.starttime + pd.to_timedelta(rel_time, unit='s')

        data_to_save = {
            'absolute_time': abs_time,
            'maac': self.ans.maac[mask],
            'beam_max': self.ans.beam_max[mask],
            'beam_avg': self.ans.beam[mask]
        }

        data_to_save['baz_min']  = self.ans.baz[mask, 0]
        data_to_save['baz'] = self.ans.baz[mask, 1]
        data_to_save['baz_max']  = self.ans.baz[mask, 2]

        data_to_save['slow_min']  = self.ans.slow[mask, 0]
        data_to_save['slow_mean'] = self.ans.slow[mask, 1]
        data_to_save['slow_max']  = self.ans.slow[mask, 2]

        df = pd.DataFrame(data_to_save)

        timestamp = abs_time[0].strftime("%Y%m%d_%H%M%S")
        filename = f"zlcc_export_{timestamp}.csv"
        filepath = os.path.join(self.outpath, filename)

        # Crear carpeta si no existe
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        df.to_csv(filepath, index=False)

    def on_scatter_clicked(self, plot, points):
        point = points[0]
        clicked_time = point.pos().x()
        idx = np.argmin(np.abs(self.ans.time_s - clicked_time))
        self.update_lower_plots(idx)
    
    def update_lower_plots(self, index):
        self.current_index = index

        # Actualizar la posición de las líneas de Highlight
        current_time = self.ans.time_s[index]
        self.hl_line0.setPos(current_time)
        self.hl_line1.setPos(current_time)

        # ===================
        # ACTUALIZAR SLOWMAP
        # ===================
        if hasattr(self.ans, 'slowmap'):
            slomap = self.ans.slowmap[index]
            slowxy = self.ans.s_vals
            
            slomap_xy = slomap
            self.img_slowmap.setImage(slomap_xy, autoLevels=True)
            
            s_min, s_max = slowxy.min(), slowxy.max()
            self.img_slowmap.setRect(QtCore.QRectF(s_min, s_min, s_max - s_min, s_max - s_min))

            maacth = slomap_xy.max() * self.ans.ccerr
            self.iso_slowmap.setData(slomap_xy)
            self.iso_slowmap.setLevel(maacth)

            maxpos = np.unravel_index(np.argmax(slomap_xy), slomap_xy.shape)
            idx_x, idx_y = maxpos[0], maxpos[1]
            
            # Bug solucionado: usar slowxy para ambos
            peak_x = slowxy[idx_x]
            peak_y = slowxy[idx_y]

            self.line_slowmap.setData([0, peak_x], [0, peak_y])
            self.scatter_max.setData(x=[peak_x], y=[peak_y])

        # ===================
        # ACTUALIZAR TRAZAS
        # ===================
        try:
            aligned, _, _ = self.ans.stack_traces(self.data, index, pad_sec=self.pad_sec)
            
            n_samples = aligned.shape[0]
            num_traces = aligned.shape[1]
            t_axis = np.linspace(-self.pad_sec, (self.ans.lwin / self.ans.fs) + self.pad_sec, n_samples)

            for i in range(num_traces):
                trace_data = aligned[:, i]
                trace_centered = trace_data - np.mean(trace_data)
                self.trace_curves[i].setData(t_axis, trace_centered)

            y_max = np.max(np.abs(aligned)) * 1.1
            if y_max > 0:
                self.p_traces.setLimits(yMin=-y_max, yMax=y_max)
                self.p_traces.setYRange(-y_max, y_max, padding=0)
                    
        except Exception as e:
            print(f"No se pudo cargar la traza para el índice {index}: {e}")


def init_ZLCCViewer(*args):
    app = QtWidgets.QApplication.instance()
    must_run_loop = (app is None)
    
    if app is None:
        app = QtWidgets.QApplication(sys.argv) 

    viewer = ZLCCViewer(*args)
    viewer.show()
    
    if must_run_loop:
        app.exec_()
        
    return viewer