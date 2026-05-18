#!/usr/bin/env python3
# coding=utf-8

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph
import datetime as dt


class MultiStreamWidget(QtWidgets.QWidget):
    def __init__(self, network, starttime, id_list, id_ref=0, interval=15, **kwargs):
        
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()

        # Configuración de datos
        self.network = network
        self.id_list = sta_chan_list # ej: [("STA1", "HHZ"), ("STA2", "HHZ")]

        if isinstance(id_ref, int)
            self.id_ref  = self.id_list[id_ref]   # ej: ("STA1", "HHZ") para el dayplot
        else:
            if id_ref in self.id_list:
                self.id_ref = id_ref
            else:
                raise ValueError(" id_ref is not correct")

        self.sta_kwargs = kwargs
        
        # Configuración de tiempo
        self.interval = dt.timedelta(minutes=interval)
        self.starttime = starttime
        self.endtime   = starttime + self.interval

        # Lienzos (Canvas)
        self.stream_canvas = MultiStreamCanvas(self)
        self.layout.addWidget(self.stream_canvas)
        self.setLayout(self.layout)
        
        # Ventana del Dayplot (puede iniciar oculta o mostrarse con "d")
        self.dayplot_widget = None
        self.setCursor(QtCore.Qt.CrossCursor)
        self.stream_canvas.setFocus()


    def plot(self, new_starttime):
        self.starttime = new_starttime
        self.endtime   = self.starttime + self.interval
        self.stream_canvas.plot()

        # 2. Actualiza la ventana de contexto (Dayplot) marcando el cuadro rojo
        if self.dayplot_widget and self.dayplot_widget.isVisible():
            self.dayplot_widget.update_highlight(self.starttime, self.endtime)

    def on_key(self, key):
        if key == QtCore.Qt.Key_Right:
            self.plot(self.starttime + self.interval)
        
        elif key == QtCore.Qt.Key_Left:
            self.plot(self.starttime - self.interval)
        
        elif key == "d":
            if not self.dayplot_widget:
                self.dayplot_widget = DayPlotWidget(self)
            
            self.dayplot_widget.show()
            self.dayplot_widget.update_highlight(self.starttime, self.endtime)


class MultiStreamCanvas(FigureCanvas):
    def __init__(self, parent):
        self.parent = parent
        self.fig = Figure(figsize=(12, 8), constrained_layout=True)
        super().__init__(self.fig)
        
        self.mpl_connect('key_press_event', self.on_key)
        self.plot()

    def plot(self):
        with pyqtgraph.BusyCursor():

            # st = self.parent.network.get_empty_stream() # Asumiendo un método para iniciar un Stream vacío
            
            for sta, cha in self.parent.sta_chan_list:
                tr = self.parent.network.get_trace(
                    sta, cha, 
                    self.parent.starttime, 
                    self.parent.endtime, 
                    **self.parent.sta_kwargs
                )
                if tr:
                    st += tr

            # Llamamos a tu función orquestadora de Matplotlib
            self.fig, self.axes = plot_stream(st, time_mode='datetime', fig=self.fig)
            self.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.parent.on_key(QtCore.Qt.Key_Right)
        
        elif event.key == 'left':
            self.parent.on_key(QtCore.Qt.Key_Left)
        
        elif event.key == 'd':
            self.parent.on_key("d")

