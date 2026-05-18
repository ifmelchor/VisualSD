#!/usr/bin/env python3
# coding=utf-8

import pyqtgraph as pg
import datetime as dt
from PyQt5 import QtCore
from obspy import UTCDateTime

P_WEIGHT_MAP = {'p': 0, '1': 1, '2': 2, '3': 3, '4': 4}
S_WEIGHT_MAP = {'s': 0, '5': 1, '6': 2, '7': 3, '8': 4}
C_WEIGHT_MAP = {'c': None}

P_ERASE_KEY = 'P'
S_ERASE_KEY = 'S'
C_ERASE_KEY = 'C'

P_PICKER_KEYS = list(P_WEIGHT_MAP.keys()) + [P_ERASE_KEY]
S_PICKER_KEYS = list(S_WEIGHT_MAP.keys()) + [S_ERASE_KEY]
C_PICKER_KEYS = list(C_WEIGHT_MAP.keys()) + [C_ERASE_KEY]

ONSET_KEYS    = ["u", "d", "U", "D"]


class NavigatePG:
    def __init__(self, gl_widget, axes_list):
        """
        Clase de navegación y "picking" optimizada para PyQtGraph.
        :param gl_widget: GraphicsLayoutWidget
        :param axes_list: Lista de PlotItems
        """

        self.gl_widget = gl_widget
        self.axes = axes_list
        
        # Desactivamos el menú por defecto
        for ax in self.axes:
            ax.setMenuEnabled(False)

        # --- CONFIGURACIÓN DEL CURSOR ---
        self.vLines = []
        self.hLines = []
        
        for ax in self.axes:
            vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', style=QtCore.Qt.DashLine, alpha=0.5))
            hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', style=QtCore.Qt.DashLine, alpha=0.5))
            ax.addItem(vLine, ignoreBounds=True)
            ax.addItem(hLine, ignoreBounds=True)
            self.vLines.append(vLine)
            self.hLines.append(hLine)

        # --- CONFIGURACIÓN DE LOS TICKS (L Y R) ---
        self.ticks = {'left': None, 'right': None}
        self.tick_lines = {'left': [], 'right': []}

        # Creamos las líneas rojas y verdes pero las mantenemos ocultas hasta hacer clic
        for ax in self.axes:
            left_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5))
            right_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=1.5))
            left_line.setVisible(False)
            right_line.setVisible(False)
            ax.addItem(left_line, ignoreBounds=True)
            ax.addItem(right_line, ignoreBounds=True)
            self.tick_lines['left'].append(left_line)
            self.tick_lines['right'].append(right_line)

        # --- CONEXIÓN DE EVENTOS ---
        # SignalProxy optimiza el movimiento del mouse
        self.proxy = pg.SignalProxy(self.gl_widget.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_moved)

        self.gl_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)


    def on_mouse_moved(self, evt):
        """Mueve el cursor (crosshair) siguiendo el ratón."""
        pos = evt[0]  # Posición en la escena global
        
        for i, ax in enumerate(self.axes):
            if ax.sceneBoundingRect().contains(pos):
                # Traducir pixeles a coordenadas de datos (tiempo y amplitud)
                mousePoint = ax.vb.mapSceneToView(pos)
                
                # Mover TODAS las líneas verticales
                for v in self.vLines:
                    v.setPos(mousePoint.x())
                
                # Mover solo la horizontal
                for j, h in enumerate(self.hLines):
                    if i == j:
                        h.setPos(mousePoint.y())
                        h.setVisible(True)
                    else:
                        h.setVisible(False)


    def on_mouse_clicked(self, evt):
        """Registra el clic para los Ticks L y R, ignora si el usuario está arrastrando."""
        for ax in self.axes:
            if ax.sceneBoundingRect().contains(evt.scenePos()):
                mousePoint = ax.vb.mapSceneToView(evt.scenePos())
                x_val = mousePoint.x()

                # Doble clic -> Resetear el Zoom
                if evt.double():
                    for a in self.axes:
                        a.autoRange()
                    evt.accept()

                # Clic Izquierdo -> Tick Rojo
                elif evt.button() == QtCore.Qt.LeftButton:
                    self.ticks['left'] = x_val
                    for line in self.tick_lines['left']:
                        line.setPos(x_val)
                        line.setVisible(True)
                    evt.accept()

                # Clic Derecho -> Tick Verde
                elif evt.button() == QtCore.Qt.RightButton:
                    self.ticks['right'] = x_val
                    for line in self.tick_lines['right']:
                        line.setPos(x_val)
                        line.setVisible(True)
                    evt.accept()

                break


    def reset_ticks(self):
        """Oculta las marcas y limpia los datos."""
        self.ticks = {'left': None, 'right': None}
        for line in self.tick_lines['left']:
            line.setVisible(False)
        for line in self.tick_lines['right']:
            line.setVisible(False)


    def get_time_ticks(self):
        l_tick = self.ticks['left']
        r_tick = self.ticks['right']

        l_dt = dt.datetime.fromtimestamp(l_tick, tz=dt.timezone.utc).replace(tzinfo=None) if l_tick is not None else None
        r_dt = dt.datetime.fromtimestamp(r_tick, tz=dt.timezone.utc).replace(tzinfo=None) if r_tick is not None else None

        return l_dt, r_dt


class PickerPG(NavigatePG):
    def __init__(self, gl_widget, axes_list, phase_dict, on_pick_callback=None, **kwargs):
        """
        :param on_pick_callback: Una función que se ejecutará cada vez que se cree o borre un pick.
        """
        super().__init__(gl_widget, axes_list)
        
        self.phase = phase_dict
        self.on_pick_callback = on_pick_callback 

        self.current_x = None
        self.phase_colors = kwargs.get("phase_colors", {"P": "r", "S": "g", "C": "b"})

        self.init_phase_artists()

        # Llamalo en el Widget, desde afuera!
        # self.draw_initial_phases()

    def init_phase_artists(self):
        """Crea los objetos visuales para P, S y F una sola vez."""
        for wave in ["P", "S", "C"]:
            color = self.phase_colors[wave]
            self.phase[wave]["artist"] = []
            
            # Línea vertical en cada canal
            for ax in self.axes:
                line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color, width=1.5))
                line.setVisible(False)
                ax.addItem(line)
                self.phase[wave]["artist"].append(line)
            
            # Texto descriptivo (se pone en el eje superior)
            text = pg.TextItem(text="", color=color, anchor=(0, 1))
            text.setVisible(False)
            self.axes[0].addItem(text)
            self.phase[wave]["artist_text"] = text


    def draw_initial_phases(self):
        """Lee el diccionario self.phase y dibuja las líneas si hay datos."""

        for wave in ["P", "S", "C"]:
            p_data = self.phase[wave]
            
            # Si hay un tiempo guardado, activamos la visualización
            if p_data.get("time") is not None:
                # Convertimos el datetime a timestamp
                t_unix = UTCDateTime(p_data["time"]).timestamp
                
                # Posicionar y mostrar líneas en todos los canales
                for line in p_data["artist"]:
                    line.setPos(t_unix)
                    line.setVisible(True)
                
                # 2. Posicionar y mostrar el texto
                txt = self.phase_text(wave, p_data)
                p_data["artist_text"].setText(txt)

                # Lo ponemos arriba del todo en el eje Y
                y_max = self.axes[0].viewRange()[1][1] 
                p_data["artist_text"].setPos(t_unix, y_max)
                p_data["artist_text"].setVisible(True)


    def on_mouse_moved(self, evt):
        """Sobrescribe el método del padre para capturar la X actual para el picker."""
        super().on_mouse_moved(evt)
        
        # Guarda la posición X exacta para cuando el usuario presione una tecla
        pos = evt[0]
        
        for ax in self.axes:
            if ax.sceneBoundingRect().contains(pos):
                self.current_x = ax.vb.mapSceneToView(pos).x()
                break


    def _pick_phase(self, wave, key, erase_key, weight_map, t_dt):
        if key == erase_key:
            if self.phase[wave]["time"] is not None:
                self.clear_phase(wave)
                return True

        else:
            w = weight_map[key]
            if self.phase[wave]["time"] != t_dt or self.phase[wave]["weight"] != w:
                self.phase[wave].update({"time": t_dt, "weight": w})
                self.update_phase_ui(wave)
                return True

        return False


    def keyPressEvent(self, event):
        if self.current_x is None:
            return

        key = event.text()
        t_dt = dt.datetime.fromtimestamp(self.current_x, tz=dt.timezone.utc).replace(tzinfo=None)
        changed = False

        # --- LÓGICA PARA FASES P/S ---
        if key in P_PICKER_KEYS:
            changed = self._pick_phase("P", key, "P", P_WEIGHT_MAP, t_dt)
        
        elif key in S_PICKER_KEYS:
            changed = self._pick_phase("S", key, "S", S_WEIGHT_MAP, t_dt)

        # --- LÓGICA PARA FASE C (Coda) ---
        elif key in C_PICKER_KEYS:
            changed = self._pick_phase("C", key, "C", C_WEIGHT_MAP, t_dt)

        # --- LÓGICA PARA ONSET / POLARIDAD ---
        elif key in ONSET_KEYS:
            # Solo podemos ponerle polaridad a la P si ya está picada
            if self.phase["P"]["time"] is not None:
                letra = key.upper()
                current_onset = self.phase["P"].get("onset", "")
                if current_onset == letra:
                    self.phase["P"]["onset"] = ""
                else:
                    self.phase["P"]["onset"] = letra

                self.update_phase_ui("P")
                changed = True

        # Solo si hubo un cambio real, disparamos el callback
        if changed and self.on_pick_callback:
            self.on_pick_callback(self.phase)


    def update_phase_ui(self, wave):
        """Actualiza la posición de las líneas de fase."""
        p = self.phase[wave]
        t_unix = UTCDateTime(p["time"]).timestamp
        
        for line in p["artist"]:
            line.setPos(t_unix)
            line.setVisible(True)
        
        txt = self.phase_text(wave, p)
        p["artist_text"].setText(txt)
        p["artist_text"].setPos(t_unix, self.axes[0].viewRange()[1][1])
        p["artist_text"].setVisible(True)


    def clear_phase(self, wave):
        """Borra visualmente y limpia datos."""
        for line in self.phase[wave]["artist"]:
            line.setVisible(False)

        self.phase[wave]["artist_text"].setVisible(False)
        self.phase[wave]["time"] = None

        if wave in ("P", "S"):
            self.phase[wave]["weight"] = None

        if wave == "P":
            self.phase[wave]["onset"] = ""


    @staticmethod
    def phase_text(wave, p):
        weight_val = p.get('weight')
        weight_str = str(weight_val) if weight_val is not None else ""
        
        if wave == "P":
            onset = str(p.get('onset') or '')
            return f"{onset}P{weight_str}"
            
        elif wave == "S":
            return f"S{weight_str}"
            
        elif wave == "C":
            return "C"
            
        return wave