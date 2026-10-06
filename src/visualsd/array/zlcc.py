#!/usr/bin/env python3
# coding=utf-8

import os
import copy
import numpy as np


class ZLCC:
    """
    ZLCC sobre un Array del inventario
    """

    def __init__(self, array, station_list=None):
        self.array = array.subarray(station_list) if station_list else array

        ids, x, y = array.positions_km()
        self.ids = ids
        self._xy = {c: (xi, yi) for c, xi, yi in zip(ids, x, y)}


    def min_toff(self, fs, slowmax):
        return np.ceil(slowmax * self.array.aperture * fs + 2) / fs


    def run(self, starttime, endtime, toff, *, lwin, nadv, fmin, fmax, slowmax, maac_th=0.5, ccerr=0.95, slowint_c=0.1, slowint_f=0.01, slowfw=0.5, return_cmap=True, gaps="reject", **kw):
        """
        Lee [starttime, endtime] (+ toff) del (sub)array y corre ZLCC.
        toff=None: el mínimo necesario. gaps: 'reject' | 'interpolate' (ver get_stream).
        **kw va a get_stream (preprocesado). Devuelve un ZLCCResult.
        """

        from ..jlwrap import jl_call

        # get data
        ids, data, t0, fs = self.array.get_stream(starttime, endtime, toff, gaps=gaps, return_array=True, **kw)

        if data is None:
            raise ValueError(f"[{self.array.code}] sin datos entre {starttime} y {endtime}")

        # check toff
        need = self.min_toff(fs, slowmax)
        toff = need if toff is None else round(float(toff) * fs) / fs
        if toff < need - 1e-9:
            raise ValueError(f"toff={toff:.3f} s < mínimo {need:.3f} s (slowmax × apertura)")
        
        # coordenadas
        codes = [i.rsplit(".", 1)[0] for i in ids]
        x, y = np.array([self._xy[c] for c in codes]).T
        if len(codes) < len(self.ids):
            print(f" >> [ZLCC] corre con {len(codes)} de {len(self.ids)} estaciones")

        # load wrapper
        jz = jl_call("ZLCC")
        ans_jl = jz.zlcc(data, x, y, fs=fs, lwin=lwin, nadv=nadv, fmin=fmin, fmax=fmax, slowmax=slowmax, toff=toff, slowint_c=slowint_c, slowint_f=slowint_f, ccerr=ccerr, maac_th=maac_th, slowfw=slowfw, return_cmap=return_cmap)

        ans_py = parse_zlcc_output(jz.jl, ans_jl) if ans_jl is not None else {"maac": np.empty(0)}

        res = ZLCCResult(f"{self.array.code}.{self.array.component}", t0 + toff, fmin, fmax, x, y, fs, lwin, slowmax, slowint_f, toff, ccerr, ans_py, data)
        res.codes = codes
        return res


def parse_zlcc_output(jl, julia_output):
    """
    Convierte ZLCCOutput (struct Julia) a diccionario Python/NumPy.
      - Vectores / Matrices numéricas  → np.ndarray
      - Vector{Union{Matrix,Nothing}}  → lista de ndarray | None
      - NaN Julia                      → np.nan
    """

    def _is_nothing(value):
        return value is None or type(value).__name__ in ('NothingValue', 'Nothing')

    def _fix_shape(arr: np.ndarray):
        """Corrige dimensiones espurias que introduce juliacall."""
        if arr.ndim == 3 and arr.shape[-1] == 1:  # (N, 3, 1) → (N, 3)
            arr = arr[:, :, 0]
        if arr.ndim == 2 and min(arr.shape) == 1:  # (N, 1) o (1, N) → (N,)
            arr = arr.ravel()
        return arr


    def _convert_field(value, fname: str):
        if _is_nothing(value):
            return None

        # smap: Vector{Union{Matrix{T}, Nothing}} → numpy array 3D

        if fname == 'smap':
            elements = [
                None if _is_nothing(elem)
                else _fix_shape(np.asarray(elem, dtype=float))
                for elem in value
            ]
            first = next((e for e in elements if e is not None), None)
            
            if first is None:
                return np.empty((len(elements), 0, 0), dtype=float)
            
            n = first.shape[0]
            arr3d = np.full((len(elements), n, n), np.nan, dtype=float)
            for i, e in enumerate(elements):
                if e is not None:
                    arr3d[i] = e
            
            return arr3d

        # Arrays numéricos (vectores y matrices)
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                return _fix_shape(np.asarray(value, dtype=float))
            except Exception:
                return value

        # Escalares y strings
        return value


    field_names = [str(f) for f in jl.fieldnames(jl.typeof(julia_output))]
    return {
        fname: _convert_field(getattr(julia_output, fname), fname)
        for fname in field_names
    }


class ZLCCResult:
    """
    Clase para almacenar, analizar y graficar los resultados de ZLCC.
    """
    def __init__(self, code, starttime, fmin, fmax, posx, posy, fs, lwin, slowmax, slowint, toff, ccerr, result_dict, data):
        self.id     = code
        self.starttime = starttime
        self.fs     = fs
        self.lwin   = lwin
        self.toff   = toff
        self.fmin   = fmin
        self.fmax   = fmax
        self.posx   = posx
        self.posy   = posy
        self.ccerr  = ccerr
        self.slowmax = slowmax
        self.slowint = slowint
        self.n_max   = len(result_dict["maac"])
        self.data    = data

        if self.n_max > 0:
            self.__dict__.update(result_dict)

        # Variables para Lazy Loading de Julia
        self._jl_loaded = False
        self._jl = None

    @property
    def s_vals(self):
        return np.arange(-self.slowmax, self.slowmax + self.slowint * 0.5, self.slowint)

    def __str__(self):
        st_str = self.starttime.strftime("%Y-%m-%d %H:%M:%S") if hasattr(self, 'starttime') else "N/A"
        
        summary = [
            "--- ZLCC Result Summary ---",
            f"Start Time:        {st_str}",
            f"Windows processed: {self.n_max}",
            f"Sampling Rate:     {self.fs} Hz",
            f"Window Length:     {self.lwin} samples ({self.lwin/self.fs:.2f} s)",
            f"Slowness range:    [{min(self.s_vals):.2f}, {max(self.s_vals):.2f}] s/km",
            f"Stations:          {len(self.posx)}",
            f"CC Error Thresh:   {self.ccerr}",
            "---------------------------"
        ]
        return "\n".join(summary)

    def __repr__(self):
        st_repr = self.starttime.isoformat() if hasattr(self, 'starttime') else "None"
        
        return (f"ZLCCResult(start='{st_repr}', n_windows={self.n_max}, "
                f"fs={self.fs}, lwin={self.lwin}, ccerr={self.ccerr})")

    @classmethod
    def from_file(cls, filepath):
        """Constructor alternativo: Crea una instancia a partir de un archivo."""
        from .io import load_zlcc

        datafile = load_zlcc(filepath)

        try:
            starttime = datafile.pop('starttime')
            code      = datafile.pop('code')
            raw_data  = datafile.pop('data')
            posx      = datafile.pop('posx')
            posy      = datafile.pop('posy')
            fs        = float(datafile.pop('fs'))
            fmin      = float(datafile.pop('fmin'))
            fmax      = float(datafile.pop('fmax'))
            lwin      = int(datafile.pop('lwin'))
            slowmax   = float(datafile.pop('slowmax'))
            slowint   = float(datafile.pop('slowint'))
            toff      = float(datafile.pop('toff'))
            ccerr     = float(datafile.pop('ccerr'))

        except KeyError as e:
            raise KeyError(f"El archivo no contiene el metadato necesario: {e}")

        return cls(code, starttime, fmin, fmax, posx, posy, fs, lwin, slowmax, slowint, toff, ccerr, datafile, raw_data)

    def select(self, **kwargs):
        """
        Filtra las ventanas procesadas utilizando criterios dinámicos.
        Devuelve una nueva instancia de ZLCCResult con los datos filtrados.
        
        Sufijos soportados en kwargs:
        -----------------------------
        _max   : Filtra valores menores o iguales al umbral (ej. baz_width_max=10.5)
        _min   : Filtra valores mayores o iguales al umbral (ej. baz_min=0.5)
        _range : Filtra valores dentro de un rango tupla (min, max) (ej. slow_range=(0.1, 0.4))
        """

        if self.n_max == 0:
            return copy.deepcopy(self)

        # Crear la máscara booleana inicial
        mask = np.ones(self.n_max, dtype=bool)

        # Procesar los kwargs recibidos
        for key, value in kwargs.items():
            if value is None:
                continue

            # Filtro Máximo (_max)
            if key.endswith('_max'):
                attr = key[:-4]
                if hasattr(self, attr):
                    mask &= (getattr(self, attr) <= value)

            # Filtro Mínimo (_min)
            elif key.endswith('_min'):
                attr = key[:-4]
                if hasattr(self, attr):
                    mask &= (getattr(self, attr) >= value)

            # Filtro por Rango (_range)
            elif key.endswith('_range'):
                attr = key[:-6]
                if hasattr(self, attr):
                    data_attr = getattr(self, attr)
                    
                    # Lógica especial si es Backazimuth (baz) y el rango cruza el Norte (ej: [350, 15])
                    if attr == 'baz' and value[0] > value[1]:
                        mask &= (data_attr >= value[0]) | (data_attr <= value[1])
                    else:
                        mask &= (data_attr >= value[0]) & (data_attr <= value[1])

        # 3. Clonar el objeto actual (copia superficial)
        new_result = copy.copy(self)
        
        # 4. Actualizar el conteo de ventanas que sobrevivieron al filtro
        new_result.n_max = int(np.sum(mask))

        # 5. Filtrar los arrays que dependen del número de ventanas (eje 0 == n_max original)
        for attr, val in self.__dict__.items():
            if isinstance(val, np.ndarray) and val.shape[0] == self.n_max:
                if attr not in ['posx', 'posy', 's_vals']: 
                    setattr(new_result, attr, val[mask])

        return new_result

    def delay(self, index):
        sx = self.sx[index]
        sy = self.sy[index]
        return delay_matrix(sx, sy, self.posx, self.posy, is_samples=False, fs=self.fs)

    def plot_slowmap(self, index, **kwargs):
        """
        Grafica el slowmap si está disponible en los resultados.
        """

        if hasattr(self, 'smap') and np.any(np.isfinite(self.smap[index])):

            from .plotting import slowmap
            power = self.smap[index]
            kwargs["ccerr"] = self.ccerr
            ans = slowmap(power, self.s_vals, self.s_vals, **kwargs)

            return ans

    def stack_traces(self, index, pad_sec=0, **kwargs):

        n_start0  =  int(round(self.time_s[index] * self.fs))
        n_start0 += int(self.toff * self.fs)
        n_end0    =  int(round(n_start0 + self.lwin))

        n_pad = int(round(pad_sec * self.fs))
        n_start = max(0, n_start0 - n_pad)
        n_end   = min(self.data.shape[0], n_end0 + n_pad)

        wdata = self.data[n_start:n_end, :]
        delay = self.delay(index)
        
        return stack_traces(wdata, delay, self.fs, **kwargs)

    def stack_slowmap(self, bazw_th, **kwargs):
        # genera el mapa de lentitud stacked y calcula su incertidumbre

        if hasattr(self, 'smap'):

            from .plotting import slowmap

            if not self._jl_loaded:
                from juliacall import Main as jl 
                jl.seval("using SeisArrays")
                self._jl = jl
                self._jl_loaded = True

            baz_mask = self.baz_width <= bazw_th
            valid_mask = baz_mask & np.isfinite(self.smap[:, 0, 0])

            if not np.any(valid_mask):
                raise ValueError("La máscara no selecciona ningún elemento válido.")

            power   = np.mean(self.smap[valid_mask], axis=0)

            kwargs["ccerr"] = self.ccerr
            level = power.max()*self.ccerr
            sx = self._jl.Array(self.s_vals)
            result = self._jl.uncertainty_contour(sx, sx, self._jl.Array(power), level)

            if result is None:
                contans = None

            else:
                contans = {
                    "ratio":      float(result.ratio),
                    "slow":       float(result.slow),
                    "slowmin":    float(result.slowmin),
                    "slowmax":    float(result.slowmax),
                    "slow_width": float(result.sloww),
                    "baz":        float(result.baz),
                    "bazmin":     float(result.bazmin),
                    "bazmax":     float(result.bazmax),
                    "baz_width":  float(result.bazw),
                }

            ans = slowmap(power, self.s_vals, self.s_vals, **kwargs)

            return ans, contans

    def view(self,  pad_sec=0):
        from .gui.zlcc import init_ZLCCViewer
        self._zlcc_viewer = init_ZLCCViewer(self, pad_sec)

    def write(self, filename=None, outpath="./"):
        from visualsd.io import save_zlcc

        if not filename:
            time_str = self.starttime.strftime("%Y%m%d_%H%M")
            filename = f"{self.id}.{time_str}.{self.fmin:g}-{self.fmax:g}.zlcc.{self.lwin/self.fs:g}.npz"

        filepath = os.path.join(outpath, filename)
        save_zlcc(filepath, self)

        print(f" >>> file {filepath} created!")

