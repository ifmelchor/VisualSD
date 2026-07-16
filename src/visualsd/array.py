#!/usr/bin/env python3
# coding=utf-8

import os
import copy
import numpy as np

def array_transfunc(posx, posy, slomax, sloinc, fmin, fmax, finc):
    
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    n_sta = len(posx)
    
    # Evita división repetida en el bucle
    inv_nsta_sq = 1.0 / (n_sta * n_sta)
    
    freqs = np.arange(fmin, fmax + finc * 0.5, finc)
    omegas = 2.0 * np.pi * freqs
    s_vals = np.arange(-slomax, slomax + sloinc * 0.5, sloinc)
    n_s = len(s_vals)
    
    SX, SY = np.meshgrid(s_vals, s_vals, indexing='ij')
    slomax2 = slomax ** 2
    valid_mask = (SX**2 + SY**2) <= slomax2
    
    sx_valid = SX[valid_mask]
    sy_valid = SY[valid_mask]
    n_valid = len(sx_valid)
    
    power_valid = np.zeros(n_valid, dtype=np.float64)
    
    # Bucle sobre frecuencias
    for w in omegas:
        delay = sx_valid[:, None] * posx + sy_valid[:, None] * posy
        beam_sum = np.sum(np.exp(1j * w * delay), axis=1)
        power_valid += np.abs(beam_sum) ** 2 * inv_nsta_sq
        
    power_valid /= len(omegas)
    
    # Reconstruir matriz
    power = np.zeros((n_s, n_s), dtype=np.float64)
    power[valid_mask] = power_valid
    
    return s_vals, power


def delay_matrix(sx, sy, posx, posy, is_samples=False, fs=100.0):
    
    """
    Construye la matriz antisimétrica de delays (en segundos) a partir de un vector de lentitud.

    :param sx, sy: Lentitud (s/km).
    :param posx, posy: Arrays 1D con las coordenadas de cada estación (km).
    :param is_samples: Booleano. True si deseas muestras
    :param fs: Frecuencia de muestreo
    """
    
    t_abs = (sx * posx) + (sy * posy)
    delay = t_abs[:, np.newaxis] - t_abs[np.newaxis, :]

    if is_samples:
        delay /= fs

    return delay


def stack_traces(data, delay, fs, ref_index=0, normalize_product=True):
    """
    Calcula el Linear y Product Beam usando operaciones puras de matrices.
    
    :param data: Array 2D (M, N) con N trazas y M muestras temporales.
    :param delay: Matriz 2D antisimétrica (N, N) de desfases en segundos.
    :param fs: Frecuencia de muestreo en Hz
    :param ref_index: Índice de la fila que servirá como ancla (default 0).
    :param normalize_product: Normaliza antes de multiplicar para evitar under/overflow.
    
    :return: (beam_stacked, beam_product) - Arrays 1D de longitud M.
    """

    M, N = data.shape
    aligned_data = np.zeros_like(data)
    
    for i in range(N):
        # Desfase de la traza 'i' respecto a la referencia
        shift_sec = -delay[i, ref_index]
        n_shift = int(round(shift_sec * fs))
        row_data = data[:, i]
        
        # Alinear usando slicing en el eje M
        if n_shift > 0:    # Retrasar
            length = min(M, M - n_shift)
            if length > 0:
                aligned_data[n_shift : n_shift + length, i] = row_data[:length]
        
        elif n_shift < 0:  # Adelantar
            length = min(M + n_shift, M)
            if length > 0:
                aligned_data[:length, i] = row_data[-n_shift : -n_shift + length]
        
        else:
            aligned_data[:, i] = row_data
    
    beam_stacked = np.mean(aligned_data, axis=1)
    
    if normalize_product:
        # Normalización robusta
        max_vals = np.max(np.abs(aligned_data), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        norm_data = aligned_data / max_vals

        sign_prod = np.prod(np.sign(norm_data), axis=1)
        log_abs_sum = np.sum(np.log(np.abs(norm_data) + 1e-15), axis=1)
        beam_product = sign_prod * np.exp(log_abs_sum)
    else:
        beam_product = np.prod(aligned_data, axis=1)

    return aligned_data, beam_stacked, beam_product


def parse_zlcc_output(jl, julia_output):
    """
    Convierte ZLCCOutput (struct Julia) a diccionario Python/NumPy.
      - Vectores / Matrices numéricas  → np.ndarray
      - Vector{Union{Matrix,Nothing}}  → lista de ndarray | None
      - NaN Julia                      → np.nan
    """

    def _is_nothing(value) -> bool:
        return value is None or type(value).__name__ in ('NothingValue', 'Nothing')

    def _fix_shape(arr: np.ndarray) -> np.ndarray:
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

