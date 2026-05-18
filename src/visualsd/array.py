#!/usr/bin/env python3
# coding=utf-8

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


def parse_julia_dict(julia_dict):
    """
    Convierte un diccionario proveniente de Julia a estructuras nativas de Python y NumPy.
    
    Reglas de conversión:
    - Vectores y Matrices numéricas -> numpy.ndarray
    - Union{Nothing, Matrix} -> Lista de Python con (numpy.ndarray o None)
    - NaN de Julia -> numpy.nan
    - Cadenas y escalares -> str, float, int nativos
    """
    python_dict = {}
    
    try:
        items = julia_dict.items()
    
    except AttributeError:
        items = dict(julia_dict).items()

    for key, value in items:
        k_str = str(key) 

        if isinstance(value, dict) or type(value).__name__ in ['Dict', 'PyDict', 'juliacall.DictValue']:
            python_dict[k_str] = parse_julia_dict(value)
            continue

        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                # 1. Lo pasamos a array genérico primero
                arr = np.array(value)
                
                # 2. ¡EL TRUCO PARA EL SLOWMAP!
                # Si NumPy lo dejó como un array de objetos (ej. (39,) con matrices adentro),
                # forzamos la extracción y lo apilamos en un cubo 3D (39, 201, 201)
                if arr.dtype == object:
                    arr = np.stack([np.asarray(v, dtype=float) for v in arr])
                else:
                    arr = np.asarray(arr, dtype=float)
                
                # 3. Quitar dimensión fantasma si viene como (N, 3, 1) -> (N, 3)
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[:, :, 0] 
                
                # 4. Aplanar estrictamente a 1D si es vector (N, 1) o (1, N) -> (N,)
                if arr.ndim == 2 and (arr.shape[1] == 1 or arr.shape[0] == 1):
                    arr = arr.ravel()
                        
                python_dict[k_str] = arr
                
            except Exception:
                # Si algo de lo anterior falla (ej. si es una lista de strings o tipos raros)
                python_dict[k_str] = value
            continue
            
        python_dict[k_str] = value
            
    return python_dict


class ZLCCResult:
    """
    Clase para almacenar, analizar y graficar los resultados de ZLCC.
    """
    def __init__(self, starttime, posx, posy, fs, lwin, slowmax, slowint, toff, ccerr, result_dict):
        self.starttime = starttime
        self.fs     = fs
        self.lwin   = lwin
        self.toff   = toff
        self.posx   = posx
        self.posy   = posy
        self.ccerr  = ccerr
        self.s_vals = np.arange(-slowmax, slowmax + slowint * 0.5, slowint)
        self.n_max  = len(result_dict["maac"])

        if self.n_max > 0:
            self.__dict__.update(result_dict)

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

        data = load_zlcc(filepath)

        try:
            starttime = data.pop('starttime')
            posx = data.pop('posx')
            posy = data.pop('posy')
            fs = float(data.pop('fs'))
            lwin = int(data.pop('lwin'))
            slowmax = float(data.pop('slowmax'))
            slowint = float(data.pop('slowint'))
            toff = float(data.pop('toff'))
            ccerr = float(data.pop('ccerr'))

        except KeyError as e:
            raise KeyError(f"El archivo no contiene el metadato necesario: {e}")

        return cls(starttime, posx, posy, fs, lwin, slowmax, slowint, toff, ccerr, data)

    def delay(self, index):
        sx = self.sx[index]
        sy = self.sy[index]
        return delay_matrix(sx, sy, self.posx, self.posy, is_samples=False, fs=self.fs)

    def plot_slowmap(self, index, **kwargs):
        """
        Grafica el slowmap si está disponible en los resultados.
        """

        if hasattr(self, 'slowmap'):

            import matplotlib.pyplot as plt
            from .plotting import slowmap

            power = self.slowmap[index]
            kwargs["ccerr"] = self.ccerr
            slowmap(power, self.s_vals, self.s_vals, **kwargs)

            plt.show()

    def stack_traces(self, data, index, pad_sec=0, **kwargs):

        n_start0  =  int(round(self.time_s[index] * self.fs))
        n_start0 += int(self.toff * self.fs)
        n_end0    =  int(round(n_start0 + self.lwin))

        n_pad = int(round(pad_sec * self.fs))
        n_start = max(0, n_start0 - n_pad)
        n_end   = min(data.shape[0], n_end0 + n_pad)

        wdata = data[n_start:n_end, :]
        delay = self.delay(index)
        
        return stack_traces(wdata, delay, self.fs, **kwargs)

    def view(self, data, pad_sec=0.0):
        from .gui.zlcc import init_ZLCCViewer
        self._zlcc_viewer = init_ZLCCViewer(self, data, pad_sec)