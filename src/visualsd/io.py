#!/usr/bin/python3

import os
import numpy as np
import datetime as dt

def save_zlcc(filepath, ans):
    """
    Guarda los resultados del ZLCC y las trazas originales en un archivo .npz
    """

    out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    # metadata
    save_dict = {
        'code':      ans.id,
        'starttime': str(ans.starttime),
        'data':      ans.data,
        'slowmax':   ans.slowmax,
        'slowint':   ans.slowint,
        'fmin':      ans.fmin,
        'fmax':      ans.fmax,
        'posx':      ans.posx,
        'posy':      ans.posy,
        'fs':        ans.fs,
        'lwin':      ans.lwin,
        'toff':      ans.toff,
        'ccerr':     ans.ccerr,
    }

    atributos_resultados = [
        'time_s', 'maac', 'beam_max', 'beam', 
        'baz', 'slow', 's_ratio', 'baz_width',
        'fpeak', 'slow_width', 'sx', 'sy', 'smap'
    ]

    for attr in atributos_resultados:
        if hasattr(ans, attr):
            save_dict[attr] = getattr(ans, attr)

    np.savez_compressed(filepath, **save_dict)


def load_zlcc(filepath):
    """
    Carga un archivo .npz y reconstruye los objetos necesarios para el visor.
    """

    with np.load(filepath, allow_pickle=True) as loader:
        data = {k: loader[k] for k in loader.files}

    data['starttime'] = dt.datetime.fromisoformat(str(data['starttime']))

    for key in ['code', 'fmin', 'fmax','fs', 'lwin', 'slowmax', 'slowint', 'toff', 'ccerr']:
        if key in data and isinstance(data[key], np.ndarray):
            data[key] = data[key].item()

    return data