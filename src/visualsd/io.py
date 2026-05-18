#!/usr/bin/python3

from os import path
import numpy as np
import datetime as dt

def save_zlcc(filepath, data, slowmax, slowint, ans):
    """
    Guarda los resultados del ZLCC y las trazas originales en un archivo .npz
    """

    out_dir = path.dirname(filepath)
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # metadata
    save_dict = {
        'starttime': str(ans.starttime),
        'data': data,
        'slowmax':slowmax,
        'slowint':slowint,
        'posx': ans.posx,
        'posy': ans.posy,
        'fs': ans.fs,
        'lwin': ans.lwin,
        'toff': ans.toff,
        'ccerr': ans.ccerr,
    }

    atributos_resultados = [
        'time_s', 'maac', 'beam_max', 'beam', 
        'baz', 'slow', 'slowmap', 'baz_width',
        'fpeak', 'slow_width', 'sx', 'sy'
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

    st_str = str(data['starttime'])
    data['starttime'] = dt.datetime.fromisoformat(st_str)

    for key in ['fs', 'lwin', 'slowmax', 'slowint', 'toff', 'ccerr']:
        if key in data and isinstance(data[key], np.ndarray):
            data[key] = data[key].item()

    return data