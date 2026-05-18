#!/usr/bin/python3

import os
import sys
import argparse
from .array import ZLCCResult
from .gui.zlcc import ZLCCViewer
from PyQt5 import QtWidgets

def zlcc_viewer():
    parser = argparse.ArgumentParser(description="ZLCCViewer - VisualSD")
    
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Ruta al archivo zlcc .npz")

    parser.add_argument("-p", "--pad", type=float, default=1.0, dest="pad_sec",
                        help="Segundos de margen (padding) a los lados de la ventana (defecto: 1.0)")
    
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="Directorio para exportar archivos CSV (defecto: ./)")

    args = parser.parse_args()

    try:
        print(f" Cargando datos de: {args.input}...")
        ans = ZLCCResult.from_file(args.input)
        
        raw_data = getattr(ans, 'data', None)
        
        if raw_data is None:
            print(" Error: El archivo no contiene la matriz de señales 'data'.")
            sys.exit(1)
            
    except Exception as e:
        print(f" Error al abrir el archivo: {e}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)

    outpath = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.input))
    
    print(raw_data.shape)
    viewer = ZLCCViewer(ans, raw_data, args.pad_sec, outpath)
    
    viewer.show()
    sys.exit(app.exec_())