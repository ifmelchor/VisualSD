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

    print(f" Leyendo {args.input}...")
    ans = ZLCCResult.from_file(args.input)

    app = QtWidgets.QApplication(sys.argv)

    outpath = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.input))
    
    viewer = ZLCCViewer(ans, args.pad_sec, outpath)
    
    viewer.show()
    sys.exit(app.exec_())