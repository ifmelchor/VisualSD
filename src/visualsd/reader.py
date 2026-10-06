#!/usr/bin/python3

import os
import numpy as np
import datetime as dt
from pathlib import Path
from obspy import read, UTCDateTime, Stream

def sds_filepath(station, channel, date):
    """
    Ruta estándar SDS para un día específico.
    """
    year = date.year
    yday = date.timetuple().tm_yday
    
    # Extraemos info del objeto station
    net_code = station.net
    sta_code = station.code
    loc_code = station.loc if station.loc else ""
    sds_path = station.sds_path
    
    # Formato: NET.STA.LOC.CHAN.D.YEAR.JDAY
    file_name = f"{net_code}.{sta_code}.{loc_code}.{channel}.D.{year}.{yday:03d}"
    file_path = os.path.join(sds_path, str(year), net_code, sta_code, f"{channel}.D", file_name)
    
    return file_path


def scan_available_dates(station):
    """
    Escanea la ruta SDS y devuelve la fecha más antigua y más reciente.
    """
    start = dt.datetime(3000, 1, 1)
    end   = dt.datetime(1000, 1, 1)
    
    # Crea el objeto Path
    sds_path = Path(station.sds_path)
    net_code = station.net
    sta_code = station.code
    loc_code = station.loc if station.loc else ""

    for chan_name, chan in station.channels.items():
        search_pattern = f"*/{net_code}/{sta_code}/{chan_name}.D/{net_code}.{sta_code}.{loc_code}.{chan_name}.D.*.*"
        
        flist = []
        for p in sds_path.glob(search_pattern):
            if p.is_file() and len(p.name.split('.')[-1]) == 3:
                flist.append(p)
        
        if not flist:
            continue
            
        # Ordenamos por nombre de archivo
        datelist = sorted(flist)
        
        first_file = str(datelist[0])
        last_file = str(datelist[-1])

        st_start = read(first_file, headonly=True)
        st_end   = read(last_file, headonly=True)

        starttime = st_start[0].stats.starttime.datetime
        endtime   = st_end[0].stats.endtime.datetime
        
        if starttime < start: start = starttime
        if endtime > end: end = endtime

    if start.year == 3000: 
        return None, None
        
    return start, end


def get_station_stream(station, channel, starttime, endtime):
    """
    Devuelve un Stream con UN segmento continuo por traza (una sola traza si no hay huecos, varias si los hay), o None si no hay archivos o datos.
    """

    t0, t1 = UTCDateTime(starttime), UTCDateTime(endtime)
    tag    = f"{station.id}.{channel}"

    day0, day1 = t0.datetime.date(), t1.datetime.date()
    files = [sds_filepath(station, channel, day0 + dt.timedelta(days=i)) for i in range((day1 - day0).days + 1)]
    files = [f for f in files if os.path.isfile(f)]

    if not files:
        print(f" >> [{tag}] sin archivos entre {t0} y {t1}")
        return None

    st = Stream()
    for f in files:
        try:
            st += read(f, starttime=t0, endtime=t1)
        except Exception as e:
            print(f" >> [{tag}] error leyendo {f}: {e}")
            return None

    st = st.select(channel=channel)

    if not st:
        print(f" >> [{tag}] sin datos entre {t0} y {t1}")
        return None

    st.merge(method=1, fill_value=None)
    st = st.split()
    st.sort(keys=["starttime"])
    return st