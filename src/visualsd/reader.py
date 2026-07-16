#!/usr/bin/python3

import os
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

    # Asegurar que las fechas sean UTCDateTime para ObsPy
    if not isinstance(starttime, UTCDateTime): 
        starttime = UTCDateTime(starttime)

    if not isinstance(endtime, UTCDateTime): 
        endtime = UTCDateTime(endtime)

    # Calcular numero de dias
    day_diff = (endtime.datetime.date() - starttime.datetime.date()).days 
    date_list = [starttime.datetime + dt.timedelta(days=i) for i in
    range(day_diff + 1)]
    
    # Buscar los archivos
    files_to_read = []
    for date in date_list:
        file_path = sds_filepath(station, channel, date)
        if os.path.isfile(file_path):
            files_to_read.append(file_path)
            
    if not files_to_read:
        print(f" >> No data found for {station.code} between {starttime} and {endtime}")
        return None

    # Leer y unir
    st = Stream()
    try:
        for f in files_to_read:
            st += read(f, starttime=starttime, endtime=endtime)
            
        if not st:
            return None
            
        # Limpiar huecos pequeños y solapamientos
        st.merge(method=1, fill_value='None') 
        
        return st
    
    except Exception as e:
        print(f" >> Error reading data: {e}")
        return None