#!/usr/bin/python3

import os
import json
import numpy as np
from pathlib import Path
from obspy import Stream, UTCDateTime
import datetime as dt
import matplotlib.pyplot as plt

def load_network(net_code):

    net_path = os.environ.get("VISUALSD_PATH", None)

    if not net_path:
        # ask user for the net_path nad warning that is not at bashrc!
        print(f"\n [!] WARNING: 'VISUALSD_PATH' is not defined in your .bashrc (environment variables).")
        manual_path = input(" >>> Please, enter the VISUALSD_PATH: ").strip()

        if not manual_path:
            print(" [X] Error: No path provided. Aborting.")
            return None

    base_dir = Path(net_path)
    net_dir = base_dir / net_code
    expected_file = net_dir / f"{net_code}.json"
    
    if expected_file.is_file():
        with open(expected_file, 'r') as f:
            network_dict = json.load(f)
            
        network_dict["net_file"] = str(expected_file)
        network_dict["net_path"] = str(net_dir)
        return network_dict
    
    print(f"\n >> visualSD network directory: {base_dir}")
    print(f" >> no network (JSON) file found for '{net_code}'")
    
    available_codes = []
    if base_dir.is_dir():
        for d in base_dir.iterdir():
            if d.is_dir() and (d / f"{d.name}.json").is_file():
                available_codes.append(d.name)
                
    print(f"            Loaded networks are: {available_codes}")
    
    return None


def stream2array(stream, id_list):
    """
    Convierte un Stream de ObsPy en una matriz NumPy 2D alineada.
    
    :param stream: Objeto Stream de ObsPy.
    :param id_list: Lista de strings con los id esperados (net.sta.loc.chan).
    :return: (matriz_numpy, lista_canales_faltantes)
    """
    
    # Si el stream está vacío, devolvemos None
    if not stream:
        return [], None

    max_npts = max([len(tr.data) for tr in stream])
    ncha     = len(id_list)
    array = np.full((max_npts, ncha), np.nan)
    
    valid_codes = []
    for i, code in enumerate(id_list):
        sta, loc, cha = code.split(".")
        tr = stream.select(station=sta, location=loc, channel=cha)

        if not tr:
            print(f" [Warn] :: {code} no se encontró en el Stream")
            continue
        else:
            valid_codes.append(code)
            npts = tr[0].stats.npts
            array[:npts, i] = tr[0].data

    return valid_codes, array


class Network:
    def __init__(self, net, **kwargs):

        self.code = net
        self._net_dict = load_network(net)
        self._stations = {}

        if self._net_dict:
            self.name = self._net_dict.get("name")
            self.net_file = self._net_dict.get("net_file")
            self.net_path = self._net_dict.get("net_path")
            self.sds_path = self._net_dict.get("sds_path")
            self.resp_path = self.net_path + "/resp"
            self._build_stations(**kwargs)

    def __str__(self):
        lines = [
            f"\n{'='*60}",
            f" NETWORK: {self.name} ({self.code})",
            f" SDS PATH: {self.sds_path}",
            f" RESP PATH: {self.resp_path}",
            f" STATIONS LOADED: {len(self._stations)}",
            f"{'='*60}"
        ]
        if self._stations:
            for sta_id, sta_obj in self._stations.items():
                chans = list(sta_obj.channels.keys())
                lat = f"{sta_obj.lat:.5f}" if sta_obj.lat is not None else "N/A"
                lon = f"{sta_obj.lon:.5f}" if sta_obj.lon is not None else "N/A"
                lines.append(f"  [{sta_id:<6}] -> Channels: {str(chans):<20} | Coords: ({lat}, {lon})")
        else:
            lines.append("  No hay estaciones cargadas.")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Network {self.code} ({len(self._stations)} stations)>"

    def _build_stations(self, **kwargs):
        stations_list = self._net_dict.get("stations", [])

        target_code = kwargs.get("target_code", None)
        target_comp = kwargs.get("target_comp", None)

        for sta_info in stations_list:
            code = sta_info.get("code")

            if target_code and target_code != code:
                continue

            loc  = sta_info.get("location")
            sta_id = f"{code}.{loc}" if loc else code

            if sta_id not in self._stations:
                lat   = sta_info.get("lat")
                lon   = sta_info.get("lon")
                elev  = sta_info.get("elevation")
                
                sta = Station(self.code, code, loc, lat, lon, elev, self.sds_path)
                self._stations[sta.id] = sta
            else:
                print(f" [!] Warning: Station '{sta_id}' already exist!")
                sta = self._stations[sta_id]

            channels_dict = sta_info.get("channels", {})

            for ch_code, ch_info in channels_dict.items():

                if target_comp and ch_code[-1] != target_comp:
                    continue
            
                if sta.is_channel(ch_code):
                    print(f" [!] Warning: Channel {channel} already exist in {sta_id}")
                    continue

                sample_rate = ch_info.get("sampling_rate")
                sensitivity = ch_info.get("sensitivity")
                resp_file   = ch_info.get("response_file")

                if resp_file:
                    resp_file = os.path.join(self.resp_path, resp_file)

                chan = Channel(ch_code, sample_rate, sensitivity, resp_file)
                sta.add_channel(chan)

    def __len__(self):
        return len(self._stations)

    def __iter__(self):
        return iter(self._stations.values())

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self._stations.values())[item]
        
        return self._stations.get(item)

    def print_available_dates(self):
        print(f"{'='*50}")
        for sta in self:
            s, e = sta.get_available_dates()
            str_start = s.strftime('%Y-%m-%d %H:%M:%S') if s else "No data   "
            str_end   = e.strftime('%Y-%m-%d %H:%M:%S') if e else "No data   "
            print(f"  [{sta.id:<6}] -> {str_start} to {str_end}")
        print(f"{'='*50}\n")

    def get_position(self, utm=False, plot=False, **plot_kwargs):
        code = []
        lat_list = []
        lon_list = []

        if plot:
            from .plotting import plot_utm
            utm = True

        for sta in self:
            code.append(sta.id)

            if utm:
                pos = sta.get_utm()
                lat_list.append(pos["x"])
                lon_list.append(pos["y"])
            else:
                lat_list.append(sta.lat)
                lon_list.append(sta.lon)

        lat = np.array(lat_list)
        lon = np.array(lon_list)

        if plot:
            plot_utm(code, lat, lon, **plot_kwargs)
        else:
            return code, lat, lon


    def get_stream(self, starttime, endtime, toff, stations=None, component="Z", return_array=False, **kwargs):

        if not stations:
            stations_to_query = self._stations.values()
        else:
            if isinstance(stations, str):
                stations = [stations]
            
            stations_to_query = [self._stations[k] for k in stations if k in self._stations]

        net_stream = Stream()
        valid_id = []

        sta_kwargs = kwargs.copy()
        sta_kwargs["toff"] = toff
        for sta in stations_to_query:
            st = sta.get_stream(starttime, endtime, **sta_kwargs)
            if st:
                for tr in st:
                    if tr.stats.channel[-1] == component:
                        net_stream.append(tr)
                        valid_id.append(sta.id + "." + tr.stats.channel)

        if return_array:
            valid_id, array = stream2array(net_stream, valid_id)
            return valid_id, array

        return net_stream


class Station:
    def __init__(self, net, code, loc, lat, lon, elev, sds_path):
        self.net = net
        self.code = code
        self.id = ".".join((code, loc)) if loc else code
        self.loc = loc
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.channels = {}
        self.sds_path = sds_path
        self._start_date = None
        self._end_date = None
        self._utm = None

    def __repr__(self):
        return f"<Station {self.net}.{self.id} ({len(self.channels)} channels)>"

    def __str__(self):
        # Representación detallada y legible para cuando haces print(estacion)
        lat_str = f"{self.lat:.5f}" if self.lat is not None else "N/A"
        lon_str = f"{self.lon:.5f}" if self.lon is not None else "N/A"
        elev_str = f"{self.elev:.1f} m" if self.elev is not None else "N/A"
        
        chans_list = list(self.channels.keys())
        chans_str = ", ".join(chans_list) if chans_list else "None"

        # Formato de "tarjeta de presentación" de la estación
        lines = [
            f" Station: {self.id} (Network: {self.net})",
            f"   Coordinates : ({lat_str}, {lon_str})",
            f"   Elevation   : {elev_str}",
            f"   Channels    : [{chans_str}]"
        ]
        
        # Si ya se calcularon las fechas, las mostramos
        if self._start_date or self._end_date:
            s_str = self._start_date.strftime('%Y-%m-%d') if self._start_date else "N/A"
            e_str = self._end_date.strftime('%Y-%m-%d') if self._end_date else "N/A"
            lines.append(f"   Data range  : {s_str} to {e_str}")

        return "\n".join(lines)

    def is_channel(self, chan):
        return chan in self.channels

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self.channels.values())[item]

    def add_channel(self, chan_oj):
        self.channels[chan_oj.channel] = chan_oj

    def get_available_dates(self):
        from .reader import scan_available_dates
        if self._start_date is None:
            self._start_date, self._end_date = scan_available_dates(self)
        
        return self._start_date, self._end_date

    def get_stream(self, starttime, endtime, channel=None, pad=10, return_array=False, **kwargs):

        from .reader import get_station_stream
        from .signal import stream_preprocess

        # 1. IDENTIFICA CANALES
        if isinstance(channel, list):
            ch_names = channel
        elif channel:
            ch_names = [channel]
        else:
            ch_names = list(self.channels.keys())

        valid_channels = []
        for ch in ch_names:
            if self.is_channel(ch):
                valid_channels.append(ch)
            else:
                print(f" >>> Channel '{ch}' not found in {self.id}")

        if not valid_channels:
            return None

        ncha = len(valid_channels)

        pad = dt.timedelta(minutes=pad)
        t_start_pad = starttime - pad
        t_end_pad   = endtime + pad

        # Lectura de datos crudos
        st_raw = Stream()
        for ch in valid_channels:
            st_ch = get_station_stream(self, ch, t_start_pad, t_end_pad)
            if st_ch:
                st_raw += st_ch

        if not st_raw:
            print(f" [{self.code}] No data to read.")
            return None

        # Verificar tasa de muestreo
        sample_rates = list(set([tr.stats.sampling_rate for tr in st_raw]))
        if len(sample_rates) > 1:
            
            print(f" Imposible realizar un preprocesado automatico con multiples frecuencias de muestreo.")
            
            for tr in st_raw:
                print(tr.stats.id, tr.stats.sampling_rate)

            st_final = st_raw.slice(UTCDateTime(starttime), UTCDateTime(endtime))

            return st_final

        fs = sample_rates[0]

        # prepara el archivo de configuracion
        config_kwargs = kwargs.copy()
        rm_sens = config_kwargs.get('rm_sens', True)
        rm_resp =config_kwargs.get('rm_resp', False)

        if rm_sens or rm_resp:
            sens_dict = {}
            for ch in self:
                if ch.channel in valid_channels:
                    if rm_resp:
                        sens_dict[ch.channel] = ch.get_resp()
                    else:
                        sens_dict[ch.channel] = ch.sensitivity

            if not sens_dict:
                config_kwargs['rm_sens'] = False
                config_kwargs['rm_resp'] = False
            else:
                config_kwargs['rm_dict'] = sens_dict

        st_processed = stream_preprocess(st_raw, config=config_kwargs)

        # Cortar el pad
        toff = config_kwargs.get('toff', 0) # in seconds
        toff = dt.timedelta(seconds=toff)
        st_final = st_processed.slice(UTCDateTime(starttime-toff), UTCDateTime(endtime+toff))

        if return_array:
            for i in range(len(valid_channels)):
                valid_channels[i] += self.id + "."
            
            valid, array = stream2array(st_final, valid_channels)
            return valid, array
        else:
            return st_final

    def get_utm(self):
        if not self._utm:
            import utm
            (x, y, code, nro) = utm.from_latlon(self.lat, self.lon)
            self._utm = {"x":x, "y":y, "code":code, "nro":nro}
        return self._utm


class Channel:
    def __init__(self, name, sample_rate, sensitivity, resp_file):

        self.channel = name
        self.component = name[-1]
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity # should be counts/UNITS
        self.resp_file = resp_file
        self._resp = None

    def __repr__(self):
        return f"<Channel {self.channel} | {self.sample_rate}Hz>"

    def __getitem__(self, item):
        return self.channels[item]

    def __iter__(self):
        return iter(self.channels)

    def __len__(self):
        return len(self.channels)

    def get_resp(self):
        if self.resp_file:

            if self._resp:
                return self._resp

            else:
                from obspy import read_inventory
                self._resp = read_inventory(self.resp_file)
                return self._resp


class Array(Network):
    def __init__(self, net, code, component="Z"):
        super().__init__(net, target_code=code, target_comp=component)
        self.net  = net
        self.code = code
        self.component = component
        self._reindex_by_loc()

        # Variables para Lazy Loading de Julia
        self._jl_loaded = False
        self._jl = None

    def _reindex_by_loc(self):
        new_stations = {}
        for sta in self._stations.values():
            loc_key = sta.loc if sta.loc else ""
            new_stations[loc_key] = sta
        self._stations = new_stations

    def __repr__(self):
        return f"<Array {self.code} ({len(self._stations)} stations)>"

    def delay_matrix(self, sx, sy):
        """
        Calcula delays usando las coordenadas de este Array.
        :param sx, sy: Lentitud (s/km).
        """

        from .array import delay_matrix

        _, posx, posy = self.get_position(utm=True)

        return delay_matrix(sx, sy, posx/1000, posy/1000, is_samples=False)

    def response(self, slomax, fmin, fmax, sloinc=0.01, finc=0.05, plot=True):

        from .array import array_transfunc

        _, posx, posy = self.get_position(utm=True)

        s_vals, power = array_transfunc(posx=posx/1000, posy=posy/1000,
            slomax=slomax, sloinc=sloinc, fmin=fmin, fmax=fmax, finc=finc
        )

        if not plot:
            return s_vals, power

        from .plotting import slowmap

        _, axes = slowmap(power, s_vals, s_vals, v_min=0, v_max=power.max())
        axes[0].set_title(f"{self.code} • f={fmin}-{fmax} Hz • N={len(self)}")

        plt.show()

        return

    def zlcc(self, starttime, data, fs, lwin, nadv, fmin, fmax, slowmax, tof, **zlcc_kwargs):

        from .array import parse_julia_dict, ZLCCResult

        # Cargar la librería en el entorno de Julia
        if not self._jl_loaded:
            from juliacall import Main as jl 
            jl.seval("using SeisArrays")
            self._jl = jl
            self._jl_loaded = True

        _, posx, posy = self.get_position(utm=True)

        x_jl   = self._jl.Array(posx/1000)
        y_jl   = self._jl.Array(posy/1000)
        sa_jl  = self._jl.SeisArray2D(x_jl, y_jl, self._jl.Array(data), fs)
        ans_jl = self._jl.zlcc(sa_jl, lwin, nadv, fmin, fmax, slowmax, tof, **zlcc_kwargs)
        ans_py = parse_julia_dict(ans_jl)

        slowint = zlcc_kwargs.get("slowint_f", 0.01)
        ccerr   = zlcc_kwargs.get("ccerr", 0.95)

        zlcc_ob = ZLCCResult(starttime, posx/1000, posy/1000, fs, lwin, slowmax, slowint, tof, ccerr, ans_py)

        return zlcc_ob

    def get_beam(self, data, sx, sy, fs, **kwargs):

        from .array import compute_beams_matrix

        delay = self.delay_matrix(sx, sy)

        return compute_beams_matrix(data, delay, fs, **kwargs)

