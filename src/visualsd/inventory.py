#!/usr/bin/python3

import os
import json
import copy 
import numpy as np
from functools import cached_property
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
        net_path = manual_path

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
    Stream de ObsPy -> matriz (npts, ncanales) alineada en el tramo común a todas las trazas.
      id_list: ids esperados (sta.loc.chan o sta.chan), en el orden de las columnas
    Devuelve (ids, data, t0, fs); ([], None, None, None) si no hay ninguna traza.
    """
    trs, ids = [], []
    for code in id_list:
        parts = code.split(".")
        sta, cha = parts[0], parts[-1]
        loc = parts[1] if len(parts) == 3 else ""
        tr = stream.select(station=sta, location=loc, channel=cha) if stream else None
        if not tr:
            print(f" [Warn] :: {code} no se encontró en el Stream")
            continue
        ids.append(code)
        trs.append(tr[0])

    if not trs:
        return [], None, None, None

    rates = {t.stats.sampling_rate for t in trs}
    if len(rates) > 1:
        raise ValueError(f"frecuencias de muestreo distintas entre canales: {sorted(rates)}")
    fs = rates.pop()

    # tramo común
    t0 = max(t.stats.starttime for t in trs)
    t1 = min(t.stats.endtime for t in trs)
    if t1 <= t0:
        print(f" [Warn] :: los canales no se solapan en el tiempo")
        return [], None, None, None

    n = int(round((t1 - t0) * fs)) + 1
    cols = []
    for t in trs:
        i0 = int(round((t0 - t.stats.starttime) * fs))
        cols.append(np.asarray(t.data[i0:i0 + n], dtype=float))
    n = min(len(c) for c in cols)

    return ids, np.column_stack([c[:n] for c in cols]), t0, fs


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
                    print(f" [!] Warning: Channel {ch_code} already exist in {sta_id}")
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
            return plot_utm(code, lat, lon, **plot_kwargs)
        else:
            return code, lat, lon

    def get_aperture(self):

        labels, x_coords, y_coords = self.get_position(utm=True, plot=False)

        num_sensores = len(x_coords)
        if num_sensores < 2:
            print(" >> num_sensores < 2!")
            return

        max_dist = 0.0
        sensor_a_max, sensor_b_max = None, None
        sensor_a, sensor_b = "", ""

        print("\n--- Distancias entre pares de sensores ---")

        x_coords /= 1000
        y_coords /= 1000

        for i in range(num_sensores):
            for j in range(i + 1, num_sensores):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = np.sqrt(dx**2 + dy**2)
                sla = labels[i]
                slb = labels[j]

                print(f" {sla} -- {slb}: {dist:.2f} km")

                if dist > max_dist:
                    max_dist = dist
                    sensor_a = sla
                    sensor_b = slb
                    sensor_a_max = i
                    sensor_b_max = j

        return (sensor_a, sensor_b), max_dist

    def get_stream(self, starttime, endtime, toff, stations=None, component="Z", return_array=False, **kwargs):

        if not stations:
            stations_to_query = self._stations.values()
        else:
            if isinstance(stations, str):
                stations = [stations]
            unknown = [k for k in stations if k not in self._stations]
            if unknown:
                raise ValueError(f"[{self.code}] estaciones desconocidas: {unknown} (disponibles: {list(self._stations)})")
            stations_to_query = [self._stations[k] for k in stations if k in self._stations]

        net_stream = Stream()
        valid_id = []
        sta_kwargs = dict(kwargs, toff=toff)
        for sta in stations_to_query:
            chans = [ch for ch in sta.channels if component is None or ch[-1] == component]
            if not chans:
                continue

            st = sta.get_stream(starttime, endtime,  channel=chans, **sta_kwargs)

            if st:
                for tr in st:
                        net_stream.append(tr)
                        valid_id.append(f"{sta.id}.{tr.stats.channel}")

        if return_array:
            return stream2array(net_stream, valid_id)

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

    def get_stream(self, starttime, endtime, channel=None, pad=600, return_array=False, **kwargs):

        from .reader import get_station_stream
        from .signal import stream_preprocess

        config = kwargs.copy()
        empty  = ([], None, None, None) if return_array else None

        # Canales
        names = list(self.channels) if channel is None else (channel if isinstance(channel, list) else [channel])
        valid = [ch for ch in names if self.is_channel(ch)]
        for ch in names:
            if ch not in valid:
                print(f" >>> Channel '{ch}' not found in {self.id}")
        if not valid:
            return empty

        # Padding
        toff = float(config.pop("toff", 0))
        if toff >= pad:
            raise ValueError(f"toff={toff} s debe ser menor que el pad ({pad} s)")
        config["taper_max_length"] = pad - toff

        # Lectura miniseeds
        t0, t1 = UTCDateTime(starttime), UTCDateTime(endtime)
        r0, r1 = t0 - pad, t1 + pad
        config["starttime"], config["endtime"] = r0, r1
        st_raw = Stream()
        for ch in valid:
            st_ch = get_station_stream(self, ch, r0, r1)
            if st_ch:
                st_raw += st_ch
        if not st_raw:
            return empty

        # check tasa de muestreo
        rates = {tr.stats.sampling_rate for tr in st_raw}
        if len(rates) > 1:
            raise ValueError(f"[{self.id}] frecuencias de muestreo distintas: {sorted(rates)}")
        
        # preprocesado + política de huecos
        rm_resp = config.get("rm_resp", False)
        rm_sens = config.get("rm_sens", True) and not rm_resp
        config["rm_resp"], config["rm_sens"] = rm_resp, rm_sens
        if rm_resp or rm_sens:
            chans = {tr.stats.channel for tr in st_raw}
            config["rm_dict"] = {ch: (self.channels[ch].get_resp() if rm_resp else self.channels[ch].sensitivity) for ch in chans}
        config.setdefault("gaps", "reject")
        st = stream_preprocess(st_raw, config)
        if not st:
            return empty

        # recorte
        st = st.slice(t0 - toff, t1 + toff)
        if return_array:
            return stream2array(st, [f"{self.id}.{tr.stats.channel}" for tr in st])
        return st

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

    def _reindex_by_loc(self):
        new_stations = {}
        for sta in self._stations.values():
            loc_key = sta.loc if sta.loc else ""
            new_stations[loc_key] = sta
        self._stations = new_stations

    @property
    def nsta(self):
        return len(self._stations)

    def __repr__(self):
        return f"<Array {self.code} ({len(self._stations)} stations)>"

    @cached_property
    def _positions(self):
        ids, x, y = self.get_position(utm=True)
        ids = list(ids)
        x = (np.asarray(x, dtype=float) - np.mean(x)) / 1000.0
        y = (np.asarray(y, dtype=float) - np.mean(y)) / 1000.0
        return ids, x, y

    def positions_km(self):
        """(ids, x, y) en km, centradas en el array."""
        ids, x, y = self._positions
        return list(ids), x.copy(), y.copy()

    @cached_property
    def distances(self):
        """Distancias entre pares (km), triángulo superior, en el orden de positions_km."""
        _, x, y = self._positions
        d = np.hypot(np.subtract.outer(x, x), np.subtract.outer(y, y))
        return d[np.triu_indices(len(x), 1)]

    @property
    def aperture(self):
        """Distancia máxima entre estaciones (km)."""
        return float(self.distances.max())

    @property
    def dmin(self):
        """Distancia mínima entre estaciones (km)."""
        return float(self.distances.min())

    def subarray(self, station_list):
        """
        Nuevo Array con solo esas estaciones. Acepta las claves del array (location) o los sta.id. Comparte los objetos Station.
        """

        station_list = list(station_list)

        if len(set(station_list)) != len(station_list):
            raise ValueError(f"estaciones repetidas en {station_list}")

        # traducir cada entrada a su clave (location)
        by_id = {sta.id: k for k, sta in self._stations.items()}
        wanted, unknown = set(), []
        for s in station_list:
            if s in self._stations:
                wanted.add(s)
            elif s in by_id:
                wanted.add(by_id[s])
            else:
                unknown.append(s)
        if unknown:
            raise ValueError(f"[{self.code}] estaciones desconocidas {unknown}; disponibles: {list(self._stations)}")
        if len(wanted) != len(station_list):          # p.ej. "01" y "PJ01.00": la misma estación dos veces
            raise ValueError(f"la misma estación aparece dos veces en {station_list}")

        # orden del array, no el del usuario
        keys = [k for k in self._stations if k in wanted]

        sub = copy.copy(self)
        sub._stations = {k: self._stations[k] for k in keys}

        # borrar lo cacheado se recalcula para el subarray
        for name in list(vars(sub)):
            if isinstance(getattr(type(sub), name, None), cached_property):
                del sub.__dict__[name]
        return sub

    def get_stream(self, starttime, endtime, toff, return_array=False, **kwargs):
        return super().get_stream(starttime, endtime, toff, component=self.component, return_array=return_array, **kwargs)

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
        
        else:
            from .plotting import slowmap
            fig, _, axes = slowmap(power, s_vals, s_vals, v_min=0, v_max=power.max())
            axes[0].set_title(f"freq={fmin}-{fmax} Hz")
            return fig, axes

    def get_beam(self, data, sx, sy, fs, **kwargs):
        from .array import compute_beams_matrix

        delay = self.delay_matrix(sx, sy)
        return compute_beams_matrix(data, delay, fs, **kwargs)

    def get_trias(self, station_list=None, *, fs, fmax, fmin=None, nulldir="nulltest", g_min=0.1):
        from .array import TRIAS
        return TRIAS(self, fs=fs, fmax=fmax, fmin=fmin, station_list=station_list, nulldir=nulldir, g_min=g_min)

    def get_zlcc(self, station_list=None):
        from .array import ZLCC
        return ZLCC(self, station_list=station_list)