#!/usr/bin/env python3
# coding=utf-8

from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker, declarative_base
import sqlalchemy as sql

Base = declarative_base()

class SDERow(Base):
    __tablename__ = 'sde_events'

    id = sql.Column(sql.Integer, primary_key=True)

    event_id = sql.Column(sql.Integer, nullable=False, index=True)
    event_type = sql.Column(sql.String, nullable=False)
        
    network  = sql.Column(sql.String, nullable=False)
    station  = sql.Column(sql.String, nullable=False)
    location = sql.Column(sql.String, nullable=True)
        
    label     = sql.Column(sql.String, nullable=False)
    sublabel  = sql.Column(sql.String, nullable=True)
    
    starttime = sql.Column(sql.DateTime(timezone=False), nullable=False)
    duration  = sql.Column(sql.Float, nullable=False)

    # event attributes
    time_P   = sql.Column(sql.DateTime(timezone=False), nullable=True)
    weight_P = sql.Column(sql.Integer, nullable=True)
    onset_P  = sql.Column(sql.String, nullable=True)
    time_S   = sql.Column(sql.DateTime(timezone=False), nullable=True)
    weight_S = sql.Column(sql.Integer, nullable=True)
    time_C   = sql.Column(sql.DateTime(timezone=False), nullable=True)    

    # additional floats
    value_1 = sql.Column(sql.Float, nullable=True) # peak to peak
    value_2 = sql.Column(sql.Float, nullable=True) # max amplitude
    value_3 = sql.Column(sql.Float, nullable=True)
    value_4 = sql.Column(sql.Float, nullable=True)
    value_5 = sql.Column(sql.Float, nullable=True)
    
        # additional strings
    string_1 = sql.Column(sql.String, nullable=True)
    string_2 = sql.Column(sql.String, nullable=True) # hypoellipse loc path 
    string_3 = sql.Column(sql.String, nullable=True)
    string_4 = sql.Column(sql.String, nullable=True)
    string_5 = sql.Column(sql.String, nullable=True)

    def get_station_id(self):
        return f"{self.network}.{self.station}.{self.location}"





class DatabaseManager:
    def __init__(self, sql_path):
        self.sql_path = sql_path
        self.engine = sql.create_engine(f'sqlite:///{self.sql_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class SDEDatabase(DatabaseManager):
    def __init__(self, sql_path):
        super().__init__(sql_path)
        self._update_metadata()

    def _update_metadata(self):
        eid_list = self.get_event_list() 
        if eid_list:
            self.first_eid = eid_list[0]
            self.last_eid  = eid_list[-1]
        else:
            print(" [info]  Empty database")
            self.first_eid = None
            self.last_eid = 0

    def add_event(self, info_dict):
        with self.get_session() as session:
            new_row = SDERow(**info_dict)
            session.add(new_row)
            session.flush() # Obtiene el ID sin cerrar la transacción
            return new_row.id

    def get_event_rows(self, event_id):
        """Devuelve los registros crudos de la BD."""
        with self.get_session() as session:
            return session.query(SDERow).filter(SDERow.event_id == event_id).all()
    
    def get_event_object(self, event_id):
        """Devuelve un objeto de Lógica de Negocio (Event)."""
        rows = self.get_event_rows(event_id)
        if not rows:
            return None
        return Event(event_id, rows)


class Event:
    def __init__(self, event_id, db_rows):
        self.id = event_id
        self.rows = db_rows
        
        self.label = self.rows[0].label
        self.starttime = self.rows[0].starttime
        self.duration = self.rows[0].duration
        self.stations = [row.get_station_id() for row in self.rows]
        self.nro_phase = self.get_phases(nro_phases=True)