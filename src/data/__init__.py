# src/data/__init__.py
from .data_processing import format_data_for_modeling
from .usgs_api import get_instantaneous_data,get_historical_data
from .station_config import MONITORING_STATIONS,WATER_QUALITY_PARAMS

# Now you can do:
# from src.data import format_data_for_modeling
# Instead of:
# from src.data.data_processing import format_data_for_modeling