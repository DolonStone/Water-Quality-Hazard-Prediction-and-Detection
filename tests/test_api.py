import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import format_data_for_modeling, MONITORING_STATIONS,get_instantaneous_data,get_historical_data

df = get_instantaneous_data(site_id='USGS-07374000')
df_formatted = format_data_for_modeling(df[0], df[1])
print(df_formatted.head())