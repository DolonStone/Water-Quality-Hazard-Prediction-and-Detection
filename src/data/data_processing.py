from usgs_api import get_instantaneous_data
from usgs_api import get_historical_data
from station_config import MONITORING_STATIONS, WATER_QUALITY_PARAMS
import pandas as pd


def format_data_for_modeling(df, metadata):
    """
    Format the retrieved data for machine learning modeling.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the retrieved water quality data.
    metadata : dict
        Dictionary containing metadata about the retrieved data.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for modeling.
    """


    # Pivot to wide format: one column per parameter
    df_wide = df.pivot(index='time', columns='parameter_code', values='value')

    # Rename columns to human-readable names using metadata
    df_wide.columns = df_wide.columns.map(WATER_QUALITY_PARAMS)
    df_clean = interpolate_missing_values(df_wide)
    # Remove Rows with missing values
    #required_cols = [col for col in df_wide.columns]
    #df_clean = df_wide.dropna(subset=required_cols, how='any')
    return df_clean

def interpolate_missing_values(df, method='linear', limit_direction='both'):
    """
    Interpolate missing values in water quality data.
    
    Args:
        df: DataFrame with DatetimeIndex
        method: 'linear' (average of neighbors), 'forward_fill', 'bfill'
        limit_direction: 'both', 'forward', 'backward'
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Linear interpolation (takes average of previous and subsequent)
    df_interpolated = df.interpolate(
        method=method,
        limit_direction=limit_direction,
        limit=1  # Only fill gaps up to 1 missing value (optional)
    )
    
    return df_interpolated

df = get_instantaneous_data(site_id='USGS-07374000')
df_formatted = format_data_for_modeling(df[0], df[1])
print(df_formatted.head())