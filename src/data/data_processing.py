from src.data.station_config import MONITORING_STATIONS, WATER_QUALITY_PARAMS
import pandas as pd


def format_data_for_modeling(df, metadata):
    """
    Format retrieved water quality data for anomaly detection.
    """

    # Ensure timestamp is datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Pivot to wide format
    df_wide = df.pivot(
        index="time",
        columns="parameter_code",
        values="value"
    )

    # Rename parameters
    df_wide.columns = df_wide.columns.map(WATER_QUALITY_PARAMS)

    # Sort timestamps
    df_wide = df_wide.sort_index()

    # Remove duplicate timestamps
    df_wide = df_wide[~df_wide.index.duplicated(keep="first")]

    # Force uniform 15-minute sampling
    df_wide = df_wide.resample("15T").mean()

    # Interpolate missing values
    df_clean = interpolate_missing_values(df_wide)

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

