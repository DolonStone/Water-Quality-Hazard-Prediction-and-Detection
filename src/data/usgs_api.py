"""
USGS API module for water quality data retrieval.

This module provides functions to query the USGS NWIS service for
water quality and streamflow data from monitoring stations.
"""

from datetime import datetime, timedelta
from typing import Tuple, Dict, Set, List
from dataretrieval import waterdata


import requests
import pandas as pd

from station_config import (
    WATER_QUALITY_PARAMS,
    PARAM_CODES,
    USGS_API_BASE_URL,
    SITE_SEARCH_PARAMS,
)


def get_instantaneous_data(
    site_id: str,
    param_codes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Retrieve instantaneous (real-time) water quality data from USGS.

    Parameters
    ----------
    site_id : str
        USGS site identification number.
    param_codes : list, optional
        List of USGS parameter codes to retrieve.
        If None, uses PARAM_CODES from station_config.
    start_date : str, optional
        Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'.
        If None, defaults to 24 hours ago.
    end_date : str, optional
        End date in format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM'.
        If None, defaults to current time.

    Returns
    -------
    tuple(pd.DataFrame, dict)
        - DataFrame containing the retrieved data
        - Dictionary containing metadata
    """
    if param_codes is None:
        param_codes = PARAM_CODES

    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    if start_date is None:
        start_date = end_date - timedelta(hours=24)
    elif isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%dT%H:%M')
    end_str = end_date.strftime('%Y-%m-%dT%H:%M')

    # Retrieve data
    data, metadata = waterdata.get_continuous(
    monitoring_location_id=site_id,
    parameter_code=param_codes,
    time=[start_str, end_str],
    )

    return data, metadata


def find_sites_with_parameters(
    param_codes: List[str] = None,
    state_code: str = None,
    site_type: str = 'ST',
    site_status: str = 'active',
) -> List[str]:
    """
    Search for monitoring sites that measure specified parameters.

    Parameters
    ----------
    param_codes : list, optional
        List of USGS parameter codes to search for.
        If None, uses PARAM_CODES from station_config.
    state_code : str, optional
        Two-letter state code (e.g., 'SC', 'LA').
        If None, searches all states.
    site_type : str, default 'ST'
        Type of site ('ST' for stream).
    site_status : str, default 'active'
        Site status ('active' or 'all').

    Returns
    -------
    list
        List of site IDs that have all specified parameters.
    """
    if param_codes is None:
        param_codes = PARAM_CODES

    # Build request parameters
    params = {
        'format': 'json',
        'parameterCd': ','.join(param_codes),
        'siteType': site_type,
        'siteStatus': site_status,
    }

    if state_code:
        params['stateCd'] = state_code

    # Query API
    response = requests.get(USGS_API_BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse response to find sites with all parameters
    sites_dict: Dict[str, Set[str]] = {}

    for ts in data.get('value', {}).get('timeSeries', []):
        site_id = ts['sourceInfo']['siteCode'][0]['value']
        param = ts['variable']['variableCode'][0]['value']

        if site_id not in sites_dict:
            sites_dict[site_id] = set()
        sites_dict[site_id].add(param)

    # Filter for sites that have all requested parameters
    desired_params = set(param_codes)
    sites_with_all = [
        site for site, params in sites_dict.items()
        if desired_params.issubset(params)
    ]

    return sites_with_all


def get_historical_data(
    site_id: str,
    param_codes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    days_back: int = 500,
) -> Tuple[pd.DataFrame, dict]:
    """
    Retrieve historical instantaneous data from USGS.

    Parameters
    ----------
    site_id : str
        USGS site identification number.
    param_codes : list, optional
        List of USGS parameter codes to retrieve.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format.
        If None, calculated from days_back.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format.
        If None, defaults to today.
    days_back : int, default 500
        Number of days to retrieve if start_date is not provided.

    Returns
    -------
    tuple(pd.DataFrame, dict)
        - DataFrame containing the retrieved data
        - Dictionary containing metadata
    """
    if param_codes is None:
        param_codes = PARAM_CODES

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days_back)
        start_date = start_dt.strftime('%Y-%m-%d')

    return get_instantaneous_data(
        site_id=site_id,
        param_codes=param_codes,
        start_date=start_date,
        end_date=end_date,
    )
