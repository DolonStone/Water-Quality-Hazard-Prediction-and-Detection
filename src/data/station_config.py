"""
Station configuration and parameter definitions.

This module contains configuration for water quality monitoring stations
and their associated parameters.
"""

# USGS parameter codes and their descriptions
WATER_QUALITY_PARAMS = {
    '00010': 'Temperature (°C)',
    '00060': 'Streamflow (cfs)',
    '00095': 'Specific Conductance (uS/cm)',
    '00300': 'Dissolved Oxygen (mg/L)',
    '00400': 'pH',
}

# Parameter codes as a list for API queries
PARAM_CODES = list(WATER_QUALITY_PARAMS.keys())

# Monitoring stations
MONITORING_STATIONS = {
    'USGS-07374000': {
        'name': 'Lake Moultrie Tailrace Canal at Moncks Corner, SC',
        'location': 'South Carolina',
        'state_code': 'SC',
    },
    'USGS-07381600': {
        'name': 'Monitoring Station 2',
        'location': 'Louisiana',
        'state_code': 'LA',
    },
}

# API Configuration
USGS_API_BASE_URL = 'https://waterservices.usgs.gov/nwis/iv/'

# Site search criteria
SITE_SEARCH_PARAMS = {
    'format': 'json',
    'parameterCd': ','.join(PARAM_CODES),
    'siteType': 'ST',  # Stream sites
    'siteStatus': 'active',
}
