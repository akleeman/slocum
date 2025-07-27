"""
Standard variable name conventions for the slocum codebase.

This module defines the standard variable names used throughout the slocum
codebase to ensure consistency when working with different data sources.
"""

# Wind components (standard names used internally)
U_WIND = "u_wind"
V_WIND = "v_wind"

# Alternative wind component names from different data sources
WIND_U_10M = "wind_u_10m"
WIND_V_10M = "wind_v_10m"
UGRD10M = "ugrd10m"
VGRD10M = "vgrd10m"

# Wind speed and direction
WIND_SPEED = "speed"
WIND_DIRECTION = "direction"

# Temperature variables
AIR_TEMPERATURE = "air_temperature"
TEMP_2M = "tmp2m"
TEMPERATURE_2M = "temperature_2m"

# Pressure variables
AIR_PRESSURE_AT_SEA_LEVEL = "air_pressure_at_sea_level"
PRMSLMSL = "prmslmsl"
PRESSURE_SURFACE = "pressure_surface"

# Current variables
SEA_WATER_X_VELOCITY = "sea_water_x_velocity"
SEA_WATER_Y_VELOCITY = "sea_water_y_velocity"
U_VELOCITY = "u_velocity"
V_VELOCITY = "v_velocity"

# Wave variables
SEA_SURFACE_WAVE_SIGNIFICANT_HEIGHT = "sea_surface_wave_significant_height"
SEA_SURFACE_WAVE_TO_DIRECTION = "sea_surface_wave_to_direction"
HTSGWSFC = "htsgwsfc"
DIRPWSFC = "dirpwsfc"

# Sea surface temperature
SEA_SURFACE_TEMPERATURE = "sea_surface_temperature"
SST = "sst"

# Coordinate variables
LATITUDE = "latitude"
LONGITUDE = "longitude"
INIT_TIME = "init_time"
LEAD_TIME = "lead_time"
REALIZATION = "realization"
ENS = "ens"

# Mapping from source-specific names to standard names
WIND_COMPONENT_MAPPING = {
    # Standard names (no mapping needed)
    U_WIND: U_WIND,
    V_WIND: V_WIND,
    
    # Dynamical.org zarr names
    WIND_U_10M: U_WIND,
    WIND_V_10M: V_WIND,
    
    # GRIB/NOAA names
    UGRD10M: U_WIND,
    VGRD10M: V_WIND,
    
    # Alternative names
    "u10": U_WIND,
    "v10": V_WIND,
    "u_wind": U_WIND,
    "v_wind": V_WIND,
}

# Temperature variable mapping
TEMPERATURE_MAPPING = {
    AIR_TEMPERATURE: AIR_TEMPERATURE,
    TEMP_2M: AIR_TEMPERATURE,
    TEMPERATURE_2M: AIR_TEMPERATURE,
    "tmp2m": AIR_TEMPERATURE,
    "temperature_2m": AIR_TEMPERATURE,
}

# Pressure variable mapping
PRESSURE_MAPPING = {
    AIR_PRESSURE_AT_SEA_LEVEL: AIR_PRESSURE_AT_SEA_LEVEL,
    PRMSLMSL: AIR_PRESSURE_AT_SEA_LEVEL,
    PRESSURE_SURFACE: AIR_PRESSURE_AT_SEA_LEVEL,
    "prmslmsl": AIR_PRESSURE_AT_SEA_LEVEL,
    "pressure_surface": AIR_PRESSURE_AT_SEA_LEVEL,
}

def get_standard_name(variable_name):
    """
    Get the standard variable name for a given variable name.
    
    Parameters:
    -----------
    variable_name : str
        The variable name to standardize
        
    Returns:
    --------
    str
        The standard variable name, or the original name if no mapping exists
    """
    # Check wind component mapping first
    if variable_name in WIND_COMPONENT_MAPPING:
        return WIND_COMPONENT_MAPPING[variable_name]
    
    # Check temperature mapping
    if variable_name in TEMPERATURE_MAPPING:
        return TEMPERATURE_MAPPING[variable_name]
    
    # Check pressure mapping
    if variable_name in PRESSURE_MAPPING:
        return PRESSURE_MAPPING[variable_name]
    
    # Return original name if no mapping exists
    return variable_name

def rename_variables_to_standard(ds):
    """
    Rename variables in a dataset to standard names.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with variables to rename
        
    Returns:
    --------
    xarray.Dataset
        Dataset with variables renamed to standard names
    """
    rename_dict = {}
    
    for var_name in ds.data_vars:
        standard_name = get_standard_name(var_name)
        if standard_name != var_name:
            rename_dict[var_name] = standard_name
    
    if rename_dict:
        ds = ds.rename(rename_dict)
        print(f"Renamed variables to standard names: {rename_dict}")
    
    return ds

def has_wind_components(ds):
    """
    Check if a dataset has wind components (either standard or mappable names).
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to check
        
    Returns:
    --------
    bool
        True if the dataset has wind components, False otherwise
    """
    available_vars = list(ds.data_vars.keys())
    
    # Check for standard names
    if U_WIND in available_vars and V_WIND in available_vars:
        return True
    
    # Check for mappable names
    for var_name in available_vars:
        if var_name in WIND_COMPONENT_MAPPING:
            mapped_name = WIND_COMPONENT_MAPPING[var_name]
            if mapped_name == U_WIND:
                # Check if we have the corresponding V component
                for other_var in available_vars:
                    if other_var in WIND_COMPONENT_MAPPING and WIND_COMPONENT_MAPPING[other_var] == V_WIND:
                        return True
    
    return False

def get_wind_component_names(ds):
    """
    Get the wind component variable names from a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to check
        
    Returns:
    --------
    tuple
        (u_wind_name, v_wind_name) or (None, None) if not found
    """
    available_vars = list(ds.data_vars.keys())
    
    # Check for standard names first
    if U_WIND in available_vars and V_WIND in available_vars:
        return U_WIND, V_WIND
    
    # Look for mappable names
    u_wind_name = None
    v_wind_name = None
    
    for var_name in available_vars:
        if var_name in WIND_COMPONENT_MAPPING:
            mapped_name = WIND_COMPONENT_MAPPING[var_name]
            if mapped_name == U_WIND:
                u_wind_name = var_name
            elif mapped_name == V_WIND:
                v_wind_name = var_name
    
    if u_wind_name and v_wind_name:
        return u_wind_name, v_wind_name
    
    return None, None 