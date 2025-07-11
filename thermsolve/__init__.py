"""
ThermSolve - A thermophysical property library for chemical and process engineering.

This package provides easy access to temperature-dependent properties of common
fluids, gases, and materials with unit safety, validity ranges, and custom 
substance support.
"""

__version__ = "0.1.0"

# Core classes
from .substances import Substance, SubstanceDatabase

# Interpolation utilities
try:
    from .interpolation import (
        PropertyInterpolator, 
        TemperatureDataSeries, 
        enhance_substance_with_data,
        create_property_function
    )
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False

# I/O utilities
try:
    from .io import load_substance_data, save_substance_data
    HAS_IO = True
except ImportError:
    HAS_IO = False

# Plotting utilities  
try:
    from .plotting import plot_property_vs_temperature
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Units handling
try:
    from .units import convert_units, get_unit_registry
    HAS_UNITS = True
except ImportError:
    HAS_UNITS = False

__all__ = [
    "Substance",
    "SubstanceDatabase", 
    "__version__"
]

# Add interpolation exports if available
if HAS_INTERPOLATION:
    __all__.extend([
        "PropertyInterpolator",
        "TemperatureDataSeries", 
        "enhance_substance_with_data",
        "create_property_function"
    ])

# Add other exports conditionally
if HAS_IO:
    __all__.extend(["load_substance_data", "save_substance_data"])
    
if HAS_PLOTTING:
    __all__.append("plot_property_vs_temperature")
    
if HAS_UNITS:
    __all__.extend(["convert_units", "get_unit_registry"])