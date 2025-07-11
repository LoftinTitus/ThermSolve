"""
Interpolation utilities for temperature-dependent thermophysical properties.

This module provides interpolation and extrapolation functions for handling
discrete experimental data points and converting them to continuous property
functions over temperature ranges.
"""

import numpy as np
import warnings
from typing import List, Tuple, Union, Optional, Dict, Any
from scipy import interpolate
from scipy.optimize import curve_fit


class PropertyInterpolator:
    """
    A class for interpolating thermophysical properties as a function of temperature.
    
    Supports various interpolation methods including linear, cubic spline, and
    polynomial fitting with extrapolation warnings.
    """
    
    def __init__(self, 
                 temperatures: List[float], 
                 values: List[float],
                 property_name: str = "property",
                 method: str = "cubic",
                 extrapolation_method: str = "linear"):
        """
        Initialize the PropertyInterpolator.
        
        Parameters:
        -----------
        temperatures : List[float]
            Temperature data points in Kelvin
        values : List[float]
            Property values corresponding to temperatures
        property_name : str
            Name of the property for warning messages
        method : str
            Interpolation method: 'linear', 'cubic', 'polynomial'
        extrapolation_method : str
            How to handle extrapolation: 'linear', 'constant', 'polynomial', 'warning'
        """
        if len(temperatures) != len(values):
            raise ValueError("Temperature and value arrays must have the same length")
        
        if len(temperatures) < 2:
            raise ValueError("At least 2 data points are required for interpolation")
        
        # Sort data by temperature
        sorted_data = sorted(zip(temperatures, values))
        self.temperatures = np.array([t for t, v in sorted_data])
        self.values = np.array([v for t, v in sorted_data])
        
        self.property_name = property_name
        self.method = method
        self.extrapolation_method = extrapolation_method
        
        # Temperature range
        self.temp_min = self.temperatures.min()
        self.temp_max = self.temperatures.max()
        
        # Create interpolation function
        self._create_interpolator()
    
    def _create_interpolator(self):
        """Create the appropriate interpolation function."""
        if self.method == "linear":
            self.interpolator = interpolate.interp1d(
                self.temperatures, self.values, 
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
        elif self.method == "cubic":
            if len(self.temperatures) >= 4:
                self.interpolator = interpolate.CubicSpline(
                    self.temperatures, self.values, 
                    bc_type='natural', extrapolate=True
                )
            else:
                # Fall back to linear for insufficient points
                warnings.warn(f"Insufficient points for cubic interpolation of {self.property_name}, using linear")
                self.interpolator = interpolate.interp1d(
                    self.temperatures, self.values,
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
        elif self.method == "polynomial":
            degree = min(len(self.temperatures) - 1, 3)  # Limit to cubic
            self.poly_coeffs = np.polyfit(self.temperatures, self.values, degree)
            self.interpolator = lambda T: np.polyval(self.poly_coeffs, T)
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
    
    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate property value at given temperature(s).
        
        Parameters:
        -----------
        T : float or array-like
            Temperature(s) in Kelvin
            
        Returns:
        --------
        float or array-like
            Interpolated property value(s)
        """
        T = np.asarray(T)
        scalar_input = T.ndim == 0
        T = np.atleast_1d(T)
        
        # Check for extrapolation
        extrapolating = (T < self.temp_min) | (T > self.temp_max)
        if np.any(extrapolating):
            if self.extrapolation_method == "warning":
                warnings.warn(
                    f"Extrapolating {self.property_name} outside data range "
                    f"[{self.temp_min:.1f}, {self.temp_max:.1f}] K"
                )
            elif self.extrapolation_method == "constant":
                # Use nearest boundary values for extrapolation
                result = self.interpolator(T)
                result[T < self.temp_min] = self.values[0]
                result[T > self.temp_max] = self.values[-1]
                return result.item() if scalar_input else result
        
        # Perform interpolation
        result = self.interpolator(T)
        
        return result.item() if scalar_input else result
    
    def derivative(self, T: Union[float, np.ndarray], order: int = 1) -> Union[float, np.ndarray]:
        """
        Calculate derivative of property with respect to temperature.
        
        Parameters:
        -----------
        T : float or array-like
            Temperature(s) in Kelvin
        order : int
            Order of derivative (1 or 2)
            
        Returns:
        --------
        float or array-like
            Derivative value(s)
        """
        if self.method == "cubic" and hasattr(self.interpolator, 'derivative'):
            return self.interpolator.derivative(order)(T)
        elif self.method == "polynomial":
            deriv_coeffs = np.polyder(self.poly_coeffs, order)
            return np.polyval(deriv_coeffs, T)
        else:
            # Numerical derivative for linear interpolation
            h = 1e-6  # Small temperature step
            if order == 1:
                return (self(T + h) - self(T - h)) / (2 * h)
            elif order == 2:
                return (self(T + h) - 2*self(T) + self(T - h)) / (h**2)
            else:
                raise ValueError("Only first and second derivatives are supported")


class TemperatureDataSeries:
    """
    Container for temperature-dependent property data with metadata.
    """
    
    def __init__(self, 
                 temperatures: List[float],
                 values: List[float],
                 property_name: str,
                 units: str = "",
                 source: str = "",
                 uncertainty: Optional[List[float]] = None):
        """
        Initialize temperature-dependent data series.
        
        Parameters:
        -----------
        temperatures : List[float]
            Temperature data points in Kelvin
        values : List[float]
            Property values
        property_name : str
            Name of the property
        units : str
            Units of the property
        source : str
            Data source reference
        uncertainty : List[float], optional
            Uncertainty values for each data point
        """
        self.temperatures = np.array(temperatures)
        self.values = np.array(values)
        self.property_name = property_name
        self.units = units
        self.source = source
        self.uncertainty = np.array(uncertainty) if uncertainty else None
        
        # Validate data
        if len(self.temperatures) != len(self.values):
            raise ValueError("Temperature and value arrays must have same length")
        
        if self.uncertainty is not None and len(self.uncertainty) != len(self.values):
            raise ValueError("Uncertainty array must have same length as values")
    
    def to_interpolator(self, method: str = "cubic", **kwargs) -> PropertyInterpolator:
        """Convert to PropertyInterpolator object."""
        return PropertyInterpolator(
            self.temperatures.tolist(),
            self.values.tolist(),
            property_name=self.property_name,
            method=method,
            **kwargs
        )
    
    def fit_correlation(self, correlation_type: str = "polynomial") -> Dict[str, Any]:
        """
        Fit a correlation to the data.
        
        Parameters:
        -----------
        correlation_type : str
            Type of correlation: 'polynomial', 'arrhenius', 'antoine'
            
        Returns:
        --------
        dict
            Correlation parameters and statistics
        """
        if correlation_type == "polynomial":
            return self._fit_polynomial()
        elif correlation_type == "arrhenius":
            return self._fit_arrhenius()
        elif correlation_type == "antoine":
            return self._fit_antoine()
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")
    
    def _fit_polynomial(self, degree: int = 3) -> Dict[str, Any]:
        """Fit polynomial correlation."""
        coeffs = np.polyfit(self.temperatures, self.values, degree)
        fitted_values = np.polyval(coeffs, self.temperatures)
        r_squared = 1 - np.sum((self.values - fitted_values)**2) / np.sum((self.values - np.mean(self.values))**2)
        
        return {
            "type": "polynomial",
            "coefficients": coeffs.tolist(),
            "degree": degree,
            "r_squared": r_squared,
            "rmse": np.sqrt(np.mean((self.values - fitted_values)**2))
        }
    
    def _fit_arrhenius(self) -> Dict[str, Any]:
        """Fit Arrhenius-type correlation: y = A * exp(B/T)."""
        def arrhenius(T, A, B):
            return A * np.exp(B / T)
        
        try:
            popt, pcov = curve_fit(arrhenius, self.temperatures, self.values, 
                                 p0=[1.0, 1000.0], maxfev=5000)
            fitted_values = arrhenius(self.temperatures, *popt)
            r_squared = 1 - np.sum((self.values - fitted_values)**2) / np.sum((self.values - np.mean(self.values))**2)
            
            return {
                "type": "arrhenius",
                "A": popt[0],
                "B": popt[1],
                "r_squared": r_squared,
                "rmse": np.sqrt(np.mean((self.values - fitted_values)**2))
            }
        except:
            warnings.warn(f"Failed to fit Arrhenius correlation for {self.property_name}")
            return self._fit_polynomial()
    
    def _fit_antoine(self) -> Dict[str, Any]:
        """Fit Antoine equation: log10(P) = A - B/(C + T)."""
        if np.any(self.values <= 0):
            raise ValueError("Antoine equation requires positive values")
        
        def antoine(T, A, B, C):
            return A - B / (C + T)
        
        try:
            log_values = np.log10(self.values)
            popt, pcov = curve_fit(antoine, self.temperatures, log_values,
                                 p0=[8.0, 1500.0, -50.0], maxfev=5000)
            fitted_log = antoine(self.temperatures, *popt)
            fitted_values = 10 ** fitted_log
            r_squared = 1 - np.sum((self.values - fitted_values)**2) / np.sum((self.values - np.mean(self.values))**2)
            
            return {
                "type": "antoine",
                "A": popt[0],
                "B": popt[1], 
                "C": popt[2],
                "r_squared": r_squared,
                "rmse": np.sqrt(np.mean((self.values - fitted_values)**2))
            }
        except:
            warnings.warn(f"Failed to fit Antoine equation for {self.property_name}")
            return self._fit_polynomial()


# Utility functions
def linear_interpolation(x: List[float], y: List[float], x_new: float) -> float:
    """Simple linear interpolation between two points."""
    x, y = np.array(x), np.array(y)
    return np.interp(x_new, x, y)


def check_extrapolation(T: float, T_range: Tuple[float, float], property_name: str = "property"):
    """Check if temperature is outside valid range and warn if extrapolating."""
    if T < T_range[0] or T > T_range[1]:
        warnings.warn(
            f"Temperature {T:.1f} K is outside valid range "
            f"[{T_range[0]:.1f}, {T_range[1]:.1f}] K for {property_name}. "
            f"Extrapolating may give unreliable results."
        )


def create_property_function(temperatures: List[float], 
                           values: List[float],
                           method: str = "cubic") -> callable:
    """
    Create a callable function from temperature-property data.
    
    Parameters:
    -----------
    temperatures : List[float]
        Temperature data points in Kelvin
    values : List[float] 
        Property values
    method : str
        Interpolation method
        
    Returns:
    --------
    callable
        Function that takes temperature and returns interpolated property
    """
    interpolator = PropertyInterpolator(temperatures, values, method=method)
    return interpolator


# Integration with existing Substance class
def enhance_substance_with_data(substance, property_name: str, 
                              temperatures: List[float], 
                              values: List[float],
                              method: str = "cubic"):
    """
    Add interpolated property data to an existing Substance object.
    
    Parameters:
    -----------
    substance : Substance
        Existing substance object
    property_name : str
        Name of property to add ('cp', 'viscosity', 'density', etc.)
    temperatures : List[float]
        Temperature data points in Kelvin
    values : List[float]
        Property values
    method : str
        Interpolation method
    """
    interpolator = PropertyInterpolator(temperatures, values, 
                                      property_name=property_name, method=method)
    
    # Store as coefficients in the expected format
    coeffs_attr = f"{property_name}_coefficients"
    if hasattr(substance, coeffs_attr):
        setattr(substance, coeffs_attr, {
            "type": "interpolated",
            "interpolator": interpolator,
            "temp_range": (min(temperatures), max(temperatures))
        })
    
    # Update temperature range
    current_range = getattr(substance, 'temp_range', (0, 1000))
    new_range = (
        max(current_range[0], min(temperatures)),
        min(current_range[1], max(temperatures))
    )
    substance.temp_range = new_range