"""
Unit handling and conversion utilities for thermodynamic calculations.

This module provides a comprehensive unit system for chemical and process
engineering calculations, including temperature, pressure, density, heat capacity,
enthalpy, entropy, viscosity, thermal conductivity, and other thermophysical properties.
"""

import pint
from typing import Union, Dict, Any, Optional, List
import numpy as np

# Create a unit registry with additional thermodynamic units
ureg = pint.UnitRegistry()

# Define additional compound units commonly used in thermodynamics
ureg.define('Btu_IT = 1055.05585262 * joule = Btu')  # International Table Btu
ureg.define('cal_IT = 4.1868 * joule = cal')  # International Table calorie
ureg.define('psia = pound_force_per_square_inch')  # Absolute pressure
ureg.define('psig = pound_force_per_square_inch')  # Gauge pressure (same unit, different reference)


class ThermodynamicUnits:
    """
    A comprehensive unit handling class for thermodynamic calculations.
    
    This class provides conversion utilities, validation, and standardized
    unit handling for common thermophysical properties used in unit operations.
    """
    
    def __init__(self):
        self.ureg = ureg
        
        # Standard SI base units for each property type
        self.standard_units = {
            'temperature': 'K',
            'pressure': 'Pa',
            'density': 'kg/m^3',
            'molar_density': 'mol/m^3',
            'specific_heat': 'J/(kg*K)',
            'molar_heat_capacity': 'J/(mol*K)',
            'enthalpy': 'J/kg',
            'molar_enthalpy': 'J/mol',
            'entropy': 'J/(kg*K)',
            'molar_entropy': 'J/(mol*K)',
            'viscosity_dynamic': 'Pa*s',
            'viscosity_kinematic': 'm^2/s',
            'thermal_conductivity': 'W/(m*K)',
            'surface_tension': 'N/m',
            'diffusivity': 'm^2/s',
            'heat_transfer_coefficient': 'W/(m^2*K)',
            'mass_flow_rate': 'kg/s',
            'volumetric_flow_rate': 'm^3/s',
            'molar_flow_rate': 'mol/s',
            'heat_flux': 'W/m^2',
            'power': 'W',
            'energy': 'J',
            'volume': 'm^3',
            'area': 'm^2',
            'length': 'm',
            'mass': 'kg',
            'molar_mass': 'kg/mol',
            'concentration': 'mol/m^3',
            'mass_fraction': 'dimensionless',
            'mole_fraction': 'dimensionless',
            'dimensionless': 'dimensionless'
        }
        
        # Common alternative units for each property
        self.common_units = {
            'temperature': ['K', 'degC', 'degF', 'degR'],
            'pressure': ['Pa', 'bar', 'atm', 'psi', 'psia', 'psig', 'mmHg', 'torr', 'kPa', 'MPa'],
            'density': ['kg/m^3', 'g/cm^3', 'lb/ft^3', 'kg/L'],
            'molar_density': ['mol/m^3', 'mol/L', 'kmol/m^3'],
            'specific_heat': ['J/(kg*K)', 'kJ/(kg*K)', 'cal/(g*K)', 'Btu/(lb*degF)'],
            'molar_heat_capacity': ['J/(mol*K)', 'kJ/(mol*K)', 'cal/(mol*K)', 'Btu/(lbmol*degF)'],
            'enthalpy': ['J/kg', 'kJ/kg', 'cal/g', 'Btu/lb'],
            'molar_enthalpy': ['J/mol', 'kJ/mol', 'cal/mol', 'Btu/lbmol'],
            'entropy': ['J/(kg*K)', 'kJ/(kg*K)', 'cal/(g*K)', 'Btu/(lb*degF)'],
            'molar_entropy': ['J/(mol*K)', 'kJ/(mol*K)', 'cal/(mol*K)', 'Btu/(lbmol*degF)'],
            'viscosity_dynamic': ['Pa*s', 'cP', 'poise', 'lb/(ft*s)'],
            'viscosity_kinematic': ['m^2/s', 'cSt', 'stokes', 'ft^2/s'],
            'thermal_conductivity': ['W/(m*K)', 'cal/(s*cm*K)', 'Btu/(hr*ft*degF)'],
            'surface_tension': ['N/m', 'dyn/cm', 'mN/m'],
            'diffusivity': ['m^2/s', 'cm^2/s', 'ft^2/s'],
            'heat_transfer_coefficient': ['W/(m^2*K)', 'cal/(s*cm^2*K)', 'Btu/(hr*ft^2*degF)'],
            'mass_flow_rate': ['kg/s', 'kg/hr', 'lb/s', 'lb/hr', 'g/s'],
            'volumetric_flow_rate': ['m^3/s', 'L/s', 'L/min', 'ft^3/s', 'gal/min'],
            'molar_flow_rate': ['mol/s', 'mol/hr', 'kmol/s', 'kmol/hr', 'lbmol/hr'],
            'power': ['W', 'kW', 'MW', 'hp', 'Btu/hr'],
            'energy': ['J', 'kJ', 'MJ', 'cal', 'kcal', 'Btu', 'kWh']
        }
    
    def convert(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        Convert a value from one unit to another.
        
        Args:
            value: The numerical value to convert
            from_unit: Source unit string
            to_unit: Target unit string
            
        Returns:
            Converted value in target units
            
        Examples:
            >>> units = ThermodynamicUnits()
            >>> units.convert(100, 'degC', 'K')
            373.15
            >>> units.convert(1, 'bar', 'Pa')
            100000.0
        """
        try:
            quantity = self.ureg.Quantity(value, from_unit)
            converted = quantity.to(to_unit)
            return converted.magnitude
        except Exception as e:
            raise ValueError(f"Cannot convert from '{from_unit}' to '{to_unit}': {e}")
    
    def create_quantity(self, value: Union[float, np.ndarray], unit: str) -> pint.Quantity:
        """
        Create a pint Quantity object with units.
        
        Args:
            value: The numerical value
            unit: Unit string
            
        Returns:
            pint.Quantity object
        """
        return self.ureg.Quantity(value, unit)
    
    def get_standard_unit(self, property_type: str) -> str:
        """
        Get the standard SI unit for a given property type.
        
        Args:
            property_type: Type of thermodynamic property
            
        Returns:
            Standard unit string
        """
        if property_type not in self.standard_units:
            raise ValueError(f"Unknown property type: {property_type}")
        return self.standard_units[property_type]
    
    def get_common_units(self, property_type: str) -> List[str]:
        """
        Get list of common units for a given property type.
        
        Args:
            property_type: Type of thermodynamic property
            
        Returns:
            List of common unit strings
        """
        return self.common_units.get(property_type, [])
    
    def validate_unit(self, unit: str, property_type: str) -> bool:
        """
        Validate that a unit is appropriate for a given property type.
        
        Args:
            unit: Unit string to validate
            property_type: Expected property type
            
        Returns:
            True if unit is valid for the property type
        """
        try:
            # Create a test quantity and convert to standard unit
            test_quantity = self.ureg.Quantity(1.0, unit)
            standard_unit = self.get_standard_unit(property_type)
            test_quantity.to(standard_unit)
            return True
        except Exception:
            return False
    
    def normalize_to_standard(self, value: Union[float, np.ndarray], unit: str, property_type: str) -> Union[float, np.ndarray]:
        """
        Convert a value to standard SI units for the given property type.
        
        Args:
            value: The numerical value
            unit: Current unit
            property_type: Type of property
            
        Returns:
            Value converted to standard units
        """
        if property_type == 'dimensionless' or unit == 'dimensionless':
            return value
        
        standard_unit = self.get_standard_unit(property_type)
        return self.convert(value, unit, standard_unit)
    
    def temperature_convert(self, value: Union[float, np.ndarray], from_scale: str, to_scale: str) -> Union[float, np.ndarray]:
        """
        Specialized temperature conversion with common scale names.
        
        Args:
            value: Temperature value
            from_scale: Source temperature scale ('C', 'F', 'K', 'R')
            to_scale: Target temperature scale ('C', 'F', 'K', 'R')
            
        Returns:
            Converted temperature
        """
        scale_map = {
            'C': 'degC',
            'F': 'degF', 
            'K': 'K',
            'R': 'degR'
        }
        
        from_unit = scale_map.get(from_scale, from_scale)
        to_unit = scale_map.get(to_scale, to_scale)
        
        return self.convert(value, from_unit, to_unit)
    
    def pressure_convert(self, value: Union[float, np.ndarray], from_unit: str, to_unit: str, 
                        gauge_to_absolute: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Specialized pressure conversion with gauge/absolute handling.
        
        Args:
            value: Pressure value
            from_unit: Source pressure unit
            to_unit: Target pressure unit
            gauge_to_absolute: Atmospheric pressure for gauge/absolute conversion (in from_unit)
            
        Returns:
            Converted pressure
        """
        # Handle gauge to absolute conversion if needed
        if gauge_to_absolute is not None:
            if 'psig' in from_unit.lower() and 'psia' in to_unit.lower():
                value = value + gauge_to_absolute
            elif 'psia' in from_unit.lower() and 'psig' in to_unit.lower():
                value = value - gauge_to_absolute
        
        return self.convert(value, from_unit, to_unit)
    
    def get_unit_registry(self) -> pint.UnitRegistry:
        """
        Get the underlying pint unit registry for advanced operations.
        
        Returns:
            pint.UnitRegistry instance
        """
        return self.ureg
    
    def format_quantity(self, value: Union[float, np.ndarray], unit: str, 
                       precision: int = 3, scientific: bool = False) -> str:
        """
        Format a quantity with units for display.
        
        Args:
            value: Numerical value
            unit: Unit string
            precision: Number of decimal places
            scientific: Whether to use scientific notation
            
        Returns:
            Formatted string with value and units
        """
        quantity = self.create_quantity(value, unit)
        if scientific:
            return f"{quantity.magnitude:.{precision}e} {quantity.units:~}"
        else:
            return f"{quantity.magnitude:.{precision}f} {quantity.units:~}"
    
    def get_dimensionality(self, unit: str) -> str:
        """
        Get the dimensionality of a unit (e.g., '[length]', '[mass] / [time]').
        
        Args:
            unit: Unit string
            
        Returns:
            Dimensionality string
        """
        return str(self.ureg.Quantity(1, unit).dimensionality)


# Convenience functions for common conversions
def convert_units(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Quick unit conversion function.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    units = ThermodynamicUnits()
    return units.convert(value, from_unit, to_unit)


def get_unit_registry() -> pint.UnitRegistry:
    """
    Get the global unit registry.
    
    Returns:
        pint.UnitRegistry instance
    """
    return ureg


# Create a global instance for convenience
_global_units = ThermodynamicUnits()

# Export commonly used functions
def temperature_to_kelvin(temp: Union[float, np.ndarray], from_scale: str) -> Union[float, np.ndarray]:
    """Convert temperature to Kelvin."""
    return _global_units.temperature_convert(temp, from_scale, 'K')


def kelvin_to_celsius(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Kelvin to Celsius."""
    return _global_units.temperature_convert(temp, 'K', 'C')


def celsius_to_kelvin(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius to Kelvin."""
    return _global_units.temperature_convert(temp, 'C', 'K')


def fahrenheit_to_celsius(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Fahrenheit to Celsius."""
    return _global_units.temperature_convert(temp, 'F', 'C')


def celsius_to_fahrenheit(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Celsius to Fahrenheit."""
    return _global_units.temperature_convert(temp, 'C', 'F')


def bar_to_pascal(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert bar to Pascal."""
    return _global_units.convert(pressure, 'bar', 'Pa')


def pascal_to_bar(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Pascal to bar.""" 
    return _global_units.convert(pressure, 'Pa', 'bar')


def psi_to_pascal(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert psi to Pascal."""
    return _global_units.convert(pressure, 'psi', 'Pa')


def pascal_to_psi(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Pascal to psi."""
    return _global_units.convert(pressure, 'Pa', 'psi')