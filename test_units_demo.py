"""
Test script to demonstrate the ThermodynamicUnits class functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermsolve.units import (
    ThermodynamicUnits, 
    convert_units,
    temperature_to_kelvin,
    celsius_to_kelvin,
    celsius_to_fahrenheit,
    bar_to_pascal,
    psi_to_pascal
)
import numpy as np

def test_basic_conversions():
    """Test basic unit conversions."""
    print("=== Basic Unit Conversions ===")
    
    units = ThermodynamicUnits()
    
    # Temperature conversions
    temp_c = 25.0
    temp_k = units.temperature_convert(temp_c, 'C', 'K')
    temp_f = units.temperature_convert(temp_c, 'C', 'F')
    print(f"Temperature: {temp_c}°C = {temp_k} K = {temp_f}°F")
    
    # Pressure conversions
    pressure_bar = 2.5
    pressure_pa = units.convert(pressure_bar, 'bar', 'Pa')
    pressure_psi = units.convert(pressure_bar, 'bar', 'psi')
    print(f"Pressure: {pressure_bar} bar = {pressure_pa} Pa = {pressure_psi:.2f} psi")
    
    # Density conversions
    density_kg_m3 = 1000
    density_g_cm3 = units.convert(density_kg_m3, 'kg/m^3', 'g/cm^3')
    print(f"Density: {density_kg_m3} kg/m³ = {density_g_cm3} g/cm³")
    
    # Heat capacity conversions
    cp_j_kg_k = 4184
    cp_cal_g_k = units.convert(cp_j_kg_k, 'J/(kg*K)', 'cal/(g*K)')
    print(f"Heat capacity: {cp_j_kg_k} J/(kg·K) = {cp_cal_g_k:.3f} cal/(g·K)")

def test_array_conversions():
    """Test conversions with numpy arrays."""
    print("\n=== Array Conversions ===")
    
    units = ThermodynamicUnits()
    
    # Temperature array
    temps_c = np.array([0, 25, 50, 100])
    temps_k = units.temperature_convert(temps_c, 'C', 'K')
    temps_f = units.temperature_convert(temps_c, 'C', 'F')
    
    print("Temperature conversions:")
    for c, k, f in zip(temps_c, temps_k, temps_f):
        print(f"  {c}°C = {k} K = {f}°F")

def test_unit_validation():
    """Test unit validation functionality."""
    print("\n=== Unit Validation ===")
    
    units = ThermodynamicUnits()
    
    # Test valid units
    valid_tests = [
        ('K', 'temperature'),
        ('Pa', 'pressure'),
        ('kg/m^3', 'density'),
        ('J/(kg*K)', 'specific_heat'),
        ('W/(m*K)', 'thermal_conductivity')
    ]
    
    print("Valid unit tests:")
    for unit, prop_type in valid_tests:
        is_valid = units.validate_unit(unit, prop_type)
        print(f"  {unit} for {prop_type}: {'✓' if is_valid else '✗'}")
    
    # Test invalid units
    invalid_tests = [
        ('m', 'temperature'),  # Length unit for temperature
        ('kg', 'pressure'),    # Mass unit for pressure
    ]
    
    print("\nInvalid unit tests:")
    for unit, prop_type in invalid_tests:
        is_valid = units.validate_unit(unit, prop_type)
        print(f"  {unit} for {prop_type}: {'✓' if is_valid else '✗'}")

def test_standard_units():
    """Test standard unit functionality."""
    print("\n=== Standard Units ===")
    
    units = ThermodynamicUnits()
    
    properties = ['temperature', 'pressure', 'density', 'viscosity_dynamic', 'thermal_conductivity']
    
    for prop in properties:
        standard = units.get_standard_unit(prop)
        common = units.get_common_units(prop)
        print(f"{prop}: standard = {standard}, common = {common[:3]}...")

def test_quantity_formatting():
    """Test quantity formatting."""
    print("\n=== Quantity Formatting ===")
    
    units = ThermodynamicUnits()
    
    # Various quantities with formatting
    examples = [
        (298.15, 'K', 'Temperature'),
        (101325, 'Pa', 'Pressure'),
        (1000, 'kg/m^3', 'Density'),
        (0.001, 'Pa*s', 'Viscosity'),
        (2.5e6, 'J/kg', 'Enthalpy')
    ]
    
    for value, unit, description in examples:
        formatted_normal = units.format_quantity(value, unit, precision=2)
        formatted_sci = units.format_quantity(value, unit, precision=2, scientific=True)
        print(f"{description}:")
        print(f"  Normal: {formatted_normal}")
        print(f"  Scientific: {formatted_sci}")

def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Convenience Functions ===")
    
    # Temperature conversions
    temp_c = 25.0
    temp_k = celsius_to_kelvin(temp_c)
    temp_f = celsius_to_fahrenheit(temp_c)
    print(f"Using convenience functions:")
    print(f"  {temp_c}°C = {temp_k} K = {temp_f}°F")
    
    # Pressure conversions
    pressure_bar = 1.5
    pressure_pa = bar_to_pascal(pressure_bar)
    pressure_psi_val = 20.0
    pressure_pa_from_psi = psi_to_pascal(pressure_psi_val)
    
    print(f"  {pressure_bar} bar = {pressure_pa} Pa")
    print(f"  {pressure_psi_val} psi = {pressure_pa_from_psi:.0f} Pa")

def test_thermodynamic_calculations():
    """Test units in context of thermodynamic calculations."""
    print("\n=== Thermodynamic Calculation Example ===")
    
    units = ThermodynamicUnits()
    
    # Example: Heat required to heat water
    mass_kg = 1.0  # kg of water
    cp_water = 4184  # J/(kg·K) specific heat of water
    temp_initial_c = 20  # °C
    temp_final_c = 80   # °C
    
    # Convert temperatures to Kelvin
    temp_initial_k = units.temperature_convert(temp_initial_c, 'C', 'K')
    temp_final_k = units.temperature_convert(temp_final_c, 'C', 'K')
    
    delta_t = temp_final_k - temp_initial_k
    
    # Calculate heat required
    heat_j = mass_kg * cp_water * delta_t
    heat_kj = units.convert(heat_j, 'J', 'kJ')
    heat_cal = units.convert(heat_j, 'J', 'cal')
    heat_btu = units.convert(heat_j, 'J', 'Btu')
    
    print(f"Heating {mass_kg} kg water from {temp_initial_c}°C to {temp_final_c}°C:")
    print(f"  Heat required: {heat_j:.0f} J = {heat_kj:.1f} kJ = {heat_cal:.0f} cal = {heat_btu:.2f} Btu")

if __name__ == "__main__":
    print("ThermodynamicUnits Class Demonstration")
    print("=" * 50)
    
    test_basic_conversions()
    test_array_conversions()
    test_unit_validation()
    test_standard_units()
    test_quantity_formatting()
    test_convenience_functions()
    test_thermodynamic_calculations()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
