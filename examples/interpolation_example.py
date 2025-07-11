#!/usr/bin/env python3
"""
Example demonstrating interpolation functionality in ThermSolve.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

import numpy as np
from interpolation import PropertyInterpolator, TemperatureDataSeries, enhance_substance_with_data
from substances import Substance

def main():
    print("=== ThermSolve Interpolation Example ===\n")
    
    # Example 1: Basic interpolation
    print("Example 1: Basic Property Interpolation")
    print("-" * 40)
    
    # Temperature-dependent heat capacity data for water (example data)
    temperatures = [273.15, 298.15, 323.15, 373.15, 423.15, 473.15]  # K
    cp_values = [4217, 4184, 4179, 4217, 4312, 4459]  # J/(kg·K)
    
    # Create interpolator
    cp_interpolator = PropertyInterpolator(
        temperatures, cp_values, 
        property_name="heat_capacity",
        method="cubic"
    )
    
    # Test interpolation at various temperatures
    test_temps = [280, 300, 350, 400, 450]
    print("Temperature (K) | Heat Capacity (J/kg·K)")
    print("-" * 35)
    for T in test_temps:
        cp = cp_interpolator(T)
        print(f"{T:12.1f} | {cp:18.1f}")
    
    print()
    
    # Example 2: Using TemperatureDataSeries
    print("Example 2: Temperature Data Series")
    print("-" * 40)
    
    # Viscosity data for ethanol
    temps_visc = [273.15, 298.15, 323.15, 348.15, 373.15]
    visc_values = [1.773e-3, 1.074e-3, 0.694e-3, 0.476e-3, 0.346e-3]  # Pa·s
    
    visc_data = TemperatureDataSeries(
        temperatures=temps_visc,
        values=visc_values,
        property_name="viscosity",
        units="Pa·s",
        source="NIST Webbook"
    )
    
    # Fit correlations
    poly_fit = visc_data.fit_correlation("polynomial")
    arrhenius_fit = visc_data.fit_correlation("arrhenius")
    
    print(f"Polynomial fit R² = {poly_fit['r_squared']:.4f}")
    print(f"Arrhenius fit R² = {arrhenius_fit['r_squared']:.4f}")
    
    # Convert to interpolator
    visc_interpolator = visc_data.to_interpolator(method="cubic")
    
    print("\nViscosity interpolation:")
    print("Temperature (K) | Viscosity (mPa·s)")
    print("-" * 32)
    for T in [280, 310, 340]:
        visc = visc_interpolator(T) * 1000  # Convert to mPa·s
        print(f"{T:12.1f} | {visc:13.2f}")
    
    print()
    
    # Example 3: Integration with Substance class
    print("Example 3: Enhanced Substance with Interpolated Data")
    print("-" * 50)
    
    # Create a basic substance
    ethanol = Substance(
        name="ethanol",
        formula="C2H5OH",
        molecular_weight=46.069,
        boiling_point=351.44
    )
    
    # Add interpolated viscosity data
    enhance_substance_with_data(
        ethanol, "viscosity", 
        temps_visc, visc_values,
        method="cubic"
    )
    
    # Test the enhanced substance
    print(f"Ethanol viscosity at 300 K: {ethanol.viscosity(300.0):.6f} Pa·s")
    print(f"Ethanol viscosity at 320 K: {ethanol.viscosity(320.0):.6f} Pa·s")
    print(f"Temperature range: {ethanol.temp_range}")
    
    print()
    
    # Example 4: Derivative calculations
    print("Example 4: Property Derivatives")
    print("-" * 30)
    
    # Calculate temperature derivative of heat capacity
    T_test = 350.0
    dcp_dT = cp_interpolator.derivative(T_test, order=1)
    d2cp_dT2 = cp_interpolator.derivative(T_test, order=2)
    
    print(f"At {T_test} K:")
    print(f"dCp/dT = {dcp_dT:.2f} J/(kg·K²)")
    print(f"d²Cp/dT² = {d2cp_dT2:.4f} J/(kg·K³)")
    
    print()
    
    # Example 5: Extrapolation warnings
    print("Example 5: Extrapolation Behavior")
    print("-" * 35)
    
    print("Testing extrapolation beyond data range:")
    try:
        # This should trigger a warning
        cp_high = cp_interpolator(500.0)  # Beyond max temperature
        print(f"Cp at 500 K (extrapolated): {cp_high:.1f} J/(kg·K)")
        
        cp_low = cp_interpolator(250.0)   # Below min temperature  
        print(f"Cp at 250 K (extrapolated): {cp_low:.1f} J/(kg·K)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
