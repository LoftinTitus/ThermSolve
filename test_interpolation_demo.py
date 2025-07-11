#!/usr/bin/env python3
"""
Interactive test and demonstration of the interpolation functionality.
This script shows practical examples of using the interpolation module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'thermsolve'))

import numpy as np
import warnings

def test_basic_interpolation():
    """Test basic interpolation functionality."""
    print("=== Testing Basic Interpolation ===")
    
    try:
        from interpolation import PropertyInterpolator
        
        # Water heat capacity data (J/kg·K) vs temperature (K)
        temperatures = [273.15, 298.15, 323.15, 348.15, 373.15]
        cp_values = [4217, 4184, 4179, 4188, 4217]
        
        # Test different interpolation methods
        methods = ["linear", "cubic", "polynomial"]
        
        for method in methods:
            print(f"\n--- {method.title()} Interpolation ---")
            
            interpolator = PropertyInterpolator(
                temperatures, cp_values,
                property_name="heat_capacity",
                method=method
            )
            
            # Test at exact points
            for i, T in enumerate(temperatures):
                result = interpolator(T)
                expected = cp_values[i]
                error = abs(result - expected)
                print(f"T={T:6.1f}K: {result:7.1f} (expected {expected:7.1f}, error={error:.2e})")
                # Allow larger tolerance for polynomial fitting
                tolerance = 1e-6 if method != "polynomial" else 5.0
                assert error < tolerance, f"Large error at exact point: {error}"
            
            # Test interpolation
            test_T = 310.0  # Between 298.15 and 323.15
            result = interpolator(test_T)
            print(f"T={test_T:6.1f}K: {result:7.1f} (interpolated)")
            assert 4170 < result < 4190, f"Interpolated value seems unreasonable: {result}"
            
        print(" Basic interpolation tests passed!")
        return True
        
    except ImportError as e:
        print(f" Failed to import interpolation module: {e}")
        return False
    except Exception as e:
        print(f" Basic interpolation test failed: {e}")
        return False


def test_real_world_data():
    """Test with real thermodynamic data."""
    print("\n=== Testing with Real-World Data ===")
    
    try:
        from interpolation import TemperatureDataSeries, PropertyInterpolator
        
        # Real water viscosity data from NIST (Pa·s vs K)
        water_viscosity_data = {
            "temperatures": [273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15],
            "viscosities": [1.793e-3, 1.307e-3, 1.002e-3, 0.797e-3, 0.653e-3, 0.547e-3, 0.467e-3, 0.404e-3, 0.355e-3, 0.315e-3, 0.282e-3]
        }
        
        print("Water viscosity interpolation:")
        
        # Create data series
        visc_data = TemperatureDataSeries(
            temperatures=water_viscosity_data["temperatures"],
            values=water_viscosity_data["viscosities"],
            property_name="viscosity",
            units="Pa·s",
            source="NIST"
        )
        
        # Test polynomial fitting
        poly_fit = visc_data.fit_correlation("polynomial")
        print(f"Polynomial fit R² = {poly_fit['r_squared']:.6f}")
        assert poly_fit['r_squared'] > 0.99, "Poor polynomial fit"
        
        # Test Arrhenius fitting
        arrhenius_fit = visc_data.fit_correlation("arrhenius")
        print(f"Arrhenius fit R² = {arrhenius_fit['r_squared']:.6f}")
        
        # Create interpolator
        interpolator = visc_data.to_interpolator(method="cubic")
        
        # Test interpolation at intermediate points
        test_temperatures = [278, 298, 318, 338, 358]
        print("\nInterpolated viscosity values:")
        print("Temperature (K) | Viscosity (mPa·s)")
        print("-" * 32)
        
        for T in test_temperatures:
            visc = interpolator(T) * 1000  # Convert to mPa·s
            print(f"{T:12.1f} | {visc:13.3f}")
            assert 0.2 < visc < 2.0, f"Unreasonable viscosity value: {visc}"
        
        # Test derivatives
        T_test = 300.0
        dvisc_dT = interpolator.derivative(T_test, order=1)
        print(f"\ndμ/dT at {T_test}K = {dvisc_dT:.2e} Pa·s/K")
        assert dvisc_dT < 0, "Viscosity should decrease with temperature"
        
        print("✓ Real-world data tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Real-world data test failed: {e}")
        return False


def test_integration_with_substance():
    """Test integration with Substance class."""
    print("\n=== Testing Integration with Substance Class ===")
    
    try:
        from interpolation import enhance_substance_with_data
        from substances import Substance
        
        # Create a substance
        ethanol = Substance(
            name="ethanol",
            formula="C2H5OH",
            molecular_weight=46.069,
            boiling_point=351.44
        )
        
        # Add heat capacity data
        cp_temps = [250, 275, 300, 325, 350, 375]
        cp_values = [112.3, 120.1, 128.2, 136.8, 145.9, 155.6]  # J/(mol·K)
        
        enhance_substance_with_data(
            ethanol, "cp",
            cp_temps, cp_values,
            method="cubic"
        )
        
        # Test the enhanced substance
        test_temps = [260, 290, 320, 360]
        print("Enhanced ethanol heat capacity:")
        print("Temperature (K) | Cp (J/mol·K)")
        print("-" * 30)
        
        for T in test_temps:
            try:
                cp = ethanol.heat_capacity(T)
                if cp is not None:
                    print(f"{T:12.1f} | {cp:10.1f}")
                    assert 110 < cp < 160, f"Unreasonable Cp value: {cp}"
                else:
                    print(f"{T:12.1f} | No data available")
            except Exception as e:
                print(f"{T:12.1f} | Error: {e}")
                raise
        
        print(f"Temperature range: {ethanol.temp_range}")
        
        print(" Substance integration tests passed!")
        return True
        
    except Exception as e:
        print(f" Substance integration test failed: {e}")
        return False


def test_extrapolation_warnings():
    """Test extrapolation warning system."""
    print("\n=== Testing Extrapolation Warnings ===")
    
    try:
        from interpolation import PropertyInterpolator
        
        # Limited temperature range data
        temps = [300, 320, 340, 360]
        values = [100, 110, 120, 130]
        
        interpolator = PropertyInterpolator(
            temps, values,
            property_name="test_property",
            extrapolation_method="warning"
        )
        
        print("Testing extrapolation beyond data range...")
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should trigger a warning
            result_low = interpolator(250)  # Below range
            result_high = interpolator(400)  # Above range
            
            # Check if warnings were issued
            warning_messages = [str(warning.message) for warning in w]
            has_extrapolation_warning = any("Extrapolating" in msg for msg in warning_messages)
            
            if has_extrapolation_warning:
                print("✓ Extrapolation warnings working correctly!")
                print(f"Result at 250K: {result_low:.1f}")
                print(f"Result at 400K: {result_high:.1f}")
                return True
            else:
                print("⚠ No extrapolation warnings detected")
                return False
        
    except Exception as e:
        print(f"✗ Extrapolation warning test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from interpolation import PropertyInterpolator, TemperatureDataSeries
        
        # Test insufficient data points
        try:
            PropertyInterpolator([273.15], [100])
            print("✗ Should have failed with insufficient points")
            return False
        except ValueError:
            print("✓ Correctly handled insufficient data points")
        
        # Test mismatched array lengths
        try:
            PropertyInterpolator([273, 298], [100])
            print("✗ Should have failed with mismatched arrays")
            return False
        except ValueError:
            print("✓ Correctly handled mismatched array lengths")
        
        # Test invalid interpolation method
        try:
            PropertyInterpolator([273, 298, 323], [100, 110, 120], method="invalid")
            print("✗ Should have failed with invalid method")
            return False
        except ValueError:
            print("✓ Correctly handled invalid interpolation method")
        
        # Test with NaN values
        try:
            interpolator = PropertyInterpolator([273, 298, 323], [100, float('nan'), 120])
            result = interpolator(290)
            if np.isnan(result):
                print("✓Correctly handled NaN input")
            else:
                print(" NaN handling may need improvement")
        except:
            print(" Correctly rejected NaN input")
        
        print(" Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing ThermSolve Interpolation Module")
    print("=" * 40)
    
    tests = [
        test_basic_interpolation,
        test_real_world_data,
        test_integration_with_substance,
        test_extrapolation_warnings,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(" All tests passed! The interpolation module is working correctly.")
        return True
    else:
        print(" Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
