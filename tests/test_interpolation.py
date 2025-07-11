"""
Tests for the interpolation module.
"""

import pytest
import numpy as np
import sys
import os

# Add the thermsolve directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

try:
    from interpolation import PropertyInterpolator, TemperatureDataSeries
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="Interpolation module not available")
class TestPropertyInterpolator:
    """Test the PropertyInterpolator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.temperatures = [273.15, 298.15, 323.15, 373.15]
        self.values = [1000, 1100, 1200, 1400]
        
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        interpolator = PropertyInterpolator(
            self.temperatures, self.values, method="linear"
        )
        
        # Test exact points
        assert abs(interpolator(273.15) - 1000) < 1e-10
        assert abs(interpolator(373.15) - 1400) < 1e-10
        
        # Test interpolated point
        result = interpolator(348.15)  # Midpoint between 323.15 and 373.15
        expected = (1200 + 1400) / 2
        assert abs(result - expected) < 1  # Allow some tolerance
        
    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        interpolator = PropertyInterpolator(
            self.temperatures, self.values, method="cubic"
        )
        
        # Test exact points
        assert abs(interpolator(273.15) - 1000) < 1e-6
        assert abs(interpolator(373.15) - 1400) < 1e-6
        
        # Test smoothness (no large jumps)
        T_test = np.linspace(273.15, 373.15, 10)
        results = interpolator(T_test)
        
        # All values should be reasonable
        assert np.all(results > 900)
        assert np.all(results < 1500)
        
    def test_polynomial_interpolation(self):
        """Test polynomial interpolation."""
        interpolator = PropertyInterpolator(
            self.temperatures, self.values, method="polynomial"
        )
        
        # Test exact points
        assert abs(interpolator(273.15) - 1000) < 1e-6
        assert abs(interpolator(373.15) - 1400) < 1e-6
        
    def test_array_input(self):
        """Test interpolation with array input."""
        interpolator = PropertyInterpolator(
            self.temperatures, self.values, method="linear"
        )
        
        T_array = np.array([280, 300, 350])
        results = interpolator(T_array)
        
        assert len(results) == 3
        assert np.all(np.isfinite(results))
        
    def test_derivatives(self):
        """Test derivative calculations."""
        # Use more data points for better derivatives
        T = np.linspace(273, 373, 10)
        y = 1000 + 2*T + 0.01*T**2  # Quadratic function
        
        interpolator = PropertyInterpolator(T.tolist(), y.tolist(), method="cubic")
        
        # Test first derivative (should be close to 2 + 0.02*T)
        T_test = 300
        dydT = interpolator.derivative(T_test, order=1)
        expected_dydT = 2 + 0.02 * T_test
        
        assert abs(dydT - expected_dydT) < 0.5  # Allow some numerical error
        
    def test_insufficient_points(self):
        """Test behavior with insufficient points."""
        with pytest.raises(ValueError):
            PropertyInterpolator([273.15], [1000])
            
    def test_mismatched_arrays(self):
        """Test error handling for mismatched array lengths."""
        with pytest.raises(ValueError):
            PropertyInterpolator([273, 298, 323], [1000, 1100])


@pytest.mark.skipif(not HAS_INTERPOLATION, reason="Interpolation module not available")
class TestTemperatureDataSeries:
    """Test the TemperatureDataSeries class."""
    
    def setup_method(self):
        """Set up test data."""
        self.temperatures = [273.15, 298.15, 323.15, 373.15]
        self.values = [1.5, 1.2, 0.9, 0.6]  # Decreasing values (like viscosity)
        
    def test_initialization(self):
        """Test basic initialization."""
        data = TemperatureDataSeries(
            self.temperatures, self.values,
            property_name="viscosity",
            units="mPa·s"
        )
        
        assert data.property_name == "viscosity"
        assert data.units == "mPa·s"
        assert len(data.temperatures) == 4
        assert len(data.values) == 4
        
    def test_to_interpolator(self):
        """Test conversion to interpolator."""
        data = TemperatureDataSeries(
            self.temperatures, self.values,
            property_name="viscosity"
        )
        
        interpolator = data.to_interpolator(method="linear")
        
        # Test that interpolator works
        result = interpolator(300)
        assert isinstance(result, float)
        assert result > 0
        
    def test_polynomial_fitting(self):
        """Test polynomial correlation fitting."""
        data = TemperatureDataSeries(
            self.temperatures, self.values,
            property_name="test_property"
        )
        
        fit_result = data.fit_correlation("polynomial")
        
        assert fit_result["type"] == "polynomial"
        assert "coefficients" in fit_result
        assert "r_squared" in fit_result
        assert 0 <= fit_result["r_squared"] <= 1
        
    def test_uncertainty_handling(self):
        """Test handling of uncertainty data."""
        uncertainties = [0.1, 0.08, 0.06, 0.04]
        
        data = TemperatureDataSeries(
            self.temperatures, self.values,
            property_name="test_property",
            uncertainty=uncertainties
        )
        
        assert data.uncertainty is not None
        assert len(data.uncertainty) == 4


if __name__ == "__main__":
    pytest.main([__file__])
