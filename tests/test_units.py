"""
Unit tests for the thermsolve.units module.
"""

import pytest
import numpy as np
from thermsolve.units import (
    ThermodynamicUnits,
    convert_units,
    temperature_to_kelvin,
    celsius_to_kelvin,
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    bar_to_pascal,
    pascal_to_bar,
    psi_to_pascal,
    pascal_to_psi,
    get_unit_registry
)


class TestThermodynamicUnits:
    """Test cases for ThermodynamicUnits class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.units = ThermodynamicUnits()
    
    def test_temperature_conversions(self):
        """Test temperature unit conversions."""
        # Celsius to Kelvin
        assert abs(self.units.temperature_convert(0, 'C', 'K') - 273.15) < 1e-10
        assert abs(self.units.temperature_convert(100, 'C', 'K') - 373.15) < 1e-10
        
        # Celsius to Fahrenheit
        assert abs(self.units.temperature_convert(0, 'C', 'F') - 32.0) < 1e-10
        assert abs(self.units.temperature_convert(100, 'C', 'F') - 212.0) < 1e-10
        
        # Fahrenheit to Celsius
        assert abs(self.units.temperature_convert(32, 'F', 'C') - 0.0) < 1e-10
        assert abs(self.units.temperature_convert(212, 'F', 'C') - 100.0) < 1e-10
    
    def test_pressure_conversions(self):
        """Test pressure unit conversions."""
        # Bar to Pascal
        assert abs(self.units.convert(1, 'bar', 'Pa') - 100000) < 1e-6
        
        # Atmosphere to Pascal
        assert abs(self.units.convert(1, 'atm', 'Pa') - 101325) < 1
        
        # PSI to Pascal
        psi_to_pa = self.units.convert(1, 'psi', 'Pa')
        assert abs(psi_to_pa - 6894.76) < 1
    
    def test_density_conversions(self):
        """Test density unit conversions."""
        # kg/m³ to g/cm³
        assert abs(self.units.convert(1000, 'kg/m^3', 'g/cm^3') - 1.0) < 1e-10
        
        # g/cm³ to kg/m³
        assert abs(self.units.convert(1, 'g/cm^3', 'kg/m^3') - 1000) < 1e-10
    
    def test_heat_capacity_conversions(self):
        """Test heat capacity unit conversions."""
        # J/(kg·K) to cal/(g·K)
        cp_water_j = 4184  # J/(kg·K)
        cp_water_cal = self.units.convert(cp_water_j, 'J/(kg*K)', 'cal/(g*K)')
        assert abs(cp_water_cal - 1.0) < 0.01  # Should be approximately 1 cal/(g·K)
    
    def test_array_conversions(self):
        """Test conversions with numpy arrays."""
        temps_c = np.array([0, 25, 50, 100])
        temps_k = self.units.temperature_convert(temps_c, 'C', 'K')
        expected_k = np.array([273.15, 298.15, 323.15, 373.15])
        
        np.testing.assert_allclose(temps_k, expected_k, rtol=1e-10)
    
    def test_unit_validation(self):
        """Test unit validation functionality."""
        # Valid units
        assert self.units.validate_unit('K', 'temperature')
        assert self.units.validate_unit('Pa', 'pressure') 
        assert self.units.validate_unit('kg/m^3', 'density')
        assert self.units.validate_unit('J/(kg*K)', 'specific_heat')
        
        # Invalid units
        assert not self.units.validate_unit('m', 'temperature')
        assert not self.units.validate_unit('kg', 'pressure')
    
    def test_standard_units(self):
        """Test standard unit retrieval."""
        assert self.units.get_standard_unit('temperature') == 'K'
        assert self.units.get_standard_unit('pressure') == 'Pa'
        assert self.units.get_standard_unit('density') == 'kg/m^3'
        
        # Test unknown property
        with pytest.raises(ValueError):
            self.units.get_standard_unit('unknown_property')
    
    def test_common_units(self):
        """Test common unit retrieval."""
        temp_units = self.units.get_common_units('temperature')
        assert 'K' in temp_units
        assert 'degC' in temp_units
        assert 'degF' in temp_units
        
        pressure_units = self.units.get_common_units('pressure')
        assert 'Pa' in pressure_units
        assert 'bar' in pressure_units
        assert 'atm' in pressure_units
    
    def test_normalize_to_standard(self):
        """Test normalization to standard units."""
        # Temperature
        temp_k = self.units.normalize_to_standard(25, 'degC', 'temperature')
        assert abs(temp_k - 298.15) < 1e-10
        
        # Pressure
        pressure_pa = self.units.normalize_to_standard(1, 'bar', 'pressure')
        assert abs(pressure_pa - 100000) < 1e-6
    
    def test_quantity_creation(self):
        """Test pint quantity creation."""
        quantity = self.units.create_quantity(25, 'degC')
        assert quantity.magnitude == 25
        assert str(quantity.units) == 'degree_Celsius'
    
    def test_quantity_formatting(self):
        """Test quantity formatting."""
        formatted = self.units.format_quantity(298.15, 'K', precision=2)
        assert '298.15' in formatted
        assert 'K' in formatted
        
        formatted_sci = self.units.format_quantity(298.15, 'K', precision=2, scientific=True)
        assert '2.98e+02' in formatted_sci
    
    def test_dimensionality(self):
        """Test dimensionality checking."""
        temp_dim = self.units.get_dimensionality('K')
        assert '[temperature]' in temp_dim
        
        pressure_dim = self.units.get_dimensionality('Pa')
        assert '[mass]' in pressure_dim and '[length]' in pressure_dim and '[time]' in pressure_dim
    
    def test_pressure_gauge_absolute(self):
        """Test gauge/absolute pressure conversions."""
        # Convert 15 psig to psia (assuming 14.7 psi atmospheric)
        pressure_psia = self.units.pressure_convert(15, 'psig', 'psia', gauge_to_absolute=14.7)
        assert abs(pressure_psia - 29.7) < 1e-10
        
        # Convert 29.7 psia to psig
        pressure_psig = self.units.pressure_convert(29.7, 'psia', 'psig', gauge_to_absolute=14.7)
        assert abs(pressure_psig - 15.0) < 1e-10


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_temperature_functions(self):
        """Test temperature convenience functions."""
        assert abs(celsius_to_kelvin(0) - 273.15) < 1e-10
        assert abs(celsius_to_kelvin(100) - 373.15) < 1e-10
        
        assert abs(celsius_to_fahrenheit(0) - 32.0) < 1e-10
        assert abs(celsius_to_fahrenheit(100) - 212.0) < 1e-10
        
        assert abs(fahrenheit_to_celsius(32) - 0.0) < 1e-10
        assert abs(fahrenheit_to_celsius(212) - 100.0) < 1e-10
        
        assert abs(temperature_to_kelvin(0, 'C') - 273.15) < 1e-10
        assert abs(temperature_to_kelvin(32, 'F') - 273.15) < 1e-10
    
    def test_pressure_functions(self):
        """Test pressure convenience functions."""
        assert abs(bar_to_pascal(1) - 100000) < 1e-6
        assert abs(pascal_to_bar(100000) - 1.0) < 1e-10
        
        psi_pa = psi_to_pascal(1)
        assert abs(psi_pa - 6894.76) < 1
        
        pa_psi = pascal_to_psi(6894.76)
        assert abs(pa_psi - 1.0) < 1e-3
    
    def test_convert_units_function(self):
        """Test the standalone convert_units function."""
        assert abs(convert_units(100, 'degC', 'K') - 373.15) < 1e-10
        assert abs(convert_units(1, 'bar', 'Pa') - 100000) < 1e-6
    
    def test_get_unit_registry_function(self):
        """Test getting the unit registry."""
        ureg = get_unit_registry()
        quantity = ureg.Quantity(100, 'degC')
        assert quantity.magnitude == 100


class TestThermodynamicCalculations:
    """Test units in realistic thermodynamic calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.units = ThermodynamicUnits()
    
    def test_heat_capacity_calculation(self):
        """Test heat capacity calculation with unit conversions."""
        # Heat water from 20°C to 80°C
        mass_kg = 1.0
        cp_j_kg_k = 4184  # J/(kg·K)
        temp_initial_c = 20
        temp_final_c = 80
        
        # Convert to Kelvin
        temp_initial_k = self.units.temperature_convert(temp_initial_c, 'C', 'K')
        temp_final_k = self.units.temperature_convert(temp_final_c, 'C', 'K')
        
        delta_t = temp_final_k - temp_initial_k
        heat_j = mass_kg * cp_j_kg_k * delta_t
        
        # Convert to other units
        heat_kj = self.units.convert(heat_j, 'J', 'kJ')
        heat_cal = self.units.convert(heat_j, 'J', 'cal')
        
        # Check results
        assert abs(delta_t - 60) < 1e-10  # 60 K temperature difference
        assert abs(heat_j - 251040) < 1e-6  # Expected heat in Joules
        assert abs(heat_kj - 251.04) < 1e-6  # Expected heat in kJ
        assert abs(heat_cal - 59960) < 1  # Expected heat in cal (approximate)
    
    def test_ideal_gas_calculation(self):
        """Test ideal gas law calculation with unit conversions."""
        # PV = nRT calculation
        R = 8.314  # J/(mol·K)
        n_mol = 1.0  # mol
        T_c = 25.0  # °C
        P_bar = 1.0  # bar
        
        # Convert units
        T_k = self.units.temperature_convert(T_c, 'C', 'K')
        P_pa = self.units.convert(P_bar, 'bar', 'Pa')
        
        # Calculate volume
        V_m3 = (n_mol * R * T_k) / P_pa
        V_l = self.units.convert(V_m3, 'm^3', 'L')
        
        # Check results (should be close to 24.5 L for 1 mol at STP)
        expected_v_l = 24.79  # L (approximately)
        assert abs(V_l - expected_v_l) < 0.1
    
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation with unit conversions."""
        # Re = ρvD/μ
        rho_kg_m3 = 1000  # kg/m³ (water density)
        v_m_s = 1.0  # m/s (velocity)
        D_mm = 50  # mm (diameter)
        mu_cp = 1.0  # cP (dynamic viscosity)
        
        # Convert units
        D_m = self.units.convert(D_mm, 'mm', 'm')
        mu_pa_s = self.units.convert(mu_cp, 'cP', 'Pa*s')
        
        # Calculate Reynolds number (dimensionless)
        Re = (rho_kg_m3 * v_m_s * D_m) / mu_pa_s
        
        # Check result
        expected_Re = 50000  # Expected Reynolds number
        assert abs(Re - expected_Re) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
