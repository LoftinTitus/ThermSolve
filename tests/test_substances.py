"""
Test suite for the Substance class and SubstanceDatabase.
"""

import pytest
import json
import csv
import os
import tempfile
from pathlib import Path
import sys

# Add the thermsolve package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

from substances import Substance, SubstanceDatabase


class TestSubstance:
    """Test cases for the Substance class."""
    
    def test_substance_initialization(self):
        """Test basic substance initialization."""
        substance = Substance(
            name="water",
            formula="H2O",
            molecular_weight=18.015,
            melting_point=273.15,
            boiling_point=373.15
        )
        
        assert substance.name == "water"
        assert substance.formula == "H2O"
        assert substance.molecular_weight == 18.015
        assert substance.melting_point == 273.15
        assert substance.boiling_point == 373.15
        assert substance.reference_temperature == 298.15  # default
        assert substance.temp_range == (0, 1000)  # default
    
    def test_substance_from_dict(self):
        """Test creating substance from dictionary."""
        data = {
            "name": "ethanol",
            "formula": "C2H6O",
            "molecular_weight": 46.068,
            "melting_point": 159.05,
            "boiling_point": 351.44,
            "cp_coefficients": {
                "type": "constant",
                "value": 2440
            }
        }
        
        substance = Substance.from_dict(data)
        
        assert substance.name == "ethanol"
        assert substance.formula == "C2H6O"
        assert substance.molecular_weight == 46.068
        assert substance.cp_coefficients["type"] == "constant"
        assert substance.cp_coefficients["value"] == 2440
    
    def test_substance_to_dict(self):
        """Test converting substance to dictionary."""
        substance = Substance(
            name="methane",
            formula="CH4",
            molecular_weight=16.043,
            melting_point=90.7,
            boiling_point=111.66
        )
        
        data = substance.to_dict()
        
        assert data["name"] == "methane"
        assert data["formula"] == "CH4"
        assert data["molecular_weight"] == 16.043
        assert data["melting_point"] == 90.7
        assert data["boiling_point"] == 111.66
        assert "reference_temperature" in data
        assert "temp_range" in data
    
    def test_heat_capacity_constant(self):
        """Test heat capacity calculation with constant value."""
        substance = Substance(
            name="test_substance",
            cp_coefficients={
                "type": "constant",
                "value": 4184
            }
        )
        
        cp = substance.heat_capacity(298.15)
        assert cp == 4184
        
        # Should be same at any temperature for constant
        cp2 = substance.heat_capacity(350.0)
        assert cp2 == 4184
    
    def test_heat_capacity_polynomial(self):
        """Test heat capacity calculation with polynomial coefficients."""
        substance = Substance(
            name="test_substance",
            cp_coefficients={
                "type": "polynomial",
                "coefficients": [1000, 2, 0.001]  # Cp = 1000 + 2*T + 0.001*T^2
            }
        )
        
        T = 300.0
        expected_cp = 1000 + 2*T + 0.001*T**2
        cp = substance.heat_capacity(T)
        assert abs(cp - expected_cp) < 1e-6
    
    def test_heat_capacity_no_data(self):
        """Test heat capacity calculation when no data is available."""
        substance = Substance(name="test_substance")
        
        with pytest.raises(ValueError, match="Heat capacity data not available"):
            substance.heat_capacity(298.15)
    
    def test_viscosity_arrhenius(self):
        """Test viscosity calculation with Arrhenius-type equation."""
        substance = Substance(
            name="test_substance",
            viscosity_coefficients={
                "type": "arrhenius",
                "A": 0.001,
                "B": 1000,
                "n": 0.5,
                "C": 100
            }
        )
        
        T = 300.0
        # μ = A * T^n * (1 + C/T)
        expected_viscosity = 0.001 * (T**0.5) * (1 + 100/T)
        viscosity = substance.viscosity(T)
        assert abs(viscosity - expected_viscosity) < 1e-6
    
    def test_density_linear(self):
        """Test density calculation with linear temperature dependence."""
        substance = Substance(
            name="test_substance",
            density_coefficients={
                "type": "linear",
                "A": 1000,
                "B": -0.5
            }
        )
        
        T = 300.0
        expected_density = 1000 - 0.5 * T
        density = substance.density(T)
        assert density == expected_density
    
    def test_density_constant(self):
        """Test density calculation with constant value."""
        substance = Substance(
            name="test_substance",
            density_coefficients={
                "type": "constant",
                "value": 997.0
            }
        )
        
        density = substance.density(298.15)
        assert density == 997.0
    
    def test_vapor_pressure_antoine(self):
        """Test vapor pressure calculation using Antoine equation."""
        substance = Substance(
            name="test_substance",
            vapor_pressure_coefficients={
                "type": "antoine",
                "A": 8.20417,
                "B": 1642.89,
                "C": -42.85
            }
        )
        
        T = 298.15
        # Antoine: log10(P) = A - B/(C + T)
        expected_log_p = 8.20417 - 1642.89/(-42.85 + T)
        expected_p = 10**expected_log_p
        
        p = substance.vapor_pressure(T)
        assert abs(p - expected_p) < 1e-6
    
    def test_thermal_conductivity_polynomial(self):
        """Test thermal conductivity calculation with polynomial coefficients."""
        substance = Substance(
            name="test_substance",
            thermal_conductivity_coefficients={
                "type": "polynomial",
                "coefficients": [0.1, 0.0001, -1e-7]  # k = 0.1 + 0.0001*T - 1e-7*T^2
            }
        )
        
        T = 300.0
        expected_k = 0.1 + 0.0001*T - 1e-7*T**2
        k = substance.thermal_conductivity(T)
        assert abs(k - expected_k) < 1e-9
    
    def test_temperature_range_warning(self):
        """Test temperature range validation warnings."""
        substance = Substance(
            name="test_substance",
            temp_range=(200, 400),
            cp_coefficients={
                "type": "constant",
                "value": 1000
            }
        )
        
        # Should work without warning
        cp = substance.heat_capacity(300)
        assert cp == 1000
        
        # Should issue warning for out-of-range temperature
        with pytest.warns(UserWarning, match="Temperature.*outside valid range"):
            substance.heat_capacity(500)  # Above range
        
        with pytest.warns(UserWarning, match="Temperature.*outside valid range"):
            substance.heat_capacity(100)  # Below range
    
    def test_json_serialization(self):
        """Test JSON save and load functionality."""
        substance = Substance(
            name="test_substance",
            formula="H2O",
            molecular_weight=18.015,
            melting_point=273.15,
            cp_coefficients={
                "type": "constant",
                "value": 4184
            }
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to JSON
            substance.to_json(temp_path)
            
            # Load from JSON
            loaded_substance = Substance.from_json(temp_path)
            
            # Verify data
            assert loaded_substance.name == substance.name
            assert loaded_substance.formula == substance.formula
            assert loaded_substance.molecular_weight == substance.molecular_weight
            assert loaded_substance.melting_point == substance.melting_point
            assert loaded_substance.cp_coefficients == substance.cp_coefficients
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_string_representation(self):
        """Test string representation methods."""
        substance = Substance(
            name="water",
            formula="H2O",
            molecular_weight=18.015
        )
        
        str_repr = str(substance)
        assert "water" in str_repr
        assert "H2O" in str_repr
        assert "18.015" in str_repr
        
        # __repr__ should be same as __str__
        assert repr(substance) == str(substance)


class TestSubstanceDatabase:
    """Test cases for the SubstanceDatabase class."""
    
    def test_database_initialization(self):
        """Test database initialization."""
        db = SubstanceDatabase()
        assert len(db.substances) == 0
        assert db.list_substances() == []
    
    def test_add_and_get_substance(self):
        """Test adding and retrieving substances."""
        db = SubstanceDatabase()
        
        water = Substance(name="Water", formula="H2O", molecular_weight=18.015)
        db.add_substance(water)
        
        # Should be stored with lowercase name
        assert len(db.substances) == 1
        assert "water" in db.substances
        
        # Should be retrievable by name (case-insensitive)
        retrieved = db.get_substance("water")
        assert retrieved is not None
        assert retrieved.name == "Water"
        
        retrieved2 = db.get_substance("WATER")
        assert retrieved2 is not None
        assert retrieved2.name == "Water"
    
    def test_list_substances(self):
        """Test listing all substances."""
        db = SubstanceDatabase()
        
        water = Substance(name="Water", formula="H2O")
        ethanol = Substance(name="Ethanol", formula="C2H6O")
        
        db.add_substance(water)
        db.add_substance(ethanol)
        
        substances = db.list_substances()
        assert len(substances) == 2
        assert "water" in substances
        assert "ethanol" in substances
    
    def test_search_by_formula(self):
        """Test searching by chemical formula."""
        db = SubstanceDatabase()
        
        water = Substance(name="Water", formula="H2O")
        heavy_water = Substance(name="Heavy Water", formula="D2O")
        ethanol = Substance(name="Ethanol", formula="C2H6O")
        
        db.add_substance(water)
        db.add_substance(heavy_water)
        db.add_substance(ethanol)
        
        # Search for H2O
        h2o_substances = db.search_by_formula("H2O")
        assert len(h2o_substances) == 1
        assert h2o_substances[0].name == "Water"
        
        # Search for non-existent formula
        results = db.search_by_formula("XYZ")
        assert len(results) == 0
    
    def test_search_by_cas(self):
        """Test searching by CAS number."""
        db = SubstanceDatabase()
        
        water = Substance(name="Water", cas_number="7732-18-5")
        ethanol = Substance(name="Ethanol", cas_number="64-17-5")
        
        db.add_substance(water)
        db.add_substance(ethanol)
        
        # Search for water's CAS
        found = db.search_by_cas("7732-18-5")
        assert found is not None
        assert found.name == "Water"
        
        # Search for non-existent CAS
        not_found = db.search_by_cas("12345-67-8")
        assert not_found is None
    
    def test_csv_operations(self):
        """Test CSV save and load operations."""
        db = SubstanceDatabase()
        
        # Create test substances
        water = Substance(
            name="water",
            formula="H2O",
            molecular_weight=18.015,
            melting_point=273.15,
            boiling_point=373.15
        )
        
        ethanol = Substance(
            name="ethanol",
            formula="C2H6O",
            molecular_weight=46.068,
            melting_point=159.05,
            boiling_point=351.44
        )
        
        db.add_substance(water)
        db.add_substance(ethanol)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to CSV
            db.save_to_csv(temp_path)
            
            # Create new database and load from CSV
            new_db = SubstanceDatabase()
            new_db.load_from_csv(temp_path)
            
            # Verify data
            assert len(new_db.substances) == 2
            
            loaded_water = new_db.get_substance("water")
            assert loaded_water is not None
            assert loaded_water.name == "water"
            assert loaded_water.formula == "H2O"
            assert loaded_water.molecular_weight == 18.015
            
            loaded_ethanol = new_db.get_substance("ethanol")
            assert loaded_ethanol is not None
            assert loaded_ethanol.name == "ethanol"
            assert loaded_ethanol.formula == "C2H6O"
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_empty_database_csv_save(self):
        """Test saving empty database to CSV."""
        db = SubstanceDatabase()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should handle empty database gracefully
            db.save_to_csv(temp_path)
            
            # File should exist but be empty (or just headers)
            assert os.path.exists(temp_path)
            
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete workflow from creation to calculation."""
        # Create substance with all properties
        ethanol = Substance(
            name="ethanol",
            formula="C2H6O",
            molecular_weight=46.068,
            melting_point=159.05,
            boiling_point=351.44,
            cp_coefficients={
                "type": "polynomial",
                "coefficients": [2440, 0.5, -0.001]
            },
            density_coefficients={
                "type": "linear",
                "A": 850,
                "B": -1.0
            },
            vapor_pressure_coefficients={
                "type": "antoine",
                "A": 8.20417,
                "B": 1642.89,
                "C": -42.85
            },
            temp_range=(200, 400)
        )
        
        # Test property calculations
        T = 298.15
        
        cp = ethanol.heat_capacity(T)
        assert cp is not None
        assert cp > 0
        
        density = ethanol.density(T)
        assert density is not None
        assert density > 0
        
        vp = ethanol.vapor_pressure(T)
        assert vp is not None
        assert vp > 0
        
        # Test serialization
        data = ethanol.to_dict()
        ethanol2 = Substance.from_dict(data)
        
        # Properties should be identical
        assert ethanol2.heat_capacity(T) == ethanol.heat_capacity(T)
        assert ethanol2.density(T) == ethanol.density(T)
        assert ethanol2.vapor_pressure(T) == ethanol.vapor_pressure(T)
    
    def test_database_with_calculations(self):
        """Test database operations with property calculations."""
        db = SubstanceDatabase()
        
        # Add substances with different property types
        water = Substance(
            name="water",
            cp_coefficients={"type": "constant", "value": 4184}
        )
        
        ethanol = Substance(
            name="ethanol",
            cp_coefficients={
                "type": "polynomial",
                "coefficients": [2440, 0.5]
            }
        )
        
        db.add_substance(water)
        db.add_substance(ethanol)
        
        # Test calculations for substances from database
        T = 298.15
        
        water_from_db = db.get_substance("water")
        cp_water = water_from_db.heat_capacity(T)
        assert cp_water == 4184
        
        ethanol_from_db = db.get_substance("ethanol")
        cp_ethanol = ethanol_from_db.heat_capacity(T)
        expected_cp = 2440 + 0.5 * T
        assert abs(cp_ethanol - expected_cp) < 1e-6


# Test fixtures and utilities
@pytest.fixture
def sample_substance():
    """Fixture providing a sample substance for testing."""
    return Substance(
        name="test_substance",
        formula="C2H6O",
        molecular_weight=46.068,
        melting_point=159.05,
        boiling_point=351.44,
        cp_coefficients={
            "type": "constant",
            "value": 2440
        }
    )


@pytest.fixture
def sample_database():
    """Fixture providing a sample database with substances."""
    db = SubstanceDatabase()
    
    water = Substance(name="water", formula="H2O", molecular_weight=18.015)
    ethanol = Substance(name="ethanol", formula="C2H6O", molecular_weight=46.068)
    
    db.add_substance(water)
    db.add_substance(ethanol)
    
    return db


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
