#!/usr/bin/env python3
"""
Simple test runner for ThermSolve without pytest dependency.
This can be used if pytest is not available.
"""

import sys
import os
import unittest
from io import StringIO

# Add the thermsolve package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

try:
    from substances import Substance, SubstanceDatabase
    SUBSTANCES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import substances module: {e}")
    SUBSTANCES_AVAILABLE = False


class TestSubstanceBasic(unittest.TestCase):
    """Basic tests for Substance class using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SUBSTANCES_AVAILABLE:
            self.skipTest("Substances module not available")
    
    def test_substance_creation(self):
        """Test basic substance creation."""
        substance = Substance(
            name="water",
            formula="H2O",
            molecular_weight=18.015,
            melting_point=273.15,
            boiling_point=373.15
        )
        
        self.assertEqual(substance.name, "water")
        self.assertEqual(substance.formula, "H2O")
        self.assertEqual(substance.molecular_weight, 18.015)
        self.assertEqual(substance.melting_point, 273.15)
        self.assertEqual(substance.boiling_point, 373.15)
    
    def test_substance_from_dict(self):
        """Test creating substance from dictionary."""
        data = {
            "name": "ethanol",
            "formula": "C2H6O",
            "molecular_weight": 46.068,
            "melting_point": 159.05,
            "boiling_point": 351.44
        }
        
        substance = Substance.from_dict(data)
        
        self.assertEqual(substance.name, "ethanol")
        self.assertEqual(substance.formula, "C2H6O")
        self.assertEqual(substance.molecular_weight, 46.068)
    
    def test_heat_capacity_constant(self):
        """Test heat capacity with constant value."""
        substance = Substance(
            name="test",
            cp_coefficients={
                "type": "constant",
                "value": 4184
            }
        )
        
        cp = substance.heat_capacity(298.15)
        self.assertEqual(cp, 4184)
    
    def test_heat_capacity_polynomial(self):
        """Test heat capacity with polynomial coefficients."""
        substance = Substance(
            name="test",
            cp_coefficients={
                "type": "polynomial",
                "coefficients": [1000, 2, 0.001]
            }
        )
        
        T = 300.0
        expected_cp = 1000 + 2*T + 0.001*T**2
        cp = substance.heat_capacity(T)
        self.assertAlmostEqual(cp, expected_cp, places=6)
    
    def test_heat_capacity_no_data(self):
        """Test heat capacity when no data available."""
        substance = Substance(name="test")
        
        with self.assertRaises(ValueError):
            substance.heat_capacity(298.15)
    
    def test_density_constant(self):
        """Test density with constant value."""
        substance = Substance(
            name="test",
            density_coefficients={
                "type": "constant",
                "value": 997.0
            }
        )
        
        density = substance.density(298.15)
        self.assertEqual(density, 997.0)
    
    def test_vapor_pressure_antoine(self):
        """Test vapor pressure with Antoine equation."""
        substance = Substance(
            name="test",
            vapor_pressure_coefficients={
                "type": "antoine",
                "A": 8.20417,
                "B": 1642.89,
                "C": -42.85
            }
        )
        
        T = 298.15
        expected_log_p = 8.20417 - 1642.89/(-42.85 + T)
        expected_p = 10**expected_log_p
        
        p = substance.vapor_pressure(T)
        self.assertAlmostEqual(p, expected_p, places=6)
    
    def test_to_dict(self):
        """Test converting substance to dictionary."""
        substance = Substance(
            name="methane",
            formula="CH4",
            molecular_weight=16.043
        )
        
        data = substance.to_dict()
        
        self.assertEqual(data["name"], "methane")
        self.assertEqual(data["formula"], "CH4")
        self.assertEqual(data["molecular_weight"], 16.043)
        self.assertIn("reference_temperature", data)
        self.assertIn("temp_range", data)


class TestSubstanceDatabaseBasic(unittest.TestCase):
    """Basic tests for SubstanceDatabase class using unittest."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SUBSTANCES_AVAILABLE:
            self.skipTest("Substances module not available")
        
        self.db = SubstanceDatabase()
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertEqual(len(self.db.substances), 0)
        self.assertEqual(self.db.list_substances(), [])
    
    def test_add_and_get_substance(self):
        """Test adding and retrieving substances."""
        water = Substance(name="Water", formula="H2O", molecular_weight=18.015)
        self.db.add_substance(water)
        
        self.assertEqual(len(self.db.substances), 1)
        self.assertIn("water", self.db.substances)
        
        retrieved = self.db.get_substance("water")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Water")
        
        # Test case insensitive retrieval
        retrieved2 = self.db.get_substance("WATER")
        self.assertIsNotNone(retrieved2)
        self.assertEqual(retrieved2.name, "Water")
    
    def test_list_substances(self):
        """Test listing all substances."""
        water = Substance(name="Water", formula="H2O")
        ethanol = Substance(name="Ethanol", formula="C2H6O")
        
        self.db.add_substance(water)
        self.db.add_substance(ethanol)
        
        substances = self.db.list_substances()
        self.assertEqual(len(substances), 2)
        self.assertIn("water", substances)
        self.assertIn("ethanol", substances)
    
    def test_search_by_formula(self):
        """Test searching by chemical formula."""
        water = Substance(name="Water", formula="H2O")
        ethanol = Substance(name="Ethanol", formula="C2H6O")
        
        self.db.add_substance(water)
        self.db.add_substance(ethanol)
        
        h2o_substances = self.db.search_by_formula("H2O")
        self.assertEqual(len(h2o_substances), 1)
        self.assertEqual(h2o_substances[0].name, "Water")
        
        results = self.db.search_by_formula("XYZ")
        self.assertEqual(len(results), 0)
    
    def test_search_by_cas(self):
        """Test searching by CAS number."""
        water = Substance(name="Water", cas_number="7732-18-5")
        ethanol = Substance(name="Ethanol", cas_number="64-17-5")
        
        self.db.add_substance(water)
        self.db.add_substance(ethanol)
        
        found = self.db.search_by_cas("7732-18-5")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Water")
        
        not_found = self.db.search_by_cas("12345-67-8")
        self.assertIsNone(not_found)


def run_tests():
    """Run all tests and return results."""
    print("=" * 60)
    print("Running ThermSolve Substance Tests")
    print("=" * 60)
    
    # Capture test output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubstanceBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestSubstanceDatabaseBasic))
    
    # Run tests
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll tests passed! ✅")
        return True
    else:
        print("\nSome tests failed! ❌")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
