#!/usr/bin/env python3
"""
Quick test script to verify the Substance class is working properly.
This script can be run without installing pytest.
"""

import sys
import os
import traceback

# Add the thermsolve package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

def test_basic_functionality():
    """Test basic functionality of the Substance class."""
    print("Testing basic Substance functionality...")
    
    try:
        from substances import Substance, SubstanceDatabase
        print(" Successfully imported Substance and SubstanceDatabase")
    except ImportError as e:
        print(f" Failed to import substances module: {e}")
        return False
    
    try:
        # Test 1: Create a basic substance
        print("\n1. Testing basic substance creation...")
        water = Substance(
            name="water",
            formula="H2O",
            molecular_weight=18.015,
            melting_point=273.15,
            boiling_point=373.15
        )
        assert water.name == "water"
        assert water.formula == "H2O"
        assert water.molecular_weight == 18.015
        print(" Basic substance creation works")
        
        # Test 2: Test from_dict creation
        print("\n2. Testing substance creation from dictionary...")
        ethanol_data = {
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
        ethanol = Substance.from_dict(ethanol_data)
        assert ethanol.name == "ethanol"
        assert ethanol.cp_coefficients["value"] == 2440
        print(" Substance creation from dictionary works")
        
        # Test 3: Test heat capacity calculation
        print("\n3. Testing heat capacity calculations...")
        cp = ethanol.heat_capacity(298.15)
        assert cp == 2440
        print(f" Heat capacity calculation works: {cp} J/(mol·K)")
        
        # Test 4: Test polynomial heat capacity
        print("\n4. Testing polynomial heat capacity...")
        poly_substance = Substance(
            name="test",
            cp_coefficients={
                "type": "polynomial",
                "coefficients": [1000, 2, 0.001]  # Cp = 1000 + 2*T + 0.001*T^2
            }
        )
        T = 300.0
        cp_poly = poly_substance.heat_capacity(T)
        expected = 1000 + 2*T + 0.001*T**2
        assert abs(cp_poly - expected) < 1e-6
        print(f" Polynomial heat capacity works: {cp_poly} J/(mol·K)")
        
        # Test 5: Test vapor pressure calculation
        print("\n5. Testing vapor pressure calculations...")
        vp_substance = Substance(
            name="test",
            vapor_pressure_coefficients={
                "type": "antoine",
                "A": 8.20417,
                "B": 1642.89,
                "C": -42.85
            }
        )
        vp = vp_substance.vapor_pressure(298.15)
        assert vp is not None and vp > 0
        print(f" Vapor pressure calculation works: {vp:.2f} Pa")
        
        # Test 6: Test database operations
        print("\n6. Testing database operations...")
        db = SubstanceDatabase()
        db.add_substance(water)
        db.add_substance(ethanol)
        
        assert len(db.substances) == 2
        retrieved_water = db.get_substance("water")
        assert retrieved_water is not None
        assert retrieved_water.name == "water"
        print(" Database operations work")
        
        # Test 7: Test serialization
        print("\n7. Testing serialization...")
        data_dict = water.to_dict()
        assert data_dict["name"] == "water"
        assert data_dict["formula"] == "H2O"
        
        water_from_dict = Substance.from_dict(data_dict)
        assert water_from_dict.name == water.name
        assert water_from_dict.formula == water.formula
        print(" Serialization works")
        
        # Test 8: Test error handling
        print("\n8. Testing error handling...")
        try:
            substance_no_cp = Substance(name="test")
            substance_no_cp.heat_capacity(298.15)
            print(" Should have raised ValueError")
            return False
        except ValueError:
            print(" Error handling works correctly")
        
        print("\n" + "="*50)
        print("🎉 All basic functionality tests PASSED!")
        print("="*50)
        return True
        
    except Exception as e:
        print(f" Test failed with error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_csv_data():
    """Test loading the CSV data file."""
    print("\nTesting CSV data loading...")
    
    try:
        from substances import SubstanceDatabase
        
        # Check if CSV file exists
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'thermsolve', 'data', 'substance_list.csv')
        if not os.path.exists(csv_path):
            print(f" CSV file not found at: {csv_path}")
            return False
        
        # Try to load CSV
        db = SubstanceDatabase()
        db.load_from_csv(csv_path)
        
        if len(db.substances) == 0:
            print(" No substances loaded from CSV")
            return False
        
        print(f" Successfully loaded {len(db.substances)} substances from CSV")
        print("Available substances:", ", ".join(db.list_substances()))
        
        # Test accessing a substance
        water = db.get_substance("water")
        if water:
            print(f" Found water: {water.formula}, MW: {water.molecular_weight}")
        else:
            print("Could not find water in database")
            return False
        
        return True
        
    except Exception as e:
        print(f" CSV loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print(" ThermSolve Substance Class Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run basic functionality tests
    if not test_basic_functionality():
        success = False
    
    # Run CSV data tests
    if not test_csv_data():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print(" All tests pasted, Your Substance class is working correctly.")
    else:
        print(" Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
