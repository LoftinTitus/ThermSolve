#!/usr/bin/env python3
"""
Example usage of the ThermSolve substance database.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'thermsolve'))

from substances import Substance, SubstanceDatabase

def main():
    print("=== ThermSolve Substance Example ===\n")
    
    # Create a substance database
    db = SubstanceDatabase()
    
    # Load substances from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'thermsolve', 'data', 'substance_list.csv')
    if os.path.exists(csv_path):
        db.load_from_csv(csv_path)
        print(f"Loaded {len(db.substances)} substances from database")
        print("Available substances:", ", ".join(db.list_substances()))
    else:
        print("CSV file not found, creating substances manually...")
    
    # Example 1: Create a substance manually
    print("\n=== Example 1: Manual Substance Creation ===")
    water = Substance(
        name="water",
        formula="H2O",
        molecular_weight=18.015,
        melting_point=273.15,
        boiling_point=373.15,
        cp_coefficients={
            "type": "constant",
            "value": 4184  # J/(kg·K)
        },
        density_coefficients={
            "type": "constant", 
            "value": 997.0  # kg/m³ at 20°C
        }
    )
    
    print(f"Created substance: {water}")
    print(f"Molecular weight: {water.molecular_weight} g/mol")
    print(f"Boiling point: {water.boiling_point} K")
    
    # Example 2: Create substance from dictionary
    print("\n=== Example 2: Substance from Dictionary ===")
    ethanol_data = {
        "name": "ethanol",
        "formula": "C2H6O", 
        "molecular_weight": 46.068,
        "melting_point": 159.05,
        "boiling_point": 351.44,
        "cp_coefficients": {
            "type": "polynomial",
            "coefficients": [2440, 0.5, -0.001]  # Example: Cp = a + b*T + c*T^2
        },
        "vapor_pressure_coefficients": {
            "type": "antoine",
            "A": 8.20417,
            "B": 1642.89,
            "C": -42.85
        },
        "temp_range": [159, 400],
        "data_sources": {"cp": "NIST", "vapor_pressure": "DIPPR"}
    }
    
    ethanol = Substance.from_dict(ethanol_data)
    print(f"Created substance: {ethanol}")
    
    # Example 3: Property calculations
    print("\n=== Example 3: Property Calculations ===")
    T = 298.15  # 25°C
    
    try:
        cp = ethanol.heat_capacity(T)
        print(f"Heat capacity of ethanol at {T} K: {cp} J/(mol·K)")
    except Exception as e:
        print(f"Heat capacity calculation failed: {e}")
    
    try:
        vp = ethanol.vapor_pressure(T)
        print(f"Vapor pressure of ethanol at {T} K: {vp} Pa")
    except Exception as e:
        print(f"Vapor pressure calculation failed: {e}")
    
    # Example 4: Save and load from JSON
    print("\n=== Example 4: JSON Save/Load ===")
    json_path = "ethanol_example.json"
    ethanol.to_json(json_path)
    print(f"Saved ethanol data to {json_path}")
    
    # Load it back
    ethanol_loaded = Substance.from_json(json_path)
    print(f"Loaded substance: {ethanol_loaded}")
    
    # Example 5: Database operations
    print("\n=== Example 5: Database Operations ===")
    db.add_substance(water)
    db.add_substance(ethanol)
    
    # Search by name
    found_water = db.get_substance("water")
    if found_water:
        print(f"Found water: {found_water}")
    
    # List all substances
    print(f"Database contains: {db.list_substances()}")
    
    # Clean up
    if os.path.exists(json_path):
        os.remove(json_path)
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
