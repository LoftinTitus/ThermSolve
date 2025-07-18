"""
Main script for ThermSolve package demo.
"""
from thermsolve.substances import Substance, SubstanceDatabase
from thermsolve.units import ureg

# Load substances from CSV database
sub_db = SubstanceDatabase()
sub_db.load_from_csv("thermsolve/data/substance_list.csv")

# Get water from database
water = sub_db.get_substance("water")
T = 298.15 * ureg.kelvin
T_val = T.magnitude  # Kelvin as float

print(f"Water at {T}:")
print(f"  Cp: {(water.heat_capacity(T_val) * ureg('J/kg/K')):~P}")
print(f"  Viscosity: {(water.viscosity(T_val) * ureg('Pa*s')):~P}")
print(f"  Density: {(water.density(T_val) * ureg('kg/m^3')):~P}")
print(f"  Boiling point: {(water.boiling_point * ureg.kelvin):~P}")
# Example: Add a custom substance
kcl = Substance.from_dict({
    "name": "KCl",
    "cp_coefficients": {"type": "constant", "value": 800, "units": "J/kg/K"},
    "melting_point": 1040 * ureg.kelvin,
    "density_coefficients": {"type": "constant", "value": 1980, "units": "kg/m^3"}
})

print("\nCustom substance (KCl):")
print(f"  Cp: {(kcl.heat_capacity(T_val) * ureg('J/kg/K')):~P}")
print(f"  Melting point: {kcl.melting_point:~P}")
print(f"  Density: {(kcl.density(T_val) * ureg('kg/m^3')):~P}")

# Range warning demo (if implemented)
try:
    print(f"Water Cp at 600 K: {(water.heat_capacity(600) * ureg('J/kg/K')):~P}")
except Exception as e:
    print(f"Warning: {e}")

# CLI entry point (optional)
if __name__ == "__main__":
    print("\nThermSolve main script executed.")
