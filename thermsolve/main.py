"""
Main script for ThermSolve package demo.
"""
from thermsolve.substances import Substance
from thermsolve.units import ureg

# Example: Load a built-in substance
water = Substance("water")
T = 298.15 * ureg.kelvin
T_val = T.magnitude  # Kelvin as float

print(f"Water at {T}:")
print(f"  Cp: {water.heat_capacity(T_val):~P}")
print(f"  Viscosity: {water.viscosity(T_val):~P}")
print(f"  Density: {water.density(T_val):~P}")
print(f"  Boiling point: {water.boiling_point():~P}")

# Example: Add a custom substance
kcl = Substance.from_dict({
    "name": "KCl",
    "cp": {"type": "constant", "value": 800, "units": "J/kg/K"},
    "melting_point": 1040 * ureg.kelvin,
    "density": 1980 * ureg.kg / ureg.meter**3
})

print("\nCustom substance (KCl):")
print(f"  Cp: {kcl.cp(T_val):~P}")
print(f"  Melting point: {kcl.melting_point:~P}")
print(f"  Density: {kcl.density:~P}")

# Range warning demo (if implemented)
try:
    print(f"Water Cp at 600 K: {water.cp(600):~P}")
except Exception as e:
    print(f"Warning: {e}")

# CLI entry point (optional)
if __name__ == "__main__":
    print("\nThermSolve main script executed.")
