# ThermSolve

An extensible, lightweight thermophysical property library for chemical and process engineering.

Thermsolve makes it easy to retrieve temperature-dependent properties of common fluids, gases, and materials — with unit safety, validity ranges, and custom substance support.

---

## Features

- Look up Cp, viscosity, density, boiling/melting point, enthalpy, entropy, etc.
- Temperature-dependent property fitting/interpolation
- Fully unit-aware (using Pint)
- Built-in and user-defined property database (JSON/CSV)
- Range warnings when extrapolating properties
- CLI or API access to properties
- Compatible with ReactorPy and simulation libraries

---

## To-do
- [ ] Core `Substance` class for property access
- [ ] Build initial CSV/JSON database for ~20 common substances
- [ ] Add temperature-dependent interpolation (Cp, viscosity, etc.)
- [ ] Integrate Pint for unit-aware values
- [ ] Implement property validity range checking
- [ ] Enable user-defined substances from JSON, CSV, or dict
- [ ] Add CLI tool for property lookup
- [ ] Create optional SQLite backend for scalable storage
- [ ] Add plotting functions for property trends (e.g., Cp vs T)
- [ ] Include citation metadata for each property source (e.g., NIST, DIPPR)
- [ ] Export properties to JSON, LaTeX, and human-readable tables

## Example

```python
from thermoprops import Substance

ethanol = Substance("ethanol")
cp = ethanol.cp(T=300)       # Specific heat [J/kg-K]
mu = ethanol.viscosity(T=350)
bp = ethanol.boiling_point()

# Add a custom substance
my_material = Substance.from_dict({
    "name": "KCl",
    "cp": {"type": "constant", "value": 800, "units": "J/kg-K"},
    "melting_point": 1040,
    "density": 1980
})
