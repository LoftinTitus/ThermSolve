import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Demo for ThermPlotter class in ThermSolve.
"""
import numpy as np
from thermsolve.plotting import ThermPlotter

# Example data: Heat capacity of water
temperatures = np.array([273.15, 298.15, 323.15, 373.15, 423.15, 473.15])
cp_values = np.array([4217, 4184, 4179, 4217, 4312, 4459])

# Example data: Viscosity of ethanol
temps_visc = np.array([273.15, 298.15, 323.15, 348.15, 373.15])
visc_values = np.array([1.773e-3, 1.074e-3, 0.694e-3, 0.476e-3, 0.346e-3])

plotter = ThermPlotter(style='seaborn-v0_8')

# Property vs temperature plot
plotter.plot_property_vs_temperature(
    temperatures, cp_values,
    property_name="Heat Capacity",
    property_units="J/(kg·K)",
    substance_name="Water",
    data_points=(temperatures, cp_values),
    validity_range=(273.15, 473.15),
    show_plot=True
)

# Compare substances
substance_data = {
    "Water": {"temperatures": temperatures, "property_values": cp_values},
    "Ethanol": {"temperatures": temps_visc, "property_values": visc_values}
}
plotter.compare_substances(
    substance_data,
    property_name="Property",
    property_units="(units)",
    show_plot=True
)

# Multiple properties
properties_data = {
    "Heat Capacity": {"values": cp_values, "units": "J/(kg·K)"},
    "Viscosity": {"values": np.interp(temperatures, temps_visc, visc_values), "units": "Pa·s"}
}
plotter.plot_multiple_properties(
    temperatures,
    properties_data,
    substance_name="Water",
    show_plot=True
)
