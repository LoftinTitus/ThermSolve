import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Basic tests for ThermPlotter class in ThermSolve.
"""
import numpy as np
import pytest
from thermsolve.plotting import ThermPlotter

@pytest.fixture
def example_data():
    temperatures = np.array([273.15, 298.15, 323.15, 373.15, 423.15, 473.15])
    cp_values = np.array([4217, 4184, 4179, 4217, 4312, 4459])
    return temperatures, cp_values

def test_plot_property_vs_temperature(example_data):
    temperatures, cp_values = example_data
    plotter = ThermPlotter()
    fig = plotter.plot_property_vs_temperature(
        temperatures, cp_values,
        property_name="Heat Capacity",
        property_units="J/(kg·K)",
        show_plot=False
    )
    assert fig is not None

def test_compare_substances(example_data):
    temperatures, cp_values = example_data
    plotter = ThermPlotter()
    substance_data = {
        "Water": {"temperatures": temperatures, "property_values": cp_values},
        "Test": {"temperatures": temperatures, "property_values": cp_values + 100}
    }
    fig = plotter.compare_substances(
        substance_data,
        property_name="Heat Capacity",
        property_units="J/(kg·K)",
        show_plot=False
    )
    assert fig is not None

def test_plot_multiple_properties(example_data):
    temperatures, cp_values = example_data
    plotter = ThermPlotter()
    properties_data = {
        "Heat Capacity": {"values": cp_values, "units": "J/(kg·K)"},
        "Fake Property": {"values": cp_values * 2, "units": "unit"}
    }
    fig = plotter.plot_multiple_properties(
        temperatures,
        properties_data,
        substance_name="Water",
        show_plot=False
    )
    assert fig is not None
