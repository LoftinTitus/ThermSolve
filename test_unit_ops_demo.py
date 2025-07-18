"""
Test and demonstration script for unit operations classes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermsolve.unit_operations import (
    HeatExchanger, Reactor, DistillationColumn, Pump, Compressor,
    calculate_reynolds_number, calculate_friction_factor, calculate_pressure_drop_pipe
)
from thermsolve.substances import Substance
import numpy as np

def test_heat_exchanger():
    """Test heat exchanger calculations."""
    print("=== Heat Exchanger Example ===")
    
    # Create a shell-and-tube heat exchanger
    hx = HeatExchanger(name="Oil Cooler", heat_exchanger_type="shell_and_tube")
    hx.flow_arrangement = "counter_current"
    
    # Define fluids (using simplified substances)
    hot_oil = Substance(name="Hot Oil")
    cooling_water = Substance(name="Cooling Water")
    
    # Set fluid conditions
    hx.set_hot_fluid(hot_oil, flow_rate=2.5, flow_rate_unit='kg/s',
                     inlet_temp=150, temp_unit='degC')
    
    hx.set_cold_fluid(cooling_water, flow_rate=5.0, flow_rate_unit='kg/s',
                      inlet_temp=25, temp_unit='degC')
    
    # Calculate LMTD
    hot_outlet = 80  # °C
    cold_outlet = 45  # °C
    
    lmtd = hx.calculate_lmtd(hot_outlet, cold_outlet, 'degC')
    print(f"LMTD: {lmtd:.2f} K")
    
    # Calculate heat duty
    heat_duty = hx.calculate_heat_duty(hot_outlet, cold_outlet, 'degC')
    print(f"Heat duty (hot side): {heat_duty['heat_duty_hot']/1000:.1f} kW")
    print(f"Heat duty (cold side): {heat_duty['heat_duty_cold']/1000:.1f} kW")
    print(f"Energy balance error: {heat_duty['energy_balance_error_percent']:.2f}%")
    
    # Calculate required area
    U = 500  # W/(m²·K)
    area = hx.calculate_area(U, 'W/(m^2*K)', hot_outlet, cold_outlet, 'degC')
    print(f"Required heat transfer area: {area:.2f} m²")

def test_pump():
    """Test pump calculations."""
    print("\n=== Pump Example ===")
    
    # Create a centrifugal pump
    pump = Pump(name="Feed Pump")
    
    # Set fluid properties (water)
    pump.set_fluid_properties(density=1000, density_unit='kg/m^3',
                             viscosity=0.001, viscosity_unit='Pa*s')
    
    # Set design parameters
    pump.set_parameter('flow_rate', 0.05, 'm^3/s')  # 50 L/s
    pump.set_parameter('head', 50, 'm')  # 50 m head
    pump.set_parameter('efficiency', 0.75, 'dimensionless')
    
    # Calculate pump power
    flow_rate = pump.get_parameter('flow_rate')
    head = pump.get_parameter('head')
    efficiency = pump.get_parameter('efficiency')
    
    power_results = pump.calculate_pump_power(flow_rate, 'm^3/s', head, 'm', efficiency)
    
    print(f"Flow rate: {flow_rate*1000:.1f} L/s")
    print(f"Head: {head:.1f} m")
    print(f"Efficiency: {efficiency*100:.1f}%")
    print(f"Hydraulic power: {power_results['hydraulic_power_kw']:.2f} kW" 
          if 'hydraulic_power_kw' in power_results 
          else f"Hydraulic power: {power_results['hydraulic_power_w']/1000:.2f} kW")
    print(f"Brake power: {power_results['brake_power_hp']:.2f} hp")
    
    # Calculate NPSH required
    npsh_r = pump.calculate_npsh_required(flow_rate, 'm^3/s')
    print(f"NPSH required: {npsh_r:.2f} m")

def test_compressor():
    """Test compressor calculations."""
    print("\n=== Compressor Example ===")
    
    # Create a centrifugal compressor
    compressor = Compressor(name="Air Compressor", compressor_type="centrifugal")
    
    # Set gas properties (air)
    compressor.set_gas_properties(molecular_weight=29, k_ratio=1.4, z_factor=1.0)
    
    # Calculate compression power
    power_results = compressor.calculate_adiabatic_power(
        inlet_pressure=1, inlet_pressure_unit='bar',
        outlet_pressure=5, outlet_pressure_unit='bar',
        flow_rate=2.0, flow_unit='kg/s',
        inlet_temperature=25, temp_unit='degC',
        efficiency=0.8
    )
    
    print(f"Compression ratio: {power_results['compression_ratio']:.2f}")
    print(f"Outlet temperature (ideal): {power_results['outlet_temperature_ideal_k']-273.15:.1f}°C")
    print(f"Ideal power: {power_results['ideal_power_w']/1000:.1f} kW")
    print(f"Actual power: {power_results['actual_power_hp']:.1f} hp")

def test_distillation():
    """Test distillation column calculations."""
    print("\n=== Distillation Column Example ===")
    
    # Create a binary distillation column
    column = DistillationColumn(name="Benzene-Toluene Column")
    
    # Add components
    benzene = Substance(name="Benzene")
    toluene = Substance(name="Toluene")
    
    column.add_component("Benzene", benzene, feed_mole_fraction=0.5)
    column.add_component("Toluene", toluene, feed_mole_fraction=0.5)
    
    # Set feed conditions
    column.set_feed_conditions(flow_rate=100, flow_unit='mol/s',
                              temperature=80, temp_unit='degC',
                              pressure=1, pressure_unit='bar')
    
    # Calculate minimum reflux and stages
    alpha = 2.5  # Relative volatility
    xD = 0.95   # Distillate purity
    xB = 0.95   # Bottoms purity (of heavy component)
    
    Rm = column.calculate_minimum_reflux_binary(alpha, xD)
    Nm = column.calculate_minimum_stages_binary(alpha, xD, xB)
    
    print(f"Minimum reflux ratio: {Rm:.2f}")
    print(f"Minimum number of stages: {Nm:.1f}")
    print(f"Actual reflux ratio (1.5 × minimum): {1.5*Rm:.2f}")

def test_reactor():
    """Test reactor calculations."""
    print("\n=== Reactor Example ===")
    
    # Create a CSTR
    reactor = Reactor(name="CSTR", reactor_type="cstr")
    
    # Add components
    reactant_a = Substance(name="Reactant A")
    product_b = Substance(name="Product B")
    
    reactor.add_component("A", reactant_a, inlet_flow=1.0, flow_unit='kg/s')
    reactor.add_component("B", product_b, inlet_flow=0.0, flow_unit='kg/s')
    
    # Add reaction: A → B
    reactor.add_reaction(
        reaction_rate_constant=0.1,  # 1/s
        activation_energy=50000,     # J/mol
        stoichiometry={"A": -1, "B": 1},
        reaction_order={"A": 1}
    )
    
    # Calculate reaction rate at operating conditions
    concentrations = {"A": 2.0, "B": 0.5}  # mol/L
    temperature = 350  # K
    
    rates = reactor.calculate_reaction_rate(concentrations, temperature)
    print(f"Reaction rate at {temperature} K: {rates[0]:.4f} mol/(L·s)")
    
    # Perform reactor calculation
    results = reactor.calculate()
    print(f"Reactor type: {results['reactor_type']}")
    print(f"Number of reactions: {results['number_of_reactions']}")

def test_fluid_mechanics():
    """Test fluid mechanics calculations."""
    print("\n=== Fluid Mechanics Example ===")
    
    # Pipe flow calculation
    diameter = 0.1  # m
    velocity = 2.0  # m/s
    density = 1000  # kg/m³
    viscosity = 0.001  # Pa·s
    
    # Calculate Reynolds number
    Re = calculate_reynolds_number(velocity, diameter, density, viscosity)
    print(f"Reynolds number: {Re:.0f}")
    
    # Determine flow regime
    if Re < 2300:
        flow_regime = "Laminar"
    elif Re < 4000:
        flow_regime = "Transitional"
    else:
        flow_regime = "Turbulent"
    
    print(f"Flow regime: {flow_regime}")
    
    # Calculate friction factor
    roughness = 0.000045  # m (commercial steel)
    relative_roughness = roughness / diameter
    f = calculate_friction_factor(Re, relative_roughness)
    print(f"Friction factor: {f:.4f}")
    
    # Calculate pressure drop
    length = 100  # m
    flow_rate = density * velocity * (3.14159 * diameter**2 / 4)  # kg/s
    
    delta_p = calculate_pressure_drop_pipe(flow_rate, diameter, length, 
                                         density, viscosity, roughness)
    
    print(f"Pressure drop over {length} m: {delta_p/1000:.2f} kPa")
    print(f"Pressure drop per 100 m: {delta_p/length*100/1000:.2f} kPa/100m")

def test_integrated_example():
    """Test an integrated process example."""
    print("\n=== Integrated Process Example ===")
    print("Process: Pump → Heat Exchanger → Reactor")
    
    # Step 1: Pump to increase pressure
    feed_pump = Pump(name="Feed Pump")
    feed_pump.set_fluid_properties(density=1000, density_unit='kg/m^3',
                                  viscosity=0.001, viscosity_unit='Pa*s')
    
    pump_power = feed_pump.calculate_pump_power(
        flow_rate=0.02, flow_unit='m^3/s',
        head=30, head_unit='m',
        efficiency=0.75
    )
    
    print(f"1. Pump power required: {pump_power['brake_power_w']/1000:.1f} kW")
    
    # Step 2: Heat exchanger to heat feed
    preheater = HeatExchanger(name="Feed Preheater")
    
    process_fluid = Substance(name="Process Fluid")
    steam = Substance(name="Steam")
    
    preheater.set_cold_fluid(process_fluid, flow_rate=20, flow_rate_unit='kg/s',
                            inlet_temp=25, temp_unit='degC')
    
    preheater.set_hot_fluid(steam, flow_rate=2, flow_rate_unit='kg/s',
                           inlet_temp=150, temp_unit='degC')
    
    heat_duty = preheater.calculate_heat_duty(120, 80, 'degC')  # Steam out, process out
    
    print(f"2. Heat exchanger duty: {heat_duty['average_heat_duty']/1000:.1f} kW")
    
    # Step 3: Reactor
    main_reactor = Reactor(name="Main Reactor", reactor_type="cstr")
    
    print(f"3. Reactor feed temperature: 80°C")
    print(f"   Process integration complete!")

if __name__ == "__main__":
    print("Unit Operations Classes Demonstration")
    print("=" * 60)
    
    test_heat_exchanger()
    test_pump()
    test_compressor()
    test_distillation()
    test_reactor()
    test_fluid_mechanics()
    test_integrated_example()
    
    print("\n" + "=" * 60)
    print("All unit operations tests completed successfully!")
