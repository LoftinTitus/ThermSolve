"""
Unit operations classes for chemical and process engineering calculations.

This module provides classes for common unit operations including heat exchangers,
reactors, distillation columns, pumps, compressors, and other process equipment
with integrated unit handling and thermodynamic property calculations.
"""

import numpy as np
import math
from typing import Optional, Dict, Any, Union, Tuple, List
from abc import ABC, abstractmethod

from .units import ThermodynamicUnits
from .substances import Substance


class UnitOperation(ABC):
    """
    Abstract base class for all unit operations.
    
    Provides common functionality for unit handling, validation,
    and basic thermodynamic calculations.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.units = ThermodynamicUnits()
        self.design_parameters = {}
        self.operating_conditions = {}
        self.results = {}
    
    @abstractmethod
    def calculate(self) -> Dict[str, Any]:
        """Perform the unit operation calculations."""
        pass
    
    def set_parameter(self, parameter: str, value: float, unit: str):
        """Set a design parameter with units."""
        self.design_parameters[parameter] = {
            'value': value,
            'unit': unit,
            'si_value': self.units.normalize_to_standard(value, unit, self._get_property_type(parameter))
        }
    
    def get_parameter(self, parameter: str, target_unit: str = None) -> float:
        """Get a parameter value, optionally in specified units."""
        if parameter not in self.design_parameters:
            raise ValueError(f"Parameter '{parameter}' not set")
        
        param_data = self.design_parameters[parameter]
        if target_unit is None:
            return param_data['si_value']
        else:
            property_type = self._get_property_type(parameter)
            standard_unit = self.units.get_standard_unit(property_type)
            return self.units.convert(param_data['si_value'], standard_unit, target_unit)
    
    def _get_property_type(self, parameter: str) -> str:
        """Map parameter names to property types for unit validation."""
        property_map = {
            'temperature': 'temperature',
            'pressure': 'pressure', 
            'flow_rate': 'volumetric_flow_rate',  # Changed to volumetric for pumps
            'mass_flow_rate': 'mass_flow_rate',
            'volumetric_flow_rate': 'volumetric_flow_rate',
            'heat_duty': 'power',
            'area': 'area',
            'diameter': 'length',
            'length': 'length',
            'volume': 'volume',
            'u_overall': 'heat_transfer_coefficient',
            'delta_p': 'pressure',
            'efficiency': 'dimensionless',
            'head': 'length'
        }
        
        for key, prop_type in property_map.items():
            if key in parameter.lower():
                return prop_type
        
        return 'dimensionless'  # Default


class HeatExchanger(UnitOperation):
    """
    Heat exchanger unit operation class.
    
    Supports shell-and-tube, plate, and other heat exchanger types
    with LMTD, NTU-effectiveness, and other calculation methods.
    """
    
    def __init__(self, name: str = None, heat_exchanger_type: str = "shell_and_tube"):
        super().__init__(name)
        self.type = heat_exchanger_type
        self.hot_fluid = None
        self.cold_fluid = None
        self.flow_arrangement = "counter_current"  # counter_current, co_current, cross_flow
    
    def set_hot_fluid(self, fluid: Substance, flow_rate: float, flow_rate_unit: str,
                     inlet_temp: float, temp_unit: str, pressure: float = None, pressure_unit: str = None):
        """Set hot fluid properties."""
        self.hot_fluid = {
            'substance': fluid,
            'flow_rate': self.units.normalize_to_standard(flow_rate, flow_rate_unit, 'mass_flow_rate'),
            'inlet_temp': self.units.normalize_to_standard(inlet_temp, temp_unit, 'temperature'),
            'pressure': self.units.normalize_to_standard(pressure, pressure_unit, 'pressure') if pressure else None
        }
    
    def set_cold_fluid(self, fluid: Substance, flow_rate: float, flow_rate_unit: str,
                      inlet_temp: float, temp_unit: str, pressure: float = None, pressure_unit: str = None):
        """Set cold fluid properties."""
        self.cold_fluid = {
            'substance': fluid,
            'flow_rate': self.units.normalize_to_standard(flow_rate, flow_rate_unit, 'mass_flow_rate'),
            'inlet_temp': self.units.normalize_to_standard(inlet_temp, temp_unit, 'temperature'),
            'pressure': self.units.normalize_to_standard(pressure, pressure_unit, 'pressure') if pressure else None
        }
    
    def calculate_lmtd(self, hot_outlet_temp: float, cold_outlet_temp: float, temp_unit: str = 'K') -> float:
        """Calculate Log Mean Temperature Difference."""
        if not self.hot_fluid or not self.cold_fluid:
            raise ValueError("Both hot and cold fluids must be defined")
        
        # Convert temperatures to Kelvin
        T_h_in = self.hot_fluid['inlet_temp']
        T_c_in = self.cold_fluid['inlet_temp']
        T_h_out = self.units.normalize_to_standard(hot_outlet_temp, temp_unit, 'temperature')
        T_c_out = self.units.normalize_to_standard(cold_outlet_temp, temp_unit, 'temperature')
        
        # Calculate temperature differences
        if self.flow_arrangement == "counter_current":
            delta_T1 = T_h_in - T_c_out
            delta_T2 = T_h_out - T_c_in
        else:  # co_current
            delta_T1 = T_h_in - T_c_in
            delta_T2 = T_h_out - T_c_out
        
        if delta_T1 <= 0 or delta_T2 <= 0:
            raise ValueError("Invalid temperature differences - check flow arrangement and temperatures")
        
        if abs(delta_T1 - delta_T2) < 1e-6:
            lmtd = delta_T1
        else:
            lmtd = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
        
        return lmtd
    
    def calculate_heat_duty(self, hot_outlet_temp: float, cold_outlet_temp: float, 
                           temp_unit: str = 'K') -> Dict[str, float]:
        """Calculate heat duty and validate energy balance."""
        if not self.hot_fluid or not self.cold_fluid:
            raise ValueError("Both hot and cold fluids must be defined")
        
        T_h_out = self.units.normalize_to_standard(hot_outlet_temp, temp_unit, 'temperature')
        T_c_out = self.units.normalize_to_standard(cold_outlet_temp, temp_unit, 'temperature')
        
        # Get average temperatures for property evaluation
        T_h_avg = (self.hot_fluid['inlet_temp'] + T_h_out) / 2
        T_c_avg = (self.cold_fluid['inlet_temp'] + T_c_out) / 2
        
        # Get heat capacities (assuming constant)
        cp_hot = 4184 if not hasattr(self.hot_fluid['substance'], 'cp') else self.hot_fluid['substance'].cp
        cp_cold = 4184 if not hasattr(self.cold_fluid['substance'], 'cp') else self.cold_fluid['substance'].cp
        
        # Calculate heat duties
        Q_hot = self.hot_fluid['flow_rate'] * cp_hot * (self.hot_fluid['inlet_temp'] - T_h_out)
        Q_cold = self.cold_fluid['flow_rate'] * cp_cold * (T_c_out - self.cold_fluid['inlet_temp'])
        
        # Energy balance check
        energy_balance_error = abs(Q_hot - Q_cold) / max(Q_hot, Q_cold) * 100
        
        return {
            'heat_duty_hot': Q_hot,
            'heat_duty_cold': Q_cold,
            'average_heat_duty': (Q_hot + Q_cold) / 2,
            'energy_balance_error_percent': energy_balance_error
        }
    
    def calculate_area(self, overall_heat_transfer_coefficient: float, u_unit: str,
                      hot_outlet_temp: float, cold_outlet_temp: float, temp_unit: str = 'K') -> float:
        """Calculate required heat transfer area."""
        U = self.units.normalize_to_standard(overall_heat_transfer_coefficient, u_unit, 'heat_transfer_coefficient')
        
        heat_duty_results = self.calculate_heat_duty(hot_outlet_temp, cold_outlet_temp, temp_unit)
        Q = heat_duty_results['average_heat_duty']
        
        lmtd = self.calculate_lmtd(hot_outlet_temp, cold_outlet_temp, temp_unit)
        
        area = Q / (U * lmtd)
        return area
    
    def calculate(self) -> Dict[str, Any]:
        """Perform complete heat exchanger calculation."""
        if 'u_overall' not in self.design_parameters:
            raise ValueError("Overall heat transfer coefficient must be specified")
        if 'area' not in self.design_parameters:
            raise ValueError("Heat transfer area must be specified")
        
        # This is a simplified calculation - in practice, you'd solve iteratively
        results = {
            'equipment_type': 'heat_exchanger',
            'heat_exchanger_type': self.type,
            'flow_arrangement': self.flow_arrangement
        }
        
        self.results = results
        return results


class Reactor(UnitOperation):
    """
    Chemical reactor unit operation class.
    
    Supports CSTR, PFR, batch reactors with reaction kinetics,
    heat effects, and mass balance calculations.
    """
    
    def __init__(self, name: str = None, reactor_type: str = "cstr"):
        super().__init__(name)
        self.type = reactor_type  # cstr, pfr, batch, semi_batch
        self.reactions = []
        self.components = {}
        
    def add_component(self, name: str, substance: Substance, inlet_flow: float = 0, 
                     flow_unit: str = 'kg/s'):
        """Add a component to the reactor."""
        self.components[name] = {
            'substance': substance,
            'inlet_flow': self.units.normalize_to_standard(inlet_flow, flow_unit, 'mass_flow_rate')
        }
    
    def add_reaction(self, reaction_rate_constant: float, activation_energy: float,
                    stoichiometry: Dict[str, float], reaction_order: Dict[str, float] = None):
        """Add a reaction to the reactor."""
        reaction = {
            'k0': reaction_rate_constant,
            'Ea': activation_energy,
            'stoichiometry': stoichiometry,
            'order': reaction_order or {comp: 1.0 for comp in stoichiometry.keys()}
        }
        self.reactions.append(reaction)
    
    def calculate_reaction_rate(self, concentrations: Dict[str, float], temperature: float) -> List[float]:
        """Calculate reaction rates at given conditions."""
        R = 8.314  # J/(mol·K)
        rates = []
        
        for reaction in self.reactions:
            k = reaction['k0'] * math.exp(-reaction['Ea'] / (R * temperature))
            
            rate = k
            for component, order in reaction['order'].items():
                if component in concentrations:
                    rate *= concentrations[component] ** order
            
            rates.append(rate)
        
        return rates
    
    def calculate_cstr_volume(self, conversion: float, temperature: float, 
                             temp_unit: str = 'K') -> float:
        """Calculate CSTR volume for given conversion."""
        T = self.units.normalize_to_standard(temperature, temp_unit, 'temperature')
        
        if not self.reactions:
            raise ValueError("No reactions defined")
        
        # Simplified calculation for single reaction
        # V = F₀X / (-rₐ)
        # This would need more sophisticated implementation for multiple reactions
        
        return 0.0  # Placeholder
    
    def calculate(self) -> Dict[str, Any]:
        """Perform reactor calculation."""
        results = {
            'equipment_type': 'reactor',
            'reactor_type': self.type,
            'number_of_reactions': len(self.reactions),
            'number_of_components': len(self.components)
        }
        
        self.results = results
        return results


class DistillationColumn(UnitOperation):
    """
    Distillation column unit operation class.
    
    Supports binary and multicomponent distillation with
    McCabe-Thiele, Fenske-Underwood-Gilliland methods.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.components = []
        self.feed_conditions = {}
        self.column_specs = {}
    
    def add_component(self, name: str, substance: Substance, feed_mole_fraction: float):
        """Add a component to the distillation system."""
        self.components.append({
            'name': name,
            'substance': substance,
            'feed_mole_fraction': feed_mole_fraction
        })
    
    def set_feed_conditions(self, flow_rate: float, flow_unit: str, temperature: float,
                           temp_unit: str, pressure: float, pressure_unit: str):
        """Set feed stream conditions."""
        self.feed_conditions = {
            'flow_rate': self.units.normalize_to_standard(flow_rate, flow_unit, 'molar_flow_rate'),
            'temperature': self.units.normalize_to_standard(temperature, temp_unit, 'temperature'),
            'pressure': self.units.normalize_to_standard(pressure, pressure_unit, 'pressure')
        }
    
    def calculate_minimum_reflux_binary(self, relative_volatility: float, 
                                      distillate_purity: float) -> float:
        """Calculate minimum reflux ratio for binary distillation."""
        if len(self.components) != 2:
            raise ValueError("Binary calculation requires exactly 2 components")
        
        alpha = relative_volatility
        xD = distillate_purity
        xF = self.components[0]['feed_mole_fraction']
        
        # Underwood equation for minimum reflux
        Rm = (xD / (alpha - 1)) * ((alpha * xF) / (xD - xF) - 1)
        
        return Rm
    
    def calculate_minimum_stages_binary(self, relative_volatility: float,
                                     distillate_purity: float, bottoms_purity: float) -> float:
        """Calculate minimum number of stages using Fenske equation."""
        alpha = relative_volatility
        xD = distillate_purity
        xB = 1 - bottoms_purity  # Mole fraction of light component in bottoms
        
        Nm = math.log((xD / (1 - xD)) * ((1 - xB) / xB)) / math.log(alpha)
        
        return Nm
    
    def calculate(self) -> Dict[str, Any]:
        """Perform distillation column calculation."""
        results = {
            'equipment_type': 'distillation_column',
            'number_of_components': len(self.components),
            'feed_conditions': self.feed_conditions
        }
        
        self.results = results
        return results


class Pump(UnitOperation):
    """
    Centrifugal pump unit operation class.
    
    Calculates pump power requirements, head, efficiency,
    and NPSH requirements.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.fluid_properties = {}
    
    def set_fluid_properties(self, density: float, density_unit: str,
                           viscosity: float, viscosity_unit: str):
        """Set fluid properties for pump calculations."""
        self.fluid_properties = {
            'density': self.units.normalize_to_standard(density, density_unit, 'density'),
            'viscosity': self.units.normalize_to_standard(viscosity, viscosity_unit, 'viscosity_dynamic')
        }
    
    def calculate_pump_power(self, flow_rate: float, flow_unit: str,
                           head: float, head_unit: str, efficiency: float) -> Dict[str, float]:
        """Calculate pump power requirements."""
        Q = self.units.normalize_to_standard(flow_rate, flow_unit, 'volumetric_flow_rate')
        H = self.units.convert(head, head_unit, 'm')  # Convert head to meters
        
        if not self.fluid_properties:
            raise ValueError("Fluid properties must be set")
        
        rho = self.fluid_properties['density']
        g = 9.81  # m/s²
        
        # Hydraulic power
        P_hydraulic = rho * g * Q * H  # Watts
        
        # Brake power (accounting for efficiency)
        P_brake = P_hydraulic / efficiency
        
        return {
            'hydraulic_power_w': P_hydraulic,
            'brake_power_w': P_brake,
            'hydraulic_power_hp': self.units.convert(P_hydraulic, 'W', 'hp'),
            'brake_power_hp': self.units.convert(P_brake, 'W', 'hp')
        }
    
    def calculate_npsh_required(self, flow_rate: float, flow_unit: str) -> float:
        """Calculate NPSH required (simplified correlation)."""
        Q = self.units.normalize_to_standard(flow_rate, flow_unit, 'volumetric_flow_rate')
        
        # Simplified correlation - actual values depend on pump design
        NPSH_r = 0.5 + 0.1 * (Q * 3600)  # Very rough approximation
        
        return NPSH_r
    
    def calculate(self) -> Dict[str, Any]:
        """Perform pump calculation."""
        if 'flow_rate' not in self.design_parameters or 'head' not in self.design_parameters:
            raise ValueError("Flow rate and head must be specified")
        
        flow_rate = self.get_parameter('flow_rate')
        head = self.get_parameter('head')
        efficiency = self.get_parameter('efficiency', 'dimensionless') if 'efficiency' in self.design_parameters else 0.75
        
        power_results = self.calculate_pump_power(flow_rate, 'm^3/s', head, 'm', efficiency)
        
        results = {
            'equipment_type': 'pump',
            **power_results,
            'efficiency': efficiency
        }
        
        self.results = results
        return results


class Compressor(UnitOperation):
    """
    Compressor unit operation class.
    
    Supports centrifugal and reciprocating compressors with
    polytropic and adiabatic compression calculations.
    """
    
    def __init__(self, name: str = None, compressor_type: str = "centrifugal"):
        super().__init__(name)
        self.type = compressor_type
        self.gas_properties = {}
    
    def set_gas_properties(self, molecular_weight: float, k_ratio: float,
                          z_factor: float = 1.0):
        """Set gas properties for compressor calculations."""
        self.gas_properties = {
            'molecular_weight': molecular_weight,  # kg/kmol
            'k_ratio': k_ratio,  # Cp/Cv
            'z_factor': z_factor  # Compressibility factor
        }
    
    def calculate_adiabatic_power(self, inlet_pressure: float, inlet_pressure_unit: str,
                                outlet_pressure: float, outlet_pressure_unit: str,
                                flow_rate: float, flow_unit: str,
                                inlet_temperature: float, temp_unit: str,
                                efficiency: float = 0.8) -> Dict[str, float]:
        """Calculate adiabatic compression power."""
        P1 = self.units.normalize_to_standard(inlet_pressure, inlet_pressure_unit, 'pressure')
        P2 = self.units.normalize_to_standard(outlet_pressure, outlet_pressure_unit, 'pressure')
        m_dot = self.units.normalize_to_standard(flow_rate, flow_unit, 'mass_flow_rate')
        T1 = self.units.normalize_to_standard(inlet_temperature, temp_unit, 'temperature')
        
        if not self.gas_properties:
            raise ValueError("Gas properties must be set")
        
        k = self.gas_properties['k_ratio']
        MW = self.gas_properties['molecular_weight']
        z = self.gas_properties['z_factor']
        
        R = 8314  # J/(kmol·K)
        
        # Compression ratio
        r = P2 / P1
        
        # Adiabatic temperature rise
        T2_ideal = T1 * (r ** ((k - 1) / k))
        
        # Ideal work
        W_ideal = (m_dot / MW) * z * R * T1 * (k / (k - 1)) * ((r ** ((k - 1) / k)) - 1)
        
        # Actual work (accounting for efficiency)
        W_actual = W_ideal / efficiency
        
        return {
            'compression_ratio': r,
            'outlet_temperature_ideal_k': T2_ideal,
            'ideal_power_w': W_ideal,
            'actual_power_w': W_actual,
            'actual_power_hp': self.units.convert(W_actual, 'W', 'hp')
        }
    
    def calculate(self) -> Dict[str, Any]:
        """Perform compressor calculation."""
        results = {
            'equipment_type': 'compressor',
            'compressor_type': self.type,
            'gas_properties': self.gas_properties
        }
        
        self.results = results
        return results


# Utility functions for common unit operation calculations
def calculate_reynolds_number(velocity: float, diameter: float, density: float,
                            viscosity: float, units: ThermodynamicUnits = None) -> float:
    """Calculate Reynolds number."""
    if units is None:
        units = ThermodynamicUnits()
    
    Re = (density * velocity * diameter) / viscosity
    return Re


def calculate_friction_factor(reynolds_number: float, relative_roughness: float = 0) -> float:
    """Calculate Darcy friction factor using Colebrook equation."""
    if reynolds_number < 2300:
        # Laminar flow
        f = 64 / reynolds_number
    else:
        # Turbulent flow - simplified approximation
        if relative_roughness == 0:
            # Smooth pipe (Blasius for Re < 100,000)
            f = 0.316 * (reynolds_number ** -0.25)
        else:
            # Rough pipe - iterative solution needed for exact Colebrook
            # Using approximation
            f = 0.02 + 0.001 * relative_roughness
    
    return f


def calculate_pressure_drop_pipe(flow_rate: float, diameter: float, length: float,
                               density: float, viscosity: float, roughness: float = 0,
                               units: ThermodynamicUnits = None) -> float:
    """Calculate pressure drop in a pipe using Darcy-Weisbach equation."""
    if units is None:
        units = ThermodynamicUnits()
    
    # Calculate velocity
    area = math.pi * (diameter / 2) ** 2
    velocity = flow_rate / (density * area)
    
    # Calculate Reynolds number
    Re = calculate_reynolds_number(velocity, diameter, density, viscosity, units)
    
    # Calculate friction factor
    relative_roughness = roughness / diameter if diameter > 0 else 0
    f = calculate_friction_factor(Re, relative_roughness)
    
    # Calculate pressure drop
    delta_p = f * (length / diameter) * (density * velocity ** 2) / 2
    
    return delta_p
