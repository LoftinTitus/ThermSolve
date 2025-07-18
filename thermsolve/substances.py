"""
Substance class for thermophysical property calculations.
"""
import json
import csv
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import warnings

try:
    from .interpolation import PropertyInterpolator
    HAS_INTERPOLATION = True
except ImportError:
    try:
        from interpolation import PropertyInterpolator
        HAS_INTERPOLATION = True
    except ImportError:
        HAS_INTERPOLATION = False
        warnings.warn("Interpolation module not available. Some features will be disabled.")

try:
    import pint
    ureg = pint.UnitRegistry()
    HAS_PINT = True
except ImportError:
    HAS_PINT = False
    warnings.warn("Pint not available. Unit conversions will be disabled.")


class Substance:
    """
    A class representing a chemical substance with its thermophysical properties.
    
    This class provides access to temperature-dependent and constant properties
    such as heat capacity, viscosity, density, phase transition temperatures, etc.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize a Substance object.
        
        Parameters:
        -----------
        name : str
            Name of the substance
        **kwargs : dict
            Additional properties as keyword arguments
        """
        self.name = name
        self.cas_number = kwargs.get('cas_number', None)
        
        # Basic properties
        self.molecular_weight = kwargs.get('molecular_weight', None)  # g/mol
        self.formula = kwargs.get('formula', None)
        
        # Phase transition properties
        self.melting_point = kwargs.get('melting_point', None)  # K
        self.boiling_point = kwargs.get('boiling_point', None)  # K
        self.critical_temperature = kwargs.get('critical_temperature', None)  # K
        self.critical_pressure = kwargs.get('critical_pressure', None)  # Pa
        self.critical_density = kwargs.get('critical_density', None)  # kg/m³
        
        # Enthalpy properties
        self.enthalpy_of_fusion = kwargs.get('enthalpy_of_fusion', None)  # J/mol
        self.enthalpy_of_vaporization = kwargs.get('enthalpy_of_vaporization', None)  # J/mol
        self.enthalpy_of_formation = kwargs.get('enthalpy_of_formation', None)  # J/mol
        
        # Reference state properties
        self.reference_temperature = kwargs.get('reference_temperature', 298.15)  # K
        self.reference_pressure = kwargs.get('reference_pressure', 101325)  # Pa
        
        # Temperature-dependent properties (stored as coefficients or data points)
        self.cp_coefficients = kwargs.get('cp_coefficients', None)  # Heat capacity
        # Support legacy or alternate keys for heat capacity
        if self.cp_coefficients is None:
            # If cp_coefficients is a number, treat as constant
            if 'cp_constant' in kwargs:
                self.cp_coefficients = {
                    'type': 'constant',
                    'value': kwargs['cp_constant'],
                    'units': kwargs.get('cp_units', 'J/kg/K')
                }
            # If cp_coefficients is a number (from CSV), treat as constant
            elif 'cp_coefficients' in kwargs and isinstance(kwargs['cp_coefficients'], (int, float)):
                self.cp_coefficients = {
                    'type': 'constant',
                    'value': kwargs['cp_coefficients'],
                    'units': 'J/kg/K'
                }
        self.viscosity_coefficients = kwargs.get('viscosity_coefficients', None)
        self.density_coefficients = kwargs.get('density_coefficients', None)
        self.vapor_pressure_coefficients = kwargs.get('vapor_pressure_coefficients', None)
        self.thermal_conductivity_coefficients = kwargs.get('thermal_conductivity_coefficients', None)
        
        # Validity ranges for properties
        self.temp_range = kwargs.get('temp_range', (0, 1000))  # K
        self.pressure_range = kwargs.get('pressure_range', (0, 1e6))  # Pa
        
        # Source metadata
        self.data_sources = kwargs.get('data_sources', {})
        
        # Units handling
        self.units = kwargs.get('units', {})
        
    def heat_capacity(self, T: float, pressure: Optional[float] = None) -> float:
        """
        Calculate heat capacity at constant pressure.
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin
        pressure : float, optional
            Pressure in Pa (for pressure-dependent correlations)
            
        Returns:
        --------
        float
            Heat capacity in J/(mol·K)
        """
        self._check_temperature_range(T)
        
        if self.cp_coefficients is None:
            raise ValueError(f"Heat capacity data not available for {self.name}")
        
        # Handle different coefficient types
        if isinstance(self.cp_coefficients, dict):
            if self.cp_coefficients.get('type') == 'polynomial':
                coeffs = self.cp_coefficients['coefficients']
                cp = sum(coeff * T**i for i, coeff in enumerate(coeffs))
                return cp
            elif self.cp_coefficients.get('type') == 'constant':
                return self.cp_coefficients['value']
            elif self.cp_coefficients.get('type') == 'interpolated' and HAS_INTERPOLATION:
                interpolator = self.cp_coefficients['interpolator']
                return interpolator(T)
        
        return None
    
    def viscosity(self, T: float, pressure: Optional[float] = None) -> float:
        """
        Calculate dynamic viscosity.
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin
        pressure : float, optional
            Pressure in Pa
            
        Returns:
        --------
        float
            Dynamic viscosity in Pa·s
        """
        self._check_temperature_range(T)
        
        if self.viscosity_coefficients is None:
            raise ValueError(f"Viscosity data not available for {self.name}")
        
        # Handle different coefficient types
        if isinstance(self.viscosity_coefficients, dict):
            if self.viscosity_coefficients.get('type') == 'arrhenius':
                A = self.viscosity_coefficients['A']
                B = self.viscosity_coefficients['B']
                return A * (T ** self.viscosity_coefficients.get('n', 0)) * \
                       (1 + self.viscosity_coefficients.get('C', 0) / T)
            elif self.viscosity_coefficients.get('type') == 'interpolated' and HAS_INTERPOLATION:
                interpolator = self.viscosity_coefficients['interpolator']
                return interpolator(T)
        
        return None
    
    def density(self, T: float, pressure: Optional[float] = None) -> float:
        """
        Calculate density.
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin
        pressure : float, optional
            Pressure in Pa
            
        Returns:
        --------
        float
            Density in kg/m³
        """
        self._check_temperature_range(T)
        
        if self.density_coefficients is None:
            raise ValueError(f"Density data not available for {self.name}")
        
        # Handle different coefficient types
        if isinstance(self.density_coefficients, dict):
            if self.density_coefficients.get('type') == 'linear':
                A = self.density_coefficients['A']
                B = self.density_coefficients['B']
                return A + B * T
            elif self.density_coefficients.get('type') == 'constant':
                return self.density_coefficients['value']
            elif self.density_coefficients.get('type') == 'interpolated' and HAS_INTERPOLATION:
                interpolator = self.density_coefficients['interpolator']
                return interpolator(T)
        
        return None
    
    def vapor_pressure(self, T: float) -> float:
        """
        Calculate vapor pressure using Antoine equation
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin
            
        Returns:
        --------
        float
            Vapor pressure in Pa
        """
        self._check_temperature_range(T)
        
        if self.vapor_pressure_coefficients is None:
            raise ValueError(f"Vapor pressure data not available for {self.name}")
        
        # Antoine equation: log10(P) = A - B/(C + T)
        if isinstance(self.vapor_pressure_coefficients, dict):
            if self.vapor_pressure_coefficients.get('type') == 'antoine':
                A = self.vapor_pressure_coefficients['A']
                B = self.vapor_pressure_coefficients['B']
                C = self.vapor_pressure_coefficients['C']
                log_p = A - B / (C + T)
                return 10 ** log_p
        
        return None
    
    def thermal_conductivity(self, T: float, pressure: Optional[float] = None) -> float:
        """
        Calculate thermal conductivity.
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin
        pressure : float, optional
            Pressure in Pa
            
        Returns:
        --------
        float
            Thermal conductivity in W/(m·K)
        """
        self._check_temperature_range(T)
        
        if self.thermal_conductivity_coefficients is None:
            raise ValueError(f"Thermal conductivity data not available for {self.name}")
        
        # Example: polynomial correlation k = A + B*T + C*T^2
        if isinstance(self.thermal_conductivity_coefficients, dict):
            if self.thermal_conductivity_coefficients.get('type') == 'polynomial':
                coeffs = self.thermal_conductivity_coefficients['coefficients']
                k = sum(coeff * T**i for i, coeff in enumerate(coeffs))
                return k
        
        return None
    
    def _check_temperature_range(self, T: float):
        """Check if temperature is within valid range."""
        if not (self.temp_range[0] <= T <= self.temp_range[1]):
            warnings.warn(f"Temperature {T} K is outside valid range "
                         f"{self.temp_range[0]}-{self.temp_range[1]} K for {self.name}")
    
    def _check_pressure_range(self, P: float):
        """Check if pressure is within valid range."""
        if not (self.pressure_range[0] <= P <= self.pressure_range[1]):
            warnings.warn(f"Pressure {P} Pa is outside valid range "
                         f"{self.pressure_range[0]}-{self.pressure_range[1]} Pa for {self.name}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Substance':
        """
        Create a Substance from a dictionary.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing substance properties
            
        Returns:
        --------
        Substance
            New Substance instance
        """
        name = data.pop('name')
        return cls(name, **data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'Substance':
        """
        Load substance data from JSON file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to JSON file
            
        Returns:
        --------
        Substance
            New Substance instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert substance to dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of substance
        """
        return {
            'name': self.name,
            'cas_number': self.cas_number,
            'molecular_weight': self.molecular_weight,
            'formula': self.formula,
            'melting_point': self.melting_point,
            'boiling_point': self.boiling_point,
            'critical_temperature': self.critical_temperature,
            'critical_pressure': self.critical_pressure,
            'critical_density': self.critical_density,
            'enthalpy_of_fusion': self.enthalpy_of_fusion,
            'enthalpy_of_vaporization': self.enthalpy_of_vaporization,
            'enthalpy_of_formation': self.enthalpy_of_formation,
            'reference_temperature': self.reference_temperature,
            'reference_pressure': self.reference_pressure,
            'cp_coefficients': self.cp_coefficients,
            'viscosity_coefficients': self.viscosity_coefficients,
            'density_coefficients': self.density_coefficients,
            'vapor_pressure_coefficients': self.vapor_pressure_coefficients,
            'thermal_conductivity_coefficients': self.thermal_conductivity_coefficients,
            'temp_range': self.temp_range,
            'pressure_range': self.pressure_range,
            'data_sources': self.data_sources,
            'units': self.units
        }
    
    def to_json(self, filepath: Union[str, Path]):
        """
        Save substance data to JSON file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to output JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __str__(self) -> str:
        """String representation of substance."""
        return f"Substance(name='{self.name}', formula='{self.formula}', MW={self.molecular_weight})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


class SubstanceDatabase:
    """
    A database for managing multiple substances and their properties.
    """
    
    def __init__(self):
        self.substances = {}
    
    def add_substance(self, substance: Substance):
        """Add a substance to the database."""
        self.substances[substance.name.lower()] = substance
    
    def get_substance(self, name: str) -> Optional[Substance]:
        """Get a substance by name."""
        return self.substances.get(name.lower())
    
    def load_from_csv(self, filepath: Union[str, Path]):
        """
        Load substances from CSV file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to CSV file
        """
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                for key, value in row.items():
                    if value and value != '':
                        try:
                            row[key] = float(value)
                        except ValueError:
                            pass  # Keep as string
                
                substance = Substance.from_dict(row)
                self.add_substance(substance)
    
    def save_to_csv(self, filepath: Union[str, Path]):
        """
        Save substances to CSV file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to output CSV file
        """
        if not self.substances:
            return
        
        # Get all possible fields from all substances
        all_fields = set()
        for substance in self.substances.values():
            all_fields.update(substance.to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
            writer.writeheader()
            for substance in self.substances.values():
                writer.writerow(substance.to_dict())
    
    def list_substances(self) -> list:
        """List all substance names in the database."""
        return list(self.substances.keys())
    
    def search_by_formula(self, formula: str) -> list:
        """Search substances by chemical formula."""
        return [s for s in self.substances.values() if s.formula == formula]
    
    def search_by_cas(self, cas_number: str) -> Optional[Substance]:
        """Search substance by CAS number."""
        for substance in self.substances.values():
            if substance.cas_number == cas_number:
                return substance
        return None