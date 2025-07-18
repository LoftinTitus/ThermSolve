"""
Plotting utilities for visualizing thermophysical property results.

This module provides a comprehensive plotting class for creating publication-quality
plots of substance properties, interpolation results, and process engineering data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Union, Tuple, Any
import warnings

try:
    from .substances import Substance
    from .interpolation import PropertyInterpolator, TemperatureDataSeries
except ImportError:
    try:
        from substances import Substance
        from interpolation import PropertyInterpolator, TemperatureDataSeries
    except ImportError:
        warnings.warn("Could not import Substance or interpolation classes")


class ThermPlotter:
    """
    A comprehensive plotting class for thermophysical property visualization.
    
    This class provides methods for creating various types of plots commonly used
    in chemical and process engineering, including:
    - Property vs temperature plots
    - Comparison plots for multiple substances
    - Interpolation visualization with data points
    - Phase diagrams
    - Process operation plots
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[float, float] = (10, 6)):
        """
        Initialize the ThermPlotter.
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use ('default', 'seaborn', 'ggplot', etc.)
        figsize : Tuple[float, float]
            Default figure size (width, height) in inches
        """
        self.style = style
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set matplotlib style
        if style != 'default':
            plt.style.use(style)
    
    def plot_property_vs_temperature(self, 
                                   temperatures: Union[List[float], np.ndarray],
                                   property_values: Union[List[float], np.ndarray],
                                   property_name: str = "Property",
                                   property_units: str = "",
                                   temperature_units: str = "K",
                                   title: Optional[str] = None,
                                   substance_name: Optional[str] = None,
                                   data_points: Optional[Tuple[List[float], List[float]]] = None,
                                   validity_range: Optional[Tuple[float, float]] = None,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> plt.Figure:
        """
        Create a property vs temperature plot.
        
        Parameters:
        -----------
        temperatures : array-like
            Temperature values
        property_values : array-like
            Property values corresponding to temperatures
        property_name : str
            Name of the property being plotted
        property_units : str
            Units of the property
        temperature_units : str
            Units of temperature
        title : str, optional
            Custom title for the plot
        substance_name : str, optional
            Name of the substance
        data_points : tuple, optional
            (temp_data, prop_data) for showing original data points
        validity_range : tuple, optional
            (min_temp, max_temp) for highlighting valid temperature range
        save_path : str, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Main property curve
        ax.plot(temperatures, property_values, 'b-', linewidth=2, 
                label=f'{property_name} (interpolated)' if data_points else property_name)
        
        # Plot original data points if provided
        if data_points is not None:
            temp_data, prop_data = data_points
            ax.scatter(temp_data, prop_data, color='red', s=50, zorder=5,
                      label='Data points', edgecolors='black', linewidth=1)
        
        # Highlight validity range
        if validity_range is not None:
            min_temp, max_temp = validity_range
            y_min, y_max = ax.get_ylim()
            valid_mask = (np.array(temperatures) >= min_temp) & (np.array(temperatures) <= max_temp)
            if np.any(~valid_mask):
                # Shade extrapolated regions
                ax.axvspan(min(temperatures), min_temp, alpha=0.2, color='orange', 
                          label='Extrapolated region')
                ax.axvspan(max_temp, max(temperatures), alpha=0.2, color='orange')
        
        # Labels and title
        temp_label = f"Temperature ({temperature_units})"
        prop_label = f"{property_name}"
        if property_units:
            prop_label += f" ({property_units})"
        
        ax.set_xlabel(temp_label, fontsize=12)
        ax.set_ylabel(prop_label, fontsize=12)
        
        if title is None:
            title = f"{property_name} vs Temperature"
            if substance_name:
                title += f" for {substance_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        if data_points is not None or validity_range is not None:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def compare_substances(self,
                          substance_data: Dict[str, Dict[str, Union[List[float], np.ndarray]]],
                          property_name: str = "Property",
                          property_units: str = "",
                          temperature_units: str = "K",
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """
        Compare the same property for multiple substances.
        
        Parameters:
        -----------
        substance_data : dict
            Dictionary with substance names as keys and each value containing
            'temperatures' and 'property_values' arrays
        property_name : str
            Name of the property being compared
        property_units : str
            Units of the property
        temperature_units : str
            Units of temperature
        title : str, optional
            Custom title for the plot
        save_path : str, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (substance_name, data) in enumerate(substance_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(data['temperatures'], data['property_values'], 
                   color=color, linewidth=2, label=substance_name)
        
        # Labels and title
        temp_label = f"Temperature ({temperature_units})"
        prop_label = f"{property_name}"
        if property_units:
            prop_label += f" ({property_units})"
        
        ax.set_xlabel(temp_label, fontsize=12)
        ax.set_ylabel(prop_label, fontsize=12)
        
        if title is None:
            title = f"{property_name} Comparison"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_interpolation_analysis(self,
                                  interpolator: 'PropertyInterpolator',
                                  temperature_range: Optional[Tuple[float, float]] = None,
                                  n_points: int = 100,
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> plt.Figure:
        """
        Plot interpolation results showing data points, interpolated curve, and analysis.
        
        Parameters:
        -----------
        interpolator : PropertyInterpolator
            The interpolator object to analyze
        temperature_range : tuple, optional
            (min_temp, max_temp) for the plot range
        n_points : int
            Number of points for the interpolated curve
        title : str, optional
            Custom title for the plot
        save_path : str, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        
        # Get data from interpolator
        temps_data = interpolator.temperatures
        values_data = interpolator.values
        
        if temperature_range is None:
            temp_min = min(temps_data) - 0.1 * (max(temps_data) - min(temps_data))
            temp_max = max(temps_data) + 0.1 * (max(temps_data) - min(temps_data))
        else:
            temp_min, temp_max = temperature_range
        
        # Generate interpolated curve
        temps_interp = np.linspace(temp_min, temp_max, n_points)
        values_interp = [interpolator(T) for T in temps_interp]
        
        # Main plot
        ax1.plot(temps_interp, values_interp, 'b-', linewidth=2, label='Interpolated')
        ax1.scatter(temps_data, values_data, color='red', s=60, zorder=5,
                   label='Data points', edgecolors='black', linewidth=1)
        
        # Highlight validity range
        valid_min, valid_max = min(temps_data), max(temps_data)
        extrapolated_mask = (temps_interp < valid_min) | (temps_interp > valid_max)
        if np.any(extrapolated_mask):
            ax1.fill_between(temps_interp, values_interp, alpha=0.2, 
                           where=extrapolated_mask, color='orange', 
                           label='Extrapolated region')
        
        ax1.set_xlabel('Temperature (K)', fontsize=12)
        ax1.set_ylabel(f'{interpolator.property_name}', fontsize=12)
        ax1.set_title(title or f'{interpolator.property_name} Interpolation Analysis', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Residual plot
        residuals = []
        for i, (T, val_true) in enumerate(zip(temps_data, values_data)):
            val_interp = interpolator(T)
            residual = val_true - val_interp
            residuals.append(residual)
        
        ax2.scatter(temps_data, residuals, color='green', s=40)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.set_title('Interpolation Residuals', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_phase_diagram(self,
                          temperatures: Union[List[float], np.ndarray],
                          pressures: Union[List[float], np.ndarray],
                          phase_labels: Optional[List[str]] = None,
                          critical_point: Optional[Tuple[float, float]] = None,
                          triple_point: Optional[Tuple[float, float]] = None,
                          substance_name: Optional[str] = None,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """
        Create a simple phase diagram plot.
        
        Parameters:
        -----------
        temperatures : array-like
            Temperature values for phase boundaries
        pressures : array-like
            Pressure values for phase boundaries
        phase_labels : list, optional
            Labels for different phases
        critical_point : tuple, optional
            (T_critical, P_critical)
        triple_point : tuple, optional
            (T_triple, P_triple)
        substance_name : str, optional
            Name of the substance
        title : str, optional
            Custom title for the plot
        save_path : str, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot phase boundary
        ax.plot(temperatures, pressures, 'b-', linewidth=2, label='Phase boundary')
        
        # Mark special points
        if critical_point:
            ax.scatter(*critical_point, color='red', s=100, marker='*', 
                      label='Critical point', zorder=5, edgecolors='black')
        
        if triple_point:
            ax.scatter(*triple_point, color='green', s=80, marker='o', 
                      label='Triple point', zorder=5, edgecolors='black')
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Pressure (Pa)', fontsize=12)
        
        if title is None:
            title = "Phase Diagram"
            if substance_name:
                title += f" for {substance_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_multiple_properties(self,
                                temperatures: Union[List[float], np.ndarray],
                                properties_data: Dict[str, Dict[str, Union[List[float], np.ndarray, str]]],
                                substance_name: Optional[str] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None,
                                show_plot: bool = True) -> plt.Figure:
        """
        Plot multiple properties on separate subplots.
        
        Parameters:
        -----------
        temperatures : array-like
            Temperature values
        properties_data : dict
            Dictionary with property names as keys and values containing
            'values' (property values) and 'units' (property units)
        substance_name : str, optional
            Name of the substance
        title : str, optional
            Custom title for the plot
        save_path : str, optional
            Path to save the figure
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        n_props = len(properties_data)
        n_cols = min(2, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0]*n_cols/2, 
                                                         self.figsize[1]*n_rows/2))
        
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if n_props > 1 else axes
        
        for i, (prop_name, prop_data) in enumerate(properties_data.items()):
            ax = axes_flat[i]
            
            values = prop_data['values']
            units = prop_data.get('units', '')
            color = self.colors[i % len(self.colors)]
            
            ax.plot(temperatures, values, color=color, linewidth=2)
            ax.set_xlabel('Temperature (K)', fontsize=10)
            
            ylabel = prop_name
            if units:
                ylabel += f" ({units})"
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(prop_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_props, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        if title is None:
            title = "Multiple Properties vs Temperature"
            if substance_name:
                title += f" for {substance_name}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig


# Convenience function for backwards compatibility
def plot_property_vs_temperature(temperatures, property_values, property_name="Property", 
                                property_units="", temperature_units="K", **kwargs):
    """
    Convenience function for creating a simple property vs temperature plot.
    
    This function maintains backwards compatibility with existing code.
    """
    plotter = ThermPlotter()
    return plotter.plot_property_vs_temperature(
        temperatures, property_values, property_name, property_units, 
        temperature_units, **kwargs
    )