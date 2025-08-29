"""
Module base class for Phase V modules.
This provides a common foundation for all Phase V modules.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Import utilities
from .utils.fingerprinting import AttractorFingerprinter
from .utils.visualization import RCFTVisualizer
from .utils.metrics import IdentityMetrics, IdentityTrace

class PhaseVModule(ABC):
    """Base class for all Phase V modules."""
    
    def __init__(self, output_dir=None, **kwargs):
        """
        Initialize the module.
        
        Args:
            output_dir: Directory for output files
            **kwargs: Additional module-specific parameters
        """
        # Set up output directory
        self.module_name = self.__class__.__name__
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            self.output_dir = os.path.join(output_dir, f"{self.module_name}_{self.timestamp}")
        else:
            self.output_dir = f"{self.module_name}_{self.timestamp}"
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize utilities
        self.fingerprinter = AttractorFingerprinter()
        self.visualizer = RCFTVisualizer(output_dir=self.output_dir)
        
        # Initialize parameters
        self.params = kwargs
        self.save_parameters()
        
        # Results storage
        self.results = {}
        self.metrics = {}
        self.identity_traces = []
        
    def save_parameters(self):
        """Save parameters to JSON file."""
        params_file = os.path.join(self.output_dir, "parameters.json")
        with open(params_file, 'w') as f:
            json.dump(self.params, f, indent=2)
            
    def save_results(self, results=None):
        """
        Save results to JSON file.
        
        Args:
            results: Optional results dictionary to save
                    (uses self.results if None)
        """
        if results is None:
            results = self.results
            
        # Handle numpy arrays for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
                
        results_json = convert_arrays(results)
        
        results_file = os.path.join(self.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)
            
    def save_metrics(self, metrics=None):
        """
        Save metrics to CSV file.
        
        Args:
            metrics: Optional metrics dictionary to save
                    (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
            
        # Convert metrics to CSV format
        import pandas as pd
        
        # Try to convert to DataFrame
        try:
            # If metrics are all scalar values, we need to create a single-row DataFrame
            if all(not isinstance(v, (list, tuple, np.ndarray)) for v in metrics.values()):
                df = pd.DataFrame([metrics])  # Wrap in list to create a single row
            else:
                df = pd.DataFrame(metrics)
                
            metrics_file = os.path.join(self.output_dir, "metrics.csv")
            df.to_csv(metrics_file, index=False)
        except (ValueError, TypeError):
            # If metrics has irregular structure, save as JSON
            metrics_file = os.path.join(self.output_dir, "metrics.json")
            # Define the conversion function
            def convert_arrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_arrays(item) for item in obj]
                else:
                    return obj
                    
            metrics_json = convert_arrays(metrics)
            with open(metrics_file, 'w') as f:
                json.dump(metrics_json, f, indent=2)
                
    def load_config(self, config_file):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Dictionary of configuration parameters
        """
        if not os.path.exists(config_file):
            print(f"Warning: Config file {config_file} not found.")
            return {}
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Update parameters
        self.params.update(config)
        self.save_parameters()
        
        return config
        
    @abstractmethod
    def run(self, experiment, **kwargs):
        """
        Run the module. Must be implemented by subclasses.
        
        Args:
            experiment: RCFT experiment instance
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary
        """
        pass
    
    def track_identity(self, experiment, label=None):
        """
        Create an identity trace for tracking a pattern.
        
        Args:
            experiment: RCFT experiment
            label: Optional label for the trace
            
        Returns:
            IdentityTrace instance
        """
        if label is None:
            label = f"Trace_{len(self.identity_traces)}"
            
        trace = IdentityTrace(experiment.state.copy(), label=label)
        self.identity_traces.append(trace)
        
        return trace
        
    def update_identity(self, experiment, trace_index=-1):
        """
        Update an identity trace with the current state.
        
        Args:
            experiment: RCFT experiment
            trace_index: Index of trace to update
            
        Returns:
            Updated metrics
        """
        if not self.identity_traces:
            return None
            
        trace = self.identity_traces[trace_index]
        metrics = trace.add_state(experiment.state.copy())
        
        return metrics
        
    def visualize_identity_metrics(self, trace_indices=None, metrics_list=None, 
                                title=None, save_path=None):
        """
        Visualize identity metrics.
        
        Args:
            trace_indices: Indices of traces to visualize (all if None)
            metrics_list: List of metrics to visualize
            title: Optional title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not self.identity_traces:
            return None
            
        # Default to all traces
        if trace_indices is None:
            trace_indices = list(range(len(self.identity_traces)))
            
        # Get traces to visualize
        traces = [self.identity_traces[i] for i in trace_indices 
                if i >= 0 and i < len(self.identity_traces)]
                
        # Default metrics
        if metrics_list is None:
            metrics_list = ['correlation', 'coherence', 'ccdi']
            
        # Default save path
        if save_path is None:
            save_path = os.path.join(self.output_dir, "identity_metrics.png")
            
        # Create visualization
        return self.visualizer.visualize_identity_metrics(
            traces, metrics_list, title=title, save_path=save_path)
            
    def create_summary(self):
        """
        Create a summary of module results.
        
        Returns:
            Summary dictionary
        """
        # Create a summary of key results and metrics
        summary = {
            'module_name': self.module_name,
            'timestamp': self.timestamp,
            'parameters': self.params
        }
        
        # Add key metrics if available
        if self.metrics:
            summary['metrics'] = {}
            
            # Look for final values of common metrics
            for metric_name in ['correlation', 'coherence', 'ccdi', 
                              'identity_persistence', 'self_distinction']:
                if metric_name in self.metrics:
                    if isinstance(self.metrics[metric_name], list):
                        summary['metrics'][f'final_{metric_name}'] = self.metrics[metric_name][-1]
                    else:
                        summary['metrics'][metric_name] = self.metrics[metric_name]
                        
        # Add identity trace summaries
        if self.identity_traces:
            summary['identity_traces'] = []
            
            for trace in self.identity_traces:
                trace_summary = {
                    'label': trace.label,
                    'iterations': trace.iteration,
                    'final_correlation': trace.correlations[-1],
                    'final_coherence': trace.coherence[-1],
                    'final_ccdi': trace.ccdi[-1],
                    'identity_persistence': trace.compute_identity_persistence(),
                    'self_distinction': trace.compute_self_distinction_index()
                }
                
                summary['identity_traces'].append(trace_summary)
                
        # Add any additional results
        if self.results:
            summary['results'] = {}
            
            # Look for key result metrics
            for key in ['success_rate', 'recovery_quality', 'echo_correction']:
                if key in self.results:
                    summary['results'][key] = self.results[key]
                    
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary