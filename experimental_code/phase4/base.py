# phase4/base.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from rcft_framework import RCFTExperiment

class DirectedMemoryExperiment:
    """Base class for all Phase IV experiments with directed memory manipulation"""
    
    def __init__(self, 
                 output_dir="phase4_results",
                 alpha=0.35, 
                 beta=0.5, 
                 gamma=0.92, 
                 pattern_type="fractal"):
        """Initialize base experiment parameters"""
        self.output_dir = output_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pattern_type = pattern_type
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(output_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Placeholder for experiment-specific attributes
        self.experiment = None
        self.metrics_history = []
        self.field_snapshots = []
    
    def initialize_experiment(self):
        """Initialize the RCFT experiment with standard parameters"""
        self.experiment = RCFTExperiment(
            memory_strength=self.alpha,
            coupling_strength=self.beta,
            memory_decay=self.gamma
        )
        self.experiment.initialize_pattern(pattern_type=self.pattern_type)
        
        # Save initial state
        self.initial_state = self.experiment.state.copy()
        self.field_snapshots.append(self.initial_state)
        
        return self.experiment
    
    def save_field_snapshot(self, description=""):
        """Save current field state as snapshot"""
        snapshot = self.experiment.state.copy()
        self.field_snapshots.append(snapshot)
        return len(self.field_snapshots) - 1  # Return index
    
    def load_field_snapshot(self, snapshot_idx=-1):
        """Load field from snapshot"""
        if 0 <= snapshot_idx < len(self.field_snapshots):
            self.experiment.state = self.field_snapshots[snapshot_idx].copy()
            return True
        return False
    
    def compute_metrics(self):
        """Compute core metrics and return them"""
        if self.experiment is None:
            return None
            
        metrics = {
            'correlation': self.experiment.metrics['correlation'][-1] if len(self.experiment.metrics['correlation']) > 0 else 0,
            'coherence': self.experiment.metrics['coherence'][-1] if len(self.experiment.metrics['coherence']) > 0 else 0,
            'ccdi': self.experiment.metrics['coherence'][-1] - self.experiment.metrics['correlation'][-1] if len(self.experiment.metrics['correlation']) > 0 else 0,
            'spectral_entropy': self.experiment.metrics['spectral_entropy'][-1] if len(self.experiment.metrics['spectral_entropy']) > 0 else 0,
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def run(self, **kwargs):
        """Abstract method to run the experiment"""
        raise NotImplementedError("Subclasses must implement run()")
    
    def save_results(self, results, filename):
        """Save results to CSV"""
        results_df = pd.DataFrame(results)
        output_path = os.path.join(self.output_dir, filename)
        results_df.to_csv(output_path, index=False)
        return output_path
    
    def visualize_results(self, **kwargs):
        """Abstract method for result visualization"""
        raise NotImplementedError("Subclasses must implement visualize_results()")