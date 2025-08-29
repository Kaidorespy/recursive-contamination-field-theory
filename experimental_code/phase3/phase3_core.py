"""
Phase 3 Core - Base class for Phase 3 RCFT experiments
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import logging
from tqdm import tqdm
import multiprocessing as mp

# Import from previous phase code
from rcft_framework import RCFTExperiment
from rcft_metrics import (
    compute_ccdi, 
    classify_recovery, 
    extract_attractor_residual,
    compute_spectral_entropy
)

class FalseAttractorSample:
    """Container for false attractor data and metadata"""
    
    def __init__(self, alpha, gamma, initial_state, final_state, 
                 perturbation_info, recovery_trajectory, metrics):
        """
        Initialize a false attractor sample
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        initial_state : ndarray
            Initial field state
        final_state : ndarray
            Final attractor state
        perturbation_info : dict
            Information about applied perturbations
        recovery_trajectory : dict
            Time series of recovery metrics
        metrics : dict
            Final metrics and classifications
        """
        self.alpha = alpha
        self.gamma = gamma
        self.initial_state = initial_state
        self.final_state = final_state
        self.perturbation_info = perturbation_info
        self.recovery_trajectory = recovery_trajectory
        self.metrics = metrics
        
        # Derived properties
        self.residual = final_state - initial_state
        self.ccdi = metrics.get('ccdi', 0.0)
        self.recovery_class = metrics.get('recovery_class', 'unknown')
        self.is_anomalous = metrics.get('is_anomalous', False)
    
    def to_dict(self):
        """Convert to serializable dictionary (without numpy arrays)"""
        return {
            'alpha': float(self.alpha),
            'gamma': float(self.gamma),
            'ccdi': float(self.ccdi),
            'recovery_class': self.recovery_class,
            'is_anomalous': bool(self.is_anomalous),
            'final_correlation': float(self.metrics.get('final_correlation', 0.0)),
            'final_coherence': float(self.metrics.get('final_coherence', 0.0)),
            'final_entropy': float(self.metrics.get('final_entropy', 0.0)),
            'residual_norm': float(np.linalg.norm(self.residual)),
        }
    
    def save(self, directory):
        """Save the attractor sample to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save metadata as JSON
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        # Save states as NumPy arrays
        np.save(os.path.join(directory, 'initial_state.npy'), self.initial_state)
        np.save(os.path.join(directory, 'final_state.npy'), self.final_state)
        np.save(os.path.join(directory, 'residual.npy'), self.residual)
        
        # Save recovery trajectory as CSV
        trajectory_df = pd.DataFrame(self.recovery_trajectory)
        trajectory_df.to_csv(os.path.join(directory, 'recovery_trajectory.csv'), index=False)
    
    @classmethod
    def load(cls, directory):
        """Load a false attractor sample from disk"""
        # Load metadata
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        # Load states
        initial_state = np.load(os.path.join(directory, 'initial_state.npy'))
        final_state = np.load(os.path.join(directory, 'final_state.npy'))
        
        # Load trajectory
        trajectory_df = pd.read_csv(os.path.join(directory, 'recovery_trajectory.csv'))
        recovery_trajectory = trajectory_df.to_dict('list')
        
        # Create a minimal perturbation_info if we can't load it
        perturbation_info = {
            'type': 'flip',
            'magnitude': 1.0,
            'steps': [1, metadata.get('delay', 10) + 1]
        }
        
        # Create instance
        return cls(
            alpha=metadata['alpha'],
            gamma=metadata['gamma'],
            initial_state=initial_state,
            final_state=final_state,
            perturbation_info=perturbation_info,
            recovery_trajectory=recovery_trajectory,
            metrics=metadata
        )


class Phase3Core:
    """Base class for Phase 3 RCFT experiments"""
    
    def __init__(self, output_dir="phase3_results", log_dir="phase3_logs"):
        """Initialize the Phase 3 core with default settings"""
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        # Default parameters
        self.beta = 0.5  # Coupling strength (fixed)
        self.ccdi_threshold = 0.08  # Anomaly threshold
        self.base_seed = 42  # Random seed
        
        # Setup directories and logging
        self._setup_directories()
        self._setup_logging()
        
        # Storage for results
        self.results_df = None
        self.samples = []
    
    def _setup_directories(self):
        """Create necessary directories"""
        # Create main output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create subdirectories for different data types
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        
        # Create log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _setup_logging(self):
        """Configure logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"phase3_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}. Output dir: {self.output_dir}")
    
    def run_rcft_simulation(self, alpha, gamma, pattern_type, delay=10, 
                           run_id=None, seed=None):
        """
        Run a single RCFT simulation with specified parameters
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        pattern_type : str
            Pattern type to initialize
        delay : int
            Steps between perturbations
        run_id : str, optional
            Identifier for this run
        seed : int, optional
            Random seed
            
        Returns:
        --------
        FalseAttractorSample
            Sample containing simulation results
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"a{alpha:.4f}_g{gamma:.4f}"
        
        # Create output directory
        run_dir = os.path.join(self.output_dir, "samples", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize experiment
        exp = RCFTExperiment(
            memory_strength=alpha,
            coupling_strength=self.beta,
            memory_decay=gamma
        )
        
        # Initialize pattern
        exp.initialize_pattern(pattern_type=pattern_type)
        initial_state = exp.state.copy()
        
        # Apply first perturbation
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        first_perturbation = exp.perturbation_step
        
        # Let system recover for specified delay
        exp.update(steps=delay)
        
        # Record mid-state
        mid_state = exp.state.copy()
        
        # Apply second perturbation
        exp.apply_perturbation(
            perturbation_type="flip", 
            magnitude=1.0, 
            center=(20, 20),  # Different location
            radius=10
        )
        second_perturbation = len(exp.history) - 1
        
        # Let system evolve to completion
        exp.update(steps=50)
        
        # Capture final state and metrics
        final_state = exp.state.copy()
        final_correlation = exp.metrics['correlation'][-1]
        final_coherence = exp.metrics['coherence'][-1]
        final_mutual_info = exp.metrics['mutual_info'][-1]
        final_entropy = exp.metrics['spectral_entropy'][-1]
        
        # Calculate CCDI
        ccdi = compute_ccdi(final_correlation, final_coherence)
        
        # Classify recovery trajectory
        recovery_class = classify_recovery(
            exp.metrics['correlation'], 
            [first_perturbation, second_perturbation]
        )
        
        # Construct metrics dictionary
        metrics = {
            'final_correlation': final_correlation,
            'final_coherence': final_coherence,
            'final_mutual_info': final_mutual_info,
            'final_entropy': final_entropy,
            'ccdi': ccdi,
            'recovery_class': recovery_class,
            'is_anomalous': ccdi > self.ccdi_threshold,
            'delay': delay
        }
        
        # Construct perturbation info
        perturbation_info = {
            'type': 'flip',
            'magnitude': 1.0,
            'steps': [first_perturbation, second_perturbation]
        }
        
        # Extract recovery trajectory
        recovery_trajectory = {
            'step': list(range(len(exp.metrics['correlation']))),
            'correlation': exp.metrics['correlation'],
            'coherence': exp.metrics['coherence'],
            'mutual_info': exp.metrics['mutual_info'],
            'spectral_entropy': exp.metrics['spectral_entropy'],
            'ccdi': [exp.metrics['coherence'][i] - exp.metrics['correlation'][i] 
                     for i in range(len(exp.metrics['correlation']))]
        }
        
        # Create sample
        sample = FalseAttractorSample(
            alpha=alpha,
            gamma=gamma,
            initial_state=initial_state,
            final_state=final_state,
            perturbation_info=perturbation_info,
            recovery_trajectory=recovery_trajectory,
            metrics=metrics
        )
        
        # Save visualizations
        self._visualize_experiment(
            exp, initial_state, mid_state, final_state,
            [first_perturbation, second_perturbation],
            run_dir
        )
        
        # Save residual visualization
        self._visualize_residual(
            sample.residual, 
            os.path.join(run_dir, "residual.png")
        )
        
        # Save sample to disk
        sample.save(run_dir)
        
        return sample
    
    def _visualize_experiment(self, exp, initial_state, mid_state, final_state, 
                             perturbation_steps, output_dir):
        """Create visualizations for a single experiment"""
        # Plot metrics
        plt.figure(figsize=(12, 8))
        
        # Plot correlation
        plt.subplot(2, 2, 1)
        plt.plot(exp.metrics['correlation'], label='Correlation')
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--', 
                      label=f'Perturbation {i+1}' if i == 0 else None)
        plt.title('Correlation with Initial State')
        plt.xlabel('Time Step')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.legend()
        
        # Plot coherence
        plt.subplot(2, 2, 2)
        plt.plot(exp.metrics['coherence'], label='Coherence')
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--')
        plt.title('Field Coherence')
        plt.xlabel('Time Step')
        plt.ylabel('Coherence')
        plt.grid(True)
        
        # Plot entropy
        plt.subplot(2, 2, 3)
        plt.plot(exp.metrics['spectral_entropy'], label='Entropy')
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--')
        plt.title('Spectral Entropy')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.grid(True)
        
        # Plot CCDI
        plt.subplot(2, 2, 4)
        ccdi = [exp.metrics['coherence'][i] - exp.metrics['correlation'][i] 
                for i in range(len(exp.metrics['coherence']))]
        plt.plot(ccdi, label='CCDI')
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--')
        plt.axhline(y=self.ccdi_threshold, color='g', linestyle='--', label='Anomaly Threshold')
        plt.title('Coherence-Correlation Divergence Index')
        plt.xlabel('Time Step')
        plt.ylabel('CCDI')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics.png"), dpi=300)
        plt.close()
        
        # Create key frames visualization
        plt.figure(figsize=(16, 4))
        
        # Initial state
        plt.subplot(1, 4, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Initial State")
        plt.axis('off')
        
        # After first perturbation
        plt.subplot(1, 4, 2)
        plt.imshow(exp.history[perturbation_steps[0]], cmap='viridis', vmin=-1, vmax=1)
        plt.title("After First Perturbation")
        plt.axis('off')
        
        # Mid recovery (before second perturbation)
        plt.subplot(1, 4, 3)
        plt.imshow(mid_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Mid Recovery")
        plt.axis('off')
        
        # Final state
        plt.subplot(1, 4, 4)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Final State")
        plt.axis('off')
        
        plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "key_frames.png"), dpi=300)
        plt.close()
    
    def _visualize_residual(self, residual, save_path):
        """Visualize the attractor residual (Delta field)"""
        plt.figure(figsize=(8, 6))
        
        # Plot the residual field
        im = plt.imshow(residual, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(im, label='Delta Value')
        plt.title('Attractor Residual (Final - Initial)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def save_results(self, results_df, filename):
        """Save results DataFrame to CSV"""
        results_path = os.path.join(self.output_dir, filename)
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Saved results to {results_path}")
        return results_path
        
    def load_samples(self, directory=None):
        """Load all attractor samples from the samples directory"""
        if directory is None:
            directory = os.path.join(self.output_dir, "samples")
            
        if not os.path.exists(directory):
            self.logger.warning(f"Samples directory not found: {directory}")
            return []
            
        # List all subdirectories (each should be a sample)
        sample_dirs = [d for d in os.listdir(directory) 
                      if os.path.isdir(os.path.join(directory, d))]
        
        if not sample_dirs:
            self.logger.warning("No sample directories found")
            return []
            
        self.logger.info(f"Found {len(sample_dirs)} sample directories")
        
        # Load each sample
        samples = []
        for sample_dir in tqdm(sample_dirs, desc="Loading samples"):
            try:
                sample_path = os.path.join(directory, sample_dir)
                sample = FalseAttractorSample.load(sample_path)
                samples.append(sample)
            except Exception as e:
                self.logger.error(f"Error loading sample {sample_dir}: {e}")
                
        self.logger.info(f"Successfully loaded {len(samples)} samples")
        return samples
    
    def extract_trajectory_fingerprint(self, sample, n_points=10):
        """
        Extract a fixed-length fingerprint from a recovery trajectory
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Sample containing recovery trajectory
        n_points : int
            Number of points to sample from trajectory
            
        Returns:
        --------
        ndarray
            Fingerprint array combining correlation, coherence, and entropy
        """
        # Get trajectory data
        corr = np.array(sample.recovery_trajectory['correlation'])
        cohf = np.array(sample.recovery_trajectory['coherence'])
        entr = np.array(sample.recovery_trajectory['spectral_entropy'])
        steps = np.array(sample.recovery_trajectory['step'])
        
        # Find second perturbation step
        pert_step = sample.perturbation_info['steps'][1]
        
        # Extract post-perturbation trajectory
        post_idx = np.where(steps >= pert_step)[0]
        if len(post_idx) == 0:
            self.logger.warning(f"No post-perturbation steps found for sample {sample.alpha}_{sample.gamma}")
            return np.zeros(n_points * 3)  # Return zeros if no data
            
        post_steps = steps[post_idx]
        post_corr = corr[post_idx]
        post_cohf = cohf[post_idx]
        post_entr = entr[post_idx]
        
        # Sample at evenly spaced points
        if len(post_steps) < n_points:
            # Pad with last value if not enough points
            pad_len = n_points - len(post_steps)
            post_corr = np.pad(post_corr, (0, pad_len), 'edge')
            post_cohf = np.pad(post_cohf, (0, pad_len), 'edge')
            post_entr = np.pad(post_entr, (0, pad_len), 'edge')
            sample_idx = np.arange(len(post_corr))
        else:
            # Select evenly spaced indices
            sample_idx = np.linspace(0, len(post_steps)-1, n_points, dtype=int)
            
        # Create fingerprint
        corr_fp = post_corr[sample_idx]
        cohf_fp = post_cohf[sample_idx]
        entr_fp = post_entr[sample_idx]
        
        # Combine into a single array
        fingerprint = np.concatenate([corr_fp, cohf_fp, entr_fp])
        
        return fingerprint
