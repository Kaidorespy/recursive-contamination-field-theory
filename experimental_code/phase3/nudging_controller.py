"""
Nudging Controller - Module 2 for Phase 3

Applies nudging interventions to guide memory recovery back to original attractors.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import time
from tqdm import tqdm
import multiprocessing as mp
# Add this at the top of phase3/nudging_controller.py
from rcft_framework import RCFTExperiment

from phase3.phase3_core import Phase3Core, FalseAttractorSample
from phase3.boundary_cartography import AttractorBoundaryMapper

class NudgingController(Phase3Core):
    """Applies nudging interventions to guide memory recovery"""
    
    def __init__(self, output_dir="phase3_results/nudging_experiments", log_dir="phase3_logs"):
        """Initialize the nudging controller"""
        super().__init__(output_dir, log_dir)
        
        # Nudging parameters
        self.nudge_times = [30, 40, 50]  # Time steps for nudging
        self.nudge_amplitudes = [0.01, 0.05, 0.1]  # Nudge amplitudes to test
        self.nudge_types = ['uniform', 'patch', 'echo']  # Types of nudges to apply
        
        # Source of false attractor samples
        self.source_samples_dir = None
        
        # Run ID for this experiment
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def set_source_samples(self, source_dir):
        """Set the source directory for false attractor samples"""
        self.source_samples_dir = source_dir
        self.logger.info(f"Set source samples directory: {source_dir}")
    
    def load_edge_cases(self, boundary_mapper=None, n_samples=5):
        """
        Load edge cases from a boundary mapper or the source directory
        
        Parameters:
        -----------
        boundary_mapper : AttractorBoundaryMapper, optional
            Boundary mapper to extract edge cases from
        n_samples : int
            Number of edge cases to load
            
        Returns:
        --------
        list
            List of FalseAttractorSample objects
        """
        if boundary_mapper is not None:
            # Extract edge cases from boundary mapper
            self.logger.info(f"Extracting {n_samples} edge cases from boundary mapper")
            return boundary_mapper.identify_edge_cases(n_samples=n_samples)
        elif self.source_samples_dir is not None:
            # Load samples from source directory
            self.logger.info(f"Loading edge cases from source directory: {self.source_samples_dir}")
            all_samples = self.load_samples(self.source_samples_dir)
            
            if not all_samples:
                self.logger.warning("No samples found in source directory")
                return []
                
            # Sort by CCDI to find edge cases
            all_samples.sort(key=lambda x: abs(x.ccdi - self.ccdi_threshold))
            
            # Return top N samples
            return all_samples[:min(n_samples, len(all_samples))]
        else:
            self.logger.warning("No source for edge cases specified")
            return []
    
    def run_nudging_experiments(self, edge_cases=None, nudge_times=None, 
                              nudge_amplitudes=None, nudge_types=None,
                              parallel=False):
        """
        Run nudging experiments on edge cases
        
        Parameters:
        -----------
        edge_cases : list, optional
            List of FalseAttractorSample objects to use
        nudge_times : list, optional
            List of time steps for nudging
        nudge_amplitudes : list, optional
            List of nudge amplitudes to test
        nudge_types : list, optional
            List of nudge types to apply
        parallel : bool
            Whether to use multiprocessing
            
        Returns:
        --------
        pd.DataFrame
            Results DataFrame
        """
        # Use provided parameters or defaults
        if nudge_times is not None:
            self.nudge_times = nudge_times
        if nudge_amplitudes is not None:
            self.nudge_amplitudes = nudge_amplitudes
        if nudge_types is not None:
            self.nudge_types = nudge_types
            
        # Load edge cases if not provided
        if edge_cases is None or len(edge_cases) == 0:
            edge_cases = self.load_edge_cases()
            
        if not edge_cases:
            self.logger.error("No edge cases available for nudging experiments")
            return None
            
        self.logger.info(f"Running nudging experiments on {len(edge_cases)} edge cases")
        self.logger.info(f"Testing {len(self.nudge_types)} nudge types at {len(self.nudge_times)} time points " +
                        f"with {len(self.nudge_amplitudes)} amplitudes")
        
        # Create parameter combinations
        param_grid = [
            (sample, nudge_type, nudge_time, nudge_amp)
            for sample in edge_cases
            for nudge_type in self.nudge_types
            for nudge_time in self.nudge_times
            for nudge_amp in self.nudge_amplitudes
        ]
        
        total_runs = len(param_grid)
        self.logger.info(f"Nudging experiments will perform {total_runs} total simulations")
        
        # Initialize results storage
        results_list = []
        self.samples = []
        
        # Run the experiments
        if parallel:
            self.logger.info("Using parallel processing")
            
            # Configure process pool
            num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
            self.logger.info(f"Using {num_cores} cores")
            
            # Create a pool of workers
            with mp.Pool(processes=num_cores) as pool:
                # Process parameters in parallel
                results = []
                for i, (sample, nudge_type, nudge_time, nudge_amp) in enumerate(param_grid):
                    # Create a run ID
                    run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_{nudge_type}_{nudge_time}_{nudge_amp:.4f}"
                    
                    # Submit job to pool
                    result = pool.apply_async(
                        self._run_single_nudge_experiment, 
                        (sample, nudge_type, nudge_time, nudge_amp, run_id)
                    )
                    results.append(result)
                
                # Collect results as they complete
                for i, result in enumerate(tqdm(results, desc="Nudging Experiments")):
                    try:
                        nudged_sample, result_dict = result.get()
                        self.samples.append(nudged_sample)
                        results_list.append(result_dict)
                        
                        # Save ongoing results
                        if (i + 1) % 5 == 0 or (i + 1) == len(results):
                            self._save_interim_results(results_list)
                    except Exception as e:
                        self.logger.error(f"Error in parallel execution: {e}")
        else:
            self.logger.info("Using sequential processing")
            
            # Process parameters sequentially
            for i, (sample, nudge_type, nudge_time, nudge_amp) in enumerate(tqdm(param_grid, desc="Nudging Experiments")):
                # Create a run ID
                run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_{nudge_type}_{nudge_time}_{nudge_amp:.4f}"
                
                try:
                    nudged_sample, result_dict = self._run_single_nudge_experiment(
                        sample, nudge_type, nudge_time, nudge_amp, run_id
                    )
                    self.samples.append(nudged_sample)
                    results_list.append(result_dict)
                except Exception as e:
                    self.logger.error(f"Error in nudge experiment: {e}")
                    
                # Save ongoing results
                if (i + 1) % 5 == 0 or (i + 1) == len(param_grid):
                    self._save_interim_results(results_list)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        # Generate visualizations
        self._generate_nudging_visualizations()
        
        # Save final results
        self.save_results(self.results_df, f"nudging_results_{self.run_id}.csv")
        
        self.logger.info("Nudging experiments completed successfully")
        return self.results_df
    
    def _run_single_nudge_experiment(self, sample, nudge_type, nudge_time, nudge_amplitude, run_id=None):
        """
        Run a single nudging experiment
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Sample to apply nudging to
        nudge_type : str
            Type of nudge to apply
        nudge_time : int
            Time step at which to apply nudge
        nudge_amplitude : float
            Amplitude of the nudge
        run_id : str, optional
            Identifier for this run
            
        Returns:
        --------
        tuple
            (FalseAttractorSample, dict) - Sample object and result dictionary
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_{nudge_type}_{nudge_time}_{nudge_amplitude:.4f}"
            
        # Create output directory
        run_dir = os.path.join(self.output_dir, "samples", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Initialize experiment with sample parameters
        exp = self._setup_experiment_from_sample(sample)
        
        # Run simulation until the nudge time
        self._run_until_nudge_time(exp, sample, nudge_time)
        
        # Apply nudge
        self._apply_nudge(exp, sample, nudge_type, nudge_amplitude, nudge_time)
        
        # Complete simulation
        self._complete_simulation(exp, sample)
        
        # Extract results
        nudged_sample, result_dict = self._extract_results(exp, sample, nudge_type, nudge_time, nudge_amplitude, run_id)
        
        # Save visualizations
        self._visualize_nudge_experiment(exp, sample, nudged_sample, nudge_time, nudge_type, run_dir)
        
        # Save sample to disk
        nudged_sample.save(run_dir)
        
        return nudged_sample, result_dict
    
    def _setup_experiment_from_sample(self, sample):
        """Initialize an experiment from a sample's parameters"""
        # Create experiment with sample parameters
        exp = RCFTExperiment(
            memory_strength=sample.alpha,
            coupling_strength=self.beta,
            memory_decay=sample.gamma
        )
        
        # Initialize with the same initial state
        exp.state = sample.initial_state.copy()
        exp.memory = sample.initial_state.copy()
        exp.initial_state = sample.initial_state.copy()
        
        # Initialize history and metrics
        exp.history = [exp.state.copy()]
        exp.memory_history = [exp.memory.copy()]
        exp._calculate_metrics()
        
        return exp
    
    def _run_until_nudge_time(self, exp, sample, nudge_time):
        """Run simulation until the nudge time"""
        # Run first perturbation sequence as in original sample
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        first_perturbation = exp.perturbation_step
        
        # Run until delay
        delay = sample.metrics.get('delay', 10)
        exp.update(steps=delay)
        
        # Apply second perturbation
        exp.apply_perturbation(
            perturbation_type="flip", 
            magnitude=1.0, 
            center=(20, 20),
            radius=10
        )
        second_perturbation = len(exp.history) - 1
        
        # Let system evolve until nudge time, adjusting for perturbation steps
        steps_to_run = nudge_time - (second_perturbation + 1)
        if steps_to_run > 0:
            exp.update(steps=steps_to_run)
    
    def _apply_nudge(self, exp, sample, nudge_type, nudge_amplitude, nudge_time):
        """Apply a nudge to the experimental state"""
        # Extract current state and size
        current_state = exp.state.copy()
        size = current_state.shape[0]
        
        # Create nudge profile based on type
        if nudge_type == 'uniform':
            # Uniform coherence pulse
            nudge_profile = np.ones_like(current_state) * nudge_amplitude
            
        elif nudge_type == 'patch':
            # Inject a patch of the original pattern (25% centered)
            nudge_profile = np.zeros_like(current_state)
            center = size // 2
            radius = size // 4
            
            # Create circular mask
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y)
            mask = ((X - center)**2 + (Y - center)**2 <= radius**2)
            
            # Apply patch of original pattern
            nudge_profile[mask] = sample.initial_state[mask] * nudge_amplitude
            
        elif nudge_type == 'echo':
            # Phase-conjugated echo - use time-reversed history
            if len(exp.history) > 2:
                # Find post-perturbation segment
                second_perturbation = sample.perturbation_info['steps'][1]
                post_idx = list(range(second_perturbation, second_perturbation + nudge_time))
                
                # Extract states
                post_states = [exp.history[idx] if idx < len(exp.history) else exp.history[-1] 
                              for idx in post_idx]
                
                # Time-reverse
                reversed_states = post_states[::-1]
                
                # Average and scale
                nudge_profile = np.mean(reversed_states, axis=0) * nudge_amplitude
            else:
                # Fallback if not enough history
                nudge_profile = current_state * nudge_amplitude
        else:
            # Unknown nudge type
            self.logger.warning(f"Unknown nudge type: {nudge_type}, using uniform")
            nudge_profile = np.ones_like(current_state) * nudge_amplitude
        
        # Apply nudge to the state
        exp.state += nudge_profile
        
        # Ensure state stays in valid range [-1, 1]
        exp.state = np.clip(exp.state, -1, 1)
        
        # Record the nudged state
        exp.history.append(exp.state.copy())
        exp.memory_history.append(exp.memory.copy())
        exp._calculate_metrics()
    
    def _complete_simulation(self, exp, sample):
        """Complete the simulation after nudging"""
        # Determine remaining steps to match original sample length
        target_length = len(sample.recovery_trajectory['step'])
        remaining_steps = max(0, target_length - len(exp.history))
        
        # Update for remaining steps
        if remaining_steps > 0:
            exp.update(steps=remaining_steps)
    
    def _extract_results(self, exp, sample, nudge_type, nudge_time, nudge_amplitude, run_id):
        """Extract results from the nudged experiment"""
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
            sample.perturbation_info['steps']
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
            'nudge_type': nudge_type,
            'nudge_time': nudge_time,
            'nudge_amplitude': nudge_amplitude,
            'original_ccdi': sample.ccdi,
            'original_correlation': sample.metrics['final_correlation'],
            'ccdi_delta': ccdi - sample.ccdi,
            'correlation_delta': final_correlation - sample.metrics['final_correlation']
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
        
        # Create nudged sample
        nudged_sample = FalseAttractorSample(
            alpha=sample.alpha,
            gamma=sample.gamma,
            initial_state=sample.initial_state,
            final_state=final_state,
            perturbation_info=sample.perturbation_info,
            recovery_trajectory=recovery_trajectory,
            metrics=metrics
        )
        
        # Create result dictionary
        result_dict = {
            'alpha': sample.alpha,
            'gamma': sample.gamma,
            'nudge_type': nudge_type,
            'nudge_time': nudge_time,
            'nudge_amplitude': nudge_amplitude,
            'original_correlation': sample.metrics['final_correlation'],
            'nudged_correlation': final_correlation,
            'correlation_delta': final_correlation - sample.metrics['final_correlation'],
            'original_ccdi': sample.ccdi,
            'nudged_ccdi': ccdi,
            'ccdi_delta': ccdi - sample.ccdi,
            'original_recovery_class': sample.recovery_class,
            'nudged_recovery_class': recovery_class,
            'is_improved': final_correlation > sample.metrics['final_correlation'],
            'residual_norm': np.linalg.norm(final_state - sample.initial_state),
            'run_id': run_id
        }
        
        return nudged_sample, result_dict
    
    def _visualize_nudge_experiment(self, exp, original_sample, nudged_sample, 
                                  nudge_time, nudge_type, output_dir):
        """Create side-by-side visualizations for the nudging experiment"""
        # Extract trajectories
        original_corr = original_sample.recovery_trajectory['correlation']
        nudged_corr = nudged_sample.recovery_trajectory['correlation']
        
        original_ccdi = original_sample.recovery_trajectory.get('ccdi', 
                    [original_sample.recovery_trajectory['coherence'][i] - original_sample.recovery_trajectory['correlation'][i]
                     for i in range(len(original_sample.recovery_trajectory['correlation']))])
        
        nudged_ccdi = nudged_sample.recovery_trajectory['ccdi']
        
        # Plot recovery comparison
        plt.figure(figsize=(12, 10))
        
        # Correlation comparison
        plt.subplot(2, 1, 1)
        plt.plot(original_corr, label='Original', color='blue')
        plt.plot(nudged_corr, label='Nudged', color='red', linestyle='--')
        
        # Mark perturbations and nudge
        for i, step in enumerate(original_sample.perturbation_info['steps']):
            plt.axvline(x=step, color='gray', linestyle='--', 
                      label=f'Perturbation {i+1}' if i == 0 else None)
            
        plt.axvline(x=nudge_time, color='green', linestyle='-', 
                  label=f'Nudge ({nudge_type}, t={nudge_time})')
        
        plt.title('Correlation Recovery Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Correlation with Initial State')
        plt.grid(True)
        plt.legend()
        
        # CCDI comparison
        plt.subplot(2, 1, 2)
        plt.plot(original_ccdi, label='Original', color='blue')
        plt.plot(nudged_ccdi, label='Nudged', color='red', linestyle='--')
        
        # Mark perturbations and nudge
        for i, step in enumerate(original_sample.perturbation_info['steps']):
            plt.axvline(x=step, color='gray', linestyle='--')
            
        plt.axvline(x=nudge_time, color='green', linestyle='-', 
                  label=f'Nudge ({nudge_type}, t={nudge_time})')
        
        # Mark anomaly threshold
        plt.axhline(y=self.ccdi_threshold, color='black', linestyle=':', label='Anomaly Threshold')
        
        plt.title('CCDI Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('CCDI')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recovery_comparison.png"), dpi=300)
        plt.close()
        
        # Create state comparison visualization
        plt.figure(figsize=(12, 4))
        
        # Original final state
        plt.subplot(1, 3, 1)
        plt.imshow(original_sample.final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Original Final State")
        plt.axis('off')
        
        # Nudged final state
        plt.subplot(1, 3, 2)
        plt.imshow(nudged_sample.final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Nudged Final State")
        plt.axis('off')
        
        # Initial state (target)
        plt.subplot(1, 3, 3)
        plt.imshow(original_sample.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Initial State (Target)")
        plt.axis('off')
        
        plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "state_comparison.png"), dpi=300)
        plt.close()
        
        # Create difference visualization
        plt.figure(figsize=(12, 4))
        
        # Original residual
        plt.subplot(1, 3, 1)
        plt.imshow(original_sample.residual, cmap='RdBu', vmin=-1, vmax=1)
        plt.title("Original Residual")
        plt.axis('off')
        
        # Nudged residual
        nudged_residual = nudged_sample.final_state - original_sample.initial_state
        plt.subplot(1, 3, 2)
        plt.imshow(nudged_residual, cmap='RdBu', vmin=-1, vmax=1)
        plt.title("Nudged Residual")
        plt.axis('off')
        
        # Difference between residuals
        residual_diff = nudged_residual - original_sample.residual
        plt.subplot(1, 3, 3)
        plt.imshow(residual_diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        plt.title("Residual Difference")
        plt.axis('off')
        
        plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residual_comparison.png"), dpi=300)
        plt.close()
    
    def _save_interim_results(self, results_list):
        """Save interim results during processing"""
        interim_df = pd.DataFrame(results_list)
        
        # Save DataFrame to CSV
        interim_path = os.path.join(self.output_dir, f"interim_results_{self.run_id}.csv")
        interim_df.to_csv(interim_path, index=False)
        
        # Save samples metadata
        samples_meta = [sample.to_dict() for sample in self.samples]
        meta_path = os.path.join(self.output_dir, f"samples_metadata_{self.run_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(samples_meta, f, indent=2)
            
        self.logger.info(f"Saved interim results: {len(results_list)} runs")
    
    def _generate_nudging_visualizations(self):
        """Generate visualizations for nudging results"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for visualizations")
            return
            
        # Create directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Nudge effectiveness by type and time
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate mean correlation improvement by type and time
            pivot_data = self.results_df.pivot_table(
                index='nudge_type',
                columns='nudge_time',
                values='correlation_delta',
                aggfunc='mean'
            )
            
            # Create heatmap
            ax = sns.heatmap(
                pivot_data, 
                cmap='RdYlGn',
                annot=True, 
                fmt=".3f",
                linewidths=.5,
                center=0,
                cbar_kws={'label': 'Mean Correlation Improvement'}
            )
            
            # Set labels
            plt.title("Nudge Effectiveness by Type and Time")
            plt.xlabel("Nudge Time Step")
            plt.ylabel("Nudge Type")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"nudge_type_time_heatmap_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating nudge type/time heatmap: {e}")
            
        # Nudge effectiveness by type and amplitude
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate mean correlation improvement by type and amplitude
            pivot_data = self.results_df.pivot_table(
                index='nudge_type',
                columns='nudge_amplitude',
                values='correlation_delta',
                aggfunc='mean'
            )
            
            # Create heatmap
            ax = sns.heatmap(
                pivot_data, 
                cmap='RdYlGn',
                annot=True, 
                fmt=".3f",
                linewidths=.5,
                center=0,
                cbar_kws={'label': 'Mean Correlation Improvement'}
            )
            
            # Set labels
            plt.title("Nudge Effectiveness by Type and Amplitude")
            plt.xlabel("Nudge Amplitude")
            plt.ylabel("Nudge Type")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"nudge_type_amp_heatmap_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating nudge type/amplitude heatmap: {e}")
            
        # Box plot of correlation improvement by nudge type
        try:
            plt.figure(figsize=(12, 8))
            
            # Create box plot
            sns.boxplot(x='nudge_type', y='correlation_delta', data=self.results_df)
            
            # Set labels
            plt.title("Correlation Improvement by Nudge Type")
            plt.xlabel("Nudge Type")
            plt.ylabel("Correlation Improvement (Δ)")
            plt.axhline(y=0, color='black', linestyle='--')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"nudge_type_boxplot_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating nudge type boxplot: {e}")
            
        # Scatter plot of original vs nudged correlation
        try:
            plt.figure(figsize=(10, 10))
            
            # Create scatter plot
            g = sns.scatterplot(
                x='original_correlation', 
                y='nudged_correlation', 
                hue='nudge_type',
                style='nudge_type',
                s=100,
                data=self.results_df
            )
            
            # Add diagonal line
            xlim = plt.xlim()
            ylim = plt.ylim()
            lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
            plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            plt.xlim(lims)
            plt.ylim(lims)
            
            # Set labels
            plt.title("Original vs Nudged Correlation")
            plt.xlabel("Original Correlation")
            plt.ylabel("Nudged Correlation")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            plt.legend(title="Nudge Type")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"correlation_scatter_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating correlation scatter plot: {e}")
            
        # Summary bar chart - success rate by nudge type
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate success rate (improvement in correlation)
            success_rate = self.results_df.groupby('nudge_type')['is_improved'].mean() * 100
            
            # Create bar chart
            ax = success_rate.plot(kind='bar', color='lightgreen')
            
            # Add value labels
            for i, v in enumerate(success_rate):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            # Set labels
            plt.title("Success Rate by Nudge Type")
            plt.xlabel("Nudge Type")
            plt.ylabel("Success Rate (%)")
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"success_rate_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating success rate bar chart: {e}")

    def summarize_nudging_results(self):
        """Print a summary of the nudging results"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for summary")
            return
            
        # Count successful nudges
        n_improved = self.results_df['is_improved'].sum()
        n_total = len(self.results_df)
        
        print("\nNudging Experiments Summary:")
        print(f"Total Experiments: {n_total}")
        print(f"Successful Nudges: {n_improved} ({n_improved/n_total*100:.1f}%)")
        
        # Success by nudge type
        type_success = self.results_df.groupby('nudge_type')['is_improved'].mean() * 100
        print("\nSuccess Rate by Nudge Type:")
        for nudge_type, success_rate in type_success.items():
            print(f"  {nudge_type}: {success_rate:.1f}%")
            
        # Success by nudge time
        time_success = self.results_df.groupby('nudge_time')['is_improved'].mean() * 100
        print("\nSuccess Rate by Nudge Time:")
        for nudge_time, success_rate in time_success.items():
            print(f"  t={nudge_time}: {success_rate:.1f}%")
            
        # Success by nudge amplitude
        amp_success = self.results_df.groupby('nudge_amplitude')['is_improved'].mean() * 100
        print("\nSuccess Rate by Nudge Amplitude:")
        for nudge_amp, success_rate in amp_success.items():
            print(f"  η={nudge_amp}: {success_rate:.1f}%")
            
        # Find best configuration
        best_rows = self.results_df.sort_values('correlation_delta', ascending=False).head(3)
        print("\nTop 3 Nudging Configurations:")
        for i, (_, row) in enumerate(best_rows.iterrows()):
            print(f"  {i+1}. Type: {row['nudge_type']}, Time: {row['nudge_time']}, " +
                 f"Amplitude: {row['nudge_amplitude']}, Improvement: {row['correlation_delta']:.4f}")
            
        # Find worst configuration
        worst_rows = self.results_df.sort_values('correlation_delta').head(3)
        print("\nBottom 3 Nudging Configurations:")
        for i, (_, row) in enumerate(worst_rows.iterrows()):
            print(f"  {i+1}. Type: {row['nudge_type']}, Time: {row['nudge_time']}, " +
                 f"Amplitude: {row['nudge_amplitude']}, Change: {row['correlation_delta']:.4f}")


# Function to import for compatibility with previous code
def compute_ccdi(correlation, coherence):
    """Compute Coherence-Correlation Divergence Index (CCDI)"""
    return coherence - correlation

def classify_recovery(correlation_curve, perturbation_steps):
    """Classify recovery trajectory as true, false, or oscillatory"""
    # Extract correlation after second perturbation
    second_perturbation = perturbation_steps[1]
    post_perturb = correlation_curve[second_perturbation:]
    
    if len(post_perturb) < 3:
        return "unknown"  # Not enough data
    
    # Find peaks and valleys
    peaks = []
    valleys = []
    
    for i in range(1, len(post_perturb)-1):
        if post_perturb[i] > post_perturb[i-1] and post_perturb[i] > post_perturb[i+1]:
            peaks.append(i)
        if post_perturb[i] < post_perturb[i-1] and post_perturb[i] < post_perturb[i+1]:
            valleys.append(i)
    
    # Analyze pattern
    if len(peaks) == 0:
        # No peaks - check if monotonically increasing or decreasing
        if post_perturb[-1] > post_perturb[0]:
            return "true"  # Monotonic recovery
        else:
            return "false"  # Monotonic decline
    
    elif len(peaks) == 1:
        # One peak - check for peak-then-decline
        peak_idx = peaks[0]
        if peak_idx < len(post_perturb) // 2 and post_perturb[-1] < post_perturb[peak_idx]:
            return "false"  # Peak-then-decline
        else:
            return "true"  # Still recovering
    
    else:
        # Multiple peaks and valleys - oscillatory
        return "oscillatory"


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RCFT Nudging Experiments')
    parser.add_argument('--output_dir', type=str, default='phase3_results/nudging_experiments',
                        help='Output directory for results')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Source directory for false attractor samples')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of edge cases to analyze')
    parser.add_argument('--nudge_times', type=str, default='30,40,50',
                        help='Comma-separated list of nudge time steps')
    parser.add_argument('--nudge_amplitudes', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated list of nudge amplitudes')
    parser.add_argument('--nudge_types', type=str, default='uniform,patch,echo',
                        help='Comma-separated list of nudge types')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = NudgingController(output_dir=args.output_dir)
    
    # Set source samples directory
    if args.source_dir:
        controller.set_source_samples(args.source_dir)
    
    # Parse parameters
    nudge_times = [int(t) for t in args.nudge_times.split(',')]
    nudge_amplitudes = [float(a) for a in args.nudge_amplitudes.split(',')]
    nudge_types = args.nudge_types.split(',')
    
    # Load edge cases
    edge_cases = controller.load_edge_cases(n_samples=args.n_samples)
    
    if not edge_cases:
        print("Error: No edge cases found. Please provide a valid source directory.")
        exit(1)
    
    # Run nudging experiments
    results = controller.run_nudging_experiments(
        edge_cases=edge_cases,
        nudge_times=nudge_times,
        nudge_amplitudes=nudge_amplitudes,
        nudge_types=nudge_types,
        parallel=args.parallel
    )
    
    # Print summary
    controller.summarize_nudging_results()
    
    print(f"\nNudging experiments complete. Results saved to {args.output_dir}")
