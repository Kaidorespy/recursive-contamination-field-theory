"""
Temporal Coherence Reinforcement module for Phase V.
This module explores memory reinforcement through delayed feedback and echo effects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from .module_base import PhaseVModule
from .utils.metrics import IdentityMetrics

class TemporalCoherenceReinforcer(PhaseVModule):
    """Module for testing memory reinforcement through temporal feedback."""
    
    def __init__(self, output_dir=None, buffer_size=10, reactivation_strength=0.15,
                reactivation_schedule="periodic", reactivation_interval=10,
                **kwargs):
        """
        Initialize the Temporal Coherence Reinforcer.
        
        Args:
            output_dir: Directory for output files
            buffer_size: Size of the memory reactivation buffer
            reactivation_strength: Strength of memory reactivation (ε)
            reactivation_schedule: Schedule type ("periodic", "decay", "adaptive")
            reactivation_interval: Steps between reactivations for periodic schedule
            **kwargs: Additional parameters
        """
        super().__init__(output_dir=output_dir, buffer_size=buffer_size,
                       reactivation_strength=reactivation_strength,
                       reactivation_schedule=reactivation_schedule,
                       reactivation_interval=reactivation_interval, **kwargs)
        
        # Memory buffer for reactivation
        self.memory_buffer = []
        self.buffer_size = buffer_size
        
        # Reactivation parameters
        self.reactivation_strength = reactivation_strength
        self.reactivation_schedule = reactivation_schedule
        self.reactivation_interval = reactivation_interval
        
        # Tracking
        self.reactivation_steps = []
        
    def _parse_schedule(self, step, max_steps):
        """
        Determine if reactivation should occur at this step based on schedule.
        
        Args:
            step: Current step
            max_steps: Maximum number of steps
            
        Returns:
            Boolean indicating whether to reactivate and strength modifier
        """
        if self.reactivation_schedule == "periodic":
            # Reactivate every N steps
            return step > 0 and step % self.reactivation_interval == 0, 1.0
            
        elif self.reactivation_schedule == "decay":
            # Reactivate with decreasing frequency
            interval = max(1, int(self.reactivation_interval * (1.0 + step / max_steps)))
            return step > 0 and step % interval == 0, 1.0
            
        elif self.reactivation_schedule == "adaptive":
            # For now, just use periodic schedule
            # In a real implementation, this would adapt based on recovery metrics
            return step > 0 and step % self.reactivation_interval == 0, 1.0
            
        else:
            return False, 0.0
            
    def add_to_buffer(self, state):
        """
        Add a state to the memory buffer.
        
        Args:
            state: State to add
        """
        # Add to buffer, maintaining size limit
        self.memory_buffer.append(state.copy())
        
        if len(self.memory_buffer) > self.buffer_size:
            self.memory_buffer.pop(0)  # Remove oldest
            
    def get_reactivation_state(self):
        """
        Get reactivation state from buffer.
        
        Returns:
            Reactivation state or None if buffer empty
        """
        if not self.memory_buffer:
            return None
            
        # For now, simple implementation: use the oldest state
        # Could be extended to use weighted combinations, etc.
        return self.memory_buffer[0].copy()
        
    def run(self, experiment, n_steps=200, perturb_step=50, perturbation_type="flip",
           perturbation_magnitude=1.0, run_baseline=True):
        """
        Run the temporal coherence reinforcement experiment.
        
        Args:
            experiment: RCFT experiment instance
            n_steps: Number of steps to run
            perturb_step: Step at which to apply perturbation
            perturbation_type: Type of perturbation
            perturbation_magnitude: Magnitude of perturbation
            run_baseline: Whether to run a baseline without reinforcement
            
        Returns:
            Results dictionary
        """
        # Initialize results tracking
        self.results = {
            'reinforced': {'steps': [], 'states': [], 'metrics': {}},
            'baseline': {'steps': [], 'states': [], 'metrics': {}}
        }
        
        # Define common metrics to track
        metric_names = ['correlation', 'coherence', 'ccdi', 
                      'spectral_entropy', 'spatial_entropy']
        
        for mode in ['reinforced', 'baseline']:
            for metric in metric_names:
                self.results[mode]['metrics'][metric] = []
        
        # Run baseline experiment first if requested
        if run_baseline:
            print("Running baseline experiment (no reinforcement)...")
            baseline_results = self._run_baseline(
                experiment, n_steps, perturb_step, 
                perturbation_type, perturbation_magnitude)
                
            # Update results
            self.results['baseline'] = baseline_results
            
        # Run reinforced experiment
        print("Running reinforced experiment...")
        reinforced_results = self._run_reinforced(
            experiment, n_steps, perturb_step, 
            perturbation_type, perturbation_magnitude)
            
        # Update results
        self.results['reinforced'] = reinforced_results
        
        # Compare results
        self._compare_results()
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        return self.results
        
    def _run_baseline(self, experiment, n_steps, perturb_step, 
                     perturbation_type, perturbation_magnitude):
        """
        Run baseline experiment without reinforcement.
        
        Args:
            experiment: RCFT experiment instance
            n_steps: Number of steps to run
            perturb_step: Step at which to apply perturbation
            perturbation_type: Type of perturbation
            perturbation_magnitude: Magnitude of perturbation
            
        Returns:
            Baseline results dictionary
        """
        # Save original state
        original_state = experiment.state.copy()
        
        # Initialize results
        results = {
            'steps': [],
            'states': [original_state.copy()],
            'metrics': {
                'correlation': [1.0],  # Initial correlation is 1
                'coherence': [IdentityMetrics.coherence(original_state)],
                'ccdi': [0.0],  # Initial CCDI is 0
                'spectral_entropy': [IdentityMetrics.spectral_entropy(original_state)],
                'spatial_entropy': [IdentityMetrics.spatial_entropy(original_state)]
            }
        }
        
        # Create identity trace
        baseline_trace = self.track_identity(experiment, label="Baseline")
        
        # Run simulation
        for step in range(1, n_steps + 1):
            # Apply perturbation if at perturb_step
            if step == perturb_step:
                experiment.apply_perturbation(
                    perturbation_type=perturbation_type,
                    magnitude=perturbation_magnitude
                )
                
                # Record perturbation step
                results['perturbation_step'] = step
                
            # Update experiment
            experiment.update(steps=1)
            
            # Record state
            current_state = experiment.state.copy()
            results['states'].append(current_state)
            results['steps'].append(step)
            
            # Update identity trace
            self.update_identity(experiment, trace_index=0)
            
            # Calculate metrics
            correlation = IdentityMetrics.correlation(current_state, original_state)
            coherence = IdentityMetrics.coherence(current_state)
            ccdi = coherence - correlation
            spectral_entropy = IdentityMetrics.spectral_entropy(current_state)
            spatial_entropy = IdentityMetrics.spatial_entropy(current_state)
            
            # Record metrics
            results['metrics']['correlation'].append(correlation)
            results['metrics']['coherence'].append(coherence)
            results['metrics']['ccdi'].append(ccdi)
            results['metrics']['spectral_entropy'].append(spectral_entropy)
            results['metrics']['spatial_entropy'].append(spatial_entropy)
            
        # Calculate recovery quality
        if 'perturbation_step' in results and perturb_step < n_steps:
            perturbed_state = results['states'][perturb_step]
            final_state = results['states'][-1]
            
            recovery_quality = IdentityMetrics.recovery_quality(
                original_state, perturbed_state, final_state)
                
            results['recovery_quality'] = recovery_quality
            
        return results
        
    def _run_reinforced(self, experiment, n_steps, perturb_step, 
                      perturbation_type, perturbation_magnitude):
        """
        Run reinforced experiment with temporal coherence.
        
        Args:
            experiment: RCFT experiment instance
            n_steps: Number of steps to run
            perturb_step: Step at which to apply perturbation
            perturbation_type: Type of perturbation
            perturbation_magnitude: Magnitude of perturbation
            
        Returns:
            Reinforced results dictionary
        """
        # Reset to original state
        original_state = experiment.initial_state.copy()
        experiment.state = original_state.copy()
        experiment.memory = original_state.copy()
        
        # Clear memory buffer
        self.memory_buffer = []
        self.reactivation_steps = []
        
        # Initialize results
        results = {
            'steps': [],
            'states': [original_state.copy()],
            'metrics': {
                'correlation': [1.0],  # Initial correlation is 1
                'coherence': [IdentityMetrics.coherence(original_state)],
                'ccdi': [0.0],  # Initial CCDI is 0
                'spectral_entropy': [IdentityMetrics.spectral_entropy(original_state)],
                'spatial_entropy': [IdentityMetrics.spatial_entropy(original_state)]
            },
            'reactivation_steps': []
        }
        
        # Create identity trace
        reinforced_trace = self.track_identity(experiment, label="Reinforced")
        
        # Run simulation
        for step in range(1, n_steps + 1):
            # Add current state to buffer before perturbation
            if step < perturb_step:
                self.add_to_buffer(experiment.state.copy())
                
            # Apply perturbation if at perturb_step
            if step == perturb_step:
                experiment.apply_perturbation(
                    perturbation_type=perturbation_type,
                    magnitude=perturbation_magnitude
                )
                
                # Record perturbation step
                results['perturbation_step'] = step
                
            # Check if reactivation should occur
            should_reactivate, strength_mod = self._parse_schedule(step, n_steps)
            
            if should_reactivate and step > perturb_step:
                # Get reactivation state
                reactivation_state = self.get_reactivation_state()
                
                if reactivation_state is not None:
                    # Apply reactivation: F(t+1) = RCFT(F(t)) + ε * F(t - Δ)
                    # Note: We're manually injecting here since there's no direct RCFT method
                    
                    # First compute standard RCFT update
                    experiment.update(steps=1)
                    
                    # Then add the reactivation term
                    reactivation_strength = self.reactivation_strength * strength_mod
                    experiment.state += reactivation_strength * reactivation_state
                    
                    # Record reactivation
                    self.reactivation_steps.append(step)
                    results['reactivation_steps'].append(step)
                else:
                    # No reactivation state available, just update normally
                    experiment.update(steps=1)
            else:
                # Normal update
                experiment.update(steps=1)
                
            # Record state
            current_state = experiment.state.copy()
            results['states'].append(current_state)
            results['steps'].append(step)
            
            # Update identity trace
            self.update_identity(experiment, trace_index=1)
            
            # Calculate metrics
            correlation = IdentityMetrics.correlation(current_state, original_state)
            coherence = IdentityMetrics.coherence(current_state)
            ccdi = coherence - correlation
            spectral_entropy = IdentityMetrics.spectral_entropy(current_state)
            spatial_entropy = IdentityMetrics.spatial_entropy(current_state)
            
            # Record metrics
            results['metrics']['correlation'].append(correlation)
            results['metrics']['coherence'].append(coherence)
            results['metrics']['ccdi'].append(ccdi)
            results['metrics']['spectral_entropy'].append(spectral_entropy)
            results['metrics']['spatial_entropy'].append(spatial_entropy)
            
        # Calculate recovery quality
        if 'perturbation_step' in results and perturb_step < n_steps:
            perturbed_state = results['states'][perturb_step]
            final_state = results['states'][-1]
            
            recovery_quality = IdentityMetrics.recovery_quality(
                original_state, perturbed_state, final_state)
                
            results['recovery_quality'] = recovery_quality
            
        return results
        
    def _compare_results(self):
        """Compare reinforced and baseline results."""
        # Skip if no baseline
        if 'baseline' not in self.results or not self.results['baseline']['steps']:
            return
            
        # Calculate comparative metrics
        baseline_results = self.results['baseline']
        reinforced_results = self.results['reinforced']
        
        # Temporal coherence
        if 'states' in baseline_results and 'states' in reinforced_results:
            baseline_tc, _ = IdentityMetrics.temporal_coherence(baseline_results['states'])
            reinforced_tc, _ = IdentityMetrics.temporal_coherence(reinforced_results['states'])
            
            self.metrics['temporal_coherence'] = {
                'baseline': baseline_tc,
                'reinforced': reinforced_tc,
                'improvement': reinforced_tc - baseline_tc
            }
            
        # Recovery quality
        if 'recovery_quality' in baseline_results and 'recovery_quality' in reinforced_results:
            baseline_rq = baseline_results['recovery_quality']
            reinforced_rq = reinforced_results['recovery_quality']
            
            self.metrics['recovery_quality'] = {
                'baseline': baseline_rq,
                'reinforced': reinforced_rq,
                'improvement': reinforced_rq - baseline_rq
            }
            
        # Final correlation
        if ('metrics' in baseline_results and 'correlation' in baseline_results['metrics'] and
            'metrics' in reinforced_results and 'correlation' in reinforced_results['metrics']):
            
            baseline_corr = baseline_results['metrics']['correlation'][-1]
            reinforced_corr = reinforced_results['metrics']['correlation'][-1]
            
            self.metrics['final_correlation'] = {
                'baseline': baseline_corr,
                'reinforced': reinforced_corr,
                'improvement': reinforced_corr - baseline_corr
            }
            
        # Calculate stability gain
        if ('metrics' in baseline_results and 'correlation' in baseline_results['metrics'] and
            'metrics' in reinforced_results and 'correlation' in reinforced_results['metrics']):
            
            # Get correlation curves after perturbation
            if 'perturbation_step' in baseline_results:
                perturb_step = baseline_results['perturbation_step']
                
                baseline_curve = baseline_results['metrics']['correlation'][perturb_step:]
                reinforced_curve = reinforced_results['metrics']['correlation'][perturb_step:]
                
                # Calculate stability as variance of the curve (lower is more stable)
                baseline_var = np.var(baseline_curve)
                reinforced_var = np.var(reinforced_curve)
                
                # Stability gain (negative variance change means more stable)
                stability_gain = baseline_var - reinforced_var
                
                self.metrics['stability'] = {
                    'baseline_variance': baseline_var,
                    'reinforced_variance': reinforced_var,
                    'stability_gain': stability_gain
                }
                
    def _create_visualizations(self):
        """Create visualizations of results."""
        # Original, perturbed and final states
        if 'reinforced' in self.results and 'states' in self.results['reinforced']:
            states = self.results['reinforced']['states']
            
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                
                # Extract key frames
                key_states = [
                    states[0],                   # Original
                    states[perturb_step],        # Perturbed
                    states[-1]                   # Final
                ]
                
                # Create visualization
                titles = ["Original", "Perturbed", "Final"]
                self.visualizer.visualize_state_sequence(
                    key_states, titles=titles, title="Key States",
                    save_path=os.path.join(self.output_dir, "key_states.png")
                )
        
                
        # Metrics over time
        if ('reinforced' in self.results and 'metrics' in self.results['reinforced'] and
        'baseline' in self.results and 'metrics' in self.results['baseline']):
        
            # Compare correlation
            plt.figure(figsize=(10, 6))
            
            baseline_corr = self.results['baseline']['metrics']['correlation']
            reinforced_corr = self.results['reinforced']['metrics']['correlation']
            
            # Ensure steps and metric arrays have same length
            baseline_steps = [0] + self.results['baseline']['steps'] if len(baseline_corr) > len(self.results['baseline']['steps']) else self.results['baseline']['steps']
            reinforced_steps = [0] + self.results['reinforced']['steps'] if len(reinforced_corr) > len(self.results['reinforced']['steps']) else self.results['reinforced']['steps']
            
            # Make sure arrays match in length before plotting
            if len(baseline_steps) > len(baseline_corr):
                baseline_steps = baseline_steps[:len(baseline_corr)]
            elif len(baseline_steps) < len(baseline_corr):
                baseline_corr = baseline_corr[:len(baseline_steps)]
                
            if len(reinforced_steps) > len(reinforced_corr):
                reinforced_steps = reinforced_steps[:len(reinforced_corr)]
            elif len(reinforced_steps) < len(reinforced_corr):
                reinforced_corr = reinforced_corr[:len(reinforced_steps)]
            
            plt.plot(baseline_steps, baseline_corr, 'b-', label='Baseline')
            plt.plot(reinforced_steps, reinforced_corr, 'g-', label='Reinforced')
                
                # Mark perturbation step
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                plt.axvline(x=perturb_step, color='r', linestyle='--', 
                       label='Perturbation')
                
            # Mark reactivation steps
            for step in self.reactivation_steps:
                plt.axvline(x=step, color='m', linestyle=':', alpha=0.5)
                
            plt.axvline(x=self.reactivation_steps[0] if self.reactivation_steps else 0, 
                      color='m', linestyle=':', label='Reactivation')
            
            plt.title('Correlation with Original State')
            plt.xlabel('Step')
            plt.ylabel('Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, "correlation_comparison.png"))
            plt.close()
            
            # Compare coherence
            plt.figure(figsize=(10, 6))
            
            baseline_coh = self.results['baseline']['metrics']['coherence']
            reinforced_coh = self.results['reinforced']['metrics']['coherence']
            
            # Get coherence metrics
            baseline_coh = self.results['baseline']['metrics']['coherence']
            reinforced_coh = self.results['reinforced']['metrics']['coherence']

            # Fix array lengths
            baseline_steps = [0] + self.results['baseline']['steps'] if len(baseline_coh) > len(self.results['baseline']['steps']) else self.results['baseline']['steps']
            reinforced_steps = [0] + self.results['reinforced']['steps'] if len(reinforced_coh) > len(self.results['reinforced']['steps']) else self.results['reinforced']['steps']

            # Make sure arrays match in length before plotting
            if len(baseline_steps) > len(baseline_coh):
                baseline_steps = baseline_steps[:len(baseline_coh)]
            elif len(baseline_steps) < len(baseline_coh):
                baseline_coh = baseline_coh[:len(baseline_steps)]
                
            if len(reinforced_steps) > len(reinforced_coh):
                reinforced_steps = reinforced_steps[:len(reinforced_coh)]
            elif len(reinforced_steps) < len(reinforced_corr):
                reinforced_coh = reinforced_coh[:len(reinforced_steps)]

            # Now plot with fixed arrays
            plt.plot(baseline_steps, baseline_coh, 'b-', label='Baseline')
            plt.plot(reinforced_steps, reinforced_coh, 'g-', label='Reinforced')
            
            # Mark perturbation step
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                plt.axvline(x=perturb_step, color='r', linestyle='--', 
                          label='Perturbation')
                
            # Mark reactivation steps
            for step in self.reactivation_steps:
                plt.axvline(x=step, color='m', linestyle=':', alpha=0.5)
                
            plt.axvline(x=self.reactivation_steps[0] if self.reactivation_steps else 0, 
                      color='m', linestyle=':', label='Reactivation')
            
            plt.title('Coherence')
            plt.xlabel('Step')
            plt.ylabel('Coherence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, "coherence_comparison.png"))
            plt.close()
            
            # Compare CCDI
            plt.figure(figsize=(10, 6))
            
            baseline_ccdi = self.results['baseline']['metrics']['ccdi']
            reinforced_ccdi = self.results['reinforced']['metrics']['ccdi']
            
            # Get coherence metrics
            baseline_coh = self.results['baseline']['metrics']['coherence']
            reinforced_coh = self.results['reinforced']['metrics']['coherence']

            # Fix array lengths
            baseline_steps = [0] + self.results['baseline']['steps'] if len(baseline_coh) > len(self.results['baseline']['steps']) else self.results['baseline']['steps']
            reinforced_steps = [0] + self.results['reinforced']['steps'] if len(reinforced_coh) > len(self.results['reinforced']['steps']) else self.results['reinforced']['steps']

            # Make sure arrays match in length before plotting
            if len(baseline_steps) > len(baseline_coh):
                baseline_steps = baseline_steps[:len(baseline_coh)]
            elif len(baseline_steps) < len(baseline_coh):
                baseline_coh = baseline_coh[:len(baseline_steps)]
                
            if len(reinforced_steps) > len(reinforced_coh):
                reinforced_steps = reinforced_steps[:len(reinforced_coh)]
            elif len(reinforced_steps) < len(reinforced_corr):
                reinforced_coh = reinforced_coh[:len(reinforced_steps)]

            # Now plot with fixed arrays
            plt.plot(baseline_steps, baseline_coh, 'b-', label='Baseline')
            plt.plot(reinforced_steps, reinforced_coh, 'g-', label='Reinforced')
            
            # Mark perturbation step
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                plt.axvline(x=perturb_step, color='r', linestyle='--', 
                          label='Perturbation')
                
            # Mark reactivation steps
            for step in self.reactivation_steps:
                plt.axvline(x=step, color='m', linestyle=':', alpha=0.5)
                
            plt.axvline(x=self.reactivation_steps[0] if self.reactivation_steps else 0, 
                      color='m', linestyle=':', label='Reactivation')
            
            plt.title('Coherence-Correlation Divergence Index (CCDI)')
            plt.xlabel('Step')
            plt.ylabel('CCDI')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, "ccdi_comparison.png"))
            plt.close()
            
        # Visualize identity metrics
        self.visualize_identity_metrics(
            metrics_list=['correlation', 'coherence', 'ccdi'],
            title="Identity Metrics Comparison",
            save_path=os.path.join(self.output_dir, "identity_metrics.png")
        )
        
        # Create summary visualization
        self._create_summary_visualization()
        
    def _create_summary_visualization(self):
        """Create a summary visualization with key findings."""
        plt.figure(figsize=(12, 10))
        
        # Layout
        gs = plt.GridSpec(2, 2)
        
        # Top left: Correlation curves
        ax1 = plt.subplot(gs[0, 0])
        
        if ('reinforced' in self.results and 'metrics' in self.results['reinforced'] and
            'baseline' in self.results and 'metrics' in self.results['baseline']):
            
            baseline_corr = self.results['baseline']['metrics']['correlation']
            reinforced_corr = self.results['reinforced']['metrics']['correlation']
            
            # Get coherence metrics
            baseline_coh = self.results['baseline']['metrics']['coherence']
            reinforced_coh = self.results['reinforced']['metrics']['coherence']

            # Fix array lengths
            baseline_steps = [0] + self.results['baseline']['steps'] if len(baseline_coh) > len(self.results['baseline']['steps']) else self.results['baseline']['steps']
            reinforced_steps = [0] + self.results['reinforced']['steps'] if len(reinforced_coh) > len(self.results['reinforced']['steps']) else self.results['reinforced']['steps']

            # Make sure arrays match in length before plotting
            if len(baseline_steps) > len(baseline_coh):
                baseline_steps = baseline_steps[:len(baseline_coh)]
            elif len(baseline_steps) < len(baseline_coh):
                baseline_coh = baseline_coh[:len(baseline_steps)]
                
            if len(reinforced_steps) > len(reinforced_coh):
                reinforced_steps = reinforced_steps[:len(reinforced_coh)]
            elif len(reinforced_steps) < len(reinforced_corr):
                reinforced_coh = reinforced_coh[:len(reinforced_steps)]

            # Now plot with fixed arrays
            plt.plot(baseline_steps, baseline_coh, 'b-', label='Baseline')
            plt.plot(reinforced_steps, reinforced_coh, 'g-', label='Reinforced')
            
            # Mark perturbation step
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                ax1.axvline(x=perturb_step, color='r', linestyle='--', 
                          label='Perturbation')
                
            # Mark reactivation steps
            for step in self.reactivation_steps:
                ax1.axvline(x=step, color='m', linestyle=':', alpha=0.5)
                
            ax1.axvline(x=self.reactivation_steps[0] if self.reactivation_steps else 0, 
                      color='m', linestyle=':', label='Reactivation')
                      
            ax1.set_title('Correlation with Original State')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
        # Top right: Recovery quality bar chart
        ax2 = plt.subplot(gs[0, 1])
        
        if 'recovery_quality' in self.metrics:
            recovery = self.metrics['recovery_quality']
            
            ax2.bar(['Baseline', 'Reinforced'], 
                  [recovery['baseline'], recovery['reinforced']],
                  color=['blue', 'green'])
                  
            # Add improvement text
            improvement = recovery['improvement']
            ax2.text(1.0, recovery['reinforced'] / 2, 
                   f"+{improvement:.4f}",
                   ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=12)
                   
            ax2.set_title('Recovery Quality')
            ax2.set_ylabel('Quality')
            ax2.grid(axis='y', alpha=0.3)
            
        # Bottom left: CCDI curves
        ax3 = plt.subplot(gs[1, 0])
        
        if ('reinforced' in self.results and 'metrics' in self.results['reinforced'] and
            'baseline' in self.results and 'metrics' in self.results['baseline']):
            
            baseline_ccdi = self.results['baseline']['metrics']['ccdi']
            reinforced_ccdi = self.results['reinforced']['metrics']['ccdi']
            
            # Get coherence metrics
            baseline_coh = self.results['baseline']['metrics']['coherence']
            reinforced_coh = self.results['reinforced']['metrics']['coherence']

            # Fix array lengths
            baseline_steps = [0] + self.results['baseline']['steps'] if len(baseline_coh) > len(self.results['baseline']['steps']) else self.results['baseline']['steps']
            reinforced_steps = [0] + self.results['reinforced']['steps'] if len(reinforced_coh) > len(self.results['reinforced']['steps']) else self.results['reinforced']['steps']

            # Make sure arrays match in length before plotting
            if len(baseline_steps) > len(baseline_coh):
                baseline_steps = baseline_steps[:len(baseline_coh)]
            elif len(baseline_steps) < len(baseline_coh):
                baseline_coh = baseline_coh[:len(baseline_steps)]
                
            if len(reinforced_steps) > len(reinforced_coh):
                reinforced_steps = reinforced_steps[:len(reinforced_coh)]
            elif len(reinforced_steps) < len(reinforced_corr):
                reinforced_coh = reinforced_coh[:len(reinforced_steps)]

            # Now plot with fixed arrays
            plt.plot(baseline_steps, baseline_coh, 'b-', label='Baseline')
            plt.plot(reinforced_steps, reinforced_coh, 'g-', label='Reinforced')
            
            # Mark perturbation step
            if 'perturbation_step' in self.results['reinforced']:
                perturb_step = self.results['reinforced']['perturbation_step']
                ax3.axvline(x=perturb_step, color='r', linestyle='--', 
                          label='Perturbation')
                
            # Mark reactivation steps
            for step in self.reactivation_steps:
                ax3.axvline(x=step, color='m', linestyle=':', alpha=0.5)
                
            ax3.axvline(x=self.reactivation_steps[0] if self.reactivation_steps else 0, 
                      color='m', linestyle=':', label='Reactivation')
                      
            ax3.set_title('Coherence-Correlation Divergence Index')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('CCDI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
        # Bottom right: Stability comparison
        ax4 = plt.subplot(gs[1, 1])
        
        if 'stability' in self.metrics:
            stability = self.metrics['stability']
            
            ax4.bar(['Baseline', 'Reinforced'], 
                  [stability['baseline_variance'], stability['reinforced_variance']],
                  color=['blue', 'green'])
                  
            # Add gain text
            gain = stability['stability_gain']
            if gain > 0:
                label = f"+{gain:.6f}"
            else:
                label = f"{gain:.6f}"
                
            ax4.text(1.0, stability['reinforced_variance'] / 2, 
                   label,
                   ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=12)
                   
            ax4.set_title('Trajectory Variance (Lower is More Stable)')
            ax4.set_ylabel('Variance')
            ax4.grid(axis='y', alpha=0.3)
            
        plt.suptitle(f'Temporal Coherence Reinforcement Summary\n' + 
                   f'ε={self.reactivation_strength}, Schedule: {self.reactivation_schedule}',
                   fontsize=16)
                   
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary.png"))
        plt.close()
        
    def save_metrics(self, metrics=None):
        """
        Save metrics to CSV file.
        
        Args:
            metrics: Optional metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
            
        # Flatten metrics for CSV
        flat_metrics = {}
        
        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_metrics[f"{category}_{key}"] = value
            else:
                flat_metrics[category] = values
                
        # Create CSV file
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(flat_metrics.keys())
            
            # Write values
            writer.writerow(flat_metrics.values())
            
        # Also save in JSON format for completeness
        super().save_metrics(metrics)