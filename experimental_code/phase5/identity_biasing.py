"""
Identity Biasing module for Phase V.
This module explores biasing recovery toward previously-formed attractors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from .module_base import PhaseVModule
from .utils.metrics import IdentityMetrics

class IdentityBiaser(PhaseVModule):
    """Module for testing attractor bias injection."""
    
    def __init__(self, output_dir=None, bias_strength=0.3, bias_method="centroid", 
                perturbation_type="flip", perturbation_magnitude=1.0,
                n_trials=3, steps_per_trial=100, **kwargs):
        """
        Initialize the Identity Biaser.
        
        Args:
            output_dir: Directory for output files
            bias_strength: Strength of bias injection (η)
            bias_method: Method for bias ("centroid", "fingerprint", "gradient")
            perturbation_type: Type of perturbation to apply
            perturbation_magnitude: Magnitude of perturbation
            n_trials: Number of trials to run
            steps_per_trial: Steps per trial
            **kwargs: Additional parameters
        """
        super().__init__(output_dir=output_dir, 
                       bias_strength=bias_strength,
                       bias_method=bias_method,
                       perturbation_type=perturbation_type,
                       perturbation_magnitude=perturbation_magnitude,
                       n_trials=n_trials,
                       steps_per_trial=steps_per_trial, **kwargs)
        
        # Bias parameters
        self.bias_strength = bias_strength
        self.bias_method = bias_method
        
        # Perturbation parameters
        self.perturbation_type = perturbation_type
        self.perturbation_magnitude = perturbation_magnitude
        
        # Trial parameters
        self.n_trials = n_trials
        self.steps_per_trial = steps_per_trial
        
        # Bias reference
        self.reference_attractor = None
        self.reference_fingerprint = None
    
    def run(self, experiment, reference_experiment=None, n_trials=None, 
           steps_per_trial=None, run_unbiased=True):
        """
        Run the identity biasing experiment.
        
        Args:
            experiment: RCFT experiment instance to run trials on
            reference_experiment: Optional reference for bias (uses experiment if None)
            n_trials: Number of trials (overrides init param)
            steps_per_trial: Steps per trial (overrides init param)
            run_unbiased: Whether to run unbiased trials for comparison
            
        Returns:
            Results dictionary
        """
        # Override parameters if provided
        if n_trials is not None:
            self.n_trials = n_trials
            
        if steps_per_trial is not None:
            self.steps_per_trial = steps_per_trial
            
        # Store reference attractor
        if reference_experiment is not None:
            self.reference_attractor = reference_experiment.state.copy()
            self.reference_fingerprint = self.fingerprinter.compute_fingerprint(
                self.reference_attractor, reference_id="reference")
        else:
            self.reference_attractor = experiment.state.copy()
            self.reference_fingerprint = self.fingerprinter.compute_fingerprint(
                self.reference_attractor, reference_id="reference")
            
        # Initialize results tracking
        self.results = {
            'biased': {'trials': [], 'states': [], 'recovery_quality': [], 'metrics': {}},
            'unbiased': {'trials': [], 'states': [], 'recovery_quality': [], 'metrics': {}}
        }
        
        # Define common metrics to track
        metric_names = ['correlation', 'coherence', 'ccdi', 
                      'spectral_entropy', 'spatial_entropy']
        
        for mode in ['biased', 'unbiased']:
            self.results[mode]['final_states'] = []
            
            for metric in metric_names:
                self.results[mode]['metrics'][metric] = []
                
        # Save original state
        original_state = self.reference_attractor.copy()
        
        # Run unbiased trials first if requested
        if run_unbiased:
            print("Running unbiased trials...")
            
            for trial in range(self.n_trials):
                print(f"Unbiased Trial {trial+1}/{self.n_trials}")
                
                # Run unbiased trial
                unbiased_results = self._run_unbiased_trial(
                    experiment, trial, original_state)
                    
                # Save results
                self.results['unbiased']['trials'].append(trial)
                self.results['unbiased']['states'].append(unbiased_results['states'])
                self.results['unbiased']['recovery_quality'].append(
                    unbiased_results['recovery_quality'])
                self.results['unbiased']['final_states'].append(
                    unbiased_results['states'][-1])
                    
                # Save metrics
                for metric in metric_names:
                    if metric in unbiased_results['metrics']:
                        self.results['unbiased']['metrics'][metric].append(
                            unbiased_results['metrics'][metric])
                            
        # Run biased trials
        print("Running biased trials...")
        
        for trial in range(self.n_trials):
            print(f"Biased Trial {trial+1}/{self.n_trials}")
            
            # Run biased trial
            biased_results = self._run_biased_trial(
                experiment, trial, original_state)
                
            # Save results
            self.results['biased']['trials'].append(trial)
            self.results['biased']['states'].append(biased_results['states'])
            self.results['biased']['recovery_quality'].append(
                biased_results['recovery_quality'])
            self.results['biased']['final_states'].append(
                biased_results['states'][-1])
                
            # Save metrics
            for metric in metric_names:
                if metric in biased_results['metrics']:
                    self.results['biased']['metrics'][metric].append(
                        biased_results['metrics'][metric])
                        
        # Calculate summary metrics
        self._calculate_summary_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        return self.results
        
    def _run_unbiased_trial(self, experiment, trial, original_state):
        """
        Run an unbiased trial.
        
        Args:
            experiment: RCFT experiment instance
            trial: Trial number
            original_state: Original state
            
        Returns:
            Trial results
        """
        # Reset to original state
        experiment.state = original_state.copy()
        experiment.memory = original_state.copy()
        
        # Initialize results
        results = {
            'states': [original_state.copy()],
            'metrics': {
                'correlation': [1.0],
                'coherence': [IdentityMetrics.coherence(original_state)],
                'ccdi': [0.0],
                'spectral_entropy': [IdentityMetrics.spectral_entropy(original_state)],
                'spatial_entropy': [IdentityMetrics.spatial_entropy(original_state)]
            }
        }
        
        # Create identity trace
        unbiased_trace = self.track_identity(experiment, label=f"Unbiased_{trial}")
        
        # Apply perturbation
        experiment.apply_perturbation(
            perturbation_type=self.perturbation_type,
            magnitude=self.perturbation_magnitude
        )
        
        # Save perturbed state
        perturbed_state = experiment.state.copy()
        results['states'].append(perturbed_state)
        
        # Calculate metrics for perturbed state
        correlation = IdentityMetrics.correlation(perturbed_state, original_state)
        coherence = IdentityMetrics.coherence(perturbed_state)
        ccdi = coherence - correlation
        spectral_entropy = IdentityMetrics.spectral_entropy(perturbed_state)
        spatial_entropy = IdentityMetrics.spatial_entropy(perturbed_state)
        
        # Record metrics
        results['metrics']['correlation'].append(correlation)
        results['metrics']['coherence'].append(coherence)
        results['metrics']['ccdi'].append(ccdi)
        results['metrics']['spectral_entropy'].append(spectral_entropy)
        results['metrics']['spatial_entropy'].append(spatial_entropy)
        
        # Run recovery
        experiment.update(steps=self.steps_per_trial)
        
        # Save final state
        final_state = experiment.state.copy()
        results['states'].append(final_state)
        
        # Update identity trace
        self.update_identity(experiment)
        
        # Calculate metrics for final state
        correlation = IdentityMetrics.correlation(final_state, original_state)
        coherence = IdentityMetrics.coherence(final_state)
        ccdi = coherence - correlation
        spectral_entropy = IdentityMetrics.spectral_entropy(final_state)
        spatial_entropy = IdentityMetrics.spatial_entropy(final_state)
        
        # Record metrics
        results['metrics']['correlation'].append(correlation)
        results['metrics']['coherence'].append(coherence)
        results['metrics']['ccdi'].append(ccdi)
        results['metrics']['spectral_entropy'].append(spectral_entropy)
        results['metrics']['spatial_entropy'].append(spatial_entropy)
        
        # Calculate recovery quality
        recovery_quality = IdentityMetrics.recovery_quality(
            original_state, perturbed_state, final_state)
            
        results['recovery_quality'] = recovery_quality
        
        return results
        
    def _run_biased_trial(self, experiment, trial, original_state):
        """
        Run a biased trial.
        
        Args:
            experiment: RCFT experiment instance
            trial: Trial number
            original_state: Original state
            
        Returns:
            Trial results
        """
        # Reset to original state
        experiment.state = original_state.copy()
        experiment.memory = original_state.copy()
        
        # Initialize results
        results = {
            'states': [original_state.copy()],
            'metrics': {
                'correlation': [1.0],
                'coherence': [IdentityMetrics.coherence(original_state)],
                'ccdi': [0.0],
                'spectral_entropy': [IdentityMetrics.spectral_entropy(original_state)],
                'spatial_entropy': [IdentityMetrics.spatial_entropy(original_state)]
            }
        }
        
        # Create identity trace
        biased_trace = self.track_identity(experiment, label=f"Biased_{trial}")
        
        # Apply bias
        biased_init = self._apply_bias(experiment.state, self.reference_attractor)
        
        # Set biased state
        experiment.state = biased_init
        experiment.memory = biased_init.copy()
        
        # Apply perturbation
        experiment.apply_perturbation(
            perturbation_type=self.perturbation_type,
            magnitude=self.perturbation_magnitude
        )
        
        # Save perturbed state
        perturbed_state = experiment.state.copy()
        results['states'].append(perturbed_state)
        
        # Calculate metrics for perturbed state
        correlation = IdentityMetrics.correlation(perturbed_state, original_state)
        coherence = IdentityMetrics.coherence(perturbed_state)
        ccdi = coherence - correlation
        spectral_entropy = IdentityMetrics.spectral_entropy(perturbed_state)
        spatial_entropy = IdentityMetrics.spatial_entropy(perturbed_state)
        
        # Record metrics
        results['metrics']['correlation'].append(correlation)
        results['metrics']['coherence'].append(coherence)
        results['metrics']['ccdi'].append(ccdi)
        results['metrics']['spectral_entropy'].append(spectral_entropy)
        results['metrics']['spatial_entropy'].append(spatial_entropy)
        
        # Run recovery
        experiment.update(steps=self.steps_per_trial)
        
        # Save final state
        final_state = experiment.state.copy()
        results['states'].append(final_state)
        
        # Update identity trace
        self.update_identity(experiment)
        
        # Calculate metrics for final state
        correlation = IdentityMetrics.correlation(final_state, original_state)
        coherence = IdentityMetrics.coherence(final_state)
        ccdi = coherence - correlation
        spectral_entropy = IdentityMetrics.spectral_entropy(final_state)
        spatial_entropy = IdentityMetrics.spatial_entropy(final_state)
        
        # Record metrics
        results['metrics']['correlation'].append(correlation)
        results['metrics']['coherence'].append(coherence)
        results['metrics']['ccdi'].append(ccdi)
        results['metrics']['spectral_entropy'].append(spectral_entropy)
        results['metrics']['spatial_entropy'].append(spatial_entropy)
        
        # Calculate recovery quality
        recovery_quality = IdentityMetrics.recovery_quality(
            original_state, perturbed_state, final_state)
            
        results['recovery_quality'] = recovery_quality
        
        return results
        
    def _apply_bias(self, state, reference_attractor):
        """
        Apply bias to a state based on reference attractor.
        
        Args:
            state: State to bias
            reference_attractor: Reference attractor
            
        Returns:
            Biased state
        """
        if self.bias_method == "centroid":
            # Simple weighted combination of state and reference
            # F_0 = (1 - η) * new_init + η * attractor_centroid
            return (1 - self.bias_strength) * state + self.bias_strength * reference_attractor
            
        elif self.bias_method == "fingerprint":
            # Bias in frequency domain
            from scipy.fft import fft2, ifft2
            
            # Compute FFTs
            state_fft = fft2(state)
            ref_fft = fft2(reference_attractor)
            
            # Blend the FFTs
            blended_fft = (1 - self.bias_strength) * state_fft + self.bias_strength * ref_fft
            
            # Inverse transform
            biased_state = np.real(ifft2(blended_fft))
            
            # Rescale to original range
            min_val = np.min(state)
            max_val = np.max(state)
            biased_state = (biased_state - np.min(biased_state)) / (np.max(biased_state) - np.min(biased_state))
            biased_state = biased_state * (max_val - min_val) + min_val
            
            return biased_state
            
        elif self.bias_method == "gradient":
            # Compute gradient toward reference
            gradient = reference_attractor - state
            
            # Apply scaled gradient
            return state + self.bias_strength * gradient
            
        else:
            # Default to centroid method
            return (1 - self.bias_strength) * state + self.bias_strength * reference_attractor
            
    def _calculate_summary_metrics(self):
        """Calculate summary metrics for the experiment."""
        # Skip if no trials completed
        if not self.results['biased']['trials']:
            return
            
        # Calculate metrics
        
        # 1. Average recovery quality
        biased_recovery = np.mean(self.results['biased']['recovery_quality'])
        self.metrics['biased_recovery_quality'] = biased_recovery
        
        if self.results['unbiased']['trials']:
            unbiased_recovery = np.mean(self.results['unbiased']['recovery_quality'])
            self.metrics['unbiased_recovery_quality'] = unbiased_recovery
            self.metrics['recovery_improvement'] = biased_recovery - unbiased_recovery
            
        # 2. Bias success rate
        if self.results['unbiased']['final_states']:
            success_rates = []
            
            for i in range(min(len(self.results['biased']['final_states']), 
                             len(self.results['unbiased']['final_states']))):
                
                # Get fingerprints
                biased_state = self.results['biased']['final_states'][i]
                unbiased_state = self.results['unbiased']['final_states'][i]
                
                biased_fingerprint = self.fingerprinter.compute_fingerprint(biased_state)
                unbiased_fingerprint = self.fingerprinter.compute_fingerprint(unbiased_state)
                
                # Calculate success rate
                success_rate = IdentityMetrics.bias_success_rate(
                    self.reference_fingerprint, biased_fingerprint, unbiased_fingerprint)
                    
                success_rates.append(success_rate)
                
            self.metrics['bias_success_rate'] = np.mean(success_rates)
            
        # 3. Coherence amplification
        if ('metrics' in self.results['biased'] and 
            'coherence' in self.results['biased']['metrics'] and
            'metrics' in self.results['unbiased'] and
            'coherence' in self.results['unbiased']['metrics']):
            
            # Extract final coherence values
            biased_coherence = [coh[-1] for coh in self.results['biased']['metrics']['coherence']]
            
            if self.results['unbiased']['metrics']['coherence']:
                unbiased_coherence = [coh[-1] for coh in self.results['unbiased']['metrics']['coherence']]
                
                # Calculate average coherence
                avg_biased = np.mean(biased_coherence)
                avg_unbiased = np.mean(unbiased_coherence)
                
                self.metrics['biased_coherence'] = avg_biased
                self.metrics['unbiased_coherence'] = avg_unbiased
                self.metrics['coherence_amplification'] = avg_biased - avg_unbiased
                
        # 4. CCDI reduction
        if ('metrics' in self.results['biased'] and 
            'ccdi' in self.results['biased']['metrics'] and
            'metrics' in self.results['unbiased'] and
            'ccdi' in self.results['unbiased']['metrics']):
            
            # Extract final CCDI values
            biased_ccdi = [ccdi[-1] for ccdi in self.results['biased']['metrics']['ccdi']]
            
            if self.results['unbiased']['metrics']['ccdi']:
                unbiased_ccdi = [ccdi[-1] for ccdi in self.results['unbiased']['metrics']['ccdi']]
                
                # Calculate average CCDI
                avg_biased = np.mean(biased_ccdi)
                avg_unbiased = np.mean(unbiased_ccdi)
                
                self.metrics['biased_ccdi'] = avg_biased
                self.metrics['unbiased_ccdi'] = avg_unbiased
                self.metrics['ccdi_reduction'] = avg_unbiased - avg_biased
                
    def _create_visualizations(self):
        """Create visualizations of results."""
        # 1. State comparison (original, biased, unbiased)
        self._visualize_state_comparison()
        
        # 2. Recovery quality comparison
        self._visualize_recovery_comparison()
        
        # 3. Metrics comparison
        self._visualize_metrics_comparison()
        
        # 4. Summary visualization
        self._create_summary_visualization()
        
    def _visualize_state_comparison(self):
        """Visualize state comparison."""
        if (not self.results['biased']['final_states'] or 
            not self.results['unbiased']['final_states']):
            return
            
        # Select states to compare
        original_state = self.reference_attractor
        
        # Average final states across trials
        biased_final = np.mean([state for state in self.results['biased']['final_states']], axis=0)
        
        if self.results['unbiased']['final_states']:
            unbiased_final = np.mean([state for state in self.results['unbiased']['final_states']], axis=0)
            
            # Create visualization
            states = [original_state, biased_final, unbiased_final]
            titles = ['Original', 'Biased Final (Avg)', 'Unbiased Final (Avg)']
            
            self.visualizer.visualize_state_sequence(
                states, titles=titles, title="State Comparison",
                save_path=os.path.join(self.output_dir, "state_comparison.png")
            )
        else:
            # Just original and biased
            states = [original_state, biased_final]
            titles = ['Original', 'Biased Final (Avg)']
            
            self.visualizer.visualize_state_sequence(
                states, titles=titles, title="State Comparison",
                save_path=os.path.join(self.output_dir, "state_comparison.png")
            )
            
    def _visualize_recovery_comparison(self):
        """Visualize recovery quality comparison."""
        plt.figure(figsize=(10, 6))
        
        # Compare recovery quality for each trial
        if (self.results['biased']['recovery_quality'] and 
            self.results['unbiased']['recovery_quality']):
            
            n_trials = min(len(self.results['biased']['recovery_quality']), 
                         len(self.results['unbiased']['recovery_quality']))
                         
            trials = list(range(1, n_trials + 1))
            
            biased_recovery = self.results['biased']['recovery_quality'][:n_trials]
            unbiased_recovery = self.results['unbiased']['recovery_quality'][:n_trials]
            
            # Create bar chart
            x = np.arange(len(trials))
            width = 0.35
            
            plt.bar(x - width/2, biased_recovery, width, label='Biased')
            plt.bar(x + width/2, unbiased_recovery, width, label='Unbiased')
            
            plt.xlabel('Trial')
            plt.ylabel('Recovery Quality')
            plt.title('Recovery Quality Comparison')
            plt.xticks(x, trials)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "recovery_comparison.png"))
            plt.close()
            
    def _visualize_metrics_comparison(self):
        """Visualize metrics comparison."""
        # Correlation, coherence, CCDI
        plt.figure(figsize=(15, 10))
        
        # Define metrics to compare
        metric_names = ['correlation', 'coherence', 'ccdi', 'spectral_entropy']
        subplot_positions = [1, 2, 3, 4]
        titles = ['Correlation', 'Coherence', 'CCDI', 'Spectral Entropy']
        
        for name, pos, title in zip(metric_names, subplot_positions, titles):
            plt.subplot(2, 2, pos)
            
            if (name in self.results['biased']['metrics'] and 
                name in self.results['unbiased']['metrics']):
                
                # Get average metric values across trials
                biased_values = []
                unbiased_values = []
                
                for trial in range(min(len(self.results['biased']['metrics'][name]), 
                                     len(self.results['unbiased']['metrics'][name]))):
                    # Extract values (initial, perturbed, final)
                    if (len(self.results['biased']['metrics'][name][trial]) == 3 and 
                        len(self.results['unbiased']['metrics'][name][trial]) == 3):
                        
                        biased_vals = self.results['biased']['metrics'][name][trial]
                        unbiased_vals = self.results['unbiased']['metrics'][name][trial]
                        
                        biased_values.append(biased_vals)
                        unbiased_values.append(unbiased_vals)
                
                if biased_values and unbiased_values:
                    # Calculate average
                    biased_avg = np.mean(biased_values, axis=0)
                    unbiased_avg = np.mean(unbiased_values, axis=0)
                    
                    # Calculate standard deviation
                    biased_std = np.std(biased_values, axis=0)
                    unbiased_std = np.std(unbiased_values, axis=0)
                    
                    # Plot
                    x = ['Initial', 'Perturbed', 'Final']
                    x_pos = np.arange(len(x))
                    width = 0.35
                    
                    plt.bar(x_pos - width/2, biased_avg, width, 
                          yerr=biased_std, label='Biased', alpha=0.7)
                    plt.bar(x_pos + width/2, unbiased_avg, width, 
                          yerr=unbiased_std, label='Unbiased', alpha=0.7)
                    
                    plt.title(title)
                    plt.xlabel('State')
                    plt.ylabel('Value')
                    plt.xticks(x_pos, x)
                    plt.legend()
                    plt.grid(axis='y', alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_comparison.png"))
        plt.close()
        
    def _create_summary_visualization(self):
        """Create a summary visualization with key findings."""
        plt.figure(figsize=(15, 10))
        
        # Layout
        gs = plt.GridSpec(2, 3)
        
        # 1. Recovery quality comparison
        ax1 = plt.subplot(gs[0, 0])
        
        if ('biased_recovery_quality' in self.metrics and 
            'unbiased_recovery_quality' in self.metrics):
            
            biased = self.metrics['biased_recovery_quality']
            unbiased = self.metrics['unbiased_recovery_quality']
            improvement = self.metrics['recovery_improvement']
            
            bars = ax1.bar(['Unbiased', 'Biased'], [unbiased, biased])
            bars[0].set_color('gray')
            bars[1].set_color('green')
            
            # Add improvement text
            label = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            ax1.text(1, biased / 2, label, ha='center', va='center', 
                   fontweight='bold', color='white')
            
        ax1.set_title('Recovery Quality')
        ax1.set_ylabel('Quality')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Bias success rate
        ax2 = plt.subplot(gs[0, 1])
        
        if 'bias_success_rate' in self.metrics:
            success_rate = self.metrics['bias_success_rate']
            
            ax2.bar(['Success Rate'], [success_rate], color='green')
            
            # Add text
            ax2.text(0, success_rate / 2, f"{success_rate:.4f}", 
                   ha='center', va='center', fontweight='bold', color='white')
            
        ax2.set_title('Bias Success Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Coherence and CCDI comparison
        ax3 = plt.subplot(gs[0, 2])
        
        metrics_to_show = []
        labels = []
        colors = []
        
        if 'coherence_amplification' in self.metrics:
            metrics_to_show.append(self.metrics['coherence_amplification'])
            labels.append('Coherence\nAmplification')
            colors.append('green')
            
        if 'ccdi_reduction' in self.metrics:
            metrics_to_show.append(self.metrics['ccdi_reduction'])
            labels.append('CCDI\nReduction')
            colors.append('blue')
            
        if metrics_to_show:
            bars = ax3.bar(labels, metrics_to_show, color=colors)
            
            # Add text labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height/2, 
                       f"{height:.4f}", ha='center', va='center', 
                       color='white', fontweight='bold')
                
        ax3.set_title('Bias Effects')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. State comparison
        ax4 = plt.subplot(gs[1, :])
        
        if (self.results['biased']['final_states'] and 
            self.results['unbiased']['final_states']):
            
            # Get states
            original = self.reference_attractor
            
            # Average final states
            biased_final = np.mean([state for state in self.results['biased']['final_states']], axis=0)
            unbiased_final = np.mean([state for state in self.results['unbiased']['final_states']], axis=0)
            
            # Create RGB visualization
            # Red channel: original
            # Green channel: biased
            # Blue channel: unbiased
            
            # Normalize states
            def normalize(state):
                return (state - np.min(state)) / (np.max(state) - np.min(state))
                
            norm_original = normalize(original)
            norm_biased = normalize(biased_final)
            norm_unbiased = normalize(unbiased_final)
            
            # Create RGB image
            rgb = np.zeros((original.shape[0], original.shape[1], 3))
            rgb[:, :, 0] = norm_original  # Red channel (original)
            rgb[:, :, 1] = norm_biased    # Green channel (biased)
            rgb[:, :, 2] = norm_unbiased  # Blue channel (unbiased)
            
            ax4.imshow(rgb)
            ax4.set_title('State Comparison (Red: Original, Green: Biased, Blue: Unbiased)')
            ax4.set_axis_off()
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', edgecolor='w', label='Original'),
                Patch(facecolor='green', edgecolor='w', label='Biased'),
                Patch(facecolor='blue', edgecolor='w', label='Unbiased'),
                Patch(facecolor='yellow', edgecolor='w', label='Original + Biased'),
                Patch(facecolor='magenta', edgecolor='w', label='Original + Unbiased'),
                Patch(facecolor='cyan', edgecolor='w', label='Biased + Unbiased'),
                Patch(facecolor='white', edgecolor='w', label='All Three')
            ]
            
            ax4.legend(handles=legend_elements, loc='upper right')
            
        plt.suptitle(f'Identity Biasing Summary\n' + 
                   f'Bias Strength (η)={self.bias_strength}, Method={self.bias_method}',
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
            
        # Create CSV file
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(metrics.keys())
            
            # Write values
            writer.writerow(metrics.values())
            
        # Also save in JSON format for completeness
        super().save_metrics(metrics)