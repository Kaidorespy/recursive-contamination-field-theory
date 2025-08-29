"""
Self-Distinction Index module for Phase V.
This module measures how much a memory "remembers itself" as it evolves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from scipy.fft import fft2, fftshift
from .module_base import PhaseVModule
from .utils.metrics import IdentityMetrics, IdentityTrace

class SelfDistinctionAnalyzer(PhaseVModule):
    """Module for tracking attractor drift and self-distinction."""
    
    def __init__(self, output_dir=None, fingerprint_dimensions=7, 
                perturbation_type="flip", perturbation_magnitude=1.0,
                n_iterations=5, steps_per_iteration=100, **kwargs):
        """
        Initialize the Self-Distinction Analyzer.
        
        Args:
            output_dir: Directory for output files
            fingerprint_dimensions: Number of dimensions for fingerprinting
            perturbation_type: Type of perturbation to apply
            perturbation_magnitude: Magnitude of perturbation
            n_iterations: Number of iterations to run
            steps_per_iteration: Steps per iteration
            **kwargs: Additional parameters
        """
        super().__init__(output_dir=output_dir, 
                       fingerprint_dimensions=fingerprint_dimensions,
                       perturbation_type=perturbation_type,
                       perturbation_magnitude=perturbation_magnitude,
                       n_iterations=n_iterations,
                       steps_per_iteration=steps_per_iteration, **kwargs)
        
        # Fingerprinting parameters
        self.fingerprint_dimensions = fingerprint_dimensions
        
        # Perturbation parameters
        self.perturbation_type = perturbation_type
        self.perturbation_magnitude = perturbation_magnitude
        
        # Iteration parameters
        self.n_iterations = n_iterations
        self.steps_per_iteration = steps_per_iteration
        
        # PCA for attractor tracking
        self.pca = PCA(n_components=2)
        
        # Tracking
        self.attractor_fingerprints = []
        self.attractor_positions = []
        self.initial_fingerprint = None
    
    def run(self, experiment, n_iterations=None, steps_per_iteration=None):
        """
        Run the self-distinction analysis experiment.
        
        Args:
            experiment: RCFT experiment instance
            n_iterations: Number of iterations (overrides init param)
            steps_per_iteration: Steps per iteration (overrides init param)
            
        Returns:
            Results dictionary
        """
        # Override parameters if provided
        if n_iterations is not None:
            self.n_iterations = n_iterations
            
        if steps_per_iteration is not None:
            self.steps_per_iteration = steps_per_iteration
            
        # Initialize results tracking
        self.results = {
            'iterations': [],
            'states': [],
            'fingerprints': [],
            'metrics': {
                'correlation': [],
                'coherence': [],
                'ccdi': [],
                'spectral_entropy': [],
                'spatial_entropy': [],
                'self_distinction': [],
                'attractor_drift': []
            }
        }
        
        # Save initial state
        original_state = experiment.state.copy()
        self.results['states'].append(original_state)
        
        # Compute initial fingerprint
        initial_fingerprint = self.fingerprinter.compute_fingerprint(
            original_state, reference_id="initial")
        self.initial_fingerprint = initial_fingerprint
        self.results['fingerprints'].append(initial_fingerprint)
        
        # Create identity trace
        self.track_identity(experiment, label="Self-Distinction")
        
        # Initial metrics
        self.results['metrics']['correlation'].append(1.0)
        self.results['metrics']['coherence'].append(IdentityMetrics.coherence(original_state))
        self.results['metrics']['ccdi'].append(0.0)
        self.results['metrics']['spectral_entropy'].append(IdentityMetrics.spectral_entropy(original_state))
        self.results['metrics']['spatial_entropy'].append(IdentityMetrics.spatial_entropy(original_state))
        self.results['metrics']['self_distinction'].append(1.0)  # Perfect self-distinction initially
        self.results['metrics']['attractor_drift'].append(0.0)  # No drift initially
        
        # Initial attractor position
        self._add_attractor_position(original_state, iteration=0)
        
        # Run iterations
        for i in range(1, self.n_iterations + 1):
            print(f"Running iteration {i}/{self.n_iterations}...")
            
            # Apply perturbation
            experiment.apply_perturbation(
                perturbation_type=self.perturbation_type,
                magnitude=self.perturbation_magnitude
            )
            
            # Run recovery
            experiment.update(steps=self.steps_per_iteration)
            
            # Record state
            current_state = experiment.state.copy()
            self.results['states'].append(current_state)
            self.results['iterations'].append(i)
            
            # Update identity trace
            self.update_identity(experiment)
            
            # Compute fingerprint
            current_fingerprint = self.fingerprinter.compute_fingerprint(
                current_state, reference_id=f"iteration_{i}")
            self.results['fingerprints'].append(current_fingerprint)
            
            # Add attractor position
            self._add_attractor_position(current_state, iteration=i)
            
            # Calculate metrics
            correlation = IdentityMetrics.correlation(current_state, original_state)
            coherence = IdentityMetrics.coherence(current_state)
            ccdi = coherence - correlation
            spectral_entropy = IdentityMetrics.spectral_entropy(current_state)
            spatial_entropy = IdentityMetrics.spatial_entropy(current_state)
            
            # Calculate self-distinction metrics
            self_distinction = self._calculate_self_distinction(current_state, i)
            attractor_drift = self._calculate_attractor_drift(i)
            
            # Record metrics
            self.results['metrics']['correlation'].append(correlation)
            self.results['metrics']['coherence'].append(coherence)
            self.results['metrics']['ccdi'].append(ccdi)
            self.results['metrics']['spectral_entropy'].append(spectral_entropy)
            self.results['metrics']['spatial_entropy'].append(spatial_entropy)
            self.results['metrics']['self_distinction'].append(self_distinction)
            self.results['metrics']['attractor_drift'].append(attractor_drift)
            
        # Calculate summary metrics
        self._calculate_summary_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        return self.results
    
    def _add_attractor_position(self, state, iteration):
        """
        Add attractor position for tracking drift.
        
        Args:
            state: Current state
            iteration: Iteration number
        """
        # Reshape state for PCA
        flattened = state.flatten().reshape(1, -1)
        
        # Fit or transform depending on whether this is the first state
        if iteration == 0:
            # Initialize PCA with appropriate number of components
            n_components = min(2, flattened.shape[0])  # Limit to number of samples
            self.pca = PCA(n_components=n_components)
            # Fit PCA model
            position = self.pca.fit_transform(flattened)[0]
        else:
            # Transform using existing PCA model
            position = self.pca.transform(flattened)[0]
            
        # Save position
        self.attractor_positions.append((position, iteration))
        
    def _calculate_self_distinction(self, current_state, current_iteration):
        """
        Calculate self-distinction index.
        
        Args:
            current_state: Current state
            current_iteration: Current iteration
            
        Returns:
            Self-distinction index
        """
        # Get all previous states
        previous_states = [self.results['states'][i] for i in range(current_iteration)]
        
        # Calculate self-distinction using previous states as echoes
        self_distinction = IdentityMetrics.self_distinction_index(
            current_state, previous_states, reference_state=self.results['states'][0])
            
        return self_distinction
        
    def _calculate_attractor_drift(self, current_iteration):
        """
        Calculate attractor drift.
        
        Args:
            current_iteration: Current iteration
            
        Returns:
            Drift distance
        """
        # Need at least 2 positions
        if current_iteration < 1 or len(self.attractor_positions) < 2:
            return 0.0
            
        # Get current and previous positions
        current_pos, _ = self.attractor_positions[-1]
        prev_pos, _ = self.attractor_positions[-2]
        
        # Calculate Euclidean distance
        drift = np.sqrt(np.sum((current_pos - prev_pos)**2))
        
        return drift
        
    def _calculate_summary_metrics(self):
        """Calculate summary metrics for the experiment."""
        # Skip if no iterations completed
        if not self.results['iterations']:
            return
            
        # Calculate metrics
        
        # 1. Autocorrelation decay
        correlations = self.results['metrics']['correlation']
        
        # Fit exponential decay curve: y = a * exp(-b * x)
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b):
                return a * np.exp(-b * x)
                
            x = np.array([0] + self.results['iterations'])
            y = np.array(correlations)
            
            popt, _ = curve_fit(exp_decay, x, y, p0=[1.0, 0.1])
            
            # Extract decay rate
            decay_rate = popt[1]
            self.metrics['autocorrelation_decay_rate'] = decay_rate
            
        except:
            # Fallback: calculate simple decay rate
            if len(correlations) > 1:
                initial = correlations[0]
                final = correlations[-1]
                n_steps = len(correlations) - 1
                
                # Average decay per step
                decay_rate = (initial - final) / n_steps if n_steps > 0 else 0
                self.metrics['autocorrelation_decay_rate'] = decay_rate
                
        # 2. Total attractor drift
        if len(self.attractor_positions) > 1:
            first_pos, _ = self.attractor_positions[0]
            last_pos, _ = self.attractor_positions[-1]
            
            total_drift = np.sqrt(np.sum((last_pos - first_pos)**2))
            self.metrics['total_attractor_drift'] = total_drift
            
        # 3. Fingerprint divergence
        if self.initial_fingerprint and len(self.results['fingerprints']) > 1:
            final_fingerprint = self.results['fingerprints'][-1]
            
            # Calculate distance from initial fingerprint
            distances = self.fingerprinter.compute_fingerprint_distance(
                self.initial_fingerprint, final_fingerprint)
                
            self.metrics['fingerprint_divergence'] = distances.get('aggregate_distance', 0.0)
            
        # 4. Average self-distinction
        self_distinction_values = self.results['metrics']['self_distinction'][1:]  # Skip initial
        if self_distinction_values:
            self.metrics['average_self_distinction'] = np.mean(self_distinction_values)
            self.metrics['min_self_distinction'] = np.min(self_distinction_values)
            
        # 5. Stability metrics
        states = self.results['states']
        if len(states) > 1:
            stability = IdentityMetrics.attractor_stability(states)
            self.metrics['attractor_stability'] = stability
            
    def _create_visualizations(self):
        """Create visualizations of results."""
        # 1. Attractor trajectory
        self._visualize_attractor_trajectory()
        
        # 2. Metrics over iterations
        self._visualize_metrics_over_iterations()
        
        # 3. FFT fingerprints
        self._visualize_fingerprints()
        
        # 4. State evolution
        self._visualize_state_evolution()
        
        # 5. Summary visualization
        self._create_summary_visualization()
        
    def _visualize_attractor_trajectory(self):
        """Visualize attractor trajectory in PCA space."""
        if len(self.attractor_positions) < 2:
            return
            
        plt.figure(figsize=(10, 8))
        
        # Extract positions and iterations
        positions = np.array([pos for pos, _ in self.attractor_positions])
        iterations = np.array([it for _, it in self.attractor_positions])
        
        # Check dimensions of positions
        if positions.shape[1] >= 2:
            # 2D visualization
            sc = plt.scatter(positions[:, 0], positions[:, 1], 
                        c=iterations, cmap='viridis',
                        s=100, alpha=0.7)
                    
            # Add colorbar
            cbar = plt.colorbar(sc)
            cbar.set_label('Iteration')
            
            # Connect points with arrows
            for i in range(len(positions) - 1):
                plt.arrow(positions[i, 0], positions[i, 1],
                        positions[i+1, 0] - positions[i, 0],
                        positions[i+1, 1] - positions[i, 1],
                        head_width=0.05, head_length=0.1, fc='black', ec='black')
                
            # Add labels for start and end
            plt.annotate('Start', positions[0], xytext=(-20, -20),
                    textcoords='offset points', ha='center',
                    arrowprops=dict(arrowstyle='->', color='green'))
                
            plt.annotate('End', positions[-1], xytext=(20, 20),
                    textcoords='offset points', ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        else:
            # 1D visualization - plot against iteration number
            plt.scatter(iterations, positions[:, 0], 
                    c=iterations, cmap='viridis',
                    s=100, alpha=0.7)
                    
            # Connect points with lines
            plt.plot(iterations, positions[:, 0], 'k-', alpha=0.3)
            
            # Add labels for start and end
            plt.annotate('Start', (iterations[0], positions[0, 0]), 
                    xytext=(-20, -20),
                    textcoords='offset points', ha='center',
                    arrowprops=dict(arrowstyle='->', color='green'))
                
            plt.annotate('End', (iterations[-1], positions[-1, 0]), 
                    xytext=(20, 20),
                    textcoords='offset points', ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.xlabel('Iteration')
            plt.ylabel('PC1')
        
        plt.title('Attractor Trajectory in PCA Space')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "attractor_trajectory.png"))
        plt.close()
        
    def _visualize_metrics_over_iterations(self):
        """Visualize metrics over iterations."""
        # Correlation, coherence, CCDI
        plt.figure(figsize=(12, 8))
        
        # Get iterations (including initial)
        iterations = [0] + self.results['iterations']
        
        # Plot correlation
        plt.subplot(2, 2, 1)
        plt.plot(iterations, self.results['metrics']['correlation'], 'o-', 
               label='Correlation')
        plt.title('Correlation with Initial State')
        plt.xlabel('Iteration')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        
        # Plot coherence
        plt.subplot(2, 2, 2)
        plt.plot(iterations, self.results['metrics']['coherence'], 'o-', 
               label='Coherence')
        plt.title('Field Coherence')
        plt.xlabel('Iteration')
        plt.ylabel('Coherence')
        plt.grid(True, alpha=0.3)
        
        # Plot CCDI
        plt.subplot(2, 2, 3)
        plt.plot(iterations, self.results['metrics']['ccdi'], 'o-', 
               label='CCDI')
        plt.title('Coherence-Correlation Divergence Index')
        plt.xlabel('Iteration')
        plt.ylabel('CCDI')
        plt.grid(True, alpha=0.3)
        
        # Plot self-distinction
        plt.subplot(2, 2, 4)
        plt.plot(iterations, self.results['metrics']['self_distinction'], 'o-',
               label='Self-Distinction')
        plt.title('Self-Distinction Index')
        plt.xlabel('Iteration')
        plt.ylabel('Self-Distinction')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_over_iterations.png"))
        plt.close()
        
        # Entropy and drift
        plt.figure(figsize=(12, 8))
        
        # Plot spectral entropy
        plt.subplot(2, 2, 1)
        plt.plot(iterations, self.results['metrics']['spectral_entropy'], 'o-',
               label='Spectral Entropy')
        plt.title('Spectral Entropy')
        plt.xlabel('Iteration')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
        
        # Plot spatial entropy
        plt.subplot(2, 2, 2)
        plt.plot(iterations, self.results['metrics']['spatial_entropy'], 'o-',
               label='Spatial Entropy')
        plt.title('Spatial Entropy')
        plt.xlabel('Iteration')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
        
        # Plot attractor drift
        plt.subplot(2, 2, 3)
        # Skip first point (initial has no drift)
        drift_iterations = iterations[1:]
        drift_values = self.results['metrics']['attractor_drift'][1:]
        
        plt.plot(drift_iterations, drift_values, 'o-',
               label='Attractor Drift')
        plt.title('Attractor Drift Between Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Drift Distance')
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative drift
        plt.subplot(2, 2, 4)
        cumulative_drift = np.cumsum(self.results['metrics']['attractor_drift'])
        plt.plot(iterations, cumulative_drift, 'o-',
               label='Cumulative Drift')
        plt.title('Cumulative Attractor Drift')
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Drift')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "entropy_drift_metrics.png"))
        plt.close()
        
    def _visualize_fingerprints(self):
        """Visualize FFT fingerprints for each iteration."""
        if not self.results['fingerprints']:
            return
            
        plt.figure(figsize=(12, 10))
        
        # Get fingerprints
        fingerprints = self.results['fingerprints']
        
        # Check if FFT spectra are available
        if 'fft_spectrum' in fingerprints[0]:
            # Plot FFT spectra for each iteration
            plt.subplot(2, 1, 1)
            
            for i, fingerprint in enumerate(fingerprints):
                label = 'Initial' if i == 0 else f'Iteration {i}'
                plt.plot(fingerprint['fft_spectrum'], label=label)
                
            plt.title('FFT Power Spectra')
            plt.xlabel('Frequency Bin')
            plt.ylabel('Power')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot spectral entropy
            plt.subplot(2, 1, 2)
            iterations = [0] + self.results['iterations']
            
            entropies = [fingerprint.get('spectral_entropy', 0) 
                       for fingerprint in fingerprints]
                       
            plt.plot(iterations, entropies, 'o-')
            plt.title('Spectral Entropy')
            plt.xlabel('Iteration')
            plt.ylabel('Entropy')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fingerprint_evolution.png"))
        plt.close()
        
        # Create fingerprint comparison between initial and final
        if len(fingerprints) > 1:
            labels = ['Initial', 'Final']
            self.fingerprinter.visualize_fingerprint_comparison(
                [fingerprints[0], fingerprints[-1]], 
                labels=labels,
                title="Initial vs Final Fingerprint Comparison"
            )
            
            plt.savefig(os.path.join(self.output_dir, "fingerprint_comparison.png"))
            plt.close()
        
    def _visualize_state_evolution(self):
        """Visualize state evolution across iterations."""
        # Show key states (initial, middle, final)
        states = self.results['states']
        
        if len(states) < 2:
            return
            
        # Select states to visualize
        if len(states) > 5:
            # Select initial, final, and 3 intermediates
            indices = [0, len(states)//4, len(states)//2, 3*len(states)//4, -1]
            selected_states = [states[i] for i in indices]
            
            # Generate titles
            if indices[0] == 0:
                titles = ['Initial']
            else:
                titles = [f'Iteration {indices[0]}']
                
            for i in indices[1:-1]:
                titles.append(f'Iteration {i}')
                
            if indices[-1] == len(states) - 1:
                titles.append('Final')
            else:
                titles.append(f'Iteration {indices[-1]}')
        else:
            # Use all states
            selected_states = states
            titles = ['Initial'] + [f'Iteration {i}' for i in self.results['iterations']]
        
        # Create visualization
        self.visualizer.visualize_state_sequence(
            selected_states, titles=titles, title="State Evolution",
            save_path=os.path.join(self.output_dir, "state_evolution.png")
        )
        
        # Create trajectory overlay
        self.visualizer.visualize_attractor_trajectory(
            states, reference_state=states[0], decay=0.95,
            title="Attractor Trajectory Overlay",
            save_path=os.path.join(self.output_dir, "trajectory_overlay.png")
        )
        
    def _create_summary_visualization(self):
        """Create a summary visualization with key findings."""
        plt.figure(figsize=(15, 12))
        
        # Create layout
        gs = plt.GridSpec(3, 3)
        
        # ... (existing code remains unchanged) ...
        
        # 3. Attractor trajectory in PCA space
        ax3 = plt.subplot(gs[0, 2])
        
        if len(self.attractor_positions) >= 2:
            positions = np.array([pos for pos, _ in self.attractor_positions])
            iterations_array = np.array([it for _, it in self.attractor_positions])
            
            # Check dimensions of positions
            if positions.shape[1] >= 2:
                # 2D visualization
                sc = ax3.scatter(positions[:, 0], positions[:, 1], 
                            c=iterations_array, cmap='viridis',
                            s=100, alpha=0.7)
                            
                plt.colorbar(sc, ax=ax3, label='Iteration')
                
                # Connect points with arrows
                for i in range(len(positions) - 1):
                    ax3.arrow(positions[i, 0], positions[i, 1],
                        positions[i+1, 0] - positions[i, 0],
                        positions[i+1, 1] - positions[i, 1],
                        head_width=0.05, head_length=0.1, fc='black', ec='black')
                        
                ax3.annotate('Start', positions[0], xytext=(-20, -20),
                        textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='green'))
                        
                ax3.annotate('End', positions[-1], xytext=(20, 20),
                        textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='red'))
                
                ax3.set_xlabel('PC1')
                ax3.set_ylabel('PC2')
            else:
                # 1D visualization - plot against iteration number
                sc = ax3.scatter(iterations_array, positions[:, 0], 
                            c=iterations_array, cmap='viridis',
                            s=100, alpha=0.7)
                            
                plt.colorbar(sc, ax=ax3, label='Iteration')
                            
                # Connect points with lines
                ax3.plot(iterations_array, positions[:, 0], 'k-', alpha=0.3)
                
                # Add labels for start and end
                ax3.annotate('Start', (iterations_array[0], positions[0, 0]), 
                        xytext=(-20, -20),
                        textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='green'))
                    
                ax3.annotate('End', (iterations_array[-1], positions[-1, 0]), 
                        xytext=(20, 20),
                        textcoords='offset points', ha='center',
                        arrowprops=dict(arrowstyle='->', color='red'))
                
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('PC1')
        
        ax3.set_title('Attractor Trajectory in PCA Space')
        ax3.grid(True, alpha=0.3)
        
        # ... (rest of the method remains unchanged) ...
        
        # 4. Spectral fingerprint evolution
        ax4 = plt.subplot(gs[1, 0:2])
        
        fingerprints = self.results['fingerprints']
        if fingerprints and 'fft_spectrum' in fingerprints[0]:
            for i, fingerprint in enumerate(fingerprints):
                label = 'Initial' if i == 0 else (
                       'Final' if i == len(fingerprints)-1 else None)
                alpha = 1.0 if label else 0.3
                
                ax4.plot(fingerprint['fft_spectrum'], alpha=alpha, label=label)
                
        ax4.set_title('FFT Power Spectrum Evolution')
        ax4.set_xlabel('Frequency Bin')
        ax4.set_ylabel('Power')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Key states visualization
        ax5 = plt.subplot(gs[1, 2])
        
        states = self.results['states']
        if states and len(states) >= 2:
            # Show initial and final
            initial_state = states[0]
            final_state = states[-1]
            
            # Compute difference
            difference = final_state - initial_state
            
            # Create RGB image with initial in red, final in green
            scaled_initial = (initial_state - initial_state.min()) / (initial_state.max() - initial_state.min())
            scaled_final = (final_state - final_state.min()) / (final_state.max() - final_state.min())
            
            rgb = np.zeros((initial_state.shape[0], initial_state.shape[1], 3))
            rgb[:, :, 0] = scaled_initial  # Red channel = initial
            rgb[:, :, 1] = scaled_final  # Green channel = final
            
            ax5.imshow(rgb)
            ax5.set_title('Initial (Red) vs Final (Green)')
            ax5.set_axis_off()
            
        # 6. CCDI vs Correlation space
        ax6 = plt.subplot(gs[2, 0])
        
        if ('correlation' in self.results['metrics'] and 
            'ccdi' in self.results['metrics']):
            
            corr = self.results['metrics']['correlation']
            ccdi = self.results['metrics']['ccdi']
            
            ax6.plot(corr, ccdi, 'o-')
            
            # Add arrows to show direction
            for i in range(len(corr) - 1):
                ax6.annotate('', xy=(corr[i+1], ccdi[i+1]), 
                          xytext=(corr[i], ccdi[i]),
                          arrowprops=dict(arrowstyle='->', color='black'))
                          
            # Mark start and end
            ax6.plot(corr[0], ccdi[0], 'o', color='green', markersize=10, label='Start')
            ax6.plot(corr[-1], ccdi[-1], 'o', color='red', markersize=10, label='End')
            
        ax6.set_title('Correlation-CCDI Space')
        ax6.set_xlabel('Correlation')
        ax6.set_ylabel('CCDI')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary metrics
        ax7 = plt.subplot(gs[2, 1:3])
        
        metric_names = ['autocorrelation_decay_rate', 'total_attractor_drift', 
                      'fingerprint_divergence', 'average_self_distinction', 
                      'attractor_stability']
                      
        metric_labels = ['Autocorrelation\nDecay Rate', 'Total\nAttractor Drift',
                       'Fingerprint\nDivergence', 'Average\nSelf-Distinction',
                       'Attractor\nStability']
                      
        metrics_to_show = []
        labels_to_show = []
        
        for name, label in zip(metric_names, metric_labels):
            if name in self.metrics:
                metrics_to_show.append(self.metrics[name])
                labels_to_show.append(label)
                
        if metrics_to_show:
            # Normalize for better visualization
            metrics_norm = np.array(metrics_to_show)
            metrics_norm = metrics_norm / np.max(np.abs(metrics_norm))
            
            bars = ax7.bar(labels_to_show, metrics_norm)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, metrics_to_show)):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f"{value:.4f}", ha='center')
                       
        ax7.set_title('Summary Metrics (Normalized)')
        ax7.set_ylim(0, 1.2)
        ax7.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Self-Distinction Analysis Summary\n' + 
                   f'n_iterations={self.n_iterations}, steps_per_iteration={self.steps_per_iteration}',
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
        
        for category, value in metrics.items():
            flat_metrics[category] = value
                
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