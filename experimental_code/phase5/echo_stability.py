"""
Echo Trail Stability module for Phase V.
This module tests if a system "remembers that it remembered" through echo recursion.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from .module_base import PhaseVModule
from .utils.metrics import IdentityMetrics

class EchoTrailAnalyzer(PhaseVModule):
    """Module for analyzing echo trail effects on recovery."""
    
    def __init__(self, output_dir=None, echo_depth=3, echo_strength=0.15,
                 echo_decay=0.8, perturbation_type="flip", 
                 perturbation_magnitude=1.0, steps_per_iteration=100,
                 n_iterations=5, **kwargs):
        """
        Initialize the Echo Trail Analyzer.
        
        Args:
            output_dir: Directory for output files
            echo_depth: Number of past recovery steps to include in echo
            echo_strength: Strength of echo reinforcement
            echo_decay: Decay rate for older echoes
            perturbation_type: Type of perturbation to apply
            perturbation_magnitude: Magnitude of perturbation
            steps_per_iteration: Steps per iteration
            n_iterations: Number of iterations to run
            **kwargs: Additional parameters
        """
        super().__init__(output_dir=output_dir, 
                       echo_depth=echo_depth,
                       echo_strength=echo_strength,
                       echo_decay=echo_decay,
                       perturbation_type=perturbation_type,
                       perturbation_magnitude=perturbation_magnitude,
                       steps_per_iteration=steps_per_iteration,
                       n_iterations=n_iterations, **kwargs)
        
        # Echo parameters
        self.echo_depth = echo_depth
        self.echo_strength = echo_strength
        self.echo_decay = echo_decay
        
        # Perturbation parameters
        self.perturbation_type = perturbation_type
        self.perturbation_magnitude = perturbation_magnitude
        
        # Iteration parameters
        self.steps_per_iteration = steps_per_iteration
        self.n_iterations = n_iterations
        
        # Storage for recovery trajectories
        self.echo_recovery_trajectory = []
        self.standard_recovery_trajectory = []
    
    def run(self, experiment, n_iterations=None, steps_per_iteration=None,
            run_standard=True):
        """
        Run the echo trail stability experiment.
        
        Args:
            experiment: RCFT experiment instance
            n_iterations: Number of iterations (overrides init param)
            steps_per_iteration: Steps per iteration (overrides init param)
            run_standard: Whether to run standard recovery for comparison
            
        Returns:
            Results dictionary
        """
        # Override parameters if provided
        if n_iterations is not None:
            self.n_iterations = n_iterations
            
        if steps_per_iteration is not None:
            self.steps_per_iteration = steps_per_iteration
            
        # Initialize results
        self.results = {
            'echo': {'iterations': [], 'states': [], 'recovery_quality': []},
            'standard': {'iterations': [], 'states': [], 'recovery_quality': []}
        }
        
        # Clear trajectories
        self.echo_recovery_trajectory = []
        self.standard_recovery_trajectory = []
        
        # Save original state
        original_state = experiment.state.copy()
        
        # Run standard recovery first if requested
        if run_standard:
            print("Running standard recovery (no echo)...")
            standard_results = self._run_standard_recovery(experiment, original_state)
            self.results['standard'] = standard_results
            self.standard_recovery_trajectory = standard_results['recovery_trajectory']
            
        # Run echo recovery
        print("Running echo recovery...")
        echo_results = self._run_echo_recovery(experiment, original_state)
        self.results['echo'] = echo_results
        self.echo_recovery_trajectory = echo_results['recovery_trajectory']
        
        # Calculate summary metrics
        self._calculate_summary_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        return self.results
        
    def _run_standard_recovery(self, experiment, original_state):
        """
        Run standard recovery without echo.
        
        Args:
            experiment: RCFT experiment instance
            original_state: Original state
            
        Returns:
            Standard recovery results
        """
        # Reset to original state
        experiment.state = original_state.copy()
        experiment.memory = original_state.copy()
        
        # Initialize results
        results = {
            'iterations': [],
            'states': [original_state.copy()],
            'perturbed_states': [],
            'final_states': [],
            'recovery_quality': [],
            'recovery_trajectory': []
        }
        
        # Create identity trace
        standard_trace = self.track_identity(experiment, label="Standard")
        
        # Run iterations
        for i in range(self.n_iterations):
            print(f"Standard iteration {i+1}/{self.n_iterations}")
            
            # Apply perturbation
            experiment.apply_perturbation(
                perturbation_type=self.perturbation_type,
                magnitude=self.perturbation_magnitude
            )
            
            # Save perturbed state
            perturbed_state = experiment.state.copy()
            results['perturbed_states'].append(perturbed_state)
            
            # Run standard recovery
            recovery_states = []
            for step in range(self.steps_per_iteration):
                # Update experiment
                experiment.update(steps=1)
                
                # Save state
                state = experiment.state.copy()
                recovery_states.append(state)
                
            # Save final state
            final_state = experiment.state.copy()
            results['final_states'].append(final_state)
            
            # Update identity trace
            self.update_identity(experiment)
            
            # Calculate recovery quality
            recovery_quality = IdentityMetrics.recovery_quality(
                original_state, perturbed_state, final_state)
                
            # Save results
            results['iterations'].append(i)
            results['states'].append(final_state)
            results['recovery_quality'].append(recovery_quality)
            results['recovery_trajectory'].append(recovery_states)
            
        return results
        
    def _run_echo_recovery(self, experiment, original_state):
        """
        Run recovery with echo trail.
        
        Args:
            experiment: RCFT experiment instance
            original_state: Original state
            
        Returns:
            Echo recovery results
        """
        # Reset to original state
        experiment.state = original_state.copy()
        experiment.memory = original_state.copy()
        
        # Initialize results
        results = {
            'iterations': [],
            'states': [original_state.copy()],
            'perturbed_states': [],
            'final_states': [],
            'recovery_quality': [],
            'recovery_trajectory': [],
            'echo_contributions': []
        }
        
        # Create identity trace
        echo_trace = self.track_identity(experiment, label="Echo")
        
        # Previous recovery states (empty initially)
        previous_recovery = []
        
        # Run iterations
        for i in range(self.n_iterations):
            print(f"Echo iteration {i+1}/{self.n_iterations}")
            
            # Apply perturbation
            experiment.apply_perturbation(
                perturbation_type=self.perturbation_type,
                magnitude=self.perturbation_magnitude
            )
            
            # Save perturbed state
            perturbed_state = experiment.state.copy()
            results['perturbed_states'].append(perturbed_state)
            
            # Run echo recovery
            recovery_states = []
            echo_contributions = []
            
            for step in range(self.steps_per_iteration):
                # Calculate echo contribution
                echo = self._calculate_echo(previous_recovery, step)
                echo_contributions.append(echo)
                
                # Apply echo if available
                if echo is not None:
                    # First do standard update
                    experiment.update(steps=1)
                    
                    # Then add echo contribution
                    experiment.state += echo
                else:
                    # Just do standard update
                    experiment.update(steps=1)
                    
                # Save state
                state = experiment.state.copy()
                recovery_states.append(state)
                
            # Save final state
            final_state = experiment.state.copy()
            results['final_states'].append(final_state)
            
            # Update identity trace
            self.update_identity(experiment)
            
            # Calculate recovery quality
            recovery_quality = IdentityMetrics.recovery_quality(
                original_state, perturbed_state, final_state)
                
            # Update previous recovery
            previous_recovery.append(recovery_states)
            if len(previous_recovery) > self.echo_depth:
                previous_recovery.pop(0)  # Remove oldest
                
            # Save results
            results['iterations'].append(i)
            results['states'].append(final_state)
            results['recovery_quality'].append(recovery_quality)
            results['recovery_trajectory'].append(recovery_states)
            results['echo_contributions'].append(echo_contributions)
            
        return results
        
    def _calculate_echo(self, previous_recovery, step):
        """
        Calculate echo contribution from previous recovery trajectories.
        
        Args:
            previous_recovery: List of previous recovery trajectories
            step: Current step in recovery
            
        Returns:
            Echo contribution or None
        """
        if not previous_recovery:
            return None
            
        # Calculate weighted echo
        echo = None
        
        # Apply echo decay to weight older trajectories less
        for i, trajectory in enumerate(previous_recovery):
            # Weight by position in history (newer = stronger)
            weight = self.echo_strength * (self.echo_decay ** (len(previous_recovery) - i - 1))
            
            # Get state from this trajectory if step is valid
            if step < len(trajectory):
                state = trajectory[step]
                
                if echo is None:
                    echo = weight * state
                else:
                    echo += weight * state
                    
        return echo
        
    def _calculate_summary_metrics(self):
        """Calculate summary metrics for the experiment."""
        # Average recovery quality
        if self.results['echo']['recovery_quality']:
            echo_recovery = np.mean(self.results['echo']['recovery_quality'])
            self.metrics['echo_recovery_quality'] = echo_recovery
            
        if self.results['standard']['recovery_quality']:
            standard_recovery = np.mean(self.results['standard']['recovery_quality'])
            self.metrics['standard_recovery_quality'] = standard_recovery
            
            # Calculate improvement
            self.metrics['recovery_improvement'] = echo_recovery - standard_recovery
            
        # Echo correction delta
        if ('recovery_quality' in self.results['echo'] and 
            'recovery_quality' in self.results['standard']):
            
            echo_quality = self.results['echo']['recovery_quality']
            standard_quality = self.results['standard']['recovery_quality']
            
            min_iterations = min(len(echo_quality), len(standard_quality))
            
            if min_iterations > 0:
                # Calculate delta for each iteration
                deltas = []
                
                for i in range(min_iterations):
                    echo_q = echo_quality[i]
                    standard_q = standard_quality[i]
                    
                    delta = IdentityMetrics.echo_correction_delta(standard_q, echo_q)
                    deltas.append(delta)
                    
                # Average delta
                avg_delta = np.mean(deltas)
                self.metrics['echo_correction_delta'] = avg_delta
                
        # Recovery trajectory convergence rate
        if self.results['echo']['recovery_trajectory']:
            # Calculate convergence (correlation improvement over steps)
            all_correlations = []
            
            for trajectory in self.results['echo']['recovery_trajectory']:
                # Get correlation with original at each step
                original_state = self.results['echo']['states'][0]
                correlations = []
                
                for state in trajectory:
                    corr = IdentityMetrics.correlation(state, original_state)
                    correlations.append(corr)
                    
                if correlations:
                    all_correlations.append(correlations)
                    
            if all_correlations:
                # Average correlation curve
                max_steps = max(len(corr) for corr in all_correlations)
                avg_correlations = np.zeros(max_steps)
                count = np.zeros(max_steps)
                
                for correlations in all_correlations:
                    for i, corr in enumerate(correlations):
                        avg_correlations[i] += corr
                        count[i] += 1
                        
                # Normalize by count
                for i in range(max_steps):
                    if count[i] > 0:
                        avg_correlations[i] /= count[i]
                        
                # Calculate convergence rate
                if len(avg_correlations) > 1:
                    # Fit exponential approach to 1.0
                    try:
                        from scipy.optimize import curve_fit
                        
                        def exp_approach(x, a, b, c):
                            return a - b * np.exp(-c * x)
                            
                        x = np.arange(len(avg_correlations))
                        y = avg_correlations
                        
                        # Reasonable initial guesses
                        p0 = [1.0, 1.0, 0.1]
                        
                        try:
                            popt, _ = curve_fit(exp_approach, x, y, p0=p0)
                            
                            # Extract convergence rate
                            convergence_rate = popt[2]
                            self.metrics['convergence_rate'] = convergence_rate
                            
                            # Save parameters for plotting
                            self.metrics['convergence_params'] = popt.tolist()
                        except:
                            # Fallback: simple measure
                            if len(avg_correlations) > 1:
                                initial_corr = avg_correlations[0]
                                final_corr = avg_correlations[-1]
                                
                                convergence_rate = (final_corr - initial_corr) / len(avg_correlations)
                                self.metrics['convergence_rate'] = convergence_rate
                    except ImportError:
                        # scipy not available, use simple measure
                        if len(avg_correlations) > 1:
                            initial_corr = avg_correlations[0]
                            final_corr = avg_correlations[-1]
                            
                            convergence_rate = (final_corr - initial_corr) / len(avg_correlations)
                            self.metrics['convergence_rate'] = convergence_rate
                            
                # Save average correlation curve
                self.metrics['avg_correlation_curve'] = avg_correlations.tolist()
                
    def _create_visualizations(self):
        """Create visualizations of results."""
        # 1. Recovery quality comparison
        self._visualize_recovery_comparison()
        
        # 2. Recovery trajectories
        self._visualize_recovery_trajectories()
        
        # 3. Echo contribution
        self._visualize_echo_contribution()
        
        # 4. Convergence rate
        self._visualize_convergence_rate()
        
        # 5. State comparison
        self._visualize_state_comparison()
        
        # 6. Summary visualization
        self._create_summary_visualization()
        
    def _visualize_recovery_comparison(self):
        """Visualize recovery quality comparison."""
        if (not self.results['echo']['recovery_quality'] or 
            not self.results['standard']['recovery_quality']):
            return
            
        plt.figure(figsize=(10, 6))
        
        # Get data
        echo_quality = self.results['echo']['recovery_quality']
        standard_quality = self.results['standard']['recovery_quality']
        
        # Get iterations (limit to common iterations)
        min_iterations = min(len(echo_quality), len(standard_quality))
        iterations = list(range(1, min_iterations + 1))
        
        echo_quality = echo_quality[:min_iterations]
        standard_quality = standard_quality[:min_iterations]
        
        # Calculate improvement
        improvement = [echo - standard for echo, standard in zip(echo_quality, standard_quality)]
        
        # Create plots
        plt.subplot(2, 1, 1)
        
        plt.plot(iterations, echo_quality, 'go-', label='Echo')
        plt.plot(iterations, standard_quality, 'bo-', label='Standard')
        
        plt.title('Recovery Quality by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Recovery Quality')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        
        bars = plt.bar(iterations, improvement)
        
        # Color bars by sign
        for i, bar in enumerate(bars):
            bar.set_color('green' if improvement[i] >= 0 else 'red')
            
        plt.title('Echo Improvement by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Improvement')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recovery_comparison.png"))
        plt.close()
        
    def _visualize_recovery_trajectories(self):
        """Visualize recovery trajectories."""
        if (not self.echo_recovery_trajectory or 
            not self.standard_recovery_trajectory):
            return
            
        plt.figure(figsize=(12, 8))
        
        # For visualization clarity, only show trajectories from last iteration
        if self.echo_recovery_trajectory:
            echo_trajectory = self.echo_recovery_trajectory[-1]
            
            # Calculate correlation with original at each step
            original_state = self.results['echo']['states'][0]
            echo_correlations = []
            
            for state in echo_trajectory:
                corr = IdentityMetrics.correlation(state, original_state)
                echo_correlations.append(corr)
                
        if self.standard_recovery_trajectory:
            standard_trajectory = self.standard_recovery_trajectory[-1]
            
            # Calculate correlation with original at each step
            original_state = self.results['standard']['states'][0]
            standard_correlations = []
            
            for state in standard_trajectory:
                corr = IdentityMetrics.correlation(state, original_state)
                standard_correlations.append(corr)
                
        # Plot correlations
        if 'echo_correlations' in locals() and 'standard_correlations' in locals():
            plt.subplot(2, 1, 1)
            
            steps = list(range(1, min(len(echo_correlations), len(standard_correlations)) + 1))
            plt.plot(steps, echo_correlations[:len(steps)], 'go-', label='Echo')
            plt.plot(steps, standard_correlations[:len(steps)], 'bo-', label='Standard')
            
            plt.title('Recovery Correlation Trajectory (Last Iteration)')
            plt.xlabel('Recovery Step')
            plt.ylabel('Correlation with Original')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        # Plot convergence rate if available
        if ('avg_correlation_curve' in self.metrics and 
            'convergence_params' in self.metrics):
            
            plt.subplot(2, 1, 2)
            
            avg_curve = self.metrics['avg_correlation_curve']
            steps = list(range(1, len(avg_curve) + 1))
            
            plt.plot(steps, avg_curve, 'ko-', label='Average')
            
            # Plot fitted curve
            params = self.metrics['convergence_params']
            
            def exp_approach(x, a, b, c):
                return a - b * np.exp(-c * x)
                
            x_fit = np.linspace(1, len(avg_curve), 100)
            y_fit = exp_approach(x_fit - 1, *params)
            
            plt.plot(x_fit, y_fit, 'r--', 
                   label=f'Fit (rate={params[2]:.4f})')
                   
            plt.title('Average Recovery Correlation with Fitted Curve')
            plt.xlabel('Recovery Step')
            plt.ylabel('Correlation with Original')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recovery_trajectories.png"))
        plt.close()
        
    def _visualize_echo_contribution(self):
        """Visualize echo contribution."""
        if not self.results['echo']['echo_contributions']:
            return
            
        # Create visualization showing echo strength
        echo_strengths = []
        
        for contributions in self.results['echo']['echo_contributions']:
            # Calculate magnitude of echo for each step
            strengths = []
            
            for echo in contributions:
                if echo is not None:
                    strength = np.sqrt(np.sum(echo**2))  # L2 norm
                    strengths.append(strength)
                else:
                    strengths.append(0)
                    
            echo_strengths.append(strengths)
            
        # Create heatmap visualization
        plt.figure(figsize=(12, 6))
        
        # Combine all contributions into a matrix
        max_steps = max(len(strengths) for strengths in echo_strengths)
        strength_matrix = np.zeros((len(echo_strengths), max_steps))
        
        for i, strengths in enumerate(echo_strengths):
            strength_matrix[i, :len(strengths)] = strengths
            
        # Create heatmap
        im = plt.imshow(strength_matrix, aspect='auto', cmap='viridis')
        
        plt.colorbar(im, label='Echo Strength')
        
        plt.title('Echo Contribution Strength by Iteration and Step')
        plt.xlabel('Recovery Step')
        plt.ylabel('Iteration')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "echo_contribution.png"))
        plt.close()
        
        # Create a line plot of average echo strength over recovery
        plt.figure(figsize=(10, 6))
        
        avg_strength = np.mean(strength_matrix, axis=0)
        std_strength = np.std(strength_matrix, axis=0)
        
        plt.plot(range(1, len(avg_strength) + 1), avg_strength, 'o-')
        plt.fill_between(range(1, len(avg_strength) + 1), 
                       avg_strength - std_strength, 
                       avg_strength + std_strength, 
                       alpha=0.3)
                       
        plt.title('Average Echo Strength During Recovery')
        plt.xlabel('Recovery Step')
        plt.ylabel('Echo Strength')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "avg_echo_strength.png"))
        plt.close()
        
    def _visualize_convergence_rate(self):
        """Visualize convergence rate comparison."""
        if 'convergence_rate' not in self.metrics:
            return
            
        plt.figure(figsize=(8, 6))
        
        # Get data
        convergence_rate = self.metrics['convergence_rate']
        
        # Create bar chart
        plt.bar(['Convergence Rate'], [convergence_rate], color='green')
        
        # Add text label
        plt.text(0, convergence_rate / 2, f"{convergence_rate:.6f}", 
               ha='center', va='center', color='white', fontweight='bold')
               
        plt.title('Recovery Trajectory Convergence Rate')
        plt.ylabel('Rate')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "convergence_rate.png"))
        plt.close()
        
    def _visualize_state_comparison(self):
        """Visualize state comparison."""
        if (not self.results['echo']['final_states'] or 
            not self.results['standard']['final_states']):
            return
            
        # Get original, final echo, and final standard states
        original_state = self.results['echo']['states'][0]
        echo_final = self.results['echo']['final_states'][-1]
        standard_final = self.results['standard']['final_states'][-1]
        
        # Create visualization
        states = [original_state, echo_final, standard_final]
        titles = ['Original', 'Echo Final', 'Standard Final']
        
        self.visualizer.visualize_state_sequence(
            states, titles=titles, title="State Comparison",
            save_path=os.path.join(self.output_dir, "state_comparison.png")
        )
        
        # Visualize echo recovery as an animation-like sequence
        if self.echo_recovery_trajectory:
            # Select key frames from the last echo recovery
            trajectory = self.echo_recovery_trajectory[-1]
            
            # Select frames at regular intervals
            n_frames = min(5, len(trajectory))
            indices = np.linspace(0, len(trajectory) - 1, n_frames).astype(int)
            selected_frames = [trajectory[i] for i in indices]
            
            # Add original and perturbed states
            if self.results['echo']['perturbed_states']:
                perturbed = self.results['echo']['perturbed_states'][-1]
                frames = [original_state, perturbed] + selected_frames
                titles = ['Original', 'Perturbed'] + [f'Recovery Step {i+1}' for i in range(n_frames)]
            else:
                frames = [original_state] + selected_frames
                titles = ['Original'] + [f'Recovery Step {i+1}' for i in range(n_frames)]
                
            # Create visualization
            self.visualizer.visualize_state_sequence(
                frames, titles=titles, title="Echo Recovery Sequence",
                save_path=os.path.join(self.output_dir, "echo_recovery_sequence.png")
            )
            
            # Create echo visualization
            if len(self.results['echo']['echo_contributions']) > 0:
                # Get echo states from last iteration
                echo_contributions = self.results['echo']['echo_contributions'][-1]
                
                # Select non-None echo states at regular intervals
                non_none_echoes = [e for e in echo_contributions if e is not None]
                
                if non_none_echoes:
                    n_echoes = min(3, len(non_none_echoes))
                    indices = np.linspace(0, len(non_none_echoes) - 1, n_echoes).astype(int)
                    selected_echoes = [non_none_echoes[i] for i in indices]
                    
                    # Create visualization
                    self.visualizer.visualize_echo_effect(
                        original_state, selected_echoes, echo_final,
                        title="Echo Effect",
                        save_path=os.path.join(self.output_dir, "echo_effect.png")
                    )
        
    def _create_summary_visualization(self):
        """Create a summary visualization with key findings."""
        plt.figure(figsize=(15, 12))
        
        # Layout
        gs = plt.GridSpec(2, 2)
        
        # 1. Recovery quality comparison
        ax1 = plt.subplot(gs[0, 0])
        
        if ('echo_recovery_quality' in self.metrics and 
            'standard_recovery_quality' in self.metrics):
            
            echo_quality = self.metrics['echo_recovery_quality']
            standard_quality = self.metrics['standard_recovery_quality']
            improvement = self.metrics['recovery_improvement']
            
            bars = ax1.bar(['Standard', 'Echo'], [standard_quality, echo_quality])
            bars[0].set_color('blue')
            bars[1].set_color('green')
            
            # Add improvement text
            label = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            ax1.text(1, echo_quality / 2, label, ha='center', va='center', 
                   fontweight='bold', color='white')
                   
        ax1.set_title('Average Recovery Quality')
        ax1.set_ylabel('Quality')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Echo correction delta
        ax2 = plt.subplot(gs[0, 1])
        
        if 'echo_correction_delta' in self.metrics:
            delta = self.metrics['echo_correction_delta']
            
            ax2.bar(['Echo Correction Delta'], [delta], 
                  color='green' if delta >= 0 else 'red')
                  
            # Add text label
            ax2.text(0, delta / 2, f"{delta:.4f}", 
                   ha='center', va='center', color='white', fontweight='bold')
                   
        ax2.set_title('Echo Correction Delta')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Recovery trajectories
        ax3 = plt.subplot(gs[1, 0])
        
        if (self.echo_recovery_trajectory and self.standard_recovery_trajectory and
            len(self.echo_recovery_trajectory) > 0 and len(self.standard_recovery_trajectory) > 0):
            
            # Use last trajectories
            echo_trajectory = self.echo_recovery_trajectory[-1]
            standard_trajectory = self.standard_recovery_trajectory[-1]
            
            # Calculate correlation with original
            original_echo = self.results['echo']['states'][0]
            original_standard = self.results['standard']['states'][0]
            
            echo_correlations = []
            for state in echo_trajectory:
                corr = IdentityMetrics.correlation(state, original_echo)
                echo_correlations.append(corr)
                
            standard_correlations = []
            for state in standard_trajectory:
                corr = IdentityMetrics.correlation(state, original_standard)
                standard_correlations.append(corr)
                
            # Plot correlation curves
            min_steps = min(len(echo_correlations), len(standard_correlations))
            steps = list(range(1, min_steps + 1))
            
            ax3.plot(steps, echo_correlations[:min_steps], 'g-', label='Echo')
            ax3.plot(steps, standard_correlations[:min_steps], 'b-', label='Standard')
            
            # Show shaded area for difference
            ax3.fill_between(steps, 
                           standard_correlations[:min_steps], 
                           echo_correlations[:min_steps], 
                           where=np.array(echo_correlations[:min_steps]) >= np.array(standard_correlations[:min_steps]),
                           color='green', alpha=0.3, label='Echo Improvement')
                           
        ax3.set_title('Recovery Trajectory (Last Iteration)')
        ax3.set_xlabel('Recovery Step')
        ax3.set_ylabel('Correlation with Original')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Echo contribution
        ax4 = plt.subplot(gs[1, 1])
        
        if 'avg_correlation_curve' in self.metrics:
            avg_curve = self.metrics['avg_correlation_curve']
            steps = list(range(1, len(avg_curve) + 1))
            
            ax4.plot(steps, avg_curve, 'ko-', label='Average')
            
            # Plot fitted curve if available
            if 'convergence_params' in self.metrics:
                params = self.metrics['convergence_params']
                
                def exp_approach(x, a, b, c):
                    return a - b * np.exp(-c * x)
                    
                x_fit = np.linspace(1, len(avg_curve), 100)
                y_fit = exp_approach(x_fit - 1, *params)
                
                ax4.plot(x_fit, y_fit, 'r--', 
                       label=f'Fit (rate={params[2]:.4f})')
                       
        ax4.set_title('Average Recovery Correlation')
        ax4.set_xlabel('Recovery Step')
        ax4.set_ylabel('Correlation')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.suptitle(f'Echo Trail Stability Summary\n' + 
                   f'Echo Depth={self.echo_depth}, Strength={self.echo_strength}, Decay={self.echo_decay}',
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
        
        for key, value in metrics.items():
            if isinstance(value, list):
                # Skip long arrays
                if key != 'avg_correlation_curve':
                    for i, v in enumerate(value):
                        flat_metrics[f"{key}_{i+1}"] = v
            else:
                flat_metrics[key] = value
                
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