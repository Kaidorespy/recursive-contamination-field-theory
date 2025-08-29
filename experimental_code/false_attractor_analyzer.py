# false_attractor_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from rcft_framework import RCFTExperiment

class FalseAttractorAnalyzer:
    """Analyzes false attractor dynamics in RCFT systems with focus on fractal patterns"""
    
    def __init__(self, output_dir="false_attractor_analysis", n_trials=5):
        """Initialize the analyzer with default parameters from Phase I"""
        self.output_dir = output_dir
        self.n_trials = n_trials
        
        # Core RCFT parameters (same as Phase I)
        self.alpha = 0.35    # Memory strength
        self.beta = 0.5      # Coupling strength 
        self.gamma = 0.92    # Memory decay rate
        
        # CCDI threshold for anomaly classification
        self.ccdi_threshold = 0.08
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Master results storage
        self.all_results = []
        self.all_residuals = []
        self.class_labels = []
    
    def run_stability_trials(self, pattern_type="fractal", delays=[10, 20, 50], n_trials=None):
        """Run multiple trials for each delay with controlled noise variations"""
        if n_trials is None:
            n_trials = self.n_trials
            
        print(f"Running stability trials for {pattern_type} pattern with {n_trials} trials per delay")
        
        # Master results list
        results = []
        
        # Store residuals for later dimensionality reduction
        residuals = []
        labels = []
        
        # Run trials for each delay
        for delay in delays:
            print(f"\nProcessing delay={delay} with {n_trials} trials...")
            
            delay_dir = os.path.join(self.output_dir, f"{pattern_type}_delay{delay}")
            if not os.path.exists(delay_dir):
                os.makedirs(delay_dir)
            
            for trial in tqdm(range(n_trials), desc=f"Delay {delay} trials"):
                # Create trial directory
                trial_dir = os.path.join(delay_dir, f"trial{trial}")
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)
                
                # Generate noise pattern (±1% variation)
                noise_factor = np.random.uniform(0.99, 1.01, size=(64, 64))  # Default grid size 64x64
                
                # Run experiment
                exp_results = self.run_layered_perturbation(
                    pattern_type=pattern_type,
                    delay=delay,
                    trial_id=trial,
                    noise_factor=noise_factor,
                    output_dir=trial_dir
                )
                
                # Compute CCDI
                ccdi = self.compute_ccdi(exp_results['final_correlation'], exp_results['final_coherence'])
                exp_results['ccdi'] = ccdi
                
                # Classify recovery trajectory
                recovery_class = self.classify_recovery(
                    exp_results['correlation_curve'], 
                    exp_results['perturbation_steps']
                )
                exp_results['recovery_class'] = recovery_class
                
                # Extract attractor residual (Δ)
                residual = self.extract_attractor_residual(
                    exp_results['final_state'], 
                    exp_results['initial_state']
                )
                
                # Save residual visualization
                self.visualize_residual(residual, os.path.join(trial_dir, "delta_field.png"))
                
                # Save residual as numpy array
                np.save(os.path.join(trial_dir, "delta_field.npy"), residual)
                
                # Add to collection for dimensionality reduction
                residuals.append(residual.flatten())  # Flatten for PCA/t-SNE
                labels.append(f"d{delay}_t{trial}")
                
                # Save CCDI value
                with open(os.path.join(trial_dir, "ccdi_value.txt"), 'w') as f:
                    f.write(f"CCDI: {ccdi:.6f}\n")
                    f.write(f"Recovery Class: {recovery_class}\n")
                    f.write(f"Final Correlation: {exp_results['final_correlation']:.6f}\n")
                    f.write(f"Final Coherence: {exp_results['final_coherence']:.6f}\n")
                
                # Add metadata for results summary
                exp_results['pattern'] = pattern_type
                exp_results['delay'] = delay
                exp_results['trial_id'] = trial
                
                # Add to results list
                results.append(exp_results)
        
        # Save master results
        self.save_results(results, os.path.join(self.output_dir, "false_memory_summary.csv"))
        
        # Apply dimensionality reduction to residuals
        if residuals:
            self.analyze_residuals(residuals, labels)
            
        # Save all data
        self.all_results = results
        self.all_residuals = residuals
        self.class_labels = labels
        
        return results
    
    def run_layered_perturbation(self, pattern_type, delay, trial_id, noise_factor=None, output_dir=None):
        """Run a single layered perturbation experiment with noise injection"""
        # Initialize experiment with default parameters
        exp = RCFTExperiment(
            memory_strength=self.alpha,
            coupling_strength=self.beta,
            memory_decay=self.gamma
        )
        
        # Initialize with fractal pattern
        exp.initialize_pattern(pattern_type=pattern_type)
        
        # Save initial state
        initial_state = exp.state.copy()
        
        # Apply first perturbation
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        
        # Record first perturbation step
        first_perturbation = exp.perturbation_step
        
        # Let system recover for specified delay
        exp.update(steps=delay)
        
        # Record state before second perturbation
        mid_state = exp.state.copy()
        mid_correlation = exp.metrics['correlation'][-1]
        mid_coherence = exp.metrics['coherence'][-1]
        
        # Apply second perturbation with noise variation if provided
        if noise_factor is not None:
            # Create a copy of the state and apply noise
            noisy_state = exp.state.copy() * noise_factor
            
            # Apply second perturbation on noisy state
            center = (20, 20)  # Different location for second perturbation
            radius = 10
            
            # Create mask for perturbation region
            x = np.arange(exp.size)
            y = np.arange(exp.size)
            X, Y = np.meshgrid(x, y)
            mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2)
            
            # Apply flip perturbation with noise
            noisy_state[mask] = -noisy_state[mask]
            
            # Update experiment state with perturbed noisy state
            exp.state = noisy_state
        else:
            # Apply standard second perturbation
            exp.apply_perturbation(
                perturbation_type="flip", 
                magnitude=1.0, 
                center=(20, 20),  # Different location
                radius=10
            )
        
        # Record second perturbation step
        second_perturbation = len(exp.history) - 1
        
        # Let system evolve to completion
        exp.update(steps=50)
        
        # Capture final state and metrics
        final_state = exp.state.copy()
        final_correlation = exp.metrics['correlation'][-1]
        final_coherence = exp.metrics['coherence'][-1]
        final_mutual_info = exp.metrics['mutual_info'][-1]
        final_entropy = exp.metrics['spectral_entropy'][-1]
        
        # Save visualizations if output directory provided
        if output_dir:
            # Save key metrics plots
            self.visualize_metrics(
                exp=exp, 
                save_dir=output_dir,
                perturbation_steps=[first_perturbation, second_perturbation]
            )
            
            # Save key frames visualization
            plt.figure(figsize=(16, 4))
            
            # Initial state
            plt.subplot(1, 4, 1)
            plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
            plt.title("Initial State")
            plt.axis('off')
            
            # After first perturbation
            plt.subplot(1, 4, 2)
            plt.imshow(exp.history[first_perturbation], cmap='viridis', vmin=-1, vmax=1)
            plt.title("After First Perturbation")
            plt.axis('off')
            
            # Mid recovery (before second perturbation)
            plt.subplot(1, 4, 3)
            plt.imshow(mid_state, cmap='viridis', vmin=-1, vmax=1)
            plt.title(f"Mid Recovery (t={delay})")
            plt.axis('off')
            
            # Final state
            plt.subplot(1, 4, 4)
            plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
            plt.title(f"Final State (t={len(exp.history)-1})")
            plt.axis('off')
            
            plt.colorbar(orientation='horizontal', fraction=0.02, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "key_frames.png"))
            plt.close()
        
        # Return results dictionary
        return {
            'initial_state': initial_state,
            'mid_state': mid_state,
            'final_state': final_state,
            'correlation_curve': exp.metrics['correlation'],
            'coherence_curve': exp.metrics['coherence'],
            'entropy_curve': exp.metrics['spectral_entropy'],
            'mutual_info_curve': exp.metrics['mutual_info'],
            'perturbation_steps': [first_perturbation, second_perturbation],
            'mid_correlation': mid_correlation,
            'mid_coherence': mid_coherence,
            'final_correlation': final_correlation,
            'final_coherence': final_coherence,
            'final_mutual_info': final_mutual_info,
            'final_entropy': final_entropy
        }
    
    def compute_ccdi(self, correlation, coherence):
        """Compute Coherence-Correlation Divergence Index (CCDI)"""
        return coherence - correlation
    
    def classify_recovery(self, correlation_curve, perturbation_steps):
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
    
    def extract_attractor_residual(self, final_state, initial_state):
        """Compute the difference between final and initial states (Δ field)"""
        return final_state - initial_state
    
    def analyze_residuals(self, residuals, labels):
        """Apply dimensionality reduction and clustering to residuals"""
        # Convert to numpy array
        residuals_array = np.array(residuals)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_pca = pca.fit_transform(residuals_array)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=min(5, len(residuals) - 1) if len(residuals) > 1 else 1)
        reduced_tsne = tsne.fit_transform(residuals_array)
        
        # Create PCA plot
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        
        # Extract delay from labels for coloring
        delays = [int(label.split('_')[0][1:]) for label in labels]
        unique_delays = sorted(list(set(delays)))
        delay_colors = {d: plt.cm.tab10(i/len(unique_delays)) for i, d in enumerate(unique_delays)}
        
        # Plot points colored by delay
        for i, (x, y) in enumerate(reduced_pca):
            delay = delays[i]
            plt.scatter(x, y, color=delay_colors[delay], label=f'Delay {delay}' if i == delays.index(delay) else "")
            plt.annotate(labels[i], (x, y), fontsize=8)
        
        plt.title('PCA of Attractor Residuals (Δ Fields)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        
        # Create t-SNE plot if enough samples
        if len(residuals) > 1:
            plt.subplot(2, 1, 2)
            
            # Plot points colored by delay
            for i, (x, y) in enumerate(reduced_tsne):
                delay = delays[i]
                plt.scatter(x, y, color=delay_colors[delay], label=f'Delay {delay}' if i == delays.index(delay) else "")
                plt.annotate(labels[i], (x, y), fontsize=8)
            
            plt.title('t-SNE of Attractor Residuals (Δ Fields)')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "residuals_dimensionality_reduction.png"))
        plt.close()
        
        # Save reduced coordinates
        pca_df = pd.DataFrame({
            'label': labels,
            'delay': delays,
            'pca_x': reduced_pca[:, 0],
            'pca_y': reduced_pca[:, 1],
        })
        pca_df.to_csv(os.path.join(self.output_dir, "pca_coordinates.csv"), index=False)
        
        if len(residuals) > 1:
            tsne_df = pd.DataFrame({
                'label': labels,
                'delay': delays,
                'tsne_x': reduced_tsne[:, 0],
                'tsne_y': reduced_tsne[:, 1],
            })
            tsne_df.to_csv(os.path.join(self.output_dir, "tsne_coordinates.csv"), index=False)
    
    def visualize_residual(self, residual, save_path):
        """Visualize the attractor residual (Δ field)"""
        plt.figure(figsize=(8, 6))
        
        # Plot the residual field
        im = plt.imshow(residual, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(im, label='Δ Value')
        plt.title('Attractor Residual (Final - Initial)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def visualize_metrics(self, exp, save_dir, perturbation_steps):
        """Create and save various metric visualizations"""
        # Plot correlation over time
        plt.figure(figsize=(10, 6))
        plt.plot(exp.metrics['correlation'], label='Correlation with Initial State')
        
        # Mark perturbations
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--', 
                      label=f'Perturbation {i+1}' if i == 0 else None)
        
        plt.title('Correlation with Initial State')
        plt.xlabel('Time Step')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "correlation_curve.png"))
        plt.close()
        
        # Plot coherence over time
        plt.figure(figsize=(10, 6))
        plt.plot(exp.metrics['coherence'], label='Field Coherence')
        
        # Mark perturbations
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--', 
                      label=f'Perturbation {i+1}' if i == 0 else None)
        
        plt.title('Field Coherence')
        plt.xlabel('Time Step')
        plt.ylabel('Coherence')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "coherence_curve.png"))
        plt.close()
        
        # Plot entropy over time
        plt.figure(figsize=(10, 6))
        plt.plot(exp.metrics['spectral_entropy'], label='Spectral Entropy')
        
        # Mark perturbations
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--', 
                      label=f'Perturbation {i+1}' if i == 0 else None)
        
        plt.title('Spectral Entropy')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "entropy_curve.png"))
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'step': range(len(exp.metrics['correlation'])),
            'correlation': exp.metrics['correlation'],
            'coherence': exp.metrics['coherence'],
            'mutual_info': exp.metrics['mutual_info'],
            'spectral_entropy': exp.metrics['spectral_entropy'],
        })
        metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    
    def save_results(self, results, output_path):
        """Save aggregated results to CSV"""
        # Create a DataFrame from results
        results_df = pd.DataFrame([
            {
                'pattern': r['pattern'],
                'delay': r['delay'],
                'trial_id': r['trial_id'],
                'final_correlation': r['final_correlation'],
                'final_coherence': r['final_coherence'],
                'final_mutual_info': r['final_mutual_info'],
                'ccdi': r['ccdi'],
                'recovery_class': r['recovery_class'],
                'is_anomalous': r['ccdi'] > self.ccdi_threshold
            }
            for r in results
        ])
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        # Print summary
        print("\nResults Summary:")
        print(f"Total trials: {len(results)}")
        
        # Count by recovery class
        class_counts = results_df['recovery_class'].value_counts()
        print("\nRecovery Classes:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} trials ({count/len(results)*100:.1f}%)")
        
        # Count anomalies
        anomaly_count = results_df['is_anomalous'].sum()
        print(f"\nAnomalies (CCDI > {self.ccdi_threshold}):")
        print(f"  {anomaly_count} trials ({anomaly_count/len(results)*100:.1f}%)")
        
        return results_df


if __name__ == "__main__":
    print("Initializing False Attractor Analysis...")
    analyzer = FalseAttractorAnalyzer(output_dir="false_attractor_analysis", n_trials=5)
    
    print("Running stability trials for fractal pattern with delays [10, 20, 50]...")
    results = analyzer.run_stability_trials(
        pattern_type="fractal",
        delays=[10, 20, 50]
    )
    
    print("Analysis complete. Check 'false_attractor_analysis' directory for detailed results.")