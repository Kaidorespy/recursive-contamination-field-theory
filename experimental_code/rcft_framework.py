import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
import os
from datetime import datetime
from scipy import stats
from sklearn.metrics import mutual_info_score

class RCFTExperiment:
    """Core class for RCFT experiments following the formal roadmap"""
    
    def __init__(self, size=64, memory_strength=0.35, coupling_strength=0.5, memory_decay=0.92):
        """Initialize RCFT experiment with core parameters
        
        Args:
            size: Grid size for the field
            memory_strength (α): Controls influence of memory on state
            coupling_strength (β): Controls influence of neighbors
            memory_decay (γ): Controls how quickly memory fades
        """
        self.size = size
        self.alpha = memory_strength    # Memory influence (α)
        self.beta = coupling_strength   # Neighbor coupling (β)
        self.gamma = memory_decay       # Memory decay (γ)
        
        # State and memory fields
        self.state = np.zeros((size, size))      # Current state s[t]
        self.memory = np.zeros((size, size))     # Memory trace m[t]
        
        # Tracking
        self.history = []                # History of states
        self.memory_history = []         # History of memory states
        self.metrics = {                 # Metrics over time
            'coherence': [],
            'correlation': [],
            'mutual_info': [],
            'spectral_entropy': [],
            'variance': []
        }
        self.initial_state = None
        self.perturbation_step = -1      # When perturbation was applied
        
        # Experiment metadata
        self.experiment_type = "pattern_specific_recovery"
        self.perturbation_type = "none"
        self.perturbation_magnitude = 0.0
        self.pattern_type = "none"
        self.parameters = {}
        
    def initialize_pattern(self, pattern_type="radial", **kwargs):
        """Initialize the field with a specific pattern"""
        # Initialize with common pattern types
        if pattern_type == "radial":
            self.state = self._create_radial_pattern(**kwargs)
        elif pattern_type == "horizontal":
            self.state = self._create_horizontal_pattern(**kwargs)
        elif pattern_type == "diagonal":
            self.state = self._create_diagonal_pattern(**kwargs)
        elif pattern_type == "fractal":
            self.state = self._create_fractal_noise(**kwargs)
        elif pattern_type == "stochastic":
            self.state = self._create_stochastic_pattern(**kwargs)
        elif pattern_type == "lattice":
            self.state = self._create_lattice_pattern(**kwargs)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        self.memory = self.state.copy()
        self.initial_state = self.state.copy()
        self.pattern_type = pattern_type
        
        # Record initial state
        self.history.append(self.state.copy())
        self.memory_history.append(self.memory.copy())
        self._calculate_metrics()
    
    def _create_radial_pattern(self, frequency=2):
        """Create a radial wave pattern"""
        x = np.linspace(0, 2*np.pi*frequency, self.size)
        y = np.linspace(0, 2*np.pi*frequency, self.size)
        X, Y = np.meshgrid(x, y)
        center = self.size // 2
        radius = np.sqrt((X - x[center])**2 + (Y - y[center])**2)
        return np.sin(radius)
    
    def _create_horizontal_pattern(self, frequency=2):
        """Create a horizontal wave pattern"""
        x = np.linspace(0, 2*np.pi*frequency, self.size)
        y = np.linspace(0, 2*np.pi*frequency, self.size)
        X, Y = np.meshgrid(x, y)
        return np.sin(X)
    
    def _create_diagonal_pattern(self, frequency=2):
        """Create a diagonal wave pattern"""
        x = np.linspace(0, 2*np.pi*frequency, self.size)
        y = np.linspace(0, 2*np.pi*frequency, self.size)
        X, Y = np.meshgrid(x, y)
        return np.sin(X + Y)
        
    def _create_fractal_noise(self, octaves=6, persistence=0.5):
        """Create fractal noise pattern"""
        noise = np.random.normal(0, 1, (self.size, self.size))
        result = np.zeros((self.size, self.size))
        
        for i in range(octaves):
            scale = 2**i
            weight = persistence**i
            scaled_noise = np.random.normal(0, 1, (self.size//scale + 1, self.size//scale + 1))
            scaled_up = np.kron(scaled_noise, np.ones((scale, scale)))[:self.size, :self.size]
            result += weight * scaled_up
            
        # Normalize to [-1, 1]
        result = 2 * (result - np.min(result)) / (np.max(result) - np.min(result)) - 1
        return result
    
    def _create_lattice_pattern(self, frequency=4):
        """Create a lattice pattern"""
        x = np.linspace(0, 2*np.pi*frequency, self.size)
        y = np.linspace(0, 2*np.pi*frequency, self.size)
        X, Y = np.meshgrid(x, y)
        return np.sin(X) * np.sin(Y)
    
    def _create_stochastic_pattern(self, smoothness=2.0):
        """Create a smoothed random pattern"""
        random_field = np.random.uniform(-1, 1, (self.size, self.size))
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(random_field, sigma=smoothness)
        
    def apply_perturbation(self, perturbation_type="flip", magnitude=1.0, **kwargs):
        """Apply perturbation to the field with controlled magnitude"""
        self.perturbation_type = perturbation_type
        self.perturbation_magnitude = magnitude
        
        # Define the perturbation region (default to center)
        center = kwargs.get("center", None)
        radius = kwargs.get("radius", self.size // 4)
        
        if center is None:
            center = (self.size // 2, self.size // 2)
            
        # Create a circular mask for the perturbation region
        x = np.arange(self.size)
        y = np.arange(self.size)
        X, Y = np.meshgrid(x, y)
        mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2)
        
        # Apply different types of perturbations based on magnitude
        if perturbation_type == "noise":
            self.state[mask] += magnitude * np.random.uniform(-1, 1, size=self.state[mask].shape)
            self.state = np.clip(self.state, -1, 1)  # Ensure values stay in range
        elif perturbation_type == "flip":
            self.state[mask] = -magnitude * self.state[mask]  # Scale and invert values
        elif perturbation_type == "zero":
            self.state[mask] = 0  # Zero out the region
        elif perturbation_type == "memory_wipe":
            # Perturbation that affects memory but not state
            self.memory[mask] *= (1 - magnitude)  # Scale down memory by magnitude
        elif perturbation_type == "displacement":
            # Shift the pattern within the mask
            shift = (int(radius * magnitude / 5), int(radius * magnitude / 5))
            temp = self.state.copy()
            for i in range(self.size):
                for j in range(self.size):
                    if mask[i, j]:
                        i_src = (i - shift[0]) % self.size
                        j_src = (j - shift[1]) % self.size
                        self.state[i, j] = temp[i_src, j_src]
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
        self.perturbation_step = len(self.history)
        
        # Record post-perturbation state
        self.history.append(self.state.copy())
        self.memory_history.append(self.memory.copy())
        self._calculate_metrics()
    
    def update(self, steps=1, memory_rule="exponential"):
        """Evolve the field for a number of steps"""
        for _ in range(steps):
            # Convolution kernel for neighbor influence
            kernel = np.array([[0.05, 0.2, 0.05], 
                               [0.2, 0, 0.2], 
                               [0.05, 0.2, 0.05]])
            
            # Compute neighbor influence via convolution
            padded = np.pad(self.state, 1, mode='wrap')
            neighbor_influence = np.zeros_like(self.state)
            
            for i in range(self.size):
                for j in range(self.size):
                    neighbor_influence[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
            
            # Calculate spatial update
            spatial_update = (1 - self.beta) * self.state + self.beta * neighbor_influence
            
            # Apply memory influence
            next_state = (1 - self.alpha) * spatial_update + self.alpha * self.memory
            
            # Update memory based on memory rule
            if memory_rule == "exponential":
                self.memory = self.gamma * self.memory + (1 - self.gamma) * self.state
            elif memory_rule == "adaptive":
                # Adaptive decay based on recent changes
                if len(self.history) > 2:
                    recent_change = np.mean(np.abs(self.history[-1] - self.history[-2]))
                    gamma_effective = self.gamma * (1 / (1 + recent_change))
                    self.memory = gamma_effective * self.memory + (1 - gamma_effective) * self.state
                else:
                    self.memory = self.gamma * self.memory + (1 - self.gamma) * self.state
            else:
                raise ValueError(f"Unknown memory rule: {memory_rule}")
            
            # Update state
            self.state = next_state
            
            # Record history
            self.history.append(self.state.copy())
            self.memory_history.append(self.memory.copy())
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate various metrics for the current state"""
        if self.initial_state is None:
            return
            
        # Coherence (inverse of variance)
        variance = np.var(self.state)
        coherence = 1 / (1 + variance)
        self.metrics['coherence'].append(coherence)
        self.metrics['variance'].append(variance)
        
        # Correlation with initial state
        correlation = np.corrcoef(self.initial_state.flatten(), self.state.flatten())[0, 1]
        self.metrics['correlation'].append(correlation)
        
        # Mutual information (bin values for discrete calculation)
        bins = 10  # Number of bins for mutual information
        x_bins = np.floor(bins * (self.initial_state.flatten() + 1) / 2).astype(int)
        y_bins = np.floor(bins * (self.state.flatten() + 1) / 2).astype(int)
        mutual_info = mutual_info_score(x_bins, y_bins)
        self.metrics['mutual_info'].append(mutual_info)
        
        # Spectral entropy (using FFT magnitude)
        fft = np.abs(np.fft.fft2(self.state))
        fft_norm = fft / np.sum(fft)
        spectral_entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))
        self.metrics['spectral_entropy'].append(spectral_entropy)
    
    def calculate_recovery_metrics(self):
        """Calculate detailed recovery metrics as specified in the roadmap"""
        if len(self.history) < 3 or self.perturbation_step == -1:
            return {
                'recovery_correlation': 0,
                'recovery_mutual_info': 0,
                'recovery_time': 0,
                'recovery_quality': 0
            }
            
        # Original metrics
        initial_correlation = self.metrics['correlation'][0]
        initial_mutual_info = self.metrics['mutual_info'][0]
        
        # Post-perturbation metrics
        perturb_correlation = self.metrics['correlation'][self.perturbation_step]
        perturb_mutual_info = self.metrics['mutual_info'][self.perturbation_step]
        
        # Final metrics
        final_correlation = self.metrics['correlation'][-1]
        final_mutual_info = self.metrics['mutual_info'][-1]
        
        # Calculate correlation recovery quality (0 to 1 scale)
        if initial_correlation == perturb_correlation:
            recovery_correlation = 1.0
        else:
            recovery_correlation = (final_correlation - perturb_correlation) / (initial_correlation - perturb_correlation)
            recovery_correlation = min(max(recovery_correlation, 0), 1)  # Clamp to [0,1]
            
        # Calculate mutual information recovery
        if initial_mutual_info == perturb_mutual_info:
            recovery_mutual_info = 1.0
        else:
            recovery_mutual_info = (final_mutual_info - perturb_mutual_info) / (initial_mutual_info - perturb_mutual_info)
            recovery_mutual_info = min(max(recovery_mutual_info, 0), 1)  # Clamp to [0,1]
            
        # Calculate recovery time (steps to reach 85% of original correlation)
        post_perturb_corr = self.metrics['correlation'][self.perturbation_step:]
        threshold = 0.85 * initial_correlation
        
        try:
            recovery_time = next(i for i, c in enumerate(post_perturb_corr) if c >= threshold)
        except StopIteration:
            recovery_time = len(post_perturb_corr)  # Did not recover
        
        # Overall recovery quality (weighted average)
        recovery_quality = 0.6 * recovery_correlation + 0.4 * recovery_mutual_info
        
        # Determine if recovery was successful per roadmap criteria
        recovery_success = recovery_quality >= 0.4 and final_correlation >= 0.85 * initial_correlation
        
        # Entropy delta (change in spectral entropy)
        entropy_delta = self.metrics['spectral_entropy'][-1] - self.metrics['spectral_entropy'][0]
        
        return {
            'recovery_correlation': recovery_correlation,
            'recovery_mutual_info': recovery_mutual_info,
            'recovery_time': recovery_time,
            'recovery_quality': recovery_quality,
            'recovery_success': recovery_success,
            'final_correlation': final_correlation,
            'entropy_delta': entropy_delta
        }
    
    def visualize_key_frames(self, save_path=None):
        """Visualize key frames of the experiment"""
        if len(self.history) < 2:
            print("Not enough frames to visualize")
            return
            
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Initial state
        im0 = axes[0].imshow(self.history[0], cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title("Initial State")
        axes[0].axis('off')
        
        # After perturbation
        perturb_idx = self.perturbation_step
        im1 = axes[1].imshow(self.history[perturb_idx], cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title(f"After {self.perturbation_type.title()} ({self.perturbation_magnitude:.2f})")
        axes[1].axis('off')
        
        # Mid recovery
        mid_idx = (len(self.history) + perturb_idx) // 2
        im2 = axes[2].imshow(self.history[mid_idx], cmap='viridis', vmin=-1, vmax=1)
        axes[2].set_title(f"Mid Recovery (t={mid_idx})")
        axes[2].axis('off')
        
        # Final state
        im3 = axes[3].imshow(self.history[-1], cmap='viridis', vmin=-1, vmax=1)
        axes[3].set_title(f"Final State (t={len(self.history)-1})")
        axes[3].axis('off')
        
        plt.colorbar(im0, ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def visualize_metrics(self, save_path=None):
        """Visualize metrics over time"""
        if len(self.history) < 2:
            print("Not enough frames to visualize metrics")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Correlation
        axes[0, 0].plot(self.metrics['correlation'])
        axes[0, 0].axvline(x=self.perturbation_step, color='r', linestyle='--', label='Perturbation')
        axes[0, 0].set_title("Correlation with Initial State")
        axes[0, 0].set_xlabel("Time step")
        axes[0, 0].set_ylabel("Correlation")
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Coherence
        axes[0, 1].plot(self.metrics['coherence'])
        axes[0, 1].axvline(x=self.perturbation_step, color='r', linestyle='--', label='Perturbation')
        axes[0, 1].set_title("Field Coherence")
        axes[0, 1].set_xlabel("Time step")
        axes[0, 1].set_ylabel("Coherence")
        axes[0, 1].grid(True)
        
        # Mutual Information
        axes[1, 0].plot(self.metrics['mutual_info'])
        axes[1, 0].axvline(x=self.perturbation_step, color='r', linestyle='--', label='Perturbation')
        axes[1, 0].set_title("Mutual Information with Initial State")
        axes[1, 0].set_xlabel("Time step")
        axes[1, 0].set_ylabel("Mutual Information")
        axes[1, 0].grid(True)
        
        # Spectral Entropy
        axes[1, 1].plot(self.metrics['spectral_entropy'])
        axes[1, 1].axvline(x=self.perturbation_step, color='r', linestyle='--', label='Perturbation')
        axes[1, 1].set_title("Spectral Entropy")
        axes[1, 1].set_xlabel("Time step")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_experiment_data(self, output_dir="experiments"):
        """Save experiment data to specified directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.pattern_type}_{self.perturbation_type}_{self.perturbation_magnitude:.2f}_{timestamp}"
        exp_dir = os.path.join(output_dir, exp_name)
        
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            
        # Save parameters
        params = {
            'size': self.size,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'pattern_type': self.pattern_type,
            'perturbation_type': self.perturbation_type,
            'perturbation_magnitude': self.perturbation_magnitude,
            'perturbation_step': self.perturbation_step,
            **self.parameters
        }
        
        with open(os.path.join(exp_dir, "parameters.txt"), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)
        
        # Save recovery metrics
        recovery = self.calculate_recovery_metrics()
        with open(os.path.join(exp_dir, "recovery.txt"), 'w') as f:
            for key, value in recovery.items():
                f.write(f"{key}: {value}\n")
        
        # Save visualizations
        self.visualize_key_frames(save_path=os.path.join(exp_dir, "key_frames.png"))
        self.visualize_metrics(save_path=os.path.join(exp_dir, "metrics.png"))
        
        return exp_dir


class ExperimentRunner:
    """Class to run and analyze experiment sets according to the roadmap"""
    
    def __init__(self, output_dir="experiments"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    def run_pattern_specific_recovery(self, pattern_types, perturbation_magnitudes, 
                                     perturbation_type="flip", recovery_steps=50):
        """Run Experiment Set A: Pattern-Specific Recovery from the roadmap"""
        results = []
        
        for pattern in pattern_types:
            for magnitude in perturbation_magnitudes:
                print(f"Running experiment: {pattern} pattern with {perturbation_type} perturbation (mag={magnitude:.2f})")
                
                # Initialize experiment
                exp = RCFTExperiment(memory_strength=0.35, coupling_strength=0.5, memory_decay=0.92)
                
                # Setup and run
                exp.initialize_pattern(pattern_type=pattern)
                exp.apply_perturbation(perturbation_type=perturbation_type, magnitude=magnitude)
                exp.update(steps=recovery_steps)
                
                # Calculate results
                recovery = exp.calculate_recovery_metrics()
                recovery['pattern_type'] = pattern
                recovery['perturbation_type'] = perturbation_type
                recovery['perturbation_magnitude'] = magnitude
                results.append(recovery)
                
                # Save experiment
                exp.save_experiment_data(self.output_dir)
                
        # Compile and save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "pattern_specific_recovery_results.csv"), index=False)
        
        # Create summary visualization
        self.visualize_pattern_recovery_results(results_df)
        
        return results_df
    
    def visualize_pattern_recovery_results(self, results_df):
        """Create visualizations for pattern-specific recovery results"""
        # Group results by pattern type
        patterns = results_df['pattern_type'].unique()
        magnitudes = sorted(results_df['perturbation_magnitude'].unique())
        
        # Create recovery quality vs magnitude plot
        plt.figure(figsize=(12, 6))
        
        for pattern in patterns:
            pattern_data = results_df[results_df['pattern_type'] == pattern]
            plt.plot(pattern_data['perturbation_magnitude'], pattern_data['recovery_quality'], 
                    'o-', label=f"{pattern}")
        
        plt.axhline(y=0.4, color='r', linestyle='--', label='Recovery Threshold')
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Recovery Quality')
        plt.title('Recovery Quality by Pattern Type and Perturbation Magnitude')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "pattern_recovery_quality.png"))
        
        # Create recovery time vs magnitude plot
        plt.figure(figsize=(12, 6))
        
        for pattern in patterns:
            pattern_data = results_df[results_df['pattern_type'] == pattern]
            plt.plot(pattern_data['perturbation_magnitude'], pattern_data['recovery_time'], 
                    'o-', label=f"{pattern}")
        
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Recovery Time (steps)')
        plt.title('Recovery Time by Pattern Type and Perturbation Magnitude')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "pattern_recovery_time.png"))
        
    def run_parameter_sweep(self, alpha_range, gamma_range, beta=0.5, 
                           pattern_type="radial", perturbation_type="flip", 
                           perturbation_magnitude=1.0, recovery_steps=50):
        """Run Parameter Sweep experiment from Phase II of the roadmap"""
        results = []
        
        for alpha in alpha_range:
            for gamma in gamma_range:
                print(f"Testing α={alpha:.2f}, γ={gamma:.2f}")
                
                # Initialize experiment
                exp = RCFTExperiment(
                    memory_strength=alpha,
                    coupling_strength=beta,
                    memory_decay=gamma
                )
                
                # Setup and run
                exp.initialize_pattern(pattern_type=pattern_type)
                exp.apply_perturbation(perturbation_type=perturbation_type, magnitude=perturbation_magnitude)
                exp.update(steps=recovery_steps)
                
                # Calculate results
                recovery = exp.calculate_recovery_metrics()
                recovery['alpha'] = alpha
                recovery['beta'] = beta
                recovery['gamma'] = gamma
                results.append(recovery)
                
                # Save experiment if it's interesting (shows threshold behavior)
                if 0.3 <= recovery['recovery_quality'] <= 0.5:
                    exp.save_experiment_data(self.output_dir)
        
        # Compile and save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "parameter_sweep_results.csv"), index=False)
        
        # Create heatmap visualization
        self.visualize_parameter_sweep(results_df, alpha_range, gamma_range)
        
        return results_df
    
    def visualize_parameter_sweep(self, results_df, alpha_range, gamma_range):
        """Create heatmap visualization for parameter sweep results"""
        plt.figure(figsize=(10, 8))
        
        # Create 2D grid of recovery quality values
        recovery_grid = results_df.pivot(index="alpha", columns="gamma", values="recovery_quality")
        
        # Create heatmap
        plt.imshow(recovery_grid, cmap='viridis', origin='lower', aspect='auto',
                  extent=[min(gamma_range), max(gamma_range), min(alpha_range), max(alpha_range)])
        
        plt.colorbar(label='Recovery Quality')
        plt.xlabel('Memory Decay (γ)')
        plt.ylabel('Memory Strength (α)')
        plt.title('Recovery Quality in Memory Parameter Space')
        
        # Add contour for the recovery threshold
        CS = plt.contour(recovery_grid, levels=[0.4], colors='r', origin='lower', 
                        extent=[min(gamma_range), max(gamma_range), min(alpha_range), max(alpha_range)])
        plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "parameter_sweep_heatmap.png"))


# Example usage for running experiments
if __name__ == "__main__":
    print("Recursive Coherence Field Theory Experiment Suite")
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir="experiments")
    
    # Run Phase I, Experiment Set A as specified in the roadmap
    patterns = ["radial", "horizontal", "diagonal", "fractal", "lattice", "stochastic"]
    magnitudes = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    print("Running Phase I, Experiment Set A: Pattern-Specific Recovery")
    results = runner.run_pattern_specific_recovery(
        pattern_types=patterns,
        perturbation_magnitudes=magnitudes,
        perturbation_type="flip",
        recovery_steps=50
    )
    
    print("\nExperiment completed. Results saved to experiments/ directory.")