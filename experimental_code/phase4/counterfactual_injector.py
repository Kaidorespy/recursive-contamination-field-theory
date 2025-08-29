# phase4/counterfactual_injector.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import mutual_info_score
import pickle

from .base import DirectedMemoryExperiment

class CounterfactualGenerator:
    """Generates counterfactual 'ghost' memories with controlled similarity to true patterns"""
    
    def __init__(self, size=64, similarity_control='blend'):
        """Initialize the counterfactual generator"""
        self.size = size
        self.similarity_control = similarity_control  # 'blend', 'noise', 'transform', 'structure'
    
    def generate_counterfactual(self, true_pattern, similarity_level, method=None):
        """Generate a counterfactual memory with controlled similarity to true pattern"""
        if method is None:
            method = self.similarity_control
            
        if method == 'blend':
            return self._generate_blend_counterfactual(true_pattern, similarity_level)
        elif method == 'noise':
            return self._generate_noise_counterfactual(true_pattern, similarity_level)
        elif method == 'transform':
            return self._generate_transform_counterfactual(true_pattern, similarity_level)
        elif method == 'structure':
            return self._generate_structure_counterfactual(true_pattern, similarity_level)
        else:
            raise ValueError(f"Unknown counterfactual generation method: {method}")
    
    def _generate_blend_counterfactual(self, true_pattern, similarity_level):
        """Generate by blending with a random pattern"""
        # Create a random pattern
        random_pattern = np.random.uniform(-1, 1, true_pattern.shape)
        
        # Blend based on similarity level (0-1)
        counterfactual = similarity_level * true_pattern + (1 - similarity_level) * random_pattern
        
        # Normalize to [-1, 1]
        counterfactual = np.clip(counterfactual, -1, 1)
        
        return counterfactual
    
    def _generate_noise_counterfactual(self, true_pattern, similarity_level):
        """Generate by adding controlled noise"""
        # Calculate noise amplitude based on similarity (higher similarity = lower noise)
        noise_amplitude = 2 * (1 - similarity_level)
        
        # Create noise
        noise = np.random.normal(0, noise_amplitude, true_pattern.shape)
        
        # Add noise to true pattern
        counterfactual = true_pattern + noise
        
        # Normalize to [-1, 1]
        counterfactual = np.clip(counterfactual, -1, 1)
        
        return counterfactual
    
    def _generate_transform_counterfactual(self, true_pattern, similarity_level):
        """Generate by transforming true pattern (rotation, shift, etc.)"""
        # Map similarity to transformation parameters
        # Lower similarity = more transformation
        max_rotation = 180 * (1 - similarity_level)
        max_shift = self.size // 4 * (1 - similarity_level)
        
        # Choose a transformation type
        transform_type = np.random.choice(['rotate', 'shift', 'flip', 'scale'])
        
        if transform_type == 'rotate':
            # Rotate by a random angle
            angle = np.random.uniform(-max_rotation, max_rotation)
            from scipy.ndimage import rotate
            counterfactual = rotate(true_pattern, angle, reshape=False, mode='nearest')
            
        elif transform_type == 'shift':
            # Shift pattern
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            from scipy.ndimage import shift
            counterfactual = shift(true_pattern, (shift_y, shift_x), mode='wrap')
            
        elif transform_type == 'flip':
            # Flip pattern
            flip_axes = np.random.choice([0, 1, None])
            if flip_axes is None:
                counterfactual = true_pattern  # No flip for high similarity
            else:
                counterfactual = np.flip(true_pattern, axis=flip_axes)
                
        elif transform_type == 'scale':
            # Scale pattern
            scale_factor = 1.0 + (1.0 - similarity_level) * np.random.choice([-1, 1])
            from scipy.ndimage import zoom
            # Zoom and crop to original size
            zoomed = zoom(true_pattern, scale_factor, order=1)
            # Extract center region or pad if smaller
            if scale_factor > 1.0:
                start_x = (zoomed.shape[0] - self.size) // 2
                start_y = (zoomed.shape[1] - self.size) // 2
                counterfactual = zoomed[start_x:start_x+self.size, start_y:start_y+self.size]
            else:
                padding = ((self.size - zoomed.shape[0]) // 2, ) * 2
                counterfactual = np.pad(zoomed, padding, mode='constant')
                
            # Ensure correct shape
            if counterfactual.shape != true_pattern.shape:
                counterfactual = np.resize(counterfactual, true_pattern.shape)
        
        # Normalize to [-1, 1]
        counterfactual = np.clip(counterfactual, -1, 1)
        
        return counterfactual
    
    def _generate_structure_counterfactual(self, true_pattern, similarity_level):
        """Generate by preserving structural properties but changing details"""
        # Use FFT to capture structure
        fft = np.fft.fft2(true_pattern)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Create new phase information with controlled randomness
        random_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
        new_phase = similarity_level * phase + (1 - similarity_level) * random_phase
        
        # Reconstruct with original magnitude but new phase
        new_fft = magnitude * np.exp(1j * new_phase)
        counterfactual = np.real(np.fft.ifft2(new_fft))
        
        # Normalize to [-1, 1]
        counterfactual = 2 * (counterfactual - np.min(counterfactual)) / (np.max(counterfactual) - np.min(counterfactual)) - 1
        
        return counterfactual
        
    def calculate_similarity(self, pattern1, pattern2, method='correlation'):
        """Calculate similarity between two patterns"""
        if method == 'correlation':
            # Flatten and compute correlation
            return np.corrcoef(pattern1.flatten(), pattern2.flatten())[0, 1]
            
        elif method == 'mutual_info':
            # Discretize for mutual information
            bins = 10
            p1_bins = np.floor(bins * (pattern1.flatten() + 1) / 2).astype(int)
            p2_bins = np.floor(bins * (pattern2.flatten() + 1) / 2).astype(int)
            return mutual_info_score(p1_bins, p2_bins)
            
        elif method == 'mse':
            # Mean squared error (convert to similarity)
            mse = np.mean((pattern1 - pattern2) ** 2)
            return 1.0 / (1.0 + mse)  # Higher value = more similar
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")

class InterferenceAnalyzer:
    """Analyzes interference effects between true memories and counterfactuals"""
    
    def __init__(self, 
                 catastrophic_threshold=0.3, 
                 graceful_threshold=0.7,
                 similarity_method='correlation'):
        """Initialize the interference analyzer"""
        self.catastrophic_threshold = catastrophic_threshold
        self.graceful_threshold = graceful_threshold
        self.similarity_method = similarity_method
        
        # Storage for analysis results
        self.interference_results = []
    
    def analyze_interference(self, baseline_metrics, interference_metrics, 
                           similarity_level, counterfactual_pattern_id):
        """Analyze interference effects between baseline and post-interference metrics"""
        # Calculate key interference metrics
        recovery_drop = baseline_metrics['recovery_quality'] - interference_metrics['recovery_quality']
        correlation_drop = baseline_metrics['final_correlation'] - interference_metrics['final_correlation']
        trace_drop = baseline_metrics.get('trace_strength', 1.0) - interference_metrics.get('trace_strength', 0.0)
        
        # Classify interference level
        if recovery_drop > 1 - self.catastrophic_threshold:
            interference_level = "catastrophic"
        elif recovery_drop < 1 - self.graceful_threshold:
            interference_level = "graceful"
        else:
            interference_level = "moderate"
            
        # Compute interference index (normalized)
        interference_index = recovery_drop / max(baseline_metrics['recovery_quality'], 0.001)
        
        # Create result
        result = {
            'similarity_level': similarity_level,
            'counterfactual_id': counterfactual_pattern_id,
            'recovery_drop': recovery_drop,
            'correlation_drop': correlation_drop,
            'trace_drop': trace_drop,
            'interference_index': interference_index,
            'interference_level': interference_level,
            'baseline_recovery': baseline_metrics['recovery_quality'],
            'interference_recovery': interference_metrics['recovery_quality']
        }
        
        self.interference_results.append(result)
        
        return result
    
    def summarize_interference(self):
        """Generate summary statistics of interference effects"""
        if not self.interference_results:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(self.interference_results)
        
        # Group by similarity level
        grouped = df.groupby('similarity_level').agg({
            'recovery_drop': ['mean', 'std'],
            'correlation_drop': ['mean', 'std'],
            'interference_index': ['mean', 'std'],
            'interference_level': lambda x: x.value_counts().index[0]  # Most common level
        })
        
        # Flatten multi-index columns
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
        
        # Count occurrences of interference levels
        level_counts = df['interference_level'].value_counts()
        
        summary = {
            'by_similarity': grouped.to_dict(),
            'interference_levels': level_counts.to_dict(),
            'total_trials': len(df),
            'mean_interference_index': df['interference_index'].mean()
        }
        
        return summary
    
    def visualize_interference(self, output_dir):
        """Create visualizations of interference effects"""
        if not self.interference_results:
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.interference_results)
        
        # Create interference by similarity plot
        plt.figure(figsize=(10, 6))
        
        # Group by similarity
        grouped = df.groupby('similarity_level')['interference_index'].agg(['mean', 'std']).reset_index()
        
        # Plot mean with error bars
        plt.errorbar(grouped['similarity_level'], grouped['mean'], yerr=grouped['std'], 
                   fmt='o-', capsize=5, label='Interference Index')
        
        # Add thresholds
        plt.axhline(y=1-self.graceful_threshold, color='green', linestyle='--', 
                  label=f'Graceful ({1-self.graceful_threshold})')
        plt.axhline(y=1-self.catastrophic_threshold, color='red', linestyle='--', 
                  label=f'Catastrophic ({1-self.catastrophic_threshold})')
        
        plt.title('Interference Index by Similarity Level')
        plt.xlabel('Similarity Level')
        plt.ylabel('Interference Index')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "interference_by_similarity.png"))
        plt.close()
        
        # Create interference level distribution
        plt.figure(figsize=(8, 6))
        
        level_counts = df['interference_level'].value_counts()
        
        plt.bar(level_counts.index, level_counts.values, 
               color=['green', 'yellow', 'red'])
        
        plt.title('Distribution of Interference Levels')
        plt.xlabel('Interference Level')
        plt.ylabel('Count')
        
        # Add count labels
        for i, v in enumerate(level_counts.values):
            plt.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "interference_level_distribution.png"))
        plt.close()
        
        # Create recovery drop vs similarity scatter plot
        plt.figure(figsize=(10, 6))
        
        plt.scatter(df['similarity_level'], df['recovery_drop'], 
                  c=df['interference_index'], cmap='viridis', 
                  alpha=0.7, s=50)
        
        plt.colorbar(label='Interference Index')
        plt.title('Recovery Quality Drop vs. Similarity Level')
        plt.xlabel('Similarity Level')
        plt.ylabel('Recovery Quality Drop')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recovery_drop_scatter.png"))
        plt.close()
        
        # Create similarity-recovery heatmap if we have enough data
        similarity_levels = df['similarity_level'].unique()
        if len(similarity_levels) >= 3:
            try:
                # Pivot data for heatmap
                sim_levels = sorted(df['similarity_level'].unique())
                recovery_bins = np.linspace(0, 1, 5)  # 5 recovery quality bins
                recovery_centers = (recovery_bins[:-1] + recovery_bins[1:]) / 2
                
                heatmap_data = np.zeros((len(sim_levels), len(recovery_centers)))
                
                for i, sim in enumerate(sim_levels):
                    sim_data = df[df['similarity_level'] == sim]
                    hist, _ = np.histogram(sim_data['interference_recovery'], bins=recovery_bins)
                    heatmap_data[i, :] = hist
                
                # Normalize each row
                row_sums = heatmap_data.sum(axis=1, keepdims=True)
                heatmap_data = heatmap_data / np.maximum(row_sums, 1)  # Avoid division by zero
                
                plt.figure(figsize=(10, 8))
                plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                
                # Set tick labels
                plt.yticks(range(len(sim_levels)), [f"{s:.2f}" for s in sim_levels])
                plt.xticks(range(len(recovery_centers)), [f"{s:.2f}" for s in recovery_centers])
                
                plt.colorbar(label='Frequency')
                plt.title('Recovery Quality Distribution by Similarity Level')
                plt.xlabel('Recovery Quality After Interference')
                plt.ylabel('Similarity Level')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "recovery_heatmap.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error creating heatmap: {e}")

class CounterfactualInjector(DirectedMemoryExperiment):
    """Injects counterfactual memories and studies interference effects"""
    
    def __init__(self, 
                 output_dir="phase4_results/counterfactual_injector",
                 alpha=0.35, 
                 beta=0.5, 
                 gamma=0.92, 
                 pattern_type="fractal",
                 max_steps=100,
                 catastrophic_threshold=0.3,
                 graceful_threshold=0.7):
        """Initialize the counterfactual injector"""
        super().__init__(output_dir, alpha, beta, gamma, pattern_type)
        
        self.max_steps = max_steps
        self.catastrophic_threshold = catastrophic_threshold
        self.graceful_threshold = graceful_threshold
        
        # Create component objects
        self.generator = CounterfactualGenerator(size=64)
        self.analyzer = InterferenceAnalyzer(
            catastrophic_threshold=catastrophic_threshold,
            graceful_threshold=graceful_threshold
        )
        
        # Create output subdirectories
        os.makedirs(os.path.join(output_dir, "true_patterns"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "counterfactuals"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "interference_analysis"), exist_ok=True)
        
        # Storage for patterns and results
        self.true_patterns = {}
        self.counterfactual_patterns = {}
        self.interference_results = []
    
    def create_true_pattern(self, pattern_id, pattern_type=None, **kwargs):
        """Create a true pattern to use as baseline"""
        if pattern_type is None:
            pattern_type = self.pattern_type
            
        # Initialize experiment
        self.initialize_experiment()
        
        # Initialize pattern
        self.experiment.initialize_pattern(pattern_type=pattern_type, **kwargs)
        
        # Store pattern
        pattern = self.experiment.state.copy()
        self.true_patterns[pattern_id] = pattern
        
        # Save visualization
        self._save_pattern_visualization(pattern_id, pattern, "true")
        
        return pattern
    
    def create_counterfactual(self, true_pattern_id, cf_pattern_id, similarity_level, method=None):
        """Create a counterfactual pattern with controlled similarity to a true pattern"""
        if true_pattern_id not in self.true_patterns:
            raise ValueError(f"True pattern '{true_pattern_id}' not found")
            
        # Get true pattern
        true_pattern = self.true_patterns[true_pattern_id]
        
        # Generate counterfactual
        counterfactual = self.generator.generate_counterfactual(
            true_pattern, 
            similarity_level, 
            method=method
        )
        
        # Calculate actual similarity
        actual_similarity = self.generator.calculate_similarity(true_pattern, counterfactual)
        
        # Store counterfactual with metadata
        self.counterfactual_patterns[cf_pattern_id] = {
            'pattern': counterfactual,
            'true_pattern_id': true_pattern_id,
            'similarity_level': similarity_level,
            'actual_similarity': actual_similarity,
            'method': method or self.generator.similarity_control
        }
        
        # Save visualization
        self._save_pattern_visualization(cf_pattern_id, counterfactual, "counterfactual")
        
        # Create comparison visualization
        self._save_comparison_visualization(true_pattern_id, cf_pattern_id)
        
        return counterfactual, actual_similarity
    
    def _save_pattern_visualization(self, pattern_id, pattern, pattern_type="true"):
        """Save visualization of a pattern"""
        plt.figure(figsize=(6, 6))
        plt.imshow(pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        
        if pattern_type == "true":
            title = f"True Pattern: {pattern_id}"
            save_dir = os.path.join(self.output_dir, "true_patterns")
        else:
            title = f"Counterfactual Pattern: {pattern_id}"
            save_dir = os.path.join(self.output_dir, "counterfactuals")
            
        plt.title(title)
        plt.axis('off')
        
        filename = f"pattern_{pattern_id}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
    
    def _save_comparison_visualization(self, true_id, cf_id):
        """Save comparison between true and counterfactual patterns"""
        # Get patterns
        true_pattern = self.true_patterns[true_id]
        cf_data = self.counterfactual_patterns[cf_id]
        cf_pattern = cf_data['pattern']
        
        plt.figure(figsize=(15, 5))
        
        # Plot true pattern
        plt.subplot(1, 3, 1)
        plt.imshow(true_pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"True Pattern: {true_id}")
        plt.axis('off')
        
        # Plot counterfactual
        plt.subplot(1, 3, 2)
        plt.imshow(cf_pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {cf_id}\nSimilarity: {cf_data['actual_similarity']:.3f}")
        plt.axis('off')
        
        # Plot difference
        plt.subplot(1, 3, 3)
        plt.imshow(cf_pattern - true_pattern, cmap='RdBu', vmin=-1, vmax=1)
        plt.title("Difference")
        plt.axis('off')
        
        plt.suptitle(f"Similarity: {cf_data['similarity_level']} (Target), {cf_data['actual_similarity']:.3f} (Actual)")
        plt.tight_layout()
        
        filename = f"compare_{true_id}_vs_{cf_id}.png"
        plt.savefig(os.path.join(self.output_dir, "counterfactuals", filename))
        plt.close()
    
    def measure_baseline_performance(self, pattern_id, num_trials=3):
        """Measure baseline recovery performance for a pattern"""
        if pattern_id not in self.true_patterns:
            raise ValueError(f"Pattern '{pattern_id}' not found")
            
        # Run multiple trials
        results = []
        
        for trial in range(num_trials):
            # Initialize experiment
            self.initialize_experiment()
            
            # Set state to pattern
            self.experiment.state = self.true_patterns[pattern_id].copy()
            self.initial_state = self.true_patterns[pattern_id].copy()
            
            # Apply perturbation
            self.experiment.apply_perturbation(perturbation_type="flip", magnitude=1.0)
            
            # Let system recover
            self.experiment.update(steps=self.max_steps)
            
            # Calculate recovery metrics
            metrics = self.compute_metrics()
            recovery = self.experiment.calculate_recovery_metrics()
            
            # Add to results
            result = {
                'pattern_id': pattern_id,
                'trial': trial,
                'final_correlation': metrics['correlation'],
                'final_coherence': metrics['coherence'],
                'recovery_quality': recovery.get('recovery_quality', 0.0),
                'recovery_time': recovery.get('recovery_time', self.max_steps)
            }
            
            results.append(result)
        
        # Compute average metrics
        avg_metrics = {
            'pattern_id': pattern_id,
            'final_correlation': np.mean([r['final_correlation'] for r in results]),
            'final_coherence': np.mean([r['final_coherence'] for r in results]),
            'recovery_quality': np.mean([r['recovery_quality'] for r in results]),
            'recovery_time': np.mean([r['recovery_time'] for r in results]),
            'trace_strength': 1.0  # Baseline strength
        }
        
        return avg_metrics
    
    def run_interference_experiment(self, true_pattern_id, cf_pattern_id, 
                                  interference_strength=0.5,
                                  learning_trials=3, 
                                  interference_trials=3,
                                  recovery_trials=3):
        """Run a complete interference experiment"""
        # Check patterns exist
        if true_pattern_id not in self.true_patterns:
            raise ValueError(f"True pattern '{true_pattern_id}' not found")
            
        if cf_pattern_id not in self.counterfactual_patterns:
            raise ValueError(f"Counterfactual pattern '{cf_pattern_id}' not found")
            
        # Get counterfactual info
        cf_info = self.counterfactual_patterns[cf_pattern_id]
        similarity_level = cf_info['actual_similarity']
        
        # Phase 1: Measure baseline performance
        print(f"Measuring baseline performance for pattern {true_pattern_id}...")
        baseline_metrics = self.measure_baseline_performance(
            true_pattern_id, 
            num_trials=learning_trials
        )
        
        # Phase 2: Train memory on true pattern
        print(f"Training memory system on true pattern {true_pattern_id}...")
        memory_field = np.zeros((self.experiment.size, self.experiment.size))
        
        for trial in range(learning_trials):
            # Initialize experiment
            self.initialize_experiment()
            
            # Set state to pattern
            self.experiment.state = self.true_patterns[true_pattern_id].copy()
            
            # Let system evolve without perturbation
            self.experiment.update(steps=self.max_steps // 2)
            
            # Accumulate memory
            memory_field += self.experiment.memory * (1.0 / learning_trials)
        
        # Phase 3: Interfere with counterfactual
        print(f"Interfering with counterfactual pattern {cf_pattern_id}...")
        
        # Mix memory with counterfactual based on interference strength
        interfered_memory = (1 - interference_strength) * memory_field + \
                           interference_strength * cf_info['pattern']
        
        # Phase 4: Test recovery after interference
        print(f"Testing recovery after interference...")
        interference_results = []
        
        for trial in range(recovery_trials):
            # Initialize experiment
            self.initialize_experiment()
            
            # Set state to true pattern
            self.experiment.state = self.true_patterns[true_pattern_id].copy()
            self.initial_state = self.true_patterns[true_pattern_id].copy()
            
            # Set memory to interfered memory
            self.experiment.memory = interfered_memory.copy()
            
            # Apply perturbation
            self.experiment.apply_perturbation(perturbation_type="flip", magnitude=1.0)
            
            # Let system recover
            self.experiment.update(steps=self.max_steps)
            
            # Calculate recovery metrics
            metrics = self.compute_metrics()
            recovery = self.experiment.calculate_recovery_metrics()
            
            # Add to results
            result = {
                'pattern_id': true_pattern_id,
                'cf_pattern_id': cf_pattern_id,
                'trial': trial,
                'final_correlation': metrics['correlation'],
                'final_coherence': metrics['coherence'],
                'recovery_quality': recovery.get('recovery_quality', 0.0),
                'recovery_time': recovery.get('recovery_time', self.max_steps)
            }
            
            interference_results.append(result)
            
            # Save visualization of the trial
            self._save_interference_trial_visualization(
                true_pattern_id, 
                cf_pattern_id, 
                interfered_memory,
                trial,
                recovery.get('recovery_quality', 0.0)
            )
        
        # Compute average interference metrics
        avg_interference_metrics = {
            'pattern_id': true_pattern_id,
            'final_correlation': np.mean([r['final_correlation'] for r in interference_results]),
            'final_coherence': np.mean([r['final_coherence'] for r in interference_results]),
            'recovery_quality': np.mean([r['recovery_quality'] for r in interference_results]),
            'recovery_time': np.mean([r['recovery_time'] for r in interference_results]),
            'trace_strength': 1.0 - interference_strength  # Approximate
        }
        
        # Analyze interference
        interference_analysis = self.analyzer.analyze_interference(
            baseline_metrics, 
            avg_interference_metrics,
            similarity_level,
            cf_pattern_id
        )
        
        # Store result
        result = {
            'true_pattern_id': true_pattern_id,
            'cf_pattern_id': cf_pattern_id,
            'similarity_level': similarity_level,
            'interference_strength': interference_strength,
            'baseline_metrics': baseline_metrics,
            'interference_metrics': avg_interference_metrics,
            'interference_analysis': interference_analysis
        }
        
        self.interference_results.append(result)
        
        # Save detailed visualization
        self._save_interference_summary_visualization(result)
        
        return result
    
    def _save_interference_trial_visualization(self, true_id, cf_id, 
                                            interfered_memory, trial, recovery_quality):
        """Save visualization of an interference trial"""
        plt.figure(figsize=(15, 5))
        
        # Plot true pattern
        plt.subplot(1, 3, 1)
        plt.imshow(self.true_patterns[true_id], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"True Pattern: {true_id}")
        plt.axis('off')
        
        # Plot interfered memory
        plt.subplot(1, 3, 2)
        plt.imshow(interfered_memory, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Interfered Memory")
        plt.axis('off')
        
        # Plot final state
        plt.subplot(1, 3, 3)
        plt.imshow(self.experiment.state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Recovered State\nQuality: {recovery_quality:.3f}")
        plt.axis('off')
        
        plt.suptitle(f"Interference Trial {trial}: {true_id} vs {cf_id}")
        plt.tight_layout()
        
        filename = f"interference_{true_id}_vs_{cf_id}_trial{trial}.png"
        plt.savefig(os.path.join(self.output_dir, "interference_analysis", filename))
        plt.close()
    
    def _save_interference_summary_visualization(self, result):
        """Save summary visualization of interference experiment"""
        true_id = result['true_pattern_id']
        cf_id = result['cf_pattern_id']
        
        plt.figure(figsize=(12, 10))
        
        # Create a 2x2 grid
        # Plot true pattern
        plt.subplot(2, 2, 1)
        plt.imshow(self.true_patterns[true_id], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"True Pattern: {true_id}")
        plt.axis('off')
        
        # Plot counterfactual
        plt.subplot(2, 2, 2)
        plt.imshow(self.counterfactual_patterns[cf_id]['pattern'], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {cf_id}\nSimilarity: {result['similarity_level']:.3f}")
        plt.axis('off')
        
        # Plot recovery comparison
        plt.subplot(2, 2, 3)
        bars = ['Baseline', 'Post-Interference']
        heights = [
            result['baseline_metrics']['recovery_quality'],
            result['interference_metrics']['recovery_quality']
        ]
        
        plt.bar(bars, heights, color=['green', 'red'])
        plt.ylim(0, 1)
        plt.title("Recovery Quality")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(heights):
            plt.text(i, v + 0.05, f"{v:.3f}", ha='center')
            
        # Highlight drop
        plt.annotate(
            f"Drop: {result['interference_analysis']['recovery_drop']:.3f}",
            xy=(0.5, min(heights) - 0.1),
            xytext=(0.5, min(heights) - 0.2),
            arrowprops=dict(arrowstyle="->"),
            ha='center'
        )
        
        # Plot interference summary
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.9, f"Interference Analysis", ha='center', fontsize=12, fontweight='bold')
        plt.text(0.5, 0.8, f"Similarity Level: {result['similarity_level']:.3f}", ha='center')
        plt.text(0.5, 0.7, f"Interference Strength: {result['interference_strength']:.3f}", ha='center')
        plt.text(0.5, 0.6, f"Interference Index: {result['interference_analysis']['interference_index']:.3f}", ha='center')
        plt.text(0.5, 0.5, f"Interference Level: {result['interference_analysis']['interference_level']}", ha='center')
        plt.text(0.5, 0.4, f"Recovery Drop: {result['interference_analysis']['recovery_drop']:.3f}", ha='center')
        plt.text(0.5, 0.3, f"Correlation Drop: {result['interference_analysis']['correlation_drop']:.3f}", ha='center')
        plt.axis('off')
        
        plt.suptitle(f"Interference Experiment: {true_id} vs {cf_id}", fontsize=14)
        plt.tight_layout()
        
        filename = f"interference_summary_{true_id}_vs_{cf_id}.png"
        plt.savefig(os.path.join(self.output_dir, "interference_analysis", filename))
        plt.close()
    
    def run_similarity_sweep(self, true_pattern_id, similarity_levels, 
                           method=None, interference_strength=0.5,
                           trials_per_level=3, parallel=False):
        """Run interference experiments across multiple similarity levels"""
        if true_pattern_id not in self.true_patterns:
            raise ValueError(f"True pattern '{true_pattern_id}' not found")
            
        results = []
        
        # Create counterfactuals for each similarity level
        counterfactuals = []
        
        for i, similarity in enumerate(similarity_levels):
            cf_id = f"{true_pattern_id}_cf_{i}"
            counterfactual, actual_similarity = self.create_counterfactual(
                true_pattern_id, 
                cf_id, 
                similarity, 
                method=method
            )
            
            counterfactuals.append({
                'cf_id': cf_id,
                'similarity': similarity,
                'actual_similarity': actual_similarity
            })
        
        # Run interference experiments
        if parallel and len(counterfactuals) > 1:
            # Parallel execution
            with ProcessPoolExecutor() as executor:
                futures = []
                
                for cf in counterfactuals:
                    futures.append(
                        executor.submit(
                            self.run_interference_experiment,
                            true_pattern_id,
                            cf['cf_id'],
                            interference_strength,
                            trials_per_level,
                            trials_per_level,
                            trials_per_level
                        )
                    )
                    
                # Collect results
                for future in futures:
                    result = future.result()
                    results.append(result)
        else:
            # Sequential execution
            for cf in tqdm(counterfactuals, desc="Similarity Levels"):
                result = self.run_interference_experiment(
                    true_pattern_id,
                    cf['cf_id'],
                    interference_strength,
                    trials_per_level,
                    trials_per_level,
                    trials_per_level
                )
                
                results.append(result)
        
        # Analyze and visualize overall results
        self.analyzer.visualize_interference(
            os.path.join(self.output_dir, "interference_analysis")
        )
        
        # Create interference threshold visualization
        self._create_threshold_visualization(true_pattern_id, results)
        
        return results
    
    def _create_threshold_visualization(self, true_pattern_id, results):
        """Create visualization of interference thresholds"""
        # Extract similarity levels and recovery qualities
        similarities = [r['similarity_level'] for r in results]
        baseline_qualities = [r['baseline_metrics']['recovery_quality'] for r in results]
        interference_qualities = [r['interference_metrics']['recovery_quality'] for r in results]
        
        # Sort by similarity level
        sorted_indices = np.argsort(similarities)
        similarities = [similarities[i] for i in sorted_indices]
        baseline_qualities = [baseline_qualities[i] for i in sorted_indices]
        interference_qualities = [interference_qualities[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        
        # Plot both qualities
        plt.plot(similarities, baseline_qualities, 'o-', color='green', 
               label='Baseline Recovery')
        plt.plot(similarities, interference_qualities, 'o-', color='red', 
               label='Post-Interference Recovery')
        
        # Fill area between curves
        plt.fill_between(similarities, baseline_qualities, interference_qualities, 
                       color='orange', alpha=0.3, label='Interference Gap')
        
        # Add threshold lines
        plt.axhline(y=self.catastrophic_threshold, color='red', linestyle='--', 
                  label=f'Catastrophic Threshold ({self.catastrophic_threshold})')
        plt.axhline(y=self.graceful_threshold, color='green', linestyle='--', 
                  label=f'Graceful Threshold ({self.graceful_threshold})')
        
        # Find approximate similarity threshold for catastrophic interference
        try:
            from scipy.interpolate import interp1d
            
            # Create interpolation function
            interp = interp1d(similarities, interference_qualities, 
                            kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Find where interference quality crosses catastrophic threshold
            sim_range = np.linspace(min(similarities), max(similarities), 1000)
            qual_range = interp(sim_range)
            
            catastrophic_idx = np.abs(qual_range - self.catastrophic_threshold).argmin()
            catastrophic_sim = sim_range[catastrophic_idx]
            
            # Find where interference quality crosses graceful threshold
            graceful_idx = np.abs(qual_range - self.graceful_threshold).argmin()
            graceful_sim = sim_range[graceful_idx]
            
            # Add vertical lines at thresholds
            plt.axvline(x=catastrophic_sim, color='red', linestyle=':', 
                      label=f'Catastrophic Similarity ({catastrophic_sim:.3f})')
            plt.axvline(x=graceful_sim, color='green', linestyle=':', 
                      label=f'Graceful Similarity ({graceful_sim:.3f})')
            
        except (ImportError, ValueError, IndexError):
            pass
        
        plt.title(f'Interference Thresholds for Pattern {true_pattern_id}')
        plt.xlabel('Similarity Level')
        plt.ylabel('Recovery Quality')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "interference_analysis", 
                               f"thresholds_{true_pattern_id}.png"))
        plt.close()
    
    def run_method_comparison(self, true_pattern_id, similarity_level=0.5, 
                            methods=None, interference_strength=0.5,
                            trials_per_method=3):
        """Compare different counterfactual generation methods"""
        if true_pattern_id not in self.true_patterns:
            raise ValueError(f"True pattern '{true_pattern_id}' not found")
            
        if methods is None:
            methods = ['blend', 'noise', 'transform', 'structure']
            
        results = []
        
        # Run experiment for each method
        for method in tqdm(methods, desc="Generation Methods"):
            # Create counterfactual
            cf_id = f"{true_pattern_id}_{method}"
            _, actual_similarity = self.create_counterfactual(
                true_pattern_id, 
                cf_id, 
                similarity_level, 
                method=method
            )
            
            # Run interference experiment
            result = self.run_interference_experiment(
                true_pattern_id,
                cf_id,
                interference_strength,
                trials_per_method,
                trials_per_method,
                trials_per_method
            )
            
            results.append(result)
        
        # Create method comparison visualization
        self._create_method_comparison_visualization(true_pattern_id, results)
        
        return results
    
    def _create_method_comparison_visualization(self, true_pattern_id, results):
        """Create visualization comparing counterfactual generation methods"""
        # Extract methods and metrics
        methods = [r['cf_pattern_id'].split('_')[-1] for r in results]
        similarities = [r['similarity_level'] for r in results]
        interference_indices = [r['interference_analysis']['interference_index'] for r in results]
        recovery_drops = [r['interference_analysis']['recovery_drop'] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, similarities, width, label='Similarity Level')
        plt.bar(x + width/2, interference_indices, width, label='Interference Index')
        
        plt.xlabel('Counterfactual Generation Method')
        plt.ylabel('Value')
        plt.title(f'Method Comparison - Pattern {true_pattern_id}')
        plt.xticks(x, methods)
        plt.legend()
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "interference_analysis", 
                               f"method_comparison_{true_pattern_id}.png"))
        plt.close()
        
        # Create recovery drop comparison
        plt.figure(figsize=(10, 6))
        
        colors = ['green' if drop < 1-self.graceful_threshold else 
                 'orange' if drop < 1-self.catastrophic_threshold else 
                 'red' for drop in recovery_drops]
        
        plt.bar(methods, recovery_drops, color=colors)
        
        plt.axhline(y=1-self.graceful_threshold, color='green', linestyle='--', 
                  label=f'Graceful Threshold ({1-self.graceful_threshold})')
        plt.axhline(y=1-self.catastrophic_threshold, color='red', linestyle='--', 
                  label=f'Catastrophic Threshold ({1-self.catastrophic_threshold})')
        
        plt.xlabel('Counterfactual Generation Method')
        plt.ylabel('Recovery Quality Drop')
        plt.title(f'Recovery Drop by Method - Pattern {true_pattern_id}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add drop values
        for i, v in enumerate(recovery_drops):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "interference_analysis", 
                               f"recovery_drop_by_method_{true_pattern_id}.png"))
        plt.close()
    
    def run(self, pattern_ids=None, similarity_levels=None, methods=None, **kwargs):
        """Run a comprehensive set of counterfactual experiments"""
        if pattern_ids is None:
            pattern_ids = ['pattern_A']
            
        if similarity_levels is None:
            similarity_levels = [0.2, 0.4, 0.6, 0.8]
            
        if methods is None:
            methods = ['blend']
            
        # Create true patterns if needed
        for pid in pattern_ids:
            if pid not in self.true_patterns:
                self.create_true_pattern(pid)
        
        # Run experiments for each pattern
        all_results = {}
        
        for pid in pattern_ids:
            # Run similarity sweep
            if len(similarity_levels) > 1:
                print(f"Running similarity sweep for pattern {pid}...")
                sweep_results = self.run_similarity_sweep(
                    pid,
                    similarity_levels,
                    method=methods[0],
                    interference_strength=kwargs.get('interference_strength', 0.5),
                    trials_per_level=kwargs.get('trials_per_level', 3),
                    parallel=kwargs.get('parallel', False)
                )
                all_results[f"{pid}_similarity_sweep"] = sweep_results
            
            # Run method comparison if multiple methods
            if len(methods) > 1:
                print(f"Running method comparison for pattern {pid}...")
                method_results = self.run_method_comparison(
                    pid,
                    similarity_level=kwargs.get('similarity_level', 0.5),
                    methods=methods,
                    interference_strength=kwargs.get('interference_strength', 0.5),
                    trials_per_method=kwargs.get('trials_per_method', 3)
                )
                all_results[f"{pid}_method_comparison"] = method_results
        
        # Create summary across all experiments
        self._create_experiment_summary()
        
        return all_results
    
    def _create_experiment_summary(self):
        """Create overall summary of all experiments"""
        if not self.interference_results:
            return
            
        # Create summary JSON
        summary = {
            'total_experiments': len(self.interference_results),
            'true_patterns': list(self.true_patterns.keys()),
            'counterfactuals': len(self.counterfactual_patterns),
            'interference_analysis': self.analyzer.summarize_interference()
        }
        
        # Save summary
        with open(os.path.join(self.output_dir, "experiment_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary visualization
        self._create_summary_visualization()
    
    def _create_summary_visualization(self):
        """Create overall summary visualization"""
        if not self.interference_results:
            return
            
        # Create overall interference level distribution
        analysis_summary = self.analyzer.summarize_interference()
        
        if analysis_summary:
            plt.figure(figsize=(10, 6))
            
            level_counts = analysis_summary['interference_levels']
            labels = list(level_counts.keys())
            sizes = list(level_counts.values())
            
            colors = ['green' if l == 'graceful' else 
                     'orange' if l == 'moderate' else 
                     'red' for l in labels]
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                  startangle=90, shadow=True)
            plt.axis('equal')
            
            plt.title('Distribution of Interference Levels')
            
            plt.savefig(os.path.join(self.output_dir, "interference_level_distribution.png"))
            plt.close()
        
        # Create similarity vs interference index scatter plot
        df = pd.DataFrame([
            {
                'similarity': r['similarity_level'],
                'interference_index': r['interference_analysis']['interference_index'],
                'recovery_drop': r['interference_analysis']['recovery_drop'],
                'cf_method': r['cf_pattern_id'].split('_')[-1] if '_' in r['cf_pattern_id'] else 'unknown'
            } for r in self.interference_results
        ])
        
        plt.figure(figsize=(10, 8))
        
        # Color by method if possible
        unique_methods = df['cf_method'].unique()
        if len(unique_methods) > 1:
            for method in unique_methods:
                method_data = df[df['cf_method'] == method]
                plt.scatter(method_data['similarity'], method_data['interference_index'], 
                          label=method, alpha=0.7, s=50)
            plt.legend()
        else:
            plt.scatter(df['similarity'], df['interference_index'], alpha=0.7, s=50)
        
        # Add regression line if scipy available
        try:
            from scipy.stats import linregress
            
            if len(df) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(
                    df['similarity'], df['interference_index']
                )
                
                x_range = np.linspace(min(df['similarity']), max(df['similarity']), 100)
                plt.plot(x_range, intercept + slope * x_range, 'r--', 
                       label=f'Trend: y={slope:.3f}x+{intercept:.3f} (R²={r_value**2:.3f})')
                plt.legend()
        except ImportError:
            pass
        
        plt.title('Similarity vs. Interference Index')
        plt.xlabel('Similarity Level')
        plt.ylabel('Interference Index')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, "similarity_vs_interference.png"))
        plt.close()
    
    def visualize_results(self, **kwargs):
        """Create comprehensive visualization of results"""
        # Additional visualizations can be added here based on specific needs
        pass