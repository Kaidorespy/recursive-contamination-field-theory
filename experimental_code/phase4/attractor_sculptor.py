# phase4/attractor_sculptor.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.fftpack import fft2, fftshift
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from .base import DirectedMemoryExperiment

class AttractorFingerprinter:
    """Generates fingerprints for attractor identification using PCA and FFT"""
    
    def __init__(self, n_components=5):
        """Initialize the fingerprinter"""
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        
    def compute_fingerprint(self, original_field, final_field):
        """Compute an attractor fingerprint from original and final fields"""
        # Compute residual field
        residual = final_field - original_field
        
        # Flatten for PCA
        residual_flat = residual.flatten().reshape(1, -1)
        
        # Apply PCA if we have enough samples to fit
        if not self.fitted:
            # We can't fit with just one sample, so return raw components for now
            pca_components = residual_flat[0, :self.n_components]
        else:
            pca_components = self.pca.transform(residual_flat)[0]
            
        # Compute FFT
        fft_result = fft2(residual)
        fft_shifted = fftshift(fft_result)
        
        # Extract magnitude spectrum (log scale)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        
        # Sample key points from magnitude spectrum (center and cardinal points)
        center = magnitude_spectrum.shape[0] // 2
        sample_points = [
            (center, center),  # Center
            (center - 10, center),  # North
            (center + 10, center),  # South
            (center, center - 10),  # West
            (center, center + 10),  # East
        ]
        
        fft_components = [magnitude_spectrum[y, x] for x, y in sample_points]
        
        # Combine PCA and FFT components
        fingerprint = np.concatenate([pca_components[:self.n_components], fft_components])
        
        return fingerprint
    
    def fit(self, residuals):
        """Fit the PCA model with a collection of residuals"""
        if len(residuals) > self.n_components:
            self.pca.fit(residuals)
            self.fitted = True
            return True
        return False
    
    def visualize_fingerprint(self, fingerprint, ax=None):
        """Visualize a fingerprint for comparison"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            
        # Split into PCA and FFT components
        pca_part = fingerprint[:self.n_components]
        fft_part = fingerprint[self.n_components:]
        
        # Plot PCA components
        ax.bar(
            range(len(pca_part)), 
            pca_part, 
            color='blue', 
            alpha=0.7, 
            label='PCA Components'
        )
        
        # Plot FFT components
        ax.bar(
            range(len(pca_part), len(fingerprint)), 
            fft_part, 
            color='red', 
            alpha=0.7, 
            label='FFT Components'
        )
        
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Component Value')
        ax.set_title('Attractor Fingerprint')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax

class AttractorSculptor(DirectedMemoryExperiment):
    """Sculpts attractor basins through meta-perturbations"""
    
    def __init__(self, 
                 output_dir="phase4_results/attractor_sculptor",
                 alpha=0.35, 
                 beta=0.5, 
                 gamma=0.92, 
                 pattern_type="fractal",
                 max_steps=100,
                 n_fingerprint_components=5):
        """Initialize the attractor sculptor"""
        super().__init__(output_dir, alpha, beta, gamma, pattern_type)
        
        self.max_steps = max_steps
        self.n_fingerprint_components = n_fingerprint_components
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, "fingerprints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "meta_perturbations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "embedding"), exist_ok=True)
        
        # Initialize fingerprinter
        self.fingerprinter = AttractorFingerprinter(n_components=n_fingerprint_components)
        
        # Results storage
        self.fingerprints = []
        self.meta_perturbation_results = []
        self.residuals = []
        self.attractor_labels = []
    
    def apply_meta_perturbation(self, meta_type, strength=0.1, **kwargs):
        """Apply a meta-perturbation to the field before standard perturbation"""
        if self.experiment is None:
            self.initialize_experiment()
            
        field = self.experiment.state
        
        if meta_type == "field_bias":
            return self._apply_field_bias(field, strength, **kwargs)
        elif meta_type == "boundary_constraint":
            return self._apply_boundary_constraint(field, strength, **kwargs)
        elif meta_type == "noise_profile":
            return self._apply_noise_profile(field, strength, **kwargs)
        else:
            raise ValueError(f"Unknown meta-perturbation type: {meta_type}")
    
    def _apply_field_bias(self, field, strength, bias_type="gradient", **kwargs):
        """Apply a field bias meta-perturbation"""
        size = field.shape[0]
        
        if bias_type == "gradient":
            # Create a gradient bias field
            x = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, x)
            bias = X  # Horizontal gradient
            
        elif bias_type == "radial":
            # Create a radial bias field
            x = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, x)
            bias = np.sqrt(X**2 + Y**2)
            bias = 1 - bias  # Invert so center is higher
            
        elif bias_type == "checkerboard":
            # Create a checkerboard bias field
            x = np.linspace(0, 7, size)
            X, Y = np.meshgrid(x, x)
            bias = np.sin(X) * np.sin(Y)
            
        else:
            raise ValueError(f"Unknown bias type: {bias_type}")
            
        # Apply bias with strength
        biased_field = field + strength * bias
        
        # Normalize to [-1, 1]
        biased_field = np.clip(biased_field, -1, 1)
        
        # Update the experiment state
        self.experiment.state = biased_field
        
        return biased_field
    
    def _apply_boundary_constraint(self, field, strength, constraint_type="edge", **kwargs):
        """Apply a boundary constraint meta-perturbation"""
        size = field.shape[0]
        
        if constraint_type == "edge":
            # Force edges toward specific values
            edge_val = kwargs.get("edge_val", 0.0)
            
            # Create mask (1 at edges, 0 in center)
            mask = np.ones_like(field)
            margin = int(size * 0.1)  # 10% margin
            mask[margin:-margin, margin:-margin] = 0
            
            # Blend field with edge value based on mask and strength
            constrained_field = field * (1 - strength * mask) + edge_val * (strength * mask)
            
        elif constraint_type == "region":
            # Constrain a specific region
            center = kwargs.get("center", (size//2, size//2))
            radius = kwargs.get("radius", size//4)
            region_val = kwargs.get("region_val", 0.5)
            
            # Create circular mask
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y)
            mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2).astype(float)
            
            # Apply constraint
            constrained_field = field * (1 - strength * mask) + region_val * (strength * mask)
            
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
            
        # Normalize to [-1, 1]
        constrained_field = np.clip(constrained_field, -1, 1)
        
        # Update the experiment state
        self.experiment.state = constrained_field
        
        return constrained_field
    
    def _apply_noise_profile(self, field, strength, noise_type="gaussian", **kwargs):
        """Apply a noise profile meta-perturbation"""
        size = field.shape[0]
        
        if noise_type == "gaussian":
            # Apply Gaussian noise
            noise = np.random.normal(0, 1, field.shape)
            
        elif noise_type == "uniform":
            # Apply uniform noise
            noise = np.random.uniform(-1, 1, field.shape)
            
        elif noise_type == "perlin":
            # Generate Perlin-like noise (approximation)
            from scipy.ndimage import gaussian_filter
            base_noise = np.random.uniform(-1, 1, field.shape)
            octaves = 4
            persistence = 0.5
            
            noise = np.zeros_like(field)
            for i in range(octaves):
                freq = 2**i
                amp = persistence**i
                
                # Sample noise at appropriate scale
                octave_noise = gaussian_filter(
                    np.random.uniform(-1, 1, field.shape), 
                    sigma=size/(freq*4)
                )
                
                noise += amp * octave_noise
                
            # Normalize noise
            noise = 2 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-10) - 1
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        # Apply noise with strength
        noisy_field = field + strength * noise
        
        # Normalize to [-1, 1]
        noisy_field = np.clip(noisy_field, -1, 1)
        
        # Update the experiment state
        self.experiment.state = noisy_field
        
        return noisy_field
    
    def run_meta_perturbation_experiment(self, meta_type, variations, strengths, 
                                        trials_per_config=3, parallel=False):
        """Run a complete meta-perturbation experiment"""
        if parallel:
            return self._run_parallel_experiment(meta_type, variations, strengths, trials_per_config)
            
        all_results = []
        all_residuals = []
        all_fingerprints = []
        all_labels = []
        
        # Create experiment configurations
        configs = []
        for variation in variations:
            for strength in strengths:
                for trial in range(trials_per_config):
                    configs.append({
                        'meta_type': meta_type,
                        'variation': variation,
                        'strength': strength,
                        'trial': trial
                    })
        
        # Run all configurations
        for config in tqdm(configs, desc=f"Running {meta_type} experiments"):
            result = self._run_single_experiment(**config)
            
            all_results.append(result)
            all_residuals.append(result['residual'])
            all_fingerprints.append(result['fingerprint'])
            all_labels.append(f"{meta_type}_{config['variation']}_{config['strength']}_{config['trial']}")
        
        # Store results
        self.meta_perturbation_results = all_results
        self.residuals = all_residuals
        self.fingerprints = all_fingerprints
        self.attractor_labels = all_labels
        
        # Fit fingerprinter with collected residuals
        if len(all_residuals) > self.n_fingerprint_components:
            self.fingerprinter.fit(np.array(all_residuals))
        
        # Analyze results
        self._analyze_results(meta_type)
        
        return all_results
    
    def _run_parallel_experiment(self, meta_type, variations, strengths, trials_per_config):
        """Run experiment configurations in parallel"""
        # Create experiment configurations
        configs = []
        for variation in variations:
            for strength in strengths:
                for trial in range(trials_per_config):
                    configs.append({
                        'meta_type': meta_type,
                        'variation': variation,
                        'strength': strength,
                        'trial': trial
                    })
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for config in configs:
                futures.append(
                    executor.submit(self._run_single_experiment, **config)
                )
            
            # Collect results
            all_results = []
            all_residuals = []
            all_fingerprints = []
            all_labels = []
            
            for future in tqdm(futures, desc=f"Collecting {meta_type} results"):
                result = future.result()
                
                all_results.append(result)
                all_residuals.append(result['residual'])
                all_fingerprints.append(result['fingerprint'])
                all_labels.append(f"{meta_type}_{result['variation']}_{result['strength']}_{result['trial']}")
        
        # Store results
        self.meta_perturbation_results = all_results
        self.residuals = all_residuals
        self.fingerprints = all_fingerprints
        self.attractor_labels = all_labels
        
        # Fit fingerprinter with collected residuals
        if len(all_residuals) > self.n_fingerprint_components:
            self.fingerprinter.fit(np.array(all_residuals))
        
        # Analyze results
        self._analyze_results(meta_type)
        
        return all_results
    
    def _run_single_experiment(self, meta_type, variation, strength, trial, **kwargs):
        """Run a single meta-perturbation experiment"""
        # Initialize the experiment
        self.initialize_experiment()
        
        # Save initial state before any perturbation
        initial_state = self.experiment.state.copy()
        
        # Apply meta-perturbation
        if meta_type == "field_bias":
            self._apply_field_bias(self.experiment.state, strength, bias_type=variation)
        elif meta_type == "boundary_constraint":
            self._apply_boundary_constraint(self.experiment.state, strength, constraint_type=variation)
        elif meta_type == "noise_profile":
            self._apply_noise_profile(self.experiment.state, strength, noise_type=variation)
        
        # Save state after meta-perturbation
        meta_perturbed_state = self.experiment.state.copy()
        
        # Apply standard perturbation
        self.experiment.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        
        # Let system evolve
        self.experiment.update(steps=self.max_steps)
        
        # Get final state
        final_state = self.experiment.state.copy()
        
        # Extract metrics
        metrics = self.compute_metrics()
        
        # Calculate residual
        residual = final_state - initial_state
        
        # Compute attractor fingerprint
        fingerprint = self.fingerprinter.compute_fingerprint(initial_state, final_state)
        
        # Save visualizations
        vis_dir = os.path.join(self.output_dir, "meta_perturbations", 
                             f"{meta_type}_{variation}")
        os.makedirs(vis_dir, exist_ok=True)
        
        self._save_meta_perturbation_visualization(
            initial_state,
            meta_perturbed_state,
            final_state,
            f"{meta_type}_{variation}_{strength}_{trial}",
            vis_dir
        )
        
        # Save fingerprint visualization
        fp_dir = os.path.join(self.output_dir, "fingerprints")
        self._save_fingerprint_visualization(
            fingerprint,
            f"{meta_type}_{variation}_{strength}_{trial}",
            fp_dir
        )
        
        # Compile result
        result = {
            'meta_type': meta_type,
            'variation': variation,
            'strength': strength,
            'trial': trial,
            'final_correlation': metrics['correlation'],
            'final_coherence': metrics['coherence'],
            'final_ccdi': metrics['ccdi'],
            'residual': residual.flatten(),
            'fingerprint': fingerprint
        }
        
        return result
    
    def _save_meta_perturbation_visualization(self, initial_state, meta_state, final_state, 
                                            label, output_dir):
        """Save visualization of a meta-perturbation experiment"""
        plt.figure(figsize=(15, 5))
        
        # Plot initial state
        plt.subplot(1, 3, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Initial State")
        plt.axis('off')
        
        # Plot meta-perturbed state
        plt.subplot(1, 3, 2)
        plt.imshow(meta_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("After Meta-Perturbation")
        plt.axis('off')
        
        # Plot final state
        plt.subplot(1, 3, 3)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Final State")
        plt.axis('off')
        
        plt.suptitle(label)
        plt.tight_layout()
        
        # Save the figure
        filename = f"{label}_states.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
        # Save residual visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(final_state - initial_state, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(label='Residual Value')
        plt.title(f"Residual Field (Final - Initial): {label}")
        plt.axis('off')
        
        filename = f"{label}_residual.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def _save_fingerprint_visualization(self, fingerprint, label, output_dir):
        """Save visualization of an attractor fingerprint"""
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        self.fingerprinter.visualize_fingerprint(fingerprint, ax=ax)
        plt.title(f"Attractor Fingerprint: {label}")
        
        filename = f"{label}_fingerprint.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    def _analyze_results(self, meta_type):
        """Analyze meta-perturbation results and create visualizations"""
        if not self.meta_perturbation_results:
            return
            
        # Extract key metrics
        df = pd.DataFrame(self.meta_perturbation_results)
        
        # Create heatmap of correlation by variation and strength
        plt.figure(figsize=(10, 8))
        
        # Pivot data for heatmap
        if 'variation' in df.columns and 'strength' in df.columns:
            heatmap_data = df.pivot_table(
                index='variation',
                columns='strength',
                values='final_correlation',
                aggfunc='mean'
            )
            
            # Create heatmap
            sns_available = False
            try:
                import seaborn as sns
                sns_available = True
            except ImportError:
                pass
                
            if sns_available:
                ax = sns.heatmap(
                    heatmap_data, 
                    annot=True, 
                    cmap='viridis', 
                    fmt='.3f',
                    vmin=0, 
                    vmax=1
                )
                plt.title(f'Mean Correlation by {meta_type} Variation and Strength')
            else:
                plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
                plt.colorbar(label='Mean Correlation')
                plt.title(f'Mean Correlation by {meta_type} Variation and Strength')
                plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
                plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{meta_type}_correlation_heatmap.png"))
            plt.close()
        
        # Create embedding visualization based on fingerprints
        if len(self.fingerprints) > 2:
            self._create_fingerprint_embedding(meta_type)
        
        # Create stability surface visualization
        if 'variation' in df.columns and 'strength' in df.columns:
            variations = df['variation'].unique()
            strengths = sorted(df['strength'].unique())
            
            for variation in variations:
                plt.figure(figsize=(10, 6))
                
                # Extract data for this variation
                var_data = df[df['variation'] == variation]
                
                # Group by strength
                grouped = var_data.groupby('strength')
                
                # Calculate mean and std for correlation
                means = grouped['final_correlation'].mean()
                stds = grouped['final_correlation'].std()
                
                # Plot mean with error bands
                plt.plot(strengths, [means.get(s, 0) for s in strengths], 'o-', 
                        label=f'Mean Correlation', color='blue')
                
                # Add error bands
                lower = [max(0, means.get(s, 0) - stds.get(s, 0)) for s in strengths]
                upper = [min(1, means.get(s, 0) + stds.get(s, 0)) for s in strengths]
                
                plt.fill_between(strengths, lower, upper, color='blue', alpha=0.2,
                               label='Standard Deviation')
                
                plt.axhline(y=0.9, color='green', linestyle='--', 
                          label='Strong Recovery (0.9)')
                plt.axhline(y=0.4, color='red', linestyle='--', 
                          label='Recovery Threshold (0.4)')
                
                plt.title(f'Recovery Stability: {meta_type}_{variation}')
                plt.xlabel('Perturbation Strength')
                plt.ylabel('Final Correlation')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"{meta_type}_{variation}_stability.png"))
                plt.close()
    
    def _create_fingerprint_embedding(self, meta_type):
        """Create embedding visualization of attractor fingerprints"""
        if not self.fingerprints or len(self.fingerprints) < 3:
            return
            
        # Convert to numpy array
        fingerprints_array = np.array(self.fingerprints)
        
        # Apply PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_pca = pca.fit_transform(fingerprints_array)
        
        # Try t-SNE if available
        tsne_available = False
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(self.fingerprints)//5))
            reduced_tsne = tsne.fit_transform(fingerprints_array)
            tsne_available = True
        except (ImportError, ValueError):
            tsne_available = False
        
        # Extract meta parameters from labels
        variations = []
        strengths = []
        
        for label in self.attractor_labels:
            parts = label.split('_')
            if len(parts) >= 4:
                variations.append(parts[1])
                try:
                    strengths.append(float(parts[2]))
                except ValueError:
                    strengths.append(0.0)
            else:
                variations.append("unknown")
                strengths.append(0.0)
        
        # Create PCA plot
        plt.figure(figsize=(12, 10))
        
        # Create colormap for variations
        unique_variations = list(set(variations))
        variation_colors = {v: plt.cm.tab10(i/len(unique_variations)) 
                          for i, v in enumerate(unique_variations)}
        
        # Plot PCA embedding colored by variation
        plt.subplot(2, 1, 1)
        for i, (x, y) in enumerate(reduced_pca):
            variation = variations[i]
            strength = strengths[i]
            color = variation_colors[variation]
            
            # Scale point size by strength
            size = 50 + 200 * strength
            
            plt.scatter(x, y, color=color, s=size, alpha=0.7,
                       label=variation if variation not in [variations[:i]] else "")
            
            plt.annotate(self.attractor_labels[i], (x, y), fontsize=8)
        
        plt.title(f'PCA Embedding of Attractor Fingerprints ({meta_type})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        
        # Plot t-SNE if available
        if tsne_available:
            plt.subplot(2, 1, 2)
            
            for i, (x, y) in enumerate(reduced_tsne):
                variation = variations[i]
                strength = strengths[i]
                color = variation_colors[variation]
                
                # Scale point size by strength
                size = 50 + 200 * strength
                
                plt.scatter(x, y, color=color, s=size, alpha=0.7)
                plt.annotate(self.attractor_labels[i], (x, y), fontsize=8)
            
            plt.title(f't-SNE Embedding of Attractor Fingerprints ({meta_type})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "embedding", f"{meta_type}_fingerprint_embedding.png"))
        plt.close()
        
        # Create interactive plot if plotly is available
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create DataFrame for plotly
            df = pd.DataFrame({
                'x': reduced_pca[:, 0],
                'y': reduced_pca[:, 1],
                'variation': variations,
                'strength': strengths,
                'label': self.attractor_labels
            })
            
            # Create figure
            fig = px.scatter(
                df, x='x', y='y', 
                color='variation', size='strength',
                hover_data=['label', 'strength'],
                title=f'PCA Embedding of Attractor Fingerprints ({meta_type})'
            )
            
            # Add labels
            fig.update_traces(textposition='top center')
            fig.update_layout(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2'
            )
            
            # Save as HTML
            html_path = os.path.join(self.output_dir, "embedding", f"{meta_type}_interactive_embedding.html")
            fig.write_html(html_path)
            
        except ImportError:
            pass  # Plotly not available, skip interactive plot
    
    def run(self, meta_types=None, **kwargs):
        """Run experiments with multiple meta-perturbation types"""
        if meta_types is None:
            meta_types = ['field_bias', 'boundary_constraint', 'noise_profile']
            
        variations = {
            'field_bias': ['gradient', 'radial', 'checkerboard'],
            'boundary_constraint': ['edge', 'region'],
            'noise_profile': ['gaussian', 'uniform', 'perlin']
        }
        
        strengths = kwargs.get('strengths', [0.05, 0.1, 0.2, 0.3])
        trials = kwargs.get('trials_per_config', 3)
        parallel = kwargs.get('parallel', False)
        
        all_results = {}
        
        for meta_type in meta_types:
            print(f"Running {meta_type} experiments...")
            
            type_variations = variations.get(meta_type, [])
            if not type_variations:
                print(f"No variations defined for {meta_type}, skipping...")
                continue
                
            results = self.run_meta_perturbation_experiment(
                meta_type, 
                type_variations, 
                strengths, 
                trials_per_config=trials,
                parallel=parallel
            )
            
            all_results[meta_type] = results
            
        # Perform cross-analysis between meta-perturbation types
        if len(all_results) > 1:
            self._cross_analyze_meta_types(all_results)
            
        return all_results
    
    def _cross_analyze_meta_types(self, results_by_type):
        """Compare results across different meta-perturbation types"""
        # Create summary table
        summary = []
        
        for meta_type, results in results_by_type.items():
            # Calculate average metrics by strength
            df = pd.DataFrame(results)
            strength_summary = df.groupby('strength').agg({
                'final_correlation': ['mean', 'std'],
                'final_ccdi': ['mean', 'std']
            }).reset_index()
            
            # Flatten multi-index columns
            strength_summary.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] 
                for col in strength_summary.columns
            ]
            
            # Add meta_type
            strength_summary['meta_type'] = meta_type
            
            summary.append(strength_summary)
        
        if summary:
            # Combine summaries
            summary_df = pd.concat(summary, ignore_index=True)
            
            # Save summary table
            summary_df.to_csv(os.path.join(self.output_dir, "meta_type_comparison.csv"), index=False)
            
            # Create comparison plots
            plt.figure(figsize=(12, 8))
            
            # Get unique meta-types and strengths
            meta_types = summary_df['meta_type'].unique()
            strengths = sorted(summary_df['strength'].unique())
            
            # Plot correlation by strength for each meta-type
            for meta_type in meta_types:
                type_data = summary_df[summary_df['meta_type'] == meta_type]
                
                plt.errorbar(
                    type_data['strength'], 
                    type_data['final_correlation_mean'],
                    yerr=type_data['final_correlation_std'],
                    fmt='o-', 
                    label=meta_type,
                    capsize=5
                )
            
            plt.axhline(y=0.9, color='green', linestyle='--', 
                      label='Strong Recovery (0.9)')
            plt.axhline(y=0.4, color='red', linestyle='--', 
                      label='Recovery Threshold (0.4)')
            
            plt.xlabel('Perturbation Strength')
            plt.ylabel('Mean Final Correlation')
            plt.title('Recovery Performance by Meta-Perturbation Type')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "meta_type_comparison.png"))
            plt.close()
    
    def visualize_results(self, **kwargs):
        """Create comprehensive visualization of results"""
        # Additional visualizations can be added here based on specific needs
        pass