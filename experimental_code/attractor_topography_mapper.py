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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import logging

# Assuming these are imported from your existing codebase
from rcft_framework import RCFTExperiment
from false_attractor_analyzer import FalseAttractorAnalyzer

class AttractorTopographyMapper:
    """Maps the attractor space of RCFT systems across memory parameters (alpha, gamma)"""
    
    def __init__(self, output_dir="phase2_results", log_dir="phase2_logs"):
        """Initialize the topography mapper with default settings"""
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        # Default parameters for experiments
        self.delay = 10          # Default delay between perturbations
        self.n_trials = 1        # Default trials per parameter point
        self.ccdi_threshold = 0.08  # Anomaly threshold
        self.clustering_method = 'kmeans'  # Default clustering method
        self.n_clusters = 3      # Default number of clusters for K-means
        self.base_seed = 42      # Base random seed
        
        # RCFT default parameters (will be varied in sweep)
        self.alpha_range = None  # Will be set in run_parameter_sweep
        self.gamma_range = None  # Will be set in run_parameter_sweep
        self.beta = 0.5          # Default coupling strength
        
        # Setup directories and logging
        self._setup_directories()
        self._setup_logging()
        
        # Storage for results
        self.results_df = None
        self.field_residuals = []
        self.parameter_labels = []
        
    def _setup_directories(self):
        """Create necessary directories"""
        # Create main output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create subdirectories for different data types
        os.makedirs(os.path.join(self.output_dir, "individual_runs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "field_data"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        
        # Create log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _setup_logging(self):
        """Configure logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"phase2_sweep_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AttractorTopographyMapper")
        self.logger.info(f"Initialized AttractorTopographyMapper. Output dir: {self.output_dir}")
    
    def run_parameter_sweep(self, alpha_range=(0.1, 0.6, 11), gamma_range=(0.8, 0.99, 11), 
                           delay=10, n_trials=1, pattern_type="fractal", parallel=False):
        """
        Run a full parameter sweep across the alpha-gamma grid
        
        Parameters:
        -----------
        alpha_range : tuple (start, stop, steps)
            Range for memory strength parameter
        gamma_range : tuple (start, stop, steps)
            Range for memory decay parameter
        delay : int
            Time steps between first and second perturbation
        n_trials : int
            Number of trials per parameter combination
        pattern_type : str
            Pattern type to initialize
        parallel : bool
            Whether to use multiprocessing
        """
        self.logger.info(f"Starting parameter sweep: alpha={alpha_range}, gamma={gamma_range}, delay={delay}, trials={n_trials}")
        
        # Generate parameter ranges
        self.alpha_range = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
        self.gamma_range = np.linspace(gamma_range[0], gamma_range[1], gamma_range[2])
        self.delay = delay
        self.n_trials = n_trials
        
        # Create a timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"sweep_{timestamp}"
        
        # Create parameter combinations
        parameter_grid = [(alpha, gamma) for alpha in self.alpha_range for gamma in self.gamma_range]
        total_runs = len(parameter_grid) * n_trials
        
        self.logger.info(f"Parameter sweep will perform {total_runs} total simulations")
        
        # Initialize results storage
        results_list = []
        
        # Run the parameter sweep
        if parallel:
            self.logger.info("Using parallel processing")
            # Configure process pool
            num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
            self.logger.info(f"Using {num_cores} cores")
            
            # Create a pool of workers
            with mp.Pool(processes=num_cores) as pool:
                # Process parameters in parallel
                results = []
                for i, (alpha, gamma) in enumerate(parameter_grid):
                    for trial in range(n_trials):
                        seed = self.base_seed + i * n_trials + trial
                        # Submit job to pool
                        result = pool.apply_async(
                            self._run_single_parameter_point, 
                            (alpha, gamma, delay, pattern_type, trial, seed)
                        )
                        results.append(result)
                
                # Collect results as they complete
                for i, result in enumerate(tqdm(results, desc="Parameter Points")):
                    try:
                        result_data = result.get()
                        results_list.append(result_data)
                        
                        # Extract residual data for attractor analysis
                        if 'residual' in result_data and 'parameter_label' in result_data:
                            self.field_residuals.append(result_data['residual'])
                            self.parameter_labels.append(result_data['parameter_label'])
                            
                        # Save ongoing results to protect against crashes
                        if (i + 1) % 10 == 0:
                            self._save_interim_results(results_list)
                    except Exception as e:
                        self.logger.error(f"Error in parallel execution: {e}")
                        
        else:
            self.logger.info("Using sequential processing")
            # Process parameters sequentially with progress bar
            for i, (alpha, gamma) in enumerate(tqdm(parameter_grid, desc="Parameter Points")):
                for trial in range(n_trials):
                    seed = self.base_seed + i * n_trials + trial
                    try:
                        # Add to results and collect field data for attractor analysis
                        result_data = self._run_single_parameter_point(alpha, gamma, delay, pattern_type, trial, seed)
                        results_list.append(result_data)
                        
                        # Extract residual data
                        if 'residual' in result_data and 'parameter_label' in result_data:
                            self.field_residuals.append(result_data['residual'])
                            self.parameter_labels.append(result_data['parameter_label'])
                    except Exception as e:
                        self.logger.error(f"Error at alpha={alpha}, gamma={gamma}, trial={trial}: {e}")
                    
                    # Save ongoing results to protect against crashes
                    if (i * n_trials + trial + 1) % 5 == 0:
                        self._save_interim_results(results_list)
        
        # Convert results to DataFrame and clean up residual data
        self.results_df = pd.DataFrame(results_list)
        
        # Remove residual arrays from the DataFrame (they're already stored in self.field_residuals)
        if 'residual' in self.results_df.columns:
            self.results_df = self.results_df.drop('residual', axis=1)
        if 'parameter_label' in self.results_df.columns:
            self.results_df = self.results_df.drop('parameter_label', axis=1)
        
        # Save final results
        self._save_final_results()
        
        # Generate visualizations
        self._generate_heatmaps()
        self._analyze_attractors()
        
        self.logger.info("Parameter sweep completed successfully")
        return self.results_df
    
    def _run_single_parameter_point(self, alpha, gamma, delay, pattern_type, trial_id, seed=None):
        """
        Run a single experiment at a specific parameter point
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        delay : int
            Time steps between perturbations
        pattern_type : str
            Pattern type to initialize
        trial_id : int
            Trial identifier
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Results dictionary for this parameter point
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Create output directory for this run
        run_dir = os.path.join(self.output_dir, "individual_runs", f"alpha{alpha:.4f}_gamma{gamma:.4f}_trial{trial_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Initialize experiment
        exp = RCFTExperiment(
            memory_strength=alpha,
            coupling_strength=self.beta,
            memory_decay=gamma
        )
        
        # Initialize pattern
        exp.initialize_pattern(pattern_type=pattern_type)
        
        # Save initial state
        initial_state = exp.state.copy()
        
        # Apply first perturbation
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        first_perturbation = exp.perturbation_step
        
        # Let system recover for specified delay
        exp.update(steps=delay)
        
        # Record mid-state
        mid_state = exp.state.copy()
        mid_correlation = exp.metrics['correlation'][-1]
        mid_coherence = exp.metrics['coherence'][-1]
        
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
        ccdi = self._compute_ccdi(final_correlation, final_coherence)
        
        # Classify recovery trajectory
        recovery_class = self._classify_recovery(
            exp.metrics['correlation'], 
            [first_perturbation, second_perturbation]
        )
        
        # Extract attractor residual (Δ field)
        residual = self._extract_attractor_residual(final_state, initial_state)
        
        # Save field data
        field_path = os.path.join(self.output_dir, "field_data", f"alpha{alpha:.4f}_gamma{gamma:.4f}_trial{trial_id}")
        os.makedirs(os.path.dirname(field_path), exist_ok=True)
        np.save(f"{field_path}_initial.npy", initial_state)
        np.save(f"{field_path}_final.npy", final_state)
        np.save(f"{field_path}_delta.npy", residual)
        
        # Save visualizations
        self._visualize_experiment(exp, initial_state, mid_state, final_state, 
                                [first_perturbation, second_perturbation], run_dir)
        self._visualize_residual(residual, os.path.join(run_dir, "delta_field.png"))
        
        # Save metrics to file
        with open(os.path.join(run_dir, "metrics.json"), 'w') as f:
            metrics = {
                'alpha': alpha,
                'gamma': gamma,
                'delay': delay,
                'trial_id': trial_id,
                'mid_correlation': float(mid_correlation),
                'mid_coherence': float(mid_coherence),
                'final_correlation': float(final_correlation),
                'final_coherence': float(final_coherence),
                'final_mutual_info': float(final_mutual_info),
                'final_entropy': float(final_entropy),
                'ccdi': float(ccdi),
                'recovery_class': recovery_class,
                'is_anomalous': "true" if ccdi > self.ccdi_threshold else "false",  # Use string instead of bool
                'residual_norm': float(np.linalg.norm(residual)),
                'runtime': float(time.time() - start_time)
            }
            json.dump(metrics, f, indent=2)
        
        # For parallel processing, we don't update self.field_residuals directly
        # Instead, we'll return the residual with the result
        elapsed_time = time.time() - start_time
        
        # Return results as dictionary
        result = {
            'alpha': float(alpha),
            'gamma': float(gamma),
            'trial_id': int(trial_id),
            'delay': int(delay),
            'mid_correlation': float(mid_correlation),
            'mid_coherence': float(mid_coherence),
            'final_correlation': float(final_correlation),
            'final_coherence': float(final_coherence),
            'final_mutual_info': float(final_mutual_info),
            'final_entropy': float(final_entropy),
            'ccdi': float(ccdi),
            'recovery_class': str(recovery_class),
            'is_anomalous': bool(ccdi > self.ccdi_threshold),  # Keep as bool for DataFrame
            'residual_norm': float(np.linalg.norm(residual)),
            'runtime': float(elapsed_time),
            # Add residual data for parallel processing
            'residual': residual.flatten(),
            'parameter_label': f"a{alpha:.3f}_g{gamma:.3f}"
        }
        
        return result
    
    def _compute_ccdi(self, correlation, coherence):
        """Compute Coherence-Correlation Divergence Index (CCDI)"""
        return coherence - correlation
    
    def _classify_recovery(self, correlation_curve, perturbation_steps):
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
    
    def _extract_attractor_residual(self, final_state, initial_state):
        """Compute the difference between final and initial states (Delta field)"""
        return final_state - initial_state
    
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
        plt.title(f"Mid Recovery (t={self.delay})")
        plt.axis('off')
        
        # Final state
        plt.subplot(1, 4, 4)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Final State (t={len(exp.history)-1})")
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
    
    def _save_interim_results(self, results_list):
        """Save interim results during processing"""
        interim_df = pd.DataFrame(results_list)
        
        # Save to CSV
        interim_csv = os.path.join(self.output_dir, f"interim_results_{self.run_id}.csv")
        interim_df.to_csv(interim_csv, index=False)
        
        # Save residuals to pickle
        interim_residuals = os.path.join(self.output_dir, f"interim_residuals_{self.run_id}.pkl")
        with open(interim_residuals, 'wb') as f:
            pickle.dump({
                'residuals': self.field_residuals,
                'labels': self.parameter_labels
            }, f)
            
        self.logger.info(f"Saved interim results: {len(results_list)} runs")
    
    def _save_final_results(self):
        """Save final results after processing"""
        # Save full results to CSV
        results_csv = os.path.join(self.output_dir, f"attractor_topography_results_{self.run_id}.csv")
        self.results_df.to_csv(results_csv, index=False)
        
        # Save residuals to pickle for later analysis
        residuals_pkl = os.path.join(self.output_dir, f"attractor_residuals_{self.run_id}.pkl")
        with open(residuals_pkl, 'wb') as f:
            pickle.dump({
                'residuals': self.field_residuals,
                'labels': self.parameter_labels,
                'alpha_range': self.alpha_range,
                'gamma_range': self.gamma_range
            }, f)
            
        self.logger.info(f"Saved final results: {len(self.results_df)} rows")
        
    def _generate_heatmaps(self):
        """Generate heatmap visualizations"""
        # Define metrics to visualize
        metrics = ['final_correlation', 'ccdi', 'final_coherence', 'residual_norm', 'final_entropy']
        
        # Make sure we have data
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No data available for generating heatmaps")
            return
            
        # Create heatmaps for each metric
        for metric in metrics:
            try:
                # Ensure publication-ready plot
                plt.figure(figsize=(10, 8))
                plt.rcParams.update({'font.size': 12})
                
                # Pivot data for heatmap
                pivot_data = self.results_df.pivot_table(
                    index='gamma', 
                    columns='alpha', 
                    values=metric,
                    aggfunc='mean'  # In case of multiple trials
                )
                
                # Create heatmap with viridis colormap (colorblind-friendly)
                ax = sns.heatmap(
                    pivot_data, 
                    cmap='viridis',
                    annot=True, 
                    fmt=".3f",
                    linewidths=.5, 
                    cbar_kws={'label': metric.replace('_', ' ').title()}
                )
                
                # Set labels
                plt.title(f"{metric.replace('_', ' ').title()} Across (Alpha, Gamma) Parameter Space")
                plt.xlabel("Memory Strength (Alpha)")
                plt.ylabel("Memory Decay (Gamma)")
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, "visualizations", f"heatmap_{metric}_{self.run_id}.png"),
                    dpi=300
                )
                plt.close()
                
            except Exception as e:
                self.logger.error(f"Error generating heatmap for {metric}: {e}")
                
        # Create a specific heatmap for anomaly classification
        try:
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({'font.size': 12})
            
            # Pivot data
            pivot_data = self.results_df.pivot_table(
                index='gamma', 
                columns='alpha', 
                values='is_anomalous',
                aggfunc='mean'  # Shows proportion of trials that are anomalous
            )
            
            # Create heatmap with RdYlGn_r colormap (red for anomalies)
            ax = sns.heatmap(
                pivot_data, 
                cmap='RdYlGn_r',
                vmin=0, 
                vmax=1,
                annot=True, 
                fmt=".2f",
                linewidths=.5,
                cbar_kws={'label': 'Proportion of Anomalous Trials'}
            )
            
            # Draw contour at 0.5 threshold
            if np.any((pivot_data.values >= 0.4) & (pivot_data.values <= 0.6)):
                # Add contour lines for phase transition
                x = np.arange(0, pivot_data.shape[1] + 1)
                y = np.arange(0, pivot_data.shape[0] + 1)
                plt.contour(x, y, pivot_data.values, levels=[0.5], colors='black', linewidths=2)
            
            # Set labels
            plt.title("Phase Transition Boundary: Normal vs. False Memory")
            plt.xlabel("Memory Strength (Alpha)")
            plt.ylabel("Memory Decay (Gamma)")
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "visualizations", f"anomaly_phase_diagram_{self.run_id}.png"),
                dpi=300
            )
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly phase diagram: {e}")
    
    def _load_field_residuals_from_files(self):
        """Load field residuals from saved npy files as a fallback"""
        self.logger.info("Attempting to load field residuals from saved files")
        
        field_data_dir = os.path.join(self.output_dir, "field_data")
        if not os.path.exists(field_data_dir):
            self.logger.warning(f"Field data directory not found: {field_data_dir}")
            return
            
        # List all delta field files
        delta_files = [f for f in os.listdir(field_data_dir) if f.endswith('_delta.npy')]
        
        if not delta_files:
            self.logger.warning("No delta field files found")
            return
            
        self.logger.info(f"Found {len(delta_files)} delta field files")
        
        # Clear existing data
        self.field_residuals = []
        self.parameter_labels = []
        
        # Load each file
        for file in delta_files:
            try:
                file_path = os.path.join(field_data_dir, file)
                residual = np.load(file_path)
                
                # Extract parameters from filename
                # Format: alpha0.1000_gamma0.8000_trial0_delta.npy
                parts = file.split('_')
                alpha_part = parts[0].replace('alpha', '')
                gamma_part = parts[1].replace('gamma', '')
                
                # Add to collections
                self.field_residuals.append(residual.flatten())
                self.parameter_labels.append(f"a{float(alpha_part):.3f}_g{float(gamma_part):.3f}")
                
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                
        self.logger.info(f"Successfully loaded {len(self.field_residuals)} field residuals")
        
        # If we have results_df, make sure the parameter labels match
        if self.results_df is not None and len(self.results_df) > 0:
            # Match parameter labels with DataFrame rows
            for i, label in enumerate(self.parameter_labels):
                parts = label.split('_')
                alpha_val = float(parts[0].replace('a', ''))
                gamma_val = float(parts[1].replace('g', ''))
                
                # Find matching row in DataFrame
                matching_rows = self.results_df[(self.results_df['alpha'] == alpha_val) & 
                                               (self.results_df['gamma'] == gamma_val)]
                
                if len(matching_rows) > 0:
                    # Update label to include more info if needed
                    row_idx = matching_rows.index[0]
                    self.parameter_labels[i] = f"{label}_c{float(matching_rows.iloc[0]['ccdi']):.2f}"
            
    def _analyze_attractors(self):
        """Analyze attractors using dimensionality reduction and clustering"""
        # Make sure we have data
        if len(self.field_residuals) == 0:
            self.logger.warning("No field residuals available for attractor analysis")
            
            # Try to load residuals from saved field data
            try:
                self._load_field_residuals_from_files()
            except Exception as e:
                self.logger.error(f"Failed to load field residuals from files: {e}")
                return
                
        if len(self.field_residuals) == 0:
            self.logger.warning("Still no field residuals available after attempting to load from files")
            return
                
        try:
            # Convert to numpy array
            residuals_array = np.array(self.field_residuals)
            
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_pca = pca.fit_transform(residuals_array)
            
            # Apply t-SNE if we have enough samples
            if len(residuals_array) > 5:
                perplexity = min(30, len(residuals_array) // 4)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                reduced_tsne = tsne.fit_transform(residuals_array)
            else:
                reduced_tsne = None
                
            # Perform clustering
            if self.clustering_method == 'kmeans':
                # Apply K-means clustering
                if len(residuals_array) >= self.n_clusters:
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(residuals_array)
                    
                    # Try to score the clustering
                    if len(residuals_array) > self.n_clusters:
                        silhouette_avg = silhouette_score(residuals_array, cluster_labels)
                        self.logger.info(f"K-means clustering silhouette score: {silhouette_avg:.4f}")
                else:
                    cluster_labels = np.zeros(len(residuals_array), dtype=int)
                    self.logger.warning(f"Too few samples for K-means with {self.n_clusters} clusters")
            elif self.clustering_method == 'dbscan':
                # Apply DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                cluster_labels = dbscan.fit_predict(residuals_array)
            else:
                cluster_labels = np.zeros(len(residuals_array), dtype=int)
                
            # Add cluster labels to results DataFrame
            if self.results_df is not None:
                # Create a mapping of alpha/gamma to cluster_id
                cluster_mapping = {}
                for i, label in enumerate(self.parameter_labels):
                    parts = label.split('_')
                    if len(parts) >= 2:  # Make sure the label has alpha and gamma parts
                        alpha_val = float(parts[0].replace('a', ''))
                        gamma_val = float(parts[1].replace('g', ''))
                        cluster_mapping[(alpha_val, gamma_val)] = cluster_labels[i]
                
                # Apply mapping to DataFrame
                self.results_df['cluster_id'] = self.results_df.apply(
                    lambda row: cluster_mapping.get((row['alpha'], row['gamma']), -1), 
                    axis=1
                )
                
                # Update CSV with cluster labels
                results_csv = os.path.join(self.output_dir, f"attractor_topography_results_{self.run_id}.csv")
                self.results_df.to_csv(results_csv, index=False)
            
            # Create alpha-gamma parameter arrays for coloring
            alpha_params = np.array([float(label.split('_')[0].replace('a', '')) for label in self.parameter_labels])
            gamma_params = np.array([float(label.split('_')[1].replace('g', '')) for label in self.parameter_labels])
            
            # Create PCA visualization
            plt.figure(figsize=(12, 10))
            
            # Plot PCA with parameter coloring
            plt.subplot(2, 1, 1)
            scatter = plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], 
                       c=alpha_params, cmap='viridis', alpha=0.8, s=80)
            plt.colorbar(scatter, label='Alpha (Memory Strength)')
            
            # Optionally annotate points
            for i in range(len(reduced_pca)):
                plt.annotate(
                    self.parameter_labels[i], 
                    (reduced_pca[i, 0], reduced_pca[i, 1]),
                    fontsize=6, 
                    alpha=0.7
                )
            
            plt.title('PCA of Attractor Residuals (Delta Fields) - Colored by Alpha')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot t-SNE if available
            if reduced_tsne is not None:
                plt.subplot(2, 1, 2)
                scatter = plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], 
                           c=gamma_params, cmap='plasma', alpha=0.8, s=80)
                plt.colorbar(scatter, label='Gamma (Memory Decay)')
                
                # Annotate points
                for i in range(len(reduced_tsne)):
                    plt.annotate(
                        self.parameter_labels[i], 
                        (reduced_tsne[i, 0], reduced_tsne[i, 1]),
                        fontsize=6, 
                        alpha=0.7
                    )
                
                plt.title('t-SNE of Attractor Residuals (Delta Fields) - Colored by Gamma')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "visualizations", f"attractor_embedding_{self.run_id}.png"),
                dpi=300
            )
            plt.close()
            
            # Create cluster visualization
            plt.figure(figsize=(12, 10))
            
            # Plot PCA with cluster coloring
            plt.subplot(2, 1, 1)
            scatter = plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], 
                       c=cluster_labels, cmap='tab10', alpha=0.8, s=80)
            plt.colorbar(scatter, label='Cluster ID')
            
            # Annotate points
            for i in range(len(reduced_pca)):
                plt.annotate(
                    self.parameter_labels[i], 
                    (reduced_pca[i, 0], reduced_pca[i, 1]),
                    fontsize=6, 
                    alpha=0.7
                )
            
            plt.title('PCA of Attractor Residuals - Colored by Cluster')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot t-SNE with cluster coloring if available
            if reduced_tsne is not None:
                plt.subplot(2, 1, 2)
                scatter = plt.scatter(reduced_tsne[:, 0], reduced_tsne[:, 1], 
                           c=cluster_labels, cmap='tab10', alpha=0.8, s=80)
                plt.colorbar(scatter, label='Cluster ID')
                
                # Annotate points
                for i in range(len(reduced_tsne)):
                    plt.annotate(
                        self.parameter_labels[i], 
                        (reduced_tsne[i, 0], reduced_tsne[i, 1]),
                        fontsize=6, 
                        alpha=0.7
                    )
                
                plt.title('t-SNE of Attractor Residuals - Colored by Cluster')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "visualizations", f"attractor_clusters_{self.run_id}.png"),
                dpi=300
            )
            plt.close()
            
            # Save reduced coordinates
            embedding_df = pd.DataFrame({
                'label': self.parameter_labels,
                'alpha': alpha_params,
                'gamma': gamma_params,
                'cluster': cluster_labels,
                'pca_x': reduced_pca[:, 0],
                'pca_y': reduced_pca[:, 1],
            })
            
            if reduced_tsne is not None:
                embedding_df['tsne_x'] = reduced_tsne[:, 0]
                embedding_df['tsne_y'] = reduced_tsne[:, 1]
                
            embedding_df.to_csv(
                os.path.join(self.output_dir, f"attractor_embeddings_{self.run_id}.csv"),
                index=False
            )
            
            self.logger.info(f"Attractor analysis completed with {self.clustering_method} clustering")
            
        except Exception as e:
            self.logger.error(f"Error in attractor analysis: {e}")

    # ---------- Extension Hooks (Stubs for future implementation) ----------
    
    def apply_recovery_nudging(self, alpha, gamma, pattern_type="fractal", 
                              detection_step=10, nudge_magnitude=0.1):
        """
        [EXTENSION HOOK] Apply nudging to guide divergent memory back to original basin
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        pattern_type : str
            Pattern type to initialize
        detection_step : int
            Step at which to check for divergence
        nudge_magnitude : float
            Magnitude of the corrective nudge
        """
        self.logger.info(f"Recovery nudging not yet implemented")
        # Placeholder for future implementation
        pass
    
    def catalog_attractor_fingerprints(self, cluster_id, save_templates=True):
        """
        [EXTENSION HOOK] Catalog and analyze attractor fingerprints for each cluster
        
        Parameters:
        -----------
        cluster_id : int
            Cluster to analyze
        save_templates : bool
            Whether to save template patterns
        """
        self.logger.info(f"Attractor fingerprinting not yet implemented")
        # Placeholder for future implementation
        pass
    
    def run_noise_sensitivity_analysis(self, alpha, gamma, n_seeds=10):
        """
        [EXTENSION HOOK] Run multiple noise seeds at specific (alpha, gamma) to test attractor robustness
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        n_seeds : int
            Number of random seeds to test
        """
        self.logger.info(f"Noise sensitivity analysis not yet implemented")
        # Placeholder for future implementation
        pass


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RCFT Attractor Topography Mapping')
    parser.add_argument('--output', type=str, default='phase2_results',
                        help='Output directory for results')
    parser.add_argument('--alpha_min', type=float, default=0.1,
                        help='Minimum value for alpha')
    parser.add_argument('--alpha_max', type=float, default=0.6,
                        help='Maximum value for alpha')
    parser.add_argument('--alpha_steps', type=int, default=11,
                        help='Number of steps for alpha')
    parser.add_argument('--gamma_min', type=float, default=0.8,
                        help='Minimum value for gamma')
    parser.add_argument('--gamma_max', type=float, default=0.99,
                        help='Maximum value for gamma')
    parser.add_argument('--gamma_steps', type=int, default=11,
                        help='Number of steps for gamma')
    parser.add_argument('--delay', type=int, default=10,
                        help='Delay between perturbations')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per parameter point')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = AttractorTopographyMapper(output_dir=args.output)
    
    # Run parameter sweep
    results = mapper.run_parameter_sweep(
        alpha_range=(args.alpha_min, args.alpha_max, args.alpha_steps),
        gamma_range=(args.gamma_min, args.gamma_max, args.gamma_steps),
        delay=args.delay,
        n_trials=args.trials,
        pattern_type="fractal",
        parallel=args.parallel
    )
    
    print(f"Parameter sweep complete. Results saved to {args.output}")