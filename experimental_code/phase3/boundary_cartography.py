"""
Boundary Cartography - Module 1 for Phase 3

Maps the fine-grained phase transition boundary between true and false memory attractors.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from phase3.phase3_core import Phase3Core, FalseAttractorSample

class AttractorBoundaryMapper(Phase3Core):
    """Maps the fine-grained transition boundary in parameter space"""
    
    def __init__(self, output_dir="phase3_results/boundary_cartography", log_dir="phase3_logs"):
        """Initialize the boundary mapper"""
        super().__init__(output_dir, log_dir)
        
        # Default parameters for boundary exploration
        self.alpha_range = np.linspace(0.32, 0.38, 7)  # 0.01 step size
        self.gamma_range = np.linspace(0.88, 0.96, 9)  # 0.01 step size
        self.delay = 10
        self.n_trials = 1
        self.pattern_type = "fractal"
        
        # For clustering
        self.n_clusters = 3
        
        # Timestamp for this run
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_boundary_exploration(self, alpha_range=None, gamma_range=None, 
                               delay=None, n_trials=None, pattern_type=None,
                               parallel=False):
        """
        Run a fine-grained parameter sweep to map transition boundaries
        
        Parameters:
        -----------
        alpha_range : ndarray, optional
            Range of alpha values to explore
        gamma_range : ndarray, optional
            Range of gamma values to explore
        delay : int, optional
            Delay between perturbations
        n_trials : int, optional
            Number of trials per parameter combination
        pattern_type : str, optional
            Pattern type to initialize
        parallel : bool
            Whether to use multiprocessing
            
        Returns:
        --------
        pd.DataFrame
            Results DataFrame
        """
        # Use provided parameters or defaults
        if alpha_range is not None:
            self.alpha_range = alpha_range
        if gamma_range is not None:
            self.gamma_range = gamma_range
        if delay is not None:
            self.delay = delay
        if n_trials is not None:
            self.n_trials = n_trials
        if pattern_type is not None:
            self.pattern_type = pattern_type
            
        self.logger.info(f"Starting boundary exploration: alpha={self.alpha_range[0]}-{self.alpha_range[-1]}, " +
                        f"gamma={self.gamma_range[0]}-{self.gamma_range[-1]}, delay={self.delay}, trials={self.n_trials}")
        
        # Create parameter combinations
        parameter_grid = [(alpha, gamma) for alpha in self.alpha_range for gamma in self.gamma_range]
        total_runs = len(parameter_grid) * self.n_trials
        
        self.logger.info(f"Boundary exploration will perform {total_runs} total simulations")
        
        # Initialize results storage
        results_list = []
        self.samples = []
        
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
                    for trial in range(self.n_trials):
                        seed = self.base_seed + i * self.n_trials + trial
                        run_id = f"a{alpha:.4f}_g{gamma:.4f}_{trial}"
                        
                        # Submit job to pool
                        result = pool.apply_async(
                            self._run_single_exploration, 
                            (alpha, gamma, self.delay, self.pattern_type, run_id, seed)
                        )
                        results.append(result)
                
                # Collect results as they complete
                for i, result in enumerate(tqdm(results, desc="Parameter Points")):
                    try:
                        sample, result_dict = result.get()
                        self.samples.append(sample)
                        results_list.append(result_dict)
                        
                        # Save ongoing results
                        if (i + 1) % 5 == 0 or (i + 1) == len(results):
                            self._save_interim_results(results_list)
                    except Exception as e:
                        self.logger.error(f"Error in parallel execution: {e}")
        else:
            self.logger.info("Using sequential processing")
            
            # Process parameters sequentially
            for i, (alpha, gamma) in enumerate(tqdm(parameter_grid, desc="Parameter Points")):
                for trial in range(self.n_trials):
                    seed = self.base_seed + i * self.n_trials + trial
                    run_id = f"a{alpha:.4f}_g{gamma:.4f}_{trial}"
                    
                    try:
                        sample, result_dict = self._run_single_exploration(alpha, gamma, self.delay, 
                                                                         self.pattern_type, run_id, seed)
                        self.samples.append(sample)
                        results_list.append(result_dict)
                    except Exception as e:
                        self.logger.error(f"Error at alpha={alpha}, gamma={gamma}, trial={trial}: {e}")
                        
                    # Save ongoing results
                    if (i * self.n_trials + trial + 1) % 5 == 0 or (i == len(parameter_grid) - 1 and trial == self.n_trials - 1):
                        self._save_interim_results(results_list)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        # Generate visualizations
        self._generate_heatmaps()
        self._analyze_clustering()
        
        # Save final results
        self.save_results(self.results_df, f"boundary_exploration_{self.run_id}.csv")
        
        self.logger.info("Boundary exploration completed successfully")
        return self.results_df
    
    def _run_single_exploration(self, alpha, gamma, delay, pattern_type, run_id, seed=None):
        """
        Run a single exploration point and return results
        
        Parameters:
        -----------
        alpha : float
            Memory strength parameter
        gamma : float
            Memory decay parameter
        delay : int
            Delay between perturbations
        pattern_type : str
            Pattern type to initialize
        run_id : str
            Identifier for this run
        seed : int, optional
            Random seed
            
        Returns:
        --------
        tuple
            (FalseAttractorSample, dict) - Sample object and result dictionary
        """
        # Run simulation
        sample = self.run_rcft_simulation(
            alpha=alpha,
            gamma=gamma,
            pattern_type=pattern_type,
            delay=delay,
            run_id=run_id,
            seed=seed
        )
        
        # Extract additional sample fingerprint
        fingerprint = self.extract_trajectory_fingerprint(sample, n_points=10)
        
        # Create result dictionary
        result_dict = {
            'alpha': alpha,
            'gamma': gamma,
            'delay': delay,
            'trial_id': 0 if seed is None else seed - self.base_seed,
            'final_correlation': sample.metrics['final_correlation'],
            'final_coherence': sample.metrics['final_coherence'],
            'final_entropy': sample.metrics['final_entropy'],
            'ccdi': sample.ccdi,
            'recovery_class': sample.recovery_class,
            'is_anomalous': sample.is_anomalous,
            'residual_norm': np.linalg.norm(sample.residual),
            'run_id': run_id
        }
        
        # Add fingerprint features
        for i in range(len(fingerprint)):
            result_dict[f'fp_{i}'] = fingerprint[i]
        
        return sample, result_dict
    
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
    
    def _generate_heatmaps(self):
        """Generate heatmap visualizations of the parameter space"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for heatmaps")
            return
            
        # Define metrics to visualize
        metrics = ['final_correlation', 'ccdi', 'final_coherence', 'residual_norm', 'final_entropy']
        
        # Create heatmaps
        for metric in metrics:
            try:
                plt.figure(figsize=(10, 8))
                plt.rcParams.update({'font.size': 12})
                
                # Pivot data for heatmap
                pivot_data = self.results_df.pivot_table(
                    index='gamma',
                    columns='alpha',
                    values=metric,
                    aggfunc='mean'  # In case of multiple trials
                )
                
                # Create heatmap
                ax = sns.heatmap(
                    pivot_data, 
                    cmap='viridis',
                    annot=True, 
                    fmt=".3f",
                    linewidths=.5, 
                    cbar_kws={'label': metric.replace('_', ' ').title()}
                )
                
                # Set labels
                plt.title(f"{metric.replace('_', ' ').title()} Across Parameter Space")
                plt.xlabel("Memory Strength (Alpha)")
                plt.ylabel("Memory Decay (Gamma)")
                
                # Save figure
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, "visualizations", f"heatmap_{metric}_{self.run_id}.png"),
                    dpi=300
                )
                plt.close()
            except Exception as e:
                self.logger.error(f"Error generating heatmap for {metric}: {e}")
        
        # Create anomaly classification heatmap
        try:
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({'font.size': 12})
            
            # Pivot data
            pivot_data = self.results_df.pivot_table(
                index='gamma',
                columns='alpha',
                values='is_anomalous',
                aggfunc='mean'  # Shows proportion of anomalous trials
            )
            
            # Create heatmap
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
            
            # Draw contour at transition boundary (0.5)
            if np.any((pivot_data.values >= 0.4) & (pivot_data.values <= 0.6)):
                # Add contour lines
                x = np.arange(0, pivot_data.shape[1] + 1)
                y = np.arange(0, pivot_data.shape[0] + 1)
                plt.contour(x, y, pivot_data.values, levels=[0.5], colors='black', linewidths=2)
                
            # Set labels
            plt.title("Phase Transition Boundary: Normal vs. False Memory")
            plt.xlabel("Memory Strength (Alpha)")
            plt.ylabel("Memory Decay (Gamma)")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "visualizations", f"anomaly_boundary_{self.run_id}.png"),
                dpi=300
            )
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating anomaly boundary plot: {e}")
    
    def _analyze_clustering(self):
        """Analyze clustering of the attractor space"""
        if len(self.samples) == 0:
            self.logger.warning("No samples available for clustering analysis")
            return
            
        try:
            # Extract residuals and flatten
            residuals = np.array([sample.residual.flatten() for sample in self.samples])
            
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=10)
            reduced_data = pca.fit_transform(residuals)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_data)
            
            # Add cluster labels to results DataFrame
            if self.results_df is not None:
                # Create mapping from run_id to cluster
                run_ids = [sample.to_dict().get('run_id', f"a{sample.alpha:.4f}_g{sample.gamma:.4f}") 
                          for sample in self.samples]
                cluster_mapping = dict(zip(run_ids, cluster_labels))
                
                # Add to DataFrame if run_id column exists
                if 'run_id' in self.results_df.columns:
                    self.results_df['cluster'] = self.results_df['run_id'].map(
                        lambda x: cluster_mapping.get(x, -1)
                    )
                    
                    # Update saved results
                    self.save_results(self.results_df, f"boundary_exploration_{self.run_id}.csv")
            
            # Create cluster heatmap
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({'font.size': 12})
            
            # Create a DataFrame with cluster labels
            cluster_df = pd.DataFrame({
                'alpha': [sample.alpha for sample in self.samples],
                'gamma': [sample.gamma for sample in self.samples],
                'cluster': cluster_labels
            })
            
            # Pivot data for heatmap
            pivot_data = cluster_df.pivot_table(
                index='gamma',
                columns='alpha',
                values='cluster',
                aggfunc='mean'  # In case of multiple trials
            )
            
            # Create heatmap
            ax = sns.heatmap(
                pivot_data, 
                cmap='tab10',
                annot=True, 
                fmt=".0f",
                linewidths=.5,
                cbar_kws={'label': 'Cluster ID'}
            )
            
            # Set labels
            plt.title("Attractor Clusters Across Parameter Space")
            plt.xlabel("Memory Strength (Alpha)")
            plt.ylabel("Memory Decay (Gamma)")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "visualizations", f"cluster_heatmap_{self.run_id}.png"),
                dpi=300
            )
            plt.close()
            
            # Save PCA components and clustering
            embedding_df = pd.DataFrame({
                'alpha': [sample.alpha for sample in self.samples],
                'gamma': [sample.gamma for sample in self.samples],
                'cluster': cluster_labels
            })
            
            # Add PCA components
            for i in range(reduced_data.shape[1]):
                embedding_df[f'pca_{i}'] = reduced_data[:, i]
                
            # Save to CSV
            embedding_path = os.path.join(self.output_dir, f"cluster_embeddings_{self.run_id}.csv")
            embedding_df.to_csv(embedding_path, index=False)
            
            self.logger.info(f"Clustering analysis completed: {self.n_clusters} clusters identified")
            
        except Exception as e:
            self.logger.error(f"Error in clustering analysis: {e}")
    
    def identify_edge_cases(self, ccdi_threshold=None, n_samples=5):
        """
        Identify edge cases for further analysis
        
        Parameters:
        -----------
        ccdi_threshold : float, optional
            CCDI threshold for anomaly classification
        n_samples : int
            Number of edge cases to identify
            
        Returns:
        --------
        list
            List of FalseAttractorSample objects representing edge cases
        """
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for edge case identification")
            return []
            
        if ccdi_threshold is None:
            ccdi_threshold = self.ccdi_threshold
            
        # First identify boundary cases - where CCDI is close to threshold
        boundary_df = self.results_df.copy()
        boundary_df['ccdi_diff'] = abs(boundary_df['ccdi'] - ccdi_threshold)
        boundary_df = boundary_df.sort_values('ccdi_diff')
        
        # Select top N boundary cases
        boundary_cases = boundary_df.head(n_samples)
        
        # Load corresponding samples
        edge_samples = []
        for _, row in boundary_cases.iterrows():
            alpha = row['alpha']
            gamma = row['gamma']
            
            # Find closest sample
            closest_sample = None
            min_distance = float('inf')
            
            for sample in self.samples:
                distance = abs(sample.alpha - alpha) + abs(sample.gamma - gamma)
                if distance < min_distance:
                    min_distance = distance
                    closest_sample = sample
                    
            if closest_sample is not None:
                edge_samples.append(closest_sample)
                
        self.logger.info(f"Identified {len(edge_samples)} edge cases for analysis")
        return edge_samples
    
    def summarize_boundary(self):
        """Print a summary of the boundary exploration"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for summary")
            return
            
        # Count anomalies
        n_anomalies = self.results_df['is_anomalous'].sum()
        n_total = len(self.results_df)
        
        print("\nBoundary Exploration Summary:")
        print(f"Parameter Space: Alpha [{self.alpha_range[0]:.2f}, {self.alpha_range[-1]:.2f}], " + 
              f"Gamma [{self.gamma_range[0]:.2f}, {self.gamma_range[-1]:.2f}]")
        print(f"Total Simulations: {n_total}")
        print(f"Anomalous Recoveries: {n_anomalies} ({n_anomalies/n_total*100:.1f}%)")
        
        # Count recovery classes
        class_counts = self.results_df['recovery_class'].value_counts()
        print("\nRecovery Classes:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/n_total*100:.1f}%)")
            
        # Find boundary edges using CCDI
        if 'ccdi' in self.results_df.columns:
            # Get parameter ranges where CCDI is close to threshold
            ccdi_diff = abs(self.results_df['ccdi'] - self.ccdi_threshold)
            boundary_df = self.results_df[ccdi_diff < 0.02]  # Within 0.02 of threshold
            
            if len(boundary_df) > 0:
                print("\nBoundary Region Parameters:")
                for _, row in boundary_df.iterrows():
                    print(f"  Alpha: {row['alpha']:.4f}, Gamma: {row['gamma']:.4f}, CCDI: {row['ccdi']:.4f}")
            else:
                print("\nNo clear boundary region identified.")
                
        # Cluster distribution if available
        if 'cluster' in self.results_df.columns:
            cluster_counts = self.results_df['cluster'].value_counts()
            print("\nCluster Distribution:")
            for cluster, count in cluster_counts.items():
                print(f"  Cluster {cluster}: {count} ({count/n_total*100:.1f}%)")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RCFT Boundary Cartography')
    parser.add_argument('--output_dir', type=str, default='phase3_results/boundary_cartography',
                        help='Output directory for results')
    parser.add_argument('--alpha_min', type=float, default=0.32,
                        help='Minimum value for alpha')
    parser.add_argument('--alpha_max', type=float, default=0.38,
                        help='Maximum value for alpha')
    parser.add_argument('--alpha_steps', type=int, default=7,
                        help='Number of steps for alpha (0.01 increment for 7 steps)')
    parser.add_argument('--gamma_min', type=float, default=0.88,
                        help='Minimum value for gamma')
    parser.add_argument('--gamma_max', type=float, default=0.96,
                        help='Maximum value for gamma')
    parser.add_argument('--gamma_steps', type=int, default=9,
                        help='Number of steps for gamma (0.01 increment for 9 steps)')
    parser.add_argument('--delay', type=int, default=10,
                        help='Delay between perturbations')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per parameter point')
    parser.add_argument('--pattern', type=str, default='fractal',
                        choices=['fractal', 'radial', 'horizontal', 'diagonal', 'lattice', 'stochastic'],
                        help='Pattern type to initialize')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = AttractorBoundaryMapper(output_dir=args.output_dir)
    
    # Run boundary exploration
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    gamma_range = np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps)
    
    results = mapper.run_boundary_exploration(
        alpha_range=alpha_range,
        gamma_range=gamma_range,
        delay=args.delay,
        n_trials=args.trials,
        pattern_type=args.pattern,
        parallel=args.parallel
    )
    
    # Print summary
    mapper.summarize_boundary()
    
    print(f"\nBoundary exploration complete. Results saved to {args.output_dir}")
