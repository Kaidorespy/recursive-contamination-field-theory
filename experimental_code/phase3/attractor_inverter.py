"""
Attractor Inverter - Module 3 for Phase 3

Uses optimization techniques to invert false attractors and infer their causal history.
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
from scipy.optimize import minimize
from scipy.spatial import distance

from phase3.phase3_core import Phase3Core, FalseAttractorSample
from rcft_framework import RCFTExperiment

class AttractorInverter(Phase3Core):
    """Inverts false attractor fields to infer their causal history"""
    
    def __init__(self, output_dir="phase3_results/attractor_inversion", log_dir="phase3_logs"):
        """Initialize the attractor inverter"""
        super().__init__(output_dir, log_dir)
        
        # Inversion parameters
        self.max_iterations = 500
        self.regularization = 0.1  # Entropy regularization parameter λ
        self.learning_rate = 0.01
        self.initial_field_scale = 0.5  # Scale factor for initial field guess
        
        # Source of false attractor samples
        self.source_samples_dir = None
        
        # Run ID for this experiment
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def set_source_samples(self, source_dir):
        """Set the source directory for false attractor samples"""
        self.source_samples_dir = source_dir
        self.logger.info(f"Set source samples directory: {source_dir}")
    
    def load_attractor_samples(self, boundary_mapper=None, n_samples=5):
        """
        Load attractor samples for inversion
        
        Parameters:
        -----------
        boundary_mapper : BoundaryCartographyMapper, optional
            Boundary mapper to extract samples from
        n_samples : int
            Number of samples to load
            
        Returns:
        --------
        list
            List of FalseAttractorSample objects
        """
        if boundary_mapper is not None:
            # Extract from boundary mapper
            self.logger.info(f"Extracting {n_samples} samples from boundary mapper")
            return boundary_mapper.identify_edge_cases(n_samples=n_samples)
        elif self.source_samples_dir is not None:
            # Load from source directory
            self.logger.info(f"Loading samples from source directory: {self.source_samples_dir}")
            all_samples = self.load_samples(self.source_samples_dir)
            
            if not all_samples:
                self.logger.warning("No samples found in source directory")
                return []
                
            # Sort by CCDI to find most anomalous samples
            all_samples.sort(key=lambda x: x.ccdi, reverse=True)
            
            # Return top N samples
            return all_samples[:min(n_samples, len(all_samples))]
        else:
            self.logger.warning("No source for attractor samples specified")
            return []
    
    def run_inversion_experiments(self, attractor_samples=None, n_samples=5, parallel=False):
        """
        Run inversion experiments on attractor samples
        
        Parameters:
        -----------
        attractor_samples : list, optional
            List of FalseAttractorSample objects to invert
        n_samples : int
            Number of samples to process
        parallel : bool
            Whether to use multiprocessing
            
        Returns:
        --------
        pd.DataFrame
            Results DataFrame
        """
        # Load samples if not provided
        if attractor_samples is None or len(attractor_samples) == 0:
            attractor_samples = self.load_attractor_samples(n_samples=n_samples)
            
        if not attractor_samples:
            self.logger.error("No attractor samples available for inversion")
            return None
            
        # Limit to n_samples
        if len(attractor_samples) > n_samples:
            self.logger.info(f"Limiting to {n_samples} samples")
            attractor_samples = attractor_samples[:n_samples]
            
        self.logger.info(f"Running inversion experiments on {len(attractor_samples)} samples")
        
        # Initialize results storage
        results_list = []
        self.samples = []
        
        # Run for each sample
        if parallel:
            self.logger.info("Using parallel processing")
            
            # Configure process pool
            import multiprocessing as mp
            num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
            self.logger.info(f"Using {num_cores} cores")
            
            # Create a pool of workers
            with mp.Pool(processes=num_cores) as pool:
                # Submit jobs to pool
                results = []
                for i, sample in enumerate(attractor_samples):
                    run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_inversion"
                    result = pool.apply_async(self._run_single_inversion, (sample, run_id))
                    results.append(result)
                
                # Collect results as they complete
                for i, result in enumerate(tqdm(results, desc="Inversion Experiments")):
                    try:
                        inverted_sample, result_dict = result.get()
                        self.samples.append(inverted_sample)
                        results_list.append(result_dict)
                    except Exception as e:
                        self.logger.error(f"Error in parallel inversion: {e}")
        else:
            self.logger.info("Using sequential processing")
            
            # Process samples sequentially
            for sample in tqdm(attractor_samples, desc="Inversion Experiments"):
                try:
                    run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_inversion"
                    inverted_sample, result_dict = self._run_single_inversion(sample, run_id)
                    self.samples.append(inverted_sample)
                    results_list.append(result_dict)
                except Exception as e:
                    self.logger.error(f"Error in inversion experiment: {e}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        # Generate visualizations
        self._generate_inversion_visualizations()
        
        # Save final results
        if len(results_list) > 0:
            self.save_results(self.results_df, f"inversion_results_{self.run_id}.csv")
            
        self.logger.info("Inversion experiments completed successfully")
        return self.results_df
    
    def _run_single_inversion(self, sample, run_id=None):
        """
        Run a single inversion experiment
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Sample to invert
        run_id : str, optional
            Identifier for this run
            
        Returns:
        --------
        tuple
            (FalseAttractorSample, dict) - Inverted sample and result dictionary
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"a{sample.alpha:.4f}_g{sample.gamma:.4f}_inversion"
            
        # Create output directory
        run_dir = os.path.join(self.output_dir, "samples", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Perform inversion
        start_time = time.time()
        inverted_field, loss_history = self._invert_attractor(sample)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        inversion_metrics = self._calculate_inversion_metrics(sample, inverted_field)
        
        # Create result dictionary
        result_dict = {
            'alpha': sample.alpha,
            'gamma': sample.gamma,
            'original_ccdi': sample.ccdi,
            'original_correlation': sample.metrics['final_correlation'],
            'inverted_correlation': inversion_metrics['correlation_with_initial'],
            'inverted_field_entropy': inversion_metrics['inverted_field_entropy'],
            'true_field_entropy': inversion_metrics['true_field_entropy'],
            'field_similarity': inversion_metrics['field_similarity'],
            'final_loss': inversion_metrics['final_loss'],
            'runtime': elapsed_time,
            'iterations': inversion_metrics['iterations'],
            'run_id': run_id
        }
        
        # Create a sample object for the inverted result
        inverted_sample = FalseAttractorSample(
            alpha=sample.alpha,
            gamma=sample.gamma,
            initial_state=sample.initial_state,
            final_state=sample.final_state,
            perturbation_info=sample.perturbation_info,
            recovery_trajectory=sample.recovery_trajectory,
            metrics={
                **sample.metrics,
                **inversion_metrics
            }
        )
        
        # Save inverted field
        np.save(os.path.join(run_dir, "inverted_field.npy"), inverted_field)
        
        # Save loss history
        np.save(os.path.join(run_dir, "loss_history.npy"), np.array(loss_history))
        
        # Save visualizations
        self._visualize_inversion(sample, inverted_field, inversion_metrics, loss_history, run_dir)
        
        # Save result
        with open(os.path.join(run_dir, "inversion_metrics.json"), 'w') as f:
            # Convert all values to serializable types
            serializable_metrics = {k: float(v) if isinstance(v, np.float) else v 
                                   for k, v in inversion_metrics.items()}
            json.dump(serializable_metrics, f, indent=2)
            
        return inverted_sample, result_dict
    
    def _invert_attractor(self, sample):
        """
        Invert an attractor to infer its causal history
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Sample to invert
            
        Returns:
        --------
        tuple
            (inverted_field, loss_history) - The inferred origin field and optimization history
        """
        # Extract original and target states
        initial_state = sample.initial_state
        false_attractor = sample.final_state
        
        # Make a first guess based on the false attractor
        current_field = false_attractor * self.initial_field_scale
        
        # Normalize the field
        current_field = current_field / np.max(np.abs(current_field))
        
        # Track loss history
        loss_history = []
        
        # Create simulation parameters matching the sample
        alpha = sample.alpha
        gamma = sample.gamma
        
        # Create a loss function for optimization
        def simulation_loss(field_params):
            # Reshape field parameters into a 2D grid
            field = field_params.reshape(initial_state.shape)
            
            # Run simulation with this field as initial state
            exp = RCFTExperiment(
                memory_strength=alpha,
                coupling_strength=self.beta,
                memory_decay=gamma
            )
            
            # Initialize with the field
            exp.state = field.copy()
            exp.memory = field.copy()
            exp.initial_state = field.copy()
            
            # Record initial state
            exp.history = [exp.state.copy()]
            exp.memory_history = [exp.memory.copy()]
            exp._calculate_metrics()
            
            # Apply perturbations matching the original sample
            exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
            
            # Run until delay
            delay = sample.metrics.get('delay', 10)
            exp.update(steps=delay)
            
            # Apply second perturbation
            exp.apply_perturbation(
                perturbation_type="flip", 
                magnitude=1.0, 
                center=(20, 20),
                radius=10
            )
            
            # Let system evolve to completion
            exp.update(steps=50)
            
            # Calculate loss as distance between simulated result and target false attractor
            simulated_result = exp.state
            result_diff = simulated_result - false_attractor
            mse_loss = np.mean(result_diff**2)
            
            # Add entropy regularization to promote smoothness
            field_entropy = -np.sum(np.abs(field) * np.log(np.abs(field) + 1e-10))
            
            # Combined loss with regularization
            loss = mse_loss + self.regularization * field_entropy
            
            return loss
        
        # Initial guess as flattened array
        initial_params = current_field.flatten()
        
        # Use scipy's L-BFGS-B optimizer
        result = minimize(
            simulation_loss,
            initial_params,
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iterations,
                'gtol': 1e-6,
                'disp': False
            },
            callback=lambda x: loss_history.append(simulation_loss(x))
        )
        
        # Reshape result back to 2D field
        inverted_field = result.x.reshape(initial_state.shape)
        
        return inverted_field, loss_history
    
    def _calculate_inversion_metrics(self, sample, inverted_field):
        """
        Calculate metrics for an inverted field
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Original sample
        inverted_field : ndarray
            Inverted field
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        # Extract original states
        initial_state = sample.initial_state
        false_attractor = sample.final_state
        
        # Calculate correlation with true initial state
        correlation_with_initial = np.corrcoef(
            inverted_field.flatten(), 
            initial_state.flatten()
        )[0, 1]
        
        # Calculate field entropy
        inverted_field_entropy = -np.sum(np.abs(inverted_field) * np.log(np.abs(inverted_field) + 1e-10))
        true_field_entropy = -np.sum(np.abs(initial_state) * np.log(np.abs(initial_state) + 1e-10))
        
        # Field similarity metric (normalized)
        field_similarity = 1 - np.mean(np.abs(inverted_field - initial_state)) / 2.0
        
        # Run forward simulation with inverted field
        exp = RCFTExperiment(
            memory_strength=sample.alpha,
            coupling_strength=self.beta,
            memory_decay=sample.gamma
        )
        
        # Initialize with inverted field
        exp.state = inverted_field.copy()
        exp.memory = inverted_field.copy()
        exp.initial_state = inverted_field.copy()
        
        # Record initial state
        exp.history = [exp.state.copy()]
        exp.memory_history = [exp.memory.copy()]
        exp._calculate_metrics()
        
        # Apply perturbations matching the original sample
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        
        # Run until delay
        delay = sample.metrics.get('delay', 10)
        exp.update(steps=delay)
        
        # Apply second perturbation
        exp.apply_perturbation(
            perturbation_type="flip", 
            magnitude=1.0, 
            center=(20, 20),
            radius=10
        )
        
        # Let system evolve to completion
        exp.update(steps=50)
        
        # Calculate final metrics
        simulated_result = exp.state
        result_diff = simulated_result - false_attractor
        mse_loss = np.mean(result_diff**2)
        
        # Additional metrics
        metrics = {
            'correlation_with_initial': correlation_with_initial,
            'inverted_field_entropy': float(inverted_field_entropy),
            'true_field_entropy': float(true_field_entropy),
            'field_similarity': field_similarity,
            'reconstruction_mse': float(mse_loss),
            'final_loss': float(mse_loss + self.regularization * inverted_field_entropy),
            'iterations': len(exp.history)
        }
        
        return metrics
    
    def _visualize_inversion(self, sample, inverted_field, metrics, loss_history, output_dir):
        """
        Visualize the inversion results
        
        Parameters:
        -----------
        sample : FalseAttractorSample
            Original sample
        inverted_field : ndarray
            Inverted field
        metrics : dict
            Inversion metrics
        loss_history : list
            Optimization loss history
        output_dir : str
            Output directory for visualizations
        """
        # Extract original states
        initial_state = sample.initial_state
        false_attractor = sample.final_state
        
        # Create field comparison visualization
        plt.figure(figsize=(15, 5))
        
        # True initial field
        plt.subplot(1, 3, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("True Initial Field")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Inverted field
        plt.subplot(1, 3, 2)
        plt.imshow(inverted_field, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Inverted Field\nCorrelation: {metrics['correlation_with_initial']:.4f}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Field difference
        plt.subplot(1, 3, 3)
        plt.imshow(inverted_field - initial_state, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"Field Difference\nSimilarity: {metrics['field_similarity']:.4f}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "field_comparison.png"), dpi=300)
        plt.close()
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Inversion Optimization")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=300)
        plt.close()
        
        # Create false attractor comparison visualization
        plt.figure(figsize=(15, 5))
        
        # False attractor (original)
        plt.subplot(1, 3, 1)
        plt.imshow(false_attractor, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Original False Attractor")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Simulated false attractor from inverted field
        # Run forward simulation with inverted field
        exp = RCFTExperiment(
            memory_strength=sample.alpha,
            coupling_strength=self.beta,
            memory_decay=sample.gamma
        )
        
        # Initialize with inverted field
        exp.state = inverted_field.copy()
        exp.memory = inverted_field.copy()
        exp.initial_state = inverted_field.copy()
        
        # Apply perturbations matching the original sample
        exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        
        # Run until delay
        delay = sample.metrics.get('delay', 10)
        exp.update(steps=delay)
        
        # Apply second perturbation
        exp.apply_perturbation(
            perturbation_type="flip", 
            magnitude=1.0, 
            center=(20, 20),
            radius=10
        )
        
        # Let system evolve to completion
        exp.update(steps=50)
        
        # Get simulated result
        simulated_result = exp.state
        
        # Plot simulated result
        plt.subplot(1, 3, 2)
        plt.imshow(simulated_result, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Simulated from Inverted Field\nMSE: {metrics['reconstruction_mse']:.6f}")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Plot difference
        plt.subplot(1, 3, 3)
        plt.imshow(simulated_result - false_attractor, cmap='RdBu', vmin=-0.2, vmax=0.2)
        plt.title("Reconstruction Difference")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attractor_reconstruction.png"), dpi=300)
        plt.close()
    
    def _generate_inversion_visualizations(self):
        """Generate summary visualizations for inversion results"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for visualizations")
            return
            
        # Create directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Scatter plot of correlation vs field entropy
        try:
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot
            sns.scatterplot(
                x='inverted_field_entropy', 
                y='inverted_correlation',
                size='field_similarity',
                hue='original_ccdi',
                sizes=(50, 200),
                palette='viridis',
                data=self.results_df
            )
            
            # Set labels
            plt.title("Inversion Quality vs Field Complexity")
            plt.xlabel("Inverted Field Entropy")
            plt.ylabel("Correlation with True Initial State")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add diagonal guidelines
            for corr in [0.2, 0.4, 0.6, 0.8]:
                plt.axhline(y=corr, color='gray', linestyle=':', alpha=0.5)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"inversion_quality_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating inversion quality plot: {e}")
            
        # Bar chart of correlation with true initial state
        try:
            plt.figure(figsize=(12, 6))
            
            # Sort by correlation
            sorted_df = self.results_df.sort_values('inverted_correlation', ascending=False)
            
            # Format labels with alpha and gamma
            labels = [f"α={row['alpha']:.2f}, γ={row['gamma']:.2f}" 
                     for _, row in sorted_df.iterrows()]
            
            # Create bar chart
            bars = plt.bar(
                range(len(sorted_df)),
                sorted_df['inverted_correlation'],
                color=plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
            )
            
            # Set labels
            plt.title("Correlation Between Inverted and True Initial Fields")
            plt.xlabel("Sample")
            plt.ylabel("Correlation")
            plt.xticks(range(len(sorted_df)), labels, rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}",
                    ha='center',
                    fontsize=9
                )
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"inversion_correlation_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating inversion correlation plot: {e}")
            
        # Bar chart of reconstruction MSE
        try:
            plt.figure(figsize=(12, 6))
            
            # Sort by MSE
            sorted_df = self.results_df.sort_values('reconstruction_mse', ascending=True)
            
            # Format labels with alpha and gamma
            labels = [f"α={row['alpha']:.2f}, γ={row['gamma']:.2f}" 
                     for _, row in sorted_df.iterrows()]
            
            # Create bar chart with logarithmic scale
            bars = plt.bar(
                range(len(sorted_df)),
                sorted_df['reconstruction_mse'],
                color=plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
            )
            
            # Set labels
            plt.title("False Attractor Reconstruction MSE")
            plt.xlabel("Sample")
            plt.ylabel("MSE (log scale)")
            plt.yscale('log')
            plt.xticks(range(len(sorted_df)), labels, rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.1,
                    f"{bar.get_height():.2e}",
                    ha='center',
                    fontsize=9
                )
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"reconstruction_mse_{self.run_id}.png"), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating reconstruction MSE plot: {e}")
    
    def summarize_inversion_results(self):
        """Print a summary of the inversion results"""
        if self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for summary")
            return
            
        print("\nAttractor Inversion Summary:")
        print(f"Total Samples: {len(self.results_df)}")
        
        # Calculate average correlation with true initial state
        avg_corr = self.results_df['inverted_correlation'].mean()
        min_corr = self.results_df['inverted_correlation'].min()
        max_corr = self.results_df['inverted_correlation'].max()
        
        print(f"\nCorrelation with True Initial State:")
        print(f"  Average: {avg_corr:.4f}")
        print(f"  Range: [{min_corr:.4f}, {max_corr:.4f}]")
        
        # Calculate average reconstruction MSE
        avg_mse = self.results_df['reconstruction_mse'].mean()
        min_mse = self.results_df['reconstruction_mse'].min()
        max_mse = self.results_df['reconstruction_mse'].max()
        
        print(f"\nReconstruction MSE:")
        print(f"  Average: {avg_mse:.6f}")
        print(f"  Range: [{min_mse:.6f}, {max_mse:.6f}]")
        
        # Find best and worst inversions
        best_idx = self.results_df['inverted_correlation'].idxmax()
        worst_idx = self.results_df['inverted_correlation'].idxmin()
        
        if best_idx is not None:
            best_row = self.results_df.loc[best_idx]
            print(f"\nBest Inversion:")
            print(f"  Alpha: {best_row['alpha']:.4f}, Gamma: {best_row['gamma']:.4f}")
            print(f"  Correlation: {best_row['inverted_correlation']:.4f}")
            print(f"  Field Similarity: {best_row['field_similarity']:.4f}")
            print(f"  MSE: {best_row['reconstruction_mse']:.6f}")
        
        if worst_idx is not None:
            worst_row = self.results_df.loc[worst_idx]
            print(f"\nWorst Inversion:")
            print(f"  Alpha: {worst_row['alpha']:.4f}, Gamma: {worst_row['gamma']:.4f}")
            print(f"  Correlation: {worst_row['inverted_correlation']:.4f}")
            print(f"  Field Similarity: {worst_row['field_similarity']:.4f}")
            print(f"  MSE: {worst_row['reconstruction_mse']:.6f}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RCFT Attractor Inversion')
    parser.add_argument('--output_dir', type=str, default='phase3_results/attractor_inversion',
                        help='Output directory for results')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Source directory for false attractor samples')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of attractor samples to analyze')
    parser.add_argument('--max_iterations', type=int, default=500,
                        help='Maximum iterations for optimization')
    parser.add_argument('--regularization', type=float, default=0.1,
                        help='Entropy regularization parameter')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Initialize inverter
    inverter = AttractorInverter(output_dir=args.output_dir)
    
    # Set parameters
    inverter.max_iterations = args.max_iterations
    inverter.regularization = args.regularization
    
    # Set source directory
    if args.source_dir:
        inverter.set_source_samples(args.source_dir)
    
    # Run inversion experiments
    results = inverter.run_inversion_experiments(
        n_samples=args.n_samples,
        parallel=args.parallel
    )
    
    # Print summary
    inverter.summarize_inversion_results()
    
    print(f"\nAttractor inversion complete. Results saved to {args.output_dir}")
