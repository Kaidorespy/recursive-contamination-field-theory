#!/usr/bin/env python3
"""
Runner script for Attractor Topography Mapping experiment suite.
"""

import argparse
import os
import time
from datetime import datetime

from attractor_topography_mapper import AttractorTopographyMapper

def main():
    """Main entry point for running the attractor topography mapping experiment."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Run RCFT Attractor Topography Mapping experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='phase2_results',
                        help='Output directory for results')
    parser.add_argument('--log_dir', type=str, default='phase2_logs',
                        help='Directory for log files')
    
    # Parameter sweep configuration
    parser.add_argument('--alpha_min', type=float, default=0.1,
                        help='Minimum value for alpha (memory strength)')
    parser.add_argument('--alpha_max', type=float, default=0.6,
                        help='Maximum value for alpha (memory strength)')
    parser.add_argument('--alpha_steps', type=int, default=11,
                        help='Number of steps for alpha')
    
    parser.add_argument('--gamma_min', type=float, default=0.8,
                        help='Minimum value for gamma (memory decay)')
    parser.add_argument('--gamma_max', type=float, default=0.99,
                        help='Maximum value for gamma (memory decay)')
    parser.add_argument('--gamma_steps', type=int, default=11,
                        help='Number of steps for gamma')
    
    # Experiment configuration
    parser.add_argument('--delay', type=int, default=10,
                        help='Delay between perturbations')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per parameter point')
    parser.add_argument('--pattern', type=str, default='fractal',
                        choices=['fractal', 'radial', 'horizontal', 'diagonal', 'lattice', 'stochastic'],
                        help='Pattern type to initialize')
    
    # Computational settings
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    parser.add_argument('--clustering', type=str, default='kmeans',
                        choices=['kmeans', 'dbscan', 'none'],
                        help='Clustering method for attractor analysis')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='Number of clusters for K-means clustering')
    
    # Extension hooks (placeholder flags)
    parser.add_argument('--run_nudging', action='store_true',
                        help='Run recovery nudging experiments (not implemented yet)')
    parser.add_argument('--run_fingerprinting', action='store_true',
                        help='Run attractor fingerprinting (not implemented yet)')
    parser.add_argument('--run_noise_sensitivity', action='store_true',
                        help='Run noise sensitivity analysis (not implemented yet)')
    
    args = parser.parse_args()
    
    # Display configuration
    print(f"\n{'='*60}")
    print(f"RCFT Attractor Topography Mapping - Phase II")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Alpha range: {args.alpha_min} to {args.alpha_max} ({args.alpha_steps} steps)")
    print(f"  Gamma range: {args.gamma_min} to {args.gamma_max} ({args.gamma_steps} steps)")
    print(f"  Perturbation delay: {args.delay}")
    print(f"  Trials per point: {args.trials}")
    print(f"  Pattern type: {args.pattern}")
    print(f"  Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"  Total parameter points: {args.alpha_steps * args.gamma_steps}")
    print(f"  Total simulations: {args.alpha_steps * args.gamma_steps * args.trials}")
    print(f"{'='*60}\n")
    
    # Record start time
    start_time = time.time()
    
    # Initialize mapper
    mapper = AttractorTopographyMapper(output_dir=args.output_dir, log_dir=args.log_dir)
    
    # Set clustering method if specified
    if args.clustering != 'none':
        mapper.clustering_method = args.clustering
        mapper.n_clusters = args.n_clusters
    
    try:
        # Run parameter sweep
        results = mapper.run_parameter_sweep(
            alpha_range=(args.alpha_min, args.alpha_max, args.alpha_steps),
            gamma_range=(args.gamma_min, args.gamma_max, args.gamma_steps),
            delay=args.delay,
            n_trials=args.trials,
            pattern_type=args.pattern,
            parallel=args.parallel
        )
        
        # Print summary statistics
        print(f"\nParameter sweep completed successfully!")
        print(f"  Total runtime: {time.time() - start_time:.1f} seconds")
        print(f"  Results saved to: {args.output_dir}")
        
        if results is not None:
            try:
                anomaly_count = results['is_anomalous'].sum()
                total_runs = len(results)
                
                print(f"\nResults Summary:")
                print(f"  Total parameter points: {total_runs}")
                print(f"  Anomalous recoveries: {anomaly_count} ({anomaly_count/total_runs*100:.1f}%)")
            except Exception as e:
                print(f"\nError calculating summary statistics: {e}")
            
            # Get recovery class counts if available
            if 'recovery_class' in results.columns:
                class_counts = results['recovery_class'].value_counts()
                print("\nRecovery Classes:")
                for cls, count in class_counts.items():
                    print(f"  {cls}: {count} ({count/total_runs*100:.1f}%)")
        
        # Run extension hooks if requested (these will just log that they're not implemented)
        if args.run_nudging:
            mapper.apply_recovery_nudging(0.3, 0.9)  # Just a test call with dummy parameters
            
        if args.run_fingerprinting:
            mapper.catalog_attractor_fingerprints(0)  # Just a test call with dummy parameters
            
        if args.run_noise_sensitivity:
            mapper.run_noise_sensitivity_analysis(0.3, 0.9)  # Just a test call with dummy parameters
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError during experiment: {e}")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()