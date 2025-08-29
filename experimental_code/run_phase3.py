#!/usr/bin/env python3
"""
Runner script for RCFT Phase III experiments.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np

from phase3.boundary_cartography import AttractorBoundaryMapper
from phase3.nudging_controller import NudgingController
# Import other modules as they are implemented


def setup_logging(log_dir="phase3_logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"phase3_run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("Phase3Runner")
    return logger


def run_phase3_module1(args, logger):
    """Run Phase III Module 1: Boundary Cartography"""
    logger.info("Starting Phase III Module 1: Boundary Cartography")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "boundary_cartography")
    
    # Initialize mapper
    mapper = AttractorBoundaryMapper(output_dir=output_dir)
    
    # Setup parameter ranges
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    gamma_range = np.linspace(args.gamma_min, args.gamma_max, args.gamma_steps)
    
    # Run boundary exploration
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
    
    logger.info(f"Completed Module 1. Results saved to {output_dir}")
    return mapper


def run_phase3_module2(args, logger, boundary_mapper=None):
    """Run Phase III Module 2: Nudging Experiments"""
    logger.info("Starting Phase III Module 2: Nudging Experiments")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "nudging_experiments")
    
    # Initialize controller
    controller = NudgingController(output_dir=output_dir)
    
    # Set source directory
    if args.source_dir:
        controller.set_source_samples(args.source_dir)
    
    # Parse nudging parameters
    nudge_times = [int(t) for t in args.nudge_times.split(',')]
    nudge_amplitudes = [float(a) for a in args.nudge_amplitudes.split(',')]
    nudge_types = args.nudge_types.split(',')
    
    # Get edge cases from boundary mapper or load from source directory
    if boundary_mapper is not None:
        edge_cases = boundary_mapper.identify_edge_cases(n_samples=args.n_samples)
    else:
        edge_cases = controller.load_edge_cases(n_samples=args.n_samples)
    
    if not edge_cases:
        logger.error("No edge cases found. Cannot run nudging experiments.")
        return None
    
    # Run nudging experiments
    results = controller.run_nudging_experiments(
        edge_cases=edge_cases,
        nudge_times=nudge_times,
        nudge_amplitudes=nudge_amplitudes,
        nudge_types=nudge_types,
        parallel=args.parallel
    )
    
    # Print summary
    controller.summarize_nudging_results()
    
    logger.info(f"Completed Module 2. Results saved to {output_dir}")
    return controller


def main():
    """Main entry point for RCFT Phase III experiments"""
    parser = argparse.ArgumentParser(
        description='Run RCFT Phase III Experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument('--output_dir', type=str, default='phase3_results',
                        help='Output directory for results')
    parser.add_argument('--modules', type=str, default='1,2',
                        help='Comma-separated list of modules to run (1=Boundary, 2=Nudging, etc.)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    
    # Module 1 options
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
    
    # Module 2 options
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Source directory for false attractor samples')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of edge cases to analyze')
    parser.add_argument('--nudge_times', type=str, default='30,40,50',
                        help='Comma-separated list of nudge time steps')
    parser.add_argument('--nudge_amplitudes', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated list of nudge amplitudes')
    parser.add_argument('--nudge_types', type=str, default='uniform,patch,echo',
                        help='Comma-separated list of nudge types')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Parse modules to run
    modules = [int(m) for m in args.modules.split(',')]
    
    # Print run configuration
    print("\n" + "="*60)
    print("RCFT Phase III Experiment Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Modules to run: {modules}")
    print(f"  Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"  Alpha range: {args.alpha_min} to {args.alpha_max} ({args.alpha_steps} steps)")
    print(f"  Gamma range: {args.gamma_min} to {args.gamma_max} ({args.gamma_steps} steps)")
    print("="*60 + "\n")
    
    # Run selected modules
    boundary_mapper = None
    nudging_controller = None
    
    try:
        # Module 1: Boundary Cartography
        if 1 in modules:
            boundary_mapper = run_phase3_module1(args, logger)
        
        # Module 2: Nudging Experiments
        if 2 in modules:
            nudging_controller = run_phase3_module2(args, logger, boundary_mapper)
        
        # Additional modules will be added here as they are implemented
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        print("\nExperiment interrupted by user.")
    except Exception as e:
        logger.error(f"Error during experiment: {e}", exc_info=True)
        print(f"\nError during experiment: {e}")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
