#!/usr/bin/env python3
"""
Runner script for RCFT Phase IV: Directed Memory Manipulation and Adaptive Dynamics
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import Phase IV modules
from phase4.adaptive_nudge import AdaptiveNudgeController
from phase4.attractor_sculptor import AttractorSculptor
from phase4.learning_field import LearningFieldSimulator
from phase4.counterfactual_injector import CounterfactualInjector

def setup_logging(log_dir="phase4_logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"phase4_run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("Phase4Runner")
    return logger, timestamp

def run_module1_adaptive_nudge(args, logger, timestamp):
    """Run Module 1: Adaptive Nudging Feedback Loop"""
    logger.info("Starting Phase IV Module 1: Adaptive Nudging")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "adaptive_nudge")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize controller
    controller = AdaptiveNudgeController(
        output_dir=output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        pattern_type=args.pattern,
        monitor_window=args.monitor_window,
        max_steps=args.max_steps,
        nudge_types=args.nudge_types.split(',') if args.nudge_types else None
    )
    
    # Run experiments
    results = controller.run(
        num_trials=args.trials,
        initial_perturbation=True,
        parallel=args.parallel
    )
    
    logger.info(f"Completed Module 1. Results saved to {output_dir}")
    return controller, results

def run_module2_attractor_sculptor(args, logger, timestamp):
    """Run Module 2: Attractor Sculpting via Meta-Perturbation"""
    logger.info("Starting Phase IV Module 2: Attractor Sculptor")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "attractor_sculptor")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sculptor
    sculptor = AttractorSculptor(
        output_dir=output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        pattern_type=args.pattern,
        max_steps=args.max_steps,
        n_fingerprint_components=args.n_components
    )
    
    # Parse meta-perturbation types
    meta_types = args.meta_types.split(',') if args.meta_types else ['field_bias', 'boundary_constraint', 'noise_profile']
    
    # Run experiments
    strengths = [float(s) for s in args.strengths.split(',')] if args.strengths else [0.05, 0.1, 0.2, 0.3]
    
    results = sculptor.run(
        meta_types=meta_types,
        strengths=strengths,
        trials_per_config=args.trials,
        parallel=args.parallel
    )
    
    logger.info(f"Completed Module 2. Results saved to {output_dir}")
    return sculptor, results

def run_module3_learning_field(args, logger, timestamp):
    """Run Module 3: Emergent Learning in RCFT Fields"""
    logger.info("Starting Phase IV Module 3: Learning Field")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "learning_field")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize learning field
    learner = LearningFieldSimulator(
        output_dir=output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        pattern_type=args.pattern,
        max_steps=args.max_steps,
        memory_decay_rate=args.decay_rate,
        memory_reinforcement_rate=args.reinforcement_rate,
        baseline_strength=args.baseline_strength,
        persistence_mode=args.persistence_mode
    )
    
    # Create patterns
    pattern_sequence = args.pattern_sequence.split(',') if args.pattern_sequence else ['A', 'B', 'C']
    
    for pattern_id in pattern_sequence:
        if pattern_id not in learner.patterns:
            learner.create_pattern(pattern_id)
    
    # Run learning experiment
    if args.run_interference:
        # Run interference experiment
        results = learner.run_interference_experiment(
            target_pattern_id=pattern_sequence[0],
            interfering_patterns=pattern_sequence[1:],
            num_repetitions=args.repetitions
        )
    else:
        # Run normal learning experiment
        results = learner.run(
            pattern_sequence=pattern_sequence,
            num_repetitions=args.repetitions,
            initialize_with_trace=True,
            apply_perturbation=True,
            persistence_between_reps=True
        )
    
    logger.info(f"Completed Module 3. Results saved to {output_dir}")
    return learner, results

def run_module4_counterfactual_injector(args, logger, timestamp):
    """Run Module 4: Counterfactual Memory Injection"""
    logger.info("Starting Phase IV Module 4: Counterfactual Injector")
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, "counterfactual_injector")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize injector
    injector = CounterfactualInjector(
        output_dir=output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        pattern_type=args.pattern,
        max_steps=args.max_steps,
        catastrophic_threshold=args.catastrophic_threshold,
        graceful_threshold=args.graceful_threshold
    )
    
    # Create patterns
    pattern_ids = args.pattern_ids.split(',') if args.pattern_ids else ['A']
    
    for pid in pattern_ids:
        injector.create_true_pattern(pid)
    
    # Parse similarity levels and methods
    similarity_levels = [float(s) for s in args.similarity_levels.split(',')] if args.similarity_levels else [0.2, 0.4, 0.6, 0.8]
    methods = args.cf_methods.split(',') if args.cf_methods else ['blend', 'noise', 'transform', 'structure']
    
    # Run experiments
    results = injector.run(
        pattern_ids=pattern_ids,
        similarity_levels=similarity_levels,
        methods=methods,
        interference_strength=args.interference_strength,
        trials_per_level=args.trials,
        trials_per_method=args.trials,
        parallel=args.parallel
    )
    
    logger.info(f"Completed Module 4. Results saved to {output_dir}")
    return injector, results

def run_combined_experiment(args, logger, timestamp):
    """Run a combined experiment using multiple modules"""
    logger.info("Starting Combined Phase IV Experiment")
    
    # First, run learning field to establish memory traces
    learner, learning_results = run_module3_learning_field(args, logger, timestamp)
    
    # Then, apply counterfactual injection to test robustness
    args.pattern_ids = args.pattern_sequence  # Use same patterns
    injector, injection_results = run_module4_counterfactual_injector(args, logger, timestamp)
    
    # Finally, test if adaptive nudging can recover from interference
    controller, nudge_results = run_module1_adaptive_nudge(args, logger, timestamp)
    
    # Create combined analysis
    combined_dir = os.path.join(args.output_dir, "combined_analysis")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Here you would add code to create visualizations that combine results
    # from all three modules to show the full experimental flow
    
    logger.info(f"Completed Combined Experiment. Results saved to {args.output_dir}")
    return {
        "learning": learning_results,
        "injection": injection_results,
        "nudging": nudge_results
    }

def main():
    """Main entry point for RCFT Phase IV experiments"""
    parser = argparse.ArgumentParser(
        description='Run RCFT Phase IV Experiments: Directed Memory Manipulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument('--output_dir', type=str, default='phase4_results',
                        help='Output directory for results')
    parser.add_argument('--modules', type=str, default='1',
                        help='Comma-separated list of modules to run (1=Nudging, 2=Sculptor, 3=Learning, 4=Counterfactual)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    parser.add_argument('--combined', action='store_true',
                        help='Run combined experiment using multiple modules')
    
    # Common parameters
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='Memory strength parameter')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Coupling strength parameter')
    parser.add_argument('--gamma', type=float, default=0.92,
                        help='Memory decay parameter')
    parser.add_argument('--pattern', type=str, default='fractal',
                        choices=['fractal', 'radial', 'horizontal', 'diagonal', 'lattice', 'stochastic'],
                        help='Pattern type to initialize')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum simulation steps')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials per configuration')
    
    # Module 1: Adaptive Nudging parameters
    parser.add_argument('--monitor_window', type=int, default=10,
                        help='Window size for monitoring recovery')
    parser.add_argument('--nudge_types', type=str, default='amplitude_echo,spatial_bias,symmetry_pulse',
                        help='Comma-separated list of nudge types')
    
    # Module 2: Attractor Sculptor parameters
    parser.add_argument('--meta_types', type=str, default='field_bias,boundary_constraint,noise_profile',
                        help='Comma-separated list of meta-perturbation types')
    parser.add_argument('--strengths', type=str, default='0.05,0.1,0.2,0.3',
                        help='Comma-separated list of perturbation strengths')
    parser.add_argument('--n_components', type=int, default=5,
                        help='Number of components for fingerprinting')
    
    # Module 3: Learning Field parameters
    parser.add_argument('--decay_rate', type=float, default=0.8,
                        help='Memory trace decay rate')
    parser.add_argument('--reinforcement_rate', type=float, default=0.2,
                        help='Memory trace reinforcement rate')
    parser.add_argument('--baseline_strength', type=float, default=0.1,
                        help='Baseline memory strength')
    parser.add_argument('--persistence_mode', type=str, default='both',
                        choices=['memory', 'file', 'both'],
                        help='Memory persistence mode')
    parser.add_argument('--pattern_sequence', type=str, default='A,B,C',
                        help='Comma-separated list of pattern IDs')
    parser.add_argument('--repetitions', type=int, default=5,
                        help='Number of repetitions for learning')
    parser.add_argument('--run_interference', action='store_true',
                        help='Run interference experiment instead of regular learning')
    
    # Module 4: Counterfactual Injector parameters
    parser.add_argument('--pattern_ids', type=str, default='A',
                        help='Comma-separated list of pattern IDs for counterfactuals')
    parser.add_argument('--similarity_levels', type=str, default='0.2,0.4,0.6,0.8',
                        help='Comma-separated list of similarity levels')
    parser.add_argument('--cf_methods', type=str, default='blend',
                        help='Comma-separated list of counterfactual generation methods')
    parser.add_argument('--interference_strength', type=float, default=0.5,
                        help='Strength of memory interference')
    parser.add_argument('--catastrophic_threshold', type=float, default=0.3,
                        help='Threshold for catastrophic forgetting')
    parser.add_argument('--graceful_threshold', type=float, default=0.7,
                        help='Threshold for graceful degradation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger, timestamp = setup_logging()
    
    # Print run configuration
    print("\n" + "="*60)
    print("RCFT Phase IV Experiment Suite")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Modules to run: {args.modules}")
    print(f"  Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"  Pattern type: {args.pattern}")
    print(f"  Parameters: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print("="*60 + "\n")
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run combined experiment
        if args.combined:
            results = run_combined_experiment(args, logger, timestamp)
        else:
            # Parse modules to run
            modules = [int(m) for m in args.modules.split(',')]
            results = {}
            
            # Run selected modules
            if 1 in modules:
                controller, module1_results = run_module1_adaptive_nudge(args, logger, timestamp)
                results["module1"] = module1_results
                
            if 2 in modules:
                sculptor, module2_results = run_module2_attractor_sculptor(args, logger, timestamp)
                results["module2"] = module2_results
                
            if 3 in modules:
                learner, module3_results = run_module3_learning_field(args, logger, timestamp)
                results["module3"] = module3_results
                
            if 4 in modules:
                injector, module4_results = run_module4_counterfactual_injector(args, logger, timestamp)
                results["module4"] = module4_results
        
        print(f"\nAll experiments completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
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