#!/usr/bin/env python3
"""
Runner script for Phase V of the RCFT experimental framework.
This phase explores recursive identity, self-preference, and temporal coherence.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

# Import RCFT experiment core
try:
    from rcft_framework import RCFTExperiment
except ImportError:
    print("Error: rcft_framework module not found. Please ensure it's in your Python path.")
    sys.exit(1)

# Import Phase V modules
from phase5.temporal_coherence import TemporalCoherenceReinforcer
from phase5.self_distinction import SelfDistinctionAnalyzer
from phase5.identity_biasing import IdentityBiaser
from phase5.temporal_adjacency import TemporalAdjacencyEncoder
from phase5.echo_stability import EchoTrailAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Phase V experiments on recursive identity, self-preference, and temporal coherence.")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="phase5_results",
                      help="Directory to store experiment results")
    
    # Module selection
    parser.add_argument("--modules", type=str, default="1",
                      help="Comma-separated list of modules to run (1-5)")
    
    # Parallel processing
    parser.add_argument("--parallel", action="store_true",
                      help="Run experiments in parallel when possible")
    
    # Combined mode
    parser.add_argument("--combined", action="store_true",
                      help="Run combined experiment across multiple modules")
    
    # Core RCFT parameters
    parser.add_argument("--alpha", type=float, default=0.35,
                      help="Memory strength (α)")
    parser.add_argument("--beta", type=float, default=0.5,
                      help="Coupling strength (β)")
    parser.add_argument("--gamma", type=float, default=0.92,
                      help="Memory decay rate (γ)")
    
    # Pattern parameters
    parser.add_argument("--pattern", type=str, default="fractal",
                      choices=["fractal", "radial", "horizontal", "diagonal", "lattice", "stochastic"],
                      help="Pattern type for initialization")
    
    # General experiment parameters
    parser.add_argument("--max_steps", type=int, default=200,
                      help="Maximum steps per experiment")
    parser.add_argument("--trials", type=int, default=3,
                      help="Number of trials per experiment")
    
    # For Temporal Coherence module
    parser.add_argument("--monitor_window", type=int, default=10,
                      help="Window size for monitoring memory changes")
    
    # For Echo Trail module
    parser.add_argument("--nudge_types", type=str, default="gradient,radial",
                      help="Comma-separated list of nudge types for echo trails")
    
    # For Identity Biasing module
    parser.add_argument("--meta_types", type=str, default="field_bias,noise_profile",
                      help="Comma-separated list of bias meta-types")
    
    # General perturbation parameters
    parser.add_argument("--strengths", type=str, default="0.1,0.2,0.3",
                      help="Comma-separated list of perturbation strengths")
    
    # Fingerprinting parameters
    parser.add_argument("--n_components", type=int, default=7,
                      help="Number of components for fingerprinting")
    
    # Echo parameters
    parser.add_argument("--decay_rate", type=float, default=0.8,
                      help="Decay rate for echo trails")
    parser.add_argument("--reinforcement_rate", type=float, default=0.15,
                      help="Reinforcement rate for echo")
    parser.add_argument("--baseline_strength", type=float, default=0.3,
                      help="Baseline strength for biasing and echo")
    
    # For Temporal Adjacency module
    parser.add_argument("--persistence_mode", type=str, default="memory",
                      choices=["memory", "file", "both"],
                      help="Mode for state persistence")
    parser.add_argument("--pattern_sequence", type=str, default=None,
                      help="Comma-separated list of pattern types for sequence")
    parser.add_argument("--repetitions", type=int, default=1,
                      help="Number of repetitions per pattern")
    
    # For Cross-module interaction
    parser.add_argument("--run_interference", action="store_true",
                      help="Run interference experiments between patterns")
    parser.add_argument("--pattern_ids", type=str, default=None,
                      help="Comma-separated list of pattern identifiers")
    parser.add_argument("--similarity_levels", type=str, default=None,
                      help="Comma-separated list of similarity levels")
    parser.add_argument("--cf_methods", type=str, default=None,
                      help="Comma-separated list of counterfactual methods")
    parser.add_argument("--interference_strength", type=float, default=0.5,
                      help="Strength of interference between patterns")
    
    # Failure mode thresholds
    parser.add_argument("--catastrophic_threshold", type=float, default=0.2,
                      help="Threshold for catastrophic recovery failure")
    parser.add_argument("--graceful_threshold", type=float, default=0.6,
                      help="Threshold for graceful recovery")
    
    return parser.parse_args()

def create_experiment(args):
    """Create an RCFT experiment with specified parameters."""
    experiment = RCFTExperiment(
        size=64,  # Standard size
        memory_strength=args.alpha,
        coupling_strength=args.beta,
        memory_decay=args.gamma
    )
    
    # Initialize with specified pattern
    experiment.initialize_pattern(pattern_type=args.pattern)
    
    return experiment

def create_pattern_sequence(args):
    """Create a sequence of RCFT experiments with different patterns."""
    if args.pattern_sequence:
        pattern_types = args.pattern_sequence.split(',')
    else:
        # Default sequence: different pattern types
        pattern_types = ["fractal", "radial", "diagonal"]
    
    # Limit to 5 patterns to avoid excessive computation
    pattern_types = pattern_types[:5]
    
    # Create experiments
    experiments = []
    for pattern_type in pattern_types:
        exp = RCFTExperiment(
            size=64,
            memory_strength=args.alpha,
            coupling_strength=args.beta,
            memory_decay=args.gamma
        )
        exp.initialize_pattern(pattern_type=pattern_type)
        experiments.append(exp)
    
    # Add repeated patterns if requested
    if args.repetitions > 1:
        for _ in range(args.repetitions - 1):
            for pattern_type in pattern_types:
                exp = RCFTExperiment(
                    size=64,
                    memory_strength=args.alpha,
                    coupling_strength=args.beta,
                    memory_decay=args.gamma
                )
                exp.initialize_pattern(pattern_type=pattern_type)
                experiments.append(exp)
    
    return experiments

def run_temporal_coherence(args):
    """Run the Temporal Coherence Reinforcement module."""
    print("\n========== Running Temporal Coherence Module ==========")
    
    # Parse reactivation schedule type
    # Prefer periodic for this experiment
    reactivation_schedule = "periodic"
    
    # Parse module-specific parameters
    output_dir = os.path.join(args.output_dir, "temporal_coherence")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = create_experiment(args)
    
    # Extract reactivation parameters
    reactivation_strength = args.reinforcement_rate
    reactivation_interval = args.monitor_window
    
    # Create and run module
    module = TemporalCoherenceReinforcer(
        output_dir=output_dir,
        buffer_size=10,
        reactivation_strength=reactivation_strength,
        reactivation_schedule=reactivation_schedule,
        reactivation_interval=reactivation_interval
    )
    
    # Run the experiment
    results = module.run(
        experiment=experiment,
        n_steps=args.max_steps,
        perturb_step=args.max_steps // 4,  # Perturb at 1/4 of the way through
        perturbation_type="flip",
        perturbation_magnitude=float(args.strengths.split(',')[0]),
        run_baseline=True
    )
    
    print(f"Results saved to {output_dir}")
    return results

def run_self_distinction(args):
    """Run the Self-Distinction Index module."""
    print("\n========== Running Self-Distinction Module ==========")
    
    # Parse module-specific parameters
    output_dir = os.path.join(args.output_dir, "self_distinction")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = create_experiment(args)
    
    # Parse perturbation parameters
    perturbation_type = "flip"
    perturbation_magnitude = float(args.strengths.split(',')[0])
    
    # Create and run module
    module = SelfDistinctionAnalyzer(
        output_dir=output_dir,
        fingerprint_dimensions=args.n_components,
        perturbation_type=perturbation_type,
        perturbation_magnitude=perturbation_magnitude,
        n_iterations=args.trials,
        steps_per_iteration=args.max_steps
    )
    
    # Run the experiment
    results = module.run(
        experiment=experiment,
        n_iterations=args.trials,
        steps_per_iteration=args.max_steps
    )
    
    print(f"Results saved to {output_dir}")
    return results

def run_identity_biasing(args):
    """Run the Identity Biasing module."""
    print("\n========== Running Identity Biasing Module ==========")
    
    # Parse module-specific parameters
    output_dir = os.path.join(args.output_dir, "identity_biasing")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = create_experiment(args)
    
    # Parse bias parameters
    bias_strength = args.baseline_strength
    bias_method = "centroid"  # Default to centroid method
    
    # Parse perturbation parameters
    perturbation_type = "flip"
    perturbation_magnitude = float(args.strengths.split(',')[0])
    
    # Create and run module
    module = IdentityBiaser(
        output_dir=output_dir,
        bias_strength=bias_strength,
        bias_method=bias_method,
        perturbation_type=perturbation_type,
        perturbation_magnitude=perturbation_magnitude,
        n_trials=args.trials,
        steps_per_trial=args.max_steps
    )
    
    # Run the experiment
    results = module.run(
        experiment=experiment,
        reference_experiment=None,  # Use the same experiment as reference
        n_trials=args.trials,
        steps_per_trial=args.max_steps,
        run_unbiased=True
    )
    
    print(f"Results saved to {output_dir}")
    return results

def run_temporal_adjacency(args):
    """Run the Temporal Adjacency Encoding module."""
    print("\n========== Running Temporal Adjacency Module ==========")
    
    # Parse module-specific parameters
    output_dir = os.path.join(args.output_dir, "temporal_adjacency")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pattern sequence
    experiments = create_pattern_sequence(args)
    
    # Parse sequence parameters
    sequence_length = min(3, len(experiments))  # Default to 3 or less
    
    # Parse perturbation parameters
    perturbation_type = "flip"
    perturbation_magnitude = float(args.strengths.split(',')[0])
    
    # Create and run module
    module = TemporalAdjacencyEncoder(
        output_dir=output_dir,
        sequence_length=sequence_length,
        perturbation_type=perturbation_type,
        perturbation_magnitude=perturbation_magnitude,
        steps_per_pattern=args.max_steps,
        n_trials=args.trials
    )
    
    # Get pattern names
    if args.pattern_sequence:
        pattern_names = args.pattern_sequence.split(',')[:sequence_length]
    else:
        pattern_names = [f"Pattern_{i+1}" for i in range(sequence_length)]
    
    # Run the experiment
    results = module.run(
        experiments=experiments,
        target_position=1,  # Perturb middle pattern
        pattern_names=pattern_names
    )
    
    print(f"Results saved to {output_dir}")
    return results

def run_echo_stability(args):
    """Run the Echo Trail Stability module."""
    print("\n========== Running Echo Trail Stability Module ==========")
    
    # Parse module-specific parameters
    output_dir = os.path.join(args.output_dir, "echo_stability")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = create_experiment(args)
    
    # Parse echo parameters
    echo_depth = 3  # Use 3 previous trajectories
    echo_strength = args.reinforcement_rate
    echo_decay = args.decay_rate
    
    # Parse perturbation parameters
    perturbation_type = "flip"
    perturbation_magnitude = float(args.strengths.split(',')[0])
    
    # Create and run module
    module = EchoTrailAnalyzer(
        output_dir=output_dir,
        echo_depth=echo_depth,
        echo_strength=echo_strength,
        echo_decay=echo_decay,
        perturbation_type=perturbation_type,
        perturbation_magnitude=perturbation_magnitude,
        steps_per_iteration=args.max_steps,
        n_iterations=args.trials
    )
    
    # Run the experiment
    results = module.run(
        experiment=experiment,
        n_iterations=args.trials,
        steps_per_iteration=args.max_steps,
        run_standard=True
    )
    
    print(f"Results saved to {output_dir}")
    return results

def run_combined_experiment(args):
    """Run a combined experiment across multiple modules."""
    print("\n========== Running Combined Experiment ==========")
    
    # Parse modules to include
    module_ids = [int(m) for m in args.modules.split(',')]
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"combined_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = create_experiment(args)
    
    # Save original state
    original_state = experiment.state.copy()
    
    # Combined results storage
    combined_results = {
        'modules': [],
        'metrics': {},
        'summary': {}
    }
    
    # Run selected modules in sequence
    if 1 in module_ids:
        # Temporal Coherence with reinforcement
        tc_dir = os.path.join(output_dir, "1_temporal_coherence")
        os.makedirs(tc_dir, exist_ok=True)
        
        module = TemporalCoherenceReinforcer(
            output_dir=tc_dir,
            buffer_size=10,
            reactivation_strength=args.reinforcement_rate,
            reactivation_schedule="periodic",
            reactivation_interval=args.monitor_window
        )
        
        # Run with reinforcement
        results = module.run(
            experiment=experiment,
            n_steps=args.max_steps,
            perturb_step=args.max_steps // 4,
            perturbation_type="flip",
            perturbation_magnitude=float(args.strengths.split(',')[0]),
            run_baseline=False  # Skip baseline for combined experiment
        )
        
        # Save results and metrics
        combined_results['modules'].append({
            'id': 1,
            'name': 'Temporal Coherence',
            'output_dir': tc_dir,
            'summary': module.create_summary()
        })
        
        # Extract key metrics
        if 'recovery_quality' in results['reinforced']:
            combined_results['metrics']['tc_recovery_quality'] = results['reinforced']['recovery_quality']
    
    # Reset experiment for next module
    experiment.state = original_state.copy()
    experiment.memory = original_state.copy()
    
    if 2 in module_ids:
        # Self-Distinction Index
        sd_dir = os.path.join(output_dir, "2_self_distinction")
        os.makedirs(sd_dir, exist_ok=True)
        
        module = SelfDistinctionAnalyzer(
            output_dir=sd_dir,
            fingerprint_dimensions=args.n_components,
            perturbation_type="flip",
            perturbation_magnitude=float(args.strengths.split(',')[0]),
            n_iterations=2,  # Fewer iterations for combined experiment
            steps_per_iteration=args.max_steps
        )
        
        # Run self-distinction analysis
        results = module.run(
            experiment=experiment,
            n_iterations=2,
            steps_per_iteration=args.max_steps
        )
        
        # Save results and metrics
        combined_results['modules'].append({
            'id': 2,
            'name': 'Self-Distinction',
            'output_dir': sd_dir,
            'summary': module.create_summary()
        })
        
        # Extract key metrics
        if 'self_distinction' in module.metrics:
            combined_results['metrics']['self_distinction'] = module.metrics['self_distinction']
    
    # Reset experiment for next module
    experiment.state = original_state.copy()
    experiment.memory = original_state.copy()
    
    if 3 in module_ids:
        # Identity Biasing
        ib_dir = os.path.join(output_dir, "3_identity_biasing")
        os.makedirs(ib_dir, exist_ok=True)
        
        module = IdentityBiaser(
            output_dir=ib_dir,
            bias_strength=args.baseline_strength,
            bias_method="centroid",
            perturbation_type="flip",
            perturbation_magnitude=float(args.strengths.split(',')[0]),
            n_trials=2,  # Fewer trials for combined experiment
            steps_per_trial=args.max_steps
        )
        
        # Run with bias
        results = module.run(
            experiment=experiment,
            reference_experiment=None,
            n_trials=2,
            steps_per_trial=args.max_steps,
            run_unbiased=True
        )
        
        # Save results and metrics
        combined_results['modules'].append({
            'id': 3,
            'name': 'Identity Biasing',
            'output_dir': ib_dir,
            'summary': module.create_summary()
        })
        
        # Extract key metrics
        if 'bias_success_rate' in module.metrics:
            combined_results['metrics']['bias_success_rate'] = module.metrics['bias_success_rate']
    
    # For modules 4 and 5, we need additional experiments
    if 4 in module_ids:
        # Temporal Adjacency Encoding
        ta_dir = os.path.join(output_dir, "4_temporal_adjacency")
        os.makedirs(ta_dir, exist_ok=True)
        
        # Create pattern sequence
        experiments = create_pattern_sequence(args)
        sequence_length = min(3, len(experiments))
        
        module = TemporalAdjacencyEncoder(
            output_dir=ta_dir,
            sequence_length=sequence_length,
            perturbation_type="flip",
            perturbation_magnitude=float(args.strengths.split(',')[0]),
            steps_per_pattern=args.max_steps,
            n_trials=2  # Fewer trials for combined experiment
        )
        
        # Run with pattern sequence
        if args.pattern_sequence:
            pattern_names = args.pattern_sequence.split(',')[:sequence_length]
        else:
            pattern_names = [f"Pattern_{i+1}" for i in range(sequence_length)]
            
        results = module.run(
            experiments=experiments,
            target_position=1,
            pattern_names=pattern_names
        )
        
        # Save results and metrics
        combined_results['modules'].append({
            'id': 4,
            'name': 'Temporal Adjacency',
            'output_dir': ta_dir,
            'summary': module.create_summary()
        })
        
        # Extract key metrics
        if 'connectivity' in module.metrics:
            combined_results['metrics']['adjacency_connectivity'] = module.metrics['connectivity']
        if 'overall_recovery' in module.metrics:
            combined_results['metrics']['adjacency_recovery'] = module.metrics['overall_recovery']
    
    # Reset experiment for next module
    experiment.state = original_state.copy()
    experiment.memory = original_state.copy()
    
    if 5 in module_ids:
        # Echo Trail Stability
        es_dir = os.path.join(output_dir, "5_echo_stability")
        os.makedirs(es_dir, exist_ok=True)
        
        module = EchoTrailAnalyzer(
            output_dir=es_dir,
            echo_depth=3,
            echo_strength=args.reinforcement_rate,
            echo_decay=args.decay_rate,
            perturbation_type="flip",
            perturbation_magnitude=float(args.strengths.split(',')[0]),
            steps_per_iteration=args.max_steps,
            n_iterations=3  # Fewer iterations for combined experiment
        )
        
        # Run with echo
        results = module.run(
            experiment=experiment,
            n_iterations=3,
            steps_per_iteration=args.max_steps,
            run_standard=True
        )
        
        # Save results and metrics
        combined_results['modules'].append({
            'id': 5,
            'name': 'Echo Stability',
            'output_dir': es_dir,
            'summary': module.create_summary()
        })
        
        # Extract key metrics
        if 'echo_correction_delta' in module.metrics:
            combined_results['metrics']['echo_correction_delta'] = module.metrics['echo_correction_delta']
        if 'recovery_improvement' in module.metrics:
            combined_results['metrics']['echo_improvement'] = module.metrics['recovery_improvement']
    
    # Create combined summary
    combined_results['summary'] = {
        'timestamp': timestamp,
        'parameters': vars(args),
        'modules_run': module_ids
    }
    
    # Save combined results
    with open(os.path.join(output_dir, "combined_results.json"), 'w') as f:
        json.dump(combined_results, f, indent=2, default=lambda x: str(x))
    
    # Create combined visualization
    create_combined_visualization(combined_results, output_dir)
    
    print(f"Combined results saved to {output_dir}")
    return combined_results

def create_combined_visualization(combined_results, output_dir):
    """Create a visualization summarizing the combined experiment."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(15, 10))
        
        # Track which modules were run
        modules_run = [m['id'] for m in combined_results['modules']]
        
        # Organize metrics by module
        module_metrics = {
            1: [],  # Temporal Coherence
            2: [],  # Self-Distinction
            3: [],  # Identity Biasing
            4: [],  # Temporal Adjacency
            5: []   # Echo Stability
        }
        
        # Collect metrics
        for key, value in combined_results['metrics'].items():
            if key.startswith('tc_'):
                module_metrics[1].append((key, value))
            elif key.startswith('self_'):
                module_metrics[2].append((key, value))
            elif key.startswith('bias_'):
                module_metrics[3].append((key, value))
            elif key.startswith('adjacency_'):
                module_metrics[4].append((key, value))
            elif key.startswith('echo_'):
                module_metrics[5].append((key, value))
        
        # Create subplots for each module that was run
        subplot_positions = {
            1: (0, 0),  # Top left
            2: (0, 1),  # Top right
            3: (1, 0),  # Middle left
            4: (1, 1),  # Middle right
            5: (2, 0)   # Bottom left
        }
        
        for module_id in modules_run:
            # Skip if no metrics for this module
            if not module_metrics[module_id]:
                continue
                
            # Create subplot
            i, j = subplot_positions[module_id]
            plt.subplot(3, 2, i*2 + j + 1)
            
            # Extract metrics for this module
            metrics = module_metrics[module_id]
            names = [m[0].split('_', 1)[1] for m in metrics]
            values = [m[1] for m in metrics]
            
            # Create bar chart
            bars = plt.bar(names, values)
            
            # Color bars based on sign
            for bar, value in zip(bars, values):
                bar.set_color('green' if value >= 0 else 'red')
                
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f"{height:.3f}", ha='center', va='bottom')
                else:
                    plt.text(bar.get_x() + bar.get_width()/2, height - 0.01,
                           f"{height:.3f}", ha='center', va='top')
            
            # Add module title
            module_names = {
                1: "Temporal Coherence",
                2: "Self-Distinction",
                3: "Identity Biasing",
                4: "Temporal Adjacency",
                5: "Echo Stability"
            }
            plt.title(module_names[module_id])
            plt.ylabel("Value")
            plt.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=30, ha='right')
        
        # Add summary title
        plt.suptitle("Combined Experiment Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "combined_summary.png"))
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create combined visualization: {e}")

def main():
    """Main function to run Phase V experiments."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse modules to run
    module_ids = [int(m) for m in args.modules.split(',')]
    
    # Validate module IDs
    valid_modules = list(range(1, 6))
    for module_id in module_ids:
        if module_id not in valid_modules:
            print(f"Error: Invalid module ID {module_id}. Valid modules are 1-5.")
            sys.exit(1)
    
    # If combined mode is enabled, run combined experiment
    if args.combined:
        run_combined_experiment(args)
        return
    
    # Otherwise, run selected modules separately
    for module_id in module_ids:
        if module_id == 1:
            run_temporal_coherence(args)
        elif module_id == 2:
            run_self_distinction(args)
        elif module_id == 3:
            run_identity_biasing(args)
        elif module_id == 4:
            run_temporal_adjacency(args)
        elif module_id == 5:
            run_echo_stability(args)

if __name__ == "__main__":
    main()