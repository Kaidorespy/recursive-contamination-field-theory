# example_rcft_experiments.py
import numpy as np
import matplotlib.pyplot as plt
from rcft_framework import RCFTExperiment, ExperimentRunner  # Assuming previous code is saved as rcft_framework.py

def run_single_experiment():
    """Run a single experiment to visualize memory resurgence"""
    print("Running single experiment with radial pattern and flip perturbation...")
    
    # Create experiment with default parameters
    exp = RCFTExperiment(
        memory_strength=0.35,    # α (alpha): Memory influence strength
        coupling_strength=0.5,    # β (beta): Neighbor coupling strength
        memory_decay=0.92        # γ (gamma): Memory decay rate
    )
    
    # Initialize with radial pattern
    exp.initialize_pattern(pattern_type="radial", frequency=2)
    
    # Apply a flip perturbation with magnitude 1.0
    exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
    
    # Let the system evolve for 50 steps
    exp.update(steps=50)
    
    # Calculate and print recovery metrics
    recovery = exp.calculate_recovery_metrics()
    print("\nRecovery Metrics:")
    for key, value in recovery.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize results
    exp.visualize_key_frames()
    exp.visualize_metrics()
    
    # Save experiment data
    exp_dir = exp.save_experiment_data("experiment_results")
    print(f"\nExperiment saved to: {exp_dir}")
    
    return exp

def run_pattern_comparison():
    """Compare recovery across different patterns"""
    print("\nComparing recovery across different pattern types...")
    
    runner = ExperimentRunner(output_dir="pattern_comparison")
    
    # Define patterns and perturbation magnitudes to test
    patterns = ["radial", "horizontal", "diagonal", "lattice", "stochastic"]
    magnitudes = [0.5, 1.0, 1.5]
    
    # Run the experiment set
    results = runner.run_pattern_specific_recovery(
        pattern_types=patterns,
        perturbation_magnitudes=magnitudes,
        perturbation_type="flip",
        recovery_steps=50
    )
    
    print("\nPattern comparison complete. Results saved to pattern_comparison/ directory.")
    return results

def run_parameter_exploration():
    """Explore the effect of different parameters on recovery"""
    print("\nExploring parameter space for memory resilience...")
    
    runner = ExperimentRunner(output_dir="parameter_space")
    
    # Define parameter ranges to explore
    alpha_range = np.linspace(0.1, 0.6, 6)  # Memory strength
    gamma_range = np.linspace(0.7, 0.99, 6)  # Memory decay
    
    # Run parameter sweep
    results = runner.run_parameter_sweep(
        alpha_range=alpha_range,
        gamma_range=gamma_range,
        beta=0.5,                     # Fixed coupling strength
        pattern_type="radial",
        perturbation_type="flip",
        perturbation_magnitude=1.0,
        recovery_steps=50
    )
    
    print("\nParameter exploration complete. Results saved to parameter_space/ directory.")
    return results

def run_perturbation_comparison():
    """Compare different perturbation types"""
    print("\nComparing different perturbation types...")
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir="perturbation_types")
    
    # Define patterns and perturbation types
    perturbation_types = ["flip", "noise", "zero", "memory_wipe", "displacement"]
    magnitudes = [1.0]  # Fixed magnitude for comparison
    
    # Run separate experiments for each perturbation type
    for pert_type in perturbation_types:
        print(f"Testing {pert_type} perturbation...")
        
        # Initialize experiment
        exp = RCFTExperiment(
            memory_strength=0.35,
            coupling_strength=0.5,
            memory_decay=0.92
        )
        
        # Setup and run
        exp.initialize_pattern(pattern_type="radial")
        exp.apply_perturbation(perturbation_type=pert_type, magnitude=1.0)
        exp.update(steps=50)
        
        # Save experiment
        exp_dir = exp.save_experiment_data(f"perturbation_types/{pert_type}")
        
        # Print recovery quality
        recovery = exp.calculate_recovery_metrics()
        print(f"  Recovery quality: {recovery['recovery_quality']:.4f}")
    
    print("\nPerturbation comparison complete. Results saved to perturbation_types/ directory.")

def temporal_layered_experiment():
    """Run a temporal layered perturbation experiment (Phase I, Experiment Set B)"""
    print("\nRunning temporal layered perturbation experiment...")
    
    # Create experiment
    exp = RCFTExperiment(
        memory_strength=0.35,
        coupling_strength=0.5,
        memory_decay=0.92
    )
    
    # Initialize pattern
    exp.initialize_pattern(pattern_type="radial")
    
    # Apply first perturbation
    exp.apply_perturbation(perturbation_type="flip", magnitude=0.8, radius=15)
    
    # Let it recover partially
    exp.update(steps=20)
    
    # Apply second perturbation in a different location
    exp.apply_perturbation(perturbation_type="flip", magnitude=0.8, 
                         center=(20, 20), radius=10)
    
    # Let it recover fully
    exp.update(steps=50)
    
    # Visualize and save
    exp.visualize_key_frames()
    exp.visualize_metrics()
    exp_dir = exp.save_experiment_data("temporal_layered")
    
    print(f"\nTemporal layered experiment saved to: {exp_dir}")
    return exp

if __name__ == "__main__":
    print("RCFT Experiment Suite Examples")
    print("==============================\n")
    
    # Choose which experiments to run (uncomment as needed)
    exp = run_single_experiment()
    # results = run_pattern_comparison()
    # results = run_parameter_exploration()
    # run_perturbation_comparison()
    # layered_exp = temporal_layered_experiment()
    
    print("\nAll experiments completed!")