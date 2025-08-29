# This will be the enhanced Phase I implementation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from tqdm import tqdm
from datetime import datetime
from rcft_framework import RCFTExperiment

class EnhancedPhaseI:
    """Enhanced Phase I implementation with comprehensive pattern testing and anomaly detection"""
    
    def __init__(self, output_dir="phase1_results"):
        """Initialize the Phase I experiment runner"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Master results dataframe
        self.results = pd.DataFrame()
        
        # Define patterns and perturbations
        self.patterns = ["radial", "horizontal", "diagonal", "fractal", "lattice"]
        self.perturbation_types = ["flip", "noise", "zero", "memory_wipe", "displacement"]
        self.perturbation_magnitudes = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "high_quality_low_entropy": {"correlation": 0.9, "entropy_delta": 0.05},
            "slow_high_quality": {"correlation": 0.9, "recovery_time_ratio": 0.8},
            "nonmonotonic_threshold": 0.6
        }
        
    def run_pattern_perturbation_matrix(self):
        """Run the 5×5 pattern × perturbation sweep"""
        print("Running Pattern × Perturbation Matrix (5×5 = 25 runs)")
        
        # Create subdirectory for these experiments
        matrix_dir = os.path.join(self.output_dir, "pattern_perturbation_matrix")
        if not os.path.exists(matrix_dir):
            os.makedirs(matrix_dir)
        
        # Master results list
        all_results = []
        
        # Loop through all combinations
        for pattern in tqdm(self.patterns, desc="Patterns"):
            for pert_type in self.perturbation_types:
                # Use middle magnitude for perturbation type comparison
                magnitude = 1.0
                
                # Create and run experiment
                exp = RCFTExperiment(memory_strength=0.35, coupling_strength=0.5, memory_decay=0.92)
                exp.initialize_pattern(pattern_type=pattern)
                exp.apply_perturbation(perturbation_type=pert_type, magnitude=magnitude)
                exp.update(steps=50)
                
                # Calculate recovery metrics
                recovery = exp.calculate_recovery_metrics()
                
                # Add pattern and perturbation info
                recovery['pattern'] = pattern
                recovery['perturbation_type'] = pert_type
                recovery['perturbation_magnitude'] = magnitude
                
                # Add recovery curve data (full correlation time series)
                recovery['correlation_curve'] = exp.metrics['correlation']
                
                # Add anomaly detection
                anomalies = self.detect_anomalies(exp)
                recovery.update(anomalies)
                
                # Save detailed experiment data
                exp_dir = os.path.join(matrix_dir, f"{pattern}_{pert_type}")
                if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)
                
                # Save visualizations with detailed filenames
                exp.visualize_key_frames(save_path=os.path.join(exp_dir, f"{pattern}_{pert_type}_frames.png"))
                exp.visualize_metrics(save_path=os.path.join(exp_dir, f"{pattern}_{pert_type}_metrics.png"))
                
                # Save recovery curve specifically
                self.save_recovery_curve(exp, os.path.join(exp_dir, f"{pattern}_{pert_type}_recovery.png"))
                
                # Add to results
                all_results.append(recovery)
                
        # Create master dataframe
        results_df = pd.DataFrame(all_results)
        
        # Save master results
        results_df.to_csv(os.path.join(matrix_dir, "pattern_perturbation_results.csv"), index=False)
        
        # Save recovery curves separately (they're lists, not scalar values)
        curve_data = {f"{row.pattern}_{row.perturbation_type}": row.correlation_curve 
                     for _, row in results_df.iterrows()}
        with open(os.path.join(matrix_dir, "recovery_curves.json"), 'w') as f:
            json.dump(curve_data, f)
        
        # Remove correlation_curve from dataframe (it's a list, not scalar)
        results_df = results_df.drop('correlation_curve', axis=1)
        
        # Create summary visualizations
        self.create_pattern_perturbation_summary(results_df, matrix_dir)
        
        return results_df
    
    def run_perturbation_magnitude_sweep(self):
        """Run perturbation magnitude sweeps for each pattern"""
        print("Running Perturbation Magnitude Sweeps for each pattern")
        
        # Create subdirectory
        magnitude_dir = os.path.join(self.output_dir, "perturbation_magnitude_sweep")
        if not os.path.exists(magnitude_dir):
            os.makedirs(magnitude_dir)
        
        # Master results list
        all_results = []
        
        # Run for each pattern x perturbation magnitude 
        for pattern in tqdm(self.patterns, desc="Patterns"):
            for pert_type in ["flip"]:  # Use flip as standard perturbation
                for magnitude in self.perturbation_magnitudes:
                    # Create and run experiment
                    exp = RCFTExperiment(memory_strength=0.35, coupling_strength=0.5, memory_decay=0.92)
                    exp.initialize_pattern(pattern_type=pattern)
                    exp.apply_perturbation(perturbation_type=pert_type, magnitude=magnitude)
                    exp.update(steps=50)
                    
                    # Calculate recovery metrics
                    recovery = exp.calculate_recovery_metrics()
                    
                    # Add experiment info
                    recovery['pattern'] = pattern
                    recovery['perturbation_type'] = pert_type
                    recovery['perturbation_magnitude'] = magnitude
                    
                    # Add recovery curve data
                    recovery['correlation_curve'] = exp.metrics['correlation']
                    
                    # Add anomaly detection
                    anomalies = self.detect_anomalies(exp)
                    recovery.update(anomalies)
                    
                    # Save experiment
                    exp_dir = os.path.join(magnitude_dir, f"{pattern}_mag{magnitude}")
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    
                    # Save visualizations
                    exp.visualize_key_frames(save_path=os.path.join(exp_dir, f"{pattern}_mag{magnitude}_frames.png"))
                    exp.visualize_metrics(save_path=os.path.join(exp_dir, f"{pattern}_mag{magnitude}_metrics.png"))
                    self.save_recovery_curve(exp, os.path.join(exp_dir, f"{pattern}_mag{magnitude}_recovery.png"))
                    
                    # Add to results
                    all_results.append(recovery)
        
        # Create master dataframe
        results_df = pd.DataFrame(all_results)
        
        # Save master results
        results_df.to_csv(os.path.join(magnitude_dir, "magnitude_sweep_results.csv"), index=False)
        
        # Save recovery curves separately
        curve_data = {f"{row.pattern}_mag{row.perturbation_magnitude}": row.correlation_curve 
                     for _, row in results_df.iterrows()}
        with open(os.path.join(magnitude_dir, "magnitude_sweep_curves.json"), 'w') as f:
            json.dump(curve_data, f)
        
        # Remove correlation_curve from dataframe
        results_df = results_df.drop('correlation_curve', axis=1)
        
        # Create summary visualizations
        self.create_magnitude_sweep_summary(results_df, magnitude_dir)
        
        return results_df
    
    def run_layered_perturbation_experiments(self):
        """Run layered perturbation experiments with varying t_delay"""
        print("Running Layered Perturbation Experiments")
        
        # Create subdirectory
        layered_dir = os.path.join(self.output_dir, "layered_perturbations")
        if not os.path.exists(layered_dir):
            os.makedirs(layered_dir)
        
        # Define delay times to test
        t_delays = [5, 10, 20, 50]
        
        # Master results list
        all_results = []
        
        # Run for each pattern and delay time
        for pattern in tqdm(self.patterns, desc="Patterns"):
            for t_delay in t_delays:
                # Create experiment
                exp = RCFTExperiment(memory_strength=0.35, coupling_strength=0.5, memory_decay=0.92)
                exp.initialize_pattern(pattern_type=pattern)
                
                # Apply first perturbation
                exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
                
                # Let system recover for t_delay steps
                exp.update(steps=t_delay)
                
                # Record state after first recovery
                first_recovery = {
                    'correlation': exp.metrics['correlation'][-1],
                    'mutual_info': exp.metrics['mutual_info'][-1],
                    'spectral_entropy': exp.metrics['spectral_entropy'][-1]
                }
                
                # Apply second perturbation in a different location
                exp.apply_perturbation(perturbation_type="flip", magnitude=1.0, 
                                     center=(20, 20), radius=10)
                
                # Let system recover fully
                exp.update(steps=50)
                
                # Calculate final recovery metrics
                recovery = exp.calculate_recovery_metrics()
                
                # Add experiment info
                recovery['pattern'] = pattern
                recovery['t_delay'] = t_delay
                recovery['first_recovery_correlation'] = first_recovery['correlation']
                recovery['first_recovery_entropy'] = first_recovery['spectral_entropy']
                
                # Add recovery curve data (full correlation time series)
                recovery['correlation_curve'] = exp.metrics['correlation']
                
                # Add anomaly detection
                anomalies = self.detect_anomalies(exp)
                recovery.update(anomalies)
                
                # Save experiment
                exp_dir = os.path.join(layered_dir, f"{pattern}_delay{t_delay}")
                if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)
                
                # Save visualizations
                exp.visualize_key_frames(save_path=os.path.join(exp_dir, f"{pattern}_delay{t_delay}_frames.png"))
                exp.visualize_metrics(save_path=os.path.join(exp_dir, f"{pattern}_delay{t_delay}_metrics.png"))
                
                # Save with perturbation markers
                self.save_layered_recovery_curve(exp, os.path.join(exp_dir, f"{pattern}_delay{t_delay}_recovery.png"), 
                                              [1, t_delay+1])  # Mark both perturbations
                
                # Add to results
                all_results.append(recovery)
        
        # Create master dataframe
        results_df = pd.DataFrame(all_results)
        
        # Save master results
        results_df.to_csv(os.path.join(layered_dir, "layered_perturbation_results.csv"), index=False)
        
        # Save recovery curves separately
        curve_data = {f"{row.pattern}_delay{row.t_delay}": row.correlation_curve 
                     for _, row in results_df.iterrows()}
        with open(os.path.join(layered_dir, "layered_recovery_curves.json"), 'w') as f:
            json.dump(curve_data, f)
        
        # Remove correlation_curve from dataframe
        results_df = results_df.drop('correlation_curve', axis=1)
        
        # Create summary visualizations
        self.create_layered_perturbation_summary(results_df, layered_dir)
        
        return results_df
    
    def detect_anomalies(self, experiment):
        """Detect anomalies in recovery curves based on defined criteria"""
        # Get key metrics
        correlation_curve = experiment.metrics['correlation']
        recovery = experiment.calculate_recovery_metrics()
        entropy_delta = experiment.metrics['spectral_entropy'][-1] - experiment.metrics['spectral_entropy'][0]
        
        # Initialize anomaly flags
        anomalies = {
            'high_quality_low_entropy': False,
            'slow_high_quality': False,
            'nonmonotonic_recovery': False
        }
        
        # Check for high quality but low entropy change
        if (recovery['final_correlation'] > self.anomaly_thresholds['high_quality_low_entropy']['correlation'] and 
            abs(entropy_delta) < self.anomaly_thresholds['high_quality_low_entropy']['entropy_delta']):
            anomalies['high_quality_low_entropy'] = True
        
        # Check for slow but high quality recovery
        total_steps = len(correlation_curve) - experiment.perturbation_step
        if (recovery['final_correlation'] > self.anomaly_thresholds['slow_high_quality']['correlation'] and 
            recovery['recovery_time'] > self.anomaly_thresholds['slow_high_quality']['recovery_time_ratio'] * total_steps):
            anomalies['slow_high_quality'] = True
        
        # Check for non-monotonic recovery (dips below threshold after initial recovery)
        if experiment.perturbation_step >= 0:
            post_perturb = correlation_curve[experiment.perturbation_step:]
            
            # Find max correlation after perturbation
            max_idx = np.argmax(post_perturb)
            
            # Check if there's any point after max that dips below threshold
            if max_idx < len(post_perturb) - 1:  # If max isn't the last point
                min_after_max = np.min(post_perturb[max_idx:])
                if min_after_max < self.anomaly_thresholds['nonmonotonic_threshold']:
                    anomalies['nonmonotonic_recovery'] = True
        
        return anomalies
    
    def save_recovery_curve(self, experiment, save_path):
        """Save a detailed recovery curve plot"""
        plt.figure(figsize=(10, 6))
        
        # Plot correlation
        plt.plot(experiment.metrics['correlation'], label='Correlation with Initial State')
        
        # Mark perturbation
        plt.axvline(x=experiment.perturbation_step, color='r', linestyle='--', label='Perturbation')
        
        # Add title and labels
        plt.title(f"Recovery Curve: {experiment.pattern_type} with {experiment.perturbation_type} ({experiment.perturbation_magnitude:.1f})")
        plt.xlabel("Time Step")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.legend()
        
        # Save
        plt.savefig(save_path)
        plt.close()
    
    def save_layered_recovery_curve(self, experiment, save_path, perturbation_steps):
        """Save a recovery curve with multiple perturbation markers"""
        plt.figure(figsize=(10, 6))
        
        # Plot correlation
        plt.plot(experiment.metrics['correlation'], label='Correlation with Initial State')
        
        # Mark perturbations
        for i, step in enumerate(perturbation_steps):
            plt.axvline(x=step, color='r', linestyle='--', label=f'Perturbation {i+1}' if i == 0 else None)
        
        # Add title and labels
        plt.title(f"Layered Perturbation Recovery: {experiment.pattern_type}")
        plt.xlabel("Time Step")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.legend()
        
        # Save
        plt.savefig(save_path)
        plt.close()
    
    def create_pattern_perturbation_summary(self, results_df, output_dir):
        """Create summary visualizations for pattern × perturbation matrix"""
        # Create heatmap of recovery quality
        plt.figure(figsize=(12, 8))
        
        # Pivot the data for heatmap
        heatmap_data = results_df.pivot(index="pattern", columns="perturbation_type", values="recovery_quality")
        
        # Create heatmap
        plt.imshow(heatmap_data, cmap='viridis')
        plt.colorbar(label='Recovery Quality')
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                plt.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}', 
                        ha="center", va="center", color="white")
        
        # Add labels
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        
        plt.title('Recovery Quality by Pattern and Perturbation Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recovery_quality_heatmap.png"))
        plt.close()
        
        # Create bar chart of recovery time
        plt.figure(figsize=(12, 8))
        
        # Pivot the data for bar chart
        bar_data = results_df.pivot(index="pattern", columns="perturbation_type", values="recovery_time")
        
        # Create bar chart
        bar_data.plot(kind='bar')
        
        plt.title('Recovery Time by Pattern and Perturbation Type')
        plt.ylabel('Recovery Time (steps)')
        plt.legend(title='Perturbation Type')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recovery_time_barchart.png"))
        plt.close()
        
        # Create anomaly summary
        anomaly_counts = {
            'high_quality_low_entropy': results_df['high_quality_low_entropy'].sum(),
            'slow_high_quality': results_df['slow_high_quality'].sum(),
            'nonmonotonic_recovery': results_df['nonmonotonic_recovery'].sum()
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(anomaly_counts.keys(), anomaly_counts.values())
        plt.title('Anomaly Counts in Pattern × Perturbation Matrix')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anomaly_counts.png"))
        plt.close()
        
        # List most interesting anomalies
        anomaly_cases = results_df[results_df['high_quality_low_entropy'] | 
                                 results_df['slow_high_quality'] | 
                                 results_df['nonmonotonic_recovery']]
        
        if not anomaly_cases.empty:
            with open(os.path.join(output_dir, "interesting_anomalies.txt"), 'w') as f:
                f.write("Interesting Cases for Further Investigation:\n\n")
                
                for idx, row in anomaly_cases.iterrows():
                    f.write(f"Pattern: {row['pattern']}, Perturbation: {row['perturbation_type']}\n")
                    f.write(f"Anomalies: {', '.join([k for k, v in row.items() if k in anomaly_counts and v])}\n")
                    f.write(f"Recovery Quality: {row['recovery_quality']:.4f}\n")
                    f.write(f"Recovery Time: {row['recovery_time']}\n")
                    f.write(f"Final Correlation: {row['final_correlation']:.4f}\n\n")
    
    def create_magnitude_sweep_summary(self, results_df, output_dir):
        """Create summary visualizations for magnitude sweep experiments"""
        plt.figure(figsize=(12, 8))
        
        # Group by pattern and magnitude
        for pattern in self.patterns:
            pattern_data = results_df[results_df['pattern'] == pattern]
            
            # Sort by magnitude
            pattern_data = pattern_data.sort_values('perturbation_magnitude')
            
            plt.plot(pattern_data['perturbation_magnitude'], 
                    pattern_data['recovery_quality'], 'o-', label=pattern)
        
        # Add threshold line
        plt.axhline(y=0.4, color='r', linestyle='--', label='Recovery Threshold')
        
        # Add labels
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Recovery Quality')
        plt.title('Recovery Quality vs Perturbation Magnitude by Pattern')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "magnitude_vs_quality.png"))
        plt.close()
        
        # Recovery time vs magnitude
        plt.figure(figsize=(12, 8))
        
        for pattern in self.patterns:
            pattern_data = results_df[results_df['pattern'] == pattern]
            pattern_data = pattern_data.sort_values('perturbation_magnitude')
            
            plt.plot(pattern_data['perturbation_magnitude'], 
                    pattern_data['recovery_time'], 'o-', label=pattern)
        
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Recovery Time (steps)')
        plt.title('Recovery Time vs Perturbation Magnitude by Pattern')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "magnitude_vs_time.png"))
        plt.close()
    
    def create_layered_perturbation_summary(self, results_df, output_dir):
        """Create summary visualizations for layered perturbation experiments"""
        plt.figure(figsize=(12, 8))
        
        # Compare first vs final recovery for each pattern and delay
        for pattern in self.patterns:
            pattern_data = results_df[results_df['pattern'] == pattern]
            pattern_data = pattern_data.sort_values('t_delay')
            
            plt.plot(pattern_data['t_delay'], 
                    pattern_data['first_recovery_correlation'], 'o--', label=f"{pattern} (1st Recovery)")
            plt.plot(pattern_data['t_delay'], 
                    pattern_data['final_correlation'], 'o-', label=f"{pattern} (Final)")
        
        plt.xlabel('Delay Between Perturbations (steps)')
        plt.ylabel('Correlation')
        plt.title('Effect of Perturbation Interval on Recovery')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "perturbation_interval_effect.png"))
        plt.close()
        
        # Recovery quality vs delay
        plt.figure(figsize=(12, 8))
        
        for pattern in self.patterns:
            pattern_data = results_df[results_df['pattern'] == pattern]
            pattern_data = pattern_data.sort_values('t_delay')
            
            plt.plot(pattern_data['t_delay'], 
                    pattern_data['recovery_quality'], 'o-', label=pattern)
        
        plt.xlabel('Delay Between Perturbations (steps)')
        plt.ylabel('Final Recovery Quality')
        plt.title('Recovery Quality vs Perturbation Interval')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "delay_vs_quality.png"))
        plt.close()
    
    def run_all_phase_I(self):
        """Run all Phase I experiments"""
        print("Running all Phase I experiments...")
        
        # Create a timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"phase1_run_{timestamp}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Run all experiment sets
        print("\n1. Pattern × Perturbation Matrix")
        matrix_results = self.run_pattern_perturbation_matrix()
        
        print("\n2. Perturbation Magnitude Sweep")
        magnitude_results = self.run_perturbation_magnitude_sweep()
        
        print("\n3. Layered Perturbation Experiments")
        layered_results = self.run_layered_perturbation_experiments()
        
        # Combine all results into a master results dictionary
        master_results = {
            "pattern_perturbation_matrix": matrix_results.to_dict('records'),
            "magnitude_sweep": magnitude_results.to_dict('records'),
            "layered_perturbation": layered_results.to_dict('records')
        }
        
        # Save master results JSON
        with open(os.path.join(run_dir, "phase1_master_results.json"), 'w') as f:
            json.dump(master_results, f)
        
        print(f"\nAll Phase I experiments completed and saved to {run_dir}")
        print("\nSummary of anomalies detected:")
        print(f"- In pattern × perturbation matrix: {matrix_results[['high_quality_low_entropy', 'slow_high_quality', 'nonmonotonic_recovery']].sum().sum()} anomalies")
        print(f"- In magnitude sweep: {magnitude_results[['high_quality_low_entropy', 'slow_high_quality', 'nonmonotonic_recovery']].sum().sum()} anomalies")
        print(f"- In layered perturbation: {layered_results[['high_quality_low_entropy', 'slow_high_quality', 'nonmonotonic_recovery']].sum().sum()} anomalies")
        
        return run_dir

# Example usage
if __name__ == "__main__":
    print("Enhanced Phase I RCFT Experiments")
    runner = EnhancedPhaseI(output_dir="phase1_results")
    results_dir = runner.run_all_phase_I()
    print(f"All experiments completed. Results saved to: {results_dir}")