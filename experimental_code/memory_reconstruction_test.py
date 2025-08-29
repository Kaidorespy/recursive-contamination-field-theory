# memory_reconstruction_test.py
import numpy as np
import matplotlib.pyplot as plt
import os
from phase4.learning_field import LearningFieldSimulator
from sklearn.metrics import mutual_info_score

class MemoryReconstructionAnalyzer(LearningFieldSimulator):
    """Analyzes whether memories are reconstructed faithfully or corrupted by interference"""
    
    def __init__(self, output_dir="phase4_results/memory_reconstruction", **kwargs):
        """Initialize with standard parameters"""
        super().__init__(output_dir=output_dir, **kwargs)
        
        # Create additional output directory for analysis
        self.analysis_dir = os.path.join(output_dir, "corruption_analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Store original patterns and their memory traces
        self.original_patterns = {}
        self.original_memory_traces = {}
        self.reconstruction_traces = {}
        
    def run_reconstruction_experiment(self, pattern_ids=['A', 'B'], sequence=None, 
                                     pause_between=2, return_cycles=3):
        """Run experiment with pattern introduction, interference, and return"""
        if sequence is None:
            # Build ABA pattern with multiple returns of A
            sequence = []
            for cycle in range(return_cycles):
                sequence.extend([pattern_ids[0]] * 3)  # Pattern A phase
                if cycle < return_cycles - 1:  # Don't add B after final A cycle
                    sequence.extend([pattern_ids[1]] * 3)  # Pattern B phase
                    
                    # Add pause (random patterns) between cycles
                    if pause_between > 0:
                        sequence.extend(['random'] * pause_between)
        
        print(f"Running reconstruction experiment with sequence: {sequence}")
        
        # Create patterns if they don't exist
        for pid in pattern_ids:
            if pid not in self.patterns:
                self.create_pattern(pid)
                
        # Save original patterns before any learning
        for pid in pattern_ids:
            self.original_patterns[pid] = self.patterns[pid].copy()
        
        # Run the pattern sequence
        results = self.run_multi_pattern_sequence(
            sequence, 1,  # Just one repetition
            initialize_with_trace=True,
            apply_perturbation=True,
            persistence_between_reps=True
        )
        
        # Analyze reconstruction vs corruption
        self._analyze_reconstruction(pattern_ids[0], results)
        
        return results
    
    def _analyze_reconstruction(self, target_pattern_id, results):
        """Analyze how the target pattern changes through reconstruction cycles"""
        # Extract phases where the target pattern appears
        target_results = [r for r in results if r['pattern_id'] == target_pattern_id]
        
        # Group by "cycle" (consecutive appearances)
        cycles = []
        current_cycle = []
        
        for i, result in enumerate(target_results):
            if i > 0 and result['episode'] > target_results[i-1]['episode'] + 1:
                # Gap in episodes indicates new cycle
                cycles.append(current_cycle)
                current_cycle = [result]
            else:
                current_cycle.append(result)
                
        if current_cycle:
            cycles.append(current_cycle)
        
        # Calculate reconstruction fidelity for each cycle
        original_pattern = self.original_patterns[target_pattern_id]
        
        # Get memory traces for each cycle
        memory_traces = []
        trace_correlations = []
        trace_mutual_info = []
        
        for i, cycle in enumerate(cycles):
            # Get the stabilized memory trace from the end of each cycle
            cycle_result = cycle[-1]  # Last result in cycle
            
            # Get memory trace for this pattern
            trace, strength = self.trace_manager.get_trace(target_pattern_id)
            
            if trace is not None:
                memory_traces.append(trace)
                
                # Save first trace as "original memory trace"
                if i == 0:
                    self.original_memory_traces[target_pattern_id] = trace.copy()
                elif i > 0:
                    # Save reconstruction traces
                    self.reconstruction_traces[f"{target_pattern_id}_cycle{i}"] = trace.copy()
                
                # Compare with original pattern
                pattern_corr = np.corrcoef(original_pattern.flatten(), trace.flatten())[0, 1]
                trace_correlations.append(pattern_corr)
                
                # Calculate mutual information
                bins = 10
                p1_bins = np.floor(bins * (original_pattern.flatten() + 1) / 2).astype(int)
                p2_bins = np.floor(bins * (trace.flatten() + 1) / 2).astype(int)
                mi = mutual_info_score(p1_bins, p2_bins)
                trace_mutual_info.append(mi)
                
                # If not the first cycle, compare with original memory trace
                if i > 0 and target_pattern_id in self.original_memory_traces:
                    orig_trace = self.original_memory_traces[target_pattern_id]
                    orig_trace_corr = np.corrcoef(orig_trace.flatten(), trace.flatten())[0, 1]
                    
                    # Create visualization comparing original and reconstruction
                    self._visualize_trace_comparison(
                        target_pattern_id, 
                        original_pattern,
                        orig_trace, 
                        trace, 
                        orig_trace_corr,
                        i
                    )
        
        # Create reconstruction fidelity plot
        self._create_reconstruction_plot(target_pattern_id, trace_correlations, trace_mutual_info)
        
    def _visualize_trace_comparison(self, pattern_id, original_pattern, 
                                  original_trace, reconstructed_trace, 
                                  correlation, cycle):
        """Visualize comparison between original and reconstructed memory traces"""
        plt.figure(figsize=(15, 5))
        
        # Plot original pattern
        plt.subplot(1, 4, 1)
        plt.imshow(original_pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Original Pattern {pattern_id}")
        plt.axis('off')
        
        # Plot original memory trace
        plt.subplot(1, 4, 2)
        plt.imshow(original_trace, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Original Memory Trace")
        plt.axis('off')
        
        # Plot reconstructed memory trace
        plt.subplot(1, 4, 3)
        plt.imshow(reconstructed_trace, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Reconstructed Trace (Cycle {cycle})")
        plt.axis('off')
        
        # Plot difference
        plt.subplot(1, 4, 4)
        plt.imshow(reconstructed_trace - original_trace, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"Difference (Corr: {correlation:.3f})")
        plt.axis('off')
        
        plt.suptitle(f"Memory Reconstruction Analysis - Pattern {pattern_id}, Cycle {cycle}")
        plt.tight_layout()
        
        filename = f"reconstruction_{pattern_id}_cycle{cycle}.png"
        plt.savefig(os.path.join(self.analysis_dir, filename))
        plt.close()
        
    def _create_reconstruction_plot(self, pattern_id, correlations, mutual_info):
        """Create plot showing how reconstruction fidelity changes across cycles"""
        plt.figure(figsize=(10, 6))
        
        x = range(1, len(correlations) + 1)
        
        # Normalize MI to same scale as correlation for comparison
        max_mi = max(mutual_info) if mutual_info else 1
        norm_mi = [mi / max_mi for mi in mutual_info]
        
        # Plot correlation
        plt.plot(x, correlations, 'o-', color='blue', label='Correlation with Original Pattern')
        
        # Plot normalized mutual information
        plt.plot(x, norm_mi, 's--', color='red', label='Mutual Information (Normalized)')
        
        # Add threshold line for "high fidelity"
        plt.axhline(y=0.9, color='green', linestyle='--', label='High Fidelity (0.9)')
        
        # Add threshold line for "significant corruption"
        plt.axhline(y=0.7, color='orange', linestyle='--', label='Moderate Corruption (0.7)')
        
        # Add threshold line for "severe corruption"
        plt.axhline(y=0.5, color='red', linestyle='--', label='Severe Corruption (0.5)')
        
        plt.title(f"Memory Reconstruction Fidelity - Pattern {pattern_id}")
        plt.xlabel("Reconstruction Cycle")
        plt.ylabel("Fidelity Metric")
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f"reconstruction_fidelity_{pattern_id}.png"))
        plt.close()
        
        # Create corruption index visualization
        self._create_corruption_index_plot(pattern_id, correlations)
        
    def _create_corruption_index_plot(self, pattern_id, correlations):
        """Create visualization showing the corruption index over time"""
        if len(correlations) <= 1:
            return  # Need at least 2 points
            
        # Calculate corruption index (how much fidelity drops with each cycle)
        corruption_indices = []
        
        for i in range(1, len(correlations)):
            # How much correlation dropped from previous cycle
            corruption = correlations[i-1] - correlations[i]
            
            # Express as percentage of previous correlation
            corruption_percentage = (corruption / correlations[i-1]) * 100
            corruption_indices.append(corruption_percentage)
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(range(2, len(correlations) + 1), corruption_indices, 
                     color=['green' if c < 5 else 'orange' if c < 15 else 'red' for c in corruption_indices])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.axhline(y=5, color='green', linestyle='--', label='Minimal Corruption (5%)')
        plt.axhline(y=15, color='orange', linestyle='--', label='Moderate Corruption (15%)')
        plt.axhline(y=30, color='red', linestyle='--', label='Severe Corruption (30%)')
        
        plt.title(f"Memory Corruption Index - Pattern {pattern_id}")
        plt.xlabel("Reconstruction Cycle")
        plt.ylabel("Corruption Index (%)")
        plt.grid(True)
        plt.legend()
        
        # Create parasitic interference analysis
        self._calculate_parasitic_interference(pattern_id)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f"corruption_index_{pattern_id}.png"))
        plt.close()
        
    def _calculate_parasitic_interference(self, target_pattern_id):
        """Calculate whether reconstructed memories show evidence of parasitic interference"""
        if len(self.reconstruction_traces) < 1:
            return  # Need at least one reconstruction
        
        # Get other patterns (potential sources of parasitic interference)
        other_patterns = [pid for pid in self.patterns.keys() if pid != target_pattern_id]
        
        if not other_patterns:
            return  # Need at least one other pattern
            
        # For each reconstructed trace, calculate correlation with each potential interfering pattern
        parasitic_scores = {}
        
        for rec_id, rec_trace in self.reconstruction_traces.items():
            cycle = int(rec_id.split('cycle')[1])
            parasitic_scores[cycle] = {}
            
            # For each other pattern, calculate correlation
            for other_id in other_patterns:
                other_pattern = self.patterns[other_id]
                
                # Calculate correlation between reconstruction and other pattern
                corr = np.corrcoef(rec_trace.flatten(), other_pattern.flatten())[0, 1]
                parasitic_scores[cycle][other_id] = corr
        
        # Create parasitic influence visualization
        self._visualize_parasitic_influence(target_pattern_id, parasitic_scores)
        
    def _visualize_parasitic_influence(self, target_pattern_id, parasitic_scores):
        """Visualize parasitic influence from other patterns"""
        if not parasitic_scores:
            return
            
        plt.figure(figsize=(12, 8))
        
        # For each pattern, plot its influence over reconstruction cycles
        cycles = sorted(parasitic_scores.keys())
        
        for other_id in parasitic_scores[cycles[0]].keys():
            # Extract scores for this pattern across cycles
            scores = [parasitic_scores[c][other_id] for c in cycles]
            
            plt.plot(cycles, scores, 'o-', label=f"Pattern {other_id} Influence")
        
        plt.axhline(y=0.3, color='orange', linestyle='--', 
                  label='Moderate Parasitic Influence (0.3)')
        plt.axhline(y=0.5, color='red', linestyle='--', 
                  label='Strong Parasitic Influence (0.5)')
        
        plt.title(f"Parasitic Influence on Reconstructed Memories - Pattern {target_pattern_id}")
        plt.xlabel("Reconstruction Cycle")
        plt.ylabel("Correlation with Interfering Pattern")
        plt.ylim(-0.1, 1.0)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, f"parasitic_influence_{target_pattern_id}.png"))
        plt.close()

if __name__ == "__main__":
    # Run reconstruction experiment
    analyzer = MemoryReconstructionAnalyzer(
        output_dir="phase4_results/memory_reconstruction",
        alpha=0.35,
        beta=0.5,
        gamma=0.92,
        pattern_type="fractal",
        max_steps=200,
        memory_decay_rate=0.7,
        memory_reinforcement_rate=0.3,
        baseline_strength=0.1,
        persistence_mode="both"
    )
    
    # Create a sequence with 3 cycles of A followed by B interference
    # ABA pattern with longer sequences of each
    sequence = []
    for i in range(3):
        # Pattern A repeated 3 times
        sequence.extend(['A'] * 3)
        # Pattern B repeated 3 times
        sequence.extend(['B'] * 3)
        
    print(f"Running experiment with sequence: {sequence}")
    results = analyzer.run_reconstruction_experiment(
        pattern_ids=['A', 'B'],
        sequence=sequence,
        pause_between=0,
        return_cycles=3
    )
    
    print("Experiment completed. Results saved to phase4_results/memory_reconstruction")