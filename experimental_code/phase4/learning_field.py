# phase4/learning_field.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pickle

from .base import DirectedMemoryExperiment

class MemoryTraceManager:
    """Manages memory traces across multiple trials"""
    
    def __init__(self, decay_rate=0.8, reinforcement_rate=0.2):
        """Initialize the memory trace manager"""
        self.decay_rate = decay_rate  # Rate at which unused traces decay
        self.reinforcement_rate = reinforcement_rate  # Rate at which successful traces are reinforced
        
        self.memory_traces = {}  # Dictionary of memory traces keyed by pattern_id
        self.trace_metadata = {}  # Metadata for each trace
        self.baseline_memory = None  # Baseline memory field (can be None)
    
    def add_trace(self, pattern_id, memory_field, initial_strength=1.0, metadata=None):
        """Add a new memory trace"""
        self.memory_traces[pattern_id] = {
            'field': memory_field.copy(),
            'strength': initial_strength,
            'initial_strength': initial_strength,
            'exposure_count': 1,
            'last_accessed': datetime.now()
        }
        
        if metadata:
            self.trace_metadata[pattern_id] = metadata
    
    def update_trace(self, pattern_id, memory_field, recovery_quality):
        """Update an existing memory trace based on recovery quality"""
        if pattern_id not in self.memory_traces:
            # If trace doesn't exist, add it
            self.add_trace(pattern_id, memory_field)
            return
        
        # Get current trace
        trace = self.memory_traces[pattern_id]
        
        # Blend the existing field with the new field based on recovery quality
        # High recovery quality -> more weight to existing field (reinforcement)
        # Low recovery quality -> more weight to new field (correction)
        if recovery_quality > 0.8:
            # Successful recovery - reinforce existing trace
            blend_weight = 1.0 - self.reinforcement_rate
            trace['field'] = blend_weight * trace['field'] + self.reinforcement_rate * memory_field
            trace['strength'] = min(1.0, trace['strength'] + 0.1 * recovery_quality)
        else:
            # Poor recovery - update more with new information
            correction_rate = 0.5 * (1.0 - recovery_quality)
            trace['field'] = (1.0 - correction_rate) * trace['field'] + correction_rate * memory_field
        
        # Update metadata
        trace['exposure_count'] += 1
        trace['last_accessed'] = datetime.now()
    
    def decay_traces(self, active_pattern_id=None):
        """Decay all memory traces except the active one"""
        for pattern_id, trace in self.memory_traces.items():
            if pattern_id != active_pattern_id:
                # Decay inactive traces
                trace['strength'] *= self.decay_rate
                
                # Remove very weak traces
                if trace['strength'] < 0.01:
                    del self.memory_traces[pattern_id]
                    if pattern_id in self.trace_metadata:
                        del self.trace_metadata[pattern_id]
    
    def get_trace(self, pattern_id):
        """Get a specific memory trace"""
        if pattern_id in self.memory_traces:
            trace = self.memory_traces[pattern_id]
            trace['last_accessed'] = datetime.now()  # Update access time
            return trace['field'].copy(), trace['strength']
        return None, 0.0
    
    def get_blended_field(self, pattern_weights=None):
        """Get a blended field combining multiple traces"""
        if not self.memory_traces:
            return None
            
        # If no weights provided, use trace strengths
        if pattern_weights is None:
            pattern_weights = {pid: trace['strength'] for pid, trace in self.memory_traces.items()}
            
        # Normalize weights
        total_weight = sum(pattern_weights.values())
        if total_weight == 0:
            return None
            
        normalized_weights = {pid: w/total_weight for pid, w in pattern_weights.items()}
        
        # Blend fields
        blended_field = None
        for pattern_id, weight in normalized_weights.items():
            if pattern_id in self.memory_traces:
                field = self.memory_traces[pattern_id]['field']
                if blended_field is None:
                    blended_field = weight * field
                else:
                    blended_field += weight * field
        
        return blended_field
    
    def set_baseline_memory(self, memory_field):
        """Set a baseline memory field that's always present"""
        self.baseline_memory = memory_field.copy()
    
    def combine_with_baseline(self, memory_field, baseline_weight=0.1):
        """Combine a memory field with the baseline memory"""
        if self.baseline_memory is None:
            return memory_field
            
        # Blend with baseline
        return (1.0 - baseline_weight) * memory_field + baseline_weight * self.baseline_memory
    
    def save_traces(self, filepath):
        """Save memory traces to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'traces': self.memory_traces,
                'metadata': self.trace_metadata,
                'baseline': self.baseline_memory
            }, f)
    
    def load_traces(self, filepath):
        """Load memory traces from file"""
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.memory_traces = data.get('traces', {})
        self.trace_metadata = data.get('metadata', {})
        self.baseline_memory = data.get('baseline', None)
        
        return True
    
    def get_trace_summary(self):
        """Get a summary of all memory traces"""
        summary = []
        
        for pattern_id, trace in self.memory_traces.items():
            summary.append({
                'pattern_id': pattern_id,
                'strength': trace['strength'],
                'exposure_count': trace['exposure_count'],
                'last_accessed': trace['last_accessed'].strftime('%Y-%m-%d %H:%M:%S')
            })
            
        return summary

class LearningFieldSimulator(DirectedMemoryExperiment):
    """Simulates learning in RCFT fields through multi-episode memory retention"""
    
    def __init__(self, 
                 output_dir="phase4_results/learning_field",
                 alpha=0.35, 
                 beta=0.5, 
                 gamma=0.92, 
                 pattern_type="fractal",
                 max_steps=100,
                 memory_decay_rate=0.8,
                 memory_reinforcement_rate=0.2,
                 baseline_strength=0.1,
                 persistence_mode="both"):
        """Initialize the learning field simulator"""
        super().__init__(output_dir, alpha, beta, gamma, pattern_type)
        
        self.max_steps = max_steps
        self.memory_decay_rate = memory_decay_rate
        self.memory_reinforcement_rate = memory_reinforcement_rate
        self.baseline_strength = baseline_strength
        self.persistence_mode = persistence_mode  # "memory", "file", or "both"
        
        # Create memory trace manager
        self.trace_manager = MemoryTraceManager(
            decay_rate=memory_decay_rate,
            reinforcement_rate=memory_reinforcement_rate
        )
        
        # Create output subdirectories
        os.makedirs(os.path.join(output_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "learning_curves"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "memory_traces"), exist_ok=True)
        
        # Storage for patterns and results
        self.patterns = {}
        self.learning_results = []
        self.repetition_results = {}
    
    def create_pattern(self, pattern_id, pattern_type=None, **kwargs):
        """Create and store a new pattern"""
        if pattern_type is None:
            pattern_type = self.pattern_type
            
        # Initialize a temporary experiment to generate the pattern
        exp = self.initialize_experiment()
        exp.initialize_pattern(pattern_type=pattern_type, **kwargs)
        
        # Store the pattern
        self.patterns[pattern_id] = exp.state.copy()
        
        # Save pattern visualization
        self._save_pattern_visualization(pattern_id, exp.state)
        
        return exp.state
    
    def _save_pattern_visualization(self, pattern_id, pattern):
        """Save visualization of a pattern"""
        plt.figure(figsize=(6, 6))
        plt.imshow(pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title(f"Pattern: {pattern_id}")
        plt.axis('off')
        
        filename = f"pattern_{pattern_id}.png"
        plt.savefig(os.path.join(self.output_dir, "patterns", filename))
        plt.close()
    
    def run_learning_episode(self, pattern_id, episode_number, 
                           initialize_with_trace=True, apply_perturbation=True):
        """Run a single learning episode with a specific pattern"""
        # Initialize experiment
        self.initialize_experiment()
        
        # Get pattern from storage
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            self.experiment.state = pattern.copy()
            self.initial_state = pattern.copy()
        else:
            # If pattern doesn't exist, create it
            pattern = self.create_pattern(pattern_id)
            self.experiment.state = pattern.copy()
            self.initial_state = pattern.copy()
        
        # Get memory trace if exists and requested
        memory_field = None
        trace_strength = 0.0
        
        if initialize_with_trace:
            memory_field, trace_strength = self.trace_manager.get_trace(pattern_id)
            
            if memory_field is not None:
                # Replace experiment's memory with the trace, modified by trace strength
                self.experiment.memory = memory_field.copy() * trace_strength
                
                # If using baseline memory, blend it in
                if self.baseline_strength > 0.0 and self.trace_manager.baseline_memory is not None:
                    self.experiment.memory = (1.0 - self.baseline_strength) * self.experiment.memory + \
                                           self.baseline_strength * self.trace_manager.baseline_memory
        
        # Apply perturbation if requested
        if apply_perturbation:
            self.experiment.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
        
        # Save initial metrics
        initial_metrics = self.compute_metrics()
        
        # Run recovery
        self.experiment.update(steps=self.max_steps)
        
        # Compute final metrics
        final_metrics = self.compute_metrics()
        
        # Calculate recovery quality
        recovery = self.experiment.calculate_recovery_metrics()
        recovery_quality = recovery.get('recovery_quality', 0.0)
        
        # Update memory trace with current memory field
        current_memory = self.experiment.memory.copy()
        self.trace_manager.update_trace(pattern_id, current_memory, recovery_quality)
        
        # Decay other traces
        self.trace_manager.decay_traces(active_pattern_id=pattern_id)
        
        # Save memory trace file if using file persistence
        if self.persistence_mode in ["file", "both"]:
            trace_file = os.path.join(self.output_dir, "memory_traces", f"traces_{episode_number}.pkl")
            self.trace_manager.save_traces(trace_file)
        
        # Create result record
        result = {
            'pattern_id': pattern_id,
            'episode': episode_number,
            'trace_strength_before': trace_strength,
            'trace_strength_after': self.trace_manager.memory_traces.get(pattern_id, {}).get('strength', 0.0),
            'final_correlation': final_metrics['correlation'],
            'recovery_quality': recovery_quality,
            'recovery_time': recovery.get('recovery_time', self.max_steps)
        }
        
        # Save visualization
        self._save_episode_visualization(pattern_id, episode_number, recovery_quality)
        
        return result
    
    def _save_episode_visualization(self, pattern_id, episode, recovery_quality):
        """Save visualization of a learning episode"""
        plt.figure(figsize=(15, 5))
        
        # Plot initial state
        plt.subplot(1, 3, 1)
        plt.imshow(self.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title("Original Pattern")
        plt.axis('off')
        
        # Plot perturbed state
        plt.subplot(1, 3, 2)
        if self.experiment.perturbation_step > 0:
            plt.imshow(self.experiment.history[self.experiment.perturbation_step], 
                     cmap='viridis', vmin=-1, vmax=1)
            plt.title("After Perturbation")
        else:
            plt.imshow(self.experiment.history[0], cmap='viridis', vmin=-1, vmax=1)
            plt.title("Initial State")
        plt.axis('off')
        
        # Plot final state
        plt.subplot(1, 3, 3)
        plt.imshow(self.experiment.state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Recovered State (quality={recovery_quality:.3f})")
        plt.axis('off')
        
        plt.suptitle(f"Learning Episode {episode}: Pattern {pattern_id}")
        plt.tight_layout()
        
        filename = f"episode_{episode}_pattern_{pattern_id}.png"
        plt.savefig(os.path.join(self.output_dir, "learning_curves", filename))
        plt.close()
        
        # Plot metrics
        plt.figure(figsize=(10, 6))
        
        # Plot correlation
        plt.plot(self.experiment.metrics['correlation'], 'b-', label='Correlation')
        plt.plot(self.experiment.metrics['coherence'], 'g-', label='Coherence')
        
        # Mark perturbation if applicable
        if self.experiment.perturbation_step > 0:
            plt.axvline(x=self.experiment.perturbation_step, color='r', 
                      linestyle='--', label='Perturbation')
        
        plt.title(f"Recovery Metrics - Episode {episode}, Pattern {pattern_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Metric Value")
        plt.grid(True)
        plt.legend()
        
        filename = f"metrics_episode_{episode}_pattern_{pattern_id}.png"
        plt.savefig(os.path.join(self.output_dir, "learning_curves", filename))
        plt.close()
    
    def run_multi_pattern_sequence(self, pattern_sequence, num_repetitions, 
                                 initialize_with_trace=True, apply_perturbation=True,
                                 persistence_between_reps=True):
        """Run a sequence of patterns multiple times to observe learning"""
        all_results = []
        
        # Initialize tracking
        episode_number = 0
        
        # Set baseline memory if first run
        if not self.trace_manager.memory_traces and self.baseline_strength > 0:
            # Create a random baseline field
            baseline = np.random.uniform(-0.1, 0.1, (self.experiment.size, self.experiment.size))
            self.trace_manager.set_baseline_memory(baseline)
        
        # Load previous traces if continuing and using file persistence
        if self.persistence_mode in ["file", "both"] and persistence_between_reps:
            latest_trace_file = None
            trace_files = [f for f in os.listdir(os.path.join(self.output_dir, "memory_traces")) 
                         if f.startswith("traces_") and f.endswith(".pkl")]
            
            if trace_files:
                # Find the latest trace file
                latest_trace_file = sorted(trace_files, key=lambda f: int(f.split("_")[1].split(".")[0]))[-1]
                
                if latest_trace_file:
                    self.trace_manager.load_traces(
                        os.path.join(self.output_dir, "memory_traces", latest_trace_file)
                    )
        
        # Run repetitions
        for rep in tqdm(range(num_repetitions), desc="Repetitions"):
            rep_results = []
            
            # Run each pattern in the sequence
            for i, pattern_id in enumerate(pattern_sequence):
                episode_number += 1
                
                result = self.run_learning_episode(
                    pattern_id, 
                    episode_number,
                    initialize_with_trace=initialize_with_trace,
                    apply_perturbation=apply_perturbation
                )
                
                # Add repetition and sequence information
                result['repetition'] = rep
                result['sequence_position'] = i
                
                rep_results.append(result)
                all_results.append(result)
                
            # Create per-repetition summary
            self._save_repetition_summary(rep, rep_results)
            
            # Save traces between repetitions if requested
            if persistence_between_reps and self.persistence_mode in ["file", "both"]:
                rep_trace_file = os.path.join(self.output_dir, "memory_traces", f"traces_rep{rep}.pkl")
                self.trace_manager.save_traces(rep_trace_file)
        
        # Store results
        self.learning_results = all_results
        
        # Create learning curve visualization
        self._create_learning_curve_visualization(pattern_sequence, num_repetitions)
        
        return all_results
    
    def _save_repetition_summary(self, repetition, results):
        """Save summary of a repetition"""
        # Store in object
        self.repetition_results[repetition] = results
        
        # Group by pattern
        pattern_metrics = {}
        
        for result in results:
            pattern_id = result['pattern_id']
            if pattern_id not in pattern_metrics:
                pattern_metrics[pattern_id] = []
            pattern_metrics[pattern_id].append(result)
        
        # Create summary visualization
        plt.figure(figsize=(12, 8))
        
        # Plot recovery quality for each pattern
        for pattern_id, metrics in pattern_metrics.items():
            positions = [m['sequence_position'] for m in metrics]
            qualities = [m['recovery_quality'] for m in metrics]
            
            plt.plot(positions, qualities, 'o-', label=f"Pattern {pattern_id}")
        
        plt.axhline(y=0.9, color='green', linestyle='--', label='Strong Recovery (0.9)')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Recovery Threshold (0.4)')
        
        plt.title(f"Repetition {repetition} - Recovery Quality by Pattern Sequence Position")
        plt.xlabel("Sequence Position")
        plt.ylabel("Recovery Quality")
        plt.grid(True)
        plt.legend()
        
        filename = f"repetition_{repetition}_summary.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        # Create trace strength visualization
        plt.figure(figsize=(10, 6))
        
        for pattern_id, metrics in pattern_metrics.items():
            strengths = [m['trace_strength_after'] for m in metrics]
            plt.plot(range(len(strengths)), strengths, 'o-', label=f"Pattern {pattern_id}")
        
        plt.title(f"Repetition {repetition} - Memory Trace Strength Evolution")
        plt.xlabel("Pattern Sequence Position")
        plt.ylabel("Trace Strength")
        plt.grid(True)
        plt.legend()
        
        filename = f"repetition_{repetition}_trace_strengths.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def _create_learning_curve_visualization(self, pattern_sequence, num_repetitions):
        """Create visualization of learning curves across repetitions"""
        if not self.learning_results:
            return
            
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(self.learning_results)
        
        # Group by pattern_id and repetition
        grouped = df.groupby(['pattern_id', 'repetition']).agg({
            'recovery_quality': 'mean',
            'recovery_time': 'mean',
            'trace_strength_after': 'mean'
        }).reset_index()
        
        # Create learning curve plot
        plt.figure(figsize=(12, 8))
        
        # Plot recovery quality by repetition for each pattern
        for pattern_id in set(df['pattern_id']):
            pattern_data = grouped[grouped['pattern_id'] == pattern_id]
            
            plt.plot(pattern_data['repetition'], pattern_data['recovery_quality'], 
                   'o-', label=f"Pattern {pattern_id}")
        
        plt.axhline(y=0.9, color='green', linestyle='--', label='Strong Recovery (0.9)')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Recovery Threshold (0.4)')
        
        plt.title("Learning Curve: Recovery Quality vs. Repetition")
        plt.xlabel("Repetition Number")
        plt.ylabel("Recovery Quality")
        plt.xticks(range(num_repetitions))
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "learning_curve_quality.png"))
        plt.close()
        
        # Create recovery time learning curve
        plt.figure(figsize=(12, 8))
        
        for pattern_id in set(df['pattern_id']):
            pattern_data = grouped[grouped['pattern_id'] == pattern_id]
            
            plt.plot(pattern_data['repetition'], pattern_data['recovery_time'], 
                   'o-', label=f"Pattern {pattern_id}")
        
        plt.title("Learning Curve: Recovery Time vs. Repetition")
        plt.xlabel("Repetition Number")
        plt.ylabel("Recovery Time (steps)")
        plt.xticks(range(num_repetitions))
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "learning_curve_time.png"))
        plt.close()
        
        # Create trace strength evolution
        plt.figure(figsize=(12, 8))
        
        for pattern_id in set(df['pattern_id']):
            pattern_data = grouped[grouped['pattern_id'] == pattern_id]
            
            plt.plot(pattern_data['repetition'], pattern_data['trace_strength_after'], 
                   'o-', label=f"Pattern {pattern_id}")
        
        plt.title("Memory Trace Strength Evolution")
        plt.xlabel("Repetition Number")
        plt.ylabel("Trace Strength")
        plt.xticks(range(num_repetitions))
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "trace_strength_evolution.png"))
        plt.close()
        
        # Fit exponential curves to learning data if scipy is available
        try:
            from scipy.optimize import curve_fit
            
            def exp_func(x, a, b, c):
                return a * (1 - np.exp(-b * x)) + c
                
            plt.figure(figsize=(12, 8))
            
            for pattern_id in set(df['pattern_id']):
                pattern_data = grouped[grouped['pattern_id'] == pattern_id]
                
                x_data = pattern_data['repetition'].values
                y_data = pattern_data['recovery_quality'].values
                
                # Need at least 3 points to fit exponential
                if len(x_data) >= 3:
                    try:
                        # Fit exponential curve
                        popt, _ = curve_fit(exp_func, x_data, y_data, 
                                          p0=[0.5, 0.5, 0.4], maxfev=10000)
                        
                        # Plot data points
                        plt.plot(x_data, y_data, 'o', label=f"Pattern {pattern_id} (data)")
                        
                        # Plot fitted curve
                        x_fit = np.linspace(0, max(x_data), 100)
                        y_fit = exp_func(x_fit, *popt)
                        plt.plot(x_fit, y_fit, '-', 
                               label=f"Pattern {pattern_id} (fit: {popt[0]:.2f}(1-e^(-{popt[1]:.2f}x))+{popt[2]:.2f})")
                        
                    except RuntimeError:
                        # Fallback if fitting fails
                        plt.plot(x_data, y_data, 'o-', label=f"Pattern {pattern_id}")
                else:
                    plt.plot(x_data, y_data, 'o-', label=f"Pattern {pattern_id}")
            
            plt.axhline(y=0.9, color='green', linestyle='--', label='Strong Recovery (0.9)')
            plt.axhline(y=0.4, color='red', linestyle='--', label='Recovery Threshold (0.4)')
            
            plt.title("Learning Curve with Exponential Fit")
            plt.xlabel("Repetition Number")
            plt.ylabel("Recovery Quality")
            plt.xticks(range(num_repetitions))
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "learning_curve_fit.png"))
            plt.close()
                
        except ImportError:
            pass  # Skip curve fitting if scipy not available
    
    def run_interference_experiment(self, target_pattern_id, interfering_patterns, 
                                  num_repetitions, interfere_after=3,
                                  initialize_with_trace=True, apply_perturbation=True):
        """Run experiment to study interference between multiple patterns"""
        results = []
        
        # First, train on target pattern to establish memory trace
        print(f"Training on target pattern {target_pattern_id} for {interfere_after} repetitions...")
        
        for i in range(interfere_after):
            result = self.run_learning_episode(
                target_pattern_id, 
                i,
                initialize_with_trace=initialize_with_trace,
                apply_perturbation=apply_perturbation
            )
            result['phase'] = 'target_training'
            result['interference_pattern'] = None
            results.append(result)
        
        # Save target trace metrics after training
        target_trace_after_training = None
        if target_pattern_id in self.trace_manager.memory_traces:
            target_trace_after_training = {
                'strength': self.trace_manager.memory_traces[target_pattern_id]['strength'],
                'exposure_count': self.trace_manager.memory_traces[target_pattern_id]['exposure_count']
            }
        
        # Now introduce interfering patterns
        print(f"Introducing {len(interfering_patterns)} interfering patterns...")
        
        episode = interfere_after
        for interfering_id in interfering_patterns:
            for i in range(num_repetitions):
                episode += 1
                result = self.run_learning_episode(
                    interfering_id, 
                    episode,
                    initialize_with_trace=initialize_with_trace,
                    apply_perturbation=apply_perturbation
                )
                result['phase'] = 'interference'
                result['interference_pattern'] = interfering_id
                results.append(result)
        
        # Check target trace metrics after interference
        target_trace_after_interference = None
        if target_pattern_id in self.trace_manager.memory_traces:
            target_trace_after_interference = {
                'strength': self.trace_manager.memory_traces[target_pattern_id]['strength'],
                'exposure_count': self.trace_manager.memory_traces[target_pattern_id]['exposure_count']
            }
        
        # Test target pattern recall after interference
        print(f"Testing target pattern recall after interference...")
        
        episode += 1
        result = self.run_learning_episode(
            target_pattern_id, 
            episode,
            initialize_with_trace=initialize_with_trace,
            apply_perturbation=apply_perturbation
        )
        result['phase'] = 'recall_test'
        result['interference_pattern'] = None
        results.append(result)
        
        # Create interference analysis visualization
        self._create_interference_visualization(
            results, 
            target_pattern_id, 
            interfering_patterns,
            target_trace_after_training,
            target_trace_after_interference
        )
        
        return results
    
    def _create_interference_visualization(self, results, target_pattern_id, 
                                         interfering_patterns, 
                                         trace_before, trace_after):
        """Create visualization of interference effects"""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create phase markers
        phase_changes = np.where(df['phase'] != df['phase'].shift(1))[0]
        phase_labels = df['phase'].iloc[phase_changes].values
        
        # Plot recovery quality timeline
        plt.figure(figsize=(12, 8))
        
        # Plot by pattern
        for pattern_id in set(df['pattern_id']):
            pattern_data = df[df['pattern_id'] == pattern_id]
            
            marker = 'o' if pattern_id == target_pattern_id else 's'
            plt.plot(pattern_data.index, pattern_data['recovery_quality'], 
                   marker + '-', label=f"Pattern {pattern_id}")
        
        # Add phase markers
        for i, phase in enumerate(phase_changes):
            if i < len(phase_labels):
                plt.axvline(x=phase, color='gray', linestyle='--')
                plt.text(phase, 0.1, phase_labels[i], rotation=90, verticalalignment='bottom')
        
        plt.axhline(y=0.9, color='green', linestyle='--', label='Strong Recovery (0.9)')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Recovery Threshold (0.4)')
        
        plt.title(f"Interference Experiment - Target: Pattern {target_pattern_id}")
        plt.xlabel("Episode Number")
        plt.ylabel("Recovery Quality")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "interference_timeline.png"))
        plt.close()
        
        # Plot trace strength timeline
        plt.figure(figsize=(12, 8))
        
        for pattern_id in set(df['pattern_id']):
            pattern_data = df[df['pattern_id'] == pattern_id]
            
            marker = 'o' if pattern_id == target_pattern_id else 's'
            plt.plot(pattern_data.index, pattern_data['trace_strength_after'], 
                   marker + '-', label=f"Pattern {pattern_id}")
        
        # Add phase markers
        for i, phase in enumerate(phase_changes):
            if i < len(phase_labels):
                plt.axvline(x=phase, color='gray', linestyle='--')
                plt.text(phase, 0.1, phase_labels[i], rotation=90, verticalalignment='bottom')
        
        plt.title(f"Memory Trace Strength Evolution - Target: Pattern {target_pattern_id}")
        plt.xlabel("Episode Number")
        plt.ylabel("Trace Strength")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "interference_trace_strength.png"))
        plt.close()
        
        # If we have before/after trace data, create comparison
        if trace_before and trace_after:
            plt.figure(figsize=(8, 6))
            
            labels = ['Before Interference', 'After Interference']
            strengths = [trace_before['strength'], trace_after['strength']]
            
            plt.bar(labels, strengths, color=['blue', 'orange'])
            
            change = ((trace_after['strength'] - trace_before['strength']) / 
                     trace_before['strength'] * 100)
            
            plt.title(f"Target Memory Trace Strength Change: {change:.1f}%")
            plt.ylabel("Trace Strength")
            
            # Add strength labels
            for i, v in enumerate(strengths):
                plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "interference_strength_comparison.png"))
            plt.close()
    
    def run(self, pattern_sequence=None, num_repetitions=5, **kwargs):
        """Run a multi-pattern learning experiment"""
        if pattern_sequence is None:
            # Create default pattern sequence
            pattern_sequence = ['A', 'B', 'C']
            
            # Create patterns if they don't exist
            for pid in pattern_sequence:
                if pid not in self.patterns:
                    self.create_pattern(pid)
        
        # Run the learning experiment
        results = self.run_multi_pattern_sequence(
            pattern_sequence,
            num_repetitions,
            initialize_with_trace=kwargs.get('initialize_with_trace', True),
            apply_perturbation=kwargs.get('apply_perturbation', True),
            persistence_between_reps=kwargs.get('persistence_between_reps', True)
        )
        
        return results
    
    def visualize_results(self, **kwargs):
        """Create comprehensive visualization of results"""
        if not self.learning_results:
            return
            
        # Create a learning efficacy summary
        df = pd.DataFrame(self.learning_results)
        
        # Compute learning rate
        learning_rates = {}
        
        for pattern_id in set(df['pattern_id']):
            pattern_data = df[df['pattern_id'] == pattern_id]
            
            # Group by repetition
            grouped = pattern_data.groupby('repetition')['recovery_quality'].mean()
            
            # If we have at least 2 repetitions, calculate learning rate
            if len(grouped) >= 2:
                first_quality = grouped.iloc[0]
                last_quality = grouped.iloc[-1]
                
                # Simple learning rate
                learning_rates[pattern_id] = (last_quality - first_quality) / (len(grouped) - 1)
        
        # Create learning rate visualization
        plt.figure(figsize=(10, 6))
        
        patterns = list(learning_rates.keys())
        rates = [learning_rates[p] for p in patterns]
        
        plt.bar(patterns, rates, color='blue', alpha=0.7)
        
        plt.title("Learning Rate by Pattern")
        plt.xlabel("Pattern ID")
        plt.ylabel("Learning Rate (Recovery Quality / Repetition)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add rate labels
        for i, v in enumerate(rates):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "learning_rates.png"))
        plt.close()
        
        # Create memory system status visualization
        trace_summary = self.trace_manager.get_trace_summary()
        
        if trace_summary:
            plt.figure(figsize=(10, 6))
            
            patterns = [t['pattern_id'] for t in trace_summary]
            strengths = [t['strength'] for t in trace_summary]
            exposures = [t['exposure_count'] for t in trace_summary]
            
            # Create primary strength bars
            bars = plt.bar(patterns, strengths, color='blue', alpha=0.7, label='Trace Strength')
            
            # Annotate with exposure counts
            for i, (bar, exp) in enumerate(zip(bars, exposures)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{exp} exp", ha='center', va='bottom')
            
            plt.title("Memory System Status")
            plt.xlabel("Pattern ID")
            plt.ylabel("Trace Strength")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "memory_system_status.png"))
            plt.close()