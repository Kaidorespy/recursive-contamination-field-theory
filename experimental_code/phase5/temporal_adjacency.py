"""
Temporal Adjacency Encoding module for Phase V.
This module tests memory chaining and how remembering one pattern aids recall of adjacent patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from .module_base import PhaseVModule
from .utils.metrics import IdentityMetrics

class TemporalAdjacencyEncoder(PhaseVModule):
    """Module for testing memory chaining and adjacency effects."""
    
    def __init__(self, output_dir=None, sequence_length=3, 
                perturbation_type="flip", perturbation_magnitude=1.0,
                steps_per_pattern=100, n_trials=3, **kwargs):
        """
        Initialize the Temporal Adjacency Encoder.
        
        Args:
            output_dir: Directory for output files
            sequence_length: Length of pattern sequence
            perturbation_type: Type of perturbation to apply
            perturbation_magnitude: Magnitude of perturbation
            steps_per_pattern: Steps to run for each pattern
            n_trials: Number of trials to run
            **kwargs: Additional parameters
        """
        super().__init__(output_dir=output_dir, 
                       sequence_length=sequence_length,
                       perturbation_type=perturbation_type,
                       perturbation_magnitude=perturbation_magnitude,
                       steps_per_pattern=steps_per_pattern,
                       n_trials=n_trials, **kwargs)
        
        # Sequence parameters
        self.sequence_length = sequence_length
        
        # Perturbation parameters
        self.perturbation_type = perturbation_type
        self.perturbation_magnitude = perturbation_magnitude
        
        # Trial parameters
        self.steps_per_pattern = steps_per_pattern
        self.n_trials = n_trials
        
        # Storage for patterns
        self.patterns = []
        self.pattern_names = []
        
        # Adjacency matrix
        self.adjacency_matrix = None
    
    def run(self, experiments, target_position=1, pattern_names=None):
        """
        Run the temporal adjacency experiment.
        
        Args:
            experiments: List of RCFT experiment instances with different patterns
            target_position: Position in sequence to perturb (0-indexed)
            pattern_names: Optional list of pattern names
            
        Returns:
            Results dictionary
        """
        # Validate input
        if len(experiments) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} experiments, got {len(experiments)}")
            
        if target_position < 0 or target_position >= self.sequence_length:
            raise ValueError(f"Target position must be between 0 and {self.sequence_length-1}")
            
        # Store original patterns
        self.patterns = [exp.state.copy() for exp in experiments[:self.sequence_length]]
        
        # Set pattern names
        if pattern_names is None:
            self.pattern_names = [f"Pattern_{i+1}" for i in range(self.sequence_length)]
        else:
            self.pattern_names = pattern_names[:self.sequence_length]
            
        # Initialize adjacency matrix
        self.adjacency_matrix = np.zeros((self.sequence_length, self.sequence_length))
        
        # Initialize results
        self.results = {
            'trials': [],
            'recovery_scores': [],
            'adjacency_effects': []
        }
        
        # Run trials
        for trial in range(self.n_trials):
            print(f"Running trial {trial+1}/{self.n_trials}...")
            
            # Run sequence training and testing
            trial_results = self._run_trial(experiments, target_position, trial)
            
            # Store results
            self.results['trials'].append(trial)
            self.results['recovery_scores'].append(trial_results['recovery_scores'])
            self.results['adjacency_effects'].append(trial_results['adjacency_effects'])
            
            # Update adjacency matrix
            for i in range(self.sequence_length):
                for j in range(self.sequence_length):
                    self.adjacency_matrix[i, j] += trial_results['adjacency_matrix'][i, j]
                    
        # Average adjacency matrix
        self.adjacency_matrix /= self.n_trials
        self.results['adjacency_matrix'] = self.adjacency_matrix.tolist()
        
        # Calculate summary metrics
        self._calculate_summary_metrics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save results
        self.save_results()
        self.save_metrics()
        
        return self.results
        
    def _run_trial(self, experiments, target_position, trial):
        """
        Run a single trial of sequence training and testing.
        
        Args:
            experiments: List of RCFT experiment instances
            target_position: Position to perturb
            trial: Trial number
            
        Returns:
            Trial results
        """
        # Reset experiments to original patterns
        for i, pattern in enumerate(self.patterns):
            if i < len(experiments):
                experiments[i].state = pattern.copy()
                experiments[i].memory = pattern.copy()
                
        # Set up trial results
        trial_results = {
            'sequences': [],
            'recovery_scores': np.zeros(self.sequence_length),
            'adjacency_effects': np.zeros((self.sequence_length, self.sequence_length)),
            'adjacency_matrix': np.zeros((self.sequence_length, self.sequence_length))
        }
        
        # Train the sequence
        print("  Training sequence...")
        sequence_states = self._train_sequence(experiments)
        trial_results['sequences'] = sequence_states
        
        # Perturb the target pattern
        print(f"  Perturbing pattern at position {target_position}...")
        perturbed_states = self._perturb_and_test(experiments, target_position)
        trial_results['perturbed_states'] = perturbed_states
        
        # Calculate recovery scores
        for i in range(self.sequence_length):
            initial_state = self.patterns[i]
            final_state = perturbed_states[i]
            
            # Use original and current state to get recovery score
            correlation = IdentityMetrics.correlation(initial_state, final_state)
            trial_results['recovery_scores'][i] = correlation
            
        # Calculate adjacency effects
        for i in range(self.sequence_length):
            for j in range(self.sequence_length):
                if i != j:
                    effect = self._calculate_adjacency_effect(
                        perturbed_states[i], self.patterns[i], 
                        perturbed_states[j], self.patterns[j])
                        
                    trial_results['adjacency_effects'][i, j] = effect
                    
                    # Add to adjacency matrix
                    if effect > 0:
                        # Positive effect = j helps i recover
                        trial_results['adjacency_matrix'][i, j] = effect
        
        return trial_results
        
    def _train_sequence(self, experiments):
        """
        Train the sequence of patterns.
        
        Args:
            experiments: List of RCFT experiment instances
            
        Returns:
            List of final states after training
        """
        # Train each pattern in sequence
        sequence_states = []
        
        for i in range(self.sequence_length):
            # Run the experiment for this pattern
            experiments[i].update(steps=self.steps_per_pattern)
            
            # Save final state
            sequence_states.append(experiments[i].state.copy())
            
        return sequence_states
        
    def _perturb_and_test(self, experiments, target_position):
        """
        Perturb the target pattern and test recovery with adjacent patterns.
        
        Args:
            experiments: List of RCFT experiment instances
            target_position: Position to perturb
            
        Returns:
            Final states after testing
        """
        # Apply perturbation to target
        experiments[target_position].apply_perturbation(
            perturbation_type=self.perturbation_type,
            magnitude=self.perturbation_magnitude
        )
        
        # Run recovery for all patterns again
        for i in range(self.sequence_length):
            experiments[i].update(steps=self.steps_per_pattern)
            
        # Return final states
        return [exp.state.copy() for exp in experiments[:self.sequence_length]]
        
    def _calculate_adjacency_effect(self, state_i, original_i, state_j, original_j):
        """
        Calculate how much pattern j helps pattern i recover.
        
        Args:
            state_i: Current state of pattern i
            original_i: Original state of pattern i
            state_j: Current state of pattern j
            original_j: Original state of pattern j
            
        Returns:
            Adjacency effect score
        """
        # Calculate correlation of each pattern with its original
        corr_i = IdentityMetrics.correlation(state_i, original_i)
        corr_j = IdentityMetrics.correlation(state_j, original_j)
        
        # Calculate cross-correlation between i and j
        cross_corr = IdentityMetrics.correlation(state_i, state_j)
        
        # Adjacency effect: higher cross-correlation with higher recovery of j
        # suggests j is helping i recover
        adjacency_effect = cross_corr * corr_j
        
        # Normalize by i's recovery (if i recovers well on its own, 
        # j's effect might be less significant)
        if corr_i > 0.01:
            adjacency_effect /= corr_i
            
        return adjacency_effect
        
    def _calculate_summary_metrics(self):
        """Calculate summary metrics for the experiment."""
        # Average recovery scores
        recovery_scores = np.array(self.results['recovery_scores'])
        avg_recovery = np.mean(recovery_scores, axis=0)
        
        self.metrics['average_recovery'] = avg_recovery.tolist()
        self.metrics['overall_recovery'] = np.mean(avg_recovery)
        
        # Analyze adjacency matrix
        if self.adjacency_matrix is not None:
            # Calculate total adjacency effect (sum of matrix)
            total_effect = np.sum(self.adjacency_matrix)
            self.metrics['total_adjacency_effect'] = float(total_effect)
            
            # Calculate strongest effect
            max_effect = np.max(self.adjacency_matrix)
            max_i, max_j = np.unravel_index(np.argmax(self.adjacency_matrix), 
                                          self.adjacency_matrix.shape)
                                          
            self.metrics['strongest_effect'] = float(max_effect)
            self.metrics['strongest_effect_from'] = int(max_j)
            self.metrics['strongest_effect_to'] = int(max_i)
            
            # Calculate temporal bias (is forward influence stronger than backward?)
            forward_mask = np.zeros_like(self.adjacency_matrix, dtype=bool)
            backward_mask = np.zeros_like(self.adjacency_matrix, dtype=bool)
            
            for i in range(self.sequence_length):
                for j in range(self.sequence_length):
                    if i < j:  # Forward influence (earlier helps later)
                        forward_mask[i, j] = True
                    elif i > j:  # Backward influence (later helps earlier)
                        backward_mask[i, j] = True
                        
            forward_effect = np.sum(self.adjacency_matrix[forward_mask])
            backward_effect = np.sum(self.adjacency_matrix[backward_mask])
            
            self.metrics['forward_effect'] = float(forward_effect)
            self.metrics['backward_effect'] = float(backward_effect)
            self.metrics['temporal_bias'] = float(forward_effect - backward_effect)
            
            # Calculate connectivity metrics
            # - How well connected is the adjacency graph?
            nonzero_connections = np.count_nonzero(self.adjacency_matrix)
            max_connections = self.sequence_length * (self.sequence_length - 1)
            connectivity = nonzero_connections / max_connections if max_connections > 0 else 0
            
            self.metrics['connectivity'] = float(connectivity)
            
    def _create_visualizations(self):
        """Create visualizations of results."""
        # 1. Pattern sequence
        self._visualize_pattern_sequence()
        
        # 2. Recovery scores
        self._visualize_recovery_scores()
        
        # 3. Adjacency matrix
        self._visualize_adjacency_matrix()
        
        # 4. Adjacency graph
        self._visualize_adjacency_graph()
        
        # 5. Summary visualization
        self._create_summary_visualization()
        
    def _visualize_pattern_sequence(self):
        """Visualize the pattern sequence."""
        if not self.patterns:
            return
            
        # Create visualization
        self.visualizer.visualize_state_sequence(
            self.patterns, titles=self.pattern_names, title="Pattern Sequence",
            save_path=os.path.join(self.output_dir, "pattern_sequence.png")
        )
        
    def _visualize_recovery_scores(self):
        """Visualize recovery scores."""
        if not self.results['recovery_scores']:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Get data
        recovery_scores = np.array(self.results['recovery_scores'])
        
        # Calculate average and standard deviation
        avg_recovery = np.mean(recovery_scores, axis=0)
        std_recovery = np.std(recovery_scores, axis=0)
        
        # Create bar chart
        bars = plt.bar(self.pattern_names, avg_recovery, yerr=std_recovery)
        
        # Highlight target pattern
        target_position = np.argmin(avg_recovery)  # Assume target = lowest recovery
        bars[target_position].set_color('red')
        
        plt.title('Recovery Scores by Pattern')
        plt.xlabel('Pattern')
        plt.ylabel('Recovery Score')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recovery_scores.png"))
        plt.close()
        
    def _visualize_adjacency_matrix(self):
        """Visualize adjacency matrix."""
        if self.adjacency_matrix is None:
            return
            
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        im = plt.imshow(self.adjacency_matrix, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, label='Adjacency Effect')
        
        # Add labels
        plt.xticks(range(self.sequence_length), self.pattern_names)
        plt.yticks(range(self.sequence_length), self.pattern_names)
        
        plt.title('Temporal Adjacency Matrix')
        plt.xlabel('Influence From')
        plt.ylabel('Influence To')
        
        # Add grid
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, self.sequence_length), minor=True)
        ax.set_yticks(np.arange(-0.5, self.sequence_length), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        
        # Add text labels
        for i in range(self.sequence_length):
            for j in range(self.sequence_length):
                if i != j:  # Skip diagonal
                    text = plt.text(j, i, f"{self.adjacency_matrix[i, j]:.2f}",
                                  ha="center", va="center", color="w")
                                  
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "adjacency_matrix.png"))
        plt.close()
        
    def _visualize_adjacency_graph(self):
        """Visualize adjacency graph."""
        if self.adjacency_matrix is None:
            return
            
        try:
            import networkx as nx
            
            plt.figure(figsize=(10, 8))
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, name in enumerate(self.pattern_names):
                G.add_node(i, label=name)
                
            # Add edges (with weights)
            for i in range(self.sequence_length):
                for j in range(self.sequence_length):
                    if i != j and self.adjacency_matrix[i, j] > 0.01:
                        G.add_edge(j, i, weight=self.adjacency_matrix[i, j])
                        
            # Position nodes in a circle
            pos = nx.circular_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=3000, 
                                 node_color='lightblue', alpha=0.8)
                                 
            # Draw edges with width proportional to weight
            edges = G.edges(data=True)
            weights = [d['weight'] * 5 for _, _, d in edges]  # Scale for visibility
            
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights,
                                 arrowsize=20, alpha=0.7)
                                 
            # Draw labels
            nx.draw_networkx_labels(G, pos, 
                                  labels={i: name for i, name in enumerate(self.pattern_names)},
                                  font_size=12)
                                  
            # Draw edge labels
            edge_labels = {(j, i): f"{self.adjacency_matrix[i, j]:.2f}" 
                         for i in range(self.sequence_length) 
                         for j in range(self.sequence_length)
                         if i != j and self.adjacency_matrix[i, j] > 0.01}
                         
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title('Temporal Adjacency Graph')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "adjacency_graph.png"))
            plt.close()
            
        except ImportError:
            print("NetworkX not available, skipping adjacency graph visualization")
            
    def _create_summary_visualization(self):
        """Create a summary visualization with key findings."""
        plt.figure(figsize=(15, 12))
        
        # Layout
        gs = plt.GridSpec(2, 2)
        
        # 1. Recovery scores
        ax1 = plt.subplot(gs[0, 0])
        
        if self.results['recovery_scores']:
            recovery_scores = np.array(self.results['recovery_scores'])
            avg_recovery = np.mean(recovery_scores, axis=0)
            std_recovery = np.std(recovery_scores, axis=0)
            
            bars = ax1.bar(self.pattern_names, avg_recovery, yerr=std_recovery)
            
            # Highlight target pattern
            target_position = np.argmin(avg_recovery)  # Assume target = lowest recovery
            bars[target_position].set_color('red')
            
        ax1.set_title('Recovery Scores by Pattern')
        ax1.set_xlabel('Pattern')
        ax1.set_ylabel('Recovery Score')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Adjacency matrix
        ax2 = plt.subplot(gs[0, 1])
        
        if self.adjacency_matrix is not None:
            im = ax2.imshow(self.adjacency_matrix, cmap='viridis')
            plt.colorbar(im, ax=ax2, label='Adjacency Effect')
            
            ax2.set_xticks(range(self.sequence_length))
            ax2.set_yticks(range(self.sequence_length))
            ax2.set_xticklabels(self.pattern_names)
            ax2.set_yticklabels(self.pattern_names)
            
            # Add text labels
            for i in range(self.sequence_length):
                for j in range(self.sequence_length):
                    if i != j:  # Skip diagonal
                        text = ax2.text(j, i, f"{self.adjacency_matrix[i, j]:.2f}",
                                     ha="center", va="center", color="w")
                                     
        ax2.set_title('Temporal Adjacency Matrix')
        ax2.set_xlabel('Influence From')
        ax2.set_ylabel('Influence To')
        
        # 3. Temporal bias
        ax3 = plt.subplot(gs[1, 0])
        
        if 'forward_effect' in self.metrics and 'backward_effect' in self.metrics:
            forward = self.metrics['forward_effect']
            backward = self.metrics['backward_effect']
            
            bars = ax3.bar(['Forward\n(Earlier → Later)', 'Backward\n(Later → Earlier)'], 
                         [forward, backward])
                         
            # Add text labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height * 0.5,
                       f"{height:.3f}", ha='center', va='center',
                       color='white', fontweight='bold')
                       
        ax3.set_title('Temporal Direction Bias')
        ax3.set_ylabel('Adjacency Effect')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Key metrics
        ax4 = plt.subplot(gs[1, 1])
        
        metrics_to_show = ['total_adjacency_effect', 'strongest_effect', 'connectivity']
        metric_labels = ['Total\nAdjacency Effect', 'Strongest\nEffect', 'Connectivity']
        
        values = []
        for metric in metrics_to_show:
            if metric in self.metrics:
                values.append(self.metrics[metric])
                
        if values:
            # Normalize for better visualization
            values_norm = np.array(values)
            max_val = np.max(np.abs(values_norm))
            if max_val > 0:
                values_norm = values_norm / max_val
                
            bars = ax4.bar(metric_labels, values_norm)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f"{value:.3f}", ha='center', va='bottom')
                       
        ax4.set_title('Key Metrics (Normalized)')
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Temporal Adjacency Analysis Summary\n' + 
                   f'Sequence Length={self.sequence_length}, Steps Per Pattern={self.steps_per_pattern}',
                   fontsize=16)
                   
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary.png"))
        plt.close()
        
    def save_metrics(self, metrics=None):
        """
        Save metrics to CSV file.
        
        Args:
            metrics: Optional metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
            
        # Flatten metrics for CSV
        flat_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    flat_metrics[f"{key}_{i+1}"] = v
            else:
                flat_metrics[key] = value
                
        # Create CSV file
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(flat_metrics.keys())
            
            # Write values
            writer.writerow(flat_metrics.values())
            
        # Also save in JSON format for completeness
        super().save_metrics(metrics)