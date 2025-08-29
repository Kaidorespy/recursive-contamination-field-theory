import numpy as np
import matplotlib.pyplot as plt
import os
from rcft_metrics import compute_ccdi
import pandas as pd
import seaborn as sns
import json

class ContextSwitcher:
    """Attempts to trigger different attractors based on context cues"""
    
    def __init__(self, phase6):
        """
        Initialize the context switcher
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
        self.output_dir = os.path.join(phase6.output_dir, "switching")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_context_switching_experiment(self, pattern_ids, cue_strengths, context_variations):
        """
        Run context switching experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to switch between
        cue_strengths : list
            List of cue strengths to test (0.0 to 1.0)
        context_variations : list
            List of context variations to test (spatial, frequency, gradient)
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        if len(pattern_ids) < 2:
            raise ValueError("Need at least two patterns for context switching")
            
        results = []
        
        # Get patterns
        patterns = []
        for pid in pattern_ids:
            if pid not in self.phase6.memory_bank:
                raise ValueError(f"Pattern {pid} not found in memory bank")
            patterns.append(self.phase6.memory_bank[pid])
            
        # Train each pattern to create stable attractors
        self.logger.info("Stabilizing individual pattern attractors")
        for pattern in patterns:
            self._stabilize_pattern(pattern)
            
        # For each context variation
        for variation in context_variations:
            # For each cue strength
            for strength in cue_strengths:
                # For each target pattern
                for i, target in enumerate(patterns):
                    # Try to switch from all other patterns to this target
                    for j, source in enumerate(patterns):
                        if i == j:
                            continue  # Skip same pattern
                            
                        # Run switching test
                        switching_result = self._test_context_switch(
                            source, target, variation, strength
                        )
                        
                        # Add experiment parameters
                        switching_result.update({
                            'source_pattern': source.pattern_id,
                            'target_pattern': target.pattern_id,
                            'context_variation': variation,
                            'cue_strength': strength
                        })
                        
                        results.append(switching_result)
                        
                        self.logger.info(f"Ran context switch: {source.pattern_id} → {target.pattern_id}, " +
                                      f"Variation={variation}, Strength={strength:.2f}, " +
                                      f"Success={switching_result.get('switch_success', False)}, " +
                                      f"Latency={switching_result.get('switching_latency', -1)}")
        
        # Save all results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "context_switching_results.csv"), index=False)
        
        # Create summary visualization
        self._visualize_switching_summary(results_df)
        
        # Create context transition graph
        context_graph = self.build_context_transition_graph(results_df)
        
        return results
    
    def _stabilize_pattern(self, pattern, steps=50):
        """
        Stabilize a pattern to form an attractor
        
        Parameters:
        -----------
        pattern : MemoryTrace
            Pattern to stabilize
        steps : int
            Number of steps to evolve
            
        Returns:
        --------
        dict
            Stabilization metrics
        """
        # Initialize experiment with pattern
        exp = self.phase6.base_experiment
        exp.state = pattern.initial_state.copy()
        
        # Let system evolve to form attractor
        exp.update(steps=steps)
        
        # Capture final state and metrics
        final_state = exp.state.copy()
        
        # Calculate memory metrics
        metrics = {
            'correlation': exp.metrics['correlation'][-1],
            'coherence': exp.metrics['coherence'][-1],
            'mutual_info': exp.metrics['mutual_info'][-1],
            'spectral_entropy': exp.metrics['spectral_entropy'][-1],
            'ccdi': compute_ccdi(exp.metrics['correlation'][-1], exp.metrics['coherence'][-1])
        }
        
        # Update memory trace
        pattern.final_state = final_state
        pattern.metrics = metrics
        
        # Save stabilized attractor
        np.save(os.path.join(self.output_dir, f"{pattern.pattern_id}_stabilized.npy"), final_state)
        
        return metrics
    
    def _test_context_switch(self, source_pattern, target_pattern, 
                          context_variation, cue_strength, steps=100):
        """
        Test switching from one pattern to another with context cue
        
        Parameters:
        -----------
        source_pattern : MemoryTrace
            Starting pattern
        target_pattern : MemoryTrace
            Target pattern to switch to
        context_variation : str
            Type of context variation to apply
        cue_strength : float
            Strength of context cue (0.0 to 1.0)
        steps : int
            Number of steps to evolve
            
        Returns:
        --------
        dict
            Context switching metrics
        """
        # Initialize experiment with source pattern's attractor state
        exp = self.phase6.base_experiment
        if source_pattern.final_state is not None:
            exp.state = source_pattern.final_state.copy()
        else:
            exp.state = source_pattern.initial_state.copy()
            
        # Save initial state
        initial_state = exp.state.copy()
        
        # Apply context cue (partial hint of target pattern)
        cued_state = self._apply_context_cue(
            initial_state, 
            target_pattern.initial_state, 
            context_variation, 
            cue_strength
        )
        
        # Use the cued state
        exp.state = cued_state.copy()
        
        # Track states over time for analysis
        states = [cued_state.copy()]
        correlations = []
        
        # Run system and track correlations to source and target
        source_correlations = []
        target_correlations = []
        
        for step in range(steps):
            # Update system for one step
            exp.update(steps=1)
            
            # Save state
            states.append(exp.state.copy())
            
            # Calculate correlations
            source_corr = np.corrcoef(source_pattern.initial_state.flatten(), 
                                    exp.state.flatten())[0, 1]
            target_corr = np.corrcoef(target_pattern.initial_state.flatten(), 
                                    exp.state.flatten())[0, 1]
            
            source_correlations.append(source_corr)
            target_correlations.append(target_corr)
            
            # Check for convergence
            correlations.append({'source': source_corr, 'target': target_corr, 'step': step})
        
        # Determine if switch was successful
        # Criteria: final state is closer to target than to source
        final_source_corr = source_correlations[-1]
        final_target_corr = target_correlations[-1]
        
        switch_success = final_target_corr > final_source_corr
        
        # Calculate switching latency (steps until target correlation exceeds source)
        switching_latency = -1  # Default if never switches
        
        for i, (s_corr, t_corr) in enumerate(zip(source_correlations, target_correlations)):
            if t_corr > s_corr:
                switching_latency = i
                break
                
        # Calculate transition sharpness
        # (how quickly it switches - steepness of crossover)
        transition_sharpness = 0.0
        
        if switching_latency >= 0 and switching_latency < len(source_correlations) - 1:
            # Calculate rate of change at crossover
            s_delta = source_correlations[switching_latency+1] - source_correlations[switching_latency-1]
            t_delta = target_correlations[switching_latency+1] - target_correlations[switching_latency-1]
            transition_sharpness = abs(t_delta - s_delta)
            
        # Calculate bias ratio - how strongly biased toward final state
        bias_ratio = final_target_corr / final_source_corr if final_source_corr > 0 else float('inf')
        
        # Save states for visualization
        np.save(os.path.join(self.output_dir, 
                          f"{source_pattern.pattern_id}_to_{target_pattern.pattern_id}_{context_variation}_{cue_strength:.2f}_initial.npy"), 
              initial_state)
              
        np.save(os.path.join(self.output_dir, 
                          f"{source_pattern.pattern_id}_to_{target_pattern.pattern_id}_{context_variation}_{cue_strength:.2f}_cued.npy"), 
              cued_state)
              
        np.save(os.path.join(self.output_dir, 
                          f"{source_pattern.pattern_id}_to_{target_pattern.pattern_id}_{context_variation}_{cue_strength:.2f}_final.npy"), 
              states[-1])
        
        # Create visualization
        self._visualize_context_switch(
            source_pattern.pattern_id, target_pattern.pattern_id,
            context_variation, cue_strength,
            initial_state, cued_state, states[-1],
            source_correlations, target_correlations,
            switch_success, switching_latency, bias_ratio
        )
        
        # Return results
        result = {
            'switch_success': switch_success,
            'switching_latency': switching_latency,
            'transition_sharpness': transition_sharpness,
            'final_source_correlation': final_source_corr,
            'final_target_correlation': final_target_corr,
            'bias_ratio': bias_ratio,
            'correlation_series': correlations  # Full correlation series
        }
        
        return result
    
    def _apply_context_cue(self, source_state, target_state, 
                         context_variation, cue_strength):
        """
        Apply a context cue to the source state
        
        Parameters:
        -----------
        source_state : ndarray
            Current state
        target_state : ndarray
            Target pattern to hint at
        context_variation : str
            Type of context cue: "spatial", "frequency", "gradient"
        cue_strength : float
            Strength of cue (0.0 to 1.0)
            
        Returns:
        --------
        ndarray
            State with context cue applied
        """
        # Make a copy of source state
        result = source_state.copy()
        
        if context_variation == "spatial":
            # Apply target pattern in a limited spatial region
            mask = np.zeros_like(source_state, dtype=bool)
            
            # Create circular region of influence
            center = (source_state.shape[0] // 2, source_state.shape[1] // 2)
            radius = int(cue_strength * source_state.shape[0] // 2)  # Size scales with cue strength
            
            x = np.arange(source_state.shape[0])
            y = np.arange(source_state.shape[1])
            X, Y = np.meshgrid(x, y)
            mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2)
            
            # Apply mask
            result[mask] = target_state[mask]
            
        elif context_variation == "frequency":
            # Apply cue in frequency domain (hint at frequency structure)
            source_fft = np.fft.fft2(source_state)
            target_fft = np.fft.fft2(target_state)
            
            # Create mask that focuses on low or high frequencies
            freq_mask = np.zeros_like(source_state, dtype=bool)
            center = (source_state.shape[0] // 2, source_state.shape[1] // 2)
            
            # Focus on low frequencies (central region of FFT)
            radius = int((1.0 - cue_strength) * source_state.shape[0] // 2)  # Smaller radius with stronger cue
            
            x = np.arange(source_state.shape[0])
            y = np.arange(source_state.shape[1])
            X, Y = np.meshgrid(x, y)
            freq_mask = np.fft.fftshift(((X - center[0])**2 + (Y - center[1])**2 <= radius**2))
            
            # Create blended FFT
            blended_fft = np.zeros_like(source_fft, dtype=complex)
            blended_fft[freq_mask] = source_fft[freq_mask]  # Keep source at masked frequencies
            blended_fft[~freq_mask] = target_fft[~freq_mask]  # Use target elsewhere
            
            # Convert back to spatial domain
            result = np.real(np.fft.ifft2(blended_fft))
            
            # Normalize
            result_min = np.min(result)
            result_max = np.max(result)
            if result_max > result_min:  # Avoid division by zero
                result = 2.0 * (result - result_min) / (result_max - result_min) - 1.0
            
        elif context_variation == "gradient":
            # Create a gradient blend between source and target
            x = np.linspace(0, 1, source_state.shape[0])
            y = np.linspace(0, 1, source_state.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Create gradient mask (increases from left to right)
            mask = X.copy()  # Linear gradient from 0 to 1
            
            # Scale gradient based on cue strength
            # Higher cue_strength means more of the target is visible
            mask = mask * cue_strength
            
            # Blend source and target using mask
            result = (1.0 - mask) * source_state + mask * target_state
            
        else:
            raise ValueError(f"Unknown context variation: {context_variation}")
            
        return result
    
    def build_context_transition_graph(self, results_df):
        """
        Build a graph of context transitions based on results
        
        Parameters:
        -----------
        results_df : DataFrame
            Results dataframe from context switching experiment
            
        Returns:
        --------
        dict
            Context transition graph
        """
        # Build a directed graph of transitions
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Create unique list of patterns (nodes)
        patterns = set()
        for p in results_df['source_pattern'].unique():
            patterns.add(p)
        for p in results_df['target_pattern'].unique():
            patterns.add(p)
            
        # Create node for each pattern
        for pattern in patterns:
            graph['nodes'].append({
                'id': pattern,
                'label': pattern
            })
            
        # Create edges for successful transitions
        successful = results_df[results_df['switch_success'] == True]
        
        # Group by source and target to calculate success rate
        transitions = successful.groupby(['source_pattern', 'target_pattern']).size().reset_index(name='count')
        
        # Calculate total attempts for each transition
        total_attempts = results_df.groupby(['source_pattern', 'target_pattern']).size().reset_index(name='total')
        
        # Merge to calculate success rate
        transitions = transitions.merge(total_attempts, on=['source_pattern', 'target_pattern'])
        transitions['success_rate'] = transitions['count'] / transitions['total']
        
        # Add edges
        for _, row in transitions.iterrows():
            # Calculate average latency for this transition
            latency_data = successful[
                (successful['source_pattern'] == row['source_pattern']) & 
                (successful['target_pattern'] == row['target_pattern'])
            ]
            avg_latency = latency_data['switching_latency'].mean()
            
            graph['edges'].append({
                'source': row['source_pattern'],
                'target': row['target_pattern'],
                'success_rate': float(row['success_rate']),
                'count': int(row['count']),
                'total': int(row['total']),
                'avg_latency': float(avg_latency)
            })
            
        # Save graph to JSON
        with open(os.path.join(self.output_dir, "context_transition_graph.json"), 'w') as f:
            json.dump(graph, f, indent=2)
            
        # Create graph visualization
        self._visualize_context_graph(graph)
        
        return graph
    
    def _visualize_context_switch(self, source_id, target_id, context_variation, cue_strength,
                               initial_state, cued_state, final_state, 
                               source_correlations, target_correlations,
                               switch_success, switching_latency, bias_ratio):
        """Visualize context switching experiment"""
        plt.figure(figsize=(15, 10))
        
        # Plot states
        plt.subplot(2, 3, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial State ({source_id})")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(cued_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After Context Cue\n{context_variation.title()} ({cue_strength:.2f})")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Final State\nSuccess: {'Yes' if switch_success else 'No'}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot correlation time series
        plt.subplot(2, 1, 2)
        x = range(len(source_correlations))
        plt.plot(x, source_correlations, 'b-', label=f'Correlation to {source_id}')
        plt.plot(x, target_correlations, 'g-', label=f'Correlation to {target_id}')
        
        # Mark switching point if successful
        if switching_latency >= 0:
            plt.axvline(x=switching_latency, color='r', linestyle='--', 
                      label=f'Switch Point (Step {switching_latency})')
            
        plt.title(f"Correlation Time Series: {source_id} → {target_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add switch parameters as text
        status_color = 'green' if switch_success else 'red'
        
        plt.figtext(0.01, 0.01, 
                  f"Success: {'Yes' if switch_success else 'No'} | " +
                  f"Latency: {switching_latency if switching_latency >= 0 else 'N/A'} | " +
                  f"Bias Ratio: {bias_ratio:.2f} | " + 
                  f"Final Target Corr: {target_correlations[-1]:.3f} | " +
                  f"Final Source Corr: {source_correlations[-1]:.3f}",
                  fontsize=10, color=status_color, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                              f"{source_id}_to_{target_id}_{context_variation}_{cue_strength:.2f}.png"))
        plt.close()
    
    def _visualize_context_graph(self, graph):
        """Create visualization of context transition graph"""
        # This is simplified since we can't use networkx in this environment
        plt.figure(figsize=(10, 8))
        
        # Extract node positions (simple circular layout)
        nodes = graph['nodes']
        n = len(nodes)
        
        # Create a circular layout
        node_positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            x = 0.4 * np.cos(angle)
            y = 0.4 * np.sin(angle)
            node_positions[node['id']] = (x, y)
        
        # Plot nodes
        for node in nodes:
            x, y = node_positions[node['id']]
            plt.scatter(x, y, s=500, color='skyblue')
            plt.text(x, y, node['id'], ha='center', va='center', fontsize=12)
            
        # Plot edges
        max_width = 5.0
        min_width = 0.5
        
        for edge in graph['edges']:
            source_pos = node_positions[edge['source']]
            target_pos = node_positions[edge['target']]
            
            # Calculate arrow properties based on success rate
            success_rate = edge['success_rate']
            width = min_width + (max_width - min_width) * success_rate
            
            # Determine color based on success rate
            if success_rate > 0.8:
                color = 'green'
            elif success_rate > 0.5:
                color = 'blue'
            elif success_rate > 0.2:
                color = 'orange'
            else:
                color = 'red'
                
            # Draw arrow
            dx = target_pos[0] - source_pos[0]
            dy = target_pos[1] - source_pos[1]
            
            plt.arrow(source_pos[0], source_pos[1], dx*0.8, dy*0.8, 
                    width=width/100, head_width=width/20, 
                    length_includes_head=True, color=color, alpha=0.7)
            
            # Add edge label with success rate
            mid_x = source_pos[0] + dx/2
            mid_y = source_pos[1] + dy/2
            
            plt.text(mid_x, mid_y, f"{success_rate:.2f}", 
                   color=color, fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
            
        plt.title("Context Transition Graph")
        plt.axis('equal')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "context_transition_graph.png"))
        plt.close()
        
        # Create heatmap showing transition success rates
        transition_matrix = np.zeros((n, n))
        matrix_labels = [node['id'] for node in nodes]
        
        # Fill transition matrix
        for edge in graph['edges']:
            source_idx = matrix_labels.index(edge['source'])
            target_idx = matrix_labels.index(edge['target'])
            transition_matrix[source_idx, target_idx] = edge['success_rate']
            
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', 
                  xticklabels=matrix_labels, yticklabels=matrix_labels)
        plt.title("Context Transition Success Rate")
        plt.xlabel("Target Pattern")
        plt.ylabel("Source Pattern")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "transition_matrix.png"))
        plt.close()
    
    def _visualize_switching_summary(self, results_df):
        """Create summary visualization for context switching experiments"""
        plt.figure(figsize=(15, 10))
        
        # Plot success rate vs cue strength
        plt.subplot(2, 2, 1)
        
        # Group by context variation
        for variation in results_df['context_variation'].unique():
            var_data = results_df[results_df['context_variation'] == variation]
            
            # Group by cue strength and calculate success rate
            strength_groups = var_data.groupby('cue_strength')
            success_rates = strength_groups['switch_success'].mean()
            
            plt.plot(success_rates.index, success_rates.values, 'o-', label=variation)
        
        plt.title('Switch Success Rate vs Cue Strength')
        plt.xlabel('Cue Strength')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot switching latency vs cue strength
        plt.subplot(2, 2, 2)
        
        # Only use successful switches for latency
        successful = results_df[results_df['switch_success'] == True]
        
        # Group by context variation
        for variation in successful['context_variation'].unique():
            var_data = successful[successful['context_variation'] == variation]
            
            # Group by cue strength and calculate mean latency
            strength_groups = var_data.groupby('cue_strength')
            mean_latency = strength_groups['switching_latency'].mean()
            
            plt.plot(mean_latency.index, mean_latency.values, 'o-', label=variation)
        
        plt.title('Switching Latency vs Cue Strength')
        plt.xlabel('Cue Strength')
        plt.ylabel('Latency (steps)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success rate by transition
        plt.subplot(2, 2, 3)
        
        # Group by source and target
        transition_groups = results_df.groupby(['source_pattern', 'target_pattern'])
        transition_rates = transition_groups['switch_success'].mean().reset_index()
        
        # Create labels for transitions
        transition_labels = [f"{row['source_pattern']}→{row['target_pattern']}" 
                          for _, row in transition_rates.iterrows()]
        
        plt.bar(transition_labels, transition_rates['switch_success'])
        plt.title('Success Rate by Transition')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot bias ratio distribution
        plt.subplot(2, 2, 4)
        
        # Create bins for bias ratio
        bins = [0, 1, 2, 5, 10, float('inf')]
        labels = ['0-1', '1-2', '2-5', '5-10', '>10']
        
        # Calculate bin counts
        binned_ratios = pd.cut(results_df['bias_ratio'], bins=bins, labels=labels)
        bin_counts = binned_ratios.value_counts().sort_index()
        
        plt.bar(bin_counts.index, bin_counts.values)
        plt.title('Reactivation Bias Ratio Distribution')
        plt.xlabel('Bias Ratio')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "switching_summary.png"))
        plt.close()