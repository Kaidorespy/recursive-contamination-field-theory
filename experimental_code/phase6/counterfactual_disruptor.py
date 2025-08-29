import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from rcft_metrics import compute_ccdi
from phase6.echo_buffer import EchoBuffer

class CounterfactualDisruptor:
    """Injects a false memory during or after encoding to observe the effects"""
    
    def __init__(self, phase6, echo_depth=0):
        """
        Initialize the counterfactual disruptor
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        echo_depth : int, optional
            Depth of echo buffer protection (0 = none, default)
        """
        self.phase6 = phase6
        self.logger = phase6.logger
        self.output_dir = os.path.join(phase6.output_dir, "counterfactual")
        os.makedirs(self.output_dir, exist_ok=True)
        self.echo_buffer = EchoBuffer(depth=echo_depth)
    
    # Replace the entire run_counterfactual_experiment method in counterfactual_disruptor.py:

    def run_counterfactual_experiment(self, pattern_ids, counterfactual_id='CF', 
                                    counterfactual_similarity=None, intrusion_strengths=None,
                                    trials=1, repetitions=1, injection_delay=0):
        """
        Run counterfactual intrusion experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to encode
        counterfactual_id : str
            ID of counterfactual pattern to inject
        counterfactual_similarity : list
            List of similarity levels to test (0.0 to 1.0)
        intrusion_strengths : list
            List of intrusion strengths to test
        trials : int
            Number of trials to run for each parameter combination
        repetitions : int
            Number of sequential counterfactual exposures to apply
        injection_delay : int
            Number of steps to wait after encoding before injecting counterfactual
                
        Returns:
        --------
        list
            List of result dictionaries
        """
        results = []
        
        # Make sure patterns exist
        for pid in pattern_ids:
            if pid not in self.phase6.memory_bank:
                raise ValueError(f"Pattern {pid} not found in memory bank")
        
        # For each target pattern, test counterfactual intrusion
        for target_pattern_id in pattern_ids:
            # Create experiment combinations
            for similarity in counterfactual_similarity:
                for trial in range(trials):
                    # Create counterfactual based on target pattern with a unique ID for each trial
                    cf_unique_id = f"{counterfactual_id}_{trial}" if trials > 1 else counterfactual_id
                    cf_trace = self.create_counterfactual(target_pattern_id, cf_unique_id, similarity)
                    
                    for intrusion in intrusion_strengths:
                        # Run the intrusion test with delay parameter
                        intrusion_result = self._run_intrusion_test(
                            target_pattern_id,  # This is the target pattern ID
                            cf_unique_id,       # This is the counterfactual ID
                            intrusion,          # Intrusion strength
                            cf_trace,           # Counterfactual trace
                            repetitions,        # Number of repetitions
                            injection_delay     # Delay before injection (new parameter)
                        )
                        
                        # Add experiment parameters
                        intrusion_result.update({
                            'target_pattern': target_pattern_id,
                            'counterfactual_id': cf_unique_id,
                            'similarity': similarity,
                            'intrusion_strength': intrusion,
                            'trial': trial,
                            'repetitions': repetitions,
                            'injection_delay': injection_delay
                        })
                        
                        results.append(intrusion_result)
                        
                        self.logger.info(f"Ran counterfactual intrusion: Target={target_pattern_id}, " +
                                    f"CF={cf_unique_id}, Similarity={similarity:.2f}, " +
                                    f"Strength={intrusion}, Trial={trial}, " +
                                    f"Delay={injection_delay}, " +
                                    f"Corruption={intrusion_result.get('memory_integrity_delta', 0):.3f}")
        
        # Save all results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "counterfactual_results.csv"), index=False)
        
        # Create summary visualization
        self._visualize_counterfactual_summary(results_df)
        
        return results
    
    def create_counterfactual(self, based_on_pattern_id, counterfactual_id, similarity_level=0.5):
        """
        Create a counterfactual pattern based on an existing one
        
        Parameters:
        -----------
        based_on_pattern_id : str
            ID of pattern to base counterfactual on
        counterfactual_id : str
            ID to assign to the counterfactual
        similarity_level : float
            How similar to make the counterfactual (0.0 to 1.0)
            
        Returns:
        --------
        MemoryTrace
            The counterfactual memory trace
        """
        # Get the base pattern
        if based_on_pattern_id not in self.phase6.memory_bank:
            raise ValueError(f"Base pattern {based_on_pattern_id} not found in memory bank")
            
        base_pattern = self.phase6.memory_bank[based_on_pattern_id]
        
        # Generate a new random pattern
        exp = self.phase6.base_experiment
        exp.initialize_pattern(pattern_type="stochastic")
        random_pattern = exp.state.copy()
        
        # Mix the base pattern and random pattern based on similarity level
        # similarity_level = 1.0 means identical to base pattern
        # similarity_level = 0.0 means completely random
        
        mixed_pattern = (similarity_level * base_pattern.initial_state + 
                    (1.0 - similarity_level) * random_pattern)
        
        # Normalize the mixed pattern to [-1, 1]
        pattern_min = np.min(mixed_pattern)
        pattern_max = np.max(mixed_pattern)
        normalized_pattern = mixed_pattern  # Default in case normalization isn't needed
        
        if pattern_max > pattern_min:  # Avoid division by zero
            normalized_pattern = 2.0 * (mixed_pattern - pattern_min) / (pattern_max - pattern_min) - 1.0
        
        # Get the MemoryTrace class from the main Phase6 object
        MemoryTrace = self.phase6.memory_bank[based_on_pattern_id].__class__
        
        # Create a memory trace for the counterfactual
        cf_trace = MemoryTrace(counterfactual_id, normalized_pattern)
        
        # Add to memory bank
        self.phase6.memory_bank[counterfactual_id] = cf_trace
        
        # Save counterfactual pattern
        np.save(os.path.join(self.output_dir, f"{counterfactual_id}_sim{similarity_level:.2f}.npy"), 
            normalized_pattern)
        
        # Create visualization comparing base and counterfactual
        self._visualize_counterfactual_creation(base_pattern, cf_trace, similarity_level)
        
        return cf_trace
    
    def _visualize_intrusion_test_with_repetitions(self, target_id, counterfactual_id, intrusion_strength, 
                                            target_pattern, cf_pattern, initial_state, final_state,
                                            memory_integrity_delta, cf_influence, parasite_ratio,
                                            repetitions, corruption_trend):
        """Create visualization for intrusion test with repetitions"""
        plt.figure(figsize=(15, 10))
        
        # Plot target pattern
        plt.subplot(2, 3, 1)
        plt.imshow(target_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Target Pattern: {target_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot counterfactual pattern
        plt.subplot(2, 3, 2)
        plt.imshow(cf_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {counterfactual_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot initial state
        plt.subplot(2, 3, 3)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial State")
        plt.colorbar()
        plt.axis('off')
        
        # Plot final state
        plt.subplot(2, 3, 4)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After {repetitions} Exposures")
        plt.colorbar()
        plt.axis('off')
        
        # Plot corruption trend
        plt.subplot(2, 3, 5)
        plt.plot(range(1, len(corruption_trend)+1), corruption_trend, 'o-')
        plt.title(f"Corruption Trend")
        plt.xlabel("Repetition")
        plt.ylabel("Corruption Level")
        plt.grid(True, alpha=0.3)
        
        # Add summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Color coding for integrity impact
        if memory_integrity_delta < -0.3:
            integrity_color = 'red'
        elif memory_integrity_delta < -0.1:
            integrity_color = 'orange'
        else:
            integrity_color = 'green'
                
        # Color coding for parasite formation
        if parasite_ratio > 1.0:
            parasite_color = 'red'
        elif parasite_ratio > 0.5:
            parasite_color = 'orange'
        else:
            parasite_color = 'green'
        
        # Calculate contamination velocity
        contamination_velocity = 0.0
        if len(corruption_trend) > 1:
            contamination_velocity = (corruption_trend[-1] - corruption_trend[0]) / len(corruption_trend)
        
        summary_text = [
            f"Repetitions: {repetitions}",
            f"Intrusion Strength: {intrusion_strength:.2f}",
            f"",
            f"Memory Integrity Δ: {memory_integrity_delta:.3f}",
            f"CF Influence: {cf_influence:.3f}",
            f"Parasite Ratio: {parasite_ratio:.3f}",
            f"",
            f"Contamination Velocity: {contamination_velocity:.3f}"
        ]

        # Plot the text
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)

        # Add colored indicators separately
        plt.text(0.03, 0.6, "●", fontsize=18, color=integrity_color)  # Colored dot for integrity
        plt.text(0.03, 0.5, "●", fontsize=18, color=parasite_color)   # Colored dot for parasite ratio
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                            f"{target_id}_cf{counterfactual_id}_intr{intrusion_strength:.2f}_rep{repetitions}.png"))
        plt.close()
    def _run_intrusion_test(self, target_id, counterfactual_id, intrusion_strength, cf_trace=None, repetitions=1, injection_delay=0):
        """
        Run a single intrusion test
        
        Parameters:
        -----------
        target_id : str
            ID of target pattern to corrupt
        counterfactual_id : str
            ID of counterfactual pattern
        intrusion_strength : float
            Strength of intrusion (0.0 to 1.0)
        cf_trace : MemoryTrace, optional
            Counterfactual trace if already created
        repetitions : int
            Number of repetitions of exposure
        injection_delay : int
            Number of steps to wait after encoding before injecting counterfactual
                
        Returns:
        --------
        dict
            Intrusion test results
        """
        try:
            # Get the target pattern
            target_pattern = self.phase6.memory_bank[target_id]
            
            # Get or create counterfactual pattern
            if cf_trace is None:
                if counterfactual_id not in self.phase6.memory_bank:
                    raise ValueError(f"Counterfactual {counterfactual_id} not found in memory bank")
                cf_pattern = self.phase6.memory_bank[counterfactual_id]
            else:
                cf_pattern = cf_trace
            
            # Setup experiment
            exp = self.phase6.base_experiment
            
            # First, encode the target pattern
            exp.state = target_pattern.initial_state.copy()
            
            # Allow pattern to stabilize initially
            exp.update(steps=20)  # Basic stabilization
            
            # Initial state after basic stabilization
            initial_stable_state = exp.state.copy()
            
            # Allow memory to consolidate based on delay parameter
            if injection_delay > 0:
                self.logger.info(f"Waiting {injection_delay} steps before counterfactual injection...")
                exp.update(steps=injection_delay)
            
            # Save state before counterfactual injection
            pre_injection_state = exp.state.copy()
            
            # Calculate memory stabilization - how much the pattern changed during delay
            if injection_delay > 0:
                stabilization_delta = (
                    np.corrcoef(initial_stable_state.flatten(), pre_injection_state.flatten())[0, 1] - 1.0
                )
            else:
                stabilization_delta = 0.0
            
            # Track corruption over repetitions
            corruption_trend = []
            
            # Apply sequential counterfactual exposures
            for rep in range(repetitions):
                # Measure target pattern metrics before intrusion
                target_before = {}
                target_before['state'] = exp.state.copy()
                target_before['correlation'] = exp.metrics['correlation'][-1] if 'correlation' in exp.metrics and exp.metrics['correlation'] else 0.0
                target_before['coherence'] = exp.metrics['coherence'][-1] if 'coherence' in exp.metrics and exp.metrics['coherence'] else 0.0
                target_before['mutual_info'] = exp.metrics['mutual_info'][-1] if 'mutual_info' in exp.metrics and exp.metrics['mutual_info'] else 0.0
                target_before['ccdi'] = compute_ccdi(target_before['correlation'], target_before['coherence'])
                
                # Now inject the counterfactual - weighted by intrusion strength
                if hasattr(self, 'echo_buffer'):
                    # Apply echo buffer protection
                    injected_state = self.echo_buffer.apply_protection(
                        target_before['state'],
                        cf_pattern.initial_state,
                        intrusion_strength
                    )
                    
                    # Add state to echo buffer for future protection
                    self.echo_buffer.add_state(target_before['state'])
                else:
                    # Original code without protection
                    injected_state = ((1.0 - intrusion_strength) * target_before['state'] + 
                                intrusion_strength * cf_pattern.initial_state)
                
                # Normalize the mixed state
                state_min = np.min(injected_state)
                state_max = np.max(injected_state)
                if state_max > state_min:  # Avoid division by zero
                    normalized_state = 2.0 * (injected_state - state_min) / (state_max - state_min) - 1.0
                else:
                    normalized_state = injected_state
                
                # Use the injected state
                exp.state = normalized_state.copy()
                
                # Let the system evolve to settle after intrusion
                if rep < repetitions - 1:
                    # For intermediate repetitions, shorter stabilization
                    exp.update(steps=10)
                else:
                    # Final repetition gets full stabilization
                    exp.update(steps=50)
                
                # Calculate corruption for this repetition
                rep_corruption = 1.0 - np.corrcoef(target_pattern.initial_state.flatten(), 
                                                exp.state.flatten())[0, 1]
                corruption_trend.append(rep_corruption)
            
            # Measure target pattern metrics after intrusion
            target_after = {}
            target_after['state'] = exp.state.copy()
            target_after['correlation'] = exp.metrics['correlation'][-1] if 'correlation' in exp.metrics and exp.metrics['correlation'] else 0.0
            target_after['coherence'] = exp.metrics['coherence'][-1] if 'coherence' in exp.metrics and exp.metrics['coherence'] else 0.0
            target_after['mutual_info'] = exp.metrics['mutual_info'][-1] if 'mutual_info' in exp.metrics and exp.metrics['mutual_info'] else 0.0
            target_after['ccdi'] = compute_ccdi(target_after['correlation'], target_after['coherence'])
            
            # Calculate memory integrity delta - how much was corrupted
            memory_integrity_delta = (
                np.corrcoef(target_pattern.initial_state.flatten(), target_after['state'].flatten())[0, 1] -
                np.corrcoef(target_pattern.initial_state.flatten(), pre_injection_state.flatten())[0, 1]
            )
            
            # Calculate counterfactual influence - how much of CF present in final state
            cf_influence = np.corrcoef(cf_pattern.initial_state.flatten(), 
                                    target_after['state'].flatten())[0, 1]
            
            # Calculate parasite formation - ratio of CF to target in final state
            target_final_corr = np.corrcoef(target_pattern.initial_state.flatten(), 
                                            target_after['state'].flatten())[0, 1]
            
            parasite_ratio = 0.0
            if target_final_corr > 0:
                parasite_ratio = cf_influence / target_final_corr
                
            # Calculate overwriting rate - how quickly the memory was corrupted
            overwriting_rate = -memory_integrity_delta / intrusion_strength if intrusion_strength > 0 else 0
            
            # Calculate contamination velocity - rate of corruption across repetitions
            contamination_velocity = 0.0
            if len(corruption_trend) > 1:
                contamination_velocity = (corruption_trend[-1] - corruption_trend[0]) / len(corruption_trend)
            
            # Calculate resilience ratio - inverse of corruption effectiveness based on delay
            # Higher values mean more resilience (less corruption per unit of intrusion strength)
            resilience_ratio = 0.0
            if abs(memory_integrity_delta) > 0.001:
                resilience_ratio = 1.0 / abs(memory_integrity_delta * intrusion_strength)
            else:
                resilience_ratio = 100.0  # Very high resilience for negligible corruption
            
            # Try to save states for analysis (with error handling)
            try:
                np.save(os.path.join(self.output_dir, f"{target_id}_cf{counterfactual_id}_intr{intrusion_strength:.2f}_delay{injection_delay}_before.npy"), 
                    pre_injection_state)
                np.save(os.path.join(self.output_dir, f"{target_id}_cf{counterfactual_id}_intr{intrusion_strength:.2f}_delay{injection_delay}_after.npy"), 
                    target_after['state'])
            except Exception as e:
                self.logger.error(f"Error saving state files: {e}")
            
            # Create visualization with delay info (with error handling)
            try:
                self._visualize_intrusion_test_with_delay(target_id, counterfactual_id, intrusion_strength, 
                                                        target_pattern, cf_pattern, 
                                                        initial_stable_state, pre_injection_state, target_after['state'], 
                                                        memory_integrity_delta, cf_influence, parasite_ratio,
                                                        injection_delay, stabilization_delta, resilience_ratio)
            except Exception as e:
                self.logger.error(f"Error creating visualization: {e}")
            
            # Return results
            result = {
                'memory_integrity_delta': memory_integrity_delta,
                'cf_influence': cf_influence,
                'parasite_ratio': parasite_ratio,
                'overwriting_rate': overwriting_rate,
                'contamination_velocity': contamination_velocity,
                'corruption_trend': corruption_trend,
                'stabilization_delta': stabilization_delta,
                'resilience_ratio': resilience_ratio,
                'before_correlation': target_before['correlation'],
                'after_correlation': target_after['correlation'],
                'before_ccdi': target_before['ccdi'],
                'after_ccdi': target_after['ccdi'],
                'injection_delay': injection_delay,
                'repetitions': repetitions
            }
            
            return result
            
        except Exception as e:
            # If an exception occurs, log it and return a default result
            self.logger.error(f"Error in _run_intrusion_test: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return a default result dictionary instead of None
            return {
                'memory_integrity_delta': 0.0,
                'cf_influence': 0.0,
                'parasite_ratio': 0.0,
                'overwriting_rate': 0.0,
                'contamination_velocity': 0.0,
                'corruption_trend': [0.0],
                'stabilization_delta': 0.0,
                'resilience_ratio': 0.0,
                'before_correlation': 0.0,
                'after_correlation': 0.0,
                'before_ccdi': 0.0,
                'after_ccdi': 0.0,
                'injection_delay': injection_delay,
                'repetitions': repetitions,
                'error': str(e)
            }
        
    
    def _visualize_counterfactual_creation(self, base_pattern, cf_pattern, similarity_level):
        """Create visualization comparing base pattern and counterfactual"""
        plt.figure(figsize=(12, 4))
        
        # Plot base pattern
        plt.subplot(1, 3, 1)
        plt.imshow(base_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Base Pattern: {base_pattern.pattern_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot counterfactual pattern
        plt.subplot(1, 3, 2)
        plt.imshow(cf_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {cf_pattern.pattern_id}\nSimilarity: {similarity_level:.2f}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot difference
        plt.subplot(1, 3, 3)
        diff = cf_pattern.initial_state - base_pattern.initial_state
        plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        plt.title("Difference")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{cf_pattern.pattern_id}_creation_sim{similarity_level:.2f}.png"))
        plt.close()
    
    def _visualize_intrusion_test(self, target_id, counterfactual_id, intrusion_strength, 
                               target_pattern, cf_pattern, target_before, target_after,
                               memory_integrity_delta, cf_influence, parasite_ratio):
        """Create visualization for intrusion test"""
        plt.figure(figsize=(15, 8))
        
        # Plot target pattern
        plt.subplot(2, 3, 1)
        plt.imshow(target_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Target Pattern: {target_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot counterfactual pattern
        plt.subplot(2, 3, 2)
        plt.imshow(cf_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {counterfactual_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot state before intrusion
        plt.subplot(2, 3, 3)
        plt.imshow(target_before['state'], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Before Intrusion\nCorr: {target_before['correlation']:.3f}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot state after intrusion
        plt.subplot(2, 3, 4)
        plt.imshow(target_after['state'], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After Intrusion\nCorr: {target_after['correlation']:.3f}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot difference between before and after
        plt.subplot(2, 3, 5)
        diff = target_after['state'] - target_before['state']
        plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"Difference\nΔIntegrity: {memory_integrity_delta:.3f}")
        plt.colorbar()
        plt.axis('off')
        
        # Add summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Color coding for integrity impact
        if memory_integrity_delta < -0.3:
            integrity_color = 'red'
        elif memory_integrity_delta < -0.1:
            integrity_color = 'orange'
        else:
            integrity_color = 'green'
            
        # Color coding for parasite formation
        if parasite_ratio > 1.0:
            parasite_color = 'red'
        elif parasite_ratio > 0.5:
            parasite_color = 'orange'
        else:
            parasite_color = 'green'
        
        # Replace with:
        summary_text = [
            f"Intrusion Strength: {intrusion_strength:.2f}",
            f"",
            f"Memory Integrity Δ: {memory_integrity_delta:.3f}",
            f"CF Influence: {cf_influence:.3f}",
            f"Parasite Ratio: {parasite_ratio:.3f}",
            f"",
            f"Before CCDI: {target_before['ccdi']:.3f}",
            f"After CCDI: {target_after['ccdi']:.3f}"
        ]

        # Plot the text without LaTeX color commands
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)

        # Add colored indicators separately
        plt.text(0.03, 0.55, "●", fontsize=18, color=integrity_color)  # Colored dot for integrity
        plt.text(0.03, 0.45, "●", fontsize=18, color=parasite_color)   # Colored dot for parasite ratio
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                              f"{target_id}_cf{counterfactual_id}_intr{intrusion_strength:.2f}.png"))
        plt.close()
    
    def _visualize_counterfactual_summary(self, results_df):
        """Create summary visualization for counterfactual experiments"""
        # [Keep existing code for the summary visualizations]
        
        # Add this at the end of the method:
        
        # Add a plot for repetition effects if applicable
        if 'repetitions' in results_df.columns and results_df['repetitions'].max() > 1:
            plt.figure(figsize=(12, 6))
            
            # Group by repetitions
            rep_groups = results_df.groupby(['repetitions', 'intrusion_strength'])
            rep_stats = rep_groups[['memory_integrity_delta', 'cf_influence']].mean().reset_index()
            
            # Plot memory integrity delta
            plt.subplot(1, 2, 1)
            for strength in sorted(rep_stats['intrusion_strength'].unique()):
                strength_data = rep_stats[rep_stats['intrusion_strength'] == strength]
                plt.plot(strength_data['repetitions'], 
                        strength_data['memory_integrity_delta'], 
                        'o-', label=f"Strength {strength:.1f}")
            
            plt.title('Memory Integrity vs Repetitions')
            plt.xlabel('Repetitions')
            plt.ylabel('Memory Integrity Delta')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot CF influence
            plt.subplot(1, 2, 2)
            for strength in sorted(rep_stats['intrusion_strength'].unique()):
                strength_data = rep_stats[rep_stats['intrusion_strength'] == strength]
                plt.plot(strength_data['repetitions'], 
                        strength_data['cf_influence'], 
                        'o-', label=f"Strength {strength:.1f}")
            
            plt.title('CF Influence vs Repetitions')
            plt.xlabel('Repetitions')
            plt.ylabel('CF Influence')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "repetition_effects.png"))
            plt.close()

        if 'injection_delay' in results_df.columns and results_df['injection_delay'].max() > 0:
            plt.figure(figsize=(12, 8))
            
            # Group by injection delay
            delay_groups = results_df.groupby(['injection_delay', 'intrusion_strength'])
            delay_stats = delay_groups[['memory_integrity_delta', 'cf_influence', 'resilience_ratio']].mean().reset_index()
            
            # Plot memory integrity delta
            plt.subplot(2, 2, 1)
            for strength in sorted(delay_stats['intrusion_strength'].unique()):
                strength_data = delay_stats[delay_stats['intrusion_strength'] == strength]
                plt.plot(strength_data['injection_delay'], 
                        strength_data['memory_integrity_delta'], 
                        'o-', label=f"Strength {strength:.1f}")
            
            plt.title('Memory Integrity Delta vs Delay')
            plt.xlabel('Injection Delay (steps)')
            plt.ylabel('Memory Integrity Delta')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot CF influence
            plt.subplot(2, 2, 2)
            for strength in sorted(delay_stats['intrusion_strength'].unique()):
                strength_data = delay_stats[delay_stats['intrusion_strength'] == strength]
                plt.plot(strength_data['injection_delay'], 
                        strength_data['cf_influence'], 
                        'o-', label=f"Strength {strength:.1f}")
            
            plt.title('CF Influence vs Delay')
            plt.xlabel('Injection Delay (steps)')
            plt.ylabel('CF Influence')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot Resilience Ratio
            plt.subplot(2, 2, 3)
            for strength in sorted(delay_stats['intrusion_strength'].unique()):
                strength_data = delay_stats[delay_stats['intrusion_strength'] == strength]
                plt.plot(strength_data['injection_delay'], 
                        strength_data['resilience_ratio'], 
                        'o-', label=f"Strength {strength:.1f}")
            
            plt.title('Memory Resilience vs Delay')
            plt.xlabel('Injection Delay (steps)')
            plt.ylabel('Resilience Ratio')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot comparison of all metrics normalized
            plt.subplot(2, 2, 4)
            
            # Get mean values across all strengths by delay
            delay_means = results_df.groupby('injection_delay')[['memory_integrity_delta', 'cf_influence', 'resilience_ratio']].mean()
            
            # Normalize each metric for better comparison
            for col in ['memory_integrity_delta', 'cf_influence', 'resilience_ratio']:
                col_min, col_max = delay_means[col].min(), delay_means[col].max()
                if col_max > col_min:
                    delay_means[f'{col}_norm'] = (delay_means[col] - col_min) / (col_max - col_min)
                else:
                    delay_means[f'{col}_norm'] = delay_means[col] * 0
            
            # Plot normalized metrics
            plt.plot(delay_means.index, delay_means['memory_integrity_delta_norm'], 'o-', label='Memory Integrity')
            plt.plot(delay_means.index, delay_means['cf_influence_norm'], 's-', label='CF Influence')
            plt.plot(delay_means.index, delay_means['resilience_ratio_norm'], '^-', label='Resilience')
            
            plt.title('Normalized Metrics vs Delay')
            plt.xlabel('Injection Delay (steps)')
            plt.ylabel('Normalized Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "delay_effects.png"))
            plt.close()
    def _visualize_intrusion_test_with_delay(self, target_id, counterfactual_id, intrusion_strength, 
                                        target_pattern, cf_pattern, initial_state, pre_injection_state, final_state,
                                        memory_integrity_delta, cf_influence, parasite_ratio,
                                        injection_delay, stabilization_delta, resilience_ratio):
        """Create visualization for intrusion test with memory consolidation delay"""
        plt.figure(figsize=(15, 10))
        
        # Plot target pattern
        plt.subplot(2, 3, 1)
        plt.imshow(target_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Target Pattern: {target_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot initial stabilized state
        plt.subplot(2, 3, 2)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial Stable State")
        plt.colorbar()
        plt.axis('off')
        
        # Plot pre-injection state (after delay)
        plt.subplot(2, 3, 3)
        plt.imshow(pre_injection_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After {injection_delay} Step Delay\nStabilization Δ: {stabilization_delta:.3f}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot final state
        plt.subplot(2, 3, 4)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After CF Intrusion")
        plt.colorbar()
        plt.axis('off')
        
        # Plot counterfactual pattern
        plt.subplot(2, 3, 5)
        plt.imshow(cf_pattern.initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Counterfactual: {counterfactual_id}\nStrength: {intrusion_strength:.2f}")
        plt.colorbar()
        plt.axis('off')
        
        # Add summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Color coding for integrity impact
        if memory_integrity_delta < -0.3:
            integrity_color = 'red'
        elif memory_integrity_delta < -0.1:
            integrity_color = 'orange'
        else:
            integrity_color = 'green'
                
        # Color coding for resilience
        if resilience_ratio > 5:
            resilience_color = 'green'
        elif resilience_ratio > 2:
            resilience_color = 'blue'
        else:
            resilience_color = 'red'
        
        summary_text = [
            f"Injection Delay: {injection_delay} steps",
            f"",
            f"Memory Integrity Δ: {memory_integrity_delta:.3f}",
            f"CF Influence: {cf_influence:.3f}",
            f"Parasite Ratio: {parasite_ratio:.3f}",
            f"",
            f"Stabilization Δ: {stabilization_delta:.3f}",
            f"Resilience Ratio: {resilience_ratio:.3f}"
        ]

        # Plot the text
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)

        # Add colored indicators separately
        plt.text(0.03, 0.6, "●", fontsize=18, color=integrity_color)  # Colored dot for integrity
        plt.text(0.03, 0.4, "●", fontsize=18, color=resilience_color)   # Colored dot for resilience
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                            f"{target_id}_cf{counterfactual_id}_intr{intrusion_strength:.2f}_delay{injection_delay}.png"))
        plt.close()