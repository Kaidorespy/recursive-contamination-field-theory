import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging
from datetime import datetime
from scipy.stats import entropy
import seaborn as sns

# Import from previous phases
from phase6.memory_blender import MemoryBlender
from phase6.counterfactual_disruptor import CounterfactualDisruptor
from rcft_metrics import compute_ccdi, compute_mutual_information, compute_field_statistics

class RecursiveContaminationEngine:
    """Engine for Phase VII: Self-Reflexive Contamination experiments"""
    
    def __init__(self, phase6, output_dir="phase7_results", log_level=logging.INFO):
        """
        Initialize the recursive contamination engine
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to Phase 6 system for access to memory bank and utilities
        output_dir : str
            Output directory for results
        log_level : logging level
            Logging verbosity
        """
        self.phase6 = phase6
        self.output_dir = output_dir
        
        # Create output directories
        self._setup_directories()
        
        # Configure logging
        self._setup_logging(log_level)
        
        # Initialize registry of recursive CFs
        self.recursive_cfs = {}  # Will store loaded CF_R files
        
        # Initialize lineage tracking
        self.lineage = {}  # Will store ancestry information
        
        # Initialize results storage
        self.results = []
        
        self.logger.info("Phase VII: Recursive Contamination Engine initialized")
    
    def _setup_directories(self):
        """Create necessary directories"""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sub-directories
        for subdir in ["recursive_cfs", "injection_results", "lineage_maps", "summary"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def _setup_logging(self, log_level):
        """Configure logging"""
        log_file = os.path.join(self.output_dir, "phase7.log")
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("Phase7")
    
    def load_recursive_cf(self, source_file, cf_id=None, source_info=None):
        """
        Load a state from a previous phase as a recursive counterfactual
        
        Parameters:
        -----------
        source_file : str
            Path to the .npy file containing the state
        cf_id : str, optional
            ID for this recursive CF (default: auto-generated)
        source_info : dict, optional
            Additional information about this CF's origins
            
        Returns:
        --------
        str
            ID of the loaded recursive CF
        """
        try:
            # Load the state
            state = np.load(source_file)
            
            # Generate CF ID if not provided
            if cf_id is None:
                # Extract base name from file
                base_name = os.path.basename(source_file)
                base_name = os.path.splitext(base_name)[0]
                cf_id = f"CF_R1_from_{base_name}"
            
            # Normalize state if needed
            state_min = np.min(state)
            state_max = np.max(state)
            if state_max > 1.0 or state_min < -1.0:
                # Normalize to [-1, 1]
                state = 2.0 * (state - state_min) / (state_max - state_min) - 1.0
                self.logger.info(f"Normalized state range from [{state_min:.3f}, {state_max:.3f}] to [-1, 1]")
            
            # Store in recursive CF registry
            self.recursive_cfs[cf_id] = {
                'state': state,
                'source_file': source_file,
                'source_info': source_info if source_info is not None else {},
                'metrics': compute_field_statistics(state)
            }
            
            # Initialize lineage tracking
            self.lineage[cf_id] = {
                'generation': 1,  # CF_R1
                'parent': None,   # No parent CF for first-gen
                'source_file': source_file,
                'source_info': source_info if source_info is not None else {},
                'children': []    # No children yet
            }
            
            # Save a copy in our output directory
            output_file = os.path.join(self.output_dir, "recursive_cfs", f"{cf_id}.npy")
            np.save(output_file, state)
            
            self.logger.info(f"Loaded recursive CF '{cf_id}' from {source_file}")
            
            # Create visualization
            self._visualize_recursive_cf(cf_id)
            
            return cf_id
            
        except Exception as e:
            self.logger.error(f"Error loading recursive CF from {source_file}: {e}")
            return None
    
    def create_fresh_hybrid(self, pattern_a_id, pattern_b_id, blend_ratio=0.5, method="pixel", hybrid_id=None):
        """
        Create a fresh hybrid for CF injection
        
        Parameters:
        -----------
        pattern_a_id : str
            ID of first pattern
        pattern_b_id : str
            ID of second pattern
        blend_ratio : float
            Blend ratio (0.0 = all A, 1.0 = all B)
        method : str
            Blending method ("pixel", "structured", or "frequency")
        hybrid_id : str, optional
            ID for this hybrid (default: auto-generated)
            
        Returns:
        --------
        dict
            Information about the created hybrid
        """
        # Check if patterns exist
        if pattern_a_id not in self.phase6.memory_bank:
            self.logger.error(f"Pattern {pattern_a_id} not found in memory bank")
            return None
        if pattern_b_id not in self.phase6.memory_bank:
            self.logger.error(f"Pattern {pattern_b_id} not found in memory bank")
            return None
        
        # Generate hybrid ID if not provided
        if hybrid_id is None:
            hybrid_id = f"{pattern_a_id}_{pattern_b_id}_{blend_ratio:.2f}_{method}"
        
        # Create the blend using Phase 6 blender
        try:
            blended_pattern = self.phase6.blender.blend_patterns(
                pattern_a_id, pattern_b_id, 
                ratio=blend_ratio, method=method
            )
            
            # Initialize an RCFT experiment to stabilize this blend
            exp = self.phase6.base_experiment
            exp.state = blended_pattern.copy()
            
            # Let it stabilize briefly
            exp.update(steps=20)
            
            # Capture stabilized state
            stabilized_state = exp.state.copy()
            
            # Save the hybrid
            hybrid_file = os.path.join(self.output_dir, "injection_results", f"{hybrid_id}_initial.npy")
            np.save(hybrid_file, stabilized_state)
            
            # Compute metrics
            hybrid_metrics = {
                'correlation_a': np.corrcoef(self.phase6.memory_bank[pattern_a_id].initial_state.flatten(), 
                                       stabilized_state.flatten())[0, 1],
                'correlation_b': np.corrcoef(self.phase6.memory_bank[pattern_b_id].initial_state.flatten(), 
                                       stabilized_state.flatten())[0, 1],
                'field_stats': compute_field_statistics(stabilized_state)
            }
            
            hybrid_info = {
                'id': hybrid_id,
                'pattern_a': pattern_a_id,
                'pattern_b': pattern_b_id,
                'blend_ratio': blend_ratio,
                'blend_method': method,
                'state': stabilized_state,
                'metrics': hybrid_metrics
            }
            
            self.logger.info(f"Created fresh hybrid '{hybrid_id}' with correlation to A: {hybrid_metrics['correlation_a']:.3f}, B: {hybrid_metrics['correlation_b']:.3f}")
            
            return hybrid_info
            
        except Exception as e:
            self.logger.error(f"Error creating fresh hybrid: {e}")
            return None
    
    def inject_recursive_cf(self, hybrid_info, cf_id, strength=0.6, delay=0, steps=50):
        """
        Inject a recursive CF into a fresh hybrid
        
        Parameters:
        -----------
        hybrid_info : dict
            Information about the hybrid
        cf_id : str
            ID of the recursive CF to inject
        strength : float
            Injection strength (0.0 to 1.0)
        delay : int
            Delay before injection (steps)
        steps : int
            Steps to run after injection
            
        Returns:
        --------
        dict
            Results of the injection
        """
        if cf_id not in self.recursive_cfs:
            self.logger.error(f"Recursive CF '{cf_id}' not found")
            return None
            
        try:
            # Get CF state
            cf_state = self.recursive_cfs[cf_id]['state']
            
            # Setup experiment
            exp = self.phase6.base_experiment
            exp.state = hybrid_info['state'].copy()
            
            # Let it stabilize more if delay > 0
            if delay > 0:
                exp.update(steps=delay)
            
            # Save pre-injection state
            pre_injection_state = exp.state.copy()
            
            # Inject CF
            injected_state = (1.0 - strength) * pre_injection_state + strength * cf_state
            
            # Normalize if needed
            state_min = np.min(injected_state)
            state_max = np.max(injected_state)
            if state_max > 1.0 or state_min < -1.0:
                injected_state = 2.0 * (injected_state - state_min) / (state_max - state_min) - 1.0
            
            # Set the injected state
            exp.state = injected_state.copy()
            
            # Let it evolve
            exp.update(steps=steps)
            
            # Capture final state
            final_state = exp.state.copy()
            
            # Generate result ID
            result_id = f"{hybrid_info['id']}_injected_{cf_id}_str{strength:.2f}_delay{delay}"
            
            # Calculate metrics
            pattern_a_id = hybrid_info['pattern_a']
            pattern_b_id = hybrid_info['pattern_b']
            
            # Correlation to original patterns
            corr_a_final = np.corrcoef(self.phase6.memory_bank[pattern_a_id].initial_state.flatten(), 
                                   final_state.flatten())[0, 1]
            corr_b_final = np.corrcoef(self.phase6.memory_bank[pattern_b_id].initial_state.flatten(), 
                                   final_state.flatten())[0, 1]
            
            # Correlation to pre-injection state (memory integrity)
            corr_pre = np.corrcoef(pre_injection_state.flatten(), final_state.flatten())[0, 1]
            
            # Correlation to CF
            corr_cf = np.corrcoef(cf_state.flatten(), final_state.flatten())[0, 1]
            
            # Calculate memory integrity delta
            corr_a_pre = np.corrcoef(self.phase6.memory_bank[pattern_a_id].initial_state.flatten(), 
                                 pre_injection_state.flatten())[0, 1]
            corr_b_pre = np.corrcoef(self.phase6.memory_bank[pattern_b_id].initial_state.flatten(), 
                                 pre_injection_state.flatten())[0, 1]
            
            # Use the stronger original correlation
            original_pre = max(corr_a_pre, corr_b_pre)
            original_final = max(corr_a_final, corr_b_final)
            
            memory_integrity_delta = original_final - original_pre
            
            # Calculate CF influence - how much the final correlates with CF
            cf_influence = corr_cf
            
            # Calculate recursive drift - how far from original hybrid
            recursive_drift = 1.0 - corr_pre
            
            # Calculate recovery bias ratio
            if corr_a_final > 0 and corr_b_final > 0:
                recovery_bias = corr_a_final / corr_b_final
            else:
                recovery_bias = float('inf') if corr_a_final > 0 else 0.0
            
            # Calculate Recursive Fragility Index (RFI)
            rfi = (original_pre - original_final) / 1.0  # Only one recursion level here
            
            # Check for attractor melting
            attractor_melting = (corr_a_final < 0.3 and corr_b_final < 0.3 and corr_cf < 0.3)
            
            # Field statistics
            field_stats = compute_field_statistics(final_state)
            
            # Assemble results
            metrics = {
                'memory_integrity_delta': memory_integrity_delta,
                'cf_influence': cf_influence,
                'recursive_drift': recursive_drift,
                'recovery_bias': recovery_bias,
                'rfi': rfi,
                'corr_a_final': corr_a_final,
                'corr_b_final': corr_b_final,
                'corr_cf': corr_cf,
                'corr_pre': corr_pre,
                'attractor_melting': attractor_melting,
                'entropy': field_stats['entropy'],
                'skewness': field_stats['skewness'],
                'kurtosis': field_stats['kurtosis']
            }
            
            # Save states
            np.save(os.path.join(self.output_dir, "injection_results", f"{result_id}_pre.npy"), pre_injection_state)
            np.save(os.path.join(self.output_dir, "injection_results", f"{result_id}_final.npy"), final_state)
            
            # Compile result
            result = {
                'id': result_id,
                'hybrid_id': hybrid_info['id'],
                'cf_id': cf_id,
                'strength': strength,
                'delay': delay,
                'pre_injection_state': pre_injection_state,
                'final_state': final_state,
                'metrics': metrics
            }
            
            # Update lineage - record this trial as "child" of the CF
            if cf_id in self.lineage:
                self.lineage[cf_id]['children'].append(result_id)
            
            # Add to results
            self.results.append(result)
            
            # Create visualization
            self._visualize_injection_result(result)
            
            # Log result summary
            self.logger.info(f"Completed recursive CF injection: {result_id}")
            self.logger.info(f"  Memory integrity delta: {memory_integrity_delta:.4f}")
            self.logger.info(f"  CF influence: {cf_influence:.4f}")
            self.logger.info(f"  Recursive drift: {recursive_drift:.4f}")
            self.logger.info(f"  RFI: {rfi:.4f}")
            self.logger.info(f"  Attractor melting: {attractor_melting}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in recursive CF injection: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_next_generation_cf(self, result_id, generation=2):
        """
        Create a next-generation recursive CF from an injection result
        
        Parameters:
        -----------
        result_id : str
            ID of the injection result to use as the next-gen CF
        generation : int
            Generation number (default: 2 for CF_R2)
            
        Returns:
        --------
        str
            ID of the new recursive CF
        """
        # Find the result
        result = None
        for r in self.results:
            if r['id'] == result_id:
                result = r
                break
                
        if result is None:
            self.logger.error(f"Injection result '{result_id}' not found")
            return None
            
        try:
            # Extract the final state
            final_state = result['final_state']
            
            # Generate new CF ID
            parent_cf_id = result['cf_id']
            new_cf_id = f"CF_R{generation}_from_{parent_cf_id}"
            
            # Store in recursive CF registry
            self.recursive_cfs[new_cf_id] = {
                'state': final_state.copy(),
                'source_result': result_id,
                'parent_cf': parent_cf_id,
                'generation': generation,
                'metrics': compute_field_statistics(final_state)
            }
            
            # Update lineage tracking
            self.lineage[new_cf_id] = {
                'generation': generation,
                'parent': parent_cf_id,
                'source_result': result_id,
                'children': []
            }
            
            # Update parent's children list
            if parent_cf_id in self.lineage:
                if new_cf_id not in self.lineage[parent_cf_id]['children']:
                    self.lineage[parent_cf_id]['children'].append(new_cf_id)
            
            # Save the new CF
            output_file = os.path.join(self.output_dir, "recursive_cfs", f"{new_cf_id}.npy")
            np.save(output_file, final_state)
            
            self.logger.info(f"Created generation {generation} recursive CF '{new_cf_id}' from {result_id}")
            
            # Create visualization
            self._visualize_recursive_cf(new_cf_id)
            
            return new_cf_id
            
        except Exception as e:
            self.logger.error(f"Error creating next-generation CF: {e}")
            return None
    
    def run_recursive_contamination_experiment(self, pattern_a_id, pattern_b_id, 
                                            cf_sources, injection_strengths,
                                            blend_ratio=0.5, blend_method="pixel",
                                            delay=0, steps=50, trials=1):
        """
        Run a comprehensive recursive contamination experiment
        
        Parameters:
        -----------
        pattern_a_id : str
            ID of first base pattern
        pattern_b_id : str
            ID of second base pattern
        cf_sources : list
            List of file paths to use as recursive CFs
        injection_strengths : list
            List of injection strengths to test
        blend_ratio : float
            Ratio for blending patterns (default: 0.5)
        blend_method : str
            Method for blending (default: "pixel")
        delay : int
            Delay before injection (default: 0)
        steps : int
            Steps to run after injection (default: 50)
        trials : int
            Number of trials per configuration (default: 1)
            
        Returns:
        --------
        pd.DataFrame
            Results of the experiment
        """
        self.logger.info(f"Starting recursive contamination experiment with {len(cf_sources)} CF sources and {len(injection_strengths)} injection strengths")
        
        # Load recursive CFs
        cf_ids = []
        for i, source in enumerate(cf_sources):
            cf_id = f"CF_R1_source{i+1}"
            loaded_id = self.load_recursive_cf(source, cf_id=cf_id, source_info={'index': i+1})
            if loaded_id:
                cf_ids.append(loaded_id)
        
        if not cf_ids:
            self.logger.error("No recursive CFs loaded. Experiment aborted.")
            return None
            
        # Create fresh hybrid
        hybrid_info = self.create_fresh_hybrid(
            pattern_a_id, pattern_b_id,
            blend_ratio=blend_ratio, method=blend_method
        )
        
        if hybrid_info is None:
            self.logger.error("Failed to create hybrid. Experiment aborted.")
            return None
            
        # Run all combinations
        for cf_id in cf_ids:
            for strength in injection_strengths:
                for trial in range(trials):
                    trial_suffix = f"_trial{trial+1}" if trials > 1 else ""
                    result = self.inject_recursive_cf(
                        hybrid_info,
                        cf_id,
                        strength=strength,
                        delay=delay,
                        steps=steps
                    )
                    
                    # Check if we should create a second-generation CF (for divergent cases)
                    if result and (result['metrics']['attractor_melting'] or 
                                abs(result['metrics']['recursive_drift']) > 0.5):
                        self.logger.info(f"Creating CF_R2 from divergent result {result['id']}")
                        self.create_next_generation_cf(result['id'], generation=2)
        
        # Compile results
        results_df = self.compile_results_dataframe()
        
        # Create summary visualizations
        self._create_experiment_summary()
        
        # Create lineage visualizations
        self._visualize_lineage_maps()
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, "summary", "recursive_contamination_results.csv"), index=False)
        
        return results_df
    
    def compile_results_dataframe(self):
        """Compile all results into a DataFrame"""
        rows = []
        
        for result in self.results:
            metrics = result['metrics']
            row = {
                'id': result['id'],
                'hybrid_id': result['hybrid_id'],
                'cf_id': result['cf_id'],
                'strength': result['strength'],
                'delay': result['delay'],
                'memory_integrity_delta': metrics['memory_integrity_delta'],
                'cf_influence': metrics['cf_influence'],
                'recursive_drift': metrics['recursive_drift'],
                'recovery_bias': metrics['recovery_bias'],
                'rfi': metrics['rfi'],
                'corr_a_final': metrics['corr_a_final'],
                'corr_b_final': metrics['corr_b_final'],
                'corr_cf': metrics['corr_cf'],
                'attractor_melting': metrics['attractor_melting'],
                'entropy': metrics['entropy'],
                'skewness': metrics['skewness'],
                'kurtosis': metrics['kurtosis']
            }
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def _visualize_recursive_cf(self, cf_id):
        """Create visualization for a recursive CF"""
        if cf_id not in self.recursive_cfs:
            self.logger.error(f"CF '{cf_id}' not found for visualization")
            return
            
        cf_info = self.recursive_cfs[cf_id]
        state = cf_info['state']
        
        plt.figure(figsize=(10, 8))
        
        # Plot the state
        plt.subplot(2, 2, 1)
        plt.imshow(state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Recursive CF: {cf_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot FFT magnitude
        plt.subplot(2, 2, 2)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(state)))
        plt.imshow(np.log(fft + 1), cmap='viridis')
        plt.title('FFT Magnitude')
        plt.colorbar()
        plt.axis('off')
        
        # Plot histogram
        plt.subplot(2, 2, 3)
        plt.hist(state.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title('Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        # Add information panel
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        if 'metrics' in cf_info:
            metrics = cf_info['metrics']
            info_text = [
                f"CF ID: {cf_id}",
                f"Generation: {self.lineage[cf_id]['generation'] if cf_id in self.lineage else 'Unknown'}",
                f"",
                f"Statistics:",
                f"  Mean: {metrics['mean']:.4f}",
                f"  Std Dev: {metrics['std']:.4f}",
                f"  Min/Max: [{metrics['min']:.4f}, {metrics['max']:.4f}]",
                f"  Skewness: {metrics['skewness']:.4f}",
                f"  Kurtosis: {metrics['kurtosis']:.4f}",
                f"  Entropy: {metrics['entropy']:.4f}",
                f"",
                f"Source: {cf_info.get('source_file', 'Unknown')}"
            ]
        else:
            info_text = [f"CF ID: {cf_id}", "No metrics available"]
            
        plt.text(0.1, 0.5, '\n'.join(info_text), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "recursive_cfs", f"{cf_id}_visualization.png"))
        plt.close()
    
    def _visualize_injection_result(self, result):
        """Create visualization for an injection result"""
        hybrid_id = result['hybrid_id']
        cf_id = result['cf_id']
        
        # Extract relevant info from result
        pre_state = result['pre_injection_state']
        final_state = result['final_state']
        metrics = result['metrics']
        
        plt.figure(figsize=(15, 10))
        
        # Plot states
        plt.subplot(2, 3, 1)
        plt.imshow(pre_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Pre-Injection State\n{hybrid_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot CF
        plt.subplot(2, 3, 2)
        plt.imshow(self.recursive_cfs[cf_id]['state'], cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Injected CF\n{cf_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot final state
        plt.subplot(2, 3, 3)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Final State\nStr={result['strength']:.2f}, Delay={result['delay']}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot difference (final - pre)
        plt.subplot(2, 3, 4)
        diff = final_state - pre_state
        plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        plt.title(f"Difference (Final - Pre)")
        plt.colorbar()
        plt.axis('off')
        
        # Plot correlation bar chart
        plt.subplot(2, 3, 5)
        correlations = [
            metrics['corr_a_final'], 
            metrics['corr_b_final'], 
            metrics['corr_cf'], 
            metrics['corr_pre']
        ]
        labels = ['Pattern A', 'Pattern B', 'CF', 'Pre-State']
        colors = ['blue', 'green', 'red', 'orange']
        
        plt.bar(labels, correlations, color=colors)
        plt.axhline(y=0.7, color='green', linestyle='--', label='Strong')
        plt.axhline(y=0.3, color='red', linestyle='--', label='Weak')
        plt.title('Final State Correlations')
        plt.ylabel('Correlation')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        # Add metrics summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Determine warning indicators
        memory_integrity_color = 'red' if metrics['memory_integrity_delta'] < -0.3 else 'green'
        cf_influence_color = 'red' if metrics['cf_influence'] > 0.7 else 'green'
        
        info_text = [
            f"Result ID: {result['id']}",
            f"",
            f"Metrics:",
            f"  Memory Integrity Δ: {metrics['memory_integrity_delta']:.4f}",
            f"  CF Influence: {metrics['cf_influence']:.4f}",
            f"  Recursive Drift: {metrics['recursive_drift']:.4f}",
            f"  Recovery Bias: {metrics['recovery_bias']:.4f}",
            f"  RFI: {metrics['rfi']:.4f}",
            f"",
            f"Attractor Status:",
            f"  Melting: {'Yes' if metrics['attractor_melting'] else 'No'}",
            f"  Entropy: {metrics['entropy']:.4f}",
            f"  Skewness: {metrics['skewness']:.4f}"
        ]
        
        plt.text(0.1, 0.5, '\n'.join(info_text), fontsize=10)
        
        # Add warning indicators
        plt.text(0.01, 0.62, "●", fontsize=20, color=memory_integrity_color)
        plt.text(0.01, 0.57, "●", fontsize=20, color=cf_influence_color)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "injection_results", f"{result['id']}_visualization.png"))
        plt.close()
    
    def _create_experiment_summary(self):
        """Create summary visualizations for the experiment"""
        if not self.results:
            self.logger.warning("No results to visualize in experiment summary")
            return
            
        # Compile dataframe
        df = self.compile_results_dataframe()
        
        # Plot 1: CF Influence by strength for each CF
        plt.figure(figsize=(12, 6))
        
        for cf_id in df['cf_id'].unique():
            cf_data = df[df['cf_id'] == cf_id]
            plt.plot(cf_data['strength'], cf_data['cf_influence'], 'o-', label=cf_id)
            
        plt.title('CF Influence vs. Injection Strength')
        plt.xlabel('Injection Strength')
        plt.ylabel('CF Influence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "cf_influence_by_strength.png"))
        plt.close()
        
        # Plot 2: Memory Integrity Delta by strength for each CF
        plt.figure(figsize=(12, 6))
        
        for cf_id in df['cf_id'].unique():
            cf_data = df[df['cf_id'] == cf_id]
            plt.plot(cf_data['strength'], cf_data['memory_integrity_delta'], 'o-', label=cf_id)
            
        plt.title('Memory Integrity Delta vs. Injection Strength')
        plt.xlabel('Injection Strength')
        plt.ylabel('Memory Integrity Delta')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "memory_integrity_by_strength.png"))
        plt.close()
        
        # Plot 3: Recursive Fragility Index by CF
        plt.figure(figsize=(10, 6))
        
        cf_rfi = df.groupby('cf_id')['rfi'].mean().sort_values()
        plt.bar(cf_rfi.index, cf_rfi.values)
        
        plt.title('Average Recursive Fragility Index by CF')
        plt.xlabel('Recursive CF')
        plt.ylabel('RFI (higher = more fragile)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "rfi_by_cf.png"))
        plt.close()
        
        # Plot 4: Correlation Triangle - A vs B vs CF
        plt.figure(figsize=(10, 8))
        
        for i, result in enumerate(self.results):
            metrics = result['metrics']
            plt.scatter(
                metrics['corr_a_final'], 
                metrics['corr_b_final'],
                c=[metrics['corr_cf']],
                cmap='viridis',
                s=100,
                alpha=0.7,
                label=result['cf_id'] if i == 0 else ""
            )
            
        plt.title('Correlation Triangle (A vs B colored by CF correlation)')
        plt.xlabel('Correlation to Pattern A')
        plt.ylabel('Correlation to Pattern B')
        plt.colorbar(label='Correlation to CF')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "correlation_triangle.png"))
        plt.close()
        
        # Plot 5: Attractor melting analysis
        plt.figure(figsize=(12, 5))
        
        # Count melting cases by strength
        melting_by_strength = df.groupby(['strength'])['attractor_melting'].mean()
        
        plt.subplot(1, 2, 1)
        plt.plot(melting_by_strength.index, melting_by_strength.values, 'o-')
        plt.title('Attractor Melting Probability by Strength')
        plt.xlabel('Injection Strength')
        plt.ylabel('Probability of Melting')
        plt.grid(True, alpha=0.3)
        
        # Count melting cases by CF
        melting_by_cf = df.groupby(['cf_id'])['attractor_melting'].mean()
        
        plt.subplot(1, 2, 2)
        plt.bar(melting_by_cf.index, melting_by_cf.values)
        plt.title('Attractor Melting Probability by CF')
        plt.xlabel('Recursive CF')
        plt.ylabel('Probability of Melting')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "attractor_melting_analysis.png"))
        plt.close()
        
        # Plot 6: Entropy and Skewness relationship
        plt.figure(figsize=(10, 8))
        
        plt.scatter(
            df['entropy'], 
            df['skewness'],
            c=df['cf_influence'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        plt.title('Entropy vs Skewness (colored by CF Influence)')
        plt.xlabel('Entropy')
        plt.ylabel('Skewness')
        plt.colorbar(label='CF Influence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary", "entropy_skewness_relationship.png"))
        plt.close()
    
    def _visualize_lineage_maps(self):
        """Create visualizations of memory lineage"""
        if not self.lineage:
            self.logger.warning("No lineage data available for mapping")
            return
            
        # For each root CF, create a lineage visualization
        root_cfs = [cf_id for cf_id, info in self.lineage.items() if info['generation'] == 1]
        
        for root_cf in root_cfs:
            self._visualize_single_lineage(root_cf)
            
        # Create a master lineage map
        self._create_master_lineage_map()
    
    def _visualize_single_lineage(self, root_cf):
        """Create visualization for a single CF's lineage"""
        if root_cf not in self.lineage:
            return
            
        # Collect all descendants
        descendants = self._collect_descendants(root_cf)
        
        # Create a simple tree visualization
        plt.figure(figsize=(12, 8))
        
        # Set up for simple tree drawing
        max_generation = max(info['generation'] for info in descendants.values())
        x_positions = {}
        y_positions = {}
        
        # Position root
        x_positions[root_cf] = 0
        y_positions[root_cf] = max_generation
        
        # Position children
        self._position_nodes(root_cf, descendants, x_positions, y_positions, max_generation)
        
        # Draw edges
        for cf_id, info in descendants.items():
            if info['parent'] in x_positions:
                parent_x = x_positions[info['parent']]
                parent_y = y_positions[info['parent']]
                child_x = x_positions[cf_id]
                child_y = y_positions[cf_id]
                
                plt.plot([parent_x, child_x], [parent_y, child_y], 'k-', alpha=0.6)
        
        # Draw nodes
        for cf_id, info in descendants.items():
            if cf_id in x_positions:
                x = x_positions[cf_id]
                y = y_positions[cf_id]
                
                # Color by generation
                color = plt.cm.viridis(info['generation'] / max(max_generation, 1))
                
                plt.scatter(x, y, s=200, color=color, alpha=0.8, edgecolor='black')
                plt.text(x, y - 0.2, cf_id, ha='center', va='top', fontsize=8, rotation=45)
        
        plt.title(f"Memory Lineage from {root_cf}")
        plt.xlabel('Drift Distance')
        plt.ylabel('Generation')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "lineage_maps", f"{root_cf}_lineage.png"))
        plt.close()
    
    def _collect_descendants(self, cf_id):
        """Recursively collect all descendants of a CF"""
        descendants = {cf_id: self.lineage[cf_id]}
        
        def collect_children(parent_id):
            if parent_id in self.lineage:
                for child_id in self.lineage[parent_id]['children']:
                    if child_id in self.lineage and child_id not in descendants:
                        descendants[child_id] = self.lineage[child_id]
                        collect_children(child_id)
        
        collect_children(cf_id)
        return descendants
    
    def _position_nodes(self, cf_id, descendants, x_positions, y_positions, max_generation):
        """Position nodes for tree visualization"""
        info = descendants[cf_id]
        
        # Position children
        children = [child for child in info.get('children', []) if child in descendants]
        
        if not children:
            return
            
        # Position children horizontally
        child_width = len(children)
        start_x = x_positions[cf_id] - (child_width - 1) / 2
        
        for i, child in enumerate(children):
            x = start_x + i
            x_positions[child] = x
            y_positions[child] = max_generation - descendants[child]['generation']
            
            # Recursively position grandchildren
            self._position_nodes(child, descendants, x_positions, y_positions, max_generation)
    
    def _create_master_lineage_map(self):
        """Create a master lineage map of all CFs"""
        # Get all CFs organized by generation
        generations = {}
        for cf_id, info in self.lineage.items():
            gen = info['generation']
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(cf_id)
            
        if not generations:
            return
            
        # Create a lineage network visualization
        plt.figure(figsize=(15, 10))
        
        # Position by generation
        pos = {}
        for gen, cf_ids in generations.items():
            y = max(generations.keys()) - gen
            for i, cf_id in enumerate(cf_ids):
                x = i - len(cf_ids) / 2
                pos[cf_id] = (x, y)
        
        # Draw edges
        for cf_id, info in self.lineage.items():
            if info['parent'] in pos:
                parent_x, parent_y = pos[info['parent']]
                child_x, child_y = pos[cf_id]
                
                plt.plot([parent_x, child_x], [parent_y, child_y], 'k-', alpha=0.6)
        
        # Draw
    

def run_phase7_experiment(phase6_instance, output_dir="phase7_results", 
                        pattern_ids=None, cf_sources=None, log_level=logging.INFO):
    """
    Run the Phase VII experiment suite
    
    Parameters:
    -----------
    phase6_instance : Phase6MultiMemory
        Reference to Phase 6 system
    output_dir : str
        Output directory for results
    pattern_ids : list, optional
        List of pattern IDs to use (default: ['A', 'B'])
    cf_sources : list, optional
        List of source files for recursive CFs
    log_level : logging level
        Logging verbosity
        
    Returns:
    --------
    RecursiveContaminationEngine
        The configured engine with results
    """
    # Default pattern IDs if not provided
    if pattern_ids is None:
        pattern_ids = ['A', 'B']
    
    # Default CF sources if not provided
    if cf_sources is None:
        # Look for files in the phase6 outputs that match potential CF criteria
        phase6_dir = phase6_instance.output_dir
        cf_candidates = []
        
        # Try to find counterfactual results
        counterfactual_dir = os.path.join(phase6_dir, "counterfactual")
        if os.path.exists(counterfactual_dir):
            # Look for final states from counterfactual experiments
            for filename in os.listdir(counterfactual_dir):
                if filename.endswith("_final.npy"):
                    cf_candidates.append(os.path.join(counterfactual_dir, filename))
        
        # Try to find blending results
        blending_dir = os.path.join(phase6_dir, "blending")
        if os.path.exists(blending_dir):
            # Look for final states from blend recovery experiments
            for filename in os.listdir(blending_dir):
                if filename.endswith("_final.npy"):
                    cf_candidates.append(os.path.join(blending_dir, filename))
        
        # Use the first 3 found, or create a warning if none found
        if cf_candidates:
            cf_sources = cf_candidates[:min(3, len(cf_candidates))]
        else:
            logging.warning("No CF candidates found in Phase 6 output. Will need to specify sources manually.")
            cf_sources = []
    
    # Initialize the engine
    engine = RecursiveContaminationEngine(phase6_instance, output_dir=output_dir, log_level=log_level)
    
    # If we have sources, run the experiment
    if cf_sources:
        # Define injection strengths
        injection_strengths = [0.4, 0.6, 0.8]
        
        # Run the recursive contamination experiment
        results_df = engine.run_recursive_contamination_experiment(
            pattern_ids[0], pattern_ids[1],
            cf_sources=cf_sources,
            injection_strengths=injection_strengths,
            blend_ratio=0.5,
            blend_method="pixel",
            delay=0,
            steps=50
        )
        
        print(f"Phase VII experiment completed. Results saved to {output_dir}")
    else:
        print("No CF sources provided. Experiment not run.")
    
    return engine