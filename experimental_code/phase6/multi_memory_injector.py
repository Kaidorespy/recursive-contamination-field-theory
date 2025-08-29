# Add the implementation of the MultiMemoryInjector class
import numpy as np
import matplotlib.pyplot as plt
import os
from rcft_metrics import compute_ccdi

class MultiMemoryInjector:
    """Injects multiple base memory patterns into the system"""
    
    def __init__(self, phase6):
        """
        Initialize the memory injector
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
    
    def run_sequential_encoding_experiment(self, pattern_ids, encoding_strengths, 
                                         decay_rates, delays, perturbation_strengths):
        """
        Run sequential encoding experiment for interference mapping
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to encode sequentially
        encoding_strengths : list
            List of encoding strengths to test
        decay_rates : list
            List of decay rates to test
        delays : list
            List of delays between encodings to test
        perturbation_strengths : list
            List of perturbation magnitudes to test
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        results = []
        
        # Create experiment combinations
        for enc_strength in encoding_strengths:
            for decay in decay_rates:
                for delay in delays:
                    for perturb in perturbation_strengths:
                        # Configure experiment
                        self.phase6.alpha = enc_strength
                        self.phase6.gamma = decay
                        self.phase6._initialize_base_experiment()
                        
                        # Get patterns
                        pattern_a = self.phase6.memory_bank[pattern_ids[0]]
                        pattern_b = self.phase6.memory_bank[pattern_ids[1]]
                        
                        # Train pattern A
                        self.encode_pattern(pattern_a.pattern_id)
                        
                        # Wait specified delay
                        steps_before_b = delay
                        
                        # Train pattern B
                        self.encode_pattern(pattern_b.pattern_id)
                        
                        # Recall pattern A with perturbation
                        recall_metrics = self.recall_pattern(
                            pattern_a.pattern_id, 
                            perturbation_strength=perturb
                        )
                        
                        # Compute interference metrics
                        base_recall = recall_metrics['recovery_correlation']
                        interference_index = 1.0 - base_recall  # Higher means more interference
                        
                        # Store results
                        result = {
                            'pattern_a': pattern_a.pattern_id,
                            'pattern_b': pattern_b.pattern_id,
                            'encoding_strength': enc_strength,
                            'decay_rate': decay,
                            'delay': delay,
                            'perturbation_strength': perturb,
                            'recovery_correlation': base_recall,
                            'interference_index': interference_index,
                            'memory_purity': recall_metrics.get('memory_purity', 0.0),
                            'ccdi': recall_metrics.get('ccdi', 0.0),
                            'is_anomalous': recall_metrics.get('is_anomalous', False)
                        }
                        
                        results.append(result)
                        
                        self.logger.info(f"Ran interference test: A={pattern_a.pattern_id}, B={pattern_b.pattern_id}, " +
                                      f"α={enc_strength}, γ={decay}, delay={delay}, " +
                                      f"Interference={interference_index:.3f}")
        
        return results
    
    def encode_pattern(self, pattern_id, steps=50):
        """
        Encode a pattern into the field
        
        Parameters:
        -----------
        pattern_id : str
            ID of pattern to encode
        steps : int
            Number of steps to let system evolve
            
        Returns:
        --------
        dict
            Encoding metrics
        """
        # Get pattern from memory bank
        if pattern_id not in self.phase6.memory_bank:
            raise ValueError(f"Pattern {pattern_id} not found in memory bank")
            
        pattern = self.phase6.memory_bank[pattern_id]
        
        # Initialize experiment with pattern
        exp = self.phase6.base_experiment
        exp.state = pattern.initial_state.copy()
        
        # Add the initial state to history (this ensures metrics are calculated)
        if len(exp.history) == 0:
            exp.history.append(exp.state.copy())
            # Initialize metrics if needed
            exp.metrics = {
                'correlation': [1.0],  # Correlation with itself is 1.0
                'coherence': [1.0],    # Initial coherence is typically 1.0
                'mutual_info': [1.0],  # Initial mutual info is typically 1.0
                'spectral_entropy': [0.0],  # Placeholder until calculated
                'variance': [0.0]      # Placeholder until calculated
            }
        
        # Record initial state as the pattern's attractor goal
        exp.initial_state = pattern.initial_state.copy()
        
        # Let system evolve to form attractor
        exp.update(steps=steps)
        
        # Check if metrics were calculated
        if not exp.metrics['correlation'] or len(exp.metrics['correlation']) <= 1:
            # If metrics are empty or incomplete, calculate them now
            self.phase6.logger.warning(f"Metrics not properly calculated for {pattern_id}, calculating manually")
            
            # Simple calculation of correlation with initial state
            flat_initial = pattern.initial_state.flatten()
            flat_final = exp.state.flatten()
            correlation = np.corrcoef(flat_initial, flat_final)[0, 1]
            
            # Calculate coherence as inverse of variance
            variance = np.var(exp.state)
            coherence = 1.0 / (1.0 + variance)
            
            # Use calculated metrics
            metrics = {
                'correlation': correlation,
                'coherence': coherence,
                'mutual_info': 0.0,  # Placeholder
                'spectral_entropy': 0.0,  # Placeholder
                'ccdi': coherence - correlation
            }
        else:
            # Capture final state and metrics
            metrics = {
                'correlation': exp.metrics['correlation'][-1],
                'coherence': exp.metrics['coherence'][-1],
                'mutual_info': exp.metrics['mutual_info'][-1] if 'mutual_info' in exp.metrics else 0.0,
                'spectral_entropy': exp.metrics['spectral_entropy'][-1] if 'spectral_entropy' in exp.metrics else 0.0,
                'ccdi': compute_ccdi(exp.metrics['correlation'][-1], exp.metrics['coherence'][-1])
            }
        
        # Update memory trace
        pattern.final_state = exp.state.copy()
        pattern.metrics = metrics
        pattern.compute_fingerprint()
        
        return metrics
    
    def recall_pattern(self, pattern_id, perturbation_strength=1.0, 
                     perturbation_type="flip", steps=50):
        """
        Test recall of a pattern with perturbation
        
        Parameters:
        -----------
        pattern_id : str
            ID of pattern to recall
        perturbation_strength : float
            Magnitude of perturbation
        perturbation_type : str
            Type of perturbation to apply
        steps : int
            Number of steps to let system evolve
            
        Returns:
        --------
        dict
            Recall metrics
        """
        # Get pattern from memory bank
        if pattern_id not in self.phase6.memory_bank:
            raise ValueError(f"Pattern {pattern_id} not found in memory bank")
            
        pattern = self.phase6.memory_bank[pattern_id]
        
        # Initialize experiment with pattern's final state (attractor)
        exp = self.phase6.base_experiment
        if pattern.final_state is not None:
            exp.state = pattern.final_state.copy()
        else:
            exp.state = pattern.initial_state.copy()
            
        # Apply perturbation
        exp.apply_perturbation(
            perturbation_type=perturbation_type,
            magnitude=perturbation_strength,
            radius=15
        )
        
        # Let system evolve to recover
        exp.update(steps=steps)
        
        # Capture final state and metrics
        final_state = exp.state.copy()
        
        # Calculate recall metrics
        original_correlation = np.corrcoef(pattern.initial_state.flatten(), 
                                       final_state.flatten())[0, 1]
        
        recall_metrics = {
            'correlation': exp.metrics['correlation'][-1],
            'coherence': exp.metrics['coherence'][-1],
            'mutual_info': exp.metrics['mutual_info'][-1],
            'ccdi': compute_ccdi(exp.metrics['correlation'][-1], exp.metrics['coherence'][-1]),
            'recovery_correlation': original_correlation,
            'is_anomalous': compute_ccdi(exp.metrics['correlation'][-1], 
                                       exp.metrics['coherence'][-1]) > 0.08
        }
        
        # Compute memory purity - how much does the recall match only this pattern?
        # Check similarity to other patterns
        purities = []
        for other_id, other_pattern in self.phase6.memory_bank.items():
            if other_id == pattern_id:
                continue
                
            if other_pattern.final_state is not None:
                other_corr = np.corrcoef(other_pattern.final_state.flatten(), 
                                      final_state.flatten())[0, 1]
                
                # Memory is pure if correlation with target is high and with others is low
                purity = original_correlation - other_corr
                purities.append(purity)
        
        # Average purity (higher is better - more distinct from other memories)
        if purities:
            recall_metrics['memory_purity'] = np.mean(purities)
        
        return recall_metrics


class SelectiveRecallTester:
    """Tests which memory reactivates after perturbation or noise"""
    
    def __init__(self, phase6):
        """
        Initialize the recall tester
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
    
    def test_targeted_recall(self, target_pattern_id, perturbation_type="flip", 
                           perturbation_strength=1.0, steps=50):
        """
        Test targeted recall of a specific pattern
        
        Parameters:
        -----------
        target_pattern_id : str
            ID of pattern to target for recall
        perturbation_type : str
            Type of perturbation to apply
        perturbation_strength : float
            Magnitude of perturbation
        steps : int
            Number of steps to let system evolve
            
        Returns:
        --------
        dict
            Recall metrics
        """
        # Implementation will go here
        pass
    
    def test_spontaneous_recall(self, noise_level=0.5, steps=50):
        """
        Test which memory spontaneously recovers from noise
        
        Parameters:
        -----------
        noise_level : float
            Level of noise to initialize with (0.0 to 1.0)
        steps : int
            Number of steps to let system evolve
            
        Returns:
        --------
        dict
            Recall metrics including which pattern emerged
        """
        # Implementation will go here
        pass


class CounterfactualDisruptor:
    """Injects a false memory during or after encoding to observe the effects"""
    
    def __init__(self, phase6):
        """
        Initialize the counterfactual disruptor
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
    
    def run_counterfactual_experiment(self, pattern_ids, counterfactual_id, 
                                    counterfactual_similarity, intrusion_strengths):
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
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        # Implementation will go here
        pass
    
    def create_counterfactual(self, based_on_pattern_id, similarity_level=0.5):
        """
        Create a counterfactual pattern based on an existing one
        
        Parameters:
        -----------
        based_on_pattern_id : str
            ID of pattern to base counterfactual on
        similarity_level : float
            How similar to make the counterfactual (0.0 to 1.0)
            
        Returns:
        --------
        MemoryTrace
            The counterfactual memory trace
        """
        # Implementation will go here
        pass


class MemoryBlender:
    """Deliberately mixes inputs from patterns to study hybridization"""
    
    def __init__(self, phase6):
        """
        Initialize the memory blender
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
    
    def run_blend_recovery_experiment(self, pattern_ids, blend_ratios, perturbation_types):
        """
        Run blending and recovery experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to blend
        blend_ratios : list
            List of blend ratios to test (0.0 to 1.0)
        perturbation_types : list
            List of perturbation types to test
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        # Implementation will go here
        pass
    
    def blend_patterns(self, pattern_a_id, pattern_b_id, ratio=0.5, method="pixel"):
        """
        Blend two patterns together
        
        Parameters:
        -----------
        pattern_a_id : str
            ID of first pattern
        pattern_b_id : str
            ID of second pattern
        ratio : float
            Blend ratio (0.0 = all A, 1.0 = all B)
        method : str
            Blending method: "pixel", "frequency", or "structured"
            
        Returns:
        --------
        ndarray
            Blended pattern
        """
        # Implementation will go here
        pass


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
    
    def run_context_switching_experiment(self, pattern_ids, cue_strengths, context_variations):
        """
        Run context switching experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to switch between
        cue_strengths : list
            List of cue strengths to test
        context_variations : list
            List of context variations to test
            
        Returns:
        --------
        list
            List of result dictionaries
        """
        # Implementation will go here
        pass
    
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
        # Implementation will go here
        pass