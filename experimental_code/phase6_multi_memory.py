# phase6_multi_memory.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from phase6.multi_memory_injector import MultiMemoryInjector
from phase6.selective_recall_tester import SelectiveRecallTester
from phase6.counterfactual_disruptor import CounterfactualDisruptor
from phase6.memory_blender import MemoryBlender
from phase6.context_switcher import ContextSwitcher

from rcft_framework import RCFTExperiment
from rcft_metrics import compute_ccdi, compute_mutual_information, compute_field_statistics



class MemoryTrace:
    """Class representing a memory trace with its attractor state and metrics"""
    
    def __init__(self, pattern_id, initial_state, final_state=None, fingerprint=None, metrics=None):
        """
        Initialize a memory trace
        
        Parameters:
        -----------
        pattern_id : str
            Identifier for this memory
        initial_state : ndarray
            Initial state pattern
        final_state : ndarray, optional
            Final attractor state (if already stabilized)
        fingerprint : ndarray, optional
            Compressed fingerprint representation
        metrics : dict, optional
            Dictionary of metrics for this memory
        """
        self.pattern_id = pattern_id
        self.initial_state = initial_state.copy()
        self.final_state = final_state.copy() if final_state is not None else None
        self.fingerprint = fingerprint
        self.metrics = metrics if metrics is not None else {}
        self.history = []  # For tracking changes to this memory

    def compute_fingerprint(self, n_components=20):
        """Compute a dimensionality-reduced fingerprint of the memory state"""
        if self.final_state is None:
            return None
            
        # Reshape for PCA
        flattened = self.final_state.flatten().reshape(1, -1)
        
        # Handle the single sample case - can't do PCA with more components than samples
        if n_components > 1:
            # Just store the raw flattened data or a sampled version
            # Option 1: Return a downsampled version of the flattened array
            step = max(1, len(flattened[0]) // n_components)
            fingerprint = flattened[0][::step][:n_components]
            
            # Option 2: Or just return a normalized version of the raw data
            # fingerprint = flattened[0][:n_components]
        else:
            # Use PCA with just 1 component if needed
            pca = PCA(n_components=1)
            fingerprint = pca.fit_transform(flattened)[0]
        
        self.fingerprint = fingerprint
        return fingerprint
        
    def similarity_to(self, other_trace, method='correlation'):
        """Calculate similarity to another memory trace"""
        if self.final_state is None or other_trace.final_state is None:
            return 0.0
            
        if method == 'correlation':
            return np.corrcoef(self.final_state.flatten(), 
                             other_trace.final_state.flatten())[0, 1]
        elif method == 'mutual_information':
            return compute_mutual_information(self.final_state, other_trace.final_state)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    def record_state(self, state, metrics=None):
        """Record a new state in this memory's history"""
        self.history.append({
            'state': state.copy(),
            'metrics': metrics.copy() if metrics is not None else {}
        })
        
    def __repr__(self):
        """String representation"""
        return f"MemoryTrace('{self.pattern_id}', metrics={list(self.metrics.keys()) if self.metrics else 'None'})"


class Phase6MultiMemory:
    """Main class for Phase VI: Multi-Memory Coexistence experiments"""
    
    def __init__(self, output_dir="phase6_results", alpha=0.35, beta=0.5, gamma=0.92, 
                pattern_size=64, max_steps=100, log_level=logging.INFO):
        """
        Initialize the Phase VI experimental framework
        
        Parameters:
        -----------
        output_dir : str
            Directory for saving results
        alpha : float
            Memory strength parameter (α)
        beta : float
            Coupling strength parameter (β)
        gamma : float
            Memory decay parameter (γ)
        pattern_size : int
            Size of pattern grid (default: 64x64)
        max_steps : int
            Maximum steps for evolution
        log_level : logging level
            Logging verbosity
        """
        self.output_dir = output_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pattern_size = pattern_size
        self.max_steps = max_steps
        
        # Setup directories
        self._setup_directories()
        self._setup_logging(log_level)
        
        # Initialize memory bank
        self.memory_bank = {}  # Store memory traces
        self.base_experiment = None  # RCFT experiment instance for operations
        
        # Initialize modules
        self.injector = MultiMemoryInjector(self)
        self.recall_tester = SelectiveRecallTester(self)
        self.disruptor = CounterfactualDisruptor(self, echo_depth=0)  # Default to no echo protection
        self.blender = MemoryBlender(self)
        self.switcher = ContextSwitcher(self)
        
        # Setup base experiment
        self._initialize_base_experiment()
        
        self.logger.info(f"Phase VI initialized with alpha={alpha}, beta={beta}, gamma={gamma}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        # Main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sub-directories based on output structure from spec
        for subdir in ["injection", "recall_test", "counterfactual", 
                      "blending", "switching", "summary"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def _setup_logging(self, log_level):
        """Configure logging"""
        log_file = os.path.join(self.output_dir, "phase6.log")
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("Phase6")
        
    def _initialize_base_experiment(self):
        """Initialize base RCFT experiment"""
        self.base_experiment = RCFTExperiment(
            size=self.pattern_size,
            memory_strength=self.alpha,
            coupling_strength=self.beta,
            memory_decay=self.gamma
        )
        
    def create_pattern(self, pattern_id, pattern_type="fractal", **kwargs):
        """
        Create a pattern and add it to the memory bank
        
        Parameters:
        -----------
        pattern_id : str
            Identifier for this pattern
        pattern_type : str
            Type of pattern to create (fractal, radial, etc.)
        **kwargs : dict
            Additional parameters for pattern creation
        
        Returns:
        --------
        MemoryTrace
            The created memory trace
        """
        # Initialize with pattern
        self.base_experiment.initialize_pattern(pattern_type=pattern_type, **kwargs)
        
        # Get the initial state
        initial_state = self.base_experiment.state.copy()
        
        # Create memory trace
        memory_trace = MemoryTrace(pattern_id, initial_state)
        
        # Add to memory bank
        self.memory_bank[pattern_id] = memory_trace
        
        self.logger.info(f"Created pattern '{pattern_id}' of type '{pattern_type}'")
        return memory_trace
    
    def run_interference_mapping(self, pattern_ids=None, encoding_strengths=None, decay_rates=None,
                               delays=None, perturbation_strengths=None):
        """
        Run Experiment E1: Interference Mapping
        Train A → Train B → Recall A. Sweep across parameters.
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to use (default: ['A', 'B'])
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
        pd.DataFrame
            Results of the interference mapping
        """
        if pattern_ids is None:
            pattern_ids = ['A', 'B']
            
        if encoding_strengths is None:
            encoding_strengths = [0.2, 0.35, 0.5]
            
        if decay_rates is None:
            decay_rates = [0.8, 0.92, 0.98]
            
        if delays is None:
            delays = [5, 10, 20]
            
        if perturbation_strengths is None:
            perturbation_strengths = [0.5, 1.0, 1.5]
            
        # Create patterns if they don't exist
        for pid in pattern_ids:
            if pid not in self.memory_bank:
                self.create_pattern(pid)
        
        self.logger.info(f"Running interference mapping with patterns: {pattern_ids}")
        
        # Setup experiment
        results = self.injector.run_sequential_encoding_experiment(
            pattern_ids=pattern_ids,
            encoding_strengths=encoding_strengths,
            decay_rates=decay_rates,
            delays=delays,
            perturbation_strengths=perturbation_strengths
        )
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "summary", "interference_matrix.csv"), index=False)
        
        # Generate interference matrix visualization
        self._generate_interference_matrix_viz(results_df)
        
        return results_df
    
    def run_recombination_thresholds(self, pattern_ids=None, blend_ratios=None, 
                                perturbation_types=None, blend_methods=None):
        """
        Run Experiment E2: Recombination Thresholds
        Blend A & B at varying proportions, perturb, then analyze resulting recovery.
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to use (default: ['A', 'B'])
        blend_ratios : list
            List of blend ratios to test (0.0 to 1.0)
        perturbation_types : list
            List of perturbation types to test
        blend_methods : list
            List of blend methods to test (default: ["pixel", "structured", "frequency"])
                
        Returns:
        --------
        pd.DataFrame
            Results of the recombination threshold experiment
        """
        if pattern_ids is None:
            pattern_ids = ['A', 'B']
            
        if blend_ratios is None:
            blend_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
            
        if perturbation_types is None:
            perturbation_types = ["flip", "noise", "zero"]
            
        if blend_methods is None:
            blend_methods = ["pixel", "structured", "frequency"]
            
        # Create patterns if they don't exist
        for pid in pattern_ids:
            if pid not in self.memory_bank:
                self.create_pattern(pid)
        
        self.logger.info(f"Running recombination thresholds with patterns: {pattern_ids}")
        
        # Setup experiment
        results = self.blender.run_blend_recovery_experiment(
            pattern_ids=pattern_ids,
            blend_ratios=blend_ratios,
            perturbation_types=perturbation_types,
            blend_methods=blend_methods
        )
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "summary", "hybrid_scores.csv"), index=False)
        
        # Generate hybrid score visualization
        self._generate_hybrid_score_viz(results_df)
        
        return results_df
    
    def run_counterfactual_intrusion(self, pattern_ids=None, counterfactual_similarity=None, intrusion_strengths=None, trials=1):
        """
        Run Experiment E3: Counterfactual Intrusion
        Train A & B, inject C (counterfactual), test for corruption of A/B.
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to use (default: ['A', 'B'])
        counterfactual_similarity : list
            List of similarity levels to test (0.0 to 1.0)
        intrusion_strengths : list
            List of intrusion strengths to test
        trials : int
            Number of trials to run for each parameter combination
            
        Returns:
        --------
        pd.DataFrame
            Results of the counterfactual intrusion experiment
        """
        if pattern_ids is None:
            pattern_ids = ['A', 'B']
            
        if counterfactual_similarity is None:
            counterfactual_similarity = [0.4]
            
        if intrusion_strengths is None:
            intrusion_strengths = [0.3, 0.6, 0.9]
            
        # Create patterns if they don't exist
        for pid in pattern_ids:
            if pid not in self.memory_bank:
                self.create_pattern(pid)
        
        self.logger.info(f"Running counterfactual intrusion with base patterns: {pattern_ids}")
        
        # Setup experiment
        results = self.disruptor.run_counterfactual_experiment(
            pattern_ids=pattern_ids,
            counterfactual_similarity=counterfactual_similarity,
            intrusion_strengths=intrusion_strengths,
            trials=trials  # Pass the trials parameter
        )
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "summary", "counterfactual_intrusion.csv"), index=False)
        
        # Generate counterfactual corruption visualization
        self._generate_counterfactual_viz(results_df)
        
        return results_df
    
    def register_blend(self, source_a_id, source_b_id, blend_id, ratio=0.5, method="pixel"):
        """
        Register a blended pattern as a new memory trace
        
        Parameters:
        -----------
        source_a_id : str
            First source pattern ID
        source_b_id : str
            Second source pattern ID
        blend_id : str
            ID to assign to the blend
        ratio : float
            Blend ratio (0.0 = all A, 1.0 = all B)
        method : str
            Blending method: "pixel", "structured", or "frequency"
            
        Returns:
        --------
        MemoryTrace
            The blended memory trace
        """
        # Use the blender to create the blend
        blended_pattern = self.blender.blend_patterns(
            source_a_id, source_b_id, 
            ratio=ratio, method=method
        )
        
        # Create memory trace
        memory_trace = MemoryTrace(blend_id, blended_pattern)
        
        # Add to memory bank
        self.memory_bank[blend_id] = memory_trace
        
        self.logger.info(f"Registered blend '{blend_id}' of {source_a_id} and {source_b_id} (ratio: {ratio:.2f})")
        
        return memory_trace

    def run_cue_guided_switching(self, pattern_ids=None, cue_strengths=None, context_variations=None):
        """
        Run Experiment E4: Cue-Guided Switching
        Train A & B, then provide partial cue for A or B.
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to use (default: ['A', 'B'])
        cue_strengths : list
            List of cue strengths to test
        context_variations : list
            List of context variations to test
            
        Returns:
        --------
        pd.DataFrame
            Results of the cue-guided switching experiment
        """
        if pattern_ids is None:
            pattern_ids = ['A', 'B']
            
        if cue_strengths is None:
            cue_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
            
        if context_variations is None:
            context_variations = ["spatial", "frequency", "gradient"]
            
        # Create patterns if they don't exist
        for pid in pattern_ids:
            if pid not in self.memory_bank:
                self.create_pattern(pid)
        
        self.logger.info(f"Running cue-guided switching with patterns: {pattern_ids}")
        
        # Setup experiment
        results = self.switcher.run_context_switching_experiment(
            pattern_ids=pattern_ids,
            cue_strengths=cue_strengths,
            context_variations=context_variations
        )
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "summary", "context_switching.csv"), index=False)
        
        # Generate context graph visualization
        self._generate_context_graph_viz(results_df)
        
        # Save context graph
        context_graph = self.switcher.build_context_transition_graph(results_df)
        with open(os.path.join(self.output_dir, "summary", "context_graph.json"), 'w') as f:
            json.dump(context_graph, f, indent=2)
        
        return results_df
    
    def run_all_experiments(self):
        """Run all Phase VI experiments in sequence"""
        self.logger.info("Running all Phase VI experiments")
        
        # Create patterns
        if 'A' not in self.memory_bank:
            self.create_pattern('A', pattern_type='radial')
        if 'B' not in self.memory_bank:
            self.create_pattern('B', pattern_type='diagonal')
        if 'C' not in self.memory_bank:
            self.create_pattern('C', pattern_type='fractal')
        
        # Run experiments
        interference_results = self.run_interference_mapping()
        recombination_results = self.run_recombination_thresholds()
        counterfactual_results = self.run_counterfactual_intrusion()
        switching_results = self.run_cue_guided_switching()
        
        # Generate comprehensive summary
        self._generate_comprehensive_summary(
            interference_results,
            recombination_results,
            counterfactual_results,
            switching_results
        )
        
        self.logger.info("All Phase VI experiments completed")
        
        return {
            'interference': interference_results,
            'recombination': recombination_results,
            'counterfactual': counterfactual_results,
            'switching': switching_results
        }
    
    def _generate_interference_matrix_viz(self, results_df):
        """Generate visualization for interference matrix"""
        # Implementation will go here
        self.logger.info("Generating interference matrix visualization")
        pass
    
    def _generate_hybrid_score_viz(self, results_df):
        """Generate visualization for hybrid scores"""
        # Implementation will go here
        self.logger.info("Generating hybrid score visualization")
        pass
    
    def _generate_counterfactual_viz(self, results_df):
        """Generate visualization for counterfactual intrusion"""
        # Implementation will go here
        self.logger.info("Generating counterfactual visualization")
        pass
    
    def _generate_context_graph_viz(self, results_df):
        """Generate visualization for context switching graph"""
        # Implementation will go here
        self.logger.info("Generating context graph visualization")
        pass
    
    def _generate_comprehensive_summary(self, *result_dfs):
        """Generate comprehensive summary of all experiments"""
        # Implementation will go here
        self.logger.info("Generating comprehensive summary")
        pass