import numpy as np
import matplotlib.pyplot as plt
import os
from rcft_metrics import compute_ccdi

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
        self.output_dir = os.path.join(phase6.output_dir, "recall_test")
        os.makedirs(self.output_dir, exist_ok=True)
    
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
        # Get target pattern
        if target_pattern_id not in self.phase6.memory_bank:
            raise ValueError(f"Target pattern {target_pattern_id} not found in memory bank")
            
        target_pattern = self.phase6.memory_bank[target_pattern_id]
        
        # Initialize experiment with target pattern's final state
        exp = self.phase6.base_experiment
        if target_pattern.final_state is not None:
            exp.state = target_pattern.final_state.copy()
        else:
            # If no final state, use initial state
            exp.state = target_pattern.initial_state.copy()
        
        # Apply perturbation
        initial_state = exp.state.copy()
        exp.apply_perturbation(
            perturbation_type=perturbation_type,
            magnitude=perturbation_strength,
            radius=15
        )
        perturbed_state = exp.state.copy()
        
        # Save pre-recovery states
        np.save(os.path.join(self.output_dir, f"{target_pattern_id}_initial.npy"), initial_state)
        np.save(os.path.join(self.output_dir, f"{target_pattern_id}_perturbed.npy"), perturbed_state)
        
        # Let system evolve for recovery
        exp.update(steps=steps)
        
        # Capture final state
        final_state = exp.state.copy()
        np.save(os.path.join(self.output_dir, f"{target_pattern_id}_recovered.npy"), final_state)
        
        # Calculate recall metrics for target pattern
        target_correlation = np.corrcoef(target_pattern.initial_state.flatten(), 
                                      final_state.flatten())[0, 1]
        
        # Calculate comparison to all other patterns (confusion matrix row)
        confusion_scores = {}
        for pattern_id, pattern in self.phase6.memory_bank.items():
            correlation = np.corrcoef(pattern.initial_state.flatten(), 
                                   final_state.flatten())[0, 1]
            confusion_scores[pattern_id] = correlation
            
        # Calculate competition index - ratio of target to highest competitor
        competitors = [v for k, v in confusion_scores.items() if k != target_pattern_id]
        competition_index = 0.0
        if competitors and max(competitors) > 0:
            competition_index = target_correlation / max(competitors)
        
        # Calculate CCDI
        ccdi = compute_ccdi(exp.metrics['correlation'][-1], exp.metrics['coherence'][-1])
        
        # Compile metrics
        metrics = {
            'target_pattern': target_pattern_id,
            'perturbation_type': perturbation_type,
            'perturbation_strength': perturbation_strength,
            'target_correlation': target_correlation,
            'competition_index': competition_index,
            'confusion_scores': confusion_scores,
            'recovery_quality': exp.metrics['correlation'][-1],
            'coherence': exp.metrics['coherence'][-1],
            'mutual_info': exp.metrics['mutual_info'][-1],
            'ccdi': ccdi,
            'is_anomalous': ccdi > 0.08
        }
        
        # Create confusion visualization
        self._visualize_targeted_recall(metrics, final_state)
        
        return metrics
    
    def test_spontaneous_recall(self, noise_level=0.5, steps=100, trials=5, seed=None):
        """
        Test which memory spontaneously recovers from noise
        
        Parameters:
        -----------
        noise_level : float
            Level of noise to initialize with (0.0 to 1.0)
        steps : int
            Number of steps to let system evolve
        trials : int
            Number of trials to run
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        list
            List of recall metrics including which pattern emerged
        """
        if seed is not None:
            np.random.seed(seed)
            
        results = []
        
        for trial in range(trials):
            # Generate random noise field
            noise_field = np.random.uniform(-1, 1, (self.phase6.pattern_size, self.phase6.pattern_size))
            
            # Scale noise by level parameter (1.0 = pure noise, 0.0 = no noise)
            scaled_noise = noise_field * noise_level
            
            # Initialize experiment with noise
            exp = self.phase6.base_experiment
            exp.state = scaled_noise.copy()
            
            # Save initial noisy state
            initial_state = exp.state.copy()
            np.save(os.path.join(self.output_dir, f"spontaneous_trial{trial}_initial.npy"), initial_state)
            
            # Let system evolve
            exp.update(steps=steps)
            
            # Capture final state
            final_state = exp.state.copy()
            np.save(os.path.join(self.output_dir, f"spontaneous_trial{trial}_final.npy"), final_state)
            
            # Calculate similarity to all patterns
            similarity_scores = {}
            for pattern_id, pattern in self.phase6.memory_bank.items():
                correlation = np.corrcoef(pattern.initial_state.flatten(), 
                                       final_state.flatten())[0, 1]
                similarity_scores[pattern_id] = correlation
            
            # Determine which pattern emerged (if any)
            best_match = max(similarity_scores.items(), key=lambda x: x[1])
            emergent_pattern = best_match[0]
            emergent_strength = best_match[1]
            
            # Calculate dominance ratio (how much stronger is the winner vs. runner-up)
            sorted_scores = sorted(similarity_scores.values(), reverse=True)
            dominance_ratio = 0.0
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                dominance_ratio = sorted_scores[0] / sorted_scores[1]
            
            # Calculate field metrics
            field_stats = compute_field_statistics(final_state)
            ccdi = compute_ccdi(exp.metrics['correlation'][-1], exp.metrics['coherence'][-1])
            
            # Compile trial results
            trial_result = {
                'trial': trial,
                'noise_level': noise_level,
                'emergent_pattern': emergent_pattern,
                'emergent_strength': emergent_strength,
                'dominance_ratio': dominance_ratio,
                'similarity_scores': similarity_scores,
                'coherence': exp.metrics['coherence'][-1],
                'entropy': field_stats['entropy'],
                'ccdi': ccdi,
                'skewness': field_stats['skewness'],
                'is_anomalous': ccdi > 0.08
            }
            
            results.append(trial_result)
            
            # Create visualization for this trial
            self._visualize_spontaneous_recall(trial_result, initial_state, final_state)
            
        # Calculate summary statistics
        pattern_counts = {}
        for result in results:
            pattern = result['emergent_pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        # Create bias ratio - probability of each pattern emerging
        total_trials = len(results)
        bias_ratios = {pattern: count/total_trials for pattern, count in pattern_counts.items()}
        
        # Add summary to results
        summary = {
            'pattern_counts': pattern_counts,
            'bias_ratios': bias_ratios,
            'strongest_bias': max(bias_ratios.items(), key=lambda x: x[1])[0],
            'noise_level': noise_level,
            'total_trials': total_trials
        }
        
        # Create summary visualization
        self._visualize_spontaneous_summary(results, summary)
        
        # Return individual trial results and summary
        return {
            'trials': results,
            'summary': summary
        }
    
    def _visualize_targeted_recall(self, metrics, final_state):
        """Create visualization for targeted recall test"""
        plt.figure(figsize=(12, 8))
        
        # Plot recovered state
        plt.subplot(2, 2, 1)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Recovered State\nTarget: {metrics['target_pattern']}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot confusion matrix as bar chart
        plt.subplot(2, 2, 2)
        patterns = list(metrics['confusion_scores'].keys())
        scores = list(metrics['confusion_scores'].values())
        
        # Color bars - target in blue, others in gray
        colors = ['blue' if p == metrics['target_pattern'] else 'gray' for p in patterns]
        
        plt.bar(patterns, scores, color=colors)
        plt.axhline(y=0.7, color='green', linestyle='--', label='Strong Recovery')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Weak Recovery')
        plt.title('Pattern Similarity Scores')
        plt.ylabel('Correlation')
        plt.ylim(0, 1)
        plt.legend()
        
        # Plot metrics
        plt.subplot(2, 2, 3)
        metric_names = ['target_correlation', 'competition_index', 'recovery_quality', 'ccdi']
        metric_values = [metrics[m] for m in metric_names]
        
        plt.bar(metric_names, metric_values)
        plt.title('Recall Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, max(2.0, max(metric_values) * 1.1))  # Scale based on max value
        
        # Add summary text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = [
            f"Target Pattern: {metrics['target_pattern']}",
            f"Perturbation: {metrics['perturbation_type']} ({metrics['perturbation_strength']})",
            f"Recovery Quality: {metrics['recovery_quality']:.3f}",
            f"Competition Index: {metrics['competition_index']:.3f}",
            f"CCDI: {metrics['ccdi']:.3f}",
            f"Anomalous: {'Yes' if metrics['is_anomalous'] else 'No'}"
        ]
        
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"targeted_{metrics['target_pattern']}_p{metrics['perturbation_strength']}.png"))
        plt.close()
    
    def _visualize_spontaneous_recall(self, trial_result, initial_state, final_state):
        """Create visualization for a single spontaneous recall trial"""
        plt.figure(figsize=(12, 8))
        
        # Plot initial noisy state
        plt.subplot(2, 2, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial Noisy State (Level: {trial_result['noise_level']:.2f})")
        plt.colorbar()
        plt.axis('off')
        
        # Plot final state
        plt.subplot(2, 2, 2)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Emerged State (Trial {trial_result['trial']})")
        plt.colorbar()
        plt.axis('off')
        
        # Plot similarity scores
        plt.subplot(2, 2, 3)
        patterns = list(trial_result['similarity_scores'].keys())
        scores = list(trial_result['similarity_scores'].values())
        
        # Color bars - emergent pattern in green, others in gray
        colors = ['green' if p == trial_result['emergent_pattern'] else 'gray' for p in patterns]
        
        plt.bar(patterns, scores, color=colors)
        plt.axhline(y=0.7, color='green', linestyle='--', label='Strong Match')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Weak Match')
        plt.title('Pattern Similarity Scores')
        plt.ylabel('Correlation')
        plt.ylim(0, 1)
        plt.legend()
        
        # Add summary text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = [
            f"Emergent Pattern: {trial_result['emergent_pattern']}",
            f"Emergent Strength: {trial_result['emergent_strength']:.3f}",
            f"Dominance Ratio: {trial_result['dominance_ratio']:.3f}",
            f"Coherence: {trial_result['coherence']:.3f}",
            f"CCDI: {trial_result['ccdi']:.3f}",
            f"Entropy: {trial_result['entropy']:.3f}",
            f"Anomalous: {'Yes' if trial_result['is_anomalous'] else 'No'}"
        ]
        
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"spontaneous_trial{trial_result['trial']}.png"))
        plt.close()
    
    def _visualize_spontaneous_summary(self, results, summary):
        """Create summary visualization for spontaneous recall trials"""
        plt.figure(figsize=(12, 8))
        
        # Plot pattern counts as pie chart
        plt.subplot(2, 2, 1)
        labels = list(summary['pattern_counts'].keys())
        sizes = list(summary['pattern_counts'].values())
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Emergent Pattern Distribution')
        
        # Plot strength by pattern
        plt.subplot(2, 2, 2)
        
        # Group by emergent pattern
        pattern_strengths = {}
        for result in results:
            pattern = result['emergent_pattern']
            if pattern not in pattern_strengths:
                pattern_strengths[pattern] = []
            pattern_strengths[pattern].append(result['emergent_strength'])
        
        # Create box plot
        plt.boxplot([pattern_strengths[p] for p in labels], labels=labels)
        plt.title('Emergent Strength by Pattern')
        plt.ylabel('Correlation Strength')
        
        # Plot bias ratio
        plt.subplot(2, 2, 3)
        bias_labels = list(summary['bias_ratios'].keys())
        bias_values = list(summary['bias_ratios'].values())
        
        plt.bar(bias_labels, bias_values)
        plt.title('Reactivation Bias Ratio')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add summary text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = [
            f"Total Trials: {summary['total_trials']}",
            f"Noise Level: {summary['noise_level']:.2f}",
            f"Strongest Bias: {summary['strongest_bias']}",
            f"Bias Ratio: {summary['bias_ratios'][summary['strongest_bias']]:.3f}",
            f"",
            f"Pattern Counts:"
        ]
        
        for pattern, count in summary['pattern_counts'].items():
            summary_text.append(f"  {pattern}: {count}/{summary['total_trials']}")
        
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"spontaneous_summary.png"))
        plt.close()