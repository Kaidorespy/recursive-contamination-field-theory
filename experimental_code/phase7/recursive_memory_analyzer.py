import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

class RecursiveMemoryAnalyzer:
    """Utility class for analyzing recursive memory patterns and drift"""
    
    def __init__(self, engine):
        """
        Initialize the recursive memory analyzer
        
        Parameters:
        -----------
        engine : RecursiveContaminationEngine
            Reference to the recursive contamination engine
        """
        self.engine = engine
        self.output_dir = os.path.join(engine.output_dir, "recursive_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_recursive_drift(self, results=None):
        """
        Analyze recursive drift patterns across generations
        
        Parameters:
        -----------
        results : list, optional
            List of result dictionaries (default: use engine.results)
            
        Returns:
        --------
        dict
            Drift analysis results
        """
        if results is None:
            results = self.engine.results
            
        if not results:
            return None
            
        # Group results by CF and generation
        gen_results = {}
        for cf_id, info in self.engine.lineage.items():
            gen = info['generation']
            if gen not in gen_results:
                gen_results[gen] = []
                
            # Find results that used this CF
            cf_results = [r for r in results if r['cf_id'] == cf_id]
            gen_results[gen].extend(cf_results)
        
        # Calculate average metrics by generation
        gen_metrics = {}
        for gen, gen_res in gen_results.items():
            if not gen_res:
                continue
                
            gen_metrics[gen] = {
                'memory_integrity_delta': np.mean([r['metrics']['memory_integrity_delta'] for r in gen_res]),
                'cf_influence': np.mean([r['metrics']['cf_influence'] for r in gen_res]),
                'recursive_drift': np.mean([r['metrics']['recursive_drift'] for r in gen_res]),
                'recovery_bias': np.mean([r['metrics']['recovery_bias'] for r in gen_res if not np.isinf(r['metrics']['recovery_bias'])]),
                'rfi': np.mean([r['metrics']['rfi'] for r in gen_res]),
                'entropy': np.mean([r['metrics']['entropy'] for r in gen_res]),
                'attractor_melting_rate': np.mean([1 if r['metrics']['attractor_melting'] else 0 for r in gen_res])
            }
        
        # Calculate correlations between parent and child CFs
        parent_child_correlations = []
        for cf_id, info in self.engine.lineage.items():
            if info['parent'] is not None:
                parent_state = self.engine.recursive_cfs[info['parent']]['state']
                child_state = self.engine.recursive_cfs[cf_id]['state']
                
                corr = np.corrcoef(parent_state.flatten(), child_state.flatten())[0, 1]
                parent_child_correlations.append({
                    'parent': info['parent'],
                    'child': cf_id,
                    'correlation': corr,
                    'parent_gen': self.engine.lineage[info['parent']]['generation'],
                    'child_gen': info['generation']
                })
        
        # Visualize generation metrics
        self._visualize_generation_metrics(gen_metrics)
        
        # Visualize parent-child correlations
        if parent_child_correlations:
            self._visualize_parent_child_correlations(parent_child_correlations)
        
        # Perform dimensionality reduction to visualize CF space
        self._visualize_cf_space()
        
        return {
            'generation_metrics': gen_metrics,
            'parent_child_correlations': parent_child_correlations
        }
    
    def analyze_contamination_patterns(self):
        """
        Analyze patterns of contamination across recursive generations
        
        Returns:
        --------
        dict
            Contamination pattern analysis
        """
        if not self.engine.results:
            return None
            
        # Collect all final states
        states = {}
        for result in self.engine.results:
            states[result['id']] = result['final_state']
            
        # Add base patterns
        base_patterns = {}
        for pid, pattern in self.engine.phase6.memory_bank.items():
            base_patterns[pid] = pattern.initial_state
            
        # Add CFs
        for cf_id, cf_info in self.engine.recursive_cfs.items():
            states[cf_id] = cf_info['state']
            
        # Calculate pairwise correlations
        correlations = pd.DataFrame(index=list(states.keys()) + list(base_patterns.keys()),
                                 columns=list(states.keys()) + list(base_patterns.keys()))
        
        # Fill with NaN initially
        correlations = correlations.fillna(np.nan)
        
        # Calculate correlations
        for i, (id1, state1) in enumerate(list(states.items()) + list(base_patterns.items())):
            # Only calculate upper triangle to avoid redundancy
            for id2, state2 in list(list(states.items()) + list(base_patterns.items()))[i:]:
                if id1 != id2:  # Skip self-correlation
                    corr = np.corrcoef(state1.flatten(), state2.flatten())[0, 1]
                    correlations.loc[id1, id2] = corr
                    correlations.loc[id2, id1] = corr  # Symmetrical
                else:
                    correlations.loc[id1, id2] = 1.0  # Self-correlation = 1
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, cmap='viridis', vmin=-1, vmax=1)
        plt.title('Pairwise Correlations Between States')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "state_correlation_heatmap.png"))
        plt.close()
        
        # Find clusters/convergent points
        # Identify groups with high correlation (> 0.8) as potential convergence points
        convergence_groups = []
        processed = set()
        
        for id1 in correlations.index:
            if id1 in processed:
                continue
                
            # Find highly correlated states
            group = [id1]
            processed.add(id1)
            
            for id2 in correlations.index:
                if id2 != id1 and id2 not in processed:
                    if correlations.loc[id1, id2] > 0.8:
                        group.append(id2)
                        processed.add(id2)
            
            if len(group) > 1:
                convergence_groups.append(group)
        
        # Visualize convergence groups
        if convergence_groups:
            self._visualize_convergence_groups(convergence_groups, states, base_patterns)
            
        # Calculate contamination metrics
        contamination_metrics = {
            'convergence_groups': convergence_groups,
            'avg_self_correlation': np.nanmean([
                correlations.loc[result['id'], result['cf_id']] 
                for result in self.engine.results 
                if result['id'] in correlations.index and result['cf_id'] in correlations.columns
            ]),
            'correlation_matrix': correlations
        }
        
        return contamination_metrics
    
    def compute_recursive_fragility_index(self):
        """
        Compute the Recursive Fragility Index for each generation
        
        Returns:
        --------
        dict
            RFI values by generation
        """
        # Calculate average RFI by generation
        rfi_by_gen = {}
        
        for result in self.engine.results:
            cf_id = result['cf_id']
            if cf_id in self.engine.lineage:
                gen = self.engine.lineage[cf_id]['generation']
                if gen not in rfi_by_gen:
                    rfi_by_gen[gen] = []
                rfi_by_gen[gen].append(result['metrics']['rfi'])
        
        # Compute mean RFI by generation
        mean_rfi = {gen: np.mean(values) for gen, values in rfi_by_gen.items()}
        
        # Visualize RFI by generation
        plt.figure(figsize=(10, 6))
        
        generations = sorted(mean_rfi.keys())
        rfi_values = [mean_rfi[gen] for gen in generations]
        
        plt.bar(generations, rfi_values, color='skyblue')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Moderate Fragility')
        plt.axhline(y=1.0, color='darkred', linestyle='--', label='High Fragility')
        
        plt.title('Recursive Fragility Index by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Average RFI')
        plt.xticks(generations)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rfi_by_generation.png"))
        plt.close()
        
        return mean_rfi
    
    def _visualize_generation_metrics(self, gen_metrics):
        """Visualize metrics by generation"""
        if not gen_metrics:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Setup
        generations = sorted(gen_metrics.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
        
        # Plot memory integrity delta
        plt.subplot(2, 3, 1)
        values = [gen_metrics[gen]['memory_integrity_delta'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('Memory Integrity Delta')
        plt.xlabel('Generation')
        plt.ylabel('Average Δ')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot CF influence
        plt.subplot(2, 3, 2)
        values = [gen_metrics[gen]['cf_influence'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('CF Influence')
        plt.xlabel('Generation')
        plt.ylabel('Average Influence')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot recursive drift
        plt.subplot(2, 3, 3)
        values = [gen_metrics[gen]['recursive_drift'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('Recursive Drift')
        plt.xlabel('Generation')
        plt.ylabel('Average Drift')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot recovery bias
        plt.subplot(2, 3, 4)
        values = [gen_metrics[gen]['recovery_bias'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('Recovery Bias')
        plt.xlabel('Generation')
        plt.ylabel('Average Bias Ratio')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot RFI
        plt.subplot(2, 3, 5)
        values = [gen_metrics[gen]['rfi'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('Recursive Fragility Index')
        plt.xlabel('Generation')
        plt.ylabel('Average RFI')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot attractor melting rate
        plt.subplot(2, 3, 6)
        values = [gen_metrics[gen]['attractor_melting_rate'] for gen in generations]
        plt.bar(generations, values, color=colors)
        plt.title('Attractor Melting Rate')
        plt.xlabel('Generation')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Metrics by Generation', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "generation_metrics.png"))
        plt.close()
    
    def _visualize_parent_child_correlations(self, parent_child_correlations):
        """Visualize correlations between parent and child CFs"""
        plt.figure(figsize=(10, 6))
        
        # Extract data
        generations = []
        correlations = []
        
        for item in parent_child_correlations:
            generations.append(item['child_gen'])
            correlations.append(item['correlation'])
        
        # Plot by generation
        data = pd.DataFrame({
            'Generation': generations,
            'Parent-Child Correlation': correlations
        })
        
        sns.boxplot(x='Generation', y='Parent-Child Correlation', data=data)
        
        plt.title('Parent-Child Correlation by Generation')
        plt.xlabel('Child Generation')
        plt.ylabel('Correlation to Parent')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "parent_child_correlations.png"))
        plt.close()
    
    def _visualize_cf_space(self):
        """Visualize CFs in a dimensionality-reduced space"""
        if not self.engine.recursive_cfs:
            return
            
        # Collect all CF states
        cf_states = []
        cf_ids = []
        cf_generations = []
        
        for cf_id, cf_info in self.engine.recursive_cfs.items():
            cf_states.append(cf_info['state'].flatten())
            cf_ids.append(cf_id)
            gen = self.engine.lineage[cf_id]['generation'] if cf_id in self.engine.lineage else 0
            cf_generations.append(gen)
        
        # Add base patterns
        for pid, pattern in self.engine.phase6.memory_bank.items():
            cf_states.append(pattern.initial_state.flatten())
            cf_ids.append(pid)
            cf_generations.append(0)  # Base patterns are generation 0
        
        # Convert to numpy array
        X = np.array(cf_states)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Apply t-SNE for dimensionality reduction
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1) if len(X) > 1 else 1)
            X_tsne = tsne.fit_transform(X)
            tsne_success = True
        except:
            tsne_success = False
        
        # Create PCA visualization
        plt.figure(figsize=(12, 10))
        
        # Plot in PCA space
        plt.subplot(2, 1, 1)
        
        # Color by generation
        unique_generations = sorted(set(cf_generations))
        colors = plt.cm.viridis(np.linspace(0, 1, max(unique_generations) + 1))
        
        for i, (x, y) in enumerate(X_pca):
            gen = cf_generations[i]
            color = colors[gen]
            marker = 'o' if gen > 0 else '*'  # Star for base patterns
            size = 100 if gen == 0 else 80 if gen == 1 else 60
            
            plt.scatter(x, y, color=color, s=size, marker=marker, alpha=0.8, edgecolor='black')
            plt.text(x, y, cf_ids[i], fontsize=8)
        
        # Add a legend
        for gen in unique_generations:
            label = 'Base Pattern' if gen == 0 else f'Generation {gen}'
            marker = '*' if gen == 0 else 'o'
            plt.scatter([], [], color=colors[gen], s=80, marker=marker, label=label)
            
        plt.title('CF Space Projection (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot in t-SNE space if successful
        if tsne_success:
            plt.subplot(2, 1, 2)
            
            for i, (x, y) in enumerate(X_tsne):
                gen = cf_generations[i]
                color = colors[gen]
                marker = 'o' if gen > 0 else '*'  # Star for base patterns
                size = 100 if gen == 0 else 80 if gen == 1 else 60
                
                plt.scatter(x, y, color=color, s=size, marker=marker, alpha=0.8, edgecolor='black')
                plt.text(x, y, cf_ids[i], fontsize=8)
            
            plt.title('CF Space Projection (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cf_space_projection.png"))
        plt.close()
    
    def _visualize_convergence_groups(self, convergence_groups, states, base_patterns):
        """Visualize convergence groups of states"""
        plt.figure(figsize=(15, 10))
        
        # Combine all states
        all_states = {**states, **base_patterns}
        
        # Plot up to 6 groups
        for i, group in enumerate(convergence_groups[:min(6, len(convergence_groups))]):
            plt.subplot(2, 3, i+1)
            
            # Plot the first state in the group
            first_id = group[0]
            plt.imshow(all_states[first_id], cmap='viridis', vmin=-1, vmax=1)
            plt.title(f'Group {i+1}: {len(group)} states\nRepresentative: {first_id}')
            plt.axis('off')
            
            # Add a list of members
            members_text = '\n'.join([f"- {id}" for id in group[:5]])
            if len(group) > 5:
                members_text += f"\n- ...and {len(group) - 5} more"
                
            plt.figtext(0.01 + (i % 3) * 0.33, 0.95 - (i // 3) * 0.5, members_text, fontsize=8)
        
        plt.suptitle('Convergent State Groups', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "convergence_groups.png"))
        plt.close()