"""
Morphospace Projector - Module 4 for Phase 3

Creates a morphospace map of memory types and attractor classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

from phase3.phase3_core import Phase3Core, FalseAttractorSample

class MorphospaceProjector(Phase3Core):
    """Creates a morphospace map of memory types and attractor classes"""
    
    def __init__(self, output_dir="phase3_results/morphospace_projection", log_dir="phase3_logs"):
        """Initialize the morphospace projector"""
        super().__init__(output_dir, log_dir)
        
        # Projection parameters
        self.fingerprint_length = 10  # Points to sample for trajectory fingerprints
        self.n_components = 10  # Number of PCA components to keep
        self.n_clusters = 5  # Number of clusters for K-means
        self.umap_n_neighbors = 15  # UMAP hyperparameter
        self.umap_min_dist = 0.1  # UMAP hyperparameter
        self.tsne_perplexity = 30  # t-SNE hyperparameter
        self.random_state = 42  # For reproducibility
        
        # Source directories
        self.source_dirs = []
        
        # Run ID for this experiment
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_source_directory(self, source_dir):
        """
        Add a source directory containing samples
        
        Parameters:
        -----------
        source_dir : str
            Path to directory containing samples
        """
        if os.path.exists(source_dir):
            self.source_dirs.append(source_dir)
            self.logger.info(f"Added source directory: {source_dir}")
        else:
            self.logger.warning(f"Source directory not found: {source_dir}")
    
    def run_morphospace_projection(self, samples=None, projection_method='all'):
        """
        Run morphospace projection analysis
        
        Parameters:
        -----------
        samples : list, optional
            List of FalseAttractorSample objects to project
        projection_method : str
            Projection method: 'pca', 'tsne', 'umap', or 'all'
            
        Returns:
        --------
        tuple
            (projections_df, clusters_df) - DataFrames with projection coordinates and clusters
        """
        # Load samples if not provided
        if samples is None or len(samples) == 0:
            samples = self.load_all_samples()
            
        if not samples:
            self.logger.error("No samples available for projection")
            return None, None
            
        self.logger.info(f"Running morphospace projection on {len(samples)} samples")
        
        # Extract feature vectors from samples
        feature_vectors, feature_labels, sample_metadata = self._extract_feature_vectors(samples)
        
        if len(feature_vectors) == 0:
            self.logger.error("Failed to extract feature vectors")
            return None, None
            
        self.logger.info(f"Extracted {len(feature_vectors)} feature vectors with {feature_vectors[0].shape[0]} dimensions")
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reduce dimensionality with PCA first
        pca_projections = self._run_pca(X_scaled)
        
        # Run t-SNE if requested
        tsne_projections = None
        if projection_method in ['tsne', 'all']:
            tsne_projections = self._run_tsne(X_scaled)
        
        # Run UMAP if requested
        umap_projections = None
        if projection_method in ['umap', 'all']:
            umap_projections = self._run_umap(X_scaled)
        
        # Perform clustering
        cluster_labels, silhouette_avg = self._perform_clustering(X_scaled)
        
        # Save feature vectors
        feature_df = pd.DataFrame(X, columns=feature_labels)
        feature_df['cluster'] = cluster_labels
        
        # Add metadata to feature DataFrame
        for key in sample_metadata[0].keys():
            feature_df[key] = [meta[key] for meta in sample_metadata]
            
        # Save feature vectors and clusters
        feature_csv = os.path.join(self.output_dir, f"feature_vectors_{self.run_id}.csv")
        feature_df.to_csv(feature_csv, index=False)
        
        # Create projection DataFrame
        projections_df = pd.DataFrame({
            'alpha': [meta['alpha'] for meta in sample_metadata],
            'gamma': [meta['gamma'] for meta in sample_metadata],
            'ccdi': [meta['ccdi'] for meta in sample_metadata],
            'correlation': [meta['correlation'] for meta in sample_metadata],
            'recovery_class': [meta['recovery_class'] for meta in sample_metadata],
            'cluster': cluster_labels
        })
        
        # Add PCA projections
        for i in range(pca_projections.shape[1]):
            projections_df[f'pca_{i+1}'] = pca_projections[:, i]
            
        # Add t-SNE projections if available
        if tsne_projections is not None:
            projections_df['tsne_1'] = tsne_projections[:, 0]
            projections_df['tsne_2'] = tsne_projections[:, 1]
            
        # Add UMAP projections if available
        if umap_projections is not None:
            projections_df['umap_1'] = umap_projections[:, 0]
            projections_df['umap_2'] = umap_projections[:, 1]
        
        # Save projections
        projections_csv = os.path.join(self.output_dir, f"projections_{self.run_id}.csv")
        projections_df.to_csv(projections_csv, index=False)
        
        # Generate visualizations
        self._generate_projections_visualizations(projections_df, silhouette_avg)
        
        self.logger.info("Morphospace projection completed successfully")
        return projections_df, feature_df
    
    def load_all_samples(self):
        """
        Load all samples from source directories
        
        Returns:
        --------
        list
            List of FalseAttractorSample objects
        """
        all_samples = []
        
        # Load from each source directory
        for source_dir in self.source_dirs:
            self.logger.info(f"Loading samples from {source_dir}")
            dir_samples = self.load_samples(source_dir)
            all_samples.extend(dir_samples)
            self.logger.info(f"Loaded {len(dir_samples)} samples from {source_dir}")
        
        # Add any samples that might have been set directly
        all_samples.extend(self.samples)
        
        # Remove duplicates (based on alpha and gamma)
        unique_samples = {}
        for sample in all_samples:
            key = f"{sample.alpha:.6f}_{sample.gamma:.6f}"
            unique_samples[key] = sample
            
        self.logger.info(f"Total unique samples: {len(unique_samples)}")
        return list(unique_samples.values())
    
    def _extract_feature_vectors(self, samples):
        """
        Extract feature vectors from samples
        
        Parameters:
        -----------
        samples : list
            List of FalseAttractorSample objects
            
        Returns:
        --------
        tuple
            (feature_vectors, feature_labels, sample_metadata) - Lists of feature arrays, labels, and metadata
        """
        feature_vectors = []
        feature_labels = []
        sample_metadata = []
        
        # Define feature components
        feature_components = [
            ('trajectory', lambda s: self._extract_trajectory_fingerprint(s, self.fingerprint_length)),
            ('entropy', lambda s: self._extract_entropy_fingerprint(s, self.fingerprint_length)),
            ('residual', lambda s: self._extract_residual_pca(s, n_components=5))
        ]
        
        # Loop through samples
        for sample in tqdm(samples, desc="Extracting Features"):
            try:
                # Extract individual feature components
                sample_features = []
                
                # If this is the first sample, create feature labels
                if len(feature_labels) == 0:
                    for component_name, extract_func in feature_components:
                        component_features = extract_func(sample)
                        sample_features.extend(component_features)
                        
                        # Create labels for this component
                        component_labels = [f"{component_name}_{i+1}" for i in range(len(component_features))]
                        feature_labels.extend(component_labels)
                else:
                    # Otherwise just extract features
                    for _, extract_func in feature_components:
                        component_features = extract_func(sample)
                        sample_features.extend(component_features)
                
                # Add to feature vectors
                feature_vectors.append(np.array(sample_features))
                
                # Add metadata
                meta = {
                    'alpha': sample.alpha,
                    'gamma': sample.gamma,
                    'ccdi': sample.ccdi,
                    'correlation': sample.metrics.get('final_correlation', 0.0),
                    'recovery_class': sample.recovery_class
                }
                sample_metadata.append(meta)
                
            except Exception as e:
                self.logger.error(f"Error extracting features from sample: {e}")
        
        return feature_vectors, feature_labels, sample_metadata
    
    def _extract_trajectory_fingerprint(self, sample, n_points=10):
        """Extract trajectory fingerprint from a sample"""
        try:
            # Get trajectory data
            trajectory = sample.recovery_trajectory
            
            if 'correlation' not in trajectory or 'step' not in trajectory:
                # Return zeros if data is missing
                return np.zeros(n_points)
                
            # Get correlation and step data
            correlation = np.array(trajectory['correlation'])
            steps = np.array(trajectory['step'])
            
            # Find second perturbation step if available
            if 'perturbation_info' in sample.__dict__ and 'steps' in sample.perturbation_info:
                # Use second perturbation step if available
                if len(sample.perturbation_info['steps']) > 1:
                    pert_step = sample.perturbation_info['steps'][1]
                else:
                    pert_step = sample.perturbation_info['steps'][0]
            else:
                # Assume perturbation at middle of trajectory
                pert_step = len(steps) // 2
            
            # Extract post-perturbation trajectory
            post_indices = np.where(steps >= pert_step)[0]
            
            if len(post_indices) == 0:
                # Use all data if no post-perturbation data
                post_correlation = correlation
                post_steps = steps
            else:
                post_correlation = correlation[post_indices]
                post_steps = steps[post_indices]
            
            # Sample at evenly spaced points
            if len(post_steps) < n_points:
                # Pad with last value if not enough points
                pad_length = n_points - len(post_steps)
                fingerprint = np.pad(post_correlation, (0, pad_length), 'edge')
            else:
                # Select evenly spaced indices
                indices = np.linspace(0, len(post_steps) - 1, n_points, dtype=int)
                fingerprint = post_correlation[indices]
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error extracting trajectory fingerprint: {e}")
            return np.zeros(n_points)
    
    def _extract_entropy_fingerprint(self, sample, n_points=10):
        """Extract entropy fingerprint from a sample"""
        try:
            # Get trajectory data
            trajectory = sample.recovery_trajectory
            
            if 'spectral_entropy' not in trajectory or 'step' not in trajectory:
                # Return zeros if data is missing
                return np.zeros(n_points)
                
            # Get entropy and step data
            entropy = np.array(trajectory['spectral_entropy'])
            steps = np.array(trajectory['step'])
            
            # Find second perturbation step if available
            if 'perturbation_info' in sample.__dict__ and 'steps' in sample.perturbation_info:
                # Use second perturbation step if available
                if len(sample.perturbation_info['steps']) > 1:
                    pert_step = sample.perturbation_info['steps'][1]
                else:
                    pert_step = sample.perturbation_info['steps'][0]
            else:
                # Assume perturbation at middle of trajectory
                pert_step = len(steps) // 2
            
            # Extract post-perturbation trajectory
            post_indices = np.where(steps >= pert_step)[0]
            
            if len(post_indices) == 0:
                # Use all data if no post-perturbation data
                post_entropy = entropy
                post_steps = steps
            else:
                post_entropy = entropy[post_indices]
                post_steps = steps[post_indices]
            
            # Sample at evenly spaced points
            if len(post_steps) < n_points:
                # Pad with last value if not enough points
                pad_length = n_points - len(post_steps)
                fingerprint = np.pad(post_entropy, (0, pad_length), 'edge')
            else:
                # Select evenly spaced indices
                indices = np.linspace(0, len(post_steps) - 1, n_points, dtype=int)
                fingerprint = post_entropy[indices]
            
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error extracting entropy fingerprint: {e}")
            return np.zeros(n_points)
    
    def _extract_residual_pca(self, sample, n_components=5):
        """Extract PCA components of the residual field"""
        try:
            # Check if residual is available
            if not hasattr(sample, 'residual') or sample.residual is None:
                # Try to compute residual
                if hasattr(sample, 'final_state') and hasattr(sample, 'initial_state'):
                    residual = sample.final_state - sample.initial_state
                else:
                    # Return zeros if data is missing
                    return np.zeros(n_components)
            else:
                residual = sample.residual
            
            # Flatten residual
            flat_residual = residual.flatten()
            
            # If this is a single residual, just return a normalized version
            if n_components >= len(flat_residual):
                return flat_residual / np.linalg.norm(flat_residual)
            
            # Extract PCA components
            pca = PCA(n_components=n_components)
            
            # Reshape for PCA (even though it's just one sample)
            residual_reshaped = flat_residual.reshape(1, -1)
            
            # Get features
            pca.fit(residual_reshaped)
            features = pca.transform(residual_reshaped)[0]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting residual PCA: {e}")
            return np.zeros(n_components)
    
    def _run_pca(self, X):
        """
        Run PCA projection
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        ndarray
            PCA projection
        """
        # Determine number of components (at most the minimum of n_samples and n_features)
        n_components = min(self.n_components, X.shape[0], X.shape[1])
        
        # Run PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca_result = pca.fit_transform(X)
        
        # Log variance explained
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.logger.info(f"PCA explained variance by component: {explained_variance}")
        self.logger.info(f"Cumulative explained variance: {cumulative_variance}")
        
        # Save variance explained plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, align='center')
        plt.step(range(1, n_components + 1), cumulative_variance, where='mid', label='Cumulative')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        
        plt.savefig(os.path.join(self.output_dir, "visualizations", f"pca_variance_{self.run_id}.png"), dpi=300)
        plt.close()
        
        return pca_result
    
    def _run_tsne(self, X):
        """
        Run t-SNE projection
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        ndarray
            t-SNE projection
        """
        # Determine perplexity (should be smaller than the number of samples)
        perplexity = min(self.tsne_perplexity, max(5, X.shape[0] // 5))
        
        # Run t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            init='pca',
            random_state=self.random_state
        )
        
        self.logger.info(f"Running t-SNE with perplexity {perplexity}...")
        tsne_result = tsne.fit_transform(X)
        
        return tsne_result
    
    def _run_umap(self, X):
        """
        Run UMAP projection
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        ndarray
            UMAP projection
        """
        # Determine n_neighbors (should be smaller than the number of samples)
        n_neighbors = min(self.umap_n_neighbors, max(5, X.shape[0] // 3))
        
        # Run UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.random_state
        )
        
        self.logger.info(f"Running UMAP with n_neighbors {n_neighbors}...")
        umap_result = reducer.fit_transform(X)
        
        return umap_result
    
    def _perform_clustering(self, X):
        """
        Perform clustering on the feature matrix
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        tuple
            (cluster_labels, silhouette_avg) - Cluster assignments and silhouette score
        """
        # Determine number of clusters (at most half the number of samples)
        n_clusters = min(self.n_clusters, X.shape[0] // 2)
        
        if n_clusters < 2:
            # Not enough samples for meaningful clustering
            self.logger.warning("Not enough samples for clustering")
            return np.zeros(X.shape[0], dtype=int), 0.0
        
        # Run K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        self.logger.info(f"K-means clustering with {n_clusters} clusters, silhouette score: {silhouette_avg:.4f}")
        
        return cluster_labels, silhouette_avg
    
    def _generate_projections_visualizations(self, projections_df, silhouette_score):
        """
        Generate visualizations for the projections
        
        Parameters:
        -----------
        projections_df : pd.DataFrame
            DataFrame with projection coordinates
        silhouette_score : float
            Silhouette score for clustering
        """
        # Create directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate PCA plot
        self._plot_pca_projection(projections_df, vis_dir)
        
        # Generate t-SNE plot if available
        if 'tsne_1' in projections_df.columns and 'tsne_2' in projections_df.columns:
            self._plot_tsne_projection(projections_df, vis_dir)
            
        # Generate UMAP plot if available
        if 'umap_1' in projections_df.columns and 'umap_2' in projections_df.columns:
            self._plot_umap_projection(projections_df, vis_dir)
            
        # Generate cluster analysis visualization
        self._plot_cluster_analysis(projections_df, silhouette_score, vis_dir)
    
    def _plot_pca_projection(self, df, vis_dir):
        """Plot PCA projection"""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot by cluster
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(
            df['pca_1'],
            df['pca_2'],
            c=df['cluster'],
            cmap='tab10',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('PCA Projection by Cluster')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by CCDI
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(
            df['pca_1'],
            df['pca_2'],
            c=df['ccdi'],
            cmap='viridis',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='CCDI')
        plt.title('PCA Projection by CCDI')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by correlation
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(
            df['pca_1'],
            df['pca_2'],
            c=df['correlation'],
            cmap='RdYlGn',
            alpha=0.8,
            s=100,
            edgecolors='w',
            vmin=0, vmax=1
        )
        plt.colorbar(scatter, label='Correlation')
        plt.title('PCA Projection by Correlation')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by recovery class
        plt.subplot(2, 2, 4)
        # Get unique recovery classes
        classes = df['recovery_class'].unique()
        # Create a colormap
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            mask = df['recovery_class'] == cls
            plt.scatter(
                df.loc[mask, 'pca_1'],
                df.loc[mask, 'pca_2'],
                color=colors[i],
                alpha=0.8,
                s=100,
                edgecolors='w',
                label=cls
            )
        plt.title('PCA Projection by Recovery Class')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"pca_projection_{self.run_id}.png"), dpi=300)
        plt.close()
    
    def _plot_tsne_projection(self, df, vis_dir):
        """Plot t-SNE projection"""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot by cluster
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(
            df['tsne_1'],
            df['tsne_2'],
            c=df['cluster'],
            cmap='tab10',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('t-SNE Projection by Cluster')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by CCDI
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(
            df['tsne_1'],
            df['tsne_2'],
            c=df['ccdi'],
            cmap='viridis',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='CCDI')
        plt.title('t-SNE Projection by CCDI')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by correlation
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(
            df['tsne_1'],
            df['tsne_2'],
            c=df['correlation'],
            cmap='RdYlGn',
            alpha=0.8,
            s=100,
            edgecolors='w',
            vmin=0, vmax=1
        )
        plt.colorbar(scatter, label='Correlation')
        plt.title('t-SNE Projection by Correlation')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by recovery class
        plt.subplot(2, 2, 4)
        # Get unique recovery classes
        classes = df['recovery_class'].unique()
        # Create a colormap
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            mask = df['recovery_class'] == cls
            plt.scatter(
                df.loc[mask, 'tsne_1'],
                df.loc[mask, 'tsne_2'],
                color=colors[i],
                alpha=0.8,
                s=100,
                edgecolors='w',
                label=cls
            )
        plt.title('t-SNE Projection by Recovery Class')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"tsne_projection_{self.run_id}.png"), dpi=300)
        plt.close()
    
    def _plot_umap_projection(self, df, vis_dir):
        """Plot UMAP projection"""
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot by cluster
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(
            df['umap_1'],
            df['umap_2'],
            c=df['cluster'],
            cmap='tab10',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('UMAP Projection by Cluster')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by CCDI
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(
            df['umap_1'],
            df['umap_2'],
            c=df['ccdi'],
            cmap='viridis',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        plt.colorbar(scatter, label='CCDI')
        plt.title('UMAP Projection by CCDI')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by correlation
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(
            df['umap_1'],
            df['umap_2'],
            c=df['correlation'],
            cmap='RdYlGn',
            alpha=0.8,
            s=100,
            edgecolors='w',
            vmin=0, vmax=1
        )
        plt.colorbar(scatter, label='Correlation')
        plt.title('UMAP Projection by Correlation')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot by recovery class
        plt.subplot(2, 2, 4)
        # Get unique recovery classes
        classes = df['recovery_class'].unique()
        # Create a colormap
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            mask = df['recovery_class'] == cls
            plt.scatter(
                df.loc[mask, 'umap_1'],
                df.loc[mask, 'umap_2'],
                color=colors[i],
                alpha=0.8,
                s=100,
                edgecolors='w',
                label=cls
            )
        plt.title('UMAP Projection by Recovery Class')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"umap_projection_{self.run_id}.png"), dpi=300)
        plt.close()
    
    def _plot_cluster_analysis(self, df, silhouette_score, vis_dir):
        """Plot cluster analysis visualization"""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get counts by cluster
        cluster_counts = df['cluster'].value_counts().sort_index()
        
        # Create cluster distribution
        plt.subplot(1, 2, 1)
        bars = plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.tab10(cluster_counts.index))
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.title(f'Cluster Distribution (Silhouette Score: {silhouette_score:.4f})')
        plt.xticks(cluster_counts.index)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(int(bar.get_height())),
                ha='center'
            )
        
        # Create cluster characteristics
        plt.subplot(1, 2, 2)
        
        # Get mean values by cluster
        cluster_stats = df.groupby('cluster').agg({
            'correlation': 'mean',
            'ccdi': 'mean'
        }).reset_index()
        
        # Sort by correlation (higher is better)
        cluster_stats = cluster_stats.sort_values('correlation', ascending=False)
        
        # Create a twin axes
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot correlation
        correlation_bars = ax1.bar(
            cluster_stats['cluster'],
            cluster_stats['correlation'],
            color='green',
            alpha=0.7,
            label='Correlation'
        )
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Correlation', color='green')
        ax1.tick_params(axis='y', colors='green')
        ax1.set_ylim(0, 1)
        
        # Plot CCDI
        ccdi_bars = ax2.bar(
            cluster_stats['cluster'],
            cluster_stats['ccdi'],
            color='red',
            alpha=0.7,
            label='CCDI'
        )
        ax2.set_ylabel('CCDI', color='red')
        ax2.tick_params(axis='y', colors='red')
        
        # Add title
        plt.title('Cluster Characteristics')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"cluster_analysis_{self.run_id}.png"), dpi=300)
        plt.close()
        
        # Create cluster parameter space visualization
        plt.figure(figsize=(10, 8))
        
        # Plot clusters in alpha-gamma space
        scatter = plt.scatter(
            df['alpha'],
            df['gamma'],
            c=df['cluster'],
            cmap='tab10',
            alpha=0.8,
            s=100,
            edgecolors='w'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, label='Cluster')
        cbar.set_ticks(np.arange(len(cluster_counts)))
        
        # Add labels for each point
        for i, row in df.iterrows():
            plt.text(
                row['alpha'],
                row['gamma'],
                str(int(row['cluster'])),
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                fontsize=8
            )
        
        plt.title('Clusters in Parameter Space')
        plt.xlabel('Memory Strength (Alpha)')
        plt.ylabel('Memory Decay (Gamma)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"cluster_parameter_space_{self.run_id}.png"), dpi=300)
        plt.close()
    
    def summarize_morphospace(self):
        """Print a summary of the morphospace projection"""
        if not hasattr(self, 'results_df') or self.results_df is None or len(self.results_df) == 0:
            self.logger.warning("No results available for summary")
            return
            
        # Get the projections DataFrame
        df = self.results_df
        
        print("\nMorphospace Projection Summary:")
        print(f"Total Samples: {len(df)}")
        
        # Print cluster information
        cluster_counts = df['cluster'].value_counts().sort_index()
        print("\nCluster Distribution:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} samples ({count/len(df)*100:.1f}%)")
            
        # Print cluster characteristics
        cluster_stats = df.groupby('cluster').agg({
            'correlation': ['mean', 'min', 'max'],
            'ccdi': ['mean', 'min', 'max']
        })
        
        print("\nCluster Characteristics:")
        for cluster in cluster_stats.index:
            print(f"  Cluster {cluster}:")
            print(f"    Correlation: {cluster_stats.loc[cluster, ('correlation', 'mean')]:.4f} " +
                 f"[{cluster_stats.loc[cluster, ('correlation', 'min')]:.4f}, " +
                 f"{cluster_stats.loc[cluster, ('correlation', 'max')]:.4f}]")
            print(f"    CCDI: {cluster_stats.loc[cluster, ('ccdi', 'mean')]:.4f} " +
                 f"[{cluster_stats.loc[cluster, ('ccdi', 'min')]:.4f}, " +
                 f"{cluster_stats.loc[cluster, ('ccdi', 'max')]:.4f}]")
            
        # Print recovery class distribution
        class_counts = df['recovery_class'].value_counts()
        print("\nRecovery Class Distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} samples ({count/len(df)*100:.1f}%)")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RCFT Morphospace Projection')
    parser.add_argument('--output_dir', type=str, default='phase3_results/morphospace_projection',
                        help='Output directory for results')
    parser.add_argument('--source_dirs', type=str, default='',
                        help='Comma-separated list of source directories')
    parser.add_argument('--fingerprint_length', type=int, default=10,
                        help='Number of points in trajectory fingerprints')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of PCA components')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters for K-means')
    parser.add_argument('--projection', type=str, default='all',
                        choices=['pca', 'tsne', 'umap', 'all'],
                        help='Projection method to use')
    
    args = parser.parse_args()
    
    # Initialize projector
    projector = MorphospaceProjector(output_dir=args.output_dir)
    
    # Set parameters
    projector.fingerprint_length = args.fingerprint_length
    projector.n_components = args.n_components
    projector.n_clusters = args.n_clusters
    
    # Add source directories
    if args.source_dirs:
        for source_dir in args.source_dirs.split(','):
            projector.add_source_directory(source_dir)
    
    # Run projection
    projections_df, feature_df = projector.run_morphospace_projection(
        projection_method=args.projection
    )
    
    # Print summary
    projector.summarize_morphospace()
    
    print(f"\nMorphospace projection complete. Results saved to {args.output_dir}")
