"""
Fingerprinting utilities for Phase V of the RCFT framework.
This module provides tools for extracting compact representations of attractors.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class AttractorFingerprinter:
    """Tool for creating and comparing attractor fingerprints."""
    
    def __init__(self, n_components=2, method='pca', fft_bins=32):
        """
        Initialize the fingerprinter.
        
        Args:
            n_components: Number of components for dimensionality reduction
            method: 'pca', 'fft', or 'both'
            fft_bins: Number of frequency bins for FFT
        """
        self.n_components = n_components
        self.method = method
        self.fft_bins = fft_bins
        self.pca = None
        self.tsne = None
        self.reference_fingerprints = {}
        
    def compute_fingerprint(self, state, reference_id=None, store=True):
        """
        Compute fingerprint for a state.
        
        Args:
            state: 2D array representing a field state
            reference_id: Optional identifier to store this fingerprint as reference
            store: Whether to store this fingerprint as a reference
            
        Returns:
            Dictionary with fingerprint data
        """
        fingerprint = {}
        
        if self.method in ['pca', 'both']:
            # Compute PCA fingerprint
            # Compute PCA fingerprint
            flattened = state.flatten().reshape(1, -1)

            if self.pca is None:
                # Limit components to number of samples
                n_components = min(self.n_components, flattened.shape[0])
                self.pca = PCA(n_components=n_components)
                self.pca.fit(flattened)
            
            pca_coords = self.pca.transform(flattened)[0]
            fingerprint['pca'] = pca_coords
            fingerprint['pca_explained_variance'] = self.pca.explained_variance_ratio_
            
        if self.method in ['fft', 'both']:
            # Compute FFT fingerprint
            fft_result = fft.fft2(state)
            fft_mag = np.abs(fft.fftshift(fft_result))
            
            # Compute radial power spectrum
            cy, cx = [x // 2 for x in fft_mag.shape]
            y, x = np.indices(fft_mag.shape)
            r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
            
            # Get the radial power spectrum
            radial_bins = np.bincount(r.ravel(), fft_mag.ravel())
            radial_bins = radial_bins / np.sum(radial_bins)  # Normalize
            
            # Resample to fixed number of bins
            bin_indices = np.linspace(0, len(radial_bins) - 1, self.fft_bins)
            bin_indices = np.round(bin_indices).astype(int)
            resampled_spectrum = radial_bins[bin_indices]
            
            fingerprint['fft_spectrum'] = resampled_spectrum
            
            # Compute spectral entropy
            spectrum_normalized = resampled_spectrum / np.sum(resampled_spectrum)
            nonzero_indices = spectrum_normalized > 0
            entropy = -np.sum(spectrum_normalized[nonzero_indices] * 
                             np.log2(spectrum_normalized[nonzero_indices]))
            fingerprint['spectral_entropy'] = entropy
        
        # Always compute some simple statistics
        fingerprint['mean'] = np.mean(state)
        fingerprint['std'] = np.std(state)
        fingerprint['min'] = np.min(state)
        fingerprint['max'] = np.max(state)
        fingerprint['energy'] = np.sum(state**2)
        
        # Store as reference if requested
        if reference_id is not None and store:
            self.reference_fingerprints[reference_id] = fingerprint
        
        return fingerprint
    
    def compute_fingerprint_distance(self, fingerprint1, fingerprint2):
        """
        Compute distance between two fingerprints.
        
        Args:
            fingerprint1: First fingerprint dictionary
            fingerprint2: Second fingerprint dictionary
            
        Returns:
            Dictionary with distance metrics
        """
        distances = {}
        
        if 'pca' in fingerprint1 and 'pca' in fingerprint2:
            pca_dist = np.linalg.norm(fingerprint1['pca'] - fingerprint2['pca'])
            distances['pca_distance'] = pca_dist
        
        if 'fft_spectrum' in fingerprint1 and 'fft_spectrum' in fingerprint2:
            spec1 = fingerprint1['fft_spectrum']
            spec2 = fingerprint2['fft_spectrum']
            
            # Euclidean distance between spectra
            fft_dist = np.linalg.norm(spec1 - spec2)
            distances['fft_distance'] = fft_dist
            
            # Compute KL divergence (if applicable)
            if np.all(spec1 > 0) and np.all(spec2 > 0):
                spec1_norm = spec1 / np.sum(spec1)
                spec2_norm = spec2 / np.sum(spec2)
                
                kl_div = np.sum(spec1_norm * np.log2(spec1_norm / spec2_norm))
                distances['kl_divergence'] = kl_div
        
        # Statistics distance 
        if all(k in fingerprint1 and k in fingerprint2 for k in ['mean', 'std', 'energy']):
            # Simple Euclidean distance in the space of mean, std, energy
            stat_vec1 = np.array([fingerprint1['mean'], fingerprint1['std'], 
                                 fingerprint1['energy']])
            stat_vec2 = np.array([fingerprint2['mean'], fingerprint2['std'], 
                                 fingerprint2['energy']])
            
            # Normalize to make metrics comparable
            stat_vec1 = stat_vec1 / np.linalg.norm(stat_vec1)
            stat_vec2 = stat_vec2 / np.linalg.norm(stat_vec2)
            
            distances['stats_distance'] = np.linalg.norm(stat_vec1 - stat_vec2)
        
        # Compute an aggregate distance if possible
        if distances:
            distances['aggregate_distance'] = np.mean(list(distances.values()))
        
        return distances
    
    def compare_to_reference(self, state, reference_id):
        """
        Compare a state to a stored reference fingerprint.
        
        Args:
            state: State to compare
            reference_id: ID of the reference fingerprint
            
        Returns:
            Dictionary with distance metrics
        """
        if reference_id not in self.reference_fingerprints:
            raise ValueError(f"Reference fingerprint '{reference_id}' not found")
        
        current_fingerprint = self.compute_fingerprint(state, store=False)
        reference_fingerprint = self.reference_fingerprints[reference_id]
        
        return self.compute_fingerprint_distance(current_fingerprint, reference_fingerprint)
    
    def visualize_fingerprint(self, fingerprint, title=None):
        """
        Visualize a fingerprint.
        
        Args:
            fingerprint: Fingerprint dictionary
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 5))
        
        if 'fft_spectrum' in fingerprint:
            plt.subplot(1, 2, 1)
            plt.plot(fingerprint['fft_spectrum'], 'o-')
            plt.title('FFT Power Spectrum')
            plt.xlabel('Frequency Bin')
            plt.ylabel('Power')
            plt.grid(alpha=0.3)
        
        if 'pca' in fingerprint:
            plt.subplot(1, 2, 2)
            
            # If only 2 components, plot directly
            if len(fingerprint['pca']) == 2:
                plt.scatter([fingerprint['pca'][0]], [fingerprint['pca'][1]], 
                          s=100, color='blue')
                plt.title('PCA Space')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.grid(alpha=0.3)
            else:
                # If more components, show first 3 as bar chart
                n_components = min(3, len(fingerprint['pca']))
                plt.bar(range(n_components), fingerprint['pca'][:n_components])
                plt.title('Top PCA Components')
                plt.xlabel('Component')
                plt.ylabel('Value')
                plt.grid(alpha=0.3)
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        return fig
    
    def visualize_fingerprint_comparison(self, fingerprints, labels=None, title=None):
        """
        Visualize comparison between multiple fingerprints.
        
        Args:
            fingerprints: List of fingerprint dictionaries
            labels: Optional list of labels for the fingerprints
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure
        """
        if not fingerprints:
            raise ValueError("No fingerprints provided for comparison")
            
        if labels is None:
            labels = [f"Fingerprint {i+1}" for i in range(len(fingerprints))]
            
        fig = plt.figure(figsize=(14, 8))
        
        # Check which fingerprint type is available
        has_fft = all('fft_spectrum' in fp for fp in fingerprints)
        has_pca = all('pca' in fp for fp in fingerprints)
        
        # Plot FFT spectra if available
        if has_fft:
            plt.subplot(2, 2, 1)
            for i, fp in enumerate(fingerprints):
                plt.plot(fp['fft_spectrum'], 'o-', label=labels[i])
            plt.title('FFT Power Spectra')
            plt.xlabel('Frequency Bin')
            plt.ylabel('Power')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Also show spectral entropy as a bar chart
            plt.subplot(2, 2, 2)
            entropies = [fp.get('spectral_entropy', 0) for fp in fingerprints]
            plt.bar(labels, entropies)
            plt.title('Spectral Entropy')
            plt.ylabel('Entropy')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        
        # Plot PCA components if available
        if has_pca:
            plt.subplot(2, 2, 3)
            
            # Extract PCA coordinates and plot
            pca_coords = np.array([fp['pca'] for fp in fingerprints])
            
            # If 2D, plot directly
            if pca_coords.shape[1] == 2:
                for i, (x, y) in enumerate(pca_coords):
                    plt.scatter(x, y, s=100, label=labels[i])
                plt.title('PCA Space')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.legend()
                plt.grid(alpha=0.3)
            else:
                # If more dimensions, use t-SNE to visualize
                if len(fingerprints) >= 3:  # Need at least 3 points for t-SNE
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(pca_coords)
                    
                    for i, (x, y) in enumerate(tsne_result):
                        plt.scatter(x, y, s=100, label=labels[i])
                    plt.title('t-SNE Visualization of PCA Components')
                    plt.xlabel('t-SNE 1')
                    plt.ylabel('t-SNE 2')
                    plt.legend()
                    plt.grid(alpha=0.3)
                else:
                    # Just show first component as bars
                    first_components = [fp['pca'][0] for fp in fingerprints]
                    plt.bar(labels, first_components)
                    plt.title('First PCA Component')
                    plt.ylabel('Value')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', alpha=0.3)
        
        # Plot statistics comparison
        plt.subplot(2, 2, 4)
        means = [fp.get('mean', 0) for fp in fingerprints]
        stds = [fp.get('std', 0) for fp in fingerprints]
        energies = [fp.get('energy', 0) / 1e4 for fp in fingerprints]  # Scale down for visualization
        
        x = np.arange(len(labels))
        width = 0.25
        
        plt.bar(x - width, means, width, label='Mean')
        plt.bar(x, stds, width, label='Std Dev')
        plt.bar(x + width, energies, width, label='Energy (×10⁴)')
        
        plt.title('Statistical Properties')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        return fig