    # Module 2 options
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Source directory for false attractor samples')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of edge cases to analyze')
    parser.add_argument('--nudge_times', type=str, default='30,40,50',
                        help='Comma-separated list of nudge time steps')
    parser.add_argument('--nudge_amplitudes', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated list of nudge amplitudes')
    parser.add_argument('--nudge_types', type=str, default='uniform,patch,echo',
                        help='Comma-separated list of nudge types')
    
    # Module 3 options
    parser.add_argument('--max_iterations', type=int, default=500,
                        help='Maximum iterations for inversion optimization')
    parser.add_argument('--regularization', type=float, default=0.1,
                        help='Entropy regularization parameter')
    
    # Module 4 options
    parser.add_argument('--source_dirs', type=str, default='',
                        help='Comma-separated list of source directories')
    parser.add_argument('--fingerprint_length', type=int, default=10,
                        help='Number of points in trajectory fingerprints')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of PCA components')
    parser.add_argument('--projection', type=str, default='all',
                        choices=['pca', 'tsne', 'umap', 'all'],
                        help='Projection method to use')