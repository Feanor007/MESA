import warnings
import numpy as np
import leidenalg
import igraph as ig
import anndata as ad
from sklearn.neighbors import NearestNeighbors

from mesa.multiomics.multiomics_spatial import utils


def get_spatial_knn_indices(locations, n_neighbors=15, method='kd_tree')-> np.ndarray:
    """
    Compute the k-nearest neighbors (k-NN) of each location in a given data matrix.

    Parameters
    ----------
    locations : :class:`numpy.ndarray` of shape (n_samples, 2)
        The data matrix representing locations where each row corresponds to a point in 2D space.
    n_neighbors : int, optional
        The number of nearest neighbors to identify for each location. Default is 15.
    method : str, optional
        The method used to compute the nearest neighbors. Options include 'ball_tree', 'kd_tree', and 'brute'. Default is 'kd_tree'.

    Returns
    -------
    :class:`numpy.ndarray` of shape (n_samples, n_neighbors)
        An array where each row represents the indices of the nearest neighbors for that sample, ordered by proximity.
    """
    
    locations = np.array(locations)
    assert n_neighbors <= locations.shape[0]
    # k-NN indices, may be asymmetric
    _, knn_indices = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm=method
    ).fit(locations).kneighbors(locations)
    return knn_indices

def get_spatial_knn_indices_within_distance(locations, n_neighbors=15, method='kd_tree', max_distance=np.inf):
    """
    Compute k-nearest neighbors of locations with a maximum distance constraint.

    Parameters
    ----------
    locations: np.ndarray of shape (n_samples, 2)
        Data matrix
    n_neighbors: int
        Number of nearest neighbors
    method: str, default='kd_tree'
        Method to use when computing the nearest neighbors, one of ['ball_tree', 'kd_tree', 'brute']
    max_distance: float, default=np.inf
        The maximum distance between neighbors

    Returns
    -------
    knn_indices: np.ndarray of shape (n_samples, n_neighbors)
        Each row represents the knn of that sample. Distance greater than max_distance will not be considered.
    """
    locations = np.array(locations)
    assert n_neighbors <= locations.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=method)
    nbrs.fit(locations)
    distances, indices = nbrs.kneighbors(locations)

    # Initialize the knn_indices array with -1 to indicate insufficient neighbors within max_distance
    knn_indices = -np.ones((locations.shape[0], n_neighbors), dtype=int)

    # Filter indices based on max_distance
    for i in range(locations.shape[0]):
        valid_neighbors = indices[i][distances[i] <= max_distance][:n_neighbors]
        knn_indices[i, :len(valid_neighbors)] = valid_neighbors

    return knn_indices

def leiden_clustering(n, edges, resolution=1, n_runs=1, seed=None, verbose=False):
    """
    Apply Leiden modularity maximization algorithm on the graph defined by edges and weights
    Parameters
    ----------
    n: int
        Number of edges in total
    edges: list of length two or three
        If length is two, then the graph is unweighted and each edge is (edges[0][i], edges[1][i]),
        if the length is three, then the graph is weighted and the weight of (edges[0][i], edges[1][i]) is edges[2][i].
    resolution: float, default=1
        Resolution parameter in Leiden algorithm
    n_runs: int, default=1
        Number of runs of Leiden algorithm, the run with the best modularity is taken as the output
    seed: None or int, default=None
        Random seed used. If None, use a random integer. If n_runs > 1, then seed will be reset to None.
    verbose: bool, default=True
        Whether to print progress
    Returns
    -------
    labels: np.array of shape (n,)
        Cluster labels
    """
    g = ig.Graph(directed=True)
    g.add_vertices(n)
    g.add_edges(list(zip(edges[0], edges[1])))
    if len(edges) > 2:
        g.es['weight'] = tuple(edges[2])

    if n_runs > 1 and seed is not None:
        seed = None
        warnings.warn('n_runs > 1, seed is reset to None.')

    partition_kwargs = {'n_iterations': -1, 'seed': seed,
                        'resolution_parameter': resolution}
    if len(edges) > 2:
        partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)

    partition_type = leidenalg.RBConfigurationVertexPartition

    best_modularity = float('-inf')
    best_labels = None
    if verbose and n_runs > 1:
        print('Running Leiden algorithm for {} times...'.format(n_runs), flush=True)
    for _ in range(n_runs):
        curr_part = leidenalg.find_partition(
            graph=g, partition_type=partition_type,
            **partition_kwargs
        )
        curr_modularity = curr_part.modularity
        if curr_modularity > best_modularity:
            best_modularity, best_labels = curr_modularity, np.array(curr_part.membership)

    assert best_labels is not None

    if verbose:
        if n_runs > 1:
            print('Best modularity among {} runs is {}.'.format(n_runs, best_modularity), flush=True)
        else:
            print('Modularity is {}.'.format(best_modularity), flush=True)
        print('The label has {} distinct clusters.'.format(len(np.unique(best_labels))), flush=True)

    return best_labels, best_modularity


def graph_clustering(
        n, edges, resolution=1, n_clusters=None, n_runs=1, resolution_tol=0.05, seed=None, verbose=False
):
    """
    Cluster the graph defined by edges and weights using Leiden algorithm.
    Parameters
    ----------
    n: int
        Number of edges in total
    edges: list of length two or three
        If length is two, then the graph is unweighted and each edge is (edges[0][i], edges[1][i]),
        if the length is three, then the graph is weighted and the weight of (edges[0][i], edges[1][i]) is edges[2][i].
    resolution: None or int, default=1
        If not None, then this is the resolution parameter in the clustering algorithm,
        if None, then n_clusters must be not None.
    n_clusters: None or int, default=None
        If not None, use binary search to select the resolution parameter to achieve the desired number of clusters,
        if None, then resolution must be not None.
    n_runs: int, default=1
        Number of runs of Leiden algorithm, the run with the best modularity is taken as the output.
    resolution_tol: float, default=0.05
        Any resolution within the range of plus/minus resolution_tol will not be differentiated.
    seed: None or int, default=None
        Random seed used. If None, use a random integer. If n_runs > 1, then seed will be reset to None.
    verbose: bool, default=True
        Whether to print progress
    Returns
    -------
    labels: np.array of shape (n,)
        Cluster labels
    """
    assert (resolution is not None) ^ (n_clusters is not None)

    def cluster_func(res, vb):
        return leiden_clustering(
            n=n, edges=edges, resolution=res, n_runs=n_runs, seed=seed, verbose=vb
        )

    if resolution is not None:
        return cluster_func(resolution, verbose)

    right = 1
    while True:
        curr_labels = cluster_func(right, False)
        curr_n_clusters = len(np.unique(curr_labels))
        if verbose:
            print('A resolution of {} gives {} clusters.'.format(right, curr_n_clusters), flush=True)
        if curr_n_clusters == n_clusters:
            return curr_labels
        elif curr_n_clusters < n_clusters:
            right *= 2
        else:
            break
    left = 0 if right == 1 else right / 2
    # desired resolution is in [left, right)
    while left + resolution_tol < right:
        mid = (left + right) / 2
        curr_labels = cluster_func(mid, False)
        curr_n_clusters = len(np.unique(curr_labels))
        if verbose:
            print('A resolution of {} gives {} clusters.'.format(mid, curr_n_clusters), flush=True)
        if curr_n_clusters == n_clusters:
            return curr_labels
        elif curr_n_clusters > n_clusters:
            right = mid
        else:
            left = mid
    return curr_labels


def get_feature_edges(
        arr, pca_components=None,
        n_neighbors=15, metric='correlation', verbose=False
):
    """
    Compute k-nearest neighbors of data and return the UMAP graph.
    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    pca_components: None or int
        If None, then do not do PCA,
        else, number of components to keep when doing PCA de-noising for the data matrix.
    n_neighbors: int
        Number of neighbors desired
    metric: string, default='correlation'
        Metric used when constructing the initial knn graph
    verbose: bool, default=True
        Whether to print progress
    Returns
    -------
    rows, cols, vals: list
        Each edge is rows[i], cols[i], and its weight is vals[i]
    """
    arr = utils.drop_zero_variability_columns(arr=arr)
    if verbose:
        print("Constructing the graph...", flush=True)
    # use scanpy functions to do the graph construction
    adata = ad.AnnData(arr, dtype=np.float32)
    if pca_components is not None:
        sc.tl.pca(adata, n_comps=pca_components, svd_solver='arpack')
    if pca_components is not None:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=pca_components, use_rep='X_pca', metric=metric)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=None, use_rep='X', metric=metric)

    rows, cols = adata.obsp['connectivities'].nonzero()
    vals = adata.obsp['connectivities'][(rows, cols)].A1
    if verbose:
        print("Done!", flush=True)
    return rows, cols, vals
