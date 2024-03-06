import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
from scipy import stats
from scipy import ndimage
from scipy import spatial
from pysal.lib import weights
from pysal.explore import esda
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as matpatches
from matplotlib.figure import figaspect
from itertools import combinations
from collections import defaultdict

from typing import Union, List
import math
import time

def overlap_check(new_patch, existing_patches, max_overlap_ratio):
    """
    This function checks if the new patch overlaps with any of the existing patches.
    
    Parameters:
    new_patch: tuple
        The coordinates of the new patch to be checked for overlaps.
    existing_patches: list
        A list of existing patches to check for overlaps.
    max_overlap_ratio: float
        The maximum allowable overlap ratio for a new patch.
    
    Returns:
    bool
        Returns True if the new patch doesn't overlap with any existing patch beyond the allowable overlap ratio.
        Otherwise, returns False.
    """
    
    patch_area = (new_patch[2] - new_patch[0]) * (new_patch[3] - new_patch[1])
    max_overlap = max_overlap_ratio * patch_area
    for patch in existing_patches:
        dx = min(new_patch[2], patch[2]) - max(new_patch[0], patch[0])
        dy = min(new_patch[3], patch[3]) - max(new_patch[1], patch[1])
        if (dx>=0) and (dy>=0) and (dx*dy > max_overlap):
            return False
    return True

def contains_points(patch, spatial_values, min_points):
    """
    This function checks if a patch contains a certain number of spatial values (points).
    
    Parameters:
    patch: tuple
        The coordinates of the patch to be checked.
    spatial_values: list
        A list of spatial values (cells' coordinates) to check if they are within the patch.
    min_points: int
        The minimum number of points that the patch should contain.
        
    Returns:
    bool
        Returns True if the patch contains at least 'min_points' number of spatial values（cells' coordinates）. Otherwise, returns False.
    """
    
    # Count the points within the patch
    points_in_patch = sum((patch[0] <= point[0] <= patch[2]) and (patch[1] <= point[1] <= patch[3]) for point in spatial_values)
    
    # Check if the number of points in the patch is at least min_points
    return points_in_patch >= min_points

def calculate_shannon_entropy(counts:np.ndarray):
    """
    Calculate the Shannon entropy of a set of counts.

    Parameters
    ----------
    counts : numpy.ndarray
        An array of counts.

    Returns
    -------
    float
        The Shannon entropy of the counts.
    """
    
    N = np.sum(counts)
    if N > 0:
        probabilities = counts / N
        entropy = -np.sum(probabilities * np.log2(probabilities))
    else:
        entropy = 0.0
    
    return entropy

def calculate_simpson_index(counts:np.ndarray):
    """
    This function calculates the Simpson index of a set of counts.
    
    Parameters:
    counts: numpy.ndarray
        An array of counts.
        
    Returns:
    float
        The Simpson index of the counts.
    """
    
    N = np.sum(counts)
    if N > 1.0:
        simpson_index = np.sum(counts * (counts-1))/(N*(N-1))
    else:
        simpson_index = 1.0
    
    return simpson_index

def calculate_simpsonDiversity_index(counts:np.ndarray):
    """
    This function calculates the Simpson diversity index of a set of counts.
    
    Parameters:
    counts: numpy.ndarray
        An array of counts.
        
    Returns:
    float
        The Simpson diversity index of the counts.
    """
        
    return 1.0-calculate_simpson_index(counts)
    
def create_masked_lat2W(grid):
    mask = grid != -1  # Create a mask of valid locations.
    w = weights.lat2W(*grid.shape, rook=False)  # Create spatial weights.
    flat_mask = mask.flatten()  # Flatten the mask for index matching with the weights object.
    
    new_weights = {}
    new_neighbors = {}

    for idx, neighbors in w.neighbors.items():
        # print(f"index is {idx} has value {flat_mask[idx]}")
        if not flat_mask[idx]:  # If the current index is not valid, skip it.
            continue

        # Filter the neighbors and corresponding weights by the mask.
        valid_neighbors = [neighbor for i, neighbor in enumerate(neighbors) if flat_mask[neighbor]]
        valid_weights = [w.weights[idx][i] for i, neighbor in enumerate(neighbors) if flat_mask[neighbor]]
        new_neighbors[idx] = valid_neighbors
        new_weights[idx] = valid_weights

    # Create a new weights object using the filtered neighbors and weights.
    w_new = weights.W(neighbors=new_neighbors, weights=new_weights)
    
    return w_new
    
def map_spots_back_to_grid(spots, original_grid, masked_grid, fill_value=-1):
    """
    Map the identified spots back to the original grid.
    
    Parameters:
    spots: numpy array
        Boolean array with spots (e.g., hotspots, coldspots, etc.)
    original_grid: numpy array
        The original 2D grid.
    masked_grid: numpy array
        The masked 2D grid used in the analysis.
    fill_value: int or float, optional
        The value used in the original_grid to indicate missing or masked data. 
        Defaults to -1.
    
    Returns:
    mapped_spots: numpy array
        A boolean array of the same shape as original_grid, 
        with True at the positions of the identified spots.
    """
    # Initialize an array for mapped spots
    mapped_spots = np.zeros_like(original_grid, dtype=bool)
    
    # Iterate over the original grid to identify the mapping from masked to original indices
    masked_idx = 0
    for i in range(original_grid.shape[0]):
        for j in range(original_grid.shape[1]):
            # Check if the position is not masked
            if original_grid[i, j] != fill_value:
                # If the spot is identified in the masked grid, map it back to the original grid
                # print(spots[masked_idx])
                if spots[masked_idx]:
                    mapped_spots[i, j] = True
                masked_idx += 1
                
    return mapped_spots
    
def global_moran(grid:np.ndarray, tissue_only=False, plot_weights=False):
    """
    Perform global Moran's I test for spatial autocorrelation.
    
    Parameters
    ----------
    grid : numpy.ndarray
        The 2D grid of diversity indices to be analyzed.
    tissue_only : bool, optional
        If True, the analysis is restricted to tissue regions. Defaults to False.
    plot_weights : bool, optional
        If True, visualize the spatial weights matrix. Defaults to False.
    
    Returns
    -------
    I : float
        The Moran's I statistic.
    p : float
        The p-value for the test.
    """

    # Get dimensions
    n, m = grid.shape

    # Create a spatial weights matrix
    if not tissue_only:
        w = weights.lat2W(n, m, rook=False)
        mi = esda.Moran(grid.flatten(), w, transformation = 'r')
    else:
        masked_grid_nan = np.where(grid == -1, np.nan, grid)
        masked_grid = masked_grid_nan[~np.isnan(masked_grid_nan)]
        w = create_masked_lat2W(grid)
        print(f"Global Moran restricted to tissue region with w of shape{w.full()[0].shape}")
        mi = esda.Moran(masked_grid.flatten(), w, transformation = 'o')
        
    if plot_weights:
    # Visualize the weights matrix
        weights_matrix, ids = w.full()
        print(weights_matrix.shape)
        plt.figure(figsize=(10,10))
        plt.imshow(weights_matrix, cmap='hot', interpolation='none')
        plt.colorbar(label='Spatial Weights')
        plt.title('Spatial Weights Matrix')
        plt.show()
    
    return mi.I, mi.p_sim


def local_moran(grid: np.ndarray, tissue_only=False, p_value=0.01, seed=42, plot_weights=False):
    """
    Perform local Moran's I test (LISA) for local spatial autocorrelation

    Parameters
    ----------
    grid : numpy.ndarray
        The 2D grid of diversity indices to be analyzed.
    tissue_only : bool, optional
        If True, the analysis is restricted to tissue regions. Defaults to False.
    p_value : float, optional
        The p-value threshold for significance. Defaults to 0.01.
    seed : int, optional
        The random seed for reproducibility. Defaults to 42.
    plot_weights : bool, optional
        If True, visualize the spatial weights matrix. Defaults to False.

    Returns
    -------
    hotspots : numpy.ndarray
        Boolean array indicating hotspots (high value surrounded by high values).
    coldspots : numpy.ndarray
        Boolean array indicating coldspots (low value surrounded by low values).
    doughnuts : numpy.ndarray
        Boolean array indicating doughnuts (high value surrounded by low values).
    diamonds : numpy.ndarray
        Boolean array indicating diamonds (low value surrounded by high values).
    """

    # Get dimensions
    n, m = grid.shape
    
    # Create a spatial weights matrix
    if not tissue_only:
        w = weights.lat2W(n, m, rook=False)
        lisa = esda.Moran_Local(grid.flatten(), w, transformation='r', permutations=999, seed=seed)
    else:
        masked_grid_nan = np.where(grid == -1, np.nan, grid)
        masked_grid = masked_grid_nan[~np.isnan(masked_grid_nan)]
        print(masked_grid.shape)
        w = create_masked_lat2W(grid)
        print(f"Local Moran restricted to tissue region with w of shape{w.full()[0].shape}")
        lisa = esda.Moran_Local(masked_grid.flatten(), w, transformation='o', permutations=999, seed=seed)
        
    if plot_weights:
    # Visualize the weights matrix
        weights_matrix, ids = w.full()
        plt.figure(figsize=(6,6))
        plt.imshow(weights_matrix, cmap='hot', interpolation='none')
        plt.colorbar(label='Spatial Weights')
        plt.title('Spatial Weights Matrix')
        plt.show()

    # Identify significant hotspots, coldspots, doughnuts, and diamonds
    significant = lisa.p_sim < p_value
    hotspots = np.zeros((n, m), dtype=bool)
    coldspots = np.zeros((n, m), dtype=bool)
    doughnuts = np.zeros((n, m), dtype=bool)
    diamonds = np.zeros((n, m), dtype=bool)
    
    hotspots.flat[significant * (lisa.q==1)] = True
    coldspots.flat[significant * (lisa.q==3)] = True
    doughnuts.flat[significant * (lisa.q==4)] = True
    diamonds.flat[significant * (lisa.q==2)] = True
    
    if tissue_only:
        hotspots = map_spots_back_to_grid(hotspots.flatten(), grid, masked_grid)
        coldspots = map_spots_back_to_grid(coldspots.flatten(), grid, masked_grid)
        doughnuts = map_spots_back_to_grid(doughnuts.flatten(), grid, masked_grid)
        diamonds = map_spots_back_to_grid(diamonds.flatten(), grid, masked_grid)
    
    return hotspots, coldspots, doughnuts, diamonds

def generate_patches(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                     library_key: str, 
                     library_id: str, 
                     scaling_factor: Union[int, float], 
                     spatial_key: Union[str, List[str]]):
    """
    Generate a list of patches from a spatial data object.
    
    Parameters
    ----------
    spatial_data : Union[ad.AnnData, pd.DataFrame]
        The spatial data from which to generate patches.
    library_key : str
        The key identifying the library within the spatial data.
    library_id : str
        The identifier for the library within the spatial data.
    scaling_factor : Union[int, float]
        The scaling factor to determine the size of the patches.
    spatial_key : Union[str, List[str]]
        The key or list of keys to access the spatial data.

    Returns
    -------
    list
        A list of patches.
    """
    
    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
    
    # check if spatial_data is an AnnData object or a DataFrame, and get spatial coordinates
    if isinstance(spatial_data, ad.AnnData):
        spatial_values = spatial_data_filtered.obsm[spatial_key]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_values = spatial_data_filtered[spatial_key].values
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")

        
    width = spatial_values.max(axis=0)[0] - spatial_values.min(axis=0)[0]
    height = spatial_values.max(axis=0)[1] - spatial_values.min(axis=0)[1]
    
    # Define the size of each patch
    patch_width = width / scaling_factor
    patch_height = height / scaling_factor
    
    # Create a list to store the patch coordinates
    patches = []

    for y0 in np.arange(spatial_values.min(axis=0)[1], spatial_values.max(axis=0)[1], patch_height):
        for x0 in np.arange(spatial_values.min(axis=0)[0], spatial_values.max(axis=0)[0], patch_width):
            # Define the coordinates of the current patch
            x1 = x0 + patch_width
            y1 = y0 + patch_height
            # Add the current patch to the list
            patches.append((x0, y0, x1, y1))
    return patches

def generate_patches_randomly(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                              library_key:str, 
                              library_id:str, 
                              scaling_factor:Union[int,float], 
                              spatial_key:Union[str,List[str]], 
                              max_overlap=0.0, 
                              random_seed=None, 
                              min_points=2):
    """
    This function generates a list of patches from a spatial data object in a random manner.
    
    Parameters:
    spatial_data: anndata.AnnData or pandas.DataFrame
        The spatial data from which to generate patches.
    scaling_factor: int or float
        The scaling factor to determine the size of the patches.
    spatial_key: str or list
        The key or list of keys to access the spatial data.
    max_overlap: float, default=0.0
        The maximum allowable overlap ratio for a new patch.
    random_seed: int or None, default=None
        The seed for the random number generator.
    min_points: int, default=0.0
        The minimum number of points that the patch should contain.
        
    Returns:
    list
        A list of patches.
    """
    rng = np.random.default_rng(random_seed)

    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
    
    if isinstance(spatial_data, ad.AnnData):
        spatial_values = spatial_data_filtered.obsm[spatial_key]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_values = spatial_data_filtered[spatial_key].values
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")

    width = spatial_values.max(axis=0)[0] - spatial_values.min(axis=0)[0]
    height = spatial_values.max(axis=0)[1] - spatial_values.min(axis=0)[1]

    patch_width = width / scaling_factor
    patch_height = height / scaling_factor

    num_patches = int(scaling_factor ** 2)

    patches = []
    used_coordinates = set()
    
    for _ in range(num_patches):
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:  # seconds
                print(f"Warning: Could not generate a new patch within 5 seconds. Return {len(patches)} out of {num_patches} patches")
                return patches
            
            x0 = rng.uniform(spatial_values.min(axis=0)[0], spatial_values.max(axis=0)[0] - patch_width)
            y0 = rng.uniform(spatial_values.min(axis=0)[1], spatial_values.max(axis=0)[1] - patch_height)
            x1 = x0 + patch_width
            y1 = y0 + patch_height
            new_patch = (x0, y0, x1, y1)
            if (x0, y0) not in used_coordinates and overlap_check(new_patch, patches, max_overlap) and contains_points(new_patch, spatial_values, min_points):
                used_coordinates.add((x0, y0))
                patches.append(new_patch)
                break

    return patches

def display_patches(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                    library_key: str, 
                    library_ids: List[str],
                    spatial_key: Union[str, List[str]], 
                    cluster_keys: List[str],
                    scale:Union[int,float],
                    random_patch=False,
                    **kwargs):
    
    # Count the total number of plots to make
    total_plots = len(library_ids) * len(cluster_keys)
    num_rows = len(library_ids)
    num_cols = len(cluster_keys)
    print(f" The number of plot is {total_plots}")
    
    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 8*num_rows),squeeze=False)
    
    # Iterate over each library_id and cluster_key, making a plot for each
    for i, library_id in enumerate(library_ids):
        for j, cluster_key in enumerate(cluster_keys):
            if isinstance(spatial_data, ad.AnnData):
                spatial_data_region = spatial_data[spatial_data.obs[library_key] == library_id]
            elif isinstance(spatial_data, pd.DataFrame):
                spatial_data_region = spatial_data[spatial_data[library_key] == library_id]
            else:
                raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
                
            if random_patch:
                patches_coordinates = generate_patches_randomly(spatial_data_region, 
                                                                scaling_factor=scale, 
                                                                spatial_key=spatial_key, 
                                                                **kwargs)
            else:    
                patches_coordinates = generate_patches(spatial_data_region, 
                                                       library_key,
                                                       library_id,
                                                       scaling_factor=scale, 
                                                       spatial_key=spatial_key)
            
            ax = axes[i, j] if total_plots > 1 else axes
            sns.scatterplot(data=spatial_data_region, 
                            x=spatial_key[0], 
                            y=spatial_key[1], 
                            hue=cluster_key, 
                            palette='husl', 
                            s=6, 
                            legend='full', 
                            ax=ax)

            ax.set_title('Cluster: {} Region: {}'.format(cluster_key, library_id))
            ax.legend(bbox_to_anchor=(1.0, -0.1), ncol=3, fontsize='small')

            for patch in patches_coordinates:
                rect = matpatches.Rectangle((patch[0], patch[1]), 
                                          patch[2] - patch[0], 
                                          patch[3] - patch[1], 
                                          linewidth=0.5, 
                                          edgecolor='black', 
                                          facecolor='none')
                ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
    

def calculate_diversity_index(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                              library_key: str,
                              library_id: str, 
                              spatial_key: Union[str, List[str]],
                              patches: list,
                              cluster_key: str,
                              metric: str,
                              return_comp=False):
    """
    Calculate the heterogeneity index for a set of patches.

    Parameters
    ----------
    spatial_data : Union[ad.AnnData, pd.DataFrame]
        The spatial data to be used.
    library_key : str
        The key to access the library data.
    library_id : str
        The identifier of the library.
    spatial_key : Union[str, List[str]]
        The key or list of keys to access the spatial data.
    patches : list
        The list of patches to be analyzed.
    cluster_key : str
        The key to access the cluster data.
    metric : str
        The metric to be used for the heterogeneity index calculation.
    return_comp : bool, optional
        If True, return a comprehensive object with additional details
        beyond the heterogeneity indices. Defaults to False.

    Returns
    -------
    pandas.Series
        A series of heterogeneity indices, or if `return_comp` is True,
        a more comprehensive object with additional details.
    """

    METRIC_FUNCTIONS = {
        'Shannon Diversity': calculate_shannon_entropy,
        'Simpson': calculate_simpson_index,
        'Simpson Diversity':calculate_simpsonDiversity_index
    }
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric '{metric}'. Available metrics are: {list(METRIC_FUNCTIONS.keys())}")
    else:
        metric_function = METRIC_FUNCTIONS[metric]
    
    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
        
    patch_indices = {}
    patches_comp = [None]*len(patches)
    
    # Iterate over patches
    for i, patch in enumerate(patches):
        x0, y0, x1, y1 = patch

        # Filter adata for cells within the current patch
        if isinstance(spatial_data, ad.AnnData):
            spatial_data_patch = spatial_data_filtered[
                (spatial_data_filtered.obsm[spatial_key][:, 0] >= x0) & 
                (spatial_data_filtered.obsm[spatial_key][:, 0] <= x1) & 
                (spatial_data_filtered.obsm[spatial_key][:, 1] >= y0) & 
                (spatial_data_filtered.obsm[spatial_key][:, 1] <= y1)
            ]
            try:
                # Count the number of cells/clusters of each type
                cell_type_comp = spatial_data_patch.obs[cluster_key].value_counts()
                cell_type_counts = cell_type_comp.values
            except KeyError:
                raise ValueError(f"cluster_key '{cluster_key}' not found in spatial_data_patch for AnnData")

        elif isinstance(spatial_data, pd.DataFrame):
            spatial_data_patch = spatial_data_filtered[
                (spatial_data_filtered[spatial_key[0]] >= x0) &
                (spatial_data_filtered[spatial_key[0]] <= x1) &
                (spatial_data_filtered[spatial_key[1]] >= y0) &
                (spatial_data_filtered[spatial_key[1]] <= y1)
            ]
            try:
                cell_type_comp = spatial_data_patch[cluster_key].value_counts()
                cell_type_counts = cell_type_comp.values
            except KeyError:
                raise ValueError(f"cluster_key '{cluster_key}' not found in spatial_data_patch for DataFrame")

        else:
            raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")

        if cell_type_counts.sum()!=0:
            non_zero_counts = cell_type_counts[cell_type_counts != 0]
            diversity_index = metric_function(non_zero_counts)
            patch_indices[i] = diversity_index
            patches_comp[i] = cell_type_comp
            
    print(f"{100*(1-len(patch_indices)/len(patches)):.3f} per cent patches are empty")
    if return_comp:
        return pd.Series(patch_indices), patches_comp
    
    return pd.Series(patch_indices)
        

def multiscale_diversity(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                         scales: Union[tuple, list], 
                         library_key: str,
                         library_ids: Union[tuple, list], 
                         spatial_key: Union[str, List[str]],
                         cluster_key: str,
                         mode='D',
                         random_patch=False,
                         plotfigs=False, 
                         savefigs=False,
                         patch_kwargs={},  
                         other_kwargs={}):
    """
    Calculate the multiscale neighbourhood heterogeneity.

    Parameters
    ----------
    spatial_data : Union[ad.AnnData, pd.DataFrame]
        The spatial data to be used.
    scales : Union[tuple, list]
        The scales to be used for the analysis.
    library_key : str
        The key to access the library data.
    library_ids : Union[tuple, list]
        The identifiers of the libraries.
    spatial_key : Union[str, List[str]]
        The key or list of keys to access the spatial data.
    cluster_key : str
        The key to access the cluster data.
    mode : str, optional
        The mode of operation, default is 'D'.
    random_patch : bool, optional
        Whether to generate patches in a random manner. Defaults to False.
    plotfigs : bool, optional
        Whether to plot the figures. Defaults to False.
    savefigs : bool, optional
        Whether to save the figures. Defaults to False.
    patch_kwargs : dict, optional
        Additional keyword arguments for the patch generation. Defaults to an empty dict.
    other_kwargs : dict, optional
        Other keyword arguments. Defaults to an empty dict.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        A dataframe of results and a dataframe of slopes, respectively.
    """
    
    # Prepare a dictionary to store the results
    results = pd.DataFrame(index = scales, columns = library_ids)
    slopes = {library_id: [] for library_id in library_ids}
    
    # Generate sequence of overlap for each scale if not specified in patch_kwargs
    if "max_overlap" not in patch_kwargs and random_patch:
        max_scale = max(scales)
        min_scale = min(scales)
        normalized_sequence = [(x-min_scale)/(max_scale-min_scale) for x in scales]
        overlaps = [0.5+0.4*x for x in normalized_sequence]
        print(overlaps,flush=True)
    
    for i,scale in enumerate(scales):
        
        # Use the overlap from patch_kwargs if specified, otherwise use the generated sequence
        if "max_overlap" not in patch_kwargs and random_patch:
            patch_kwargs["max_overlap"] = overlaps[i]
            
        for library_id in library_ids:
            print(f"Processing region: {library_id} at scale {scale}", flush=True)
            
            # Generate the patch coordinates 
            if random_patch:
                patches = generate_patches_randomly(spatial_data, 
                                                    library_key,
                                                    library_id,
                                                    scale,
                                                    spatial_key,
                                                    **patch_kwargs)

            else:
                patches = generate_patches(spatial_data, 
                                           library_key,
                                           library_id,
                                           scale,
                                           spatial_key)
                
            # Calculate the diversity index
            indices = calculate_diversity_index(spatial_data=spatial_data, 
                                                library_key=library_key, 
                                                library_id=library_id, 
                                                spatial_key=spatial_key,
                                                patches=patches,
                                                cluster_key=cluster_key,
                                                **other_kwargs)
                
            count = indices.sum()/len(indices)
            print(f"{library_id} at scale {scale} has {indices.eq(0.0).sum()} patches with zero diveristy", flush=True)
            print(f"{library_id} at scale {scale} diversity is {count}", flush=True)
            
            
            # Store the result
            if mode == 'D':
                results.loc[scale,library_id] = count
            elif mode == 'M':
                # Calculate Moran's I
                grid = diversity_heatmap(spatial_data=spatial_data,
                                         library_key=library_key,
                                         library_id=library_id,
                                         spatial_key=spatial_key,
                                         patches=patches, 
                                         heterogeneity_indices=indices,
                                         plot=False)
                moranI, moranI_p = global_moran(grid)
                results.loc[scale,library_id] = moranI
            else:
                print(f"{mode} currently not supported", flush=True)
            
            
    scales = np.log2(np.reciprocal(scales))
    if plotfigs:
        # Plot the results
        num_library_ids = len(library_ids)
        num_cols = min(num_library_ids, 2) 
        num_rows = math.ceil(num_library_ids / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
        if num_library_ids == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, library_id in enumerate(library_ids):
            counts = results[library_id].values.astype(np.float64)
            #counts = np.log2(counts)
            axes[i].scatter(scales, counts, marker='x', c='black', label=library_id)

            # Calculate the slope and intercept of the best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(scales, counts)
            slopes[library_id].append(slope)

            # Plot the best fit line
            axes[i].plot(scales, [slope*x + intercept for x in scales], c='black', label='best fit line')
            axes[i].text(scales[-1], counts[-1], f'slope = {slope:.4f}', ha='right')
            axes[i].set_xlabel('log(1/s)')
            axes[i].set_ylabel('N(s)')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()
        if savefigs:
            fig.savefig('fractal_dim')
    else:
        for i, library_id in enumerate(library_ids):
            counts = results[library_id].values.astype(np.float64)
            #counts = np.log2(counts)
            slope, intercept, r_value, p_value, std_err = stats.linregress(scales, counts)
            slopes[library_id].append(slope)
    return results, pd.DataFrame(slopes)

def combination_freq(series:list, n=2, top=10, cell_type_combinations=None):
    transactions = [set(serie[serie != 0].index) for serie in series if serie is not None]
    co_occurrence = defaultdict(int)
    
    for transaction in transactions:
        for comb in combinations(transaction, n): 
            
            # Ensure a consistent order for the cell types
            sorted_comb = tuple(sorted(comb)) # because sorted returns a list and I want tuple
            co_occurrence[sorted_comb] += 1
            
    # Sort the pairs by their co-occurrence count
    sorted_co_occurrence = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)

    # If user-selected combinations are provided, filter the list
    if cell_type_combinations:
        sorted_co_occurrence = [item for item in sorted_co_occurrence if item[0] in cell_type_combinations]
    
    total_hotspots = len(transactions)
    top_n_comb = {}
    
    if top:
        sorted_co_occurrence = sorted_co_occurrence[:top]
        
    for pair, count in sorted_co_occurrence:
        frequency = count / total_hotspots
        top_n_comb[pair] = frequency
        
    return pd.Series(top_n_comb)
    
# This version of combination_freq considers expected co-occurrence and compute the log2 ratio
# def combination_freq(series: list, n=2, top=10, cell_type_combinations=None):
#     transactions = [set(serie[serie != 0].index) for serie in series if serie is not None]
    
#     # Count the occurrence of each cell type across all patches
#     cell_type_counts = defaultdict(int)
#     for transaction in transactions:
#         for cell_type in transaction:
#             cell_type_counts[cell_type] += 1
    
#     total_patches = len(transactions)
    
#     # Calculate co-occurrence
#     co_occurrence = defaultdict(int)
#     for transaction in transactions:
#         for comb in combinations(transaction, n):
#             sorted_comb = tuple(sorted(comb))  # Sort to ensure consistent order
#             co_occurrence[sorted_comb] += 1
    
#     # Calculate expected co-occurrence and the ratio
#     co_occurrence_ratio = {}
#     for comb, observed in co_occurrence.items():
#         expected = (cell_type_counts[comb[0]] / total_patches) * \
#                    (cell_type_counts[comb[1]] / total_patches) * \
#                    total_patches
#         #co_occurrence_ratio[comb] = observed / expected if expected else 0  # Avoid division by zero
#         co_occurrence_ratio[comb] = np.log2(observed / expected) if expected > 0 and observed > 0 else 0

#     # Sort by the ratio of observed to expected co-occurrence
#     sorted_co_occurrence = sorted(co_occurrence_ratio.items(), key=lambda x: x[1], reverse=True)

#     # Filter for user-selected combinations if provided
#     if cell_type_combinations:
#         sorted_co_occurrence = [item for item in sorted_co_occurrence if item[0] in cell_type_combinations]

#     # Take the top N combinations if requested
#     top_n_comb = {}
#     if top:
#         sorted_co_occurrence = sorted_co_occurrence[:top]
        
#     for pair, ratio in sorted_co_occurrence:
#         top_n_comb[pair] = ratio
        
#     return pd.Series(top_n_comb)

def diversity_heatmap(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                      library_key: str,
                      library_id: str,
                      spatial_key: Union[str, List[str]], 
                      patches,
                      heterogeneity_indices, 
                      tissue_only=False,
                      plot=True,
                      return_fig=False):
    """
    This function visualizes the heterogeneity indices as a heatmap on the original spatial data.

    Parameters
    ----------
    spatial_data : Union[ad.AnnData, pd.DataFrame]
        The spatial data to be used for visualization.
    library_key : str
        The key associated with the library in the `spatial_data`.
    library_id : str
        The identifier for the library to be used in the analysis.
    spatial_key : Union[str, List[str]]
        The key(s) identifying the spatial information within `spatial_data`.
    patches : list
        The list of patches to be analyzed. Each patch should correspond to a specific region in the spatial data.
    heterogeneity_indices : pandas.Series
        The heterogeneity indices to be visualized. Each value in this series corresponds to a patch, indicating its heterogeneity level.
    tissue_only : bool, optional
        If True, only tissue regions are considered in the analysis. Defaults to False.
    plot : bool, optional
        If True, a heatmap is plotted. Defaults to True.
    return_fig : bool, optional
        If True, the matplotlib figure is returned. This is useful for further customization of the plot. Defaults to False.

    Returns
    -------
    numpy.ndarray
        A grid where each cell represents the heterogeneity index for a corresponding patch.
    matplotlib.figure.Figure, optional
        The matplotlib figure object if `return_fig` is True and `plot` is True; otherwise, this is not returned.

    Notes
    -----
    This function requires that the spatial data is properly formatted and that the heterogeneity indices have been previously calculated. 
    """  

    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")    
    
    if isinstance(spatial_data, ad.AnnData):
        x_coords = spatial_data_filtered.obsm[spatial_key][:, 0]
        y_coords = spatial_data_filtered.obsm[spatial_key][:, 1]
    elif isinstance(spatial_data, pd.DataFrame):
        x_coords = spatial_data_filtered[spatial_key[0]]
        y_coords = spatial_data_filtered[spatial_key[1]]
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
        
    if plot:
        min_x, min_y = np.min(x_coords), np.min(y_coords)
        max_x, max_y = np.max(x_coords), np.max(y_coords)
        
        # Create a 2D grid
        grid = np.zeros((int(max_y - min_y + 1), int(max_x - min_x + 1)))

        # Fill the grid with heterogeneity indices
        for patch, heterogeneity_index in heterogeneity_indices.items():
            x0, y0, x1, y1 = patches[patch]
            grid[int(y0-min_y):int(y1-min_y+1), int(x0-min_x):int(x1-min_x+1)] = heterogeneity_index
            
        # Plot the heatmap
        plt.imshow(grid, cmap='seismic', interpolation='none')
        plt.colorbar(label='Diversity Index')
        if return_fig:
            fig = plt.gcf()
        plt.show()
        
    s = int(math.sqrt(len(patches)))
    if tissue_only:
        grid = np.full((s,s), -1)
    else:
        grid = np.zeros((s,s))
        
    for patch, heterogeneity_index in heterogeneity_indices.items():
        grid[patch//s, patch%s] = heterogeneity_index
        
    if return_fig and plot:
        return grid, fig
    return grid

def diversity_clustering(spatial_data:Union[ad.AnnData,pd.DataFrame], 
                         scale:float, 
                         library_key:str,
                         library_ids:Union[tuple, list], 
                         spatial_key:Union[str,List[str]],
                         cluster_key:str,
                         hotspot=True,
                         whole_tissue=False,
                         p_value=0.01,
                         combination=2,
                         top=15,
                         selected_comb=None,
                         restricted=False,
                         **kwargs):
    merged_series_dict = {}
    global_moranI = {library_id: [] for library_id in library_ids}
    comb_freq_dict = {}
    
    for library_id in library_ids:
        print(f"Processing region: {library_id} at scale {scale}", flush=True)

        # Generate the patch coordinates 
        patches = generate_patches(spatial_data, 
                                   library_key,
                                   library_id,
                                   scale,
                                   spatial_key)

        # Calculate the heterogeneity index
        indices, patches_comp = calculate_diversity_index(spatial_data=spatial_data, 
                                                              library_key=library_key, 
                                                              library_id=library_id, 
                                                              spatial_key=spatial_key,
                                                              patches=patches,
                                                              cluster_key=cluster_key,
                                                              return_comp = True,
                                                              **kwargs)
        
        grid = diversity_heatmap(spatial_data=spatial_data,
                                 library_key=library_key,
                                 library_id=library_id,
                                 spatial_key=spatial_key,
                                 patches=patches, 
                                 heterogeneity_indices=indices,
                                 tissue_only=restricted,
                                 plot=False)
        
        moranI, moranI_p = global_moran(grid, tissue_only=restricted)
        global_moranI[library_id].append(moranI)
        
        hotspots, coldspots, doughnuts, diamonds = local_moran(grid, tissue_only=restricted, p_value = p_value)
        
        if hotspot:
            print(f"Region {library_id} contains {sum(hotspots.flatten())} diversity hotspots", flush=True)
            filtered_patches_comp = [patch for patch, is_hotspot in zip(patches_comp, hotspots.flatten()) if is_hotspot]
        else:
            print(f"Region {library_id} contains {sum(coldspots.flatten())} diversity coldspots", flush=True)
            filtered_patches_comp = [patch for patch, is_coldspot in zip(patches_comp, coldspots.flatten()) if is_coldspot]
            
        # Concatenate the series into a DataFrame
        if filtered_patches_comp and not all(item is None for item in filtered_patches_comp):
            if not whole_tissue:
                print('Considering only hotspots')
                comb_freq = combination_freq(filtered_patches_comp, n=combination, top=top, cell_type_combinations=selected_comb)
            else:
                print('Considering whole tissue')
                comb_freq = combination_freq(patches_comp, n=combination, top=top, cell_type_combinations=selected_comb)
                
            comb_freq_dict[library_id] = comb_freq
            merged_series = pd.concat(filtered_patches_comp, axis=1).sum(axis=1)

            # Store the merged_series in the dictionary with the corresponding library_id
            merged_series_dict[library_id] = merged_series
        else:
            print(f"Region {library_id} has no diversity hot/cold spot since length of filterd_patch_comp is either {len(filtered_patches_comp)} or hot/cold spots contain no cells")

    # Return the dictionary containing all merged_series, and Moran's I, and a New Metric
    return merged_series_dict, global_moranI, comb_freq_dict

def find_coordinates(array, value):
    return np.argwhere(array == value)

def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def label_islands(arr, rook=True):
    """
    Label the islands of True values in the array.
    """
    if not rook:
        s = [[1,1,1],
             [1,1,1],
             [1,1,1]]
    else:
        s = [[0,1,0],
             [1,1,1],
             [0,1,0]]
    labeled, num_features = ndimage.label(arr, structure=s)
    return labeled, num_features

def compute_areas(labeled):
    """
    Compute the area of each labeled island.
    """
    return ndimage.sum(labeled > 0, labeled, range(1, labeled.max() + 1))

def distance_transform_edt(labeled):
    """
    Compute the Euclidean distance transform.
    """
    return ndimage.distance_transform_edt(~labeled.astype(bool))

def nearest_neighbour_distance(labeled, num_features):
    """
    Compute the nearest neighbour distance for each island.
    """
    
    island_coords = {}
    for i in range(1, num_features+1):
        island_coords[i] = find_coordinates(labeled, i)
    distances = []
    
    for i in range(1, num_features+1):
        min_distance = float('inf')
        tree = spatial.cKDTree(island_coords[i])
        for j in range(1, num_features+1):
            if i != j:
                dist, _ = tree.query(island_coords[j], k=1)
                min_distance = min(min_distance, np.min(dist))
        distances.append(min_distance)
                
    return distances

def compute_proximity_index(arr, rook=True):
    """
    Compute the Proximity Index for the sample.
    """
    labelled, num_islands = label_islands(arr, rook=rook)
    print(f"{num_islands} islands identified", flush=True)
    areas = compute_areas(labelled)
    distances = nearest_neighbour_distance(labelled, num_islands)
    
    # Calculate the proximity index using the given formula
    proximity_index = sum([areas[i] / distances[i] for i in range(num_islands)])
    
    return proximity_index


def sample_proximity(spatial_data:Union[ad.AnnData,pd.DataFrame], 
                     scale:float, 
                     library_key:str,
                     library_ids:Union[tuple, list], 
                     spatial_key:Union[str,List[str]],
                     cluster_key:str,
                     hotspot=True,
                     p_value=0.01,
                     **kwargs):
    
    PX = {library_id: [] for library_id in library_ids}
    
    for library_id in library_ids:
        print(f"Processing region: {library_id} at scale {scale}", flush=True)

        # Generate the patch coordinates 
        patches = generate_patches(spatial_data, 
                                   library_key,
                                   library_id,
                                   scale,
                                   spatial_key)

        # Calculate the heterogeneity index
        indices, patches_comp = calculate_diversity_index(spatial_data=spatial_data, 
                                                          library_key=library_key, 
                                                          library_id=library_id, 
                                                          spatial_key=spatial_key,
                                                          patches=patches,
                                                          cluster_key=cluster_key,
                                                          return_comp = True,
                                                          **kwargs)
        
        grid = diversity_heatmap(spatial_data=spatial_data,
                                 library_key=library_key,
                                 library_id=library_id,
                                 spatial_key=spatial_key,
                                 patches=patches, 
                                 heterogeneity_indices=indices,
                                 plot=False)
        
        hotspots, coldspots, _, _ = local_moran(grid, p_value=p_value)
        
        if hotspot:
            print(f"Region {library_id} contains {sum(hotspots.flatten())} diversity hotspots", flush=True)
            px = compute_proximity_index(hotspots)
        else:
            print(f"Region {library_id} contains {sum(coldspots.flatten())} diversity coldspots", flush=True)
            px = compute_proximity_index(coldspots)
            
        PX[library_id].append(px)
        
    # Return the dictionary containing Proximity Index
    return PX
    
def plot_cells_patches(spatial_data, 
                       library_key, 
                       library_id, 
                       spatial_key, 
                       cluster_key, 
                       patches,
                       selected_cell_types=None,
                       marker_styles=None,
                       color_palette=None,
                       patch_alpha=0.3):
    
    all_data = []

    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
        x_coords = spatial_data_filtered.obsm[spatial_key][:, 0]
        y_coords = spatial_data_filtered.obsm[spatial_key][:, 1]
        if selected_cell_types:
            spatial_data_filtered = spatial_data_filtered[spatial_data_filtered.obs[cluster_key].isin(selected_cell_types)]
        else:
            print("No cell types are selected, will proceed with all cell types", flush=True)
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
        x_coords = spatial_data_filtered[spatial_key[0]]
        y_coords = spatial_data_filtered[spatial_key[1]]
        if selected_cell_types:
            spatial_data_filtered = spatial_data_filtered[spatial_data_filtered[cluster_key].isin(selected_cell_types)]
        else:
            print("No cell types are selected, will proceed with all cell types", flush=True)
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")

    width = x_coords.max(axis=0) - x_coords.min(axis=0)
    print(width)
    height = y_coords.max(axis=0) - y_coords.min(axis=0)
    print(height)
    w, h = figaspect(height/width)

    for patch in patches:
        x0, y0, x1, y1 = patch
        # For visualisation purpose, only one side is kept, otherwise would have non-unique warnings
        if isinstance(spatial_data, ad.AnnData):
            spatial_data_patch = spatial_data_filtered[
                (spatial_data_filtered.obsm[spatial_key][:, 0] >= x0) & 
                (spatial_data_filtered.obsm[spatial_key][:, 0] < x1) & 
                (spatial_data_filtered.obsm[spatial_key][:, 1] >= y0) & 
                (spatial_data_filtered.obsm[spatial_key][:, 1] < y1)
            ]
        elif isinstance(spatial_data, pd.DataFrame):
            spatial_data_patch = spatial_data_filtered[
                (spatial_data_filtered[spatial_key[0]] >= x0) &
                (spatial_data_filtered[spatial_key[0]] < x1) &
                (spatial_data_filtered[spatial_key[1]] >= y0) &
                (spatial_data_filtered[spatial_key[1]] < y1)
            ]

        all_data.append(spatial_data_patch)
        
    if isinstance(spatial_data, ad.AnnData):
        concatenated_data = ad.concat(all_data, join='outer')
        x_data = concatenated_data.obsm[spatial_key][:, 0]
        y_data = concatenated_data.obsm[spatial_key][:, 1]
        hue_data = concatenated_data.obs[cluster_key].astype('object')
    else:  # for pandas DataFrame
        concatenated_data = pd.concat(all_data, axis=0)
        x_data = concatenated_data[spatial_key[0]]
        y_data = concatenated_data[spatial_key[1]]
        hue_data = concatenated_data[cluster_key].astype('object')

    fig, ax = plt.subplots(figsize=(w, h))
    sns.scatterplot(
        x=x_data,
        y=y_data,
        hue=hue_data,
        palette=color_palette,
        style=hue_data, 
        markers=marker_styles,
        s=20,
        ax=ax,
        rasterized=False,
    )
    ax.set_xlim(x_coords.min(axis=0), x_coords.min(axis=0)+width)
    ax.set_ylim(y_coords.min(axis=0), y_coords.min(axis=0)+height)
    
    # Overlay patches
    for patch in patches:
        x0, y0, x1, y1 = patch
        rect = matpatches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='none', facecolor='r', alpha=patch_alpha)
        ax.add_patch(rect)

    ax.invert_yaxis()
    ax.set_title('Cells within specified patches')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(bbox_to_anchor=(1.0, -0.1), ncol=3, fontsize='small')
    plt.show()
    return fig

# Function to create a circos plot
def create_circos_plot(df, 
                       cell_type_colors_hex=None,
                       cell_abundance=None, 
                       threshold=0.05,
                       edge_weights_scaler=42,
                       highlighted_edges=None,
                       node_weights_scaler=10000,
                       figure_size=(6,6),
                       save_path=None):

    # Function to apply an offset to the label positions
    def offset_label_position(pos, x_offset=0.15, y_offset=0.1):
        pos_offset = {}
        for node, coordinates in pos.items():
            theta = np.arctan2(coordinates[1], coordinates[0])
            radius = np.sqrt(coordinates[0] ** 2 + coordinates[1] ** 2)
            radius_offset = radius + radius * x_offset  # Increase the radius by x_offset%
            pos_offset[node] = (radius_offset * np.cos(theta), radius_offset * np.sin(theta))
        return pos_offset
        
    G = nx.Graph()
    for col in df.columns:
        cell_1, cell_2 = col
        if cell_type_colors_hex:
            G.add_node(cell_1, color=cell_type_colors_hex.get(cell_1))  # Set color if provided
            G.add_node(cell_2, color=cell_type_colors_hex.get(cell_2))  # Set color if provided
        else:
            G.add_node(cell_1)
            G.add_node(cell_2)
        weight = df[col].values[0]
        if weight >= threshold:
            G.add_edge(cell_1, cell_2, weight=weight*edge_weights_scaler)
              
    pos = nx.circular_layout(G, scale=1)
    
    ## Set node color by number of edges
    if not cell_type_colors_hex:
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        cmap = plt.cm.coolwarm
        
        # Normalize the degrees to get a value between 0 and 1 to index the colormap
        norm = plt.Normalize(vmin=min_degree, vmax=max_degree)
        node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]
    else:
        node_colors = [G.nodes[node]['color'] for node in G]
        
    # Scale node sizes from cell_abundance
    node_sizes = [cell_abundance[node].values[0] * node_weights_scaler for node in G.nodes()] 
    plt.figure(figsize=figure_size)
    ax = plt.gca()
    
    # Draw all nodes at once
    nx.draw_networkx_nodes(G, 
                           pos=pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           margins=[0.25,0.25])
    
    # Draw the node labels with the abundance values
    label_pos = offset_label_position(pos, x_offset=0.25)
    node_labels = {node: f'{node}\n{cell_abundance[node].values[0]*100:.2f}%' for node in G.nodes()}
    nx.draw_networkx_labels(G, 
                            pos=label_pos,
                            labels=node_labels,
                            font_family='Arial',
                            font_size=12)

    
    ## Customized Edge Drawing
    center_x, center_y = np.mean([pos[node] for node in G.nodes()], axis=0)
    highlight_color = 'g'  
    default_edge_color = 'k'  
    
    for edge in G.edges():
        node1, node2 = edge
        
        # Calculate the angle between the nodes and the center
        angle1 = np.arctan2(pos[node1][1] - center_y, pos[node1][0] - center_x)
        angle2 = np.arctan2(pos[node2][1] - center_y, pos[node2][0] - center_x)
    
        # Normalize the angles between -pi and pi
        angle1 = (angle1 + np.pi) % (2 * np.pi) - np.pi
        angle2 = (angle2 + np.pi) % (2 * np.pi) - np.pi
    
        # Calculate the angular distance and direction
        angular_dist = angle2 - angle1
        if angular_dist > np.pi:
            angular_dist -= 2 * np.pi
        elif angular_dist < -np.pi:
            angular_dist += 2 * np.pi
        
        # Set the curvature direction based on the angular distance
        rad = 0.2 * (np.pi - abs(angular_dist)) * (-1 if angular_dist > 0 else 1)
        
        # Check if the current edge should be highlighted
        if highlighted_edges and (edge in highlighted_edges or (edge[1], edge[0]) in highlighted_edges):
            edge_color = highlight_color
            transparency = 0.7
        else:
            edge_color = default_edge_color
            transparency = 0.3
            
        # Draw the edge with the customized curvature, direction, and color
        nx.draw_networkx_edges(G, 
                               pos=pos,
                               edgelist=[edge],
                               edge_color=edge_color,
                               width=[G[u][v]['weight'] for u, v in [edge]],
                               alpha=transparency,
                               arrows=True,
                               arrowstyle='-',
                               arrowsize=0,
                               connectionstyle=f'arc3,rad={rad}')

    ## Figure Settings
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    
    ## Create a legend for edge widths
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    min_weight = np.min(edge_widths)  
    max_weight = np.max(edge_widths)
    num_values = 4 
    legend_values = np.linspace(min_weight, max_weight, num_values)
    
    for width in legend_values:
        plt.plot([0], [0], color='k', alpha=0.3, linewidth=width, 
                 label=f'{width/edge_weights_scaler:.2f}')
    plt.legend(loc='lower center', title = 'Edge width key: Co-Occurrence Frequency',bbox_to_anchor=(0.5, -0.05),ncol=len(legend_values))
    
    ## Create a colorbar for node color
    if not cell_type_colors_hex:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='Node Color Key')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
    else:
        plt.show()
