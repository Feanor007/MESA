import math
import time
from typing import Union, List, Optional, Tuple
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
from scipy import stats
from pysal.lib import weights
from pysal.explore import esda
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as matpatches
from matplotlib.figure import figaspect
from matplotlib.colors import ListedColormap

from mesa.ecospatial._utils import (
    _append_series_to_df,
    _label_islands,
    _compute_areas,
    _nearest_neighbour_distance,
    _overlap_check,
    _contains_points
)


def calculate_shannon_entropy(counts: np.ndarray) -> float:
    """
    Calculate the Shannon entropy of a set of counts.

    Parameters
    ----------
    counts : :class:`numpy.ndarray`
        An array of counts from which to calculate the entropy. Each count represents the frequency of a distinct event in the dataset.

    Returns
    -------
    :class:`float`
        The Shannon entropy calculated from the counts.
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
    :class:`float`
        The Simpson index of the counts.
    """
    counts = np.array(counts)
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
    :class:`float`
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
    
def global_spatial_stats(grid:np.ndarray, mode='MoranI', tissue_only=False, plot_weights=False)-> Tuple[float, float]:
    """
    Perform global spatial autocorrelation analysis on a 2D grid of diversity indices.

    Parameters
    ----------
    grid : :class:`numpy.ndarray`
        The 2D grid containing diversity indices for spatial autocorrelation analysis.
    mode : str, optional
        The spatial statistic method to be used, options include 'MoranI', 'GearyC', and 'GetisOrdG'. Default is 'MoranI'.
    tissue_only : bool, optional
        If set to True, the analysis is restricted to only tissue regions, excluding non-tissue areas. Default is False.
    plot_weights : bool, optional
        If set to True, visualizes the spatial weights matrix to help in understanding spatial relationships. Default is False.

    Returns
    -------
    :class:`float`
        The calculated spatial statistic indicating the degree of autocorrelation.
    :class:`float`
        The p-value associated with the spatial statistic test, indicating statistical significance.
    """
    
    # Get dimensions
    n, m = grid.shape

    # Get spatial stats function
    SPATIAL_STATS_FUNCTIONS = {
        'MoranI': esda.Moran,
        'GearyC': esda.Geary,
        'Getis-OrdG':esda.G,
        'Getis-OrdG*':esda.G
    }
    
    if mode not in SPATIAL_STATS_FUNCTIONS:
        raise ValueError(f"Unknown metric '{mode}'. Available metrics are: {list(SPATIAL_STATS_FUNCTIONS.keys())}")
    else:
        stats_function = SPATIAL_STATS_FUNCTIONS[mode]
        
    # Create a spatial weights matrix
    if not tissue_only:
        w = weights.lat2W(n, m, rook=False)
        if mode[-1] == 'I':
            # mi = esda.Moran(grid.flatten(), w, transformation = 'r')
            mi = stats_function(grid.flatten(), w, transformation = 'r')
        elif mode[-1] == 'C':
            mi = stats_function(grid.flatten(), w, transformation = 'r')
        elif mode[-1] == 'G':
            # Getis-Ord G doesn't transform weights, keep the binarized (0,1) form
            mi = stats_function(grid.flatten(), w)
        elif mode[-2:] == 'G*':
            w =  weights.fill_diagonal(w, val=1.0)
            mi = stats_function(grid.flatten(), w)
    else:
        masked_grid_nan = np.where(grid == -1, np.nan, grid)
        masked_grid = masked_grid_nan[~np.isnan(masked_grid_nan)]
        w = create_masked_lat2W(grid)
        print(f"Global spatial stats restricted to tissue region with w of shape {w.full()[0].shape}")
        if mode[-1] == 'I':
            #  mi = esda.Moran(masked_grid.flatten(), w, transformation = 'o')
            mi = stats_function(masked_grid.flatten(), w, transformation = 'o')
        elif mode[-1] == 'C':
            mi = stats_function(masked_grid.flatten(), w, transformation = 'o')
        elif mode[-1] == 'G':
            mi = stats_function(masked_grid.flatten(), w)
        elif mode[-2:] == 'G*':
            w =  weights.fill_diagonal(w, val=1.0)
            mi = stats_function(grid.flatten(), w)
        
    if plot_weights:
    # Visualize the weights matrix
        weights_matrix, ids = w.full()
        print(weights_matrix.shape)
        plt.figure(figsize=(10,10))
        plt.imshow(weights_matrix, cmap='hot', interpolation='none')
        plt.colorbar(label='Spatial Weights')
        plt.title('Spatial Weights Matrix')
        plt.show()
        
    if mode[-1] == 'I':
        return mi.I, mi.p_sim
    elif mode[-1] == 'C':
        return mi.C, mi.p_sim
    elif mode[-1] == 'G':
        return mi.G, mi.p_sim
    elif mode[-2:] == 'G*':
        grid = grid.flatten()
        grid = grid.reshape(len(grid),1)
        return mi.G*((grid*grid.T).sum()-(grid*grid).sum())/((grid*grid.T).sum()), mi.p_sim
    else:
        return None

def local_spatial_stats(grid:np.ndarray, mode='MoranI', tissue_only=False, p_value=0.01, seed=42, plot_weights=False, return_stats=False)-> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local indicators of spatial association (LISA) for local spatial autocorrelation on a 2D grid of diversity indices,
    identifying significant hotspots and coldspots based on the chosen statistical method.

    Parameters
    ----------
    grid : :class:`numpy.ndarray`
        The 2D grid containing diversity indices for local spatial autocorrelation analysis.
    mode : str, optional
        The spatial statistic method to be used, options include 'MoranI', 'GearyC', and 'GetisOrdG'. Default is 'MoranI'.
    tissue_only : bool, optional
        If set to True, restricts the analysis to tissue regions only. Default is False.
    p_value : float, optional
        The p-value cutoff for determining significance of hotspots and coldspots. Default is 0.01.
    seed : int, optional
        Random seed for ensuring reproducibility of the analysis. Default is 42.
    plot_weights : bool, optional
        If set to True, visualizes the spatial weights matrix to aid in understanding spatial relationships. Default is False.
    return_stats : bool, optional
        If True, returns the local indicators of spatial association (LISA) along with identified hotspots and coldspots. Default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        A boolean array indicating locations identified as hotspots (high value surrounded by high values).
    :class:`numpy.ndarray`
        A boolean array indicating locations identified as coldspots (low value surrounded by low values).
    """
    
    # Get dimensions
    n, m = grid.shape

    # Get spatial stats function
    SPATIAL_STATS_FUNCTIONS = {
        'MoranI': esda.Moran_Local,
        'GearyC': esda.Geary_Local,
        'Getis-OrdG':esda.G_Local,
        'Getis-OrdG*':esda.G_Local
    }
    
    if mode not in SPATIAL_STATS_FUNCTIONS:
        raise ValueError(f"Unknown metric '{mode}'. Available metrics are: {list(SPATIAL_STATS_FUNCTIONS.keys())}")
    else:
        stats_function = SPATIAL_STATS_FUNCTIONS[mode]
    
    # Create a spatial weights matrix
    if not tissue_only:
        w = weights.lat2W(n, m, rook=False)
        if mode[-1] == 'I':
            # lisa = esda.Moran_Local(grid.flatten(), w, transformation='r', permutations=999, seed=seed)
            lisa = stats_function(grid.flatten(), w, transformation='r', permutations=999, seed=seed)
            print(f"Using {mode}")
        elif mode[-1] == 'C':
            lisa = stats_function(w, labels=True, permutations=999, seed=seed).fit(grid.flatten())
            print(f"Using {mode}")
        elif mode[-1] == 'G':
            # For local G, non-binary weights are allowed
            lisa = stats_function(grid.flatten(), w, transform='r', permutations=999, seed=seed)
            print(f"Using {mode}")
        elif mode[-2:] == 'G*':
            lisa = stats_function(grid.flatten(), w, transform='b', permutations=999, star=True, seed=seed)
            print(f"Using {mode}")
    else:
        masked_grid_nan = np.where(grid == -1, np.nan, grid)
        masked_grid = masked_grid_nan[~np.isnan(masked_grid_nan)]
        print(masked_grid.shape)
        w = create_masked_lat2W(grid)
        print(f"Local spatial stats restricted to tissue region with w of shape{w.full()[0].shape}")
        if mode[-1] == 'I':
            # lisa = esda.Moran_Local(masked_grid.flatten(), w, transformation='o', permutations=999, seed=seed)
            lisa = stats_function(masked_grid.flatten(), w, transformation='o', permutations=999, seed=seed)
            print(f"Using {mode}")
        elif mode[-1] == 'C':
            lisa = stats_function(w, labels=True ,permutations=999, seed=seed).fit(masked_grid.flatten())
            print(f"Using {mode}")
        elif mode[-1] == 'G':
            # For local G, non-binary weights are allowed
            lisa = stats_function(masked_grid.flatten(), w, permutations=999, seed=seed)
            print(f"Using {mode}")
        elif mode[-2:] == 'G*':
            lisa = stats_function(masked_grid.flatten(), w, permutations=999, star=True, seed=seed)
            print(f"Using {mode}")
            
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
    if mode[-1] == 'I':
        hotspots.flat[significant * (lisa.q==1)] = True
        coldspots.flat[significant * (lisa.q==3)] = True
    elif mode[-1] == 'C':
        hotspots.flat[significant * (lisa.labs==1)] = True
        coldspots.flat[significant * (lisa.labs==3)] = True
    elif mode[-1] == 'G':
        hotspots.flat[significant * (lisa.Zs>0)] = True
        coldspots.flat[significant * (lisa.Zs<0)] = True
    elif mode[-2:] == 'G*':
        hotspots.flat[significant * (lisa.Zs>0)] = True
        coldspots.flat[significant * (lisa.Zs<0)] = True
        
    if tissue_only:
        hotspots = map_spots_back_to_grid(hotspots.flatten(), grid, masked_grid)
        coldspots = map_spots_back_to_grid(coldspots.flatten(), grid, masked_grid)

    if return_stats:
        return hotspots, coldspots, lisa
    else:
        return hotspots, coldspots

def compute_proximity_index(arr, rook=True):
    """
    Compute the Proximity Index for the sample.
    """
    labelled, num_islands = _label_islands(arr, rook=rook)
    print(f"{num_islands} islands identified", flush=True)
    areas = _compute_areas(labelled)
    distances = _nearest_neighbour_distance(labelled, num_islands)
    
    # Calculate the proximity index using the given formula
    proximity_index = sum([areas[i] / distances[i] for i in range(num_islands)])
    
    return proximity_index

def generate_patches(
    spatial_data: Union[ad.AnnData, pd.DataFrame],
    library_key: str,
    library_id: str,
    scaling_factor: Union[int, float],
    spatial_key: Union[str, List[str]]
) -> list:
    """
    Generate a list of patches from a spatial data object, scaling them according to a given factor.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data from which to generate patches. This can be either an AnnData object or a DataFrame.
    library_key : str
        Key to identify the specific library within the `spatial_data`.
    library_id : str
        Identifier for the specific library, used to segregate or identify the data subset.
    scaling_factor : Union[int, float]
        Factor by which the spatial dimensions are scaled to determine the size of each patch.
    spatial_key : Union[str, List[str]]
        Key(s) to access specific spatial information within `spatial_data`.

    Returns
    -------
    :class:`list`
        A list of spatially defined patches derived from the `spatial_data`.
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

    if scaling_factor == 0:
        raise ValueError("scaling factor cannot be zero")
        
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

def generate_patches_randomly(
    spatial_data: Union[ad.AnnData, pd.DataFrame],
    library_key: str,
    library_id: str,
    scaling_factor: Union[int, float],
    spatial_key: Union[str, List[str]],
    max_overlap: float = 0.0,
    random_seed: Optional[int] = None,
    min_points: int = 2
) -> list:
    """
    Generate a list of patches from a spatial data object in a random manner, ensuring specified constraints on overlap and point count.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data from which to generate patches. This can be either an AnnData object or a DataFrame.
    library_key : str
        Key to identify the specific library within the `spatial_data`.
    library_id : str
        Identifier for the specific library, used to segregate or identify the data subset.
    scaling_factor : Union[int, float]
        Factor by which the spatial dimensions are scaled to determine the size of each patch.
    spatial_key : Union[str, List[str]]
        Key(s) to access specific spatial information within `spatial_data`.
    max_overlap : float, optional
        The maximum allowable overlap ratio between any two patches. Default is 0.0.
    random_seed : Optional[int]
        The seed for the random number generator, if reproducibility is desired. Default is None.
    min_points : int
        The minimum number of points a patch must contain to be considered valid. Default is 2.

    Returns
    -------
    :class:`list`
        A list of spatially defined patches, generated randomly based on the specified parameters.
    """

    rng = np.random.default_rng(random_seed)

    # Filter the spatial data for the given library_id
    if isinstance(spatial_data, ad.AnnData):
        spatial_data_filtered = spatial_data[spatial_data.obs[library_key] == library_id]
        spatial_values = spatial_data_filtered.obsm[spatial_key]
    elif isinstance(spatial_data, pd.DataFrame):
        spatial_data_filtered = spatial_data[spatial_data[library_key] == library_id]
        spatial_values = spatial_data_filtered[spatial_key].values
    else:
        raise ValueError("spatial_data should be either an AnnData object or a pandas DataFrame")
        
    if scaling_factor == 0:
        raise ValueError("scaling factor cannot be zero")

    x_min, y_min = spatial_values.min(axis=0)
    x_max, y_max = spatial_values.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    patch_width = width / scaling_factor
    patch_height = height / scaling_factor

    num_patches = int(scaling_factor ** 2)

    patches = []
    used_coordinates = set()
    
    for _ in range(num_patches):
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:  # seconds
                print(f"Warning: Could not generate a new patch within 5 seconds. Returning {len(patches)} out of {num_patches} patches")
                return patches

            # Calculate high and low values for x and y
            low_x = x_min
            high_x = x_max - patch_width
            low_y = y_min
            high_y = y_max - patch_height

            # Adjust x0 and y0 if high <= low
            if high_x <= low_x:
                x0 = low_x
            else:
                x0 = rng.uniform(low_x, high_x)
            if high_y <= low_y:
                y0 = low_y
            else:
                y0 = rng.uniform(low_y, high_y)

            x1 = x0 + patch_width
            y1 = y0 + patch_height
            new_patch = (x0, y0, x1, y1)

            if ((x0, y0) not in used_coordinates and
                _overlap_check(new_patch, patches, max_overlap) and
                _contains_points(new_patch, spatial_values, min_points)):
                used_coordinates.add((x0, y0))
                patches.append(new_patch)
                break

    return patches

def display_patches(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                    library_key: str, 
                    library_id: List[str],
                    spatial_key: Union[str, List[str]], 
                    cluster_keys: List[str],
                    scale:Union[int,float],
                    random_patch=False,
                    **kwargs):
    
    # Count the total number of plots to make
    total_plots = len(library_id) * len(cluster_keys)
    num_rows = len(library_id)
    num_cols = len(cluster_keys)
    print(f" The number of plot is {total_plots}")
    
    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8*num_cols, 8*num_rows),squeeze=False)
    
    # Iterate over each library_id and cluster_key, making a plot for each
    for i, sample_id in enumerate(library_id):
        for j, cluster_key in enumerate(cluster_keys):
            if isinstance(spatial_data, ad.AnnData):
                spatial_data_region = spatial_data[spatial_data.obs[library_key] == sample_id]
            elif isinstance(spatial_data, pd.DataFrame):
                spatial_data_region = spatial_data[spatial_data[library_key] == sample_id]
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
                                                       sample_id,
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

            ax.set_title('Cluster: {} Region: {}'.format(cluster_key, sample_id))
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
    

def calculate_diversity_index(
    spatial_data: Union[ad.AnnData, pd.DataFrame],
    library_key: str,
    library_id: str,
    spatial_key: Union[str, List[str]],
    patches: list,
    cluster_key: str,
    metric: str = 'Shannon Diversity',
    return_comp: bool = False
) -> Union[pd.Series, object]:
    """
    Calculate the heterogeneity index for a set of patches using specified metrics.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data to be analyzed. This can be either an AnnData object or a DataFrame.
    library_key : str
        Key to access the library data within `spatial_data`.
    library_id : str
        Identifier for the library to analyze within `spatial_data`.
    spatial_key : Union[str, List[str]]
        Key(s) to access spatial coordinates or related data in `spatial_data`.
    patches : list
        List of spatial regions or patches to compute the diversity index for.
    cluster_key : str
        Key to access clustering data, defining groups within patches.
    metric : str, optional
        The diversity metric to use, e.g., 'Shannon Diversity'. Default is 'Shannon Diversity'.
    return_comp : bool, optional
        If `True`, returns a comprehensive result object containing additional details along with the heterogeneity indices. Default is `False`.

    Returns
    -------
    :class:`pd.Series`
        If `return_comp` is `False`, returns a :class:`pd.Series` with heterogeneity indices for each patch.
        If `return_comp` is `True`, returns an additional :class:`pd.Series` with cell compositions.
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
        

def calculate_MDI(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                  scales: Union[tuple, list], 
                  library_key: str,
                  library_id: Union[tuple, list],  
                  spatial_key: Union[str, List[str]],
                  cluster_key: str,
                  random_patch=False,
                  plotfigs=False, 
                  savefigs=False,
                  patch_kwargs={},  
                  other_kwargs={})-> pd.DataFrame:
    """
    Calculate the multiscale diversity index (MDI) for spatial data.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data to be used for calculating the diversity indices.
    scales : Union[tuple, list]
        The scales at which the diversity index is to be calculated.
    library_key : str
        The key to access the library data within `spatial_data`.
    library_id : Union[tuple, list]
        The identifiers of the libraries involved in the analysis.
    spatial_key : Union[str, List[str]]
        Key(s) used to access the spatial data from `spatial_data`.
    cluster_key : str
        The key to access the cluster data which categorizes the spatial entities.
    random_patch : bool, optional
        Specifies whether patches should be generated in a random manner. Default is False.
    plotfigs : bool, optional
        Whether to plot figures during the analysis. Default is False.
    savefigs : bool, optional
        Whether to save the generated figures to disk. Default is False.
    patch_kwargs : dict, optional
        Additional keyword arguments used for patch generation. Defaults to an empty dictionary.
    other_kwargs : dict, optional
        Other keyword arguments that may influence the analysis. Defaults to an empty dictionary.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame containing the diversity value at each scale and the overall multiscale diversity index (MDI).
    """
    
    # Prepare to store the results
    results = pd.DataFrame(index = scales, columns = library_id)
    slopes = {sample_id: [] for sample_id in library_id}
    
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
            
        for sample_id in library_id:
            print(f"Processing region: {sample_id} at scale {scale}", flush=True)
            
            # Generate the patch coordinates 
            if random_patch:
                patches = generate_patches_randomly(spatial_data, 
                                                    library_key,
                                                    sample_id,
                                                    scale,
                                                    spatial_key,
                                                    **patch_kwargs)

            else:
                patches = generate_patches(spatial_data, 
                                           library_key,
                                           sample_id,
                                           scale,
                                           spatial_key)
                
            # Calculate the diversity index
            indices = calculate_diversity_index(spatial_data=spatial_data, 
                                                library_key=library_key, 
                                                library_id=sample_id, 
                                                spatial_key=spatial_key,
                                                patches=patches,
                                                cluster_key=cluster_key,
                                                **other_kwargs)
                
            count = indices.sum()/len(indices)
            print(f"{sample_id} at scale {scale} has {indices.eq(0.0).sum()} patches with zero diveristy", flush=True)
            print(f"{sample_id} at scale {scale} diversity is {count}", flush=True)
            
            # Store the result
            results.loc[scale,sample_id] = count
            
    scales = np.log2(np.reciprocal(scales))
    if plotfigs:
        # Plot the results
        num_library_id = len(sample_id)
        num_cols = min(num_library_id, 2) 
        num_rows = math.ceil(num_library_id / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
        if num_library_id == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, sample_id in enumerate(library_id):
            counts = results[library_id].values.astype(np.float64)
            #counts = np.log2(counts)
            axes[i].scatter(scales, counts, marker='x', c='black', label=sample_id)

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
        for i, sample_id in enumerate(library_id):
            counts = results[sample_id].values.astype(np.float64)
            #counts = np.log2(counts)
            slope, intercept, r_value, p_value, std_err = stats.linregress(scales, counts)
            slopes[sample_id].append(slope)
            
    df_results = pd.concat([results.transpose(), pd.DataFrame(slopes).transpose()], axis=1)
    df_results = df_results.rename(columns={0.0: 'Slope'})
    return df_results

def calculate_GDI(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                  scale: float, 
                  library_key: str,
                  library_id: Union[tuple, list], 
                  spatial_key: Union[str, List[str]],
                  cluster_key: str,
                  hotspot: bool = True,
                  whole_tissue: bool = False,
                  p_value: float = 0.01,
                  restricted: bool = False,
                  mode: str = 'MoranI',
                  **kwargs)-> pd.DataFrame:
    """
    Calculate a Global Diversity Index (GDI) for specified samples within spatial data, incorporating spatial statistics under chosen analysis modes.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data containing library and clustering information for analysis.
    scale : float
        The scaling factor to adjust spatial coordinates for analysis.
    library_key : str
        Key associated with the library information in `spatial_data`.
    library_id : Union[tuple, list]
        Identifiers for the libraries to be analyzed.
    spatial_key : Union[str, List[str]]
        Key(s) identifying the spatial coordinates within `spatial_data`.
    cluster_key : str
        Key used to access cluster information within `spatial_data`.
    hotspot : bool, optional
        Determines whether to analyze spatial hotspots or coldspots. Default is True.
    whole_tissue : bool, optional
        Specifies whether to analyze the entire tissue or specific regions. Default is False.
    p_value : float, optional
        The p-value threshold for determining statistical significance in spatial analysis. Default is 0.01.
    restricted : bool, optional
        Restricts the analysis to specified conditions or tissue types. Default is False.
    mode : str, optional
        The mode of spatial statistics used in the analysis, such as 'MoranI'. Default is 'MoranI'.
    **kwargs
        Additional keyword arguments for customization and specific parameters in underlying functions.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with indices representing library identifiers and a single column 'GDI' containing the calculated Global Diversity Index for each sample.
    """

    global_stats = {sample_id: [] for sample_id in library_id}
    
    for sample_id in library_id:
        print(f"Processing region: {sample_id} at scale {scale}", flush=True)

        # Generate the patch coordinates 
        patches = generate_patches(spatial_data, 
                                   library_key,
                                   sample_id,
                                   scale,
                                   spatial_key)

        # Calculate the heterogeneity index
        indices, patches_comp = calculate_diversity_index(spatial_data=spatial_data, 
                                                          library_key=library_key, 
                                                          library_id=sample_id, 
                                                          spatial_key=spatial_key,
                                                          patches=patches,
                                                          cluster_key=cluster_key,
                                                          return_comp=True,
                                                          **kwargs)
        
        grid = diversity_heatmap(spatial_data=spatial_data,
                                 library_key=library_key,
                                 library_id=sample_id,
                                 spatial_key=spatial_key,
                                 patches=patches, 
                                 heterogeneity_indices=indices,
                                 tissue_only=restricted,
                                 plot=False)
        
        stats, pvals = global_spatial_stats(grid, tissue_only=restricted, mode=mode)
        global_stats[sample_id].append(stats)
        
    return pd.DataFrame(global_stats, index=['GDI']).T

def calculate_DPI(
    spatial_data: Union[ad.AnnData, pd.DataFrame],
    scale: float,
    library_key: str,
    library_id: Union[tuple, list],
    spatial_key: Union[str, List[str]],
    cluster_key: str,
    hotspot: bool = True,
    p_value: float = 0.01,
    mode: str = 'MoranI',
    restricted: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate the Diversity Proximity Index (DPI) for specified samples within spatial data.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data to be analyzed. This can be an AnnData object or a pandas DataFrame.
    scale : float
        The scale factor used for generating patches within the spatial regions.
    library_key : str
        Key in `spatial_data` that corresponds to library identifiers.
    library_id : Union[tuple, list]
        A tuple or list of identifiers for the libraries to be processed.
    spatial_key : Union[str, List[str]]
        Key(s) in `spatial_data` used to determine spatial coordinates.
    cluster_key : str
        Key used to identify different clusters or types within `spatial_data`.
    hotspot : bool, optional
        Specifies whether to identify diversity hotspots (True) or coldspots (False). Default is True.
    p_value : float, optional
        The significance level used for hotspot or coldspot identification. Default is 0.01.
    mode : str, optional
        Specifies the mode of spatial statistics to be used, such as 'MoranI'. Default is 'MoranI'.
    restricted : bool, optional
        If set to True, restricts analysis to specific tissue regions. Default is False.
    **kwargs
        Additional keyword arguments for further customization of the diversity calculations.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame where each index represents a library_id and the columns contain the calculated 
        DPI for each sample.
    """
    
    PX = {sample_id: [] for sample_id in library_id}
    
    for sample_id in library_id:
        print(f"Processing region: {sample_id} at scale {scale}", flush=True)

        # Generate the patch coordinates 
        patches = generate_patches(spatial_data, 
                                   library_key,
                                   sample_id,
                                   scale,
                                   spatial_key)

        # Calculate the heterogeneity index
        indices, patches_comp = calculate_diversity_index(spatial_data=spatial_data, 
                                                          library_key=library_key, 
                                                          library_id=sample_id, 
                                                          spatial_key=spatial_key,
                                                          patches=patches,
                                                          cluster_key=cluster_key,
                                                          return_comp = True,
                                                          **kwargs)
        
        grid = diversity_heatmap(spatial_data=spatial_data,
                                 library_key=library_key,
                                 library_id=sample_id,
                                 spatial_key=spatial_key,
                                 patches=patches, 
                                 heterogeneity_indices=indices,
                                 plot=False)
        
        hotspots, coldspots = local_spatial_stats(grid, tissue_only=restricted, mode=mode, p_value=p_value)
        
        if hotspot:
            print(f"Region {sample_id} contains {sum(hotspots.flatten())} diversity hotspots", flush=True)
            px = compute_proximity_index(hotspots)
        else:
            print(f"Region {sample_id} contains {sum(coldspots.flatten())} diversity coldspots", flush=True)
            px = compute_proximity_index(coldspots)
            
        PX[sample_id].append(px)
        
    # Return the dictionary containing Proximity Index
    return pd.DataFrame(PX, index=['DPI']).T

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
                      return_fig=False) -> Union[np.ndarray, Optional[plt.Figure]]:
    """
    Visualize the diversity indices as a heatmap on the original spatial data, optionally returning the plot figure for further customization.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data to be used for visualization. This can be either an AnnData object or a DataFrame.
    library_key : str
        The key associated with the library in `spatial_data`, used to access library-specific data.
    library_id : str
        The identifier for the library to be used in the analysis.
    spatial_key : Union[str, List[str]]
        Key(s) identifying the spatial information within `spatial_data`.
    patches : list
        The list of patches to be analyzed. Each patch corresponds to a specific region in the spatial data.
    heterogeneity_indices : :class:`pandas.Series`
        The heterogeneity indices to be visualized. Each index value corresponds to a patch, indicating its heterogeneity level.
    tissue_only : bool, optional
        If True, only tissue regions are included in the analysis. Default is False.
    plot : bool, optional
        If True, a heatmap is plotted to visualize the indices. Default is True.
    return_fig : bool, optional
        If True, the :class:`matplotlib.figure.Figure` is returned for further customization. Default is False.

    Returns
    -------
    :class:`numpy.ndarray`
        A grid where each cell represents the diversity index for a corresponding patch.
    :class:`matplotlib.figure.Figure`, optional
        The matplotlib figure object, returned if both `return_fig` is True and `plot` is True.
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

def spot_cellfreq(spatial_data: Union[ad.AnnData, pd.DataFrame], 
                  scale: float, 
                  library_key: str,
                  library_id: Union[tuple, list], 
                  spatial_key: Union[str, List[str]],
                  cluster_key: str,
                  spots: str = 'hot',
                  p_value: float = 0.01,
                  combination: int = 2,
                  top: int = 15,
                  selected_comb: Optional[list] = None,
                  mode: str = 'MoranI',
                  restricted: bool = False,
                  **kwargs)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze cell frequency and co-occurrence across different spots in spatial data.

    Parameters
    ----------
    spatial_data : Union[:class:`ad.AnnData`, :class:`pd.DataFrame`]
        The spatial data containing library information and spatial coordinates for analysis.
    scale : float
        The scaling factor for adjusting the spatial coordinates in the analysis.
    library_key : str
        Key associated with the library information in `spatial_data`.
    library_id : Union[tuple, list]
        Identifiers for the libraries to be used in the analysis.
    spatial_key : Union[str, List[str]]
        Key(s) identifying the spatial coordinates within `spatial_data`.
    cluster_key : str
        Key used to access cluster information within `spatial_data`.
    spots : str, optional
        Specifies the type of spots to analyze ('hot', 'cold', or 'global'). Default is 'hot'.
    p_value : float, optional
        The p-value threshold for significance in spatial statistics testing. Default is 0.01.
    combination : int, optional
        The number of top combinations to consider for analyzing frequency and co-occurrence. Default is 2.
    top : int, optional
        The number of top results to return for each combination. Default is 15.
    selected_comb : Optional[list], optional
        Specific combinations of clusters to analyze; if None, the top combinations are used. Default is None.
    mode : str, optional
        The mode of spatial statistics to apply (e.g., 'MoranI'). Default is 'MoranI'.
    restricted : bool, optional
        If set to True, restricts the analysis to specified conditions or regions. Default is False.
    **kwargs
        Additional keyword arguments for further customization and specific parameters in underlying functions.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame containing normalized cell frequencies for each cluster across specified regions, or across the entire tissue if 'global' is selected.
    :class:`pandas.DataFrame`
        A transposed DataFrame showing the frequency of specific cluster combinations in each region, sorted by the top specified combinations if `selected_comb` is None.
    """
    
    if spots=='global':
        spatial_df = spatial_data.obs[[library_key, cluster_key]]
        global_cell_count = spatial_df.groupby([library_key, cluster_key], observed=False).size().unstack(fill_value=0)
        cellfreq_df = global_cell_count.div(global_cell_count.sum(axis=1), axis=0)
    else:
        cellfreq_df = pd.DataFrame() 
        
    co_occurrence_df = pd.DataFrame()
    
    for library_id in library_id:
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
        
        hotspots, coldspots= local_spatial_stats(grid, tissue_only=restricted, mode=mode, p_value = p_value)
        
        if spots == 'hot':
            print(f'Region {library_id} contains {sum(hotspots.flatten())} diversity hotspots', flush=True)
            filtered_patches_comp = [patch for patch, is_hotspot in zip(patches_comp, hotspots.flatten()) if is_hotspot]
        elif spots == 'cold':
            print(f'Region {library_id} contains {sum(coldspots.flatten())} diversity coldspots', flush=True)
            filtered_patches_comp = [patch for patch, is_coldspot in zip(patches_comp, coldspots.flatten()) if is_coldspot]
        elif spots == 'global':
            print('Considering whole tissue')
            filtered_patches_comp = patches_comp 
        else:
            print(f'Your chosen {mode} is not supported; Please choose one from hot, cold and global')
        
        if filtered_patches_comp and not all(item is None for item in filtered_patches_comp):
            merged_series = pd.concat(filtered_patches_comp, axis=1).sum(axis=1)
            if mode != 'global':
                cellfreq_df = _append_series_to_df(cellfreq_df, (merged_series/merged_series.sum(axis=0)), library_id)
                
            comb_freq = combination_freq(filtered_patches_comp, n=combination, top=top, cell_type_combinations=selected_comb)
            co_occurrence_df = _append_series_to_df(co_occurrence_df, comb_freq, library_id)
        else:
            print(f"Region {library_id} has no diversity hot/cold spot since length of filterd_patch_comp is either {len(filtered_patches_comp)} or hot/cold spots contain no cells")
    
    return cellfreq_df, co_occurrence_df.T # sample id as columns (row index)
    
def signif_heatmap(spatial_data:Union[ad.AnnData,pd.DataFrame], 
                   library_key:str,
                   library_id:str,
                   spatial_key:str, 
                   patches,
                   heterogeneity_indices, 
                   tissue_only=False,
                   plot=True,
                   discrete=False,
                   return_fig=False):

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

        if not discrete:
            # Plot the heatmap
            plt.imshow(grid, cmap='Greens_r', interpolation='none')
            cb = plt.colorbar(label='P-Value', format="{x:.2f}")
            cb.ax.minorticks_on()
            
            # Define minor tick positions and labels
            minor_locator = AutoMinorLocator(5)  # This would put 1 minor tick between major ticks
            cb.ax.yaxis.set_minor_locator(minor_locator)
            
            # Define a function to format minor tick labels
            def minor_tick_format(x, pos):
                return f"{x:.2f}"  # Adjust format as needed
            
            # Apply formatter for the minor ticks
            cb.ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_format))
        else:
            # Define the categories
            categories = [0.1, 0.05, 0.01, 0.005]
            
            # Select colors from the Greens_r colormap
            greens = colormaps.get_cmap('Greens')
            colors = [greens(i / len(categories)) for i in range(len(categories) + 1)]
            cmap = ListedColormap(colors)  # Create a custom ListedColormap
    
            # Assign colors to the grid based on the categories
            grid_colored = np.digitize(grid, bins=categories, right=False)
            
            # Plot the heatmap
            plt.imshow(grid_colored, cmap=cmap, interpolation='none')
            
            # Create a legend with custom patches
            labels = ['>0.1', '<0.1', '<0.05', '<0.01', '<0.005']
            patches = [matpatches.Patch(facecolor=cmap(i), label=label, edgecolor='k') for i, label in enumerate(labels)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
        if return_fig and plot:
            fig = plt.gcf()
            return fig