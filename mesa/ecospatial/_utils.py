import numpy as np
import pandas as pd
from scipy import ndimage, spatial

def _append_series_to_df(df: pd.DataFrame, series: pd.Series, column_name: str):
    """
    Appends a Series to a DataFrame as a new column while ensuring that no data 
    is lost from the DataFrame and accommodating new indices from the Series.
    
    Parameters:
    - df: The DataFrame to which the Series should be appended.
    - series: The Series to append.
    - column_name: The name of the new column.
    
    Returns:
    - A DataFrame with the Series appended as a new column.
    """

    # Combine the indices of the DataFrame and the Series
    combined_index = df.index.union(series.index)
    
    # Reindex the DataFrame based on the combined index
    df = df.reindex(combined_index, fill_value=0)
    
    # Add the series as a new column to the DataFrame
    df[column_name] = series

    return df

def aggregate_spot_compositions(labelled, compositions):
    """
    Aggregate compositions of spots that belong to the same island.

    Parameters:
    - spots: Boolean array indicating whether each patch is a spot.
    - labelled: Array of the same shape as 'spots', where each component (island) has a unique label.
    - compositions: List of pd.Series, each representing the composition of a spot, or None for spots without compositions.

    Returns:
    - pd.DataFrame: DataFrame where each row corresponds to an island's aggregated composition.
    """
    # Initialize a DataFrame to store the aggregated data
    aggregated_df = pd.DataFrame()

    # Iterate over each unique label (island)
    for label in range(1, np.max(labelled) + 1):
        # Find indices where the current label is present
        indices = (labelled == label) # & spots
        
        # Sum compositions for the current island
        island_composition = pd.Series(dtype='float')
        for index in np.nonzero(indices.flatten())[0]:
            current_composition = compositions[index]
            if current_composition is None:
                current_composition = pd.Series(0, index=island_composition.index)
            island_composition = island_composition.add(current_composition, fill_value=0)
        
        # Append the summed Series to the DataFrame
        aggregated_df[f'Island_{label}'] = island_composition

    return aggregated_df.T

def _overlap_check(new_patch, existing_patches, max_overlap_ratio):
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

def _contains_points(patch, spatial_values, min_points):
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
    
def _find_coordinates(array, value):
    return np.argwhere(array == value)

def _calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def _label_islands(arr, rook=True):
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

def _compute_areas(labeled):
    """
    Compute the area of each labeled island.
    """
    return ndimage.sum(labeled > 0, labeled, range(1, labeled.max() + 1))

def _nearest_neighbour_distance(labeled, num_features):
    """
    Compute the nearest neighbour distance for each island.
    """
    
    island_coords = {}
    for i in range(1, num_features+1):
        island_coords[i] = _find_coordinates(labeled, i)
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