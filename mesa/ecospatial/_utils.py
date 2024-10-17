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