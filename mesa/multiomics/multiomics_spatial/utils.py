# Standard library imports
import random
import warnings
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.extmath import randomized_svd
from scipy import stats
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)


def center_scale(arr):
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)

def drop_zero_variability_columns(arr, tol=1e-8):
    """
    Drop columns for which its standard deviation is zero in any one of the arrays in arr_list.
    Parameters
    ----------
    arr: np.ndarray of shape (n_samples, n_features)
        Data matrix
    tol: float, default=1e-8
        Any number less than tol is considered as zero
    Returns
    -------
    np.ndarray where no column has zero standard deviation
    """
    
    bad_columns = set()
    curr_std = np.std(arr, axis=0)
    for col in np.nonzero(np.abs(curr_std) < tol)[0]:
        bad_columns.add(col)
    good_columns = [i for i in range(arr.shape[1]) if i not in bad_columns]
    return arr[:, good_columns]

def get_neighborhood_composition(knn_indices, labels, all_labels, percentage=True)-> np.ndarray:
    """
    Compute the global composition of neighbors for each sample, either in percentage or count form, based on k-nearest neighbors (k-NN) indices and specific cluster labels.

    Parameters
    ----------
    knn_indices : :class:`numpy.ndarray` of shape (n_samples, n_neighbors)
        An array where each row represents the k-nearest neighbors' indices for that sample, indicating the nearest neighbors.
    labels : :class:`numpy.ndarray` of shape (n_samples,)
        Cluster labels for each sample that appear in this particular region, used to determine neighborhood composition.
    all_labels : :class:`numpy.ndarray`
        All unique cluster labels across the entire dataset, used for reference in composition calculations.
    percentage : bool, optional
        Specifies whether to return the composition as a percentage of the total or as a raw count. Default is True.

    Returns
    -------
    :class:`numpy.ndarray` of shape (n_samples, len(all_labels))
        An array where each row represents the composition of neighbors for each sample, either as a percentage or a count, based on the cluster labels.
    """
    
    labels = list(labels)
    n, k = knn_indices.shape
    n_clusters = len(all_labels)
    label_to_clust_idx = {label: i for i, label in enumerate(all_labels)}
    comp = np.zeros((n, n_clusters))
    for i, neighbors in enumerate(knn_indices):
        good_neighbors = [nb for nb in neighbors if nb != -1]
        for nb in good_neighbors:
            comp[i, label_to_clust_idx[labels[nb]]] += 1
    if percentage:
        return (comp.T / comp.sum(axis=1)).T
    else:
        return comp # return the cell count instead of percentage


def robust_svd(arr, n_components, randomized=False, n_runs=1):
    """
    Do deterministic or randomized SVD on arr.
    Parameters
    ----------
    arr: np.array
        The array to do SVD on
    n_components: int
        Number of SVD components
    randomized: bool, default=False
        Whether to run randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    u, s, vh: np.array
        u @ np.diag(s) @ vh is the reconstruction of the original arr
    """
    if randomized:
        best_err = float('inf')
        u, s, vh = None, None, None
        for _ in range(n_runs):
            curr_u, curr_s, curr_vh = randomized_svd(arr, n_components=n_components, random_state=None)
            curr_err = np.sum((arr - curr_u @ np.diag(curr_s) @ curr_vh) ** 2)
            if curr_err < best_err:
                best_err = curr_err
                u, s, vh = curr_u, curr_s, curr_vh
        assert u is not None and s is not None and vh is not None
    else:
        if n_runs > 1:
            warnings.warn("Doing deterministic SVD, n_runs reset to one.")
        u, s, vh = svds(arr*1.0, k=n_components) # svds can not handle integer values
    return u, s, vh


def svd_denoise(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute best rank-n_components approximation of arr by SVD.
    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    arr: array_like of shape (n_samples, n_features)
        Rank-n_comopnents approximation of the input arr.
    """
    if n_components is None:
        return arr
    u, s, vh = robust_svd(arr, n_components=n_components, randomized=randomized, n_runs=n_runs)
    return u @ np.diag(s) @ vh


def svd_embedding(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute rank-n_components SVD embeddings of arr.
    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    embeddings: array_like of shape (n_samples, n_components)
        Rank-n_comopnents SVD embedding of arr.
    """
    if n_components is None:
        return arr
    u, s, vh = robust_svd(arr, n_components=n_components, randomized=randomized, n_runs=n_runs)
    return u @ np.diag(s)

def get_pca_expression_neighbors(exp_df: pd.DataFrame, spatial_knn_indices: np.ndarray, n_components: int=58):
    """
    Calculate the PCA-reduced expression values over the neighbors of each data point.
    Parameters
    ----------
    exp_df : pd.DataFrame
        The DataFrame to process. Each row represents a data point (e.g., a cell in single-cell data), 
        and each column represents an attribute of the data points (e.g., gene expression values).
    spatial_knn_indices : np.ndarray
        The indices that represent the k-nearest neighbors for each data point in a spatial data structure.
    n_components: int
        The number of principal components to keep.
    Returns
    -------
    np.ndarray
        The resulting np.ndarray represents the PCA-reduced expression values over the neighbors 
        of a data point.
    """
    knn_indices_df = pd.DataFrame(spatial_knn_indices)
    concat_exp_all = []
    for neighbors in tqdm(knn_indices_df.values, total=knn_indices_df.shape[0], desc='Computing PCA expression'):
        concat_df = exp_df.iloc[neighbors].values.flatten()
        #print(len(concat_df))
        concat_exp_all.append(concat_df)
        #print(len(concat_exp_all))
    
    # Convert list to np array
    concat_exp_all = np.array(concat_exp_all)
    #print(concat_exp_all.shape)
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_exp_all = pca.fit_transform(concat_exp_all)

    return pca_exp_all


def get_avg_expression_neighbors(exp_df: pd.DataFrame, spatial_knn_indices: np.ndarray):
    """
    Calculate the average expression values over the neighbors of each data point.
    
    Parameters
    ----------
    exp_df : pd.DataFrame
        The DataFrame to process. Each row represents a data point (e.g., a cell in single-cell data), 
        and each column represents an attribute of the data points (e.g., gene expression values).
    spatial_knn_indices : np.ndarray
        The indices that represent the k-nearest neighbors for each data point in a spatial data structure.
    Returns
    -------
    np.ndarray
        The resulting np.ndarray represents the average expression values over the neighbors 
        of a data point.
    """
    
    knn_indices_df = pd.DataFrame(spatial_knn_indices)
    avg_exp_all = []
    for neighbors in tqdm(knn_indices_df.values, total=knn_indices_df.shape[0], desc='Computing avg expression'):
        mean = exp_df.iloc[neighbors].mean(axis=0)
        avg_exp_all.append(mean)
    return avg_exp_all

def visualise_nbhd_freq(dataframe: pd.DataFrame, nbhd_key: str):
    """
    Visualize the frequency of different neighborhoods.
    
    Also performs a t-test between each pair of groups for every unique neighborhood, and adds the p-values to the plot.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to process. It must contain columns 'patients', 'groups' and the neighborhood key. (currently only tested on CRC dataset)
    nbhd_key : str
        The key to the neighborhoods (column in the dataframe) whose frequency to calculate and visualize.
        
    Returns
    ----------
    matplotlib.figure.Figure
        The figure object containing the generated plot. This can be used for further editing or saving the figure to a file.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('dataframe must be a pandas DataFrame')
    if not isinstance(nbhd_key, str):
        raise TypeError('nbhd_key must be a string')
    fc = dataframe.groupby(['patients', 'groups'])[nbhd_key].value_counts(normalize=True).unstack(fill_value=0)
    
    # Reset the index of the DataFrame
    fc_reset = fc.reset_index()

    # Melt the DataFrame
    melt = fc_reset.melt(id_vars=['patients', 'groups'], var_name='neighborhood', value_name='frequency of neighborhood')
    
    
    # Create a plot
    palette = dict(zip(melt['groups'].unique(), sns.color_palette("Set1", n_colors=melt['groups'].nunique())))
    fig, ax = plt.subplots(figsize=(12,5))
    # First plot
    sns.stripplot(data=melt, x='neighborhood', y='frequency of neighborhood', hue='groups', palette=palette, dodge=True, alpha=.3, ax=ax)

    # Second plot
    sns.set_palette("Set1")  # Reset the color palette
    sns.pointplot(data=melt, x='neighborhood', y='frequency of neighborhood', hue='groups', palette=palette, dodge=.5, join=False, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Groups", handletextpad=0, columnspacing=1, loc="upper left", ncol=3, frameon=True)

    p_values = []
    for neighborhood in melt['neighborhood'].unique():
        # Subset the dataframe for each neighborhood
        subset = melt[melt['neighborhood'] == neighborhood]

        # Separate this subset into two groups
        group1 = subset[subset['groups'] == 1]['frequency of neighborhood']
        group2 = subset[subset['groups'] == 2]['frequency of neighborhood']

        # Perform the t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit') # omit if there are any NaN values
        p_values.append(p_value)

    for i in range(len(p_values)):
        ax.text(i, max(melt['frequency of neighborhood']), f"p = {p_values[i]:.3f}", ha='center', fontsize=8, rotation=60)
    return fig

def neighborhood_extraction(protein: pd.DataFrame, 
                            ProteinRNA_df: pd.DataFrame, 
                            RNA:np.ndarray, 
                            nbhd_type: str, 
                            library_ids: list, 
                            n_neighbors=10):
    """
    This function clusters the given data based on various types of neighborhood compositions:
    cell type composition, protein expression, RNA expression or Protein-RNA fusion. 

    Parameters:
    ----------
        protein (pd.DataFrame): The input proteomics DataFrame which needs to be processed.
        ProteinRNA (pd.DataFrame): The input proteomics-transcriptomics fusion DataFrame which needs to be processed.
        RNA (np.ndarray): The input transcriptomics DataFrame which needs to be processed.
        nbhd_type (str): The type of neighborhood clustering method. Can be one of 'cell type composition', 'protein expression', 'RNA expression', 'Protein-RNA fusion'.
        library_ids (list): List of unique identifiers corresponding to the Regions in the input dataframe.
        n_neighbors (int): Number of nearest neighbors

    Returns:
    ----------
        tuple: A tuple where the first element is the neighborhoods and the second element is a numpy array of the original sorted indices in the dataframe.


    Note: 
        The function does not yet support 'Protein-RNA fusion' type and it will be implemented in future versions.
    """

    protein_features = list(protein.columns[13:62]) + list(protein.columns[70:79])
#     removal_list = ['CD57', 'CD71', 'CD194', 'CDX2', 'Collagen IV', 'MMP9', 'MMP12']
#     protein_features = [feature for feature in protein_features if not any(sub in feature for sub in removal_list)]
    
    all_neighbourhoods = []
    original_indices = []
    for each_region in tqdm(library_ids):
        protein_region = protein.loc[protein['File Name'] == each_region,:].copy()
        current_indices = protein_region.index.tolist()
        original_indices.extend(current_indices)
        locations = protein_region[['X:X', 'Y:Y']].values
        spatial_knn_indices = graph.get_spatial_knn_indices(locations=locations, n_neighbors=n_neighbors, method='kd_tree') 
        # Depending on the nbhd_type, execute different code blocks
        # This if-elif checking inside each for loop looks ugly :(
        if nbhd_type == 'cell type composition':
            feature_labels = protein_region['ClusterName'].values
            cell_nbhds = get_global_neighborhood_composition(knn_indices=spatial_knn_indices, labels=feature_labels, all_labels = protein['ClusterName'].unique())
            all_neighbourhoods.extend(cell_nbhds)
        elif nbhd_type == 'protein expression':
            protein_exp_region = protein_region[protein_features]
            avg_exp = get_avg_expression_neighbors(protein_exp_region, spatial_knn_indices)
            avg_exp_array = np.stack(avg_exp)
            all_neighbourhoods.extend(avg_exp_array)
        elif nbhd_type == 'RNA expression':
            rna_matched = []
            for i in tqdm(np.array(protein_region['CellID']), total=protein_region['CellID'].shape[0], desc='Assigning rna expression'):
                if i in ProteinRNA_df['protein'].values:
                    rna_idx = ProteinRNA_df[ProteinRNA_df['protein'] == i]['rna']
                    rna_exp_each = RNA[rna_idx][0]
                    rna_matched.append(rna_exp_each)
                else:
                    rna_matched.append(np.array([0]*RNA.shape[1]))

            rna_matched_protein_order = np.stack(rna_matched)
            rna_matched_df = pd.DataFrame(rna_matched_protein_order)
            # calculate the average rna expression based on rna_matched_protein_order for each cluster
            avg_exp_rna = get_avg_expression_neighbors(rna_matched_df, spatial_knn_indices)
            avg_exp_rna_array = np.stack(avg_exp_rna)
            all_neighbourhoods.extend(avg_exp_rna_array)
        elif nbhd_type == 'Protein-RNA fusion':
            pass # To be implemented
        else:
            raise ValueError("Invalid nbhd_type. Expected one of: ['cell type composition', 'protein expression', 'RNA expression', 'Protein-RNA fusion']")
            
    paired = list(zip(original_indices, all_neighbourhoods))
    paired.sort()
    all_neighbourhoods_sorted = np.array([nbhd for _, nbhd in paired])
    original_indices_sorted = np.array([idx for idx, _ in paired])

    
    return all_neighbourhoods_sorted,original_indices_sorted
