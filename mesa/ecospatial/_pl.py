import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as matpatches
from matplotlib.figure import figaspect

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
