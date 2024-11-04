API
===
Import MESA as::

    import mesa as ms

Multiomics
----------

.. module:: mesa.multiomics.multiomics_spatial
.. currentmodule:: mesa

.. autosummary::
    :toctree: api

    multiomics.multiomics_spatial.get_spatial_knn_indices
    multiomics.multiomics_spatial.get_neighborhood_composition
    multiomics.multiomics_spatial.get_avg_expression_neighbors

Ecospatial
----------

.. module:: mesa.ecospatial
.. currentmodule:: mesa

.. autosummary::
    :toctree: api

    ecospatial.generate_patches
    ecospatial.calculate_shannon_entropy
    ecospatial.calculate_diversity_index
    ecospatial.diversity_heatmap
    ecospatial.global_spatial_stats
    ecospatial.local_spatial_stats
    ecospatial.calculate_MDI
    ecospatial.calculate_GDI
    ecospatial.calculate_DPI
    ecospatial.spot_cellfreq
    