# MESA: Multiomics and Ecological Spatial Analysis for Quantitative Decoding of Tissue Disease States

<p align="center">
  <img src="https://raw.githubusercontent.com/Feanor007/MESA/refs/heads/main/docs/_static/images/figure1_new.png" width="75%" height="75%">
</p>

MESA is a novel pipeline that brings ecological principles together with multiomics data integration thereby enabling deeper and more quantitative decoding of the functional and spatial shifts of tissue remodeling across disease states. 
- Drawing inspiration from ecological studies, MESA adapts diversity metrics traditionally used to gauge biodiversity for spatial omics data, creating tools for systematic quantification of cellular diversity. Specifically, we introduce a multi-scale diversity index, alongside global and local diversity indices, to capture not only the overarching diversity of a tissue but also the localized patterns and dependencies.
- Furthermore, MESA employs a multi-omics approach to spatial omics analyses. MESA in silico amalgamates cross-modality single-cell data to enrich the context of spatial omics observations. With the additional layers of information brought to bear by multiomics, MESA facilitates an extended view of cellular neighborhoods and their spatial interactions within tissue microenvironments. MESA's approach, incorporating differential expression, gene set enrichment, and ligand-receptor interaction analyses within these spatially defined cellular assemblies, further enhances a mechanistic understanding of tissue remodeling across disease states.

## Installation

MESA is hosted on `pypi` and can be installed via `pip`. Note this package requires `python >= 3.10`.

```
pip install mesa-py
```
Visit our [documentation](https://mesa-py.readthedocs.io/en/latest/) to see examples and tutorials!

## Citation

If you use MESA in your research, please cite:

> Ding, D.Y.\*, Tang, Z.\*, Zhu, B.\*, Ren, H., Shalek, A.K., Tibshirani, R., and Nolan, G.P. (2025).
> Quantitative characterization of tissue states using multiomics and ecological spatial analysis.
> *Nature Genetics* **57**, 910–921. https://doi.org/10.1038/s41588-025-02119-z
>
> \*These authors contributed equally.

<details>
<summary>BibTeX</summary>

```bibtex
@article{ding2025mesa,
  title   = {Quantitative characterization of tissue states using multiomics and ecological spatial analysis},
  author  = {Ding, Daisy Yi and Tang, Zeyu and Zhu, Bokai and Ren, Hongyu and
             Shalek, Alex K. and Tibshirani, Robert and Nolan, Garry P.},
  journal = {Nature Genetics},
  volume  = {57},
  number  = {4},
  pages   = {910--921},
  year    = {2025},
  doi     = {10.1038/s41588-025-02119-z}
}
```
</details>

## License
```MESA``` is under the [Academic Software License Agreement](https://github.com/Feanor007/MESA/blob/main/LICENSE), please use accordingly.
