library(FlowSOM)
library(ggplot2)

data_dir <- "../../../../../tonsil/"
protein <- read.csv(file.path(data_dir, 'tonsil_codex.csv'))

protein_features <- c('CD38', 'CD19', 'CD31', 'Vimentin', 'CD22', 'Ki67', 'CD8',
                      'CD90', 'CD123', 'CD15', 'CD3', 'CD152', 'CD21', 'cytokeratin', 
                      'CD2', 'CD66', 'collagen.IV', 'CD81', 'HLA.DR', 'CD57', 'CD4', 
                      'CD7', 'CD278', 'podoplanin', 'CD45RA', 'CD34', 'CD54', 'CD9', 
                      'IGM', 'CD117', 'CD56', 'CD279', 'CD45', 'CD49f', 'CD5', 
                      'CD16', 'CD63', 'CD11b', 'CD1c', 'CD40', 'CD274', 'CD27', 
                      'CD104', 'CD273', 'FAPalpha', 'Ecadherin')
protein_exp <- protein[, protein_features]

cell_nbhd = read.csv("cell_nbhd.csv", header=FALSE)
avg_exp_df = read.csv("avg_exp_df.csv", header=FALSE)
avg_exp_rna_df = read.csv("avg_exp_rna_df.csv", header=FALSE)

# Cellular composition
num_clusters <- 10
clustering_markers <- colnames(cell_nbhd)
fsom_clustering_data = cell_nbhd
flowFrame_cluster <- flowCore::flowFrame(as.matrix(cell_nbhd))
fSOM2 <- FlowSOM(flowFrame_cluster,
                 compensate = F, scale = F, 
                 colsToUse = clustering_markers, 
                 nClus = num_clusters, seed = 3)
cluster_cell_fsom = GetMetaclusters(fSOM2)
protein$cluster_composition_fsom <- cluster_cell_fsom

ggplot(protein, aes(x = centroid_x, y = centroid_y, color = cluster_composition_fsom)) +
  geom_point(size = 0.01) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    legend.title = element_blank(),
    legend.text = element_text(size = 12.5),
    legend.position = "bottomleft"
  ) +
  labs(x = "Centroid x", y = "Centroid y") +
  guides(color = guide_legend(override.aes = list(size = 3)))


# Protein expression
clustering_markers <- colnames(avg_exp_df)
fsom_clustering_data_protein = avg_exp_df
flowFrame_cluster_protein <- flowCore::flowFrame(as.matrix(avg_exp_df))
fSOM2_protein <- FlowSOM(flowFrame_cluster_protein,
                 compensate = F, scale = F, 
                 colsToUse = clustering_markers, 
                 nClus = num_clusters, seed = 3)
cluster_cell_fsom_protein = GetMetaclusters(fSOM2_protein)

protein$cluster_protein_fsom <- cluster_cell_fsom_protein
ggplot(protein, aes(x = centroid_x, y = centroid_y, color = cluster_protein_fsom)) +
  geom_point(size = 0.000000001) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    legend.title = element_blank(),
    legend.text = element_text(size = 12.5),
    legend.position = "bottomleft"
  ) +
  labs(x = "Centroid x", y = "Centroid y") +
  guides(color = guide_legend(override.aes = list(size = 3)))



# RNA expression
clustering_markers <- colnames(avg_exp_rna_df)
fsom_clustering_data_rna = avg_exp_rna_df
flowFrame_cluster_rna <- flowCore::flowFrame(as.matrix(avg_exp_rna_df))
fSOM2_rna <- FlowSOM(flowFrame_cluster_rna,
                 compensate = F, scale = F, 
                 colsToUse = clustering_markers, 
                 nClus = num_clusters, seed = 3)
cluster_cell_fsom_rna = GetMetaclusters(fSOM2_rna)

protein$cluster_rna_fsom <- cluster_cell_fsom_rna
ggplot(protein, aes(x = centroid_x, y = centroid_y, color = cluster_rna_fsom)) +
  geom_point(size = 0.000000001) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15),
    legend.title = element_blank(),
    legend.text = element_text(size = 12.5),
    legend.position = "bottomleft"
  ) +
  labs(x = "Centroid x", y = "Centroid y") +
  guides(color = guide_legend(override.aes = list(size = 3)))

# save protein to csv
write.csv(protein, "tonsil_clustering_fsom.csv", row.names=FALSE)

