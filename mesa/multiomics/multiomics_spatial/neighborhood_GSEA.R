library(EnhancedVolcano)
library(DESeq2)
library(edgeR)
library(limma)
library(glmnet)
library(clusterProfiler)
library(msigdbr)
library(fgsea)
library(tidyverse)
library(svglite)


df = read.csv('./../results/protein_clustering_result.csv')
rna_names = read.csv('./rna_variable_2000.csv')

df_cluster_9 = df[df['cluster_protein'] == 9,]
df_cluster_3 = df[df['cluster_protein'] == 3,]
df_cluster_5 = df[df['cluster_protein'] == 5,]

process_rna_from_string = function(x){
  string_list <- gsub("\\[|\\]", "", x)
  return(as.numeric(strsplit(string_list, ", ")[[1]]))
}

rna_df_9 = lapply(df_cluster_9$all_rna, process_rna_from_string)
rna_df_9_matrix = do.call(rbind, rna_df_9) #7392 x 2000

rna_df_3 = lapply(df_cluster_3$all_rna, process_rna_from_string)
rna_df_3_matrix = do.call(rbind, rna_df_3) 

rna_df_5 = lapply(df_cluster_5$all_rna, process_rna_from_string)
rna_df_5_matrix = do.call(rbind, rna_df_5) 

# edgeR for differential expression, RNA

de_neighborhood_rna = function(df1, df2, labels, threshold = 0.0001){
  counts_rna = cbind(t(df1), t(df2))
  rownames(counts_rna) = rna_names$gene
  row_variances = apply(counts_rna, 1, var)
  col_variances = apply(counts_rna, 2, var)
  counts_filtered = counts_rna[row_variances > threshold, col_variances > threshold]
  labels_filtered = labels[col_variances > threshold]
  
  dge = DGEList(counts=counts_filtered, group=labels_filtered)
  dge = calcNormFactors(dge)
  dge = estimateCommonDisp(dge)
  dge = estimateTagwiseDisp(dge)
  dge = exactTest(dge)
  de_results = topTags(dge, n = Inf, adjust.method = "BH")
  return(de_results)
}

samples_3_9 = c(rep('cluster_3', dim(rna_df_3_matrix)[1]), 
                rep('cluster_9', dim(rna_df_9_matrix)[1]))
de_results_3_9_rna = de_neighborhood_rna(rna_df_3_matrix, rna_df_9_matrix, samples_3_9, threshold=0.001)

samples_3_5 = c(rep('cluster_3', dim(rna_df_3_matrix)[1]), 
                rep('cluster_5', dim(rna_df_5_matrix)[1]))
de_results_3_5_rna = de_neighborhood_rna(rna_df_3_matrix, rna_df_5_matrix, samples_3_5, threshold=0.001)

samples_5_9 = c(rep('cluster_5', dim(rna_df_5_matrix)[1]), 
                rep('cluster_9', dim(rna_df_9_matrix)[1]))
de_results_5_9_rna = de_neighborhood_rna(rna_df_5_matrix, rna_df_9_matrix, samples_5_9, threshold=0.001)

# Get gene set database
H = msigdbr(species = 'Homo sapiens', category = 'H')
#H.select = select(H, gs_name, gene_symbol)

C2 = msigdbr(species = 'Homo sapiens', category = 'C2')
#C2.select = select(C2, gs_name, gene_symbol)

C5 = msigdbr(species = 'Homo sapiens', category = 'C5')

# Define significant genes
#sigif = de_results_3_9_rna$table %>% filter(FDR < 0.2)

# enrich_H = enricher(gene = rownames(sigif), TERM2GENE = H.select)
# head(enrich_H@result)
# enrich_H_res = enrich_H@result
# enrich_H_df = enrich_H@result %>% 
#   separate(BgRatio, into=c("size.term", "size.category"), sep='/') %>%
#   separate(GeneRatio, into=c("size.overlap.term", "size.overlap.category"), sep='/') %>%
#   mutate_at(vars("size.term", "size.category", "size.overlap.term", "size.overlap.category"), as.numeric) %>%
#   mutate("k.K" = size.overlap.term/size.term)
# enrich_H_df %>%
#   filter(p.adjust <= 0.2) %>%
#   ggplot(aes(x = reorder(Description, -p.adjust), y = p.adjust)) + 
#   geom_col() + 
#   theme_classic() +
#   coord_flip()

get_gsea = function(gene_set, de_res){
  gene_sets_list = split(gene_set$gene_symbol, gene_set$gs_name)
  gene_ranks = de_res$table$logFC 
  names(gene_ranks) = rownames(de_res$table)
  fgsea_H = fgsea(gene_sets_list, gene_ranks)
  return(fgsea_H)
}

##################################### ##################################### 
##################################### 3 VS 9 ##################################### 
##################################### ##################################### 
set.seed(9)
fgsea_H_3_9 = get_gsea(H, de_results_3_9_rna)
fgseaResTidy <- fgsea_H_3_9 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_H_3_9, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="3 vs 9: Hallmark pathways NES from GSEA") + 
  theme_minimal()

set.seed(9)
fgsea_C2_3_9 = get_gsea(C2, de_results_3_9_rna)
fgseaResTidy <- fgsea_C2_3_9 %>%
  as_tibble() %>%
  arrange(desc(NES))  %>%
filter(padj <= 0.05) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

fgsea_C2_3_9_plot = ggplot(fgseaResTidy, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.05)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="3 vs 9: C2 pathways NES from GSEA") + 
  theme_minimal() + 
  theme(panel.background = element_rect(fill = "white"))
ggsave(filename = "./gsea_plot/fgsea_C2_3_9_plot.png", plot = fgsea_C2_3_9_plot, width = 10, height = 9, dpi = 100)


##################################### ##################################### 
##################################### 3 VS 5 ##################################### 
##################################### ##################################### 
set.seed(9)
fgsea_H_3_5 = get_gsea(H, de_results_3_5_rna)
fgseaResTidy <- fgsea_H_3_5 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

set.seed(9)
fgsea_C2_3_5 = get_gsea(C2, de_results_3_5_rna)
fgseaResTidy <- fgsea_C2_3_5 %>%
  as_tibble() %>%
  arrange(desc(NES))  %>%
  filter(padj <= 0.00001) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

fgsea_C2_3_5_plot = ggplot(fgseaResTidy, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.00001)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="3 vs 5: C2 pathways NES from GSEA") + 
  theme_minimal()
ggsave(filename = "./gsea_plot/fgsea_C2_3_5_plot.png", plot = fgsea_C2_3_5_plot, width = 10, height = 9, dpi = 100)

##################################### ##################################### 
##################################### 5 VS 9 ##################################### 
##################################### ##################################### 
set.seed(9)
fgsea_H_5_9 = get_gsea(H, de_results_5_9_rna)
fgseaResTidy <- fgsea_H_5_9 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

fgsea_H_5_9_plot = ggplot(fgsea_H_5_9, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="5 vs 9: Hallmark pathways NES from GSEA") + 
  theme_minimal()
ggsave(filename = "./gsea_plot/fgsea_H_5_9_plot.png", plot = fgsea_H_5_9_plot, width = 10, height = 9, dpi = 100)

set.seed(9)
fgsea_C2_5_9 = get_gsea(C2, de_results_5_9_rna)
fgseaResTidy <- fgsea_C2_5_9 %>%
  as_tibble() %>%
  arrange(desc(NES))  %>%
  filter(padj <= 0.00001) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

fgsea_C2_5_9_plot = ggplot(fgseaResTidy, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.00001)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="5 vs 9: C2 pathways NES from GSEA") + 
  theme_minimal()
ggsave(filename = "./gsea_plot/fgsea_C2_5_9_plot.png", plot = fgsea_C2_5_9_plot, width = 10, height = 9, dpi = 100)



gene_sets_list = split(H$gene_symbol, H$gs_name)
#gene_sets_list = split(C2$gene_symbol, C2$gs_name)
#gene_sets_list = split(C5$gene_symbol, C5$gs_name)

gene_ranks = de_results_3_9_rna$table$logFC 
names(gene_ranks) = rownames(de_results_3_9_rna$table)
fgsea_H = fgsea(gene_sets_list, gene_ranks)

fgseaRes = fgsea_H
fgseaResTidy <- fgseaRes %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
  #filter(padj <= 0.1) 

fgseaResTidy %>% 
  #filter(padj <= 0.05)  %>%
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgseaResTidy, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="Hallmark pathways NES from GSEA") + 
  theme_minimal()




