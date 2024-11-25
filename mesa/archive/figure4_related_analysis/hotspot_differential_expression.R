library(EnhancedVolcano)
library('DESeq2')
library(edgeR)
library(limma)
library(glmnet)
library(clusterProfiler)
library(msigdbr)
library(fgsea)
library(svglite)

########################################################
####################### B cell #######################
########################################################
df = read.csv('./results/protein_rna_matched_CRC_hotspot.csv')
df = data.frame(df)

df_hotspot = df[df$hotspot == "True", ]
df_coldspot = df[df$coldspot == "True", ]

df_hotspot_rna = df_hotspot[,109:2108]
df_coldspot_rna = df_coldspot[,109:2108]

counts = cbind(t(df_hotspot_rna), t(df_coldspot_rna))
counts[is.na(counts)] = 0
#rownames(counts) = 1:dim(counts)[1]

row_variances = apply(counts, 1, var)
col_variances = apply(counts, 2, var)
counts_filtered = counts[row_variances > 0.00001, col_variances > 0.00001]

labels = c(rep('hotspot', dim(df_hotspot)[1]), 
          rep('coldspot', dim(df_coldspot)[1]))
labels_filtered = labels[col_variances > 0.00001]

# check direction, right is hotspot, left is coldspot
ind = which(colnames(df_hotspot_rna) == 'FOS')
ind_fos_hotspot = mean(df_hotspot_rna[,ind], na.rm = TRUE)
ind_fos_coldspot = mean(df_coldspot_rna[,ind], na.rm = TRUE)
fos_fc = log(ind_fos_hotspot / ind_fos_coldspot)

dge = DGEList(counts=counts_filtered, group=labels_filtered)
dge = calcNormFactors(dge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dge = exactTest(dge)
de_results = topTags(dge, n = Inf, adjust.method = "BH")

png("de_plot_hotspot_b_cell.png", width = 1200, height = 800, res = 100)
plot_b = EnhancedVolcano(de_results$table,
                lab = rownames(de_results$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(min(de_results$table$logFC),max(de_results$table$logFC)),
                ylim=c(0,max(-log(de_results$table$PValue,10))),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression B-cell Hotspot vs Coldspot')
dev.off()

svglite::svglite("de_plot_hotspot_b_cell.svg", width=12, height=8)
print(plot_b)
dev.off()

# Get gene set database
H = msigdbr(species = 'Homo sapiens', category = 'H')
C2 = msigdbr(species = 'Homo sapiens', category = 'C2')
C5 = msigdbr(species = 'Homo sapiens', category = 'C5')

get_gsea = function(gene_set, de_res){
  gene_sets_list = split(gene_set$gene_symbol, gene_set$gs_name)
  gene_ranks = de_res$table$logFC 
  names(gene_ranks) = rownames(de_res$table)
  fgsea_H = fgsea(gene_sets_list, gene_ranks)
  return(fgsea_H)
}

set.seed(9)
fgsea_H = get_gsea(H, de_results)
fgsea_H_sorted <- fgsea_H %>% 
  arrange(padj)
write.csv(fgsea_H_sorted[,1:(ncol(fgsea_H_sorted)-1)], file = "./results/fgsea_H_sorted_b_cell.csv", row.names = FALSE)

fgseaResTidy <- fgsea_H %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_H, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="B-cell Hotspot vs. Coldspot") + 
  theme_minimal()

set.seed(9)
fgsea_C2 = get_gsea(C2, de_results)
fgsea_C2_sorted <- fgsea_C2 %>% arrange(padj)
write.csv(fgsea_C2_sorted[,1:(ncol(fgsea_C2_sorted)-1)], file = "./results/fgsea_C2_sorted_b_cell.csv", row.names = FALSE)

fgseaResTidy <- fgsea_C2 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_C2, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="B-cell Hotspot vs. Coldspot (C2)") + 
  theme_minimal()

set.seed(9)
fgsea_C5 = get_gsea(C5, de_results)
fgsea_C5_sorted <- fgsea_C5 %>% arrange(padj)
write.csv(fgsea_C5_sorted[,1:(ncol(fgsea_C5_sorted)-1)], file = "./results/fgsea_C5_sorted_b_cell.csv", row.names = FALSE)

fgseaResTidy <- fgsea_C5 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_C5, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="B-cell Hotspot vs. Coldspot (C5)") + 
  theme_minimal()


########################################################
####################### Macrophages #######################
########################################################
df = read.csv('./results/protein_rna_matched_CRC_hotspot_macro_163.csv')
df = data.frame(df)

### NOTE HERE ABOUT COLDSPOT
df_hotspot = df[df$hotspot == "True", ]
df_coldspot = df[df$coldspot == "True", ]
#df_coldspot = df[df$hotspot == "False", ]

df_hotspot_rna = df_hotspot[,108:2107]
df_coldspot_rna = df_coldspot[,108:2107]

counts = cbind(t(df_hotspot_rna), t(df_coldspot_rna))
#[is.na(counts)] = 0
#rownames(counts) = 1:dim(counts)[1]

row_variances = apply(counts, 1, var, na.rm=TRUE)
col_variances = apply(counts, 2, var, na.rm=TRUE)
row_variances[is.na(row_variances)] = 0
col_variances[is.na(col_variances)] = 0
counts_filtered = counts[row_variances > 0.00001, col_variances > 0.00001] #0.00001
rowname_fil = rownames(counts_filtered)

labels = c(rep('hotspot', dim(df_hotspot)[1]), 
           rep('coldspot', dim(df_coldspot)[1]))
labels_filtered = labels[col_variances > 0.00001]

# check direction, right is hotspot, left is coldspot
#ind = which(colnames(df_hotspot_rna) == 'FOS')
#ind_fos_hotspot = mean(df_hotspot_rna[,ind], na.rm = TRUE)
#ind_fos_coldspot = mean(df_coldspot_rna[,ind], na.rm = TRUE)
#fos_fc = log(ind_fos_hotspot / ind_fos_coldspot)

counts_filtered <- matrix(as.numeric(counts_filtered), 
                          nrow = nrow(counts_filtered), ncol = ncol(counts_filtered))
rownames(counts_filtered) = rowname_fil
dge = DGEList(counts=counts_filtered, group=labels_filtered)
dge = calcNormFactors(dge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dge = exactTest(dge)
de_results = topTags(dge, n = Inf, adjust.method = "fdr")

png("./results/de_plot_hotspot_macro_new.png", width = 1200, height = 800, res = 100)
plot_macro = EnhancedVolcano(de_results$table,
                         lab = rownames(de_results$table),
                         FCcutoff = 0.1,
                         pCutoff = 0.05,
                         x = 'logFC',
                         y = 'FDR',
                         labSize = 4.5,
                         xlim=c(min(de_results$table$logFC),max(de_results$table$logFC)),
                         ylim=c(0, max(-log(de_results$table$FDR, 10))),
                         drawConnectors = TRUE,
                         widthConnectors = 0.75,
                         title='Differential Expression Macrophages Hotspot vs Coldspot')
print(plot_macro)
dev.off()

svglite::svglite("./results/de_plot_hotspot_macro_new.svg", width=12, height=8)
print(plot_macro)
dev.off()

# Get gene set database
H = msigdbr(species = 'Homo sapiens', category = 'H')
C2 = msigdbr(species = 'Homo sapiens', category = 'C2')
C5 = msigdbr(species = 'Homo sapiens', category = 'C5')

get_gsea = function(gene_set, de_res){
  gene_sets_list = split(gene_set$gene_symbol, gene_set$gs_name)
  gene_ranks = de_res$table$logFC 
  names(gene_ranks) = rownames(de_res$table)
  fgsea_H = fgsea(gene_sets_list, gene_ranks)
  return(fgsea_H)
}

set.seed(9)
fgsea_H = get_gsea(H, de_results)
fgsea_H_sorted <- fgsea_H %>% 
  arrange(padj)
write.csv(fgsea_H_sorted[,1:(ncol(fgsea_H_sorted)-1)], file = "./results/fgsea_H_sorted_macro_new.csv", row.names = FALSE)

fgseaResTidy <- fgsea_H %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_H, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.1)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="Macrophage Hotspot vs. Coldspot") + 
  theme_minimal()

set.seed(9)
fgsea_C2 = get_gsea(C2, de_results)
fgsea_C2_sorted <- fgsea_C2 %>% arrange(padj)
write.csv(fgsea_C2_sorted[,1:(ncol(fgsea_C2_sorted)-1)], file = "./results/fgsea_C2_sorted_macro_new.csv", row.names = FALSE)

fgseaResTidy <- fgsea_C2 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_C2, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="Macrophage Hotspot vs. Coldspot (C2)") + 
  theme_minimal()

set.seed(9)
fgsea_C5 = get_gsea(C5, de_results)
fgsea_C5_sorted <- fgsea_C5 %>% arrange(padj)
write.csv(fgsea_C5_sorted[,1:(ncol(fgsea_C5_sorted)-1)], file = "./results/fgsea_C5_sorted_macro_new.csv", row.names = FALSE)

fgseaResTidy <- fgsea_C5 %>%
  as_tibble() %>%
  arrange(desc(NES)) # %>%
#filter(padj <= 0.1) 

fgseaResTidy %>% 
  dplyr::select(-leadingEdge, -ES) %>% 
  arrange(padj) %>% 
  DT::datatable()

ggplot(fgsea_C5, aes(reorder(pathway, NES), NES)) +
  geom_col(aes(fill=padj < 0.2)) +
  coord_flip() +
  labs(x="Pathway", y="Normalized Enrichment Score",
       title="Macrophage Hotspot vs. Coldspot (C5)") + 
  theme_minimal()

