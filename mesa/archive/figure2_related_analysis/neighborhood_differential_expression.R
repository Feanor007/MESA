
library(EnhancedVolcano)
library('DESeq2')
library(edgeR)
library(limma)
library(glmnet)
library(clusterProfiler)

df = read.csv('./../../../neighborhood/results/protein_clustering_result.csv')
rna_names = read.csv('./../../../neighborhood/neighborhood_multiomics/rna_variable_2000.csv')

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
counts = cbind(t(rna_df_9_matrix), t(rna_df_3_matrix))
rownames(counts) = 1:dim(counts)[1]

row_variances = apply(counts, 1, var)
col_variances = apply(counts, 2, var)
counts_filtered = counts[row_variances > 0.00001, col_variances > 0.00001]

samples_3_9 = c(rep('cluster_3', dim(rna_df_3_matrix)[1]), 
                rep('cluster_9', dim(rna_df_9_matrix)[1]))
samples_filtered = samples_3_9[col_variances > 0.00001]

dge = DGEList(counts=counts_filtered, group=samples_filtered)
dge = calcNormFactors(dge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dge = exactTest(dge)
de_results = topTags(dge, n = Inf, adjust.method = "BH")

de_neighborhood_rna = function(df1, df2, labels){
  counts_rna = cbind(t(df1), t(df2))
  rownames(counts_rna) = rna_names$gene
  row_variances = apply(counts_rna, 1, var)
  col_variances = apply(counts_rna, 2, var)
  counts_filtered = counts_rna[row_variances > 0.00001, col_variances > 0.00001]
  labels_filtered = labels[col_variances > 0.00001]
  
  dge = DGEList(counts=counts_filtered, group=labels_filtered)
  dge = calcNormFactors(dge)
  dge = estimateCommonDisp(dge)
  dge = estimateTagwiseDisp(dge)
  dge = exactTest(dge)
  de_results = topTags(dge, n = Inf, adjust.method = "BH")
  return(de_results)
}

de_results_3_9_rna = de_neighborhood_rna(rna_df_3_matrix, rna_df_9_matrix, samples_3_9)
png("de_plot_3_9.png", width = 1200, height = 800, res = 100)
EnhancedVolcano(de_results_3_9_rna$table,
                lab = rownames(de_results_3_9_rna$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(min(de_results_3_9_rna$table$logFC),max(de_results_3_9_rna$table$logFC)),
                ylim=c(0,max(-log(de_results_3_9_rna$table$PValue,10))),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 9 vs Cluster 3')
dev.off()

samples_3_5 = c(rep('cluster_3', dim(rna_df_3_matrix)[1]), 
                rep('cluster_5', dim(rna_df_5_matrix)[1]))
de_results_3_5_rna = de_neighborhood_rna(rna_df_3_matrix, rna_df_5_matrix, samples_3_5)
png("de_plot_3_5.png", width = 1200, height = 800, res = 100)
EnhancedVolcano(de_results_3_5_rna$table,
                lab = rownames(de_results_3_5_rna$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(min(de_results_3_5_rna$table$logFC),max(de_results_3_5_rna$table$logFC)),
                ylim=c(0,max(-log(de_results_3_5_rna$table$PValue,10))),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 5 vs Cluster 3')
dev.off()

samples_5_9 = c(rep('cluster_5', dim(rna_df_5_matrix)[1]), 
                rep('cluster_9', dim(rna_df_9_matrix)[1]))
de_results_5_9_rna = de_neighborhood_rna(rna_df_5_matrix, rna_df_9_matrix, samples_5_9)

png("de_plot_5_9.png", width = 1200, height = 800, res = 100)
EnhancedVolcano(de_results_5_9_rna$table,
                lab = rownames(de_results_5_9_rna$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(min(de_results_5_9_rna$table$logFC),max(de_results_5_9_rna$table$logFC)),
                ylim=c(0,max(-log(de_results_5_9_rna$table$PValue,10))),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 9 vs Cluster 5')
dev.off()



# edgeR for differential expression, PROTEIN
df_protein = df[,5:51]
df_cluster_9_protein = df_protein[df['cluster_protein'] == 9,]
df_cluster_3_protein = df_protein[df['cluster_protein'] == 3,]
df_cluster_5_protein = df_protein[df['cluster_protein'] == 5,]

samples_protein = c(rep('cluster_9', dim(df_cluster_9_protein)[1]), 
            rep('cluster_3', dim(df_cluster_3_protein)[1]))
counts_protein = cbind(t(df_cluster_9_protein), t(df_cluster_3_protein))

dge = DGEList(counts=counts_protein, group=samples_protein)
dge = calcNormFactors(dge)
dge = estimateCommonDisp(dge)
dge = estimateTagwiseDisp(dge)
dge = exactTest(dge)
de_results = topTags(dge, n = Inf, adjust.method = "BH")

EnhancedVolcano(de_results$table,
                lab = rownames(de_results$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(-0.8,0.8),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 9 vs Cluster 3')

de_neighborhood = function(df1, df2, labels){
  counts_protein = cbind(t(df1), t(df2))
  dge = DGEList(counts=counts_protein, group=labels)
  dge = calcNormFactors(dge)
  dge = estimateCommonDisp(dge)
  dge = estimateTagwiseDisp(dge)
  dge = exactTest(dge)
  de_results = topTags(dge, n = Inf, adjust.method = "BH")
  return(de_results)
}

samples_protein_3_5 = c(rep('cluster_3', dim(df_cluster_3_protein)[1]), 
                    rep('cluster_5', dim(df_cluster_5_protein)[1]))
de_results_3_5 = de_neighborhood(df_cluster_3_protein, df_cluster_5_protein, samples_protein_3_5)

EnhancedVolcano(de_results_3_5$table,
                lab = rownames(de_results_3_5$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(-0.8,0.8),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 5 vs Cluster 3')

samples_protein_3_9 = c(rep('cluster_3', dim(df_cluster_3_protein)[1]), 
                        rep('cluster_9', dim(df_cluster_9_protein)[1]))
de_results_3_9 = de_neighborhood(df_cluster_3_protein, df_cluster_9_protein, samples_protein_3_9)

EnhancedVolcano(de_results_3_9$table,
                lab = rownames(de_results_3_9$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(-0.8,0.8),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 9 vs Cluster 3')

samples_protein_5_9 = c(rep('cluster_5', dim(df_cluster_5_protein)[1]), 
                        rep('cluster_9', dim(df_cluster_9_protein)[1]))
de_results_5_9 = de_neighborhood(df_cluster_5_protein, df_cluster_9_protein, samples_protein_5_9)

EnhancedVolcano(de_results_5_9$table,
                lab = rownames(de_results_5_9$table),
                FCcutoff = 0.1,
                pCutoff = 0.05,
                x = 'logFC',
                y = 'FDR',
                labSize = 4.5,
                xlim=c(-0.8,0.8),
                drawConnectors = TRUE,
                widthConnectors = 0.75,
                title='Differential Expression Cluster 9 vs Cluster 5')

# Modeling to find cluster signatures
X = df_protein
y = df['cluster_protein']
fit_9 = cv.glmnet(as.matrix(X), as.matrix(y == 9), family='binomial', type.measure='auc')

index_subset = df['cluster_protein'] == 9 | df['cluster_protein'] == 3 | df['cluster_protein'] == 5
X_subset = df_protein[index_subset,]
y_subset = y[index_subset]
fit_9 = cv.glmnet(as.matrix(X_subset), as.matrix(y_subset == 9), 
                  family='binomial', type.measure='auc')

reorder_dataframe_by_absolute_value <- function(dataframe, column_name) {
  absolute_values <- abs(dataframe[[column_name]])
  dataframe$absolute_values <- absolute_values
  sorted_dataframe <- dataframe[order(dataframe$absolute_values, decreasing = TRUE), ]
  sorted_dataframe <- sorted_dataframe[, !(names(sorted_dataframe) %in% "absolute_values")]
  return(sorted_dataframe)
}

df_coef_9 = data.frame('feature'=rownames(coef(fit_9)), 'coef'=as.vector(coef(fit_9)))
df_coef_9_order = reorder_dataframe_by_absolute_value(df_coef_9, 'coef')

fit_3 = cv.glmnet(as.matrix(X_subset), as.matrix(y_subset == 3), 
                  family='binomial', type.measure='auc')
df_coef_3 = data.frame('feature'=rownames(coef(fit_3)), 'coef'=as.vector(coef(fit_3)))
df_coef_3_order = reorder_dataframe_by_absolute_value(df_coef_3, 'coef')

fit_5 = cv.glmnet(as.matrix(X_subset), as.matrix(y_subset == 5), 
                  family='binomial', type.measure='auc')
df_coef_5 = data.frame('feature'=rownames(coef(fit_5)), 'coef'=as.vector(coef(fit_5)))
df_coef_5_order = reorder_dataframe_by_absolute_value(df_coef_5, 'coef')

