library(reticulate)
library(dplyr)
library(ggplot2)
library(ggbeeswarm)
# py_install("mesa-py")
# py_install("numpy==1.24.0")
eco <- import("mesa.ecospatial")
ad <- import("anndata")
adata <- ad$read_h5ad('/Users/Emrys/Dropbox/spatial_augmentation/data/codex_mouse_spleen/codex_mouse_spleen.h5ad')
adata$obsm['spatial'] <- adata$obsm['spatial'] / 1000
library_ids <- unique(adata$obs[['sample']])
library_ids <- as.vector(library_ids) 

# Calculate MDI (Multiscale Diversity Index)
scales <- c(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
result <- eco$multiscale_diversity(spatial_data = adata,
                                   scales = scales,
                                   library_key = 'sample',
                                   library_ids = library_ids,
                                   spatial_key = 'spatial',
                                   cluster_key = 'cell_type',
                                   random_patch = FALSE,
                                   plotfigs = FALSE,
                                   savefigs = FALSE,
                                   patch_kwargs = dict(random_seed = NULL, min_points = 2),
                                   other_kwargs = dict(metric = 'Shannon Diversity'))
mdi_results <- as.data.frame(t(result[[2]]))
colnames(mdi_results) <- c("MDI")

# Add columns for 'Condition' and 'Sample_id'
mdi_results$Condition <- NA
mdi_results$Sample_id <- rownames(mdi_results)

# Assign 'Condition' based on 'Sample_id'
mdi_results <- mdi_results %>%
  mutate(Condition = case_when(
    grepl('BALBc', Sample_id) ~ 'BALBc',
    grepl('MRL', Sample_id) ~ 'MRL',
    TRUE ~ 'Other'
  ))

# Subset data for the two conditions
mdi_balbc <- mdi_results %>% filter(Condition == 'BALBc') %>% pull(MDI)
mdi_mrl <- mdi_results %>% filter(Condition == 'MRL') %>% pull(MDI)

# Perform Welch's t-test
t_test_result <- t.test(mdi_balbc, mdi_mrl, var.equal = FALSE)
p_value <- t_test_result$p.value

# Create boxplot with swarmplot overlay
p <- ggplot(mdi_results, aes(x = Condition, y = MDI, fill = Condition)) + 
  geom_boxplot(outlier.shape = NA) +    # Boxplot without outliers
  geom_beeswarm(aes(color = Condition), size = 2, alpha = 0.6) +  # Swarm plot
  labs(title = "MDI Results by Condition", x = "Condition", y = "Multiscale Diversity Index (MDI)") + 
  theme_minimal() + annotate("text", x = 1.5, y = max(mdi_results$MDI) + 0.1, 
                             label = paste0("p = ", round(p_value, 3)), size = 5, color = "black")
print(p)