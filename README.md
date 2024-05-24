# adict
Conveniently process a dictionary of anndatas (adata_dict)

#example usage:

#build a dictionary of adata (subsetted by tissue)
adata_dict = build_adata_dict(adata, ['tissue'], ['blood', 'kidney'])

#Process each adata in adata_dict independently
# Normalize each AnnData in the dictionary
normalize_adata_dict(adata_dict)

# Log transform each AnnData in the dictionary
log_transform_adata_dict(adata_dict)

# Optionally, you might subset the data to only high-variance genes:
set_high_variance_genes(adata_dict, n_top_genes=2000, subset=False)

# Scale each AnnData in the dictionary
scale_adata_dict(adata_dict)

# Perform PCA on each AnnData in the dictionary
pca_adata_dict(adata_dict, n_comps=50, mask_var='highly_variable')

#calculate and plot UMAP based on pca
calculate_umap_adata_dict(adata_dict, 'X_pca')
plot_umap_adata_dict(adata_dict, ['cell_type', 'donor_id'])

#write each adata in adata_dict separately
write_h5ad_adata_dict(adata_dict, "/Users/ggit12/adatas", file_prefix="adata_")
