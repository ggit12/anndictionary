# adict
Conveniently process a dictionary of anndatas (adata_dict)

To view the documentation, clone this repository, and then navigate to docs/build/index.html and view in a browser.

# Install
\#in bash, clone the repository  
git clone https://github.com/ggit12/adict  
cd adict  
\#activate your prefered environment, then  
pip install -e .  


# Example usage:

\# Helpful inline frequency and joint frequency tables  
display_html_summary(summarize_metadata(adata, columns = ['cell_type', 'donor_id', 'cell_type*donor_id']))


## Build and process an adata_dict
#### Processing of each dictionary entry will be independent

\# Build a dictionary of adata (subsetted by tissue)  
adata_dict = build_adata_dict(adata, ['tissue'], ['blood', 'kidney'])

\# Normalize each AnnData in the dictionary  
normalize_adata_dict(adata_dict)

\# Log transform each AnnData in the dictionary  
log_transform_adata_dict(adata_dict)

\# Optionally, you might subset the data to only high-variance genes  
set_high_variance_genes(adata_dict, n_top_genes=2000, subset=False)

\# Scale each AnnData in the dictionary  
scale_adata_dict(adata_dict)

\# Perform PCA on each AnnData in the dictionary  
pca_adata_dict(adata_dict, n_comps=50, mask_var='highly_variable')

\# Calculate and plot UMAP based on pca  
calculate_umap_adata_dict(adata_dict, 'X_pca')  
plot_umap_adata_dict(adata_dict, ['cell_type', 'donor_id'])

\# Write each adata in adata_dict separately  
write_h5ad_adata_dict(adata_dict, "/Users/ggit12/adatas", file_prefix="adata_")
