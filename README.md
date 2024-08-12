# anndict
A package for processing multiple `anndata` objects in parallel

# Documentation

To view the documentation, clone this repository, and then navigate to docs/build/index.html and view in a browser.

# Install
Copy and paste the code below into a terminal window to download this package and install it in a conda env. It assumes conda is already installed.

```bash
git clone https://github.com/ggit12/anndict
cd anndict
conda create -n anndict
conda activate anndict
pip install -e .
```


# About
`anndict` is a package that lets you process multiple `anndata` objects in parallel with a simplified interface (so that you can avoid writing a bunch of for loops). This is accomplished by a dictionary-based wrapping of `scanpy`.

## AI to make things easier
To make it easier to handle basics like cell type annotation for anndatas (one or many), and to make cell type labels agree across multiple anndatas, we provide AI-based processing to handle these tedious tasks. There are also AI-based functions to to biological process inference on gene lists.

These functions include `ai_annotate_cell_type`, `ai_annotate_biological_process`, and `ai_unify_labels`, which all operate over many adata at once.



During installation, you will be prompted to enter an OpenAI API key to allow the AI backend to operate (this will modify your .bashrc, you can also skip this step and set the OPENAI_API_KEY environemnt variable yourself). Directions on how to get an OpenAI API key can be found here: https://platform.openai.com/docs/quickstart/account-setup. You'll need to buy at least $5 of credits to start (so that you can use GPT-4+).

### If you like functions:
The main function in this package is `adata_dict_fapply()` (and its cousin `adata_dict_fapply_return()`, which does the same thing but also returns the result as a dictionary). 

`adata_dict_fapply()` works just like `lapply()` in R. It takes `adata_dict_fapply(adata_dict, func, **kwargs)`. 

- `adata_dict` is a dictionary.
- `func` is a function to apply over each element of the dictionary.
- `**kwargs` are arguments that `func` takes.

You can have `func` take the argument `adt_key` (i.e., `func(adata, adt_key=None)`) if you want the dictionary key to be available within `func`.

Many functions in `anndict` are built around `adata_dict_fapply()`, and the package provides prebuilt wrappers for several common Scanpy functions, as well as functions to build and concatenate `adata` dictionaries.

### If you like objects (under development):
This package also defines the class AdataDict(), which is a dictionary of anndatas. When a method is called on an AdataDict, it is applied independently to each adata in the dictionary. Currently in beta.

The syntax looks like this: `adata_dict.fapply(func, **kwargs)`, where `adata_dict`, `func`, and `**kwargs` are as defined above.


Read the tutorial below for basic demonstrations.

# Tutorial
This is the tutorial notebook for `anndict`. Follow the tutorial below to get started.

See `tutorial_notbooks` for other tutorials:

- Label transfer with UCE
- Automated spatial transcriptomic annotation with UCE



```python
#import the package
import anndict as adt
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
#load an anndata

#for this tutorial, we'll use Tabula Sapiens from cellxgene census
#but you could use any anndata you want
import cellxgene_census

#this command gets liver and kidney from tabula sapiens 
census = cellxgene_census.open_soma(census_version="2023-12-15")
adata = cellxgene_census.get_anndata(
    census,
    organism = "homo_sapiens",
    measurement_name = "RNA",
    obs_value_filter = "(dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc') & ((tissue_general == 'liver') | (tissue_general == 'kidney') | (tissue_general == 'spleen'))",
    obs_embeddings = ["uce"]
)


```


```python
adata
```




    AnnData object with n_obs × n_vars = 48652 × 60664
        obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
        var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
        obsm: 'uce'



`anndict` has a helpful frequency summary function to do frequencies and joint frequencies


```python
adt.display_html_summary(adt.summarize_metadata(adata, columns = ['cell_type*tissue']))

#Run the line below to see frequencies of single variables
# adt.display_html_summary(adt.summarize_metadata(adata, columns = ['donor_id', 'tissue','cell_type']))
```


<div style="display: flex; flex-wrap: wrap;"><div style="flex: 1; padding: 10px;"><h3>cell_type x tissue</h3><table border="1"><tr><th></th><th>kidney</th><th>liver</th><th>spleen</th></tr><tr><td>B cell</td><td>341</td><td>0</td><td>0</td></tr><tr><td>CD141-positive myeloid dendritic cell</td><td>0</td><td>0</td><td>51</td></tr><tr><td>CD1c-positive myeloid dendritic cell</td><td>0</td><td>0</td><td>51</td></tr><tr><td>CD4-positive helper T cell</td><td>181</td><td>0</td><td>0</td></tr><tr><td>CD4-positive, alpha-beta memory T cell</td><td>0</td><td>0</td><td>1771</td></tr><tr><td>CD8-positive, alpha-beta T cell</td><td>241</td><td>0</td><td>43</td></tr><tr><td>CD8-positive, alpha-beta memory T cell</td><td>0</td><td>0</td><td>4301</td></tr><tr><td>T cell</td><td>0</td><td>146</td><td>0</td></tr><tr><td>classical monocyte</td><td>0</td><td>0</td><td>4443</td></tr><tr><td>endothelial cell</td><td>95</td><td>267</td><td>596</td></tr><tr><td>endothelial cell of hepatic sinusoid</td><td>0</td><td>421</td><td>0</td></tr><tr><td>erythrocyte</td><td>0</td><td>108</td><td>170</td></tr><tr><td>fibroblast</td><td>0</td><td>76</td><td>0</td></tr><tr><td>hematopoietic stem cell</td><td>0</td><td>0</td><td>73</td></tr><tr><td>hepatocyte</td><td>0</td><td>1558</td><td>0</td></tr><tr><td>innate lymphoid cell</td><td>0</td><td>0</td><td>117</td></tr><tr><td>intermediate monocyte</td><td>0</td><td>0</td><td>131</td></tr><tr><td>intrahepatic cholangiocyte</td><td>0</td><td>42</td><td>0</td></tr><tr><td>kidney epithelial cell</td><td>8331</td><td>0</td><td>0</td></tr><tr><td>liver dendritic cell</td><td>0</td><td>34</td><td>0</td></tr><tr><td>macrophage</td><td>321</td><td>1381</td><td>1335</td></tr><tr><td>mature NK T cell</td><td>131</td><td>245</td><td>2439</td></tr><tr><td>memory B cell</td><td>0</td><td>0</td><td>6460</td></tr><tr><td>monocyte</td><td>0</td><td>612</td><td>0</td></tr><tr><td>naive B cell</td><td>0</td><td>0</td><td>1044</td></tr><tr><td>naive thymus-derived CD4-positive, alpha-beta T cell</td><td>0</td><td>0</td><td>1261</td></tr><tr><td>naive thymus-derived CD8-positive, alpha-beta T cell</td><td>0</td><td>0</td><td>610</td></tr><tr><td>neutrophil</td><td>0</td><td>83</td><td>4539</td></tr><tr><td>plasma cell</td><td>0</td><td>34</td><td>3171</td></tr><tr><td>plasmacytoid dendritic cell</td><td>0</td><td>0</td><td>43</td></tr><tr><td>platelet</td><td>0</td><td>0</td><td>29</td></tr><tr><td>regulatory T cell</td><td>0</td><td>0</td><td>288</td></tr><tr><td>type I NK T cell</td><td>0</td><td>0</td><td>1038</td></tr></table></div></div>


The first step is to build the dictionary of anndata.  
For this tutorial, we'll build a dictionary of tissue-specific objects, but you could do variables or combinations of variables.


```python
#This command will give an adata_dict where each value is an anndata containing cells from a single tissue
#Take only liver and kidney
adata_dict = adt.build_adata_dict(adata=adata,strata_keys=['tissue'], desired_strata=['liver','kidney'])
```


```python
adata_dict
```




    {'liver': View of AnnData object with n_obs × n_vars = 5007 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce',
     'kidney': View of AnnData object with n_obs × n_vars = 9641 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce'}



Building adata_dicts is flexible. We could also get only macrophages and endothelial cells from the desired tissues like this:


```python
strata_keys = ['tissue', 'cell_type'] #keys in .obs of each anndata
desired_strata = {'tissue': ['liver', 'kidney'],
                  'cell_type': ['macrophage', 'endothelial cell']
                  }
adata_dict_only_mac_and_endo = adt.build_adata_dict(adata=adata, strata_keys=strata_keys, desired_strata=desired_strata)
```


```python
adata_dict_only_mac_and_endo
```




    {'liver_macrophage': View of AnnData object with n_obs × n_vars = 1381 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars', 'tissue_cell_type'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce',
     'liver_endothelial cell': View of AnnData object with n_obs × n_vars = 267 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars', 'tissue_cell_type'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce',
     'kidney_macrophage': View of AnnData object with n_obs × n_vars = 321 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars', 'tissue_cell_type'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce',
     'kidney_endothelial cell': View of AnnData object with n_obs × n_vars = 95 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars', 'tissue_cell_type'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce'}



And we can summarize their metadata using adata_dict wrappers for the previously mentioned summary functions  
Note that each anndata in this dictionary only has 1 tissue and 1 cell type, as it should


```python
#Not run for brevity; uncomment and run the line below to see contents of adata_dict_only_mac_and_endo
# adt.display_html_summary_adata_dict(adt.summarize_metadata_adata_dict(adata_dict_only_mac_and_endo, columns = ["tissue", "cell_type", "donor_id"]))
```

For the rest of this tutorial, we'll use adata_dict that has multiple celltypes per tissue


```python
#Run the line below to see contents of adata_dict
# adt.display_html_summary_adata_dict(adt.summarize_metadata_adata_dict(adata_dict, columns = ["cell_type"]))
```

Now, let's do some standard scanpy preprocessing
Each anndata in the dictionary will be processed independently

Note: since many of these functions are scanpy wrappers, they take and pass any function arguments to their underlying scanpy functions.
resample_adata_dict uses sc.pp.subsample under the hood and so takes n_obs or fraction (among others)


```python
#Subset each celltype to 1000 cells and drop celltypes with fewer than 50 cells
adata_dict = adt.resample_adata_dict(adata_dict, strata_keys=['cell_type'], min_num_cells=50, n_obs=1000)
```


```python
#Confirm that the subsampling and dropping has been performed (commented out for brevity)
# adt.display_html_summary_adata_dict(adt.summarize_metadata_adata_dict(adata_dict, columns = ["cell_type"]))
```


```python
# Normalize each AnnData in the dictionary
adt.normalize_adata_dict(adata_dict)

# Log transform each AnnData in the dictionary
adt.log_transform_adata_dict(adata_dict)

# Optionally, you might subset the data to only high-variance genes
adt.set_high_variance_genes(adata_dict, n_top_genes=2000, subset=False)

# Scale each AnnData in the dictionary
adt.scale_adata_dict(adata_dict)

# Perform PCA on each AnnData in the dictionary
adt.pca_adata_dict(adata_dict, n_comps=50, mask_var='highly_variable')

# Calculate and plot UMAP based on pca
adt.calculate_umap_adata_dict(adata_dict, use_rep='X_pca')
adt.plot_umap_adata_dict(adata_dict, color_by = ['cell_type', 'donor_id'])

```

    Plotting UMAP for key: liver



    
![png](Basic_Tutorial_files/Basic_Tutorial_24_1.png)
    


    Plotting UMAP for key: kidney



    
![png](Basic_Tutorial_files/Basic_Tutorial_24_3.png)
    



```python
# Write each adata in adata_dict separately
adt.write_h5ad_adata_dict(adata_dict, "~/adatas", file_prefix="adata_")
```


```python
#Concatenate the adata_dict back to a single anndata
adata = adt.concatenate_adata_dict(adata_dict)
```


```python
#View summary table for concatenated adata
# adt.display_html_summary(adt.summarize_metadata(adata, columns = ['tissue','cell_type','cell_type*tissue']))
```


```python
#Note, you could also run scanpy functions directly on the adata_dict like this:
import scanpy as sc
adata_dict = adt.build_adata_dict(adata=adata,strata_keys=['tissue'], desired_strata=['liver','kidney'])
adata_dict.fapply(sc.pp.subsample, fraction=0.1)

```




    {'liver': None, 'kidney': None}




```python
adata_dict
```




    {'liver': AnnData object with n_obs × n_vars = 500 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce',
     'kidney': AnnData object with n_obs × n_vars = 964 × 60664
         obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
         var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length', 'nnz', 'n_measured_obs'
         obsm: 'uce'}


