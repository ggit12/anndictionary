# AnnDictionary
A package for processing multiple `anndata` objects in parallel with LLMs

# Documentation

Complete documentation covering all functions and modules is available at https://ggit12.github.io/anndictionary. We are actively working to enhance its appearance and improve the overall user experience.


# Citation

If you use this package, please cite:

> #### Benchmarking Cell Type Annotation by Large Language Models with AnnDictionary  
> **George Crowley, Tabula Sapiens Consortium, Stephen R. Quake**  
> *bioRxiv* 2024.10.10.617605  
> [doi: https://doi.org/10.1101/2024.10.10.617605](https://doi.org/10.1101/2024.10.10.617605)



# Install
Copy and paste the code below into a terminal window to download this package and install it in a conda env. It assumes conda is already installed. This should only take a few minutes total.

```bash
git clone https://github.com/ggit12/anndictionary
cd anndictionary
conda create -n anndict python=3.12
conda activate anndict
conda install -c conda-forge tbb numba
pip install -e .
```


# About
`AnnDictionary` is a package that lets you process multiple `anndata` objects in parallel with a simplified interface (so that you can avoid writing a bunch of for loops). This is accomplished by a dictionary-based wrapping of `scanpy`. We used the package to benchmark cell type annotaiton by 15 LLMs and maintain leaderboard at: https://singlecellgpt.com/celltype-annotation-leaderboard/.

## AI to make things easier
To make it easier to handle basics like cell type annotation for anndatas (one or many), and to make cell type labels agree across multiple anndatas, we provide AI-based processing to handle these tedious tasks. There are also AI-based functions to to biological process inference on gene lists.

These functions include `ai_annotate_cell_type`, `ai_annotate_biological_process`, and `ai_unify_labels`, which all operate over many adata at once.



This package supports many external LLM providers (including OpenAI, Anthropic, Google, and Bedrock). To use these, you'll need an API key. Directions on how to get an OpenAI API key can be found here: https://platform.openai.com/docs/quickstart/account-setup, and for Anthropic, here: https://docs.anthropic.com/en/api/getting-started.

### If you like functions:
The main function in this package is `adata_dict_fapply()` (and its cousin `adata_dict_fapply_return()`, which does the same thing but also returns the result as a dictionary). 

`adata_dict_fapply()` works just like `lapply()` in R. It takes `adata_dict_fapply(adata_dict, func, **kwargs)`. 

- `adata_dict` is a dictionary.
- `func` is a function to apply over each element of the dictionary.
- `**kwargs` are arguments that `func` takes.

You can have `func` take the argument `adt_key` (i.e., `func(adata, adt_key=None)`) if you want the dictionary key to be available within `func`.

The value for any **kwarg can be either: 1) a single value to be used for all anndata in adata_dict, or 2) a dictionary with the same keys as adata, and a separate value for each anndata in adata_dict.

Many functions in `anndict` are built around `adata_dict_fapply()`, and the package provides prebuilt wrappers for several common Scanpy functions, as well as functions to build and concatenate `adata` dictionaries.

### If you like objects (under development):
This package also defines the class AdataDict(), which is a dictionary of anndatas. When a method is called on an AdataDict, it is applied independently to each adata in the dictionary. Currently in beta.

The syntax looks like this: `adata_dict.fapply(func, **kwargs)`, where `adata_dict`, `func`, and `**kwargs` are as defined above.


Read the tutorial below for basic demonstrations.


## Compatibility

This package has been tested on linux (v3.10, v4.18) and macOS (v13.5, v14.7), and should work on most Unix-like operating systems. Although we haven’t formally tested it on Windows, we’re optimistic about compatibility and encourage you to reach out with any feedback or issues.

**macOS Compatibility Note:**

We configure the Numba threading layer to `tbb` to prevent concurrency issues caused by the default `workqueue` threading layer. This is automatically applied to ensure stable performance during multi-threading and parallel execution, and is done to ensure compatibility for users on macOS (especially Apple silicon).

If you encounter TBB threading layer errors, first run:
```bash
pip uninstall numba tbb intel-tbb
conda remove tbb numba
```

then reinstall `numba` and `tbb` with

```bash
conda install -c conda-forge tbb numba #need to conda install these, pip won't work
```


**How to Identify a Multithreading Issue:**

This issue typically manifests as a Jupyter kernel crash (or a Python crash with `numba` or `tbb` related errors, if running directly in Python). If you encounter these symptoms, they are likely related to the threading configuration.


# Tutorial
This is the tutorial notebook for `AnnDictionary`. Follow the tutorial below to get started.

See `tutorial_notbooks` for other tutorials:

- Basic Tutorial (learn the basic mechanics of this package)
- Label transfer with the Universal Cell Embedding (UCE)
- Automated spatial transcriptomic annotation with UCE


```python
import anndict as adt
import scanpy as sc
import time
import warnings
warnings.filterwarnings('ignore')
```

## Set up the LLM backend.
This package supports most LLMs including those offered by OpenAI, Anthropic, Google Gen AI, and Amazon Bedrock (if there's one you want that we don't support yet, let us know). Any of these can be configured or swapped with the single function call below to `adt.configure_llm_backend`. This function takes 3 required arguments, `provider`, `model`, and `api_key`, and then any provider-specific configurations as additional keyword arguments. See the documentation for examples of how to use this function with other providers.


```python
adt.configure_llm_backend(provider='anthropic',
                          model='claude-3-5-sonnet-20240620',
                          api_key='your-anthropic-api-key',
                          requests_per_minute=500
                          )
```


```python
#read data
adata_path = 'path-to-your-adata.h5ad'
adata = sc.read_h5ad(adata_path)
```


```python
#set X to be raw counts
adata.X = adata.layers['raw_counts'].copy()
```


```python
#get only protein coding genes

#load/define your list of protein-coding genes here, otherwise, annotationw will be based on all genes in object
protein_coding = None

if protein_coding:
    adata.var['protein_coding'] = [(i in protein_coding) for i in adata.var_names]
    # Subset to keep only protein-coding genes
    adata = adata[:, adata.var['protein_coding']].copy()
```

## Build the dictionary of anndatas.
The function `adt.build_adata_dict` will create separate anndatas based on the column names you provide. The following code creates a dictionary where each entry is the anndata for an individual tissue. You can also select only certain tissues by setting the argument `desired_strata`. See the Basic Tutorial notebook and function documentation for more examples and information on how to use this function.


```python
#build adata_dict
adata_dict = adt.build_adata_dict(adata, ['tissue'])
```


```python
#remove a standard list of uninformative genes
abundant_rnas = [
    "MALAT1",
    "NEAT1",
    "XIST",
    "KCNQ1OT1",
    "RPPH1",
    "RN7SL1",
    "RMRP",
    "SNHG1",
    "MIAT",
    "H19"
]

adt.remove_genes_adata_dict(adata_dict, abundant_rnas)
```

This section is just the standard Scanpy preprocessing pipeline, except here, we do it on each tissue independently and in parallel (by taking advantage of multithreading).


```python
#Run leiden clustering on each adata independently
#adata.X is raw counts, so run standard preprocessing
# Normalize each AnnData in the dictionary
adt.normalize_adata_dict(adata_dict)

# Log transform each AnnData in the dictionary
adt.log_transform_adata_dict(adata_dict)

# Optionally, you might subset the data to only high-variance genes
adt.set_high_variance_genes_adata_dict(adata_dict, n_top_genes=2000, subset=False)

# Scale each AnnData in the dictionary
adt.scale_adata_dict(adata_dict)

# Perform PCA on each AnnData in the dictionary
adt.pca_adata_dict(adata_dict, n_comps=50, mask_var='highly_variable')

#Calculate the neighborhood graph
adt.neighbors_adata_dict(adata_dict)

#Calculate the UMAP
adt.calculate_umap_adata_dict(adata_dict)
```


```python
#Determine appropriate cluster resolutions using AI. This function only works with LLMs that accept image inputs
#This will leave the final column as 'leiden' in the .obs of each anndata
# appropriate_resolution_dict = adt.ai_determine_leiden_resolution_adata_dict(adata_dict, initial_resolution=0.5)
```


```python
#recluster at 0.5 for this example
#get leiden clusters
adt.leiden_adata_dict(adata_dict, resolution=0.5)

# could also do 
# adt.leiden_adata_dict(adata_dict, resolution=appropriate_resolution_dict)
```


```python
#Run differential expression analysis independently on each anndata in adata_dict
adt.rank_genes_groups_adata_dict(adata_dict, groupby='leiden')
```

#### Now, run the LLM cell type annotation functions. Here's the rationale for this series of steps:
- First, Use an LLM to label each leiden cluster based on the top differentially expressed genes using `ai_annotate_cell_type_adata_dict`.
- Because each cluster is labelled independently, there might be some redundant labels with slight differences, for example ('Macropage' and 'macrophage.'). So, the next step is to use an LLM to merge these redundant category labels with `simplify_obs_column_adata_dict`.
- Finally, the dictionary of anndatas is merged into a single anndata. At this point, since each anndata has been processed independently, there might again be redundancies, which we remove (with an LLM) using `ensure_label_consistency_adata`. Note that this function can take a list of columns across which to unify labels (here we pass only a single column).


```python
#get the model name directly from the LLM config (just for naming the column in .obs)
model = adt.get_llm_config()['model']

#use an LLM to annotate celltypes based on the 'leiden' column, pass tissue information from 'tissue' column. The new label column will be as passed to label_column.
label_results = adt.ai_annotate_cell_type_adata_dict(adata_dict, groupby='leiden', n_top_genes=10, label_column=f'{model}_ai_cell_type', tissue_of_origin_col='tissue')

#These labels seem to have some redundancy, let's merge them with AI
ai_label_column = f'{model}_simplified_ai_cell_type'
simplified_mappings = adt.simplify_obs_column_adata_dict(adata_dict, f'{model}_ai_cell_type', ai_label_column, simplification_level='redundancy-removed')
```


```python
#Merge the adata_dict
adata = adt.concatenate_adata_dict(adata_dict)
```


```python
#unify the labels from the different adata in the adata_dict (i.e. use an LLM to merge categories like 'Macrophage' and 'macrophages.')
label_map_with_manual = adt.ensure_label_consistency_adata(adata, ai_label_column, simplification_level='unified', new_col_prefix='unified')
```


```python
#write the adata
path_to_write_adata = 'your-path-here.h5ad'
adata.write(path_to_write_adata)
```
