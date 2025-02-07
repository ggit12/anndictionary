Overview
==========

Welcome to AnnDictionary
--------------------------

Convenient parallel processing of anndata.

Why we built it
-----------------

We wanted to make it easier to analyze atlas-scale single-cell RNA sequencing data.
When using scanpy and anndata, we found three friction points when scaling up analysis.

1. We frequently wanted to do a parallel analysis separately on each stratum of an adata, and found ourselves copy-and-pasting a lot of code to do this.

   - *Example: "Normalize each cell type separately, then look for differentially expressed genes in each, then plot."*

2. Cell type and biological process annotation.

   - *Example: "Label these cell types. Try 3 different strategies, and compare them all."*

3. Managing labels within across datasets.

   - *Example 1: "I downloaded data from 3 different groups, and they each used a different label for macrophages (i.e. macrophage, Macrophage., MΦ)"*
   - *Example 2: "I want to coarsen the cell type labels."*

So, we built a backend called `AnnDictionary` aimed at making these tasks easier.

Modules
---------

AnnDictionary is organized into the following modules:

1. **Core**: This module contains the AdataDict class and looping constructs.
2. **LLM Management**: This module handles LLM configuration and use, offering support for almost all mainstream LLM providers.
3. **Annotate**: A suite of functions to annotate cells and genes with different strategies, including de novo (with LLMs) and by label transfer.
4. **Plot**: Plotting functions (mostly ones that help annotation, some that help with gene list interpretation).
5. **Automated Label Management**: For dealing with annoying category label problems.
6. **Utility Functions**: Various utility and filtering functions.
7. **Wrappers**: Pre-wrapped functions from packages that are designed to work on single adata.

The public API (i.e., how you, the end user, should use this package) is documented in the :doc:`api/index` section of the docs.

The source code for all functions (public and not) is available on `GitHub <https://github.com/ggit12/anndictionary>`_.

Tutorials
-----------

We offer a number of tutorial notebooks to allow you to familiarize yourself with using AnnDictionaries and how to adapt/scale up existing code.  
We also offer guides on using AnnDictionary to annotate cells.

Contribute
------------

The contribute section describes the architecture of this package in more detail, including guides on how to extend the LLM support and offer wrappers for other packages.

Cite
------

If you use this package, please cite:

   .. line-block::

      **Benchmarking Cell Type Annotation by Large Language Models with AnnDictionary.**
      George Crowley, Tabula Sapiens Consortium, Stephen R. Quake.
      *bioRxiv* 2024.10.10.617605
      (https://doi.org/10.1101/2024.10.10.617605)
