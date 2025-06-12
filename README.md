<div align="center"><img src="docs/_static/logo.svg" alt="AnnDictionary Logo" width="300px" /><br>A package for processing <code>anndata</code> objects in parallel with LLMs</div>

# Documentation

Complete documentation with tutorials is available at https://ggit12.github.io/anndictionary.


# Citation

If you use this package, please cite:

> #### Benchmarking Cell Type Annotation by Large Language Models with AnnDictionary  
> **George Crowley, Tabula Sapiens Consortium, Stephen R. Quake**  
> *bioRxiv* 2024.10.10.617605  
> [doi: https://doi.org/10.1101/2024.10.10.617605](https://doi.org/10.1101/2024.10.10.617605)



# Install
See [Installation Instructions](https://ggit12.github.io/anndictionary/installation.html)


# Tutorials
To get started, see [Tutorials](https://ggit12.github.io/anndictionary/tutorials/index.html) about how to use `AdataDict` and how to annotate cell types with LLMs. The runtimes on a desktop computer are no more than a few minutes.

- [Basics of AdataDict](https://ggit12.github.io/anndictionary/tutorials/adata_dict/index.html)
- [Automated LLM cell type annotation](https://ggit12.github.io/anndictionary/tutorials/annotate/index.html)
- [Automated spatial transcriptomic annotation](https://ggit12.github.io/anndictionary/tutorials/annotate/automated_spatial_transcriptomic_annotation.html)


# About
`AnnDictionary` is a package that lets you process multiple `anndata` objects in parallel with a simplified interface (so that you can avoid writing a bunch of for loops). This is accomplished by a dictionary-based wrapping of `scanpy`. We used the package to benchmark cell type annotaiton by 15 LLMs and maintain leaderboard at: https://singlecellgpt.com/celltype-annotation-leaderboard/.

## Use LLMs to simplify categorical label processing
We provide several LLM-based functions to handle tedious labeling tasks. These include cell type annotation based on differentially expressed genes with [`ai_annotate_cell_type()`](https://ggit12.github.io/anndictionary/api/annotate/cells/de_novo.html#annotation-by-marker-genes), and making cell type labels match across multiple anndata with [Automated Label Management](https://ggit12.github.io/anndictionary/api/automated_label_management/index.html). There are also AI-based functions to [annotate gene sets with biological processes](https://ggit12.github.io/anndictionary/api/annotate/cells/de_novo.html#annotate-groups-of-cells-by-biological-process).

This package supports many external LLM providers (including OpenAI, Anthropic, Google, and Bedrock). To use LLM features, you'll need an API key. Directions on how to get an OpenAI API key can be found here: https://platform.openai.com/docs/quickstart/account-setup, and for Anthropic, here: https://docs.anthropic.com/en/api/getting-started.

## Parallel processing of Anndata
This package defines the class `AdataDict`, which is a dictionary of `anndata`. There are several class methods to interact with `AdataDict` and iterate over them, see [Docs](https://ggit12.github.io/anndictionary/api/adata_dict/adata_dict.html) and [Tutorials](https://ggit12.github.io/anndictionary/tutorials/adata_dict/index.html). Additional methods and attributes are passed through to each anndata in `AdataDict`.

The core syntax for iterating a function `func` over an `AdataDict` called `adata_dict` looks like this:

    adata_dict.fapply(func, **kwargs)

where `adata_dict`, `func`, and `**kwargs` are as defined above.

`.fapply()` can also be called in a functional way using [`adata_dict_fapply()`](https://ggit12.github.io/anndictionary/api/adata_dict/generated/anndict.adata_dict.adata_dict_fapply.html#anndict.adata_dict.adata_dict_fapply):

    adata_dict_fapply(adata_dict, func, **kwargs)

In either case, `fapply()` works conceptually similar to `.map()` in python or `lapply()` in R. `fapply()` multithreads the iteration (multithreading can be turned off when needed), and uses smart argument broadcasting. This means that the value for any `**kwarg` can be either: 1) a single value to be used for all `anndata` in `adata_dict`, or 2) a `dictionary` with the same keys as `adata_dict`, and a separate value for each `anndata` in `adata_dict`.

Additionally, if you define `func` to take the argument `adt_key` (i.e., `func(adata, adt_key=None)`), `fapply` will make the respective key of `adata_dict` available to `func`.

## Compatibility

This package has been tested on linux (v3.10, v4.18) and macOS (v13.5, v14.7), and should work on most Unix-like operating systems. Although we haven’t formally tested it on Windows, we’re optimistic about compatibility and encourage you to reach out with any feedback or issues.

**macOS Compatibility Note:**

See [Install Instructions](https://ggit12.github.io/anndictionary/installation.html) if you have issues with multithreading on macOS (or others).

**How to Identify a Multithreading Issue:**

This issue typically manifests as a Jupyter kernel crash (or a Python crash with `numba` or `tbb` related errors, if running directly in Python). If you encounter these symptoms, they are likely related to the threading configuration.
