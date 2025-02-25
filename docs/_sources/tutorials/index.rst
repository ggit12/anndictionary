Tutorials
==========

List of Tutorials
---------------------------

.. toctree::
   :maxdepth: 2

   adata_dict/index
   annotate/index


Some background information
----------------------------

For reference, and to get you started, here's the rationale behind the package's core modules.

AdataDict
~~~~~~~~~~

When analyzing large amounts of single cell sequencing data, we frequently wanted to do things like:
- do the same analysis separately on each cell type
- then, separate by tissue and try again

Manually looping became too clunky, so we wrote this package to simplify the syntax. We also built in multithreading and smart argument broadcasting.

- See `AdataDict Core <https://ggit12.github.io/anndictionary/api/adata_dict/index.html>`_ documentation

Annotate
~~~~~~~~~~

When analyzing many **anndata**, there's the additional bottleneck of cell type annotation. So, we built a suite of functions to annotate cell types with LLMs, with label transfer models, etc., and functions to benchmark and compare results.

- See `Annotate Module <https://ggit12.github.io/anndictionary/api/annotate/index.html>`_ documentation

Automated Label Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, there's a major friction point--managing all these labels (for example, cell type labels that differ only typographically). So, we built functions that use LLMs to automatically clean categorical columns.

- See `Automated Label Management <https://ggit12.github.io/anndictionary/api/automated_label_management/index.html>`_ documentation
