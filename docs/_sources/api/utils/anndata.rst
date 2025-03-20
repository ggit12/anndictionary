AnnData
========

.. automodule:: anndict.utils.anndata_
   :no-index:

.. currentmodule:: anndict.utils

Get column names from ``adata.obs`` with string matching
---------------------------------------------------------

.. autosummary::
   :toctree: generated

   get_adata_columns

Remove genes
-------------

.. autosummary::
   :toctree: generated

   remove_genes

Add metadata
--------------
.. autosummary::
   :toctree: generated

   add_col_to_adata_obs
   add_col_to_adata_var
   add_col_to_pd_df


Convert column dtypes
----------------------
.. autosummary::
   :toctree: generated

   convert_obs_col_to_category
   convert_obs_col_to_string
   convert_obs_index_to_str

Filter a list of genes based on their presence in ``adata.var.index``
------------------------------------------------------------------
.. autosummary::
   :toctree: generated

   filter_gene_list
