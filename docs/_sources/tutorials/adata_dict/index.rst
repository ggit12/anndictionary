Using the AdataDict Class
==========================

These notebooks covers the basic mechanics around working with `AdataDict`s.  

This is the core class implemented in `AnnDictionary`,
a package for processing multiple `anndata` objects in parallel with LLMs.

These tutorials demonstrates how to:
   - `build <https://ggit12.github.io/anndictionary/api/adata_dict/generated/anndict.adata_dict.build_adata_dict.html#anndict.adata_dict.build_adata_dict>`_ an `AdataDict`
   - `manipulate <https://ggit12.github.io/anndictionary/api/adata_dict/adata_dict.html>`_ their structure (i.e. split by cell type, group by tissue, etc)
   - `iterate <https://ggit12.github.io/anndictionary/api/adata_dict/generated/anndict.AdataDict.fapply.html#anndict.AdataDict.fapply>`_ over their elements
   - `read <https://ggit12.github.io/anndictionary/api/adata_dict/generated/anndict.adata_dict.write_adata_dict.html#anndict.adata_dict.read_adata_dict>`_ / `write <https://ggit12.github.io/anndictionary/api/adata_dict/generated/anndict.adata_dict.write_adata_dict.html#anndict.adata_dict.write_adata_dict>`_ them from/to disk


.. toctree::
   :maxdepth: 1

   Build_AdataDict.ipynb
   Manipulate_the_Hierarchy.ipynb
   Iterate_Over_an_AdataDict.ipynb
   Write_and_Read_an_AdataDict.ipynb
