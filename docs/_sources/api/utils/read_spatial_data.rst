Read Spatial Data
===================

.. automodule:: anndict.utils.read_spatial_data
   :no-index:

.. currentmodule:: anndict.utils

Build anndata from Visium data
-------------------------------

.. autosummary::
   :toctree: generated

    build_adata_from_transcript_positions
    build_adata_from_visium

Add blank image to :class:`AnnData`
------------------------------------

.. autosummary::
   :toctree: generated

    add_blank_image_to_adata


Other functions used under-the-hood to construct :class:`Anndata` from spatial data
-------------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated

    read_transcript_coords
    get_steps_and_coords
    populate_sparse_array
    process_gene_counts
    create_anndata
