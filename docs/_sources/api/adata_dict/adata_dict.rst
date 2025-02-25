AdataDict Class
~~~~~~~~~~~~~~~~~

.. automodule:: anndict.adata_dict.adata_dict
   :noindex:

.. currentmodule:: anndict

.. autoclass:: AdataDict
   :members: hierarchy


Manipulate the hierarchy
--------------------------

.. autosummary::
   :toctree: generated

   AdataDict.set_hierarchy
   AdataDict.add_stratification
   AdataDict.flatten

.. seealso::

   :func:`~anndict.adata_dict.add_stratification`
      The function underneath ``add_stratification`` that can be used to return a new object instead of modifying in place.


Iterate over :class:`AdataDict`
--------------------------------

.. autosummary::
   :toctree: generated

   AdataDict.fapply
   AdataDict.fapply_return

.. seealso::

   :func:`~anndict.adata_dict.adata_dict_fapply`
      The function underneath ``fapply`` that can be used separatley.


Index with a boolean mask
---------------------------

.. autosummary::
   :toctree: generated

   AdataDict.index_bool

.. seealso::

   :func:`~anndict.adata_dict.adata_dict_fapply`
      Use this to generate a boolean mask of the correct format.


Set ``.obs`` and ``.var`` ``index``
------------------------------------

.. autosummary::
   :toctree: generated

   AdataDict.set_obs_index
   AdataDict.set_var_index


Miscellaneous
--------------

.. autosummary::
   :toctree: generated

   AdataDict.check_structure
   AdataDict.copy

