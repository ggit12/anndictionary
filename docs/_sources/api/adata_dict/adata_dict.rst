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
   AdataDict.flatten

Iterate over :class:`AdataDict`
--------------------------------

.. autosummary::
   :toctree: generated

   AdataDict.fapply
   AdataDict.fapply_return

.. seealso::

   :func:`~anndict.adata_dict.adata_dict_fapply`
      The function underneath ``fapply`` that can be used separatley.

   :func:`~anndict.adata_dict.adata_dict_fapply_return`
      The function underneath ``fapply_return`` that can be used separatley.

Set ``.obs`` and ``.var`` ``index``
------------------------------------

.. autosummary::
   :toctree: generated

   AdataDict.set_obs_index
   AdataDict.set_var_index

.. .. autoclass:: AdataDict
..    :members:
..    :exclude-members: flatten_nesting_list, get_levels