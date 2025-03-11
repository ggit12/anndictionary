"""
This module contains the :class:`AdataDict` class, which is basically a nested dictionary of anndata, with a 
few extra features to help restructure the nesting hierarchy and iterate over it. :class:`AdataDict` inherits from :class:`dict` and passes methods through to each :class:`AnnData` in the :class:`AdataDict`.
"""

from __future__ import annotations

from functools import wraps

from anndata import AnnData

from .adata_dict_utils import to_nested_tuple, to_nested_list, set_var_index_func, set_obs_index_func
from .adata_dict_fapply import adata_dict_fapply, adata_dict_fapply_return
from .add_stratification import add_stratification as add_stratification_func

from .dict_utils import check_dict_structure, all_leaves_are_of_type

type NestedList = str | list["NestedList"]
type NestedTuple = str | tuple["NestedTuple", ...]


class AdataDict(dict):
    """
    :class:`AdataDict` is a dictionary-like container where values are :class:`AnnData` objects. :class:`AdataDict` inherits from :class:`dict`.

    This class provides three main functionalities:

    1. It has the ``set_hierarchy`` method to restructure the nesting hierarchy, and the ``hierarchy`` attribute to keep track.
    2. It behaves like an ``AnnData`` object by passing methods through to each ``AnnData`` in the dictionary.
    3. It has methods ``fapply(func, kwargs)`` and ``fapply_return(func, kwargs)`` that apply a given function ``func`` with arguments ``kwargs`` to each ``AnnData`` object in the :class:`AdataDict`.

    Parameters
    -----------
    data
        Dictionary with keys as tuples of indices.

    hierarchy
        Tuple or list indicating the order of indices in the keys of ``data``.

    """
    # See Also
    # --------

    # :func:`adata_dict_fapply` : The function underneath ``fapply`` that can be used separatley.
    # :func:`adata_dict_fapply_return` : The function underneath ``fapply_return`` that can be used separatley.

    def __init__(
    self,
    data: dict[tuple[int, ...], any] | None = None,
    hierarchy: tuple | list | None = None,
    ) -> None:
        """
        Initialize the ``AdataDict`` with data and hierarchy.

        Parameters
        ------------
        data
            Dictionary with keys as tuples of indices.

        hierarchy
            Tuple or list indicating the order of indices.

        Returns
        -------
        None
            Initializes the ``AdataDict`` object.
        """
        if data is None:
            data = {}
        super().__init__(data)  # Initialize the dict with data
        if hierarchy is not None:
            self._hierarchy = tuple(hierarchy)  # Tuple indicating the index hierarchy
        else:
            self._hierarchy = ()

    def copy(self) -> AdataDict:
        """
        Copy the ``AdataDict``. Creates a deep copy of each ``AnnData`` object using :func:`anndata.AnnData.copy`.

        Returns
        -------
        A new ``AdataDict`` with copied ``AnnData`` objects.

        Examples
        ---------
        .. code-block:: python

            adata_dict_copy = adata_dict.copy()

        """
        def copy_adata(adata):
            return adata.copy()
        return self.fapply(copy_adata, return_as_adata_dict=True)

    @property
    def hierarchy(self):
        """
        The hierarchy of the ``AdataDict``.

        This attribute is accessed as: ``adata_dict.hierarchy``.

        Returns
        --------
        The current hierarchy of the ``AdataDict`` as a tuple.

        Examples
        ---------

        .. code-block:: python

            adata_dict.hierarchy
            > ('donor', ('tissue'))
        """
        return self._hierarchy

    def _flatten(self, parent_key=()):
        flat_data = {}
        for key, value in self.items():
            full_key = parent_key + key
            if isinstance(value, AdataDict):
                flat_data.update(value._flatten(parent_key=full_key)) # pylint: disable=protected-access
            else:
                flat_data[full_key] = value
        return flat_data

    def flatten_nesting_list(self,
    nesting_list: list | tuple
    ) -> list:
        """
        Flatten a nested list or tuple into a single list.

        Parameters
        -----------
        nesting_list
            Nested list or tuple of hierarchy levels.

        Returns
        ---------
        Flattened list of hierarchy elements.
        """
        hierarchy = []
        for item in nesting_list:
            if isinstance(item, (list, tuple)):
                hierarchy.extend(self.flatten_nesting_list(item))
            else:
                hierarchy.append(item)
        return hierarchy

    def get_levels(self, nesting_list, levels=None, depth=0):
        """
        Get the levels of hierarchy based on the nesting structure.

        :param nesting_list: Nested list indicating the new hierarchy structure.
        :param levels: List to store the levels.
        :param depth: Current depth in recursion.
        :return: List of levels with hierarchy elements.
        """
        if levels is None:
            levels = []
        if len(levels) <= depth:
            levels.append([])
        for item in nesting_list:
            if isinstance(item, list):
                self.get_levels(item, levels, depth + 1)
            else:
                levels[depth].append(item)
        return levels

    def set_hierarchy(self,
    nesting_list: NestedList | NestedTuple
    ):
        """
        Rearrange the hierarchy of :class:`AdataDict` based on the provided nesting structure.

        Parameters
        ------------
        nesting_list
            Nested list indicating the new hierarchy structure.

        Notes
        -------
        Supports input as a nested list or nested tuple. Recommended to supply as a nested list. 
        Nested tupple support is supplied for ease of use when caching, see **Case 4: Caching the hierarchy** below.

        Examples
        ---------
        **Case 1: Flat hierarchy**

        .. code-block:: python

            adata_dict.set_hierarchy(["Donor", "Tissue"])
            print(adata_dict)
            > {
            >     ("Donor1", "Tissue1"): adata1,
            >     ("Donor1", "Tissue2"): adata2,
            >     ("Donor2", "Tissue1"): adata3,
            > }

        **Case 2: Nested hierarchy**

        .. code-block:: python

            adata_dict.set_hierarchy(["Donor", ["Tissue"]])  # Note the nested list here
            print(adata_dict)
            > {
            >     ("Donor1",): {
            >         ("Tissue1",): adata1,
            >         ("Tissue2",): adata2,
            >     },
            >     ("Donor2",): {
            >         ("Tissue1",): adata3,
            >     },
            > }

        **Case 3: Complex nested hierarchy**

        .. code-block:: python

            adata_dict.set_hierarchy(["Donor", ["Tissue", "Cell Type"]])  # Note the nested list here
            print(adata_dict)
            > {
            >     ("Donor1",): {
            >         ("Tissue1", "CellType1"): adata1,
            >         ("Tissue1", "CellType2"): adata2,
            >         ("Tissue2", "CellType3"): adata3,
            >     },
            >     ("Donor2",): {
            >         ("Tissue1", "CellType1"): adata4,
            >     },
            > }

        **Case 4: Caching the hierarchy**

        .. code-block:: python

            # Initial Structure
            print(adata_dict) # Is nested
            > {
            >     ("Donor1",): {
            >         ("Tissue1",): adata1,
            >         ("Tissue2",): adata2,
            >     },
            >     ("Donor2",): {
            >         ("Tissue1",): adata3,
            >     },
            > }

            # Cache the hierarchy
            cached_hierarchy = adata_dict.hierarchy

            # Change the hierarchy (flatten for this example)
            adata_dict.flatten()
            print(adata_dict) # Is now flat
            > {
            >     ("Donor1", "Tissue1"): adata1,
            >     ("Donor1", "Tissue2"): adata2,
            >     ("Donor2", "Tissue1"): adata3,
            > }

            #Restore the hierarchy
            adata_dict.set_hierarchy(cached_hierarchy)
            print(adata_dict) # Is nested again
            > {
            >     ("Donor1",): {
            >         ("Tissue1",): adata1,
            >         ("Tissue2",): adata2,
            >     },
            >     ("Donor2",): {
            >         ("Tissue1",): adata3,
            >     },
            > }

        """
        # Convert nesting_list to list if tuple for ease of use when caching
        if isinstance(nesting_list, tuple):
            nesting_list = to_nested_list(nesting_list)

        # Flatten the nested data
        flat_data = self._flatten()
        self.clear()
        self.update(flat_data)

        # Flatten and extract the current hierarchy
        self._hierarchy = tuple(self.flatten_nesting_list(self._hierarchy))

        # Flatten the new hierarchy
        new_hierarchy = self.flatten_nesting_list(nesting_list)

        # Get the levels of the nesting structure
        levels = self.get_levels(nesting_list)
        old_hierarchy = self._hierarchy

        # Function to recursively create nested AdataDicts
        def create_nested_adata_dict(current_level, key_indices, value, level_idx):
            if level_idx == len(levels):
                return value  # Base case: return the value (AnnData object)
            level = levels[level_idx]
            level_length = len(level)
            level_key = tuple(key_indices[:level_length])
            remaining_key_indices = key_indices[level_length:]
            if level_key not in current_level:
                # Remaining hierarchy for nested AdataDict
                remaining_hierarchy = levels[level_idx + 1 :] if level_idx + 1 < len(levels) else []
                current_level[level_key] = AdataDict(hierarchy=remaining_hierarchy)
            # Recurse into the next level
            nested_dict = current_level[level_key]
            nested_value = create_nested_adata_dict(nested_dict, remaining_key_indices, value, level_idx + 1)
            if level_idx == len(levels) - 1:
                # At the last level, set the value
                current_level[level_key] = nested_value
            return current_level

        # Start building the new nested AdataDict
        new_data = AdataDict(hierarchy=new_hierarchy)
        for key, value in flat_data.items():
            # Map old indices to their values
            index_map = dict(zip(old_hierarchy, key))
            # Arrange indices according to the new hierarchy
            new_key_indices = [index_map[h] for h in new_hierarchy]
            # Recursively build the nested structure
            create_nested_adata_dict(new_data, new_key_indices, value, 0)

        # Update the hierarchy and data
        self._hierarchy = to_nested_tuple(nesting_list)  # Update with the nested structure
        # Replace the existing data in self with new_data
        self.clear()
        self.update(new_data)

    def flatten(self):
        """
        Flatten the hierarchy of :class:`AdataDict`.

        Examples
        ---------

        **Case 1: Nested hierarchy**

        .. code-block:: python

            #start with a nested hierarchy
            adata_dict.hierarchy
            > ('donor', ('tissue'))
            print(adata_dict)
            > {
            >     ("Donor1",): {
            >         ("Tissue1",): adata1,
            >         ("Tissue2",): adata2,
            >     },
            >     ("Donor2",): {
            >         ("Tissue1",): adata3,
            >     },
            > }

            #flatten adata_dict
            adata_dict.flatten()
            addata_dict.hierarchy
            > ('donor', 'tissue') #now the hierarchy is flat
            print(adata_dict)
            > {
            >     ("Donor1", "Tissue1"): adata1,
            >     ("Donor1", "Tissue2"): adata2,
            >     ("Donor2", "Tissue1"): adata3,
            > }
        """
        # Flatten the adata_dict
        flat_hierarchy = self.flatten_nesting_list(self.hierarchy)
        self.set_hierarchy(flat_hierarchy)

    def __getitem__(self, key):
        # Simplify access by converting non-tuple keys to tuple
        if not isinstance(key, tuple):
            key = (key,)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        # Simplify setting by converting non-tuple keys to tuple
        if not isinstance(key, tuple):
            key = (key,)
        super().__setitem__(key, value)

    def __getattr__(self, attr):
        # First check if this is a property of AnnData
        if hasattr(AnnData, attr):
            anndata_attr = getattr(AnnData, attr)
            if isinstance(anndata_attr, property):
                # Handle property access
                return self._handle_property_access(attr)

        # If not a property, handle as a method
        return self._handle_method_access(attr)

    def _handle_property_access(self, attr):
        """Handle property access by returning a dictionary of property values."""
        results = {}
        for key, adata in self.items():
            if isinstance(adata, AdataDict):
                # Recurse into nested AdataDict
                results[key] = getattr(adata, attr)
            else:
                # Get property value directly
                results[key] = getattr(adata, attr)
        return results

    def _handle_method_access(self, attr):
        """Handle method access by returning a wrapper function."""
        def method(*args, **kwargs):
            results = {}
            for key, adata in self.items():
                if isinstance(adata, AdataDict):
                    # Recurse into nested AdataDict
                    results[key] = getattr(adata, attr)(*args, **kwargs)
                else:
                    func = getattr(adata, attr)
                    results[key] = func(*args, **kwargs)
            return results
        return method

    # @wraps(check_dict_structure)
    def check_structure(
        self,
        input_dict: dict
    ) -> bool:
        """
        Check if an ``input_dict`` has the same structure and keys as the ``AdataDict``.
        Raises an error at the first missing key encountered.
        If no errors are raised, the structures match.

        Parameters
        ------------
        input_dict
            A dictionary to be compared with the ``AdataDict``.

        Returns
        -------
        ``True`` if the structures match (based on both keys and nesting), ``False`` otherwise.
        """
        return check_dict_structure(self, input_dict)

    @wraps(adata_dict_fapply)
    def fapply(self, func, *, use_multithreading=True, num_workers=None, max_retries=0, **kwargs_dicts):
        """Wrapper for adata_dict_fapply."""
        return adata_dict_fapply(
            self,
            func,
            use_multithreading=use_multithreading,
            num_workers=num_workers,
            max_retries=max_retries,
            **kwargs_dicts,
        )

    @wraps(adata_dict_fapply_return)
    def fapply_return(self, func, *, use_multithreading=True, num_workers=None, max_retries=0, return_as_adata_dict=False, **kwargs_dicts):
        """Wrapper for adata_dict_fapply_return."""
        return adata_dict_fapply_return(
            self,
            func,
            use_multithreading=use_multithreading,
            num_workers=num_workers,
            max_retries=max_retries,
            return_as_adata_dict=return_as_adata_dict,
            **kwargs_dicts,
        )

    @wraps(set_var_index_func)
    def set_var_index(self, cols: str | list[str]):
        """Wrapper for set_var_index_func."""
        return set_var_index_func(self, cols)

    @wraps(set_obs_index_func)
    def set_obs_index(self, cols: str | list[str]):
        """Wrapper for set_obs_index_func."""
        return set_obs_index_func(self, cols)

    @wraps(add_stratification_func)
    def add_stratification(self, strata_keys, *, desired_strata=None):
        """
        Wrapper for :func:`add_stratification` that modifies self in-place``.
        """
        adata_dict_with_new_stratification = add_stratification_func(self, strata_keys=strata_keys, desired_strata=desired_strata)
        self.clear()
        self.update(adata_dict_with_new_stratification)
        self._hierarchy = adata_dict_with_new_stratification._hierarchy # pylint: disable=protected-access

    # This function either modifies in place or returns, so disable this check
    # pylint: disable=inconsistent-return-statements
    def index_bool(self, index_dict: dict, inplace: bool = True) -> None | AdataDict:
        """
        Index the ``AdataDict`` by a dictionary of indices.

        Parameters
        -----------
        index_dict
            A :class:`dict` with the same structure as the ``AdataDict``, but values are boolean.

        inplace
            If ``True``, modifies the ``AdataDict`` in-place. Otherwise, returns a new ``AdataDict``.

        Returns
        --------
        - ``None`` and modifies the ``AdataDict`` in-place if ``inplace=True``.
        - New ``AdataDict`` object with the specified indices if ``inplace=False``.
        """

        # Check if index_dict has the same structure as self
        if not self.check_structure(index_dict):
            raise ValueError("``index_dict`` must have the same structure and keys as the ``AdataDict``")

        # Enforce that all elements in index_dict are boolean
        if not all_leaves_are_of_type(index_dict, bool):
            raise ValueError("All leaf values in ``index_dict`` must be boolean")

        # Use a new copy when not inplace
        adata_dict_to_change = self if inplace else self.copy()

        keys_to_remove = []
        def collect_keys(adata, bool_index=None, adt_key=None): # pylint: disable=unused-argument # ``adata`` arg must be included to comply with fapply syntax
            """
            Delete entries from adata based on ``bool_index``.
            """
            if not bool_index:
                keys_to_remove.append(adt_key)

        # Collect keys to remove
        adata_dict_to_change.fapply(collect_keys, bool_index=index_dict)

        # Remove the keys (can't be done with fapply)
        for key in keys_to_remove:
            del adata_dict_to_change[key]


        # Return the modified AdataDict if not inplace
        if not inplace:
            return adata_dict_to_change
