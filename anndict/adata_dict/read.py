"""
This module contains the functions necessary to read :class:`AdataDict` objects from adata on disk.
"""

import os
import json
from collections import Counter

import anndata as ad
import scanpy as sc

from .adata_dict import AdataDict, to_nested_tuple

def read_adata_dict(
    directory: str,
) -> AdataDict:
    """
    Read an :class:`AdataDict` from a previously saved :class:`AdataDict`. 
    To write an :class:`AdataDict` see :func:`~write_adata_dict`.

    Parameters
    -----------
    directory
        Base directory where the ``.h5ad`` files and hierarchy file are located.

    Returns
    --------
    An :class:`AdataDict` reconstructed from the saved files.

    Raises
    ------
    FileNotFoundError
        If the directory or required metadata files don't exist.
    ValueError
        If the metadata files are corrupted or in an unexpected format.

    Examples
    ---------
    **Case 1: Flat hierarchy**

    .. code-block:: bash

        directory/
        ├── adata_dict.hierarchy.json
        ├── adata_dict.db.json
        ├── <file_prefix>Donor1_Tissue1.h5ad
        ├── <file_prefix>Donor1_Tissue2.h5ad
        └── <file_prefix>Donor2_Tissue1.h5ad

    The reconstructed AdataDict will be:

    .. code-block:: python

        print(adata_dict)
        > {
        >     ("Donor1", "Tissue1"): adata_d1_t1,
        >     ("Donor1", "Tissue2"): adata_d1_t2,
        >     ("Donor2", "Tissue1"): adata_d2_t1,
        > }

    **Case 2: Nested hierarchy**

    .. code-block:: bash

        directory/
        ├── adata_dict.hierarchy.json
        ├── adata_dict.db.json
        ├── Donor1/
        │   ├── <file_prefix>Tissue1.h5ad
        │   └── <file_prefix>Tissue2.h5ad
        └── Donor2/
            └── <file_prefix>Tissue1.h5ad

    The reconstructed AdataDict will be:

    .. code-block:: python

        print(adata_dict)
        > {
        >     ("Donor1",): {
        >         ("Tissue1",): adata_d1_t1,
        >         ("Tissue2",): adata_d1_t2,
        >     },
        >     ("Donor2",): {
        >         ("Tissue1",): adata_d2_t1,
        >     },
        > }

    **Case 3: Nested hierarchy with multiple indices at the deepest level**

    .. code-block:: bash

        directory/
        ├── adata_dict.hierarchy.json
        ├── adata_dict.db.json
        ├── Donor1/
        │   ├── <file_prefix>Tissue1_CellType1.h5ad
        │   ├── <file_prefix>Tissue1_CellType2.h5ad
        │   └── <file_prefix>Tissue2_CellType3.h5ad
        └── Donor2/
            └── <file_prefix>Tissue1_CellType1.h5ad

    The reconstructed AdataDict will be:

    .. code-block:: python

        print(adata_dict)
        > {
        >     ("Donor1",): {
        >         ("Tissue1", "CellType1"): adata_d1_t1_c1,
        >         ("Tissue1", "CellType2"): adata_d1_t1_c2,
        >         ("Tissue2", "CellType3"): adata_d1_t2_c3,
        >     },
        >     ("Donor2",): {
        >         ("Tissue1", "CellType1"): adata_d2_t1_c1,
        >     },
        > }

    See Also
    ---------
    :func:`~write_adata_dict` : To write an :class:`AdataDict`
    """
    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # Read the hierarchy
    hierarchy_path = os.path.join(directory, "adata_dict.hierarchy.json")
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Hierarchy file not found in {directory}")

    with open(hierarchy_path, "r", encoding="utf-8") as f:
        hierarchy = json.load(f)

    # Read the database mapping files to keys
    db_path = os.path.join(directory, "adata_dict.db.json")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found in {directory}")

    with open(db_path, "r", encoding="utf-8") as f:
        db_entries = json.load(f)

    # Helper function to create nested dictionary from key path
    def create_nested_dict(data_dict: dict, key_path: tuple, value: any) -> None:
        """Create nested dictionary structure from a key path and value."""
        # For flat hierarchies, use the full key_path as a single tuple key
        if not any(isinstance(x, list) for x in hierarchy):
            data_dict[key_path] = value
            return

        # For nested hierarchies, we need to handle each level
        current = data_dict
        # Split the key_path according to the hierarchy structure
        current_pos = 0
        for level in hierarchy:
            if isinstance(level, list):
                # This level contains the remaining elements
                remaining_len = len(level)
                level_key = key_path[current_pos:current_pos + remaining_len]
                current[level_key] = value
                break

            # This level is a single element
            level_key = (key_path[current_pos],)
            if level_key not in current:
                current[level_key] = {}
            current = current[level_key]
            current_pos += 1

    # Initialize the data dictionary
    data_dict: dict = {}

    # Read each AnnData file and reconstruct the dictionary
    for entry in db_entries:
        file_path = os.path.join(directory, entry["file_path"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"AnnData file not found: {file_path}")

        key = tuple(entry["key"])  # Convert key from list to tuple
        adata = sc.read_h5ad(file_path)
        create_nested_dict(data_dict, key, adata)

    # Create and return the AdataDict
    return AdataDict(data_dict, hierarchy=to_nested_tuple(hierarchy))


def read_adata_dict_from_h5ad(
    paths: str | list[str],
    *,
    keys: list[str] | None = None,
) -> AdataDict:
    """
    Read ``.h5ad`` files from a list of paths and return them in a dictionary.

    For each element in the provided list of paths, if the element is a directory, 
    it reads all ``.h5ad`` files in that directory. If the element is an ``.h5ad`` file, 
    it reads the file directly.

    For auto-generated keys, if there are duplicate filenames, the function will 
    include parent directory names from right to left until keys are unique. 
    For example, ``dat/heart/fibroblast.h5ad`` would generate the key ``('heart', 'fibroblast')``
    if disambiguation is needed.

    Parameters
    ------------
    paths
        A string path or list of paths to directories or ``.h5ad`` files.

    keys
        A list of strings to use as keys for the adata_dict. If provided, must be equal 
        in length to the number of ``.h5ad`` files read.

    Returns
    -------
    An :class:`AdataDict`.
    """

    if isinstance(paths, str):
        paths = [paths]

    adata_dict = {}
    file_paths = []

    # First, collect all file paths recursively
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".h5ad"):
                        file_paths.append(os.path.join(root, file))
        elif path.endswith(".h5ad"):
            file_paths.append(path)

    # Check if provided keys match the number of files
    if keys is not None:
        if len(keys) != len(file_paths):
            raise ValueError(
                f"Number of provided keys ({len(keys)}) does not match the number of .h5ad files ({len(file_paths)})"
            )
        # Check for uniqueness in provided keys
        key_counts = Counter(keys)
        duplicates = [k for k, v in key_counts.items() if v > 1]
        if duplicates:
            raise ValueError(f"Duplicate keys found: {duplicates}")
        # Convert provided keys to tuples
        tuple_keys = [tuple(k) if isinstance(k, (list, tuple)) else (k,) for k in keys]
    else:
        # Generate keys from paths
        base_names = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]

        # Start with just the base names
        tuple_keys = [(name,) for name in base_names]

        # Keep extending paths to the left until all keys are unique
        while len(set(tuple_keys)) != len(tuple_keys):
            new_tuple_keys = []
            for i, file_path in enumerate(file_paths):
                path_parts = os.path.normpath(file_path).split(os.sep)
                # Find the current key's elements in the path
                current_key = tuple_keys[i]
                current_idx = (
                    len(path_parts) - 1 - len(current_key)
                )  # -1 for zero-based index
                # Add one more path element to the left if possible
                if current_idx > 0:
                    new_key = (path_parts[current_idx - 1],) + current_key
                else:
                    new_key = current_key
                new_tuple_keys.append(new_key)
            tuple_keys = new_tuple_keys

            # Safety check - if we've used all path components and still have duplicates
            if all(
                len(key) == len(os.path.normpath(fp).split(os.sep))
                for key, fp in zip(tuple_keys, file_paths)
            ):
                raise ValueError("Unable to create unique keys even using full paths")

    # Process the files with the finalized tuple keys
    for i, file_path in enumerate(file_paths):
        adata_dict[tuple_keys[i]] = ad.read_h5ad(file_path)

    return adata_dict
