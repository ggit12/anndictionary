"""
This module contains the functions write AdataDict objects to disk.
"""

import os
import json
import scanpy as sc

from anndata import AnnData

from .adata_dict import AdataDict

def write_adata_dict(
    adata_dict: AdataDict,
    directory: str,
    *,
    file_prefix: str = "",
) -> None:
    """
    Save each :class:`AnnData` object from an :class:`AdataDict` into a separate ``.h5ad`` file,
    creating a directory structure that reflects the hierarchy of the :class:`AdataDict`
    using key values as directory names. The hierarchy and structure are saved in files called
    ``adata_dict.hierarchy.json`` and ``adata_dict.db.json``, which are used internally by AnnDictionary.

    Parameters
    ----------
    adata_dict
        An :class:`AdataDict`.

    directory
        Base directory where ``.h5ad`` files will be saved.

    file_prefix
        Optional prefix for the filenames.

    Notes
    -----
    Each element in the key tuple becomes a subdirectory, ensuring the directory hierarchy 
    matches the AdataDict's hierarchy precisely. The filename itself is ``<file_prefix> + 
    key_elements_joined_by_underscores.h5ad``.

    Examples
    --------
    **Case 1: Flat hierarchy**

    .. code-block:: python

        adata_dict.set_hierarchy(["Donor", "Tissue"])
        print(adata_dict)
        > {
        >     ("Donor1", "Tissue1"): adata_d1_t1,
        >     ("Donor1", "Tissue2"): adata_d1_t2,
        >     ("Donor2", "Tissue1"): adata_d2_t1,
        > }

    The files will be saved as:

    .. code-block:: bash

        directory/
        ├── adata_dict.hierarchy.json
        ├── adata_dict.db.json
        ├── <file_prefix>Donor1_Tissue1.h5ad
        ├── <file_prefix>Donor1_Tissue2.h5ad
        └── <file_prefix>Donor2_Tissue1.h5ad

    **Case 2: Nested hierarchy**

    .. code-block:: python

        adata_dict.set_hierarchy(["Donor", ["Tissue"]])
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

    The files will be saved as:

    .. code-block:: bash

        directory/
        ├── adata_dict.hierarchy.json
        ├── adata_dict.db.json
        ├── Donor1/
        │   ├── <file_prefix>Tissue1.h5ad
        │   └── <file_prefix>Tissue2.h5ad
        └── Donor2/
            └── <file_prefix>Tissue1.h5ad

    **Case 3: Nested hierarchy with multiple indices at the deepest level**

    .. code-block:: python

        adata_dict.set_hierarchy(["Donor", ["Tissue", "Cell Type"]])
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

    The files will be saved as:

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
    """

    os.makedirs(directory, exist_ok=False)

    # Save the original hierarchy for complete reconstruction
    hierarchy_file_path = os.path.join(directory, "adata_dict.hierarchy.json")
    with open(hierarchy_file_path, "w", encoding="utf-8") as f:
        json.dump(adata_dict.hierarchy, f)

    db_file_path = os.path.join(directory, "adata_dict.db.json")
    db_entries = []

    def safe_str(val):
        s = str(val)
        return "".join(c if c not in r'\/:*?"<>|' else "_" for c in s)

    # Recursively traverse the AdataDict, yielding:
    #   full_key: tuple of all key elements from outer levels plus current level.
    #   file_key: tuple of key elements at the deepest level (used for file naming).
    #   value: the AnnData object.
    def iter_items(d, parent_key=()):
        for key_tuple, value in d.items():
            full_key = parent_key + key_tuple
            if isinstance(value, dict):
                yield from iter_items(value, full_key)
            elif isinstance(value, AnnData):
                yield full_key, key_tuple, value
            else:
                raise ValueError("Unexpected value type in adata_dict; expected AnnData or dict.")

    # Process each AnnData in the nested structure.
    for full_key, file_key, adata in iter_items(adata_dict):
        # The directory path is built from the outer (non-deepest) keys.
        dir_keys = full_key[:-len(file_key)]
        sub_dir = os.path.join(directory, *[safe_str(x) for x in dir_keys]) if dir_keys else directory
        os.makedirs(sub_dir, exist_ok=True)
        # The file name is built solely from the deepest-level key tuple.
        file_name = f"{file_prefix}{'_'.join(safe_str(x) for x in file_key)}.h5ad"
        file_path = os.path.join(sub_dir, file_name)
        sc.write(file_path, adata)
        rel_file_path = os.path.relpath(file_path, start=directory)
        db_entries.append({"file_path": rel_file_path, "key": list(full_key)})

    with open(db_file_path, "w", encoding="utf-8") as f:
        json.dump(db_entries, f)
