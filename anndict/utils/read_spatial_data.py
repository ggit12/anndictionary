"""
Functions to read spatial transcriptomic data.
"""

from typing import Literal

import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from anndata import AnnData


def read_transcript_coords(
    file_path: str,
    platform: Literal["Merscope", "Xenium"] = None
) -> pd.DataFrame:
    """
    Reads the transcript locations from a CSV file and ensures it contains the necessary columns.

    Parameters
    ----------
    file_path
        The path to the CSV file.

    platform
        The platform type.

    Returns
    -------
    A pandas DataFrame containing the transcript locations.

    Raises
    ------
    ValueError
        If the required columns are not present in the file or if the file format is unsupported.
    """
    print("reading data")

    if not file_path.endswith(".csv"):
        raise ValueError("Unsupported file format. Please provide a CSV file.")

    df = pd.read_csv(file_path)

    required_columns_merscope = {"global_x", "global_y", "gene"}
    required_columns_xenium = {"feature_name", "x_location", "y_location"}

    if platform == "Merscope":
        if not required_columns_merscope.issubset(df.columns):
            raise ValueError(
                f"The file must contain the following columns for Merscope: {required_columns_merscope}"
            )
    elif platform == "Xenium":
        if not required_columns_xenium.issubset(df.columns):
            raise ValueError(
                f"The file must contain the following columns for Xenium: {required_columns_xenium}"
            )
        df = df.rename(
            columns={
                "feature_name": "gene",
                "x_location": "global_x",
                "y_location": "global_y",
            }
        )
    else:
        raise ValueError(
            "Unsupported platform. Please provide either 'Merscope' or 'Xenium' as the platform."
        )

    return df


def get_steps_and_coords(
    df: pd.DataFrame,
    box_size: int,
    step_size: int
) -> tuple[int, int, list[list[int]]]:
    """
    Computes the number of steps and the top-left coordinates of each box.

    Parameters
    ----------
    df
        The data containing 'global_x' and 'global_y' columns.

    box_size
        The size of the box.

    step_size
        The step size.

    Returns
    -------
    A tuple containing the number of steps in x and y directions, and the list of top-left coordinates of each box.

    Raises
    ------
    ValueError
        If the box size is larger than the image dimensions.
    """
    print("getting steps and coords")
    min_x, max_x = df["global_x"].min(), df["global_x"].max()
    min_y, max_y = df["global_y"].min(), df["global_y"].max()

    x_steps = int((max_x - min_x - box_size) / step_size) + 2
    y_steps = int((max_y - min_y - box_size) / step_size) + 2

    if x_steps < 0:
        raise ValueError("box size is larger than image")

    coords_top_left = [
        [min_x + step_size * i, min_y + step_size * j]
        for i in range(0, x_steps)
        for j in range(0, y_steps)
    ]

    return x_steps, y_steps, coords_top_left


def populate_sparse_array(
    df: pd.DataFrame,
    genes: np.array,
    step_size: int
) -> csr_matrix:
    """
    Populates a sparse array with gene counts.

    Parameters
    ----------
    df
        The data containing 'global_x', 'global_y', and 'gene' columns.

    genes
        The unique genes.

    step_size
        The step size.

    Returns
    -------
    The sparse matrix with gene counts.
    """
    num_boxes_x = int((df["global_x"].max() - df["global_x"].min()) // step_size) + 1
    num_boxes_y = int((df["global_y"].max() - df["global_y"].min()) // step_size) + 1
    num_boxes = num_boxes_x * num_boxes_y

    sparse_array = dok_matrix((num_boxes, len(genes)), dtype=np.int32)
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}

    min_x, min_y = df["global_x"].min(), df["global_y"].min()
    df["box_x"] = ((df["global_x"] - min_x) // step_size).astype(int)
    df["box_y"] = ((df["global_y"] - min_y) // step_size).astype(int)

    for (box_x, box_y), box_df in df.groupby(["box_x", "box_y"]):
        index = box_x * num_boxes_y + box_y
        if 0 <= index < num_boxes:
            gene_counts = box_df["gene"].value_counts()
            for gene, count in gene_counts.items():
                if gene in gene_to_index:
                    sparse_array[index, gene_to_index[gene]] = count

    return sparse_array.tocsr()


def process_gene_counts(
    file_path: str,
    box_size: int,
    step_size: int,
    platform: Literal["Merscope", "Xenium"] = None,
) -> tuple[csr_matrix, np.array, list[list[int]]]:
    """
    Processes the gene counts from the CSV file.

    Parameters
    ----------
    file_path
        The path to the CSV file.

    box_size
        The size of the box.

    step_size
        The step size.

    platform
        The platform type.

    Returns
    -------
    A tuple containing the sparse matrix, unique genes, and list of top-left coordinates of each box.
    """
    df = read_transcript_coords(file_path, platform=platform)
    print("processing gene counts")
    genes = df["gene"].unique()
    _, _, coords_top_left = get_steps_and_coords(df, box_size, step_size)
    sparse_array = populate_sparse_array(df, genes, step_size)
    return sparse_array, genes, coords_top_left


def create_anndata(
    sparse_array: csr_matrix,
    genes: np.array,
    coords_top_left: list[list[int]]
) -> AnnData:
    """
    Creates an :class:`AnnData` object from the sparse matrix and coordinates.

    Parameters
    ----------
    sparse_array
        The sparse matrix with gene counts.

    genes
        The unique genes.

    coords_top_left
        The list of top-left coordinates of each box.

    Returns
    -------
    An :class:`AnnData` object containing the gene counts and metadata.
    """
    print("creating anndata")

    # Create the AnnData object, setting the sparse matrix as the X attribute
    adata = ad.AnnData(X=sparse_array.tocsr(), var={"gene_symbols": genes})

    # Set the gene symbols as the index of .var
    adata.var.index = adata.var["gene_symbols"]

    # Add the top-left coordinates of each box ("cell") as .obs
    adata.obs = pd.DataFrame(
        coords_top_left,
        columns=["global_x_topleft", "global_y_topleft"],
        index = [f"region_{i}" for i in range(len(coords_top_left))]
    )

    return adata


def add_blank_image_to_adata(
    adata: AnnData,
    platform: Literal["Merscope", "Xenium"] = "Merscope"
) -> AnnData:
    """
    Adds a dummy image to the AnnData object based on the platform specifications.

    Parameters
    ----------
    adata
        The :class:`AnnData` object to which the image will be added.

    platform
        The platform from which the data originates.

    Returns
    -------
    ``adata`` updated with a blank image and spatial data.
    """
    adata.obsm["spatial"] = adata.obs[
        ["global_x_topleft", "global_y_topleft"]
    ].to_numpy()

    max_x = int(adata.obs["global_x_topleft"].max())
    max_y = int(adata.obs["global_y_topleft"].max())

    dummy_image = np.ones((max_y + 1, max_x + 1, 3))

    if platform in ["Merscope", "Xenium"]:
        adata.uns["spatial"] = {
            "library_id": {
                "images": {"hires": dummy_image, "lowres": dummy_image},
                "scalefactors": {
                    "tissue_hires_scalef": 1.0,
                    "tissue_lowres_scalef": 1.0,
                    "spot_diameter_fullres": 16,
                },
            }
        }

    return adata


def build_adata_from_transcript_positions(
    paths_dict: dict[str, str],
    box_size: int = 16,
    step_size: int = 16,
    platform: Literal["Merscope", "Xenium"] = "Merscope",
) -> None:
    """
    Builds an :class:`AnnData` object from a ``detected_transcripts.csv`` (Merscope) or 
    ``transcripts.csv`` (Xenium) file and saves it to a specified output path. 
    These are the files output by most spatial transcriptomic platforms, 
    including Visium, Visium HD, Xenium, and Merscope.

    Parameters
    ----------
    paths_dict
        A dictionary with input paths as keys and output paths as values.

    box_size
        The size of the box.

    step_size
        The step size.

    platform
        The platform used, either "Merscope" (default) or "Xenium".

    Returns
    -------
    ``None``

    Notes
    -----
    Writes the processed :class:`AnnData` objects to the specified output paths.

    Example
    -------

    .. code-block:: python

        # Define input and output paths
        paths_dict = {
            'input_path1.csv': 'output_path1.h5ad',
            'input_path2.csv': 'output_path2.h5ad'
        }
        # Write AnnData objects to output paths
        build_adata_from_transcript_positions(paths_dict)

    """
    for input_path, output_path in paths_dict.items():
        sparse_array, genes, coords_top_left = process_gene_counts(
            input_path, box_size, step_size, platform=platform
        )
        adata = create_anndata(sparse_array, genes, coords_top_left)
        adata = add_blank_image_to_adata(adata, platform=platform)
        adata.write(output_path)


def build_adata_from_visium(
    paths_dict: dict[str, str],
    hd: bool = False
) -> None:
    """
    Processes Visium data from input directories and saves the processed AnnData objects to specified output paths.

    Parameters
    ----------
    paths_dict
        A dictionary with input directories as keys and output file paths as values.

    hd
        If ``True``, converts ``'spatial'`` to float and ``'obs'`` columns to integers.

    Returns
    -------
    None

    Notes
    -----
    Writes the processed :class:`AnnData` objects to the specified output paths.

    Example
    -------

    .. code-block:: python

        paths_dict = {
            'input_dir1': 'output_path1.h5ad',
            'input_dir2': 'output_path2.h5ad'
        }

        build_adata_from_visium(paths_dict, hd=True)
    """
    for input_dir, output_path in paths_dict.items():
        adata = sc.read_visium(input_dir)

        if hd:
            # Run fixes to sc.read_visium for visium HD data
            adata.obsm["spatial"] = adata.obsm["spatial"].astype(float)
            adata.obs = adata.obs.astype(int)

        adata.write(output_path)
