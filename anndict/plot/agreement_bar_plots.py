"""
Make (grouped) barplots of agreement scores between columns in ``adata.obs``.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn.matrix import ClusterGrid
from seaborn import FacetGrid
from anndata import AnnData


def plot_grouped_average(
    adata: AnnData,
    label_value: dict[str,str],
    adt_key: tuple[str,...] | None = None
):
    """
    Plots the average values specified in ``label_value`` 
    across each group of label_keys in an AnnData object.

    Parameters
    -----------
    adata: 
        An :class:`AnnData`.

    label_value: 
        A :class:`dict` where the keys are the cols in ``adata.obs`` 
        for grouping, values are the cols in ``adata.obs`` for 
        the values to average.

    adt_key: 
        to print specified key

    Returns
    --------
    None
    """
    print(adt_key)
    if not all(label in adata.obs for label in label_value.keys()):
        missing_keys = [label for label in label_value.keys() if label not in adata.obs]
        raise ValueError(f"Label key(s) {missing_keys} not found in adata.obs.")
    if not all(value in adata.obs for value in label_value.values()):
        missing_values = [value for value in label_value.values() if value not in adata.obs]
        raise ValueError(f"Value key(s) {missing_values} not found in adata.obs.")

    grouped_means = {}
    for label, value in label_value.items():
        grouped_means[label] = adata.obs.groupby(label)[value].mean()

    # Create a DataFrame from the grouped means
    df = pd.DataFrame(grouped_means)

    # Plot the results
    df.plot(kind='bar', figsize=(12, 8), color=plt.colormaps["Paired"].colors)
    plt.xlabel('Groups')
    plt.ylabel('Average Scores')
    plt.title('Average Scores across Groups')
    plt.xticks(rotation=90)
    plt.legend(title='Scores')
    plt.show()


def plot_model_agreement(
    adata: AnnData,
    group_by: str,
    sub_group_by: str,
    agreement_cols: list[str],
    granularity: int = 2
) -> tuple[Figure, Axes] | ClusterGrid:
    """
    Plots the average values of specified agreement columns across varying levels of granularity. 
    See notes for which columns are ignored based on the granularity setting.

    Parameters
    -----------
    adata
        an :class:`AnnData`.

    group_by
        Column name in ``adata.obs`` for the main grouping (e.g., ``'cell_type'``).

    sub_group_by
        Column name in ``adata.obs`` for the sub-grouping (e.g., ``'tissue'``).

    agreement_cols
        Column names specifying the agreement columns (e.g., ``['agreement_of_manual_with_model1', 'agreement_of_manual_with_model2']``).

    granularity
        Level of detail in the plot (``0`` = models only, ``1`` = models within cell types, ``2`` = models within cell types and tissues).

    Returns
    --------
    The plot of agreement, averaged according to the ``granularity`` setting.

    Notes
    ------
    If granularity is 0, ``group_by`` and ``sub_group_by`` are not used. 
    If granularity is 1, ``sub_group_by`` is not used.

    Examples
    ---------

    .. code-block:: python
    
        import anndict as adt
        adt.plot_model_agreement(adata, 'cell_type', 'tissue', ['agreement_of_manual_with_model1', 'agreement_of_manual_with_model2'], granularity=0)

    """
    if not all(col in adata.obs for col in agreement_cols):
        missing_cols = [col for col in agreement_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Pivot longer to get columns: group_by, sub_group_by, agreement, model_name
    melted = adata.obs.melt(id_vars=[group_by, sub_group_by], value_vars=agreement_cols,
                            var_name='model_name', value_name='agreement')

    if granularity == 0:
        # Calculate the average scores across all groups within each model
        grouped_means = melted.groupby('model_name')['agreement'].mean().sort_values(ascending=False)

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the bar chart
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')

        # Add value labels on top of each bar
        for i, v in enumerate(grouped_means):
            ax.text(i, v, f'{v * 100:.0f}%', ha='center', va='bottom')

    elif granularity == 1:
        # Calculate the average scores within each model and cell type
        grouped_means = melted.groupby([group_by, 'model_name'])['agreement'].mean().unstack()

        fig, ax = plt.subplots(figsize=(14, 8))
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')

    elif granularity == 2:
        # Calculate the average scores within each model, cell type, and tissue
        grouped_means = melted.groupby([group_by, sub_group_by, 'model_name'])['agreement'].mean().unstack(level=[1, 2])

        # Ensure the data is numeric and allow NaNs (missing values)
        grouped_means = grouped_means.apply(pd.to_numeric, errors='coerce')

        # Create a mask for NaN values
        mask = grouped_means.isnull()

        # Create a color mapping for tissues using the provided colors
        tissue_colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
            "#7f9ec0", "#ffab60", "#5ab4ac",
            # 5 additional concordant colors
            "#8b4513", "#ff6347", "#4682b4", "#dda0dd", "#ffd700"
        ]

        # Ensure that the number of tissues does not exceed the number of available colors
        tissues = grouped_means.columns.get_level_values(0).unique()
        tissue_colors = tissue_colors[:len(tissues)]

        # Create a color map based on the provided colors
        tissue_color_map = dict(zip(tissues, tissue_colors))

        # Create column colors based on tissues
        col_colors = [tissue_color_map[tissue] for tissue in grouped_means.columns.get_level_values(0)]

        # Plot heatmap with col_colors
        # fig, ax = plt.subplots(figsize=(16, 10))
        # Create the clustermapimport seaborn as sns

        # Use the 'viridis_r' colormap
        cmap = plt.colormaps['viridis_r']

        # Set the color for NaN values (e.g., red)
        cmap.set_bad(color='black')

        # Create the clustermap with horizontal lines
        g = sns.clustermap(grouped_means, cmap=cmap, annot=False,
                        mask=mask, cbar_kws={'label': 'Agreement'},
                        linewidths=0, linecolor='black',
                        col_colors=col_colors, col_cluster=False, row_cluster=False,
                        yticklabels=1)

        # Get the axes object
        ax = g.ax_heatmap

        # # Remove all existing lines
        # ax.grid(False)

        # Add back only horizontal lines
        # ax.set_xticks(np.arange(grouped_means.shape[1]+1)-0.5, minor=True)
        # ax.set_yticks(np.arange(grouped_means.shape[0]+1)-0.5, minor=True)
        # ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        # ax.tick_params(which="minor", bottom=False, left=False)

        # Find where col_colors change
        color_changes = []
        for i in range(1, len(col_colors)):
            if col_colors[i] != col_colors[i-1]:
                color_changes.append(i)

        # Add vertical lines at color change positions
        for pos in color_changes:
            ax.axvline(pos, color='black', linewidth=0.5)

        return g

        # Create a legend for tissues
        # tissue_handles = [plt.Rectangle((0,0),1,1, color=color) for color in tissue_color_map.values()]
        # ax.legend(tissue_handles, tissue_color_map.keys(), title=sub_group_by,
        #           loc='center left', bbox_to_anchor=(1, 0.5))
        # return fig, ax

    else:
        raise ValueError("Granularity must be 0, 1, or 2.")

    if granularity < 2:
        ax = plt.gca()  # Get current axis for granularity 0 and 1

    ax.set_xlabel(group_by if granularity > 0 else 'Model')
    ax.set_ylabel('Agreement')
    title = 'Average model agreement'
    if granularity == 0:
        title += ''
    elif granularity == 1:
        title += f' by {group_by}'
    elif granularity == 2:
        title += f' by {group_by} and {sub_group_by}'
    ax.set_title(title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    if granularity < 2:
        ax.legend(title='Models' + ('' if granularity == 0 else ' and Tissues'))

    plt.tight_layout()
    # Return the figure and axis for further editing
    return fig, ax


def plot_model_agreement_categorical(
    adata: AnnData,
    group_by: str,
    sub_group_by: str,
    agreement_cols: list[str],
    granularity: int = 2
) -> tuple[Figure, Axes] | FacetGrid | Figure:
    """
    Plots the relative proportions of categories within specified model columns across varying levels of granularity.

    Parameters
    ------------
    adata
        An :class:`AnnData`.

    group_by
        key in ``adata.obs`` for the main grouping (e.g., ``'cell_type'``).

    sub_group_by
        key in ``adata.obs`` for the sub-grouping (e.g., ``'tissue'``).

    agreement_cols
        Column names specifying the agreement columns (e.g., ``['categorical_agreement_of_manual_with_model1', 'categorical_agreement_of_manual_with_model2']``). These should be categorical.

    granularity
        level of detail in the plot (``0`` = models only, ``1`` = models within cell types, ``2`` = models within cell types and tissues).

    Returns
    --------
    The plot of agreement, averaged according to the ``granularity`` setting, split by quality of agreement.

    Notes
    ------
    If granularity is 0, ``group_by`` and ``sub_group_by`` are not used. 
    If granularity is 1, ``sub_group_by`` is not used.

    Examples
    ---------

    .. code-block:: python

        import anndict as adt
        adt.plot_model_agreement_categorical(adata, 'cell_type', 'tissue', ['agreement_of_manual_with_model1', 'agreement_of_manual_with_model2'], granularity=0)
    """
    # Verify that the required columns exist
    if not all(col in adata.obs for col in agreement_cols):
        missing_cols = [col for col in agreement_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Ensure that agreement_cols are categorical or convert numeric types to categories
    for col in agreement_cols:
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype('category')
            else:
                raise ValueError(f"Column '{col}' must be categorical or convertible to categorical.")

    # Melt the dataframe to get long format
    melted = adata.obs.melt(
        id_vars=[group_by, sub_group_by],
        value_vars=agreement_cols,
        var_name='model_name',
        value_name='agreement'
    )

    # Ensure 'agreement' is categorical and reverse the order of categories
    if not pd.api.types.is_categorical_dtype(melted['agreement']):
        melted['agreement'] = melted['agreement'].astype('category')

    # Reverse the order of 'agreement' categories
    original_categories = melted['agreement'].cat.categories.tolist()
    reversed_categories = original_categories[::-1]
    melted['agreement'] = melted['agreement'].cat.reorder_categories(reversed_categories, ordered=True)

    if granularity == 0:
        # Calculate counts and proportions
        counts = melted.groupby(['model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby('model_name')['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts
        
        # Identify the highest agreement category
        # Use the first category from the reversed categories (which should be the highest)
        highest_agreement_category = reversed_categories[0]
        
        # Sort models based on their proportion in the highest agreement category
        highest_agreement_df = counts[counts["agreement"] == highest_agreement_category].copy()
        sorted_models = highest_agreement_df.sort_values("proportion", ascending=False)["model_name"]
        
        # Convert model_name to categorical with specific order
        counts['model_name'] = pd.Categorical(counts['model_name'], categories=sorted_models, ordered=True)


        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=counts,
            x='model_name',
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            ax=ax,
            order=sorted_models
        )

        # Add proportion labels on top of each bar
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height + 0.01,
                    # f'{height:.2f}',
                    f'{height * 100:.0f}%',
                    ha="center"
                )

        # Rotate x-axis tick labels to vertical
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel('Model')
        ax.set_ylabel('Proportion')
        ax.set_title('Proportion of Agreement Categories by Model')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Agreement Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig, ax

    if granularity == 1:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Identify the highest agreement category (same as in granularity==0)
        highest_agreement_category = reversed_categories[0]
        
        # Sort models based on their proportion in the highest agreement category
        highest_agreement_df = counts[counts["agreement"] == highest_agreement_category].copy()
        sorted_models = highest_agreement_df.groupby('model_name')['proportion'].mean().reset_index()
        sorted_models = sorted_models.sort_values("proportion", ascending=False)["model_name"]
        
        # Sort 'group_by' categories based on total proportion
        total_per_group = counts.groupby(group_by)['proportion'].sum().reset_index()
        sorted_groups = total_per_group.sort_values('proportion', ascending=False)[group_by]
        counts[group_by] = pd.Categorical(counts[group_by], categories=sorted_groups, ordered=True)

        # Plot grouped bar chart with model_name as hue
        g = sns.catplot(
            data=counts,
            x=group_by,
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            col='model_name',
            kind='bar',
            height=6,
            aspect=1,
            sharey=True,
            order=sorted_groups
        )

        g.set_axis_labels(group_by, "Proportion")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1.05))

        # Rotate x-axis tick labels to vertical
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Add proportion labels on top of each bar
        for ax in g.axes.flatten():
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        height + 0.01,
                        # f'{height:.2f}',
                        f'{height * 100:.0f}%',
                        ha="center"
                    )

        plt.tight_layout()
        return g

    if granularity == 2:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, sub_group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, sub_group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Prepare data for heatmap
        pivot_table = counts.pivot_table(
            index=[group_by, sub_group_by],
            columns=['model_name', 'agreement'],
            values='proportion',
            fill_value=0
        )

        # Reverse the order of 'agreement' categories in columns
        pivot_table = pivot_table.reindex(columns=reversed_categories, level=2)

        # Sort index based on total proportion
        pivot_table['Total'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values('Total', ascending=False)
        pivot_table = pivot_table.drop(columns='Total')

        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_table,
            cmap='viridis',
            annot=True,
            fmt=".2f",
            linewidths=0.5
        )
        plt.title(f'Proportion of Agreement Categories by {group_by} and {sub_group_by}')
        plt.tight_layout()
        return plt.gcf()

    # Raise an error if granularity is not 0, 1, or 2
    raise ValueError("Granularity must be 0, 1, or 2.")
