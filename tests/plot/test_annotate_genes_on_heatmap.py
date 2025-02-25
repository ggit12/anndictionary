"""
unit tests for anndict.plot.annotate_genes_on_heatmap
"""

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


from anndict.plot.annotate_genes_on_heatmap import (
    annotate_gene_groups_with_ai_biological_process,
)


def test_annotate_gene_groups_with_ai_biological_process(large_adata):
    """Test the annotation of gene groups on a heatmap with LLM-generated biological process labels."""
    # Set up test data
    groupby = "cell_type"
    sc.tl.rank_genes_groups(large_adata, groupby, method="wilcoxon")

    # Create a heatmap plot
    plot_obj = sc.pl.rank_genes_groups_heatmap(
        large_adata,
        groupby=groupby,
        n_genes=3,
        show_gene_labels=True,
        show=False,
        return_fig=True,
        figsize=(10, 6),
    )

    # Force draw to ensure tick labels are initialized
    plot_obj["heatmap_ax"].figure.canvas.draw()

    # Get actual gene names from the plot
    xlabels = [tick.get_text() for tick in plot_obj["heatmap_ax"].get_xticklabels()]
    type_a_genes = xlabels[:3]  # First 3 genes are for TypeA
    type_b_genes = xlabels[3:]  # Last 3 genes are for TypeB

    # Create mock DataFrame that would come from ai_annotate_biological_process
    df = pd.DataFrame(
        {
            f"{groupby}": ["TypeA", "TypeB"],
            "top_3_genes": [
                type_a_genes,
                type_b_genes,
            ],
            "ai_biological_process": ["Cell growth process", "Immune response"],
        }
    )

    # Get the initial number of figure texts
    fig = plot_obj["heatmap_ax"].figure
    initial_texts = len(fig.texts)

    # Call the function
    annotate_gene_groups_with_ai_biological_process(
        plot_obj=plot_obj, df=df, groupby=groupby
    )

    # Verify that new text annotations were added to the figure
    final_texts = len(fig.texts)
    assert (
        final_texts > initial_texts
    ), f"No new texts added (before: {initial_texts}, after: {final_texts})"

    # Get all text contents from figure texts and remove newlines
    unwrapped_texts = ["".join(text.get_text().split()) for text in fig.texts]
    text_content_str = "".join(unwrapped_texts)

    # Remove spaces and newlines from expected strings for comparison
    cell_growth = "".join("Cell growth process".split())
    immune_response = "".join("Immune response".split())

    # More detailed assertions
    assert (cell_growth in text_content_str), f"Cell growth process not found in: {text_content_str}"
    assert (immune_response in text_content_str), f"Immune response not found in: {text_content_str}"

    # Print debug info
    print("Initial figure texts:", initial_texts)
    print("Final figure texts:", final_texts)
    print("Unwrapped texts:", unwrapped_texts)

    # Clean up
    plt.close("all")
