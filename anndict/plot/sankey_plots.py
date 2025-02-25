"""
Make and save sankey plots based on columns in ``adata``. 
Uses holoviews+bokeh and supports coloring edges by their weight, 
as well as other other customization parameters.
"""

import os
from typing import Literal
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from anndata import AnnData

def plot_sankey(
    adata: AnnData,
    cols: list[str],
    params: dict[str, str | int | float | bool ] | None = None
) -> Literal["holoviews.Sankey"]:
    """
    Generate a Sankey diagram from the specified columns in ``adata.obs``.

    Parameters
    ------------
    adata
        An :class:`AnnData` object.

    cols
        The list of column names in ``adata.obs`` from which to construct the Sankey diagram.

    params
        A :class:`dict` of optional parameters to customize the Sankey diagram appearance. Supported keys include:

        - ``'cmap'``: :class:`str`, colormap for node colors (default: ``'Colorblind'``).
        - ``'label_position'``: :class:`str`, position of node labels (``'outer'`` or ``'center'``, default: ``'outer'``).
        - ``'edge_line_width'``: :class:`int`, width of the edges (default: ``0``).
        - ``'edge_color'``: :class:`str`, attribute for edge coloring (default: ``'value'``, or ``'grey'`` for uniform color).
        - ``'show_values'``: :class:`bool`, whether to display flow values (default: ``False``).
        - ``'node_padding'``: :class:`int`, padding between nodes (default: ``12``).
        - ``'node_alpha'``: :class:`float`, transparency of nodes (default: ``1.0``).
        - ``'node_width'``: :class:`int`, width of nodes (default: ``30``).
        - ``'node_sort'``: :class:`bool`, whether to sort nodes (default: ``True``).
        - ``'frame_height'``: :class:`int`, height of the diagram frame (default: ``1000``).
        - ``'frame_width'``: :class:`int`, width of the diagram frame (default: ``2000``).
        - ``'bgcolor'``: :class:`str`, background color of the diagram (default: ``'white'``).
        - ``'apply_ranges'``: :class:`bool`, whether to apply range adjustments to the plot (default: ``True``).
        - ``'align_thr'``: :class:`float`, alignment threshold for colors (default: ``-0.1``).
        - ``'label_font_size'``: :class:`str`, font size for labels (default: ``'12pt'``).

    Returns
    --------
    A Holoviews Sankey diagram object.

    Example
    --------
    .. code-block:: python

        sankey = plot_sankey(adata, cols=['cell_sub_type', 'cell_type', 'compartment'], params={'cmap': 'viridis', 'frame_width': 1200})
        hv.save(sankey, 'sankey_diagram.html')
    """
    import holoviews as hv # pylint: disable=import-outside-toplevel
    hv.extension('bokeh')

    def f(plot, element): # pylint: disable=unused-argument
        plot.handles['plot'].sizing_mode = 'scale_width'
        plot.handles['plot'].x_range.start = -1000
        plot.handles['plot'].x_range.end = 1500


    if params is None:
        params = {}

    obs = adata.obs[cols]

    # Creating unique labels for each column
    unique_labels = []
    label_dict = defaultdict(dict)
    for col_index, col in enumerate(cols):
        col_data = obs[col].astype(str).tolist()
        for item in col_data:
            if item not in label_dict[col_index]:
                unique_label = f"{item} ({col})"
                label_dict[col_index][item] = unique_label
                unique_labels.append(unique_label)

    # Creating source, target and value lists
    source = []
    target = []
    value = []
    for i in range(len(cols) - 1):
        ct_dict = defaultdict(int)
        for a, b in zip(obs[cols[i]].astype(str), obs[cols[i+1]].astype(str)):
            ct_dict[(a, b)] += 1
        for (a, b), v in ct_dict.items():
            source.append(label_dict[i][a])
            target.append(label_dict[i+1][b])
            value.append(v)

    # Creating DataFrame for Sankey
    sankey_data = pd.DataFrame({
        'source': source,
        'target': target,
        'value': value
    })

    # Appearance parameters
    cmap = params.get('cmap', 'Colorblind')
    label_position = params.get('label_position', 'outer')
    edge_line_width = params.get('edge_line_width', 0)
    edge_color = params.get('edge_color', 'value')  # allows grey edges
    show_values = params.get('show_values', False)
    node_padding = params.get('node_padding', 12)
    node_alpha = params.get('node_alpha', 1.0)
    node_width = params.get('node_width', 30)
    node_sort = params.get('node_sort', True)
    frame_height = params.get('frame_height', 1000)
    frame_width = params.get('frame_width', 2000)
    bgcolor = params.get('bgcolor', 'white')
    apply_ranges = params.get('apply_ranges', True)
    align_thr = params.get('align_thr', -0.1)
    label_font_size = params.get('label_font_size', '12pt')

    colormap_max = max(sankey_data['value'])
    norm = plt.Normalize(vmin=0, vmax=colormap_max)
    colors = plt.colormaps.get_cmap("plasma")(norm(np.linspace(0, colormap_max, 128)))

    replace_these = np.where(norm(np.linspace(0, colormap_max, 128)) <= align_thr)[0]
    if replace_these.size > 0:
        colors[replace_these] = [[1, 1, 1, 0] for _ in range(len(replace_these))]

    edge_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    if edge_color == "grey":
        # edge_color = "grey"  # Set edge_color to grey
        edge_cmap = None  # No colormap for grey edges

    sankey = hv.Sankey(sankey_data, kdims=["source", "target"], vdims=["value"])
    sankey = sankey.opts(
        cmap=cmap, label_position=label_position, edge_color=edge_color, edge_cmap=edge_cmap, colorbar=True if edge_cmap else False, # pylint: disable=simplifiable-if-expression
        edge_line_width=edge_line_width, show_values=show_values, node_padding=node_padding, node_alpha=node_alpha,
        node_width=node_width, node_sort=node_sort, frame_height=frame_height, frame_width=frame_width,
        bgcolor=bgcolor, apply_ranges=apply_ranges, label_text_font_size=label_font_size, hooks=[f]
    )
    sankey = sankey.opts(clim=(0, colormap_max))

    return sankey

def save_sankey(
    plot: Literal["holoviews.Sankey"],
    filename: str,
    adt_key: tuple[str,...] | None = None
) -> None:
    """
    Save a Holoviews Sankey plot as an SVG file.

    Parameters
    ------------
    plot
        Holoviews plot, The Sankey plot to save.
    filename
        Base filename for the output SVG file.
    adt_key
        Optional identifier to append to the filename.

    Returns
    --------
    None

    Notes
    ------
    Saves sankey plot as ``svg`` at the location specified by filename.
    """
    import holoviews as hv # pylint: disable=import-outside-toplevel
    from bokeh.io.webdriver import webdriver_control # pylint: disable=import-outside-toplevel
    from bokeh.io import export_svgs # pylint: disable=import-outside-toplevel

    # Reset web driver because sometimes the max connections is hit when writing plots
    webdriver_control.reset()

    # Remove '.svg' if it exists and append '{adt_key}.svg'
    filename = os.path.splitext(filename)[0]
    if adt_key:
        filename += f"_{adt_key}"
    filename += ".svg"

    plot = hv.render(plot)
    plot.output_backend = "svg"

    export_svgs(plot, filename=filename)
