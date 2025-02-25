"""
unit tests for anndict.plot.sankey_plots.
"""

from holoviews import Sankey
from anndict.plot.sankey_plots import plot_sankey, save_sankey


def test_plot_sankey_basic(basic_adata):
    """Test that plot_sankey returns a valid Sankey diagram with expected properties."""

    # Call the function with basic parameters
    sankey = plot_sankey(
        basic_adata,
        cols=["true_label", "predicted_label"],
        params={"frame_width": 800, "frame_height": 400},
    )

    # Verify the output is a Holoviews Sankey object
    assert isinstance(sankey, Sankey)

    # Check that the data contains the expected number of flows
    # (3 classes * 3 possible transitions = 9 maximum possible flows)
    assert len(sankey.data["source"]) <= 9

    # Verify total value (should equal number of cells)
    assert sum(sankey.data["value"]) == len(basic_adata)


def test_save_sankey_basic(tmp_path, basic_adata, monkeypatch):
    """Test that save_sankey correctly processes a Sankey plot save request."""

    # Track function calls and arguments
    export_calls = []
    reset_calls = []

    def mock_export_svgs(plot, filename):
        export_calls.append((plot, filename))

    def mock_reset():
        reset_calls.append(True)

    # Patch the necessary functions
    monkeypatch.setattr("bokeh.io.export_svgs", mock_export_svgs)
    monkeypatch.setattr("bokeh.io.webdriver.webdriver_control.reset", mock_reset)

    # Create a sample Sankey plot
    sankey = plot_sankey(basic_adata, cols=["true_label", "predicted_label"])

    # Define test filename
    filename = str(tmp_path / "test_sankey.svg")

    # Save the plot
    save_sankey(sankey, filename)

    # Expected final filename wit
    expected_filename = str(tmp_path / "test_sankey.svg")

    # Verify the mocked functions were called correctly
    assert len(reset_calls) == 1  # reset was called once
    assert len(export_calls) == 1  # export was called once

    # Verify the filename passed to export_svgs
    _, actual_filename = export_calls[0]
    assert actual_filename == expected_filename
