"""
unit tests for anndict.annotate.cells.benchmarking.label_utils
"""
import pandas as pd

from anndict.annotate.cells.benchmarking.label_utils import create_label_df

def test_create_label_df(simple_adata_with_many_obs_labels):
    """Test create_label_df function with various column combinations"""

    # Test case 1: Compare 'cluster' with 'condition'
    cols1 = ['cluster']
    cols2 = ['condition']
    expected_result = pd.DataFrame({
        'col1': ['0', '1'],
        'col2': ['A', 'B']
    })
    result = create_label_df(simple_adata_with_many_obs_labels, cols1, cols2)
    pd.testing.assert_frame_equal(result, expected_result)

    # Test case 2: Compare 'cell_type_1' with 'cell_type_2' and 'cell_type_3'
    cols1 = ['cell_type_1']
    cols2 = ['cell_type_2', 'cell_type_3']
    expected_result = pd.DataFrame({
        'col1': ['T', 'B', 'T', 'B'],
        'col2': ['CD4', 'CD8', 'naive', 'memory']
    })
    result = create_label_df(simple_adata_with_many_obs_labels, cols1, cols2)
    pd.testing.assert_frame_equal(result, expected_result)

    # Test case 3: Compare 'cell_type_2' and 'cell_type_3' with 'cell_type_4'
    cols1 = ['cell_type_2', 'cell_type_3']
    cols2 = ['cell_type_4']
    expected_result = pd.DataFrame({
        'col1': ['CD4', 'CD8', 'CD4', 'CD8', 'naive', 'memory', 'naive', 'memory'],
        'col2': ['activated', 'activated', 'resting', 'resting', 'activated', 'activated', 'resting', 'resting']
    })
    result = create_label_df(simple_adata_with_many_obs_labels, cols1, cols2)
    pd.testing.assert_frame_equal(result, expected_result)
