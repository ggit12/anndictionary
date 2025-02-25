"""
unit tests for anndict.annotate.cells.benchmarking.agreement_metrics
"""

from anndict.annotate.cells.benchmarking.agreement_metrics import (
    kappa_adata,
    krippendorff_alpha_adata,
)


def test_kappa_adata_basic(simple_adata_with_many_obs_labels):
    """Test basic functionality of kappa_adata with simple test data"""
    cell_type_cols = ["cell_type_1", "cell_type_2", "cell_type_3", "cell_type_4"]

    result = kappa_adata(simple_adata_with_many_obs_labels, cell_type_cols)

    # Check that all expected keys are present in the result
    assert set(result.keys()) == {"pairwise", "average_pairwise", "fleiss"}

    # Check pairwise calculations
    assert isinstance(result["pairwise"], dict)
    for pair, value in result["pairwise"].items():
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(value, float)
        assert -1 <= value <= 1  # Kappa values are bounded between -1 and 1

    # Check average pairwise calculations
    assert isinstance(result["average_pairwise"], dict)
    for _, avg in result["average_pairwise"].items():
        assert isinstance(avg, float)
        assert -1 <= avg <= 1

    # Check Fleiss' Kappa
    assert isinstance(result["fleiss"], float)
    assert -1 <= result["fleiss"] <= 1


def test_kappa_adata_perfect_agreement(simple_adata_with_many_obs_labels):
    """Test kappa_adata with perfectly agreeing raters"""
    adata = simple_adata_with_many_obs_labels

    # Create perfect agreement by duplicating one column
    adata.obs["perfect_1"] = adata.obs["cell_type_1"]
    adata.obs["perfect_2"] = adata.obs["cell_type_1"]

    result = kappa_adata(adata, ["perfect_1", "perfect_2"])

    # Perfect agreement should yield kappa = 1
    assert result["pairwise"][("perfect_1", "perfect_2")] == 1.0
    assert all(avg == 1.0 for avg in result["average_pairwise"].values())
    assert result["fleiss"] == 1.0


def test_kappa_adata_complete_disagreement(simple_adata_with_many_obs_labels):
    """Test kappa_adata with completely disagreeing raters"""
    adata = simple_adata_with_many_obs_labels

    # Create complete disagreement
    adata.obs["disagree_1"] = ["A", "A", "A", "A"]
    adata.obs["disagree_2"] = ["B", "B", "B", "B"]

    result = kappa_adata(adata, ["disagree_1", "disagree_2"])

    # Complete disagreement should yield kappa = 0
    assert result["pairwise"][("disagree_1", "disagree_2")] == 0
    assert all(avg == 0 for avg in result["average_pairwise"].values())
    assert result["fleiss"] == -1


def test_krippendorff_alpha_adata_basic(simple_adata_with_many_obs_labels):
    """Test basic functionality of krippendorff_alpha_adata"""
    cell_type_cols = ["cell_type_1", "cell_type_2", "cell_type_3", "cell_type_4"]

    alpha = krippendorff_alpha_adata(simple_adata_with_many_obs_labels, cell_type_cols)

    assert isinstance(alpha, float)
    assert -1 <= alpha <= 1  # Krippendorff's alpha is bounded between -1 and 1


def test_krippendorff_alpha_adata_perfect_agreement(simple_adata_with_many_obs_labels):
    """Test krippendorff_alpha_adata with perfectly agreeing raters"""
    adata = simple_adata_with_many_obs_labels

    # Create perfect agreement
    adata.obs["perfect_1"] = adata.obs["cell_type_1"]
    adata.obs["perfect_2"] = adata.obs["cell_type_1"]

    alpha = krippendorff_alpha_adata(adata, ["perfect_1", "perfect_2"])

    assert alpha == 1.0  # Perfect agreement should yield alpha = 1


def test_krippendorff_alpha_adata_measurement_levels(simple_adata_with_many_obs_labels):
    """Test krippendorff_alpha_adata with different measurement levels"""
    cell_type_cols = ["cell_type_1", "cell_type_2"]

    # Test all valid measurement levels
    measurement_levels = ["nominal", "ordinal", "interval", "ratio"]
    for level in measurement_levels:
        alpha = krippendorff_alpha_adata(
            simple_adata_with_many_obs_labels, cell_type_cols, level_of_measurement=level
        )
        assert isinstance(alpha, float)
        assert -1 <= alpha <= 1
