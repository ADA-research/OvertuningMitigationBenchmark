"""
Comprehensive tests for normalization logic.

Tests cover:
1. Global normalization ranges built correctly from all models
2. Normalized scores in [0, 1] range
3. Min/max raw scores map to 0/1 normalized
4. Degenerate groups (range ≤ 0) set to NaN (not 0.0)
5. Multiple score columns normalized independently
6. Overtuning normalized by test score range
7. Per-group independence of normalization ranges
"""

import numpy as np
import pandas as pd
import pytest
from src.visualizations.preprocessing.combine_normalization import (
    build_normalization_ranges,
    apply_normalization,
)
from src.visualizations.preprocessing.preprocessing import (
    convert_negative_metric_to_loss,
)


class TestBuildNormalizationRanges:
    """Tests for build_normalization_ranges function."""

    def test_basic_range_computation(self):
        """Test that ranges are computed correctly from multiple models."""
        df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds1"],
            "repetition": [0, 0, 0],
            "outer_fold": [0, 0, 0],
            "metric": ["loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM"],
            "avg_val_score": [0.1, 0.2, 0.3],
            "retrained_test_score": [0.15, 0.25, 0.35],
            "cv_ensembled_test_score": [0.12, 0.22, 0.32],
        })
        
        ranges = build_normalization_ranges(df)
        
        assert len(ranges) == 1
        assert ranges.loc[0, "val_min"] == 0.1
        assert ranges.loc[0, "val_max"] == 0.3
        assert ranges.loc[0, "ret_min"] == 0.15
        assert ranges.loc[0, "ret_max"] == 0.35
        assert ranges.loc[0, "ens_min"] == 0.12
        assert ranges.loc[0, "ens_max"] == 0.32

    def test_multiple_groups_independence(self):
        """Test that different groups have independent ranges."""
        df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds1", "ds2", "ds2", "ds2"],
            "repetition": [0, 0, 0, 0, 0, 0],
            "outer_fold": [0, 0, 0, 0, 0, 0],
            "metric": ["loss", "loss", "loss", "loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM", "LGBM", "LGBM", "LGBM"],
            "avg_val_score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "retrained_test_score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "cv_ensembled_test_score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        
        ranges = build_normalization_ranges(df)
        
        assert len(ranges) == 2
        
        ds1_range = ranges[ranges["dataset_id"] == "ds1"].iloc[0]
        assert ds1_range["val_min"] == 1.0
        assert ds1_range["val_max"] == 3.0
        
        ds2_range = ranges[ranges["dataset_id"] == "ds2"].iloc[0]
        assert ds2_range["val_min"] == 10.0
        assert ds2_range["val_max"] == 30.0

    def test_grouping_by_all_cols(self):
        """Test that grouping respects dataset_id, repetition, outer_fold, metric, model."""
        df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds1", "ds1", "ds1", "ds1"],
            "repetition": [0, 0, 0, 1, 1, 1],
            "outer_fold": [0, 0, 0, 0, 0, 0],
            "metric": ["loss", "loss", "loss", "loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM", "LGBM", "LGBM", "LGBM"],
            "avg_val_score": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "retrained_test_score": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "cv_ensembled_test_score": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        })
        
        ranges = build_normalization_ranges(df)
        
        # Should have 2 groups: rep=0 and rep=1
        assert len(ranges) == 2
        
        rep0 = ranges[ranges["repetition"] == 0].iloc[0]
        assert rep0["val_min"] == 1.0
        assert rep0["val_max"] == 3.0
        
        rep1 = ranges[ranges["repetition"] == 1].iloc[0]
        assert rep1["val_min"] == 10.0
        assert rep1["val_max"] == 12.0

    def test_missing_columns_error(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "avg_val_score": [1.0],
            # Missing: retrained_test_score, cv_ensembled_test_score
        })
        
        with pytest.raises(ValueError, match="missing columns"):
            build_normalization_ranges(df)


class TestApplyNormalization:
    """Tests for apply_normalization function."""

    def test_basic_normalization_formula(self):
        """Test that normalization formula is correct: (x - min) / (max - min)."""
        # Create simple trajectory data
        traj_df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds1"],
            "repetition": [0, 0, 0],
            "outer_fold": [0, 0, 0],
            "metric": ["loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM"],
            "val_performance": [0.1, 0.2, 0.3],
            "retrain_test_loss": [0.15, 0.25, 0.35],
            "ensembled_test_loss": [0.12, 0.22, 0.32],
            "retrain_overtuning": [0.0, 0.05, 0.1],
            "ensemble_overtuning": [0.0, 0.05, 0.08],
        })
        
        # Create ranges
        ranges_df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "val_min": [0.1],
            "val_max": [0.3],
            "ret_min": [0.15],
            "ret_max": [0.35],
            "ens_min": [0.12],
            "ens_max": [0.32],
        })
        
        result = apply_normalization(traj_df, ranges_df)
        
        # Check validation loss normalization
        assert abs(result.loc[0, "normalized_val_loss"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_val_loss"] - 0.5) < 1e-10
        assert abs(result.loc[2, "normalized_val_loss"] - 1.0) < 1e-10
        
        # Check retrained test loss normalization
        assert abs(result.loc[0, "normalized_retrain_test_loss"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_retrain_test_loss"] - 0.5) < 1e-10
        assert abs(result.loc[2, "normalized_retrain_test_loss"] - 1.0) < 1e-10

    def test_degenerate_group_nan(self):
        """Test that degenerate groups (range ≤ 0) produce NaN."""
        traj_df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds2", "ds2"],
            "repetition": [0, 0, 0, 0],
            "outer_fold": [0, 0, 0, 0],
            "metric": ["loss", "loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM", "LGBM"],
            "val_performance": [1.0, 1.0, 2.0, 3.0],  # ds1 degenerate
            "retrain_test_loss": [1.0, 1.0, 2.0, 3.0],
            "ensembled_test_loss": [1.0, 1.0, 2.0, 3.0],
            "retrain_overtuning": [0.0, 0.0, 0.0, 0.1],
            "ensemble_overtuning": [0.0, 0.0, 0.0, 0.1],
        })
        
        ranges_df = pd.DataFrame({
            "dataset_id": ["ds1", "ds2"],
            "repetition": [0, 0],
            "outer_fold": [0, 0],
            "metric": ["loss", "loss"],
            "model": ["LGBM", "LGBM"],
            "val_min": [1.0, 2.0],
            "val_max": [1.0, 3.0],  # ds1 range = 0
            "ret_min": [1.0, 2.0],
            "ret_max": [1.0, 3.0],
            "ens_min": [1.0, 2.0],
            "ens_max": [1.0, 3.0],
        })
        
        result = apply_normalization(traj_df, ranges_df)
        
        # Degenerate group should be NaN
        assert pd.isna(result.loc[0, "normalized_val_loss"])
        assert pd.isna(result.loc[1, "normalized_val_loss"])
        
        # Non-degenerate group should not be NaN
        assert not pd.isna(result.loc[2, "normalized_val_loss"])
        assert not pd.isna(result.loc[3, "normalized_val_loss"])

    def test_overtuning_normalization(self):
        """Test that overtuning is normalized by test score range."""
        traj_df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1", "ds1"],
            "repetition": [0, 0, 0],
            "outer_fold": [0, 0, 0],
            "metric": ["loss", "loss", "loss"],
            "model": ["LGBM", "LGBM", "LGBM"],
            "val_performance": [0.2, 0.3, 0.4],
            "retrain_test_loss": [0.2, 0.3, 0.4],
            "ensembled_test_loss": [0.2, 0.3, 0.4],
            "retrain_overtuning": [0.0, 0.05, 0.1],  # max range = 0.2
            "ensemble_overtuning": [0.0, 0.05, 0.1],
        })
        
        ranges_df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "val_min": [0.2],
            "val_max": [0.4],
            "ret_min": [0.2],
            "ret_max": [0.4],  # range = 0.2
            "ens_min": [0.2],
            "ens_max": [0.4],  # range = 0.2
        })
        
        result = apply_normalization(traj_df, ranges_df)
        
        # normalized_overtuning = overtuning / test_range
        # test_range = 0.4 - 0.2 = 0.2
        assert abs(result.loc[0, "normalized_retrain_overtuning"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_retrain_overtuning"] - 0.25) < 1e-10
        assert abs(result.loc[2, "normalized_retrain_overtuning"] - 0.5) < 1e-10

    def test_multiple_score_columns_independent(self):
        """Test that different score columns are normalized independently."""
        traj_df = pd.DataFrame({
            "dataset_id": ["ds1", "ds1"],
            "repetition": [0, 0],
            "outer_fold": [0, 0],
            "metric": ["loss", "loss"],
            "model": ["LGBM", "LGBM"],
            "val_performance": [1.0, 2.0],
            "retrain_test_loss": [10.0, 20.0],  # Different scale
            "ensembled_test_loss": [100.0, 200.0],  # Different scale
            "retrain_overtuning": [0.0, 5.0],
            "ensemble_overtuning": [0.0, 50.0],
        })
        
        ranges_df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "val_min": [1.0],
            "val_max": [2.0],
            "ret_min": [10.0],
            "ret_max": [20.0],
            "ens_min": [100.0],
            "ens_max": [200.0],
        })
        
        result = apply_normalization(traj_df, ranges_df)
        
        # Each score column should independently normalize to [0, 1]
        assert abs(result.loc[0, "normalized_val_loss"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_val_loss"] - 1.0) < 1e-10
        
        assert abs(result.loc[0, "normalized_retrain_test_loss"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_retrain_test_loss"] - 1.0) < 1e-10
        
        assert abs(result.loc[0, "normalized_ensembled_test_loss"] - 0.0) < 1e-10
        assert abs(result.loc[1, "normalized_ensembled_test_loss"] - 1.0) < 1e-10

    def test_missing_trajectory_columns_error(self):
        """Test that missing required columns raise ValueError."""
        traj_df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "val_performance": [1.0],
            # Missing: retrain_test_loss, ensembled_test_loss, retrain_overtuning, ensemble_overtuning
        })
        
        ranges_df = pd.DataFrame({
            "dataset_id": ["ds1"],
            "repetition": [0],
            "outer_fold": [0],
            "metric": ["loss"],
            "model": ["LGBM"],
            "val_min": [1.0],
            "val_max": [2.0],
            "ret_min": [10.0],
            "ret_max": [20.0],
            "ens_min": [100.0],
            "ens_max": [200.0],
        })
        
        with pytest.raises(ValueError, match="missing columns"):
            apply_normalization(traj_df, ranges_df)


class TestConvertNegativeMetricToLoss:
    """Tests for convert_negative_metric_to_loss function."""

    def test_roc_auc_conversion(self):
        """Test that ROC-AUC (negated) is converted to loss form."""
        df = pd.DataFrame({
            "metric": ["roc_auc", "roc_auc", "roc_auc"],
            "avg_val_score": [-0.8, -0.7, -0.6],  # Negated ROC-AUC
            "retrained_test_score": [-0.75, -0.65, -0.55],
            "cv_ensembled_test_score": [-0.75, -0.65, -0.55],
        })
        
        result = convert_negative_metric_to_loss(df)
        
        # After conversion, should be positive loss form
        # Using approximate equality due to floating point arithmetic
        assert abs(result.loc[0, "avg_val_score"] - 0.2) < 1e-10
        assert abs(result.loc[1, "avg_val_score"] - 0.3) < 1e-10
        assert abs(result.loc[2, "avg_val_score"] - 0.4) < 1e-10

    def test_loss_metrics_unchanged(self):
        """Test that loss metrics are not affected."""
        df = pd.DataFrame({
            "metric": ["error", "error", "error"],
            "avg_val_score": [0.1, 0.2, 0.3],
            "retrained_test_score": [0.15, 0.25, 0.35],
            "cv_ensembled_test_score": [0.15, 0.25, 0.35],
        })
        
        result = convert_negative_metric_to_loss(df)
        
        # Loss metrics should be unchanged
        assert (result["avg_val_score"] == df["avg_val_score"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

