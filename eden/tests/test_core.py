import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

from eden import BacktestConfig, Eden, EdenBahamutBundle, EdenSpec, ProphetPerTargetAdapter, QuantileEnsembleAdapter


def build_panel_fixture() -> tuple[pd.DataFrame, dict[str, object], list[int], list[int], list[int]]:
    dates = pd.date_range("2024-01-01", periods=18, freq="D")
    rows = []
    for store_code, (store_id, base_sales, base_margin) in enumerate([
        ("A", 100.0, 30.0),
        ("B", 92.0, 27.0),
    ]):
        for idx, ds in enumerate(dates):
            trend = idx + 1
            promo = int(idx % 4 in {1, 3})
            planned_discount = [0.00, 0.05, 0.10, 0.02, 0.08, 0.00][idx % 6]
            price_index = [1.00, 0.99, 1.01, 0.97, 1.00, 0.98][idx % 6]
            seasonal = float(np.sin((idx + 1) / 2.7))

            y_sales = (
                base_sales
                + 3.8 * trend
                + 8.5 * promo
                - 38.0 * planned_discount
                - 14.0 * (price_index - 1.0)
                + 5.5 * seasonal
                + 3.0 * store_code
            )
            y_margin = (
                base_margin
                + 1.6 * trend
                + 2.8 * promo
                - 14.0 * planned_discount
                - 6.0 * (price_index - 1.0)
                + 1.8 * seasonal
                + 1.5 * store_code
            )

            rows.append(
                {
                    "store_id": store_id,
                    "ds": ds,
                    "trend": trend,
                    "promo": promo,
                    "planned_discount": planned_discount,
                    "price_index": price_index,
                    "store_code": store_code,
                    "y_sales": round(y_sales, 2),
                    "y_margin": round(y_margin, 2),
                }
            )

    panel_df = pd.DataFrame(rows)
    train_idx = panel_df.loc[panel_df["ds"] <= dates[10]].index.tolist()
    valid_idx = panel_df.loc[(panel_df["ds"] >= dates[11]) & (panel_df["ds"] <= dates[12])].index.tolist()
    test_idx = panel_df.loc[panel_df["ds"] >= dates[13]].index.tolist()

    feature_frame = panel_df[["ds", "store_id", "trend", "promo", "planned_discount", "price_index", "store_code"]].copy()
    target_frame = panel_df[["y_sales", "y_margin"]].copy()
    segments = {
        "X_train": feature_frame.loc[train_idx].copy(),
        "X_validation": feature_frame.loc[valid_idx].copy(),
        "X_test": feature_frame.loc[test_idx].copy(),
        "y_train": target_frame.loc[train_idx].copy(),
        "y_validation": target_frame.loc[valid_idx].copy(),
        "y_test": target_frame.loc[test_idx].copy(),
        "indices_train": pd.Index(train_idx),
        "indices_validation": pd.Index(valid_idx),
        "indices_test": pd.Index(test_idx),
        "resumen_segmentos": pd.DataFrame({"segmento": ["train", "validation", "test"]}),
    }
    return panel_df, segments, train_idx, valid_idx, test_idx


def build_common_spec_kwargs() -> dict[str, object]:
    return {
        "prediction_suffix": "_pred",
        "adapter_strategy": "multi_output",
        "interval_coverage": 0.9,
        "calibration_split": "valid",
        "metrics": ["mae", "rmse", "wape"],
        "known_future_feature_cols": ["trend", "promo", "planned_discount", "price_index", "store_code"],
        "enforce_future_covariates": True,
        "expected_frequency": "D",
        "allow_calendar_gaps": False,
    }


def test_eden_run_from_bahamut_keeps_metrics_intervals_and_backtest():
    panel_df, segments, _, _, test_idx = build_panel_fixture()
    eden = Eden.build_from_bahamut(
        segments,
        timestamp_col="ds",
        series_structure="panel",
        entity_id_col="store_id",
        model=LinearRegression(),
        **build_common_spec_kwargs(),
    )

    result = eden.run_from_bahamut(segments)
    test_bundle = eden.predict_with_metrics(panel_df.loc[test_idx].copy())
    backtest = eden.backtest_from_bahamut(
        segments,
        BacktestConfig(base_split="train", initial_train_periods=5, horizon=1, step=1, max_folds=3),
    )

    assert result["diagnostics"]["supports_interval"] is True
    assert result["diagnostics"]["interval_calibration_source"] == "valid"
    assert {"y_sales_lower_90", "y_sales_upper_90"}.issubset(result["test_predictions"].columns)
    assert test_bundle["metrics_available"] is True
    assert "y_sales__mae" in test_bundle["metrics"]
    assert backtest["fold_metrics"]["fold"].nunique() == 3


def test_eden_bundle_requires_original_dataframe_when_temporal_context_is_missing():
    panel_df, segments, _, _, _ = build_panel_fixture()
    spec = EdenSpec.from_bahamut_segments(
        segments,
        timestamp_col="ds",
        series_structure="panel",
        entity_id_col="store_id",
        **build_common_spec_kwargs(),
    )
    eden = Eden(spec=spec, model=LinearRegression())

    missing_context = {
        **segments,
        "X_train": segments["X_train"][spec.feature_cols].copy(),
        "X_validation": segments["X_validation"][spec.feature_cols].copy(),
        "X_test": segments["X_test"][spec.feature_cols].copy(),
    }

    try:
        eden.run_from_bahamut(missing_context)
    except ValueError as exc:
        assert "pass df=original_dataframe" in str(exc)
    else:
        raise AssertionError("Eden debia exigir df original si Bahamut no trae contexto temporal.")

    result = eden.run_from_bahamut(missing_context, df=panel_df)
    assert "test_predictions" in result


def test_quantile_adapter_build_keeps_interval_columns():
    _, segments, _, _, _ = build_panel_fixture()
    eden = Eden.build_from_bahamut(
        segments,
        timestamp_col="ds",
        series_structure="panel",
        entity_id_col="store_id",
        adapter=QuantileEnsembleAdapter(
            center_estimator_blueprint=LinearRegression(),
            lower_estimator_blueprint=QuantileRegressor(quantile=0.05, alpha=0.0),
            upper_estimator_blueprint=QuantileRegressor(quantile=0.95, alpha=0.0),
            lower_quantile=0.05,
            upper_quantile=0.95,
        ),
        **build_common_spec_kwargs(),
    )

    result = eden.run_from_bahamut(segments)
    assert {"y_sales_lower_90", "y_sales_upper_90"}.issubset(result["test_predictions"].columns)


def test_bahamut_bundle_infers_feature_and_target_columns():
    _, segments, _, _, _ = build_panel_fixture()
    bundle = EdenBahamutBundle.from_bahamut(segments, timestamp_col="ds", entity_id_col="store_id")

    assert bundle.feature_cols == ["trend", "promo", "planned_discount", "price_index", "store_code"]
    assert bundle.target_cols == ["y_sales", "y_margin"]