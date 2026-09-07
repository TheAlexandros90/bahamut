"""Regressions for the defects found during the 2026 audit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from eden import BacktestConfig, Eden, EdenSpec, EdenSplits


def build_series(periods: int = 48) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2021-01-01", periods=periods, freq="D"),
            "trend": np.arange(periods, dtype=float),
            "promo": (np.arange(periods) % 4 == 0).astype(float),
        }
    )
    rng = np.random.default_rng(11)
    frame["y"] = 100 + 3 * frame["trend"] + 12 * frame["promo"] + rng.normal(0, 2, periods)
    return frame


def build_spec(**overrides) -> EdenSpec:
    kwargs = dict(
        timestamp_col="ds",
        target_cols=["y"],
        feature_cols=["trend", "promo"],
        metrics=["mae", "rmse", "wape"],
        interval_coverage=0.9,
        calibration_split="valid",
    )
    kwargs.update(overrides)
    return EdenSpec(**kwargs)


# --------------------------------------------------------------------------
# Conformal calibration must never be silently in-sample
# --------------------------------------------------------------------------

def test_calibrating_on_train_emits_an_explicit_warning():
    frame = build_series()
    splits = EdenSplits(train_idx=list(range(0, 40)), test_idx=list(range(40, 48)))

    with pytest.warns(UserWarning, match="calibrated on the training split"):
        model = Eden(spec=build_spec(), model=LinearRegression()).fit(frame, splits)

    assert model.diagnostics()["interval_calibration_is_in_sample"] is True


def test_calibrating_on_a_validation_split_is_silent_and_flagged_out_of_sample():
    frame = build_series()
    splits = EdenSplits(
        train_idx=list(range(0, 32)),
        valid_idx=list(range(32, 40)),
        test_idx=list(range(40, 48)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = Eden(spec=build_spec(), model=LinearRegression()).fit(frame, splits)

    diagnostics = model.diagnostics()
    assert diagnostics["interval_calibration_source"] == "valid"
    assert diagnostics["interval_calibration_is_in_sample"] is False


def test_backtest_folds_calibrate_out_of_sample():
    frame = build_series()
    splits = EdenSplits(train_idx=list(range(0, 40)), test_idx=list(range(40, 48)))
    model = Eden(spec=build_spec(), model=LinearRegression())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = model.backtest(
            frame,
            splits,
            BacktestConfig(base_split="train", initial_train_periods=12, horizon=2, step=4, max_folds=3),
        )

    assert result["fold_metrics"]["fold"].nunique() == 3
    assert {"y_lower_90", "y_upper_90"}.issubset(result["predictions"].columns)


def test_backtest_without_intervals_uses_the_whole_training_window():
    frame = build_series()
    splits = EdenSplits(train_idx=list(range(0, 40)), test_idx=list(range(40, 48)))
    model = Eden(spec=build_spec(interval_coverage=None), model=LinearRegression())

    fit_ts, calibration_ts = model._split_fold_calibration(pd.Index(frame["ds"].head(10)))

    assert calibration_ts is None
    assert len(fit_ts) == 10

    result = model.backtest(
        frame,
        splits,
        BacktestConfig(base_split="train", initial_train_periods=12, horizon=2, step=4, max_folds=2),
    )
    assert "y_lower_90" not in result["predictions"].columns


def test_calibration_split_is_not_created_when_the_window_is_too_short():
    frame = build_series()
    model = Eden(spec=build_spec(), model=LinearRegression())

    fit_ts, calibration_ts = model._split_fold_calibration(pd.Index(frame["ds"].head(2)))

    assert calibration_ts is None
    assert len(fit_ts) == 2


# --------------------------------------------------------------------------
# Shared notebook plumbing
# --------------------------------------------------------------------------

def test_core_does_not_import_the_notebook_stack():
    import eden.core as core_module

    with open(core_module.__file__, encoding="utf-8") as handle:
        contenido = handle.read()

    assert "import ipywidgets" not in contenido
    assert "from IPython" not in contenido


def test_notebook_helpers_come_from_the_shared_module():
    import eden.interactive as interactive_module

    with open(interactive_module.__file__, encoding="utf-8") as handle:
        contenido = handle.read()

    assert "from ._notebook import" in contenido
    assert "import ipywidgets as widgets" not in contenido


# --------------------------------------------------------------------------
# Intervalos de cuantiles degenerados
# --------------------------------------------------------------------------

def test_quantile_adapter_avisa_si_los_estimadores_no_son_de_cuantiles():
    from eden import QuantileEnsembleAdapter

    frame = build_series(30)
    splits = EdenSplits(
        train_idx=list(range(0, 20)),
        valid_idx=list(range(20, 25)),
        test_idx=list(range(25, 30)),
    )
    adapter = QuantileEnsembleAdapter(
        center_estimator_blueprint=LinearRegression(),
        lower_estimator_blueprint=LinearRegression(),
        upper_estimator_blueprint=LinearRegression(),
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    model = Eden(spec=build_spec(), adapter=adapter).fit(frame, splits)

    with pytest.warns(UserWarning, match="zero width"):
        salida = model.predict_interval(frame.loc[25:29])

    assert (salida["y_upper_90"] - salida["y_lower_90"]).abs().max() == pytest.approx(0.0)


def test_quantile_adapter_no_avisa_con_estimadores_de_cuantiles_reales():
    from sklearn.linear_model import QuantileRegressor

    from eden import QuantileEnsembleAdapter

    frame = build_series(40)
    splits = EdenSplits(
        train_idx=list(range(0, 28)),
        valid_idx=list(range(28, 34)),
        test_idx=list(range(34, 40)),
    )
    adapter = QuantileEnsembleAdapter(
        center_estimator_blueprint=LinearRegression(),
        lower_estimator_blueprint=QuantileRegressor(quantile=0.05, alpha=0.0),
        upper_estimator_blueprint=QuantileRegressor(quantile=0.95, alpha=0.0),
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    model = Eden(spec=build_spec(), adapter=adapter).fit(frame, splits)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        salida = model.predict_interval(frame.loc[34:39])

    assert (salida["y_upper_90"] - salida["y_lower_90"]).min() > 0
